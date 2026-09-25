/**
 * backward.c - The differentiation contract, the losses and the optimizer
 * (plan Stage 4). See csrc/include/backward.h for the contract.
 *
 * CUDA-free on purpose, like train.c: the losses, AdamW, the accumulation schedule
 * and the checkpoint format are FP32 arithmetic over memory the caller owns, so the
 * CPU gate drives them directly and the device kernels are held to the same numbers.
 *
 * The one thing worth repeating here is why the region table is code rather than
 * prose. The plan's Stage-4 table has eleven rows, and the failure mode it is guarding
 * against is a row that is silently missing while the stage is declared done. Every
 * row therefore names the Stage-1 inventory regions it differentiates, and the CPU
 * gate walks both directions: every named region exists in the inventory, and every
 * row the plan lists has a row here. `implemented` is the row's own claim, and a row
 * that sets it is one whose entry point is gated.
 */
#include "backward.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Per-thread error text                                              */
/* ------------------------------------------------------------------ */

static _Thread_local char g_error[256];

const char *backward_last_error(void) { return g_error; }

void backward_clear_error(void) { g_error[0] = '\0'; }

static BackwardStatus fail(BackwardStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

/* ------------------------------------------------------------------ */
/* Region coverage                                                    */
/* ------------------------------------------------------------------ */

/* The plan's Stage-4 table, row by row and in its order. `stage1_regions` lists the
 * Stage-1 inventory regions the row's backward differentiates; the two tables are
 * kept in agreement by the CPU gate in tests/backward_test.c, which includes
 * regions.h and walks both directions. */
static const struct BackwardRegionInfo kRegions[BACKWARD_REGION_ROW_COUNT] = {
    {BACKWARD_REGION_EMBEDDING_TIED_LM_HEAD, "Embedding and tied LM head",
     "embedding", 1,
     "kernels/backward.cu: kernel_embedding_backward (atomic scatter-add) and "
     "backward_grad_accumulate_tied; one optimizer update per logical parameter is "
     "the Stage-3 store's tying, not a second step",
     "the LM head's vocabulary-projection weight gradient is accumulated into the "
     "embedding table's gradient because they are one logical parameter, so this row "
     "owns the tie and the GEMM row owns the projection's dX/dW; a shard-splitting "
     "vocabulary-parallel head is not covered"},

    {BACKWARD_REGION_RESIDUAL_ELEMENTWISE, "Residual and elementwise gates",
     "residual_add,silu_mul,conv_silu,attention_output_gate", 1,
     "kernels/backward.cu: kernel_residual_add_backward (branch accumulation), "
     "kernel_silu_mul_backward, kernel_silu_inplace_backward, "
     "kernel_sigmoid_mul_backward",
     "the sigmoid backward consumes the BF16-rounded sigmoid of the forward's gate, "
     "not a rounded-to-BF16 of the exact sigmoid, because the forward rounds it before "
     "multiplying"},

    {BACKWARD_REGION_ROPE_QGATE_SPLIT, "RoPE and Q/gate split",
     "rope,q_gate_split", 1,
     "kernels/backward.cu: kernel_rope_backward (the transposed rotation with the same "
     "tables) and kernel_qgate_merge (re-interleave the two gradients)",
     "the rotation tables are recomputed from positions, so the backward needs the "
     "positions and not the rotated activations"},

    {BACKWARD_REGION_NORMS, "Plain/Gemma/per-head/gated norms",
     "rmsnorm,per_head_norm,gdn_gated_norm", 1,
     "kernels/backward.cu: kernel_rmsnorm_backward (weight+1 and raw weight), "
     "kernel_l2norm_backward (the GDN Q/K normalisation, a different function from "
     "RMSNorm) and kernel_gdn_gated_norm_backward",
     "per-head norm is the same RMSNorm over [tokens*heads, head_dim] rows; the GDN "
     "gated norm's FP32 effective weight is a derived copy (Stage 3), so its gradient "
     "belongs to the BF16 source and is summed there"},

    {BACKWARD_REGION_GEMM, "GEMM including FP32 LM-head output",
     "gemm_bf16,gemm_fp32_lmhead", 1,
     "kernels/backward.cu: kernel_gemm_backward_dx / kernel_gemm_backward_dw over the "
     "cuBLAS contract the forward uses (BF16 operands, FP32 accumulate)",
     "dW is accumulated in FP32 across the rows of one call; the plan's dW guarantee "
     "is grouping-invariance under a fixed accumulation schedule, and adding tokens to "
     "a batch is explicitly outside it (Stage 2)"},

    {BACKWARD_REGION_GDN_PREPARE, "GDN prepare",
     "gdn_prepare", 1,
     "kernels/backward.cu: kernel_gdn_prepare_backward - through sigmoid(beta) with its "
     "BF16 rounding, softplus, exp(A_log) and the L2 normalisation, with the duplicated "
     "key heads reduced into their source",
     "the a/b/A_log/dt_bias chain is differentiated as the FP32 arithmetic the AOT "
     "kernel performs on BF16 inputs; a chain that reorders the FP32 additions is a "
     "different function"},

    {BACKWARD_REGION_GDN_CONV1D, "GDN conv1d",
     "gdn_conv1d", 1,
     "kernels/backward.cu: kernel_causal_conv1d_backward (dx with the state's history, "
     "and dw reduced over tokens) with the SiLU derivative as a separate pass",
     "the state's gradient is what a previous step would need; this stage returns it "
     "because a full-sequence backward crosses the boundary the state was truncated at"},

    {BACKWARD_REGION_GDN_CORE, "GDN core",
     "gdn_core", 1,
     "kernels/backward.cu: kernel_gdn_core_backward, the reverse of the decay-before-"
     "prediction recurrence, started from the retained chunk-boundary states so the "
     "gradient crosses every internal boundary (full-sequence BPTT)",
     "the chunkwise AOT cubin decomposes the recurrence with (I+A)^{-1} and BF16 "
     "intermediate MMAs; this backward differentiates the mathematical recurrence, so "
     "it is the pair for the recurrent contract and differs from the cubin by its "
     "rounding, which is a Stage-6 alignment question, not a Stage-4 one"},

    {BACKWARD_REGION_ATTENTION_CORE, "Attention core",
     "attention_core", 1,
     "kernels/backward.cu: kernel_attention_forward_lse (the forward with its base-2 "
     "LSE exported, which the engine's entry point had disabled) and "
     "kernel_attention_backward, the paired backward Stage 2's claim E validated",
     "the backward recomputes P from the saved LSE instead of retaining a [T,T] "
     "probability tensor; split-KV stays disabled so no reduction workspace is needed "
     "(Stage 2's resource note)"},

    {BACKWARD_REGION_LOSSES, "Losses",
     "masked_loss", 1,
     "backward.c: masked CE (with its log-softmax gradient), dense reverse KL, the "
     "token/sequence clipped objective and the group advantage reduction; "
     "kernels/logprob.cu: kernel_logprob_gather_backward for the fused row-at-a-time "
     "path Stage 3 started",
     "the fused device path returns dlogits for one row at a time, so a [tokens, "
     "vocab] tensor is never materialised; the CPU implementation is the independent "
     "second reading of the same contract"},

    {BACKWARD_REGION_ADAMW, "AdamW",
     "", 1,
     "backward.c: backward_adamw_step (FP32 master/m/v, bias correction, decoupled "
     "decay, then the single BF16 rounding of publication); kernels/backward.cu: "
     "kernel_adamw for the device step",
     "no parameter group, no per-tensor scaling and no gradient clipping: those are "
     "optimizer policy the plan leaves to Stage 5+, and adding them here would change "
     "the step the gate compares against torch"},
};

const struct BackwardRegionInfo *backward_region_info(int *count) {
    if (count != NULL) *count = BACKWARD_REGION_ROW_COUNT;
    return kRegions;
}

const struct BackwardRegionInfo *backward_region_at(BackwardRegion region) {
    if (region < 0 || region >= BACKWARD_REGION_ROW_COUNT) return NULL;
    return &kRegions[region];
}

int backward_region_stage1_count(BackwardRegion region) {
    const struct BackwardRegionInfo *info = backward_region_at(region);
    if (info == NULL || info->stage1_regions[0] == '\0') return 0;
    int count = 1;
    for (const char *cursor = info->stage1_regions; *cursor != '\0'; ++cursor) {
        if (*cursor == ',') ++count;
    }
    return count;
}

int backward_region_stage1_name_at(BackwardRegion region, int index, char *out, size_t out_size) {
    const struct BackwardRegionInfo *info = backward_region_at(region);
    if (info == NULL || index < 0 || out == NULL || out_size == 0) return -1;
    const char *cursor = info->stage1_regions;
    for (int i = 0; i <= index; ++i) {
        if (*cursor == '\0') return -1;
        const char *comma = strchr(cursor, ',');
        const size_t length = comma != NULL ? (size_t)(comma - cursor) : strlen(cursor);
        if (i == index) {
            if (length + 1 > out_size) return -1;
            memcpy(out, cursor, length);
            out[length] = '\0';
            return (int)length;
        }
        cursor = comma + 1;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* The differentiation convention                                     */
/* ------------------------------------------------------------------ */

/* A BF16 cast is a step function, so "identity" is a choice and the only meaningful
 * check is that the whole model was compared against a reference that behaves the same
 * way. The engine reads this rather than assuming it. */
BackwardCastMode backward_cast_mode(void) { return BACKWARD_CAST_IDENTITY; }

float backward_default_loss_scale(void) { return 1.0f; }

/* ------------------------------------------------------------------ */
/* Retained values                                                    */
/* ------------------------------------------------------------------ */

/* What each region's backward consumes. `name` is matched exactly against the step's
 * retained names by backward_check_retained. RETAINED means the forward must have
 * exported it; RECOMPUTED means the backward re-derives it from values that are
 * retained or cheap (the RoPE tables from positions, P from the base-2 LSE), which is
 * the plan's "saved or recomputed statistics". */
static const struct BackwardRequiredValue kRequired[] = {
    {BACKWARD_REGION_EMBEDDING_TIED_LM_HEAD, "token_ids", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_EMBEDDING_TIED_LM_HEAD, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_RESIDUAL_ELEMENTWISE, "residual_in", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_RESIDUAL_ELEMENTWISE, "branch_out", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_RESIDUAL_ELEMENTWISE, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_ROPE_QGATE_SPLIT, "positions", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ROPE_QGATE_SPLIT, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_NORMS, "x", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_NORMS, "inv_rms", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_NORMS, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_GEMM, "x", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GEMM, "weight", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GEMM, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_GDN_PREPARE, "conv_out", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_PREPARE, "a", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_PREPARE, "b", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_PREPARE, "A_log", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_PREPARE, "dt_bias", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_PREPARE, "beta", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_PREPARE, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_GDN_CONV1D, "x", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CONV1D, "pre_activation", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CONV1D, "weight", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CONV1D, "conv_state_in", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CONV1D, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_GDN_CORE, "prepared_q", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CORE, "prepared_k", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CORE, "v", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CORE, "log_decay", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CORE, "beta", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CORE, "chunk_state", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_GDN_CORE, "out_grad", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_ATTENTION_CORE, "prepared_q", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ATTENTION_CORE, "kv_cache", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ATTENTION_CORE, "lse", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ATTENTION_CORE, "out_grad", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ATTENTION_CORE, "probabilities", BACKWARD_SOURCE_RECOMPUTED},

    {BACKWARD_REGION_LOSSES, "logits", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_LOSSES, "labels", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_LOSSES, "loss_mask", BACKWARD_SOURCE_RETAINED},

    {BACKWARD_REGION_ADAMW, "master", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ADAMW, "grad", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ADAMW, "opt_m", BACKWARD_SOURCE_RETAINED},
    {BACKWARD_REGION_ADAMW, "opt_v", BACKWARD_SOURCE_RETAINED},
};

const struct BackwardRequiredValue *backward_required_values(int *count) {
    if (count != NULL) *count = (int)(sizeof(kRequired) / sizeof(kRequired[0]));
    return kRequired;
}

BackwardStatus backward_check_retained(BackwardRegion region, const char *const *have,
                                       int have_count) {
    if (region < 0 || region >= BACKWARD_REGION_ROW_COUNT) {
        return fail(BACKWARD_ERR_ARG, "backward_check_retained: unknown region %d", (int)region);
    }
    if (have_count < 0 || (have_count > 0 && have == NULL)) {
        return fail(BACKWARD_ERR_ARG, "backward_check_retained: invalid retained set");
    }
    for (size_t i = 0; i < sizeof(kRequired) / sizeof(kRequired[0]); ++i) {
        const struct BackwardRequiredValue *need = &kRequired[i];
        if (need->region != region) continue;
        /* A value the backward recomputes is not the step's to retain, but the step
         * has to know it will be recomputed rather than saved. Both are accepted here
         * and the distinction is what the caller reports; what is refused is a value
         * that is neither retained nor declared recomputable. */
        int found = 0;
        for (int j = 0; j < have_count; ++j) {
            if (have[j] != NULL && strcmp(have[j], need->name) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            return fail(BACKWARD_ERR_MISSING,
                        "backward_check_retained: %s needs '%s' (%s)", kRegions[region].name,
                        need->name,
                        need->source == BACKWARD_SOURCE_RECOMPUTED ? "recomputed" : "retained");
        }
    }
    return BACKWARD_OK;
}

/* ------------------------------------------------------------------ */
/* Gradient accumulation                                              */
/* ------------------------------------------------------------------ */

BackwardStatus backward_grad_zero(float *grad, long long n) {
    if (n < 0 || (n > 0 && grad == NULL)) {
        return fail(BACKWARD_ERR_ARG, "backward_grad_zero: invalid buffer");
    }
    for (long long i = 0; i < n; ++i) grad[i] = 0.0f;
    return BACKWARD_OK;
}

BackwardStatus backward_grad_accumulate(float *grad, const float *contribution, long long n) {
    if (n < 0 || (n > 0 && (grad == NULL || contribution == NULL))) {
        return fail(BACKWARD_ERR_ARG, "backward_grad_accumulate: invalid buffer");
    }
    for (long long i = 0; i < n; ++i) grad[i] += contribution[i];
    return BACKWARD_OK;
}

BackwardStatus backward_grad_accumulate_tied(float *grad, const float *contribution, long long n,
                                             long long *out_nonzero) {
    if (n < 0 || (n > 0 && (grad == NULL || contribution == NULL))) {
        return fail(BACKWARD_ERR_ARG, "backward_grad_accumulate_tied: invalid buffer");
    }
    long long nonzero = 0;
    for (long long i = 0; i < n; ++i) {
        grad[i] += contribution[i];
        if (contribution[i] != 0.0f) ++nonzero;
    }
    if (out_nonzero != NULL) *out_nonzero = nonzero;
    return BACKWARD_OK;
}

BackwardStatus backward_grad_scale(float *grad, long long n, float scale) {
    if (n < 0 || (n > 0 && grad == NULL)) {
        return fail(BACKWARD_ERR_ARG, "backward_grad_scale: invalid buffer");
    }
    if (!isfinite(scale) || scale <= 0.0f) {
        return fail(BACKWARD_ERR_RANGE, "backward_grad_scale: scale must be finite and positive");
    }
    for (long long i = 0; i < n; ++i) grad[i] *= scale;
    return BACKWARD_OK;
}

/* ------------------------------------------------------------------ */
/* Losses                                                             */
/* ------------------------------------------------------------------ */

/* A row is selected when its mask is set; the mask may be NULL, which means "all
 * rows". A label below zero deselects the row as well, so a caller that passes -1
 * for a padding target does not have to build a mask for it. */
static int row_selected(const uint8_t *mask, const int *labels, int row) {
    if (mask != NULL && mask[row] == 0) return 0;
    if (labels != NULL && labels[row] < 0) return 0;
    return 1;
}

BackwardStatus backward_masked_ce(const float *logits, const int *labels, const uint8_t *mask,
                                  int rows, int vocab, float *d_logits, double *out_sum,
                                  long long *out_count) {
    if (rows < 0 || vocab <= 0 || (rows > 0 && logits == NULL) || labels == NULL) {
        return fail(BACKWARD_ERR_ARG, "backward_masked_ce: invalid shape or null logits/labels");
    }
    /* Two passes: the count has to be final before the gradient is written, or the
     * first rows would be normalised by a smaller denominator than the last. */
    long long count = 0;
    for (int r = 0; r < rows; ++r) {
        if (!row_selected(mask, labels, r)) continue;
        if (labels[r] >= vocab) {
            return fail(BACKWARD_ERR_RANGE,
                        "backward_masked_ce: row %d has label %d outside a vocabulary of %d", r,
                        labels[r], vocab);
        }
        ++count;
    }
    if (out_count != NULL) *out_count = count;
    if (d_logits != NULL) {
        for (long long i = 0; i < (long long)rows * vocab; ++i) d_logits[i] = 0.0f;
    }
    if (count == 0) {
        /* Every row was deselected: the loss is zero and the gradient is zero, which
         * is a result rather than an error. */
        if (out_sum != NULL) *out_sum = 0.0;
        return BACKWARD_OK;
    }

    const float inv_count = 1.0f / (float)count;
    double sum = 0.0;
    for (int r = 0; r < rows; ++r) {
        if (!row_selected(mask, labels, r)) continue;
        const float *row = logits + (size_t)r * vocab;
        float row_max = -INFINITY;
        for (int j = 0; j < vocab; ++j) {
            if (!isfinite(row[j])) {
                return fail(BACKWARD_ERR_DIVERGED,
                            "backward_masked_ce: row %d logit %d is not finite", r, j);
            }
            if (row[j] > row_max) row_max = row[j];
        }
        double exp_sum = 0.0;
        for (int j = 0; j < vocab; ++j) exp_sum += exp((double)(row[j] - row_max));
        if (exp_sum <= 0.0) {
            return fail(BACKWARD_ERR_DIVERGED,
                        "backward_masked_ce: row %d has an empty softmax denominator", r);
        }
        const double log_z = (double)row_max + log(exp_sum);
        sum += log_z - (double)row[labels[r]];
        if (d_logits != NULL) {
            /* The gradient is (mask/count) * (softmax - onehot), the denominator
             * inside the loss rather than left to the caller. */
            float *drow = d_logits + (size_t)r * vocab;
            for (int j = 0; j < vocab; ++j) {
                drow[j] = (float)(exp((double)(row[j] - row_max)) / exp_sum) * inv_count;
            }
            drow[labels[r]] -= inv_count;
        }
    }
    if (out_sum != NULL) *out_sum = sum;
    return BACKWARD_OK;
}

BackwardStatus backward_reverse_kl(const float *student_logits, const float *teacher_logp,
                                   const uint8_t *mask, int rows, int vocab,
                                   float *d_student_logits, double *out_kl) {
    if (rows < 0 || vocab <= 0 || (rows > 0 && (student_logits == NULL || teacher_logp == NULL))) {
        return fail(BACKWARD_ERR_ARG,
                    "backward_reverse_kl: invalid shape or null logits/log-probabilities");
    }
    long long count = 0;
    double total = 0.0;
    for (int r = 0; r < rows; ++r) {
        float *drow = d_student_logits != NULL ? d_student_logits + (size_t)r * vocab : NULL;
        if (drow != NULL) {
            for (int j = 0; j < vocab; ++j) drow[j] = 0.0f;
        }
        if (mask != NULL && mask[r] == 0) continue;
        ++count;
        const float *srow = student_logits + (size_t)r * vocab;
        const float *trow = teacher_logp + (size_t)r * vocab;
        float row_max = -INFINITY;
        for (int j = 0; j < vocab; ++j) {
            if (!isfinite(srow[j]) || !isfinite(trow[j])) {
                return fail(BACKWARD_ERR_DIVERGED,
                            "backward_reverse_kl: row %d entry %d is not finite", r, j);
            }
            if (srow[j] > row_max) row_max = srow[j];
        }
        double exp_sum = 0.0;
        for (int j = 0; j < vocab; ++j) exp_sum += exp((double)(srow[j] - row_max));
        const double log_z = (double)row_max + log(exp_sum);
        /* KL(p||q) = sum p (log p - log q), with log p = z - log Z. */
        double kl = 0.0;
        for (int j = 0; j < vocab; ++j) {
            const double p = exp((double)(srow[j] - row_max)) / exp_sum;
            if (p == 0.0) continue;
            kl += p * ((double)srow[j] - log_z - (double)trow[j]);
        }
        total += kl;
        if (drow != NULL) {
            /* dKL/dz_i = p_i (log p_i - log q_i - KL), a result worth the derivation in
             * the header since it is not the naive (p - q) one might write. The
             * count-normalisation is applied once after the loop, so the gradient is
             * the gradient of the value this function returns rather than of an
             * unnormalised sum that happens to have the same shape. */
            for (int j = 0; j < vocab; ++j) {
                const double p = exp((double)(srow[j] - row_max)) / exp_sum;
                drow[j] = (float)(p * ((double)srow[j] - log_z - (double)trow[j] - kl));
            }
        }
    }
    if (count > 0 && d_student_logits != NULL) {
        const float inv_count = 1.0f / (float)count;
        for (long long i = 0; i < (long long)rows * vocab; ++i) d_student_logits[i] *= inv_count;
    }
    if (out_kl != NULL) *out_kl = count > 0 ? total / (double)count : 0.0;
    return BACKWARD_OK;
}

BackwardStatus backward_clipped_objective(const float *logp, const float *old_logp,
                                          const float *advantage, const uint8_t *mask, int rows,
                                          float clip_low, float clip_high, BackwardClipMode mode,
                                          float *d_logp, float *out_ratio, double *out_objective,
                                          long long *out_selected) {
    if (rows < 0 || (rows > 0 && (logp == NULL || old_logp == NULL || advantage == NULL))) {
        return fail(BACKWARD_ERR_ARG, "backward_clipped_objective: invalid shape or null input");
    }
    if (!isfinite(clip_low) || !isfinite(clip_high) || clip_low < 0.0f || clip_high < 0.0f) {
        return fail(BACKWARD_ERR_RANGE, "backward_clipped_objective: invalid clip range");
    }
    if (mode != BACKWARD_CLIP_TOKEN && mode != BACKWARD_CLIP_SEQUENCE) {
        return fail(BACKWARD_ERR_ARG, "backward_clipped_objective: unknown clip mode");
    }
    long long selected = 0;
    for (int r = 0; r < rows; ++r) {
        if (!row_selected(mask, NULL, r)) {
            if (out_ratio != NULL) out_ratio[r] = 0.0f;
            if (d_logp != NULL) d_logp[r] = 0.0f;
            continue;
        }
        ++selected;
        /* The ratio's log is the difference of the two log-probabilities; computing it
         * once is also what keeps the exponential from overflowing for a large gap. */
        const float log_ratio = logp[r] - old_logp[r];
        if (!isfinite(log_ratio) || !isfinite(advantage[r])) {
            return fail(BACKWARD_ERR_DIVERGED,
                        "backward_clipped_objective: row %d has a non-finite ratio or advantage",
                        r);
        }
    }
    if (selected == 0) {
        if (out_objective != NULL) *out_objective = 0.0;
        if (out_selected != NULL) *out_selected = 0;
        return BACKWARD_OK;
    }

    const float inv_count = 1.0f / (float)selected;
    /* The ratio the hinge uses: per row for TOKEN, and one shared ratio for the whole
     * block for SEQUENCE (which is what makes the objective a sequence-level one). */
    double shared_log_ratio = 0.0;
    if (mode == BACKWARD_CLIP_SEQUENCE) {
        for (int r = 0; r < rows; ++r) {
            if (!row_selected(mask, NULL, r)) continue;
            shared_log_ratio += (double)(logp[r] - old_logp[r]) * (double)inv_count;
        }
    }
    const float seq_ratio = (float)exp(shared_log_ratio);

    /* For a sequence-level ratio every selected row's log-probability moves the one
     * ratio, so the gradient is shared: d loss/d logp_t = -(1/N^2) * ratio * sum_r h_r',
     * where h_r' is the row's advantage in the hinge's linear region and zero outside
     * it. Computing the sum first is what makes the shared term correct rather than an
     * approximation of it. */
    double sequence_hinge_slope = 0.0;

    double objective = 0.0;
    for (int r = 0; r < rows; ++r) {
        if (!row_selected(mask, NULL, r)) continue;
        const float ratio = mode == BACKWARD_CLIP_SEQUENCE ? seq_ratio
                                                          : (float)exp((double)(logp[r] - old_logp[r]));
        const float a = advantage[r];
        if (out_ratio != NULL) out_ratio[r] = ratio;
        const float unclipped = ratio * a;
        float clipped_limit;
        float clipped;
        if (a > 0.0f) {
            clipped_limit = 1.0f + clip_high;
            clipped = fminf(ratio, clipped_limit) * a;
        } else {
            clipped_limit = 1.0f - clip_low;
            clipped = fmaxf(ratio, clipped_limit) * a;
        }
        const float hinge = a > 0.0f ? fminf(unclipped, clipped) : fmaxf(unclipped, clipped);
        objective += (double)hinge * (double)inv_count;
        /* In the clipped region the hinge is flat in the ratio, so the gradient is
         * zero; the test is the hinge's own condition rather than a comparison of the
         * two values, because exp() rarely lands exactly on the boundary. */
        int linear;
        if (a > 0.0f) {
            linear = ratio <= clipped_limit;
        } else if (a < 0.0f) {
            linear = ratio >= clipped_limit;
        } else {
            linear = 0; /* a zero advantage has no gradient whichever branch is taken */
        }
        if (linear) sequence_hinge_slope += (double)a;
        if (d_logp == NULL) continue;
        if (!linear) {
            d_logp[r] = 0.0f;
        } else if (mode == BACKWARD_CLIP_TOKEN) {
            /* dloss/dlogp = -(mask/count) * A * ratio. */
            d_logp[r] = -inv_count * a * ratio;
        }
        /* SEQUENCE mode is filled in after the loop, once the shared slope is known. */
    }
    if (d_logp != NULL && mode == BACKWARD_CLIP_SEQUENCE) {
        for (int r = 0; r < rows; ++r) {
            if (!row_selected(mask, NULL, r)) continue;
            const float a = advantage[r];
            const float ratio = seq_ratio;
            float clipped_limit = a > 0.0f ? 1.0f + clip_high : 1.0f - clip_low;
            int linear = a > 0.0f ? (ratio <= clipped_limit)
                                  : (a < 0.0f ? (ratio >= clipped_limit) : 0);
            d_logp[r] = linear ? (float)(-(double)inv_count * inv_count * (double)seq_ratio *
                                         sequence_hinge_slope)
                               : 0.0f;
        }
    }
    if (out_objective != NULL) *out_objective = objective;
    if (out_selected != NULL) *out_selected = selected;
    return BACKWARD_OK;
}

BackwardStatus backward_group_advantage(const float *rewards, const uint8_t *mask, int count,
                                        float eps, int allow_zero_variance, float *out_advantages,
                                        double *out_mean, double *out_std) {
    if (count <= 0 || rewards == NULL || out_advantages == NULL) {
        return fail(BACKWARD_ERR_ARG, "backward_group_advantage: a group needs at least one reward");
    }
    if (!isfinite(eps) || eps < 0.0f) {
        return fail(BACKWARD_ERR_RANGE, "backward_group_advantage: invalid epsilon");
    }
    long long selected = 0;
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        if (mask != NULL && mask[i] == 0) continue;
        if (!isfinite(rewards[i])) {
            return fail(BACKWARD_ERR_DIVERGED, "backward_group_advantage: reward %d is not finite",
                        i);
        }
        sum += rewards[i];
        ++selected;
    }
    if (selected == 0) {
        return fail(BACKWARD_ERR_STATE, "backward_group_advantage: the group has no selected reward");
    }
    const double mean = sum / (double)selected;
    double variance = 0.0;
    for (int i = 0; i < count; ++i) {
        if (mask != NULL && mask[i] == 0) continue;
        const double d = (double)rewards[i] - mean;
        variance += d * d;
    }
    variance /= (double)selected; /* population variance: the group is the population */
    const double std = sqrt(variance);
    if (std <= 0.0 && !allow_zero_variance) {
        return fail(BACKWARD_ERR_STATE,
                    "backward_group_advantage: the group has zero variance (all rewards equal); "
                    "scaling by eps would turn a zero advantage into 0/eps and call it a signal");
    }
    for (int i = 0; i < count; ++i) {
        if (mask != NULL && mask[i] == 0) {
            out_advantages[i] = 0.0f;
            continue;
        }
        out_advantages[i] = (float)(((double)rewards[i] - mean) / (std + (double)eps));
    }
    if (out_mean != NULL) *out_mean = mean;
    if (out_std != NULL) *out_std = std;
    return BACKWARD_OK;
}

/* ------------------------------------------------------------------ */
/* AdamW                                                             */
/* ------------------------------------------------------------------ */

BackwardStatus backward_adamw_step(float *master, const float *grad, float *m_slot, float *v_slot,
                                   long long n, const struct BackwardAdamWHyper *hyper,
                                   int step_index, uint16_t *out_bf16, long long *out_changed) {
    if (n < 0 || hyper == NULL) {
        return fail(BACKWARD_ERR_ARG, "backward_adamw_step: invalid shape or hyper-parameters");
    }
    if (n > 0 && (master == NULL || grad == NULL || m_slot == NULL || v_slot == NULL)) {
        return fail(BACKWARD_ERR_ARG, "backward_adamw_step: null buffer");
    }
    if (step_index < 1) {
        return fail(BACKWARD_ERR_STATE, "backward_adamw_step: step_index is 1-based");
    }
    const struct BackwardAdamWHyper h = *hyper;
    if (!isfinite(h.lr) || !isfinite(h.beta1) || !isfinite(h.beta2) || !isfinite(h.eps) ||
        !isfinite(h.weight_decay) || h.lr < 0.0f || h.beta1 < 0.0f || h.beta1 >= 1.0f ||
        h.beta2 < 0.0f || h.beta2 >= 1.0f || h.eps <= 0.0f || h.weight_decay < 0.0f) {
        return fail(BACKWARD_ERR_RANGE, "backward_adamw_step: hyper-parameters out of range");
    }
    /* A NaN in the gradient is the one failure this stage must not paper over: the
     * next step would silently learn nothing, so it is refused before any slot moves. */
    for (long long i = 0; i < n; ++i) {
        if (!isfinite(grad[i])) {
            return fail(BACKWARD_ERR_DIVERGED,
                        "backward_adamw_step: gradient element %lld is not finite", i);
        }
    }

    const double beta1 = (double)h.beta1;
    const double beta2 = (double)h.beta2;
    const double bc1 = 1.0 - pow(beta1, (double)step_index);
    const double bc2 = 1.0 - pow(beta2, (double)step_index);
    const double sqrt_bc2 = sqrt(bc2);
    const double step_size = (double)h.lr / bc1;

    long long changed = 0;
    for (long long i = 0; i < n; ++i) {
        const double g = (double)grad[i];
        const double m = beta1 * (double)m_slot[i] + (1.0 - beta1) * g;
        const double v = beta2 * (double)v_slot[i] + (1.0 - beta2) * g * g;
        m_slot[i] = (float)m;
        v_slot[i] = (float)v;
        /* Torch's order: the decoupled decay multiplies the weight first, and eps is
         * added after the second moment is divided by sqrt(bc2). Both small choices
         * change the step by more than the FP32 noise floor near a zero moment. */
        double theta = (double)master[i] * (1.0 - (double)h.lr * (double)h.weight_decay);
        theta -= step_size * m / (sqrt(v) / sqrt_bc2 + (double)h.eps);
        if (!isfinite(theta)) {
            return fail(BACKWARD_ERR_DIVERGED,
                        "backward_adamw_step: element %lld produced a non-finite weight", i);
        }
        const float updated = (float)theta;
        if (out_bf16 != NULL) {
            /* The single rounding of publication: FP32 master to BF16 compute
             * (round-to-nearest-even). Both are available so the caller can publish
             * without a second pass, and the count of changed elements is what "exactly
             * one optimizer update per parameter" means for a tied parameter whose two
             * readers must stay identical. */
            uint32_t bits;
            memcpy(&bits, &updated, 4);
            const uint32_t lsb = (bits >> 16) & 1u;
            const uint32_t rounded = bits + 0x7fffu + lsb;
            const uint16_t bf16 = (uint16_t)(rounded >> 16);
            if (bf16 != out_bf16[i]) ++changed;
            out_bf16[i] = bf16;
        } else if (updated != master[i]) {
            ++changed;
        }
        master[i] = updated;
    }
    if (out_changed != NULL) *out_changed = changed;
    return BACKWARD_OK;
}

/* ------------------------------------------------------------------ */
/* Reproducibility: RNG and the data cursor                           */
/* ------------------------------------------------------------------ */

/* splitmix64, evaluated from (seed, counter) rather than from a mutable state, so the
 * n-th draw is a function of the seed and n and does not depend on how many draws
 * some other part of the code made. */
#define BACKWARD_SPLITMIX_GOLDEN 0x9E3779B97F4A7C15ULL

static uint64_t splitmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void backward_rng_seed(struct BackwardRng *rng, uint64_t seed) {
    if (rng == NULL) return;
    rng->seed = seed;
    rng->counter = 0;
}

uint64_t backward_rng_next(struct BackwardRng *rng) {
    if (rng == NULL) return 0;
    const uint64_t value = splitmix64(rng->seed + rng->counter * BACKWARD_SPLITMIX_GOLDEN);
    rng->counter += 1;
    return value;
}

double backward_rng_uniform(struct BackwardRng *rng) {
    /* 53 bits of mantissa, the transform the sampling gates assume: the difference
     * between this and a 24-bit float draw is visible in a frequency test. */
    return (double)(backward_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

/* ------------------------------------------------------------------ */
/* Checkpoint / resume                                                */
/* ------------------------------------------------------------------ */

/* A table-driven CRC-32 (IEEE 802.3, reflected). A 32-bit checksum cannot detect a
 * malicious edit and is not meant to: its job is to turn a truncated, reordered or
 * half-written checkpoint into a refusal instead of a different model. A production
 * checkpoint would carry the in-tree SHA-256 (csrc/sha256.c) for the same reason the
 * manifest does. */
static uint32_t crc32_table[256];
static int crc32_ready = 0;

static void crc32_init(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
    crc32_ready = 1;
}

uint32_t backward_checksum(const void *data, size_t size) {
    if (!crc32_ready) crc32_init();
    const unsigned char *bytes = (const unsigned char *)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) crc = crc32_table[(crc ^ bytes[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

/* The format is little-endian and fixed: magic, version, the store version and step,
 * the RNG and cursor, then each parameter's logical id, element count and three FP32
 * arrays, then a checksum over everything before it. Fixed-width fields rather than
 * the C struct's layout, so a checkpoint written by one build is readable by another
 * compiled with different padding. */
#define BACKWARD_MAGIC "HIBCKPT1"
#define BACKWARD_MAGIC_LEN 8
#define BACKWARD_FORMAT_VERSION 1u

struct Writer {
    unsigned char *base;
    size_t capacity;
    size_t offset;
    int overflow;
};

static void put_bytes(struct Writer *w, const void *src, size_t n) {
    if (w->overflow || w->offset + n > w->capacity) {
        w->overflow = 1;
        return;
    }
    memcpy(w->base + w->offset, src, n);
    w->offset += n;
}

static void put_u32(struct Writer *w, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v & 0xFFu), (unsigned char)((v >> 8) & 0xFFu),
                          (unsigned char)((v >> 16) & 0xFFu), (unsigned char)((v >> 24) & 0xFFu)};
    put_bytes(w, b, 4);
}

static void put_i32(struct Writer *w, int32_t v) { put_u32(w, (uint32_t)v); }

static void put_u64(struct Writer *w, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = (unsigned char)((v >> (8 * i)) & 0xFFu);
    put_bytes(w, b, 8);
}

static void put_i64(struct Writer *w, long long v) { put_u64(w, (uint64_t)v); }

static void put_f32_array(struct Writer *w, const float *values, long long n) {
    for (long long i = 0; i < n && !w->overflow; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], 4);
        put_u32(w, bits);
    }
}

struct Reader {
    const unsigned char *base;
    size_t size;
    size_t offset;
    int underflow;
};

static int get_bytes(struct Reader *r, void *dst, size_t n) {
    if (r->underflow || r->offset + n > r->size) {
        r->underflow = 1;
        return 0;
    }
    memcpy(dst, r->base + r->offset, n);
    r->offset += n;
    return 1;
}

static uint32_t get_u32(struct Reader *r) {
    unsigned char b[4] = {0, 0, 0, 0};
    if (!get_bytes(r, b, 4)) return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static int32_t get_i32(struct Reader *r) { return (int32_t)get_u32(r); }

static uint64_t get_u64(struct Reader *r) {
    unsigned char b[8] = {0};
    if (!get_bytes(r, b, 8)) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

static long long get_i64(struct Reader *r) { return (long long)get_u64(r); }

static int get_f32_array(struct Reader *r, float *values, long long n) {
    for (long long i = 0; i < n; ++i) {
        const uint32_t bits = get_u32(r);
        if (r->underflow) return 0;
        memcpy(&values[i], &bits, 4);
    }
    return 1;
}

static int checkpoint_plan_valid(const struct BackwardCheckpoint *ckpt) {
    if (ckpt == NULL) return 0;
    if (ckpt->param_count < 0) return 0;
    if (ckpt->param_count > 0 && ckpt->params == NULL) return 0;
    if (ckpt->step_index < 0) return 0;
    if (ckpt->cursor.sample < 0) return 0;
    for (int i = 0; i < ckpt->param_count; ++i) {
        const struct BackwardCheckpointParam *p = &ckpt->params[i];
        if (p->elements < 0) return 0;
        if (p->elements > 0 && (p->master == NULL || p->m_slot == NULL || p->v_slot == NULL)) {
            return 0;
        }
    }
    return 1;
}

BackwardStatus backward_checkpoint_size(const struct BackwardCheckpoint *ckpt, size_t *out_bytes) {
    if (out_bytes == NULL) {
        return fail(BACKWARD_ERR_ARG, "backward_checkpoint_size: null output");
    }
    if (!checkpoint_plan_valid(ckpt)) {
        return fail(BACKWARD_ERR_ARG, "backward_checkpoint_size: invalid checkpoint plan");
    }
    /* magic + abi + format + param_count + store_version + step_index + rng(2) +
     * cursor(2) + checksum, plus per parameter: logical + elements + 3 FP32 arrays. */
    size_t total = BACKWARD_MAGIC_LEN + 4 + 4 + 4 + 8 + 4 + 8 + 8 + 8 + 4 + 4;
    for (int i = 0; i < ckpt->param_count; ++i) {
        const long long elements = ckpt->params[i].elements;
        if (elements > 0 &&
            (size_t)elements > (SIZE_MAX - total) / (3 * sizeof(float))) {
            return fail(BACKWARD_ERR_RANGE, "backward_checkpoint_size: plan overflows size_t");
        }
        total += 4 + 8 + (size_t)elements * 3 * sizeof(float);
    }
    *out_bytes = total;
    return BACKWARD_OK;
}

BackwardStatus backward_checkpoint_write(const struct BackwardCheckpoint *ckpt, void *buffer,
                                         size_t capacity, size_t *out_written) {
    if (buffer == NULL) {
        return fail(BACKWARD_ERR_ARG, "backward_checkpoint_write: null buffer");
    }
    size_t needed = 0;
    const BackwardStatus sized = backward_checkpoint_size(ckpt, &needed);
    if (sized != BACKWARD_OK) return sized;
    if (capacity < needed) {
        return fail(BACKWARD_ERR_NO_SPACE,
                    "backward_checkpoint_write: need %zu bytes, have %zu", needed, capacity);
    }

    struct Writer w = {(unsigned char *)buffer, capacity, 0, 0};
    put_bytes(&w, BACKWARD_MAGIC, BACKWARD_MAGIC_LEN);
    put_u32(&w, BACKWARD_ABI_VERSION);
    put_u32(&w, BACKWARD_FORMAT_VERSION);
    put_u32(&w, (uint32_t)ckpt->param_count);
    put_i64(&w, ckpt->store_version);
    put_i32(&w, ckpt->step_index);
    put_u64(&w, ckpt->rng.seed);
    put_u64(&w, ckpt->rng.counter);
    put_i64(&w, ckpt->cursor.sample);
    put_i32(&w, ckpt->cursor.epoch);
    for (int i = 0; i < ckpt->param_count; ++i) {
        const struct BackwardCheckpointParam *p = &ckpt->params[i];
        put_i32(&w, p->logical);
        put_i64(&w, p->elements);
        put_f32_array(&w, p->master, p->elements);
        put_f32_array(&w, p->m_slot, p->elements);
        put_f32_array(&w, p->v_slot, p->elements);
    }
    if (w.overflow) {
        return fail(BACKWARD_ERR_NO_SPACE, "backward_checkpoint_write: the payload overflowed");
    }
    /* The checksum covers everything before its own four bytes. */
    const uint32_t checksum = backward_checksum(buffer, w.offset);
    put_u32(&w, checksum);
    if (out_written != NULL) *out_written = w.offset;
    return BACKWARD_OK;
}

BackwardStatus backward_checkpoint_read(struct BackwardCheckpoint *ckpt, const void *buffer,
                                        size_t size, size_t *out_consumed) {
    if (ckpt == NULL || buffer == NULL) {
        return fail(BACKWARD_ERR_ARG, "backward_checkpoint_read: null checkpoint or buffer");
    }
    /* Validate everything before copying anything: a resume that writes half of a
     * different run's parameters is worse than refusing to resume. */
    struct Reader r = {(const unsigned char *)buffer, size, 0, 0};
    char magic[BACKWARD_MAGIC_LEN];
    if (!get_bytes(&r, magic, BACKWARD_MAGIC_LEN) ||
        memcmp(magic, BACKWARD_MAGIC, BACKWARD_MAGIC_LEN) != 0) {
        return fail(BACKWARD_ERR_FORMAT, "backward_checkpoint_read: bad magic");
    }
    const uint32_t abi = get_u32(&r);
    const uint32_t format = get_u32(&r);
    const uint32_t param_count = get_u32(&r);
    const long long store_version = get_i64(&r);
    const int32_t step_index = get_i32(&r);
    const uint64_t rng_seed = get_u64(&r);
    const uint64_t rng_counter = get_u64(&r);
    const long long cursor_sample = get_i64(&r);
    const int32_t cursor_epoch = get_i32(&r);
    if (r.underflow) {
        return fail(BACKWARD_ERR_FORMAT, "backward_checkpoint_read: header is truncated");
    }
    if (abi != BACKWARD_ABI_VERSION) {
        return fail(BACKWARD_ERR_FORMAT,
                    "backward_checkpoint_read: ABI %u, this build writes %u", abi,
                    (unsigned)BACKWARD_ABI_VERSION);
    }
    if (format != BACKWARD_FORMAT_VERSION) {
        return fail(BACKWARD_ERR_FORMAT, "backward_checkpoint_read: unknown format version %u",
                    (unsigned)format);
    }
    if (ckpt->params == NULL && param_count > 0) {
        return fail(BACKWARD_ERR_ARG, "backward_checkpoint_read: the caller supplied no parameter plan");
    }
    if ((int)param_count != ckpt->param_count) {
        return fail(BACKWARD_ERR_FORMAT,
                    "backward_checkpoint_read: the checkpoint holds %u parameters, the caller's plan "
                    "has %d",
                    (unsigned)param_count, ckpt->param_count);
    }
    /* The layout has to match in order, not just in count: resuming the second
     * parameter's moments into the first parameter is the failure this checks. */
    const size_t payload_end = size >= 4 ? size - 4 : 0;
    for (uint32_t i = 0; i < param_count; ++i) {
        const int32_t logical = get_i32(&r);
        const long long elements = get_i64(&r);
        if (r.underflow) {
            return fail(BACKWARD_ERR_FORMAT, "backward_checkpoint_read: parameter %u is truncated",
                        (unsigned)i);
        }
        const struct BackwardCheckpointParam *p = &ckpt->params[i];
        if (logical != p->logical) {
            return fail(BACKWARD_ERR_FORMAT,
                        "backward_checkpoint_read: parameter %u is logical %d, the plan expects %d",
                        (unsigned)i, logical, p->logical);
        }
        if (elements != p->elements) {
            return fail(BACKWARD_ERR_FORMAT,
                        "backward_checkpoint_read: parameter %d holds %lld elements, the plan "
                        "expects %lld",
                        logical, elements, p->elements);
        }
        if (elements < 0 || r.offset + (size_t)elements * 3 * sizeof(float) > payload_end) {
            return fail(BACKWARD_ERR_FORMAT,
                        "backward_checkpoint_read: parameter %d overruns the buffer", logical);
        }
        r.offset += (size_t)elements * 3 * sizeof(float);
    }
    if (r.offset != payload_end) {
        return fail(BACKWARD_ERR_FORMAT,
                    "backward_checkpoint_read: %zu payload bytes were not consumed by the plan",
                    payload_end - r.offset);
    }
    uint32_t stored_checksum = 0;
    if (size < 4) {
        return fail(BACKWARD_ERR_FORMAT, "backward_checkpoint_read: the checksum is missing");
    }
    memcpy(&stored_checksum, (const unsigned char *)buffer + payload_end, 4);
    const uint32_t computed = backward_checksum(buffer, payload_end);
    if (stored_checksum != computed) {
        return fail(BACKWARD_ERR_FORMAT,
                    "backward_checkpoint_read: checksum %08x, recomputed %08x (truncated or "
                    "reordered)",
                    stored_checksum, computed);
    }

    /* The plan and the payload agree; now copy. */
    struct Reader copy = {(const unsigned char *)buffer, size, 0, 0};
    (void)get_bytes(&copy, magic, BACKWARD_MAGIC_LEN);
    (void)get_u32(&copy); /* abi */
    (void)get_u32(&copy); /* format */
    (void)get_u32(&copy); /* param_count */
    (void)get_i64(&copy);
    (void)get_i32(&copy);
    (void)get_u64(&copy);
    (void)get_u64(&copy);
    (void)get_i64(&copy);
    (void)get_i32(&copy);
    for (uint32_t i = 0; i < param_count; ++i) {
        const struct BackwardCheckpointParam *p = &ckpt->params[i];
        float *master = p->master;
        float *m_slot = p->m_slot;
        float *v_slot = p->v_slot;
        (void)get_i32(&copy);
        (void)get_i64(&copy);
        get_f32_array(&copy, master, p->elements);
        get_f32_array(&copy, m_slot, p->elements);
        get_f32_array(&copy, v_slot, p->elements);
    }
    ckpt->store_version = store_version;
    ckpt->step_index = step_index;
    ckpt->rng.seed = rng_seed;
    ckpt->rng.counter = rng_counter;
    ckpt->cursor.sample = cursor_sample;
    ckpt->cursor.epoch = cursor_epoch;
    if (out_consumed != NULL) *out_consumed = size;
    return BACKWARD_OK;
}
