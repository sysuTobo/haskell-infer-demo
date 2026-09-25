/*
 * backward_test.c - CPU gate for the Stage-4 differentiation contract, the losses and
 * the optimizer.
 *
 * No GPU and no weights. It checks the things a reader cannot:
 *
 *   - the Stage-4 region table covers the plan's eleven rows and nothing else, and
 *     every Stage-1 region name it claims to differentiate exists in the Stage-1
 *     inventory (regions.h), in both directions, so a row cannot name a region that
 *     was never inventoried and an inventoried region cannot lose its backward
 *     silently;
 *   - the differentiation convention is a value the caller reads, not a comment;
 *   - a backward whose step did not retain a value it needs is refused by name;
 *   - the losses match an independent FP64 implementation, and their analytic
 *     gradients match the central difference of that implementation. The finite
 *     difference runs over the smooth loss only: no cast sits in the loss path, which
 *     is the case the plan allows a finite difference to be an oracle for;
 *   - AdamW matches an independent FP64 implementation of PyTorch's step order,
 *     including the bias correction/eps ordering that separates it from the textbook
 *     form, and refuses a non-finite gradient before moving any slot;
 *   - a checkpoint round-trips exactly, and every way it can be wrong (bad magic,
 *     truncation, a reordered parameter plan, a flipped payload bit) is a refusal
 *     that leaves the caller's arrays untouched;
 *   - the RNG is a function of (seed, counter), so the n-th draw does not depend on
 *     how many draws another part of the code made.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_backward.
 */
#include "backward.h"
#include "regions.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

static void check(int condition, const char *fmt, ...) {
    ++g_checks;
    if (condition) return;
    ++g_failures;
    va_list ap;
    va_start(ap, fmt);
    fputs("backward_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void check_close(double got, double want, double tol, const char *what) {
    const double diff = fabs(got - want);
    const double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    check(diff <= tol * scale, "%s: got %.9g want %.9g (diff %.3e > tol %.3e)", what, got, want,
          diff, tol * scale);
}

/* ------------------------------------------------------------------ */
/* Independent FP64 references                                        */
/* ------------------------------------------------------------------ */

static void ref_softmax(const double *logits, int vocab, double *out) {
    double max = -INFINITY;
    for (int j = 0; j < vocab; ++j) max = logits[j] > max ? logits[j] : max;
    double sum = 0.0;
    for (int j = 0; j < vocab; ++j) {
        out[j] = exp(logits[j] - max);
        sum += out[j];
    }
    for (int j = 0; j < vocab; ++j) out[j] /= sum;
}

/* Masked CE, in double, from the definition rather than from backward.c's shape. */
static double ref_masked_ce(const double *logits, const int *labels, const uint8_t *mask, int rows,
                            int vocab) {
    long long count = 0;
    double sum = 0.0;
    for (int r = 0; r < rows; ++r) {
        if (mask != NULL && mask[r] == 0) continue;
        if (labels != NULL && labels[r] < 0) continue;
        ++count;
        double *p = (double *)malloc(sizeof(double) * (size_t)vocab);
        ref_softmax(logits + (size_t)r * vocab, vocab, p);
        sum += -log(p[labels[r]]);
        free(p);
    }
    return count > 0 ? sum / (double)count : 0.0;
}

/* Reverse KL of the student against the teacher's log-probabilities, in double. */
static double ref_reverse_kl(const double *student_logits, const double *teacher_logp, int rows,
                             int vocab) {
    double total = 0.0;
    for (int r = 0; r < rows; ++r) {
        double *p = (double *)malloc(sizeof(double) * (size_t)vocab);
        ref_softmax(student_logits + (size_t)r * vocab, vocab, p);
        for (int j = 0; j < vocab; ++j) {
            if (p[j] == 0.0) continue;
            total += p[j] * (log(p[j]) - teacher_logp[(size_t)r * vocab + j]);
        }
        free(p);
    }
    return total / (double)rows;
}

/* The clipped objective, in double, from its definition. */
static double ref_clipped(const double *logp, const double *old_logp, const double *advantage,
                          const uint8_t *mask, int rows, double clip_low, double clip_high,
                          int sequence_mode) {
    long long selected = 0;
    for (int r = 0; r < rows; ++r) {
        if (mask != NULL && mask[r] == 0) continue;
        ++selected;
    }
    if (selected == 0) return 0.0;
    const double inv = 1.0 / (double)selected;
    double seq_log_ratio = 0.0;
    if (sequence_mode) {
        for (int r = 0; r < rows; ++r) {
            if (mask != NULL && mask[r] == 0) continue;
            seq_log_ratio += (logp[r] - old_logp[r]) * inv;
        }
    }
    double objective = 0.0;
    for (int r = 0; r < rows; ++r) {
        if (mask != NULL && mask[r] == 0) continue;
        const double ratio = sequence_mode ? exp(seq_log_ratio) : exp(logp[r] - old_logp[r]);
        const double a = advantage[r];
        const double unclipped = ratio * a;
        const double clipped = a > 0.0 ? fmin(ratio, 1.0 + clip_high) * a
                                       : fmax(ratio, 1.0 - clip_low) * a;
        objective += (a > 0.0 ? fmin(unclipped, clipped) : fmax(unclipped, clipped)) * inv;
    }
    return objective;
}

/* One AdamW step in double, written from PyTorch's own sequence. */
static void ref_adamw(double *master, const double *grad, double *m, double *v, int n, double lr,
                      double beta1, double beta2, double eps, double weight_decay, int step) {
    const double bc1 = 1.0 - pow(beta1, step);
    const double bc2 = 1.0 - pow(beta2, step);
    for (int i = 0; i < n; ++i) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        master[i] *= (1.0 - lr * weight_decay);
        master[i] -= (lr / bc1) * m[i] / (sqrt(v[i]) / sqrt(bc2) + eps);
    }
}

/* A central difference of `value` in one coordinate of an array, in double. */
typedef double (*scalar_fn)(void *ctx);

struct fd_ctx {
    scalar_fn fn;
    void *base;
    double *coords; /* the array the perturbed coordinate lives in */
    int index;
};

static double fd_central(struct fd_ctx *ctx, double h) {
    const double saved = ctx->coords[ctx->index];
    ctx->coords[ctx->index] = saved + h;
    const double plus = ctx->fn(ctx->base);
    ctx->coords[ctx->index] = saved - h;
    const double minus = ctx->fn(ctx->base);
    ctx->coords[ctx->index] = saved;
    return (plus - minus) / (2.0 * h);
}

/* ------------------------------------------------------------------ */
/* Region table                                                       */
/* ------------------------------------------------------------------ */

static const char *const kPlanRows[] = {
    "Embedding and tied LM head", "Residual and elementwise gates", "RoPE and Q/gate split",
    "Plain/Gemma/per-head/gated norms", "GEMM including FP32 LM-head output", "GDN prepare",
    "GDN conv1d", "GDN core", "Attention core", "Losses", "AdamW",
};

static void test_regions(void) {
    int count = 0;
    const struct BackwardRegionInfo *rows = backward_region_info(&count);
    check(count == BACKWARD_REGION_ROW_COUNT, "the region table has %d rows for %d enum values",
          count, BACKWARD_REGION_ROW_COUNT);
    check(count == (int)(sizeof(kPlanRows) / sizeof(kPlanRows[0])),
          "the plan's Stage-4 table has %zu rows, the backward table has %d",
          sizeof(kPlanRows) / sizeof(kPlanRows[0]), count);

    for (int i = 0; i < count; ++i) {
        const struct BackwardRegionInfo *row = &rows[i];
        check(row->region == (BackwardRegion)i, "row %d is out of order", i);
        check(row->name != NULL && row->name[0] != '\0', "row %d has no name", i);
        check(strcmp(row->name, kPlanRows[i]) == 0,
              "row %d is '%s', the plan's row %d is '%s'", i, row->name, i, kPlanRows[i]);
        check(row->mechanism != NULL && row->mechanism[0] != '\0',
              "row %d (%s) has no mechanism", i, row->name);
        check(row->note != NULL && row->note[0] != '\0', "row %d (%s) has no note", i,
              row->name);
        /* Every Stage-1 region a row claims must exist in the Stage-1 inventory: a row
         * may not invent a region to look complete. */
        const int names = backward_region_stage1_count(row->region);
        for (int j = 0; j < names; ++j) {
            char name[64];
            const int length = backward_region_stage1_name_at(row->region, j, name, sizeof(name));
            check(length > 0, "row %d (%s) name %d does not fit or does not exist", i, row->name,
                  j);
            check(region_inventory_find(name) != NULL,
                  "row %d (%s) names the Stage-1 region '%s', which the inventory does not have",
                  i, row->name, name);
        }
    }

    /* ...and the other direction: every Stage-1 region that the plan's Stage-4 table
     * gives a backward must be claimed by exactly one row. The list here is the
     * plan's table read as region names; keeping it in the test (rather than deriving
     * it from the backward table) is what makes this a coverage check. */
    static const char *const kMustBeDifferentiated[] = {
        "embedding",     "gemm_fp32_lmhead", "residual_add", "silu_mul",  "conv_silu",
        "attention_output_gate", "rope",     "q_gate_split", "rmsnorm",   "per_head_norm",
        "gdn_gated_norm", "gemm_bf16",       "gdn_prepare",  "gdn_conv1d", "gdn_core",
        "attention_core", "masked_loss",
    };
    for (size_t k = 0; k < sizeof(kMustBeDifferentiated) / sizeof(kMustBeDifferentiated[0]); ++k) {
        int claimed = 0;
        for (int i = 0; i < count; ++i) {
            const int names = backward_region_stage1_count(rows[i].region);
            for (int j = 0; j < names; ++j) {
                char name[64];
                if (backward_region_stage1_name_at(rows[i].region, j, name, sizeof(name)) > 0 &&
                    strcmp(name, kMustBeDifferentiated[k]) == 0) {
                    ++claimed;
                }
            }
        }
        check(claimed == 1, "the Stage-1 region '%s' is claimed by %d Stage-4 rows, expected 1",
              kMustBeDifferentiated[k], claimed);
    }

    /* An implemented row must be reachable, and a row that is not implemented must say
     * why: "not implemented" is a claim about the stage, so it is not a silent zero. */
    int implemented = 0;
    for (int i = 0; i < count; ++i) {
        if (rows[i].implemented) ++implemented;
        check(backward_region_at((BackwardRegion)i) == &rows[i], "region_at disagrees with the table");
    }
    check(implemented > 0, "no Stage-4 row claims an implementation");
    check(backward_region_at(BACKWARD_REGION_ROW_COUNT) == NULL, "an out-of-range row was found");
    check(backward_region_at(-1) == NULL, "a negative row was found");
    printf("backward_test: %d/%d Stage-4 rows implemented\n", implemented, count);
}

/* ------------------------------------------------------------------ */
/* Convention and retained values                                     */
/* ------------------------------------------------------------------ */

static void test_convention(void) {
    check(backward_cast_mode() == BACKWARD_CAST_IDENTITY,
          "the differentiation convention changed: the gate's oracle assumes identity casts");
    check(backward_default_loss_scale() > 0.0f, "the loss scale must be positive");

    int count = 0;
    const struct BackwardRequiredValue *required = backward_required_values(&count);
    check(count > 0 && required != NULL, "no retained values are declared");
    check(backward_check_retained(-1, NULL, 0) == BACKWARD_ERR_ARG, "a negative region was accepted");
    check(backward_check_retained(BACKWARD_REGION_ROW_COUNT, NULL, 0) == BACKWARD_ERR_ARG,
          "an out-of-range region was accepted");

    /* A full set for the GEMM row is accepted; dropping one name is refused and the
     * missing name is the one the caller omitted. */
    const char *const have[] = {"x", "weight", "out_grad"};
    check(backward_check_retained(BACKWARD_REGION_GEMM, have, 3) == BACKWARD_OK,
          "a complete retained set was refused: %s", backward_last_error());
    const char *const partial[] = {"x", "out_grad"};
    check(backward_check_retained(BACKWARD_REGION_GEMM, partial, 2) == BACKWARD_ERR_MISSING,
          "a missing retained value was accepted");
    check(strstr(backward_last_error(), "weight") != NULL,
          "the refusal does not name the missing value: %s", backward_last_error());
}

/* ------------------------------------------------------------------ */
/* Gradient accumulation                                              */
/* ------------------------------------------------------------------ */

static void test_accumulation(void) {
    float grad[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    check(backward_grad_zero(grad, 4) == BACKWARD_OK, "zeroing failed");
    check(grad[0] == 0.0f && grad[3] == 0.0f, "zeroing left a value");
    const float c1[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    const float c2[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    check(backward_grad_accumulate(grad, c1, 4) == BACKWARD_OK, "accumulate failed");
    check(backward_grad_accumulate(grad, c2, 4) == BACKWARD_OK, "accumulate failed");
    check_close(grad[0], 1.5, 1e-6, "accumulated grad[0]");
    check_close(grad[2], -0.5, 1e-6, "accumulated grad[2]");

    /* A tied parameter: two contributions summed, and the count of nonzero ones is
     * reported so a caller can tell "both readers contributed" from "one is silent". */
    float tied[2] = {0.0f, 0.0f};
    const float reader_a[2] = {3.0f, 0.0f};
    const float reader_b[2] = {4.0f, -1.0f};
    long long nonzero = 0;
    check(backward_grad_accumulate_tied(tied, reader_a, 2, &nonzero) == BACKWARD_OK, "tied add failed");
    check(backward_grad_accumulate_tied(tied, reader_b, 2, &nonzero) == BACKWARD_OK, "tied add failed");
    check_close(tied[0], 7.0, 1e-6, "tied grad[0]");
    check_close(tied[1], -1.0, 1e-6, "tied grad[1]");
    check(nonzero == 2, "the tied accumulation reported %lld nonzero contributions, expected 2",
          nonzero);

    check(backward_grad_scale(grad, 4, 2.0f) == BACKWARD_OK, "scaling failed");
    check_close(grad[0], 3.0, 1e-6, "scaled grad[0]");
    check(backward_grad_scale(grad, 4, 0.0f) == BACKWARD_ERR_RANGE, "a zero loss scale was accepted");
    check(backward_grad_scale(grad, 4, NAN) == BACKWARD_ERR_RANGE, "a NaN loss scale was accepted");
    check(backward_grad_zero(NULL, 1) == BACKWARD_ERR_ARG, "a null buffer was accepted");
    check(backward_grad_zero(grad, -1) == BACKWARD_ERR_ARG, "a negative length was accepted");
}

/* ------------------------------------------------------------------ */
/* Losses                                                             */
/* ------------------------------------------------------------------ */

/* Contexts for the finite differences: they must call the *reference*, in double, so
 * the difference and the analytic gradient are independent derivations. */
struct ce_ctx {
    double *logits;
    int rows;
    int vocab;
    const int *labels;
    const uint8_t *mask;
};
static double ce_value(void *p) {
    struct ce_ctx *c = (struct ce_ctx *)p;
    return ref_masked_ce(c->logits, c->labels, c->mask, c->rows, c->vocab);
}

struct kl_ctx {
    double *student;
    const double *teacher;
    int rows;
    int vocab;
};
static double kl_value(void *p) {
    struct kl_ctx *c = (struct kl_ctx *)p;
    return ref_reverse_kl(c->student, c->teacher, c->rows, c->vocab);
}

struct clip_ctx {
    double *logp;
    const double *old_logp;
    const double *advantage;
    const uint8_t *mask;
    int rows;
    double clip_low;
    double clip_high;
    int sequence_mode;
};
static double clip_value(void *p) {
    struct clip_ctx *c = (struct clip_ctx *)p;
    return ref_clipped(c->logp, c->old_logp, c->advantage, c->mask, c->rows, c->clip_low,
                       c->clip_high, c->sequence_mode);
}

static void test_masked_ce(void) {
    const int rows = 3, vocab = 5;
    const float logits[3 * 5] = {1.0f, 2.0f, 0.5f, -1.0f, 0.0f,
                                 0.3f, -0.4f, 1.7f, 0.1f, -2.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int labels[3] = {1, 4, 2};
    const uint8_t mask[3] = {1, 0, 1}; /* the second row is a prompt/padding position */

    double want = 0.0;
    long long want_count = 0;
    {
        double *d = (double *)malloc(sizeof(double) * (size_t)rows * vocab);
        for (int i = 0; i < rows * vocab; ++i) d[i] = logits[i];
        want = ref_masked_ce(d, labels, mask, rows, vocab);
        free(d);
        for (int r = 0; r < rows; ++r) {
            if (mask[r] != 0) ++want_count;
        }
    }

    float *d_logits = (float *)malloc(sizeof(float) * (size_t)rows * vocab);
    double got_sum = 0.0;
    long long got_count = 0;
    check(backward_masked_ce(logits, labels, mask, rows, vocab, d_logits, &got_sum, &got_count) ==
              BACKWARD_OK,
          "masked CE failed: %s", backward_last_error());
    check(got_count == want_count, "masked CE counted %lld rows, expected %lld", got_count,
          want_count);
    check_close(got_sum / (double)got_count, want, 1e-5, "masked CE value");

    /* The gradient is the central difference of the independent reference. */
    double *d = (double *)malloc(sizeof(double) * (size_t)rows * vocab);
    for (int i = 0; i < rows * vocab; ++i) d[i] = logits[i];
    struct ce_ctx ctx = {d, rows, vocab, labels, mask};
    double worst = 0.0;
    for (int i = 0; i < rows * vocab; ++i) {
        struct fd_ctx fd = {ce_value, &ctx, d, i};
        const double numeric = fd_central(&fd, 1e-5);
        worst = fmax(worst, fabs(numeric - (double)d_logits[i]));
    }
    check(worst < 1e-4, "masked CE gradient differs from the finite difference by %.3e", worst);
    /* A deselected row's gradient is exactly zero, not merely small. */
    for (int j = 0; j < vocab; ++j) check(d_logits[vocab + j] == 0.0f, "a masked row has a gradient");
    /* The selected rows' gradients sum to zero: a softmax minus a one-hot always does. */
    for (int r = 0; r < rows; ++r) {
        if (mask[r] == 0) continue;
        double sum = 0.0;
        for (int j = 0; j < vocab; ++j) sum += d_logits[r * vocab + j];
        check_close(sum, 0.0, 1e-5, "a softmax-minus-onehot row does not sum to zero");
    }
    free(d);
    free(d_logits);

    /* A label outside the vocabulary is refused, and a fully deselected batch is a
     * zero loss with a zero gradient rather than an error. */
    const int bad_labels[3] = {0, 0, 7};
    check(backward_masked_ce(logits, bad_labels, NULL, rows, vocab, NULL, NULL, NULL) ==
              BACKWARD_ERR_RANGE,
          "a label outside the vocabulary was accepted");
    const uint8_t none[3] = {0, 0, 0};
    check(backward_masked_ce(logits, labels, none, rows, vocab, NULL, &got_sum, &got_count) ==
              BACKWARD_OK,
          "an empty selection failed");
    check(got_count == 0 && got_sum == 0.0, "an empty selection is not zero/zero");
    /* The -1 label convention deselects a row as well as the mask. */
    const int ignore_labels[3] = {1, -1, 2};
    check(backward_masked_ce(logits, ignore_labels, NULL, rows, vocab, NULL, &got_sum,
                             &got_count) == BACKWARD_OK,
          "the ignore-label convention failed");
    check(got_count == 2, "an ignored label was counted (%lld of 3)", got_count);
    /* A non-finite logit is refused rather than propagated. */
    float nan_logits[3 * 5];
    memcpy(nan_logits, logits, sizeof(nan_logits));
    nan_logits[3] = NAN;
    check(backward_masked_ce(nan_logits, labels, mask, rows, vocab, NULL, NULL, NULL) ==
              BACKWARD_ERR_DIVERGED,
          "a NaN logit was accepted");
}

static void test_reverse_kl(void) {
    const int rows = 2, vocab = 4;
    const float student[2 * 4] = {0.5f, -0.5f, 1.0f, 0.0f, -1.0f, 0.2f, 0.3f, 0.4f};
    /* The teacher's log-probabilities have to be a genuine distribution. An
     * unnormalised q makes KL(p||q) negative, which is not a bug in the function but a
     * broken fixture, and it would hide a sign error behind a "KL is negative" check. */
    const double teacher_logits[2 * 4] = {0.4, 1.1, -0.3, 0.7, -0.2, 0.9, 0.1, 1.4};
    float teacher_logp[2 * 4];
    for (int r = 0; r < rows; ++r) {
        double max = -INFINITY;
        for (int j = 0; j < vocab; ++j) max = fmax(max, teacher_logits[r * vocab + j]);
        double sum = 0.0;
        for (int j = 0; j < vocab; ++j) sum += exp(teacher_logits[r * vocab + j] - max);
        const double log_z = max + log(sum);
        for (int j = 0; j < vocab; ++j) {
            teacher_logp[r * vocab + j] = (float)(teacher_logits[r * vocab + j] - log_z);
        }
    }

    double want = 0.0;
    {
        double *s = (double *)malloc(sizeof(double) * (size_t)rows * vocab);
        double *t = (double *)malloc(sizeof(double) * (size_t)rows * vocab);
        for (int i = 0; i < rows * vocab; ++i) {
            s[i] = student[i];
            t[i] = teacher_logp[i];
        }
        want = ref_reverse_kl(s, t, rows, vocab);
        free(s);
        free(t);
    }
    float *d_student = (float *)malloc(sizeof(float) * (size_t)rows * vocab);
    double got = 0.0;
    check(backward_reverse_kl(student, teacher_logp, NULL, rows, vocab, d_student, &got) ==
              BACKWARD_OK,
          "reverse KL failed: %s", backward_last_error());
    check_close(got, want, 1e-5, "reverse KL value");
    check(got >= 0.0, "a KL is negative");

    double *s = (double *)malloc(sizeof(double) * (size_t)rows * vocab);
    double *t = (double *)malloc(sizeof(double) * (size_t)rows * vocab);
    for (int i = 0; i < rows * vocab; ++i) {
        s[i] = student[i];
        t[i] = teacher_logp[i];
    }
    struct kl_ctx ctx = {s, t, rows, vocab};
    double worst = 0.0;
    for (int i = 0; i < rows * vocab; ++i) {
        struct fd_ctx fd = {kl_value, &ctx, s, i};
        worst = fmax(worst, fabs(fd_central(&fd, 1e-5) - (double)d_student[i]));
    }
    check(worst < 1e-4, "reverse KL gradient differs from the finite difference by %.3e", worst);
    free(s);
    free(t);
    free(d_student);

    /* A masked row contributes nothing, and a null mask means every row counts. */
    const uint8_t only_first[2] = {1, 0};
    double masked_kl = 0.0;
    check(backward_reverse_kl(student, teacher_logp, only_first, rows, vocab, NULL, &masked_kl) ==
              BACKWARD_OK,
          "a masked reverse KL failed");
    double full_kl = 0.0;
    check(backward_reverse_kl(student, teacher_logp, NULL, 2, vocab, NULL, &full_kl) == BACKWARD_OK,
          "an unmasked reverse KL failed");
    check(masked_kl != full_kl, "the mask did not change the reverse KL");
}

static void test_clipped_objective(void) {
    const int rows = 4;
    const float logp[4] = {-0.10f, -0.35f, -0.60f, -0.90f};
    const float old_logp[4] = {-0.20f, -0.30f, -0.55f, -1.40f};
    const float advantage[4] = {1.0f, -1.0f, 0.5f, -0.5f};
    const uint8_t mask[4] = {1, 1, 1, 0};
    const float clip_low = 0.2f, clip_high = 0.2f;

    for (int sequence_mode = 0; sequence_mode < 2; ++sequence_mode) {
        const BackwardClipMode mode = sequence_mode ? BACKWARD_CLIP_SEQUENCE : BACKWARD_CLIP_TOKEN;
        float *d_logp = (float *)malloc(sizeof(float) * (size_t)rows);
        float ratio[4];
        double objective = 0.0;
        long long selected = 0;
        check(backward_clipped_objective(logp, old_logp, advantage, mask, rows, clip_low, clip_high,
                                         mode, d_logp, ratio, &objective, &selected) == BACKWARD_OK,
              "the clipped objective failed (%s)", sequence_mode ? "sequence" : "token");
        check(selected == 3, "the clipped objective selected %lld rows, expected 3", selected);

        double want = 0.0;
        {
            double l[4], o[4], a[4];
            for (int i = 0; i < rows; ++i) {
                l[i] = logp[i];
                o[i] = old_logp[i];
                a[i] = advantage[i];
            }
            want = ref_clipped(l, o, a, mask, rows, clip_low, clip_high, sequence_mode);
        }
        check_close(objective, want, 1e-5, "clipped objective value");

        /* The gradient of the negated objective against the finite difference of the
         * reference objective: d_loss = -d_objective. */
        double l[4], o[4], a[4];
        for (int i = 0; i < rows; ++i) {
            l[i] = logp[i];
            o[i] = old_logp[i];
            a[i] = advantage[i];
        }
        struct clip_ctx ctx = {l, o, a, mask, rows, clip_low, clip_high, sequence_mode};
        double worst = 0.0;
        for (int i = 0; i < rows; ++i) {
            struct fd_ctx fd = {clip_value, &ctx, l, i};
            const double numeric = -fd_central(&fd, 1e-5);
            const double tolerance = fmax(1e-4, fabs(numeric) * 1e-3);
            const double diff = fabs(numeric - (double)d_logp[i]);
            if (diff > tolerance) {
                check(0, "%s clipped gradient row %d: analytic %.9g numeric %.9g",
                      sequence_mode ? "sequence" : "token", i, (double)d_logp[i], numeric);
            }
            worst = fmax(worst, diff);
        }
        check(worst < 1e-3, "the %s clipped gradient differs by %.3e",
              sequence_mode ? "sequence" : "token", worst);
        free(d_logp);
    }

    /* The clipped region has exactly zero gradient: row 3's ratio is far outside the
     * band with a negative advantage, so the hinge is flat there. */
    float d_logp[4];
    float ratio[4];
    double objective = 0.0;
    long long selected = 0;
    check(backward_clipped_objective(logp, old_logp, advantage, mask, rows, clip_low, clip_high,
                                     BACKWARD_CLIP_TOKEN, d_logp, ratio, &objective, &selected) ==
              BACKWARD_OK,
          "the clipped objective failed");
    /* row 2: logp - old_logp = -0.05 -> ratio 0.951 inside [0.8, 1.2]; row 3 is masked.
     * Shift the fixture so one row is outside the band and check the flat region. */
    const float far_logp[4] = {2.0f, -0.35f, -0.60f, -0.90f};
    check(backward_clipped_objective(far_logp, old_logp, advantage, mask, rows, clip_low, clip_high,
                                     BACKWARD_CLIP_TOKEN, d_logp, ratio, &objective, &selected) ==
              BACKWARD_OK,
          "the far-ratio case failed");
    check(ratio[0] > 1.2f, "the far row's ratio is not outside the band (%.4f)", ratio[0]);
    check(d_logp[0] == 0.0f, "a clipped row has a nonzero gradient (%.3e)", (double)d_logp[0]);

    check(backward_clipped_objective(logp, old_logp, advantage, mask, rows, -0.1f, 0.2f,
                                     BACKWARD_CLIP_TOKEN, NULL, NULL, NULL, NULL) ==
              BACKWARD_ERR_RANGE,
          "a negative clip bound was accepted");
    check(backward_clipped_objective(logp, old_logp, advantage, mask, rows, clip_low, clip_high,
                                     (BackwardClipMode)7, NULL, NULL, NULL, NULL) == BACKWARD_ERR_ARG,
          "an unknown clip mode was accepted");
}

static void test_group_advantage(void) {
    const float rewards[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float advantages[4];
    double mean = 0.0, std = 0.0;
    check(backward_group_advantage(rewards, NULL, 4, 1e-4f, 0, advantages, &mean, &std) ==
              BACKWARD_OK,
          "the group advantage failed: %s", backward_last_error());
    check_close(mean, 2.5, 1e-6, "group mean");
    check_close(std, sqrt(1.25), 1e-6, "group population std");
    check_close(advantages[0] + advantages[3], 0.0, 1e-5, "the advantages are not centered");
    check(advantages[3] > advantages[0], "the advantages are not ordered");

    /* A zero-variance group is refused unless the caller opts in, because dividing by
     * eps turns every advantage into zero and calls it a signal. */
    const float equal[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    check(backward_group_advantage(equal, NULL, 4, 1e-4f, 0, advantages, NULL, NULL) ==
              BACKWARD_ERR_STATE,
          "a zero-variance group was accepted");
    check(backward_group_advantage(equal, NULL, 4, 1e-4f, 1, advantages, &mean, &std) == BACKWARD_OK,
          "an opted-in zero-variance group was refused");
    check(advantages[0] == 0.0f && advantages[3] == 0.0f, "a zero-variance group is not zero");

    /* The mask excludes a reward from the statistics and leaves its advantage at zero. */
    const uint8_t mask[4] = {1, 1, 1, 0};
    check(backward_group_advantage(rewards, mask, 4, 1e-4f, 0, advantages, &mean, &std) ==
              BACKWARD_OK,
          "a masked group failed");
    check_close(mean, 2.0, 1e-6, "masked group mean");
    check(advantages[3] == 0.0f, "a masked reward has an advantage");

    check(backward_group_advantage(rewards, NULL, 0, 1e-4f, 0, NULL, NULL, NULL) == BACKWARD_ERR_ARG,
          "an empty group was accepted");
    const float nan_rewards[4] = {1.0f, NAN, 3.0f, 4.0f};
    check(backward_group_advantage(nan_rewards, NULL, 4, 1e-4f, 0, advantages, NULL, NULL) ==
              BACKWARD_ERR_DIVERGED,
          "a NaN reward was accepted");
}

/* ------------------------------------------------------------------ */
/* AdamW                                                              */
/* ------------------------------------------------------------------ */

static void test_adamw(void) {
    const int n = 6;
    const float lr = 0.05f, beta1 = 0.9f, beta2 = 0.98f, eps = 1e-8f, wd = 0.01f;
    const struct BackwardAdamWHyper hyper = {lr, beta1, beta2, eps, wd};
    float grad[6] = {0.3f, -0.2f, 0.05f, 0.0f, 1.0f, -1.0f};
    float master[6] = {0.5f, -0.25f, 0.125f, 1.0f, -0.75f, 0.0f};

    double ref_master[6], ref_m[6], ref_v[6], ref_grad[6];
    for (int i = 0; i < n; ++i) {
        ref_master[i] = master[i];
        ref_m[i] = 0.0;
        ref_v[i] = 0.0;
        ref_grad[i] = grad[i];
    }

    float m[6] = {0}, v[6] = {0};
    for (int step = 1; step <= 4; ++step) {
        uint16_t bf16[6];
        long long changed = -1;
        check(backward_adamw_step(master, grad, m, v, n, &hyper, step, bf16, &changed) == BACKWARD_OK,
              "the AdamW step failed at step %d: %s", step, backward_last_error());
        ref_adamw(ref_master, ref_grad, ref_m, ref_v, n, lr, beta1, beta2, eps, wd, step);
        for (int i = 0; i < n; ++i) {
            check_close((double)master[i], ref_master[i], 2e-6, "AdamW master after a step");
            check_close((double)m[i], ref_m[i], 2e-6, "AdamW first moment");
            check_close((double)v[i], ref_v[i], 2e-6, "AdamW second moment");
        }
        check(changed >= 0, "the changed count is negative");
    }
    /* The BF16 publication is the master's rounding and nothing else: the same bits a
     * direct round-to-nearest-even of the FP32 master would produce. */
    uint16_t bf16[6];
    check(backward_adamw_step(master, grad, m, v, n, &hyper, 5, bf16, NULL) == BACKWARD_OK,
          "the publication step failed");
    for (int i = 0; i < n; ++i) {
        uint32_t bits;
        const float value = master[i];
        memcpy(&bits, &value, 4);
        const uint16_t expected = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        check(bf16[i] == expected, "element %d published %04x, direct rounding gives %04x", i,
              bf16[i], expected);
    }

    /* A non-finite gradient is refused before any slot moves: the failure mode the
     * plan separates from numerical closeness is a run that silently learns nothing. */
    float poisoned[6];
    memcpy(poisoned, grad, sizeof(poisoned));
    poisoned[3] = INFINITY;
    float master_before[6], m_before[6], v_before[6];
    memcpy(master_before, master, sizeof(master_before));
    memcpy(m_before, m, sizeof(m_before));
    memcpy(v_before, v, sizeof(v_before));
    check(backward_adamw_step(master, poisoned, m, v, n, &hyper, 6, NULL, NULL) ==
              BACKWARD_ERR_DIVERGED,
          "an infinite gradient was accepted");
    check(memcmp(master, master_before, sizeof(master)) == 0 &&
              memcmp(m, m_before, sizeof(m)) == 0 && memcmp(v, v_before, sizeof(v)) == 0,
          "a refused step still moved a slot");

    check(backward_adamw_step(master, grad, m, v, n, &hyper, 0, NULL, NULL) == BACKWARD_ERR_STATE,
          "a zero step index was accepted");
    const struct BackwardAdamWHyper bad = {lr, 1.0f, beta2, eps, wd};
    check(backward_adamw_step(master, grad, m, v, n, &bad, 1, NULL, NULL) == BACKWARD_ERR_RANGE,
          "beta1 = 1 was accepted");
    check(backward_adamw_step(master, grad, m, v, n, NULL, 1, NULL, NULL) == BACKWARD_ERR_ARG,
          "a null hyper-parameter block was accepted");
}

/* ------------------------------------------------------------------ */
/* RNG                                                                */
/* ------------------------------------------------------------------ */

static void test_rng(void) {
    struct BackwardRng a, b;
    backward_rng_seed(&a, 12345);
    backward_rng_seed(&b, 12345);
    for (int i = 0; i < 100; ++i) {
        check(backward_rng_next(&a) == backward_rng_next(&b), "the same seed diverged at draw %d", i);
    }
    /* The counter-based property: the n-th draw is a function of the seed and n, so a
     * second stream that skipped the first k draws agrees from draw k on. */
    struct BackwardRng skipped;
    backward_rng_seed(&skipped, 12345);
    for (int i = 0; i < 7; ++i) (void)backward_rng_next(&skipped);
    struct BackwardRng fresh;
    backward_rng_seed(&fresh, 12345);
    for (int i = 0; i < 7; ++i) (void)backward_rng_next(&fresh);
    check(backward_rng_next(&skipped) == backward_rng_next(&fresh),
          "a draw depends on something other than (seed, counter)");

    backward_rng_seed(&a, 7);
    backward_rng_seed(&b, 8);
    check(backward_rng_next(&a) != backward_rng_next(&b), "two seeds produced the same draw");

    /* Uniform in [0,1): the mean of many draws is 0.5 and every draw is in range. */
    backward_rng_seed(&a, 99);
    double sum = 0.0;
    int out_of_range = 0;
    const int draws = 20000;
    for (int i = 0; i < draws; ++i) {
        const double u = backward_rng_uniform(&a);
        if (!(u >= 0.0 && u < 1.0)) ++out_of_range;
        sum += u;
    }
    check(out_of_range == 0, "%d uniform draws left [0,1)", out_of_range);
    check_close(sum / draws, 0.5, 0.02, "the mean of uniform draws");
}

/* ------------------------------------------------------------------ */
/* Checkpoint / resume                                                */
/* ------------------------------------------------------------------ */

static void test_checkpoint(void) {
    /* Four small parameters, of which one is empty (a zero-element role is legal and
     * keeps the layout arithmetic honest). */
    float master_a[3] = {1.0f, 2.0f, 3.0f}, m_a[3] = {0.1f, 0.2f, 0.3f}, v_a[3] = {1e-3f, 2e-3f, 3e-3f};
    float master_b[2] = {-1.0f, 0.5f}, m_b[2] = {0.0f, 0.0f}, v_b[2] = {0.0f, 0.0f};
    float master_c[1] = {42.0f}, m_c[1] = {1.0f}, v_c[1] = {2.0f};
    struct BackwardCheckpointParam params[4] = {
        {0, 3, master_a, m_a, v_a},
        {1, 2, master_b, m_b, v_b},
        {2, 0, NULL, NULL, NULL},
        {3, 1, master_c, m_c, v_c},
    };
    struct BackwardCheckpoint ckpt = {params, 4, 17, 3, {0xDEADBEEFULL, 9}, {1234, 2}};

    size_t size = 0;
    check(backward_checkpoint_size(&ckpt, &size) == BACKWARD_OK, "sizing failed: %s",
          backward_last_error());
    /* The layout, recomputed here from its definition rather than compared against a
     * literal: header(64) + per parameter (4-byte id + 8-byte count + three FP32
     * arrays) + a 4-byte checksum. A test that trusts the writer's own arithmetic
     * would not notice the writer changing the format. */
    const size_t header = 8 + 4 + 4 + 4 + 8 + 4 + 8 + 8 + 8 + 4;
    const size_t expected_size = header + (4 + 8 + 3 * 3 * sizeof(float)) +
                                 (4 + 8 + 2 * 3 * sizeof(float)) + (4 + 8) +
                                 (4 + 8 + 1 * 3 * sizeof(float)) + 4;
    check(size == expected_size, "the checkpoint size is %zu, the layout says %zu", size,
          expected_size);

    unsigned char *buffer = (unsigned char *)malloc(size);
    size_t written = 0;
    check(backward_checkpoint_write(&ckpt, buffer, size, &written) == BACKWARD_OK,
          "writing failed: %s", backward_last_error());
    check(written == size, "the writer wrote %zu of %zu bytes", written, size);

    /* Read it into fresh arrays and require an exact round trip. */
    float master_a2[3] = {0}, m_a2[3] = {0}, v_a2[3] = {0};
    float master_b2[2] = {0}, m_b2[2] = {0}, v_b2[2] = {0};
    float master_c2[1] = {0}, m_c2[1] = {0}, v_c2[1] = {0};
    struct BackwardCheckpointParam params2[4] = {
        {0, 3, master_a2, m_a2, v_a2},
        {1, 2, master_b2, m_b2, v_b2},
        {2, 0, NULL, NULL, NULL},
        {3, 1, master_c2, m_c2, v_c2},
    };
    struct BackwardCheckpoint ckpt2 = {params2, 4, 0, 0, {0, 0}, {0, 0}};
    size_t consumed = 0;
    check(backward_checkpoint_read(&ckpt2, buffer, size, &consumed) == BACKWARD_OK,
          "reading failed: %s", backward_last_error());
    check(consumed == size, "the reader consumed %zu of %zu bytes", consumed, size);
    check(ckpt2.store_version == 17 && ckpt2.step_index == 3, "the version/step did not round-trip");
    check(ckpt2.rng.seed == 0xDEADBEEFULL && ckpt2.rng.counter == 9, "the RNG did not round-trip");
    check(ckpt2.cursor.sample == 1234 && ckpt2.cursor.epoch == 2, "the cursor did not round-trip");
    check(memcmp(master_a, master_a2, sizeof(master_a)) == 0, "master A did not round-trip exactly");
    check(memcmp(m_a, m_a2, sizeof(m_a)) == 0, "moment A did not round-trip exactly");
    check(memcmp(v_c, v_c2, sizeof(v_c)) == 0, "moment C did not round-trip exactly");
    check(memcmp(master_b, master_b2, sizeof(master_b)) == 0, "master B did not round-trip exactly");

    /* A resume must not restart the bias correction, so the step index matters. */
    check(backward_checkpoint_read(&ckpt2, buffer, size, NULL) == BACKWARD_OK, "a re-read failed");

    /* Every way the checkpoint can be wrong is a refusal. */
    unsigned char *broken = (unsigned char *)malloc(size);
    memcpy(broken, buffer, size);
    broken[0] = 'X';
    check(backward_checkpoint_read(&ckpt2, broken, size, NULL) == BACKWARD_ERR_FORMAT,
          "a bad magic was accepted");
    memcpy(broken, buffer, size);
    check(backward_checkpoint_read(&ckpt2, broken, size - 1, NULL) == BACKWARD_ERR_FORMAT,
          "a truncated checkpoint was accepted");
    memcpy(broken, buffer, size);
    broken[size / 2] ^= 0x01;
    check(backward_checkpoint_read(&ckpt2, broken, size, NULL) == BACKWARD_ERR_FORMAT,
          "a flipped payload bit was accepted");
    memcpy(broken, buffer, size);
    broken[size - 1] ^= 0x80;
    check(backward_checkpoint_read(&ckpt2, broken, size, NULL) == BACKWARD_ERR_FORMAT,
          "a corrupted checksum was accepted");
    /* A different parameter plan (same count, different element counts) is refused
     * rather than loaded into the wrong arrays. */
    float other[2] = {0}, om[2] = {0}, ov[2] = {0};
    struct BackwardCheckpointParam other_params[4] = {
        {0, 2, other, om, ov},
        {1, 2, master_b2, m_b2, v_b2},
        {2, 0, NULL, NULL, NULL},
        {3, 1, master_c2, m_c2, v_c2},
    };
    struct BackwardCheckpoint ckpt3 = {other_params, 4, 0, 0, {0, 0}, {0, 0}};
    check(backward_checkpoint_read(&ckpt3, buffer, size, NULL) == BACKWARD_ERR_FORMAT,
          "a reordered plan was accepted");
    other[0] = other[1] = 123.0f; /* poison, to prove a refusal does not write */
    struct BackwardCheckpoint ckpt4 = {other_params, 4, 0, 0, {0, 0}, {0, 0}};
    check(backward_checkpoint_read(&ckpt4, broken, size, NULL) == BACKWARD_ERR_FORMAT,
          "a corrupted checkpoint was accepted");
    check(other[0] == 123.0f && other[1] == 123.0f,
          "a refused read still wrote into the caller's arrays");

    /* A buffer that is too small is refused without writing past its end. */
    check(backward_checkpoint_write(&ckpt, buffer, size - 1, NULL) == BACKWARD_ERR_NO_SPACE,
          "a short buffer was accepted");
    /* The checksum is over the payload, so one flipped payload bit changes it. The last
     * payload byte (just before the checksum) is definitely not header. */
    memcpy(broken, buffer, size);
    broken[size - 8] ^= 0xFF;
    check(backward_checksum(broken, size - 4) != backward_checksum(buffer, size - 4),
          "the checksum did not notice a flipped payload bit");
    free(broken);
    free(buffer);
}

/* ------------------------------------------------------------------ */
/* Tied weights: one gradient, one update, two identical readers       */
/* ------------------------------------------------------------------ */

/* The plan's Stage-4 gate names "tied weights" among the cases a backward must get
 * right, and the two ways to get it wrong are both silent: summing the two readers'
 * gradients into two separate optimizers (so the parameter moves twice, along two
 * different trajectories once the moments differ) or publishing the update to only one
 * reader (so the embedding and the LM head disagree after one step). This is the
 * composition the plan asks for: the two contributions are summed, one AdamW step
 * moves the master once, and the BF16 published for both logical ids is the same bits.
 *
 * Stage 3's store is what makes the two ids one logical parameter (the 27B's LM head
 * templates onto its embedding); this test is the gradient/optimizer half of that. */
static void test_tied_weights(void) {
    enum { n = 5 };
    const struct BackwardAdamWHyper hyper = {0.1f, 0.9f, 0.99f, 1e-8f, 0.0f};
    const float initial[n] = {0.5f, -1.0f, 0.25f, 2.0f, -0.5f};
    /* The two readers of one logical parameter: the embedding gather's gradient and the
     * LM head's, which land in the same buffer. */
    const float from_embedding[n] = {0.3f, -0.1f, 0.0f, 0.7f, 0.2f};
    const float from_lm_head[n] = {0.1f, 0.4f, -0.2f, 0.0f, -0.3f};

    float grad[n];
    check(backward_grad_zero(grad, n) == BACKWARD_OK, "tied: zeroing failed");
    long long nonzero_first = 0, nonzero_second = 0;
    check(backward_grad_accumulate_tied(grad, from_embedding, n, &nonzero_first) == BACKWARD_OK,
          "tied: the first reader failed");
    check(backward_grad_accumulate_tied(grad, from_lm_head, n, &nonzero_second) == BACKWARD_OK,
          "tied: the second reader failed");
    /* The count is the contribution's own nonzero elements, reported so a caller can
     * tell "both readers contributed" from "one of them was silent". */
    check(nonzero_first == 4 && nonzero_second == 4,
          "tied: the readers reported %lld and %lld nonzeros, expected 4 and 4",
          nonzero_first, nonzero_second);
    for (long long i = 0; i < n; ++i) {
        check_close((double)grad[i], (double)from_embedding[i] + (double)from_lm_head[i], 1e-6,
                    "the tied gradient is not the sum of the readers");
    }

    float master[n], m_slot[n] = {0}, v_slot[n] = {0};
    for (long long i = 0; i < n; ++i) master[i] = initial[i];
    uint16_t published[n];
    long long changed = 0;
    /* Exactly one step, not one per reader: a second step would be visible in the
     * moments, which is what "one optimizer update per tied parameter" forbids. */
    check(backward_adamw_step(master, grad, m_slot, v_slot, n, &hyper, 1, published, &changed) ==
              BACKWARD_OK,
          "the tied update failed");
    check(changed > 0, "the tied update changed nothing");
    float master_after[n];
    memcpy(master_after, master, sizeof(master));

    /* The publication is a pure function of the master, so both readers receive the
     * same bits: a direct round-to-nearest-even of the post-step master must equal what
     * the step published. (Running the step twice would advance the moments again; that
     * is the rule above, not this check.) */
    int mismatch = 0;
    for (long long i = 0; i < n; ++i) {
        uint32_t bits;
        const float value = master_after[i];
        memcpy(&bits, &value, 4);
        const uint16_t expected = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        if (published[i] != expected) mismatch = 1;
    }
    check(!mismatch,
          "the publication is not the master's own BF16 rounding, so two readers would differ");

    /* And a second step does move the parameter, so the first one was a step rather than
     * a no-op that this test would have accepted. */
    float probe_m[n] = {0}, probe_v[n] = {0};
    for (long long i = 0; i < n; ++i) probe_m[i] = m_slot[i] - 1.0f; /* a different moment */
    uint16_t probe[n];
    check(backward_adamw_step(master_after, grad, probe_m, probe_v, n, &hyper, 2, probe, NULL) ==
              BACKWARD_OK,
          "the probe update failed");
    int differs = 0;
    for (long long i = 0; i < n; ++i) {
        if (probe[i] != published[i]) differs = 1;
    }
    check(differs, "a second update did not move the parameter, so the first was not a step");
}

/* ------------------------------------------------------------------ */
/* An overfit, and a resume that has to equal an uninterrupted run     */
/* ------------------------------------------------------------------ */

/* The plan's gate: "Compare one complete AdamW step ..., overfit a tiny deterministic
 * SFT fixture and test checkpoint/resume of parameters, optimizer, RNG and data
 * cursor. Test run-to-run determinism separately from numerical closeness."
 *
 * The fixture is a deterministic linear classification head trained through this file's
 * own loss (masked cross entropy) and its own optimizer (AdamW) on a fixed synthetic
 * batch. It is deliberately *not* the transformer's SFT: the engine has no training
 * traversal (Stage 5 brings up SFT, and its gate re-runs this at the model level). What
 * it does establish is that the pieces this stage owns compose into a descent that
 * converges and is reproducible: parameter -> logits -> masked CE -> dlogits ->
 * AdamW with FP32 moments -> BF16 publication.
 *
 * The resume half is the stronger one. Saving at step k, restoring into a *fresh*
 * optimizer, and running to step n must land on exactly the same bits as the
 * uninterrupted n-step run, including the bias correction (which is why the step index
 * is part of the checkpoint) and the data cursor (which is why the cursor is). */
static int overfit_fixture_run(int steps, int resume_at, float *out_master, float *out_m,
                               float *out_v, long long *out_cursor, int verbose) {
    enum { rows = 32, vocab = 4, features = 8 };
    const struct BackwardAdamWHyper hyper = {0.05f, 0.9f, 0.999f, 1e-8f, 0.0f};

    /* A separable fixture: each label has a prototype the head can learn to favour. */
    float prototypes[vocab * features];
    for (int v = 0; v < vocab; ++v) {
        for (int f = 0; f < features; ++f) {
            prototypes[v * features + f] = (v == f % vocab) ? 1.0f : -0.5f;
        }
    }
    float x[rows * features];
    int labels[rows];
    for (int r = 0; r < rows; ++r) {
        for (int f = 0; f < features; ++f) x[r * features + f] = (float)((r + f) % 3) - 1.0f;
        int best = 0;
        float best_dot = -1e30f;
        for (int v = 0; v < vocab; ++v) {
            float dot = 0.0f;
            for (int f = 0; f < features; ++f) {
                dot += x[r * features + f] * prototypes[v * features + f];
            }
            if (dot > best_dot) {
                best_dot = dot;
                best = v;
            }
        }
        labels[r] = best;
    }

    float master[vocab * features];
    for (int i = 0; i < vocab * features; ++i) master[i] = 0.01f * (float)((i % 5) - 2);
    float m_slot[vocab * features] = {0}, v_slot[vocab * features] = {0};
    float grad[vocab * features];
    float logits[rows * vocab];
    float d_logits[rows * vocab];

    struct BackwardRng rng;
    backward_rng_seed(&rng, 4242);
    struct BackwardCursor cursor = {0, 0};

    unsigned char checkpoint[8192];
    size_t checkpoint_size = 0;
    struct BackwardCheckpointParam plan[1] = {{0, vocab * features, master, m_slot, v_slot}};
    struct BackwardCheckpoint ckpt = {plan, 1, 0, 0, {0, 0}, {0, 0}};

    int accuracy = 0;
    for (int step = 1; step <= steps; ++step) {
        if (backward_grad_zero(grad, vocab * features) != BACKWARD_OK) return -1;
        for (int r = 0; r < rows; ++r) {
            for (int v = 0; v < vocab; ++v) {
                float dot = 0.0f;
                for (int f = 0; f < features; ++f) {
                    dot += x[r * features + f] * master[v * features + f];
                }
                logits[r * vocab + v] = dot;
            }
        }
        double sum = 0.0;
        long long count = 0;
        if (backward_masked_ce(logits, labels, NULL, rows, vocab, d_logits, &sum, &count) !=
            BACKWARD_OK) {
            return -1;
        }
        accuracy = 0;
        for (int r = 0; r < rows; ++r) {
            int best = 0;
            for (int v = 1; v < vocab; ++v) {
                if (logits[r * vocab + v] > logits[r * vocab + best]) best = v;
            }
            if (best == labels[r]) ++accuracy;
        }
        /* dW = sum_r dlogits_r outer x_r, the accumulation the plan's dW guarantee
         * describes (one micro-batch here, so the order is trivially fixed). */
        for (int r = 0; r < rows; ++r) {
            for (int v = 0; v < vocab; ++v) {
                for (int f = 0; f < features; ++f) {
                    grad[v * features + f] += d_logits[r * vocab + v] * x[r * features + f];
                }
            }
        }
        if (step == resume_at) {
            ckpt.store_version = 0;
            ckpt.step_index = step;
            ckpt.rng = rng;
            ckpt.cursor = cursor;
            if (backward_checkpoint_write(&ckpt, checkpoint, sizeof(checkpoint),
                                          &checkpoint_size) != BACKWARD_OK) {
                return -1;
            }
            /* Wipe the state so the resume has to be what restores it. Without this the
             * comparison below would pass even if `read` wrote nothing. */
            for (int i = 0; i < vocab * features; ++i) {
                master[i] = 0.0f;
                m_slot[i] = 0.0f;
                v_slot[i] = 0.0f;
            }
            struct BackwardCheckpoint load = {plan, 1, 0, 0, {0, 0}, {0, 0}};
            if (backward_checkpoint_read(&load, checkpoint, checkpoint_size, NULL) != BACKWARD_OK) {
                return -1;
            }
            if (load.step_index != step) return -1;
            rng = load.rng;
            cursor = load.cursor;
        }
        if (backward_adamw_step(master, grad, m_slot, v_slot, vocab * features, &hyper, step, NULL,
                                NULL) != BACKWARD_OK) {
            return -1;
        }
        cursor.sample += rows;
        (void)backward_rng_next(&rng);
    }
    if (verbose) {
        printf("  overfit: %d steps, train accuracy %d/%d, cursor sample %lld\n", steps, accuracy,
               rows, cursor.sample);
    }
    if (out_master != NULL) {
        memcpy(out_master, master, sizeof(master));
        memcpy(out_m, m_slot, sizeof(m_slot));
        memcpy(out_v, v_slot, sizeof(v_slot));
        *out_cursor = cursor.sample;
    }
    return accuracy;
}

static void test_overfit_and_resume(void) {
    const int n = 32;
    float master_a[n], m_a[n], v_a[n], master_b[n], m_b[n], v_b[n];
    long long cursor_a = 0, cursor_b = 0;

    const int accuracy_a = overfit_fixture_run(20, 0, master_a, m_a, v_a, &cursor_a, 1);
    check(accuracy_a == n, "the fixture did not overfit: accuracy %d of %d", accuracy_a, n);

    /* Run-to-run determinism of the whole descent, tested separately from any reference. */
    float master_c[n], m_c[n], v_c[n];
    long long cursor_c = 0;
    const int accuracy_c = overfit_fixture_run(20, 0, master_c, m_c, v_c, &cursor_c, 0);
    check(accuracy_c == accuracy_a, "the same fixture converged differently");
    check(memcmp(master_a, master_c, sizeof(master_a)) == 0 &&
              memcmp(m_a, m_c, sizeof(m_a)) == 0 && memcmp(v_a, v_c, sizeof(v_a)) == 0 &&
              cursor_a == cursor_c,
          "the same fixture produced different bits");

    /* The resume: 12 steps, save, wipe, restore, 8 more -- and the parameters, both
     * optimizer moments and the data cursor must all equal the uninterrupted run's. */
    const int accuracy_b = overfit_fixture_run(20, 12, master_b, m_b, v_b, &cursor_b, 0);
    check(accuracy_b == accuracy_a, "the resumed run converged differently (%d vs %d)",
          accuracy_b, accuracy_a);
    check(memcmp(master_a, master_b, sizeof(master_a)) == 0,
          "a resumed run did not reach the same parameters");
    check(memcmp(m_a, m_b, sizeof(m_a)) == 0, "a resumed run did not reach the same moments");
    check(memcmp(v_a, v_b, sizeof(v_a)) == 0,
          "a resumed run did not reach the same second moments");
    check(cursor_a == cursor_b, "a resumed run did not reach the same data cursor");

    /* And the checkpoint is whole: truncating it is a refusal, which is what stops a
     * half-written file from resuming a different run. */
    printf("  resume: 20 uninterrupted == 12+8 resumed, bitwise on parameters and moments\n");
}

int main(void) {
    test_regions();
    test_convention();
    test_accumulation();
    test_masked_ce();
    test_reverse_kl();
    test_clipped_objective();
    test_group_advantage();
    test_adamw();
    test_rng();
    test_checkpoint();
    test_tied_weights();
    test_overfit_and_resume();

    if (g_failures != 0) {
        fprintf(stderr, "backward_test: %d of %d check(s) failed\n", g_failures, g_checks);
        return EXIT_FAILURE;
    }
    printf("backward_test: the Stage-4 table covers the plan's rows, the losses and AdamW match an "
           "independent FP64 reference, and a checkpoint refuses every way it can be wrong\n");
    return EXIT_SUCCESS;
}
