/**
 * train_loop.c - The synchronous training/rollout baseline's phases, budgets, records
 * and host sampler (plan Stage 5). See csrc/include/train_loop.h for the contract.
 *
 * CUDA-free: the phase machine, the budget arithmetic, the record's version binding and
 * the FP64 sampler are all host code, so the gate for them runs anywhere and the engine
 * only supplies the buffers and the logits.
 */
#include "train_loop.h"

#include "backward.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Per-thread error text                                              */
/* ------------------------------------------------------------------ */

static _Thread_local char g_error[256];

const char *train_loop_last_error(void) { return g_error; }

void train_loop_clear_error(void) { g_error[0] = '\0'; }

static TrainStatus fail(TrainStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

/* ------------------------------------------------------------------ */
/* Phase budgets                                                      */
/* ------------------------------------------------------------------ */

int train_loop_phase_budget(TrainPhase phase, const struct TrainLoopDims *dims, int tokens,
                            long long trainable_elements, long long retained_elements,
                            struct TrainPhaseBudget *out) {
    if (dims == NULL || out == NULL) {
        return (int)fail(TRAIN_ERR_ARG, "train_loop_phase_budget: null dims or output");
    }
    if (tokens < 1) {
        return (int)fail(TRAIN_ERR_ARG, "train_loop_phase_budget: tokens must be positive");
    }
    if (trainable_elements < 0 || retained_elements < 0) {
        return (int)fail(TRAIN_ERR_ARG,
                         "train_loop_phase_budget: element counts must not be negative");
    }
    if (phase != TRAIN_PHASE_SFT && phase != TRAIN_PHASE_ROLLOUT) {
        return (int)fail(TRAIN_ERR_ARG, "train_loop_phase_budget: unknown phase %d", (int)phase);
    }
    struct TrainPhaseBudget budget;
    memset(&budget, 0, sizeof(budget));
    budget.phase = phase;
    /* The checkpoint's own tensors, in BF16: shared by both phases and by inference. */
    long long weight_elements = 0;
    const int hidden = dims->hidden_size;
    const int layers = dims->num_layers;
    /* Only the terms the phases differ on are computed here; the weight figure is an
     * estimate from the descriptor's shapes, which is what a budget needs. */
    weight_elements += (long long)dims->vocab_size * hidden;             /* embedding (tied) */
    weight_elements += 2LL * layers * 4LL * hidden * hidden;             /* attention projections */
    weight_elements += 2LL * layers * 3LL * hidden * dims->intermediate_size; /* MLP */
    budget.weights_bytes = weight_elements * 2;
    budget.kv_cache_bytes = 2LL * dims->max_seq_len * dims->num_kv_heads * dims->head_dim * 2;
    budget.gdn_state_bytes =
        (long long)layers * ((long long)dims->gdn_conv_dim * (dims->gdn_conv_kernel - 1) * 2 +
                             (long long)dims->gdn_num_v_heads * dims->gdn_head_dim *
                                 dims->gdn_head_dim * 4);
    if (phase == TRAIN_PHASE_SFT) {
        /* A backward needs the step's retained values and the optimizer's four FP32
         * slots per trainable element (master, gradient, and two moments). */
        budget.activations_bytes = (long long)tokens * hidden * 2 * 4; /* fwd + grad streams */
        budget.retained_bytes = retained_elements * 4;
        budget.optimizer_bytes = trainable_elements * 4 * 4;
    } else {
        /* A rollout borrows the parameters and keeps nothing else: no activations for a
         * backward, no optimizer state, no retained step values. */
        budget.activations_bytes = 0;
        budget.retained_bytes = 0;
        budget.optimizer_bytes = 0;
    }
    budget.total_bytes = budget.weights_bytes + budget.kv_cache_bytes + budget.gdn_state_bytes +
                         budget.activations_bytes + budget.retained_bytes + budget.optimizer_bytes;
    *out = budget;
    return (int)TRAIN_OK;
}

/* ------------------------------------------------------------------ */
/* The phase machine                                                  */
/* ------------------------------------------------------------------ */

struct TrainLoop {
    TrainStore *store;
    TrainPhase phase;
    int open;
    long long version;
    long long sequence_resets;
    long long optimizer_offloaded_bytes;
    /* A rollout holds a real borrowing context, so the store enforces the rule. */
    TrainContext *borrow;
};

TrainLoop *train_loop_create(TrainStore *store) {
    if (store == NULL) {
        fail(TRAIN_ERR_ARG, "train_loop_create: a store is required");
        return NULL;
    }
    TrainLoop *loop = (TrainLoop *)calloc(1, sizeof(TrainLoop));
    if (loop == NULL) {
        fail(TRAIN_ERR_RANGE, "train_loop_create: out of memory");
        return NULL;
    }
    loop->store = store;
    loop->phase = TRAIN_PHASE_SFT;
    loop->version = -1;
    return loop;
}

void train_loop_destroy(TrainLoop *loop) {
    if (loop == NULL) return;
    if (loop->open) train_loop_leave(loop);
    free(loop);
}

TrainStatus train_loop_enter(TrainLoop *loop, TrainPhase phase, long long *out_version) {
    if (loop == NULL) return fail(TRAIN_ERR_ARG, "train_loop_enter: null loop");
    if (loop->open) {
        return fail(TRAIN_ERR_STATE,
                    "train_loop_enter: the %s phase is already open; leave it first",
                    loop->phase == TRAIN_PHASE_SFT ? "SFT" : "rollout");
    }
    if (phase != TRAIN_PHASE_SFT && phase != TRAIN_PHASE_ROLLOUT) {
        return fail(TRAIN_ERR_ARG, "train_loop_enter: unknown phase %d", (int)phase);
    }
    loop->phase = phase;
    loop->open = 1;
    /* Every phase enters with a clean sequence: the KV cache and the GDN state belong to
     * the sequence that produced them. */
    ++loop->sequence_resets;
    /* A rollout reads the parameters, so it borrows the version and an update is refused
     * until it leaves; SFT owns the update window, so it holds no borrow of its own. */
    loop->version = train_store_version(loop->store);
    if (phase == TRAIN_PHASE_ROLLOUT) {
        loop->optimizer_offloaded_bytes = 0;
        /* The borrow is what makes the store refuse an update while the rollout reads:
         * a recorded version would only be a promise. */
        loop->borrow = train_context_create(loop->store, /*is_rollout=*/1);
        if (loop->borrow == NULL) {
            loop->open = 0;
            loop->version = -1;
            return fail(TRAIN_ERR_STATE, "train_loop_enter: %s", train_last_error());
        }
    }
    if (out_version != NULL) *out_version = loop->version;
    return TRAIN_OK;
}

TrainStatus train_loop_leave(TrainLoop *loop) {
    if (loop == NULL) return fail(TRAIN_ERR_ARG, "train_loop_leave: null loop");
    if (!loop->open) return fail(TRAIN_ERR_STATE, "train_loop_leave: no phase is open");
    if (loop->borrow != NULL) {
        const TrainStatus status = train_context_destroy(loop->borrow);
        if (status != TRAIN_OK) return fail(status, "train_loop_leave: %s", train_last_error());
        loop->borrow = NULL;
    }
    loop->open = 0;
    loop->version = -1;
    return TRAIN_OK;
}

TrainPhase train_loop_phase(const TrainLoop *loop) {
    return loop == NULL ? TRAIN_PHASE_SFT : loop->phase;
}

int train_loop_sequence_resets(const TrainLoop *loop) {
    return loop == NULL ? 0 : (int)loop->sequence_resets;
}

long long train_loop_optimizer_offloads(const TrainLoop *loop) {
    return loop == NULL ? 0 : loop->optimizer_offloaded_bytes;
}

long long train_loop_version(const TrainLoop *loop) {
    return loop == NULL || !loop->open ? -1 : loop->version;
}

/* ------------------------------------------------------------------ */
/* Selection records                                                  */
/* ------------------------------------------------------------------ */

TrainStatus train_loop_record(TrainLoop *loop, struct TrainGroup *group,
                              const struct TrainSampleRecord *record) {
    if (loop == NULL || group == NULL || record == NULL) {
        return fail(TRAIN_ERR_ARG, "train_loop_record: null loop, group or record");
    }
    if (!loop->open) {
        return fail(TRAIN_ERR_STATE, "train_loop_record: no phase is open");
    }
    if (record->tokens < 0 || record->tokens > TRAIN_LOOP_MAX_TOKENS) {
        return fail(TRAIN_ERR_RANGE, "train_loop_record: %d tokens is out of range",
                    record->tokens);
    }
    if (group->count >= TRAIN_LOOP_MAX_GROUP) {
        return fail(TRAIN_ERR_RANGE, "train_loop_record: the group holds %d records",
                    TRAIN_LOOP_MAX_GROUP);
    }
    if (record->version != loop->version) {
        return fail(TRAIN_ERR_STATE,
                    "train_loop_record: the record is stamped version %lld but the loop reads "
                    "%lld",
                    record->version, loop->version);
    }
    if (group->count == 0) {
        /* One group, one version: mixing versions inside a group would make the group's
         * advantages compare samples from two different policies. */
        group->version = record->version;
    } else if (group->version != record->version) {
        return fail(TRAIN_ERR_STATE,
                    "train_loop_record: the group is version %lld and the record %lld",
                    group->version, record->version);
    }
    group->records[group->count] = *record;
    ++group->count;
    return TRAIN_OK;
}

TrainStatus train_loop_read_reward(const TrainLoop *loop, const struct TrainGroup *group,
                                   int index, float *out_reward) {
    if (loop == NULL || group == NULL || out_reward == NULL) {
        return fail(TRAIN_ERR_ARG, "train_loop_read_reward: null argument");
    }
    if (index < 0 || index >= group->count) {
        return fail(TRAIN_ERR_RANGE, "train_loop_read_reward: index %d of %d", index, group->count);
    }
    /* The plan's rule: never reconstruct an old denominator with updated weights. A
     * record generated under a version that is no longer current is not usable as a
     * behavior policy for a new update. */
    if (group->version != train_store_version(loop->store)) {
        return fail(TRAIN_ERR_STATE,
                    "train_loop_read_reward: the group was generated under version %lld and the "
                    "store is at %lld",
                    group->version, train_store_version(loop->store));
    }
    *out_reward = group->records[index].reward;
    return TRAIN_OK;
}

TrainStatus train_loop_ratio(const TrainLoop *loop, const struct TrainSampleRecord *record,
                             const float *new_logprobs, double *out_ratio) {
    if (loop == NULL || record == NULL || new_logprobs == NULL || out_ratio == NULL) {
        return fail(TRAIN_ERR_ARG, "train_loop_ratio: null argument");
    }
    if (record->version != train_store_version(loop->store)) {
        return fail(TRAIN_ERR_STATE,
                    "train_loop_ratio: the record is version %lld and the store is at %lld",
                    record->version, train_store_version(loop->store));
    }
    int selected = 0;
    double sum_new = 0.0;
    double sum_old = 0.0;
    for (int t = 0; t < record->tokens; ++t) {
        if (record->mask[t] == 0) continue;
        sum_new += (double)new_logprobs[t];
        sum_old += (double)record->logprobs[t];
        ++selected;
    }
    if (selected == 0) {
        return fail(TRAIN_ERR_STATE, "train_loop_ratio: the record has no selected token");
    }
    /* The sequence-level ratio the plan's objective uses: exp of the mean log-ratio over
     * the completion. At unchanged parameters `new_logprobs` is bitwise the recorded
     * one, so the mean log-ratio is exactly 0 and the ratio is exactly 1. */
    *out_ratio = exp((sum_new - sum_old) / (double)selected);
    return TRAIN_OK;
}

/* ------------------------------------------------------------------ */
/* The host FP64 sampler                                              */
/* ------------------------------------------------------------------ */

TrainStatus train_loop_sample_fp64(const float *logits, int vocab, struct BackwardRng *rng,
                                   int *out_token, float *out_model_logprob,
                                   float *out_sampled_logprob) {
    if (logits == NULL || rng == NULL || out_token == NULL) {
        return fail(TRAIN_ERR_ARG, "train_loop_sample_fp64: null argument");
    }
    if (vocab <= 0) {
        return fail(TRAIN_ERR_ARG, "train_loop_sample_fp64: vocab must be positive");
    }
    double max = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        if (!isfinite(logits[v])) {
            return fail(TRAIN_ERR_STATE, "train_loop_sample_fp64: logit %d is not finite", v);
        }
        if ((double)logits[v] > max) max = (double)logits[v];
    }
    /* FP64 throughout: the CDF the gate's frequency test compares against is this one,
     * not an FP32 approximation of it. */
    double *prob = (double *)malloc(sizeof(double) * (size_t)vocab);
    if (prob == NULL) return fail(TRAIN_ERR_RANGE, "train_loop_sample_fp64: out of memory");
    double total = 0.0;
    for (int v = 0; v < vocab; ++v) {
        prob[v] = exp((double)logits[v] - max);
        total += prob[v];
    }
    const double uniform = backward_rng_uniform(rng);
    double cumulative = 0.0;
    int chosen = vocab - 1;
    for (int v = 0; v < vocab; ++v) {
        cumulative += prob[v] / total;
        if (uniform < cumulative) {
            chosen = v;
            break;
        }
    }
    if (out_model_logprob != NULL) {
        *out_model_logprob = (float)((double)logits[chosen] - (max + log(total)));
    }
    if (out_sampled_logprob != NULL) {
        /* Temperature 1 and no truncation: the sampler's distribution *is* the model's,
         * so the two log-probabilities are the same number. A transformation would make
         * them differ, and the record keeps both for that reason. */
        *out_sampled_logprob = (float)((double)logits[chosen] - (max + log(total)));
    }
    *out_token = chosen;
    free(prob);
    return TRAIN_OK;
}

TrainStatus train_loop_sample_frequencies(const float *logits, int vocab, int draws,
                                          struct BackwardRng *rng, double *out_frequencies) {
    if (logits == NULL || rng == NULL || out_frequencies == NULL) {
        return fail(TRAIN_ERR_ARG, "train_loop_sample_frequencies: null argument");
    }
    if (vocab <= 0 || draws <= 0) {
        return fail(TRAIN_ERR_ARG, "train_loop_sample_frequencies: vocab and draws must be positive");
    }
    for (int v = 0; v < vocab; ++v) out_frequencies[v] = 0.0;
    for (int draw = 0; draw < draws; ++draw) {
        int token = 0;
        const TrainStatus status = train_loop_sample_fp64(logits, vocab, rng, &token, NULL,
                                                          NULL);
        if (status != TRAIN_OK) return status;
        out_frequencies[token] += 1.0;
    }
    for (int v = 0; v < vocab; ++v) out_frequencies[v] /= (double)draws;
    return TRAIN_OK;
}
