/**
 * train_loop.h - The synchronous training/rollout baseline's phases, budgets, records
 * and sampler (plan Stage 5, the half after "first bring up SFT").
 *
 * docs/plan-numeric-contract.md, Stage 5: "Then add phase-specific memory budgets and
 * stochastic rollout. Reuse the temperature-sampling migration rather than adding
 * another sampler here; ordinary CLI sampling does not wait for SFT. Before RL
 * admission, bind its selection records to immutable policy versions and validate
 * host-FP64 sampler versus trainer-FP32 logprob differences explicitly."
 *
 * This file is CUDA-free, like train.h and backward.c: it is the bookkeeping and the
 * host arithmetic, so every rule is testable on the CPU and the engine only supplies
 * the buffers and the logits. The rules, each a refusal rather than a comment:
 *
 *   - **A phase owns its budget.** SFT retains activations and optimizer state; a
 *     rollout retains neither (it holds a KV cache and a GDN state instead). The budget
 *     is computed per phase, so "which phase can afford what" is a number the caller
 *     reads rather than assumes.
 *   - **A rollout borrows a version and never writes.** A record is stamped with the
 *     store version it was generated under, and reading a record whose version is no
 *     longer current is refused: an old denominator may not be reconstructed from
 *     updated weights.
 *   - **A phase boundary resets the sequence state** (KV and GDN), because carrying one
 *     phase's state into the next would make the second phase's first token depend on
 *     the first phase's last one.
 *   - **The sampler is host FP64**, temperature 1, no truncation: its distribution is
 *     then the model's softmax, and the finite-RNG/CDF and FP64/FP32 differences stay
 *     subject to the sampling gates the plan's temperature-sampling track owns. A
 *     transformation (top-k/top-p/temperature) would have to record both the raw model
 *     log-probability and the sampler's, which is why the record keeps both.
 */
#ifndef HASKELL_INFER_TRAIN_LOOP_H
#define HASKELL_INFER_TRAIN_LOOP_H

#include <stddef.h>
#include <stdint.h>

#include "train.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Which phase a budget belongs to. The two differ in what they retain, which is the
 * whole content of "phase-specific": SFT keeps activations for a backward and the
 * optimizer's moments; a rollout keeps neither and cannot run a backward at all. */
typedef enum {
    TRAIN_PHASE_SFT = 0,
    TRAIN_PHASE_ROLLOUT = 1,
} TrainPhase;

/* The shape facts a budget needs, copied out of the descriptor's dimensions so this
 * header (and its gate) stays CUDA-free: the layer headers own `ModelDims` and pull in
 * the CUDA runtime, which a bookkeeping translation unit must not. */
struct TrainLoopDims {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    int gdn_conv_dim;
    int gdn_conv_kernel;
    int gdn_num_v_heads;
    int gdn_head_dim;
};

/* What a phase's step is expected to hold, in bytes. `weights` is the BF16 model, which
 * both phases share; the rest is what distinguishes them. */
struct TrainPhaseBudget {
    TrainPhase phase;
    long long weights_bytes;
    long long kv_cache_bytes;
    long long gdn_state_bytes;
    long long activations_bytes;   /* 0 for a rollout */
    long long optimizer_bytes;     /* 0 for a rollout: offload it or omit it */
    long long retained_bytes;      /* the step's saved values; 0 for a rollout */
    long long total_bytes;
};

/* The budget for one step of `tokens` under `phase`. `trainable_elements` is what the
 * optimizer state is sized from (four FP32 slots per element: master, gradient and two
 * moments), and `retained_elements` what Stage 3's step retains. A negative input is a
 * refusal (BACKWARD-style status codes are not reused here; 0 is an error and the error
 * text says which argument). */
int train_loop_phase_budget(TrainPhase phase, const struct TrainLoopDims *dims, int tokens,
                            long long trainable_elements, long long retained_elements,
                            struct TrainPhaseBudget *out);

/* ------------------------------------------------------------------ */
/* The phase machine                                                  */
/* ------------------------------------------------------------------ */

/* A phase is entered once, owns the sequence state, and is left before the other begins.
 * A rollout *borrows* the store (Stage 3's context object), so the store itself refuses
 * an update while one is open rather than this file promising not to write; the SFT phase
 * holds no borrow, because it owns the exclusive window. */
typedef struct TrainLoop TrainLoop;

TrainLoop *train_loop_create(TrainStore *store);
void train_loop_destroy(TrainLoop *loop);

/* Enter a phase: a rollout borrows the store's current version (so an update is refused
 * until it leaves), and either phase starts with a clean sequence. Returning to the same
 * phase without leaving is a refusal. */
TrainStatus train_loop_enter(TrainLoop *loop, TrainPhase phase, long long *out_version);
TrainStatus train_loop_leave(TrainLoop *loop);

TrainPhase train_loop_phase(const TrainLoop *loop);
int train_loop_sequence_resets(const TrainLoop *loop);   /* at each phase boundary */
long long train_loop_optimizer_offloads(const TrainLoop *loop); /* bytes offloaded, or kept */

/* The version this loop is running under, or -1 when no phase is open. */
long long train_loop_version(const TrainLoop *loop);

/* ------------------------------------------------------------------ */
/* Selection records                                                  */
/* ------------------------------------------------------------------ */

/* One generation's record. The plan names every field: the sampled token ids, the
 * per-token behavior log-probabilities, the rewards, the masks, the terminal reasons and
 * the version. The record is *immutable* once written and is refused for reading under a
 * different version, which is the "never reconstruct an old denominator" rule. */
#define TRAIN_LOOP_MAX_TOKENS 4096
#define TRAIN_LOOP_MAX_GROUP 64

typedef enum {
    TRAIN_TERMINAL_EOS = 0,
    TRAIN_TERMINAL_LENGTH = 1,
    TRAIN_TERMINAL_STOPPED = 2,
    TRAIN_TERMINAL_ERROR = 3,
} TrainTerminalReason;

struct TrainSampleRecord {
    long long version;             /* the parameter version this was generated under */
    long long policy_id;           /* identifies the frozen reference vs the behavior policy */
    int tokens;
    int token_ids[TRAIN_LOOP_MAX_TOKENS];
    float logprobs[TRAIN_LOOP_MAX_TOKENS];  /* the model's own log-probability of each */
    float sampled_logprobs[TRAIN_LOOP_MAX_TOKENS]; /* what the sampler actually used */
    uint8_t mask[TRAIN_LOOP_MAX_TOKENS];    /* 1 on the completion, 0 on the prompt */
    TrainTerminalReason terminal;
    float reward;
};

/* A group of G completions for one prompt, all under one version and one sampling
 * configuration (the plan's group-based requirement: serial generation is fine, mixing
 * versions is not). */
struct TrainGroup {
    long long version;
    int count;
    struct TrainSampleRecord records[TRAIN_LOOP_MAX_GROUP];
};

/* Record one sample. Refused when the loop has no phase open, when the sample's version
 * is not the loop's, or when the id count is out of range. */
TrainStatus train_loop_record(TrainLoop *loop, struct TrainGroup *group,
                              const struct TrainSampleRecord *record);

/* Read a record's reward and advantage, refusing a version that is no longer current:
 * this is the rule that keeps an old denominator from being recomputed with new weights. */
TrainStatus train_loop_read_reward(const TrainLoop *loop, const struct TrainGroup *group,
                                   int index, float *out_reward);

/* `ratio` for one recorded token: the new log-probability over the recorded behavior
 * log-probability. At unchanged parameters and the same version the two are the same
 * bitwise, so the ratio is exactly 1 - which is the plan's "ratio equals one at unchanged
 * parameters in an admitted same-case fixture" as a function rather than a claim. */
TrainStatus train_loop_ratio(const TrainLoop *loop, const struct TrainSampleRecord *record,
                             const float *new_logprobs, double *out_ratio);

/* ------------------------------------------------------------------ */
/* The host FP64 sampler                                             */
/* ------------------------------------------------------------------ */

/* The request's RNG state: the trainer's own (backward.h's counter-based generator is
 * reused rather than a second one being invented). */
struct BackwardRng;

/* Temperature-1 categorical sampling from FP32 model logits, in FP64 on the host:
 * softmax, CDF, one uniform draw. The sampled token, the model's own log-probability of
 * it and the sampler's log-probability are all returned, because a transformation would
 * make the two differ and the objective has to say which it optimizes. */
TrainStatus train_loop_sample_fp64(const float *logits, int vocab, struct BackwardRng *rng,
                                   int *out_token, float *out_model_logprob,
                                   float *out_sampled_logprob);

/* The measured frequency of each token over `draws` samples, for the sampling gate: the
 * sampler's empirical distribution has to match the model's softmax to within the
 * finite-sample band, which is a different question from whether the logits are right. */
TrainStatus train_loop_sample_frequencies(const float *logits, int vocab, int draws,
                                          struct BackwardRng *rng, double *out_frequencies);

/* ------------------------------------------------------------------ */
/* Errors                                                             */
/* ------------------------------------------------------------------ */

const char *train_loop_last_error(void);
void train_loop_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_TRAIN_LOOP_H */
