/**
 * backward.h - The differentiation contract, the losses and the optimizer
 * (plan Stage 4).
 *
 * docs/plan-numeric-contract.md, Stage 4 asks for the backward of every region in
 * the dense/dense-hybrid path, the losses, and AdamW, and it asks for one thing
 * *before* any of that: "the exact forward rounding boundaries must be documented
 * before differentiating". A comment does not stop a caller from differentiating the
 * wrong forward, so this header turns the three things a backward can silently get
 * wrong into objects:
 *
 *   1. `backward_region_info` - which regions have a backward, which region of the
 *      Stage-1 inventory each one differentiates, what mechanism implements it, and
 *      what it deliberately does not cover. The CPU gate checks that this table and
 *      the Stage-1 inventory cannot drift apart and that the plan's Stage-4 rows are
 *      all present, so "not implemented" cannot be mistaken for "not applicable".
 *
 *   2. `backward_cast_mode` - the mixed-precision differentiation convention. A BF16
 *      cast is a *step function*, so it either passes a gradient through untouched
 *      (the convention this stage adopts) or it blocks it; there is no derivative to
 *      finite-difference. The engine reads this value rather than assuming it, and
 *      the gate that trains a tiny fixture is what shows the convention is the one
 *      the reference (PyTorch, whose casts are identity) was differentially compared
 *      under.
 *
 *   3. `backward_check_retained` - the values a region's backward consumes. Stage 2
 *      established that the attention core's *natural* saving is a base-2 LSE and
 *      nothing else, and Stage 1 recorded what each region can save. A caller that runs
 *      a region's backward declares its retained set and is refused by name here rather
 *      than reading whatever happened to be in the buffer. The trainer's step does not
 *      consult this table - it keeps Stage 3's retention plan and the SFT gate (Stage 5)
 *      is what exercises that - so routing the step's plan through these names is an open
 *      item rather than a property of the trainer today.
 *
 * This file is CUDA-free, like train.h: the losses, the optimizer, the accumulation
 * schedule and the checkpoint format are pure FP32 arithmetic over host or device
 * memory that a CPU test drives directly, and the device kernels implement the same
 * contract against the same unit test's FP64 oracle.
 *
 * The losses are defined here once, in the normalisation the trainer uses:
 *
 *   masked CE         loss = -(1/count) * sum_r mask_r * logsoftmax(z_r)[y_r]
 *                     dloss/dz_r = (mask_r/count) * (softmax(z_r) - onehot(y_r))
 *   reverse KL        KL = sum_j p_j (log p_j - log q_j),  p = softmax(z_student)
 *                     dKL/dz_i = p_i * (log p_i - log q_i - KL)
 *   clipped objective ratio = exp(logp - old_logp), for a token or one sequence
 *                     dloss/dlogp = -(mask/count) * A * ratio in the linear region
 *   group advantage   A_i = (r_i - mean(r)) / (std(r) + eps), population std
 *
 * The normalisation is *inside* the gradient on purpose: a caller that divides the
 * loss by a count outside these functions gets a different gradient, and the point of
 * this stage's gate is that "which denominator" is not a free choice.
 */
#ifndef HASKELL_INFER_BACKWARD_H
#define HASKELL_INFER_BACKWARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BACKWARD_ABI_VERSION 1

/* Status codes, parallel to TrainStatus so a caller can share a switch. Every
 * fallible entry point returns one of these and leaves its outputs untouched unless
 * it returns BACKWARD_OK. */
typedef enum {
    BACKWARD_OK = 0,
    BACKWARD_ERR_ARG = 1,        /* null or invalid argument (see backward_last_error) */
    BACKWARD_ERR_RANGE = 2,      /* an index, size or shape is out of range */
    BACKWARD_ERR_STATE = 3,      /* illegal for the current lifecycle state */
    BACKWARD_ERR_MISSING = 4,    /* a value the backward needs was not retained */
    BACKWARD_ERR_UNSUPPORTED = 5,/* the region has no backward in this stage */
    BACKWARD_ERR_DIVERGED = 6,   /* a value is not finite (a NaN reached the loss) */
    BACKWARD_ERR_NO_SPACE = 7,   /* the caller's buffer is too small */
    BACKWARD_ERR_FORMAT = 8,     /* a checkpoint does not match the expected layout */
} BackwardStatus;

/* ------------------------------------------------------------------ */
/* Region coverage                                                    */
/* ------------------------------------------------------------------ */

/* The plan's Stage-4 table rows, in its order. The enum is the table. */
typedef enum {
    BACKWARD_REGION_EMBEDDING_TIED_LM_HEAD = 0,
    BACKWARD_REGION_RESIDUAL_ELEMENTWISE,
    BACKWARD_REGION_ROPE_QGATE_SPLIT,
    BACKWARD_REGION_NORMS,
    BACKWARD_REGION_GEMM,
    BACKWARD_REGION_GDN_PREPARE,
    BACKWARD_REGION_GDN_CONV1D,
    BACKWARD_REGION_GDN_CORE,
    BACKWARD_REGION_ATTENTION_CORE,
    BACKWARD_REGION_LOSSES,
    BACKWARD_REGION_ADAMW,
    BACKWARD_REGION_ROW_COUNT,
} BackwardRegion;

/* How a region's backward obtains a value: retained by the step, or recomputed from
 * something retained. The distinction is not cosmetic - a recomputed value has to be
 * recomputed under the convention that produced the forward's, which is exactly what
 * Stage 2's claim B measured for GDN and claim E for attention. */
typedef enum {
    BACKWARD_SOURCE_RETAINED = 0,
    BACKWARD_SOURCE_RECOMPUTED = 1,
    BACKWARD_SOURCE_NOT_NEEDED = 2,
} BackwardValueSource;

struct BackwardRegionInfo {
    BackwardRegion region;
    const char *name;                 /* the plan's Stage-4 table row */
    const char *stage1_regions;       /* ','-separated Stage-1 region names this row differentiates */
    int implemented;                  /* 1: an entry point exists and a gate covers it */
    const char *mechanism;            /* where the backward lives */
    const char *note;                 /* the honest remainder */
};

const struct BackwardRegionInfo *backward_region_info(int *count);
const struct BackwardRegionInfo *backward_region_at(BackwardRegion region);

/* The Stage-1 region names this row differentiates. `count` is how many there are;
 * `name_at` copies the i-th into `out` (NUL-terminated, truncated and refused with -1
 * if it does not fit) and returns the length, or -1. A copy rather than a slice into
 * the table's own comma-separated string, because a caller comparing slices with
 * strcmp would read past the comma. The CPU gate walks these in both directions to
 * prove the backward table and the Stage-1 inventory cannot drift apart. */
int backward_region_stage1_count(BackwardRegion region);
int backward_region_stage1_name_at(BackwardRegion region, int index, char *out, size_t out_size);

/* ------------------------------------------------------------------ */
/* The differentiation convention                                     */
/* ------------------------------------------------------------------ */

/* The plan: "the initial mixed-precision differentiation convention treats supported
 * casts as identity for gradient propagation, while derivatives consume the
 * appropriate saved rounded values". A cast is a step function: its derivative is 0
 * almost everywhere and undefined on the step, so "identity" is a *choice*, and the
 * only meaningful check is that the whole model was compared against a reference
 * whose casts behave the same way. This reports the choice the engine made. */
typedef enum {
    BACKWARD_CAST_IDENTITY = 0,  /* a BF16 cast passes the gradient through unchanged */
    BACKWARD_CAST_BLOCKING = 1,  /* a cast stops the gradient (not used by this stage) */
} BackwardCastMode;

BackwardCastMode backward_cast_mode(void);

/* The fixed loss-scaling contract: gradients accumulate in FP32, are scaled by one
 * factor per accumulated micro-batch before the optimizer step, and divided by it
 * before publication. `backward_check_retained`'s caller does not get to pick a
 * different schedule per region. */
float backward_default_loss_scale(void);

/* ------------------------------------------------------------------ */
/* Retained values                                                    */
/* ------------------------------------------------------------------ */

/* The value a region's backward needs and where it comes from. `name` is compared
 * exactly against the caller's retained names. */
struct BackwardRequiredValue {
    BackwardRegion region;
    const char *name;
    BackwardValueSource source;
};

const struct BackwardRequiredValue *backward_required_values(int *count);

/* Refuse a backward whose step did not retain (or cannot recompute) everything the
 * region needs. `have` is the caller's retained-name array. Returns BACKWARD_OK, or
 * BACKWARD_ERR_MISSING with the first missing name in the error text. */
BackwardStatus backward_check_retained(BackwardRegion region, const char *const *have, int have_count);

/* ------------------------------------------------------------------ */
/* Gradient accumulation                                              */
/* ------------------------------------------------------------------ */

/* The accumulation schedule the dW guarantee refers to: contributions arrive in a
 * fixed order. A contribution is added, never assigned, so a caller cannot drop a
 * micro-batch by writing twice. */
BackwardStatus backward_grad_zero(float *grad, long long n);
BackwardStatus backward_grad_accumulate(float *grad, const float *contribution, long long n);

/* A tied parameter has two readers (embedding and LM head), so its gradient is the
 * sum of the two contributions and it gets exactly one optimizer update. The sum is
 * this function, not two optimizer steps. */
BackwardStatus backward_grad_accumulate_tied(float *grad, const float *contribution, long long n,
                                             long long *out_nonzero);

/* Scale every gradient element by `scale` (the loss-scaling step). Refuses a
 * non-finite or non-positive scale rather than filling the gradients with NaN. */
BackwardStatus backward_grad_scale(float *grad, long long n, float scale);

/* ------------------------------------------------------------------ */
/* Losses                                                             */
/* ------------------------------------------------------------------ */

/* Masked cross entropy over `rows` FP32 logit rows of `vocab` entries.
 *
 * `labels[r] < 0` and `mask[r] == 0` both mean "not selected". `out_sum` is the
 * masked sum of the NLL (not divided by count), `out_count` the number of selected
 * rows, so a caller can normalise by tokens *or* by tokens-in-a-sequence explicitly.
 * `d_logits` (may be null) receives (mask_r/count) * (softmax(z_r) - onehot(y_r)),
 * zero where the row is not selected. A count of 0 is a valid result: loss 0 and an
 * all-zero gradient, not an error. */
BackwardStatus backward_masked_ce(const float *logits, const int *labels, const uint8_t *mask,
                                  int rows, int vocab, float *d_logits, double *out_sum,
                                  long long *out_count);

/* Dense reverse KL over the full vocabulary, student logits against teacher
 * log-probabilities (the frozen teacher's own log-softmax, already computed):
 *   KL = sum_r mask_r * sum_j p_j (log p_j - log q_j),  normalized by the count.
 * `d_student_logits` receives the masked, count-normalized gradient above. The
 * teacher is a constant here: this is the student's gradient only. */
BackwardStatus backward_reverse_kl(const float *student_logits, const float *teacher_logp,
                                   const uint8_t *mask, int rows, int vocab,
                                   float *d_student_logits, double *out_kl);

/* Token- or sequence-level clipped objective (the GSPO/PPO hinge without the RL
 * loop). For TOKEN mode every row is its own ratio. For SEQUENCE mode the whole
 * block is one sequence (Stage 3: "one sequence at a time") and the ratio is
 * exp(mean_t(logp_t - old_logp_t)) over the selected rows, which is what makes the
 * objective a sequence-level one.
 *
 * `out_ratio` receives the ratio actually used at each row, `out_objective` the
 * masked mean of min/max(ratio*A, clip(ratio)*A), and `d_logp` the gradient. */
typedef enum {
    BACKWARD_CLIP_TOKEN = 0,
    BACKWARD_CLIP_SEQUENCE = 1,
} BackwardClipMode;

BackwardStatus backward_clipped_objective(const float *logp, const float *old_logp,
                                          const float *advantage, const uint8_t *mask, int rows,
                                          float clip_low, float clip_high, BackwardClipMode mode,
                                          float *d_logp, float *out_ratio, double *out_objective,
                                          long long *out_selected);

/* Group advantage reduction: A_i = (r_i - mean) / (std + eps) with the population
 * standard deviation, over `count` rewards of which the ones with `mask[i] == 0` are
 * excluded. A group whose selected rewards are all equal has zero variance; scaling
 * by it is a division by eps that turns a zero advantage into 0/eps = 0, which is
 * only harmless by accident, so it is refused unless `allow_zero_variance` is set.
 * `out_advantages` is written only on BACKWARD_OK. */
BackwardStatus backward_group_advantage(const float *rewards, const uint8_t *mask, int count,
                                        float eps, int allow_zero_variance, float *out_advantages,
                                        double *out_mean, double *out_std);

/* ------------------------------------------------------------------ */
/* AdamW                                                             */
/* ------------------------------------------------------------------ */

/* The exact step PyTorch's AdamW takes, which is what the gate compares against:
 *
 *   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t
 *   m   = beta1*m + (1-beta1)*g
 *   v   = beta2*v + (1-beta2)*g*g
 *   master *= 1 - lr*weight_decay
 *   master -= (lr/bc1) * m / (sqrt(v)/sqrt(bc2) + eps)
 *
 * The eps is added *after* the bias-corrected second moment, not to `sqrt(v/bc2)`:
 * the two differ by ~eps in the denominator, which is above the FP32 noise floor for
 * a step near zero, so the order is part of the contract rather than an implementation
 * detail. `out_bf16` (may be null) receives the FP32 master cast back to BF16 - the
 * single rounding that publishes an update.
 *
 * `step_index` is 1-based. A non-finite gradient element is refused: the plan's gate
 * separates numerical closeness from a run that silently learned nothing. Returns the
 * number of elements whose BF16 value actually changed in `out_changed`. */
struct BackwardAdamWHyper {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
};

BackwardStatus backward_adamw_step(float *master, const float *grad, float *m_slot, float *v_slot,
                                   long long n, const struct BackwardAdamWHyper *hyper,
                                   int step_index, uint16_t *out_bf16, long long *out_changed);

/* ------------------------------------------------------------------ */
/* Reproducibility: RNG and the data cursor                           */
/* ------------------------------------------------------------------ */

/* A counter-based RNG (splitmix64), not a linear congruential generator: the state is
 * a function of the step and of the sample index, so two runs configured identically
 * agree without depending on how many draws another part of the code made. Sampling
 * itself is the temperature-sampling track's, not this stage's; what this owns is that
 * the *trainer's* draws (shuffling, dropout-free SFT augmentation) are reproducible
 * and that the state is checkpointed with everything else. */
struct BackwardRng {
    uint64_t seed;
    uint64_t counter;
};

void backward_rng_seed(struct BackwardRng *rng, uint64_t seed);
uint64_t backward_rng_next(struct BackwardRng *rng);
/* Uniform in [0,1) with 53 bits of mantissa, the transform the sampling gates assume. */
double backward_rng_uniform(struct BackwardRng *rng);

/* The data cursor: which sample of which epoch comes next. Checkpointing it is what
 * makes "resume" mean resume rather than restart. */
struct BackwardCursor {
    long long sample;
    int epoch;
};

/* ------------------------------------------------------------------ */
/* Checkpoint / resume                                                */
/* ------------------------------------------------------------------ */

/* One trainable parameter's slots inside a checkpoint. `master` is the FP32 weight,
 * `m_slot`/`v_slot` the optimizer's first and second moments; a frozen parameter is
 * not in the plan at all, so a resumed store cannot invent training state for it. */
struct BackwardCheckpointParam {
    int logical;
    long long elements;
    float *master;
    float *m_slot;
    float *v_slot;
};

/* The full state a resume needs: parameters, optimizer, RNG and data cursor, plus the
 * publication counter and the AdamW step index so a resumed run continues the bias
 * correction rather than restarting it at t=1. */
struct BackwardCheckpoint {
    const struct BackwardCheckpointParam *params;
    int param_count;
    long long store_version;
    int step_index;
    struct BackwardRng rng;
    struct BackwardCursor cursor;
};

/* Bytes one checkpoint occupies, or 0 with BACKWARD_ERR_ARG on an invalid plan. */
BackwardStatus backward_checkpoint_size(const struct BackwardCheckpoint *ckpt, size_t *out_bytes);

/* Write and read the whole state. The reader verifies the format, the parameter
 * layout (count and element counts, in order) and a trailing checksum, and refuses a
 * mismatch rather than loading a partial resume: a checkpoint that resumes a different
 * run than it saved is worse than no checkpoint. Both functions refuse a plan whose
 * RNG or cursor is not finite/monotonic, and `read` leaves the caller's arrays
 * untouched on any failure by checking everything before it copies. */
BackwardStatus backward_checkpoint_write(const struct BackwardCheckpoint *ckpt, void *buffer,
                                         size_t capacity, size_t *out_written);
BackwardStatus backward_checkpoint_read(struct BackwardCheckpoint *ckpt, const void *buffer,
                                        size_t size, size_t *out_consumed);

/* A 32-bit checksum over the payload the writer emitted, so a truncated or reordered
 * checkpoint is a BACKWARD_ERR_FORMAT rather than a silently different model. */
uint32_t backward_checksum(const void *data, size_t size);

/* ------------------------------------------------------------------ */
/* Errors                                                             */
/* ------------------------------------------------------------------ */

const char *backward_last_error(void);
void backward_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_BACKWARD_H */
