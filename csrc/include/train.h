/**
 * train.h - The trainable runtime's ownership objects (plan Stage 3).
 *
 * docs/plan-numeric-contract.md, Stage 3: "Haskell owns the fixed model traversal,
 * training schedule and typed opaque handles; C/CUDA owns allocations, streams and
 * kernel execution. No raw CUDA pointers become user-facing Haskell values."
 *
 * This header is the C half of that split, and it is deliberately **CUDA-free**: it
 * is the pure bookkeeping — which logical parameters exist, which of them are tied,
 * frozen or trainable, who is reading the current version, whether an update or a
 * free is legal right now, and which derived copies a publication has to refresh.
 * The buffers behind those parameters are opaque `void *` slots that the engine
 * fills with device memory and a CPU test fills with host memory, so every rule in
 * here is testable without a GPU and the same rules govern the real thing.
 *
 * The objects:
 *
 *   TrainStore        logical parameters (tying already resolved), each with a
 *                     version, its master/compute/gradient/optimizer-slot presence,
 *                     the devices that hold a copy, and the derived copies that
 *                     depend on it.
 *   TrainContext      a trainer or synchronous-rollout context. It *borrows* a
 *                     committed parameter version and owns its own scratch and
 *                     sequence state. Borrowing is what makes an update illegal:
 *                     publication needs exclusive ownership.
 *   TrainStep         one training step: it retains the forward activations and the
 *                     GDN chunk-boundary states its backward will consume, and it
 *                     records alias rules and free points for them. Stage 4 adds the
 *                     backward; the retention rules are what Stage 3 owns.
 *
 * Three rules are the point of the whole file, and each is a rejection rather than a
 * comment:
 *
 *   - a parameter update while any context still borrows the current version is
 *     refused (a reader must never see a half-updated parameter set);
 *   - a free or a store-wide destroy while a context or a step is active is refused;
 *   - publishing an update must refresh every derived copy (the FP32 GDN norm weight
 *     is cast from its BF16 source at load; an update that forgot it would leave the
 *     model with two disagreeing copies of one parameter).
 */
#ifndef HASKELL_INFER_TRAIN_H
#define HASKELL_INFER_TRAIN_H

#include <stddef.h>
#include <stdint.h>

#include "model_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRAIN_ABI_VERSION 1

/* A caller that owns a buffer computed from a parameter but absent from the
 * checkpoint - the FP32 GDN norm weight is cast from its BF16 source at load - can
 * give that buffer a logical identity for the derived-copy registry by using this
 * role code. Nothing else in the store interprets the code; the caller owns the
 * mapping from it to its buffer. */
#define TRAIN_DERIVED_ROLE_BASE 1000

/* How many parameters the store can hold: layers x roles, plus the global roles.
 * The engine refuses a descriptor whose layer/role product exceeds this. */
#define TRAIN_MAX_PARAMS (ENGINE_MAX_LAYERS * ENGINE_MAX_ROLES + ENGINE_MAX_ROLES)
#define TRAIN_MAX_LOGICAL TRAIN_MAX_PARAMS
#define TRAIN_MAX_DERIVED 16
#define TRAIN_MAX_SAVED 4096

/* Status codes. Every fallible entry point returns one of these and leaves the
 * store unchanged unless it returns TRAIN_OK. */
typedef enum {
    TRAIN_OK = 0,
    TRAIN_ERR_ARG = 1,            /* null or invalid argument (see train_last_error) */
    TRAIN_ERR_STATE = 2,          /* illegal for the current lifecycle state */
    TRAIN_ERR_BUSY = 3,           /* a reader or a step is active (update/free) */
    TRAIN_ERR_EXCLUSIVE = 4,      /* already inside an exclusive update */
    TRAIN_ERR_UNKNOWN_PARAM = 5,  /* no such (layer, role) */
    TRAIN_ERR_FROZEN = 6,         /* a frozen parameter has no training state */
    TRAIN_ERR_NO_SLOT = 7,        /* the caller did not provide a required buffer */
    TRAIN_ERR_SCHEDULE = 8,       /* an update did not respect the accumulation schedule */
    TRAIN_ERR_RANGE = 9,          /* an index, size or shape is out of range */
} TrainStatus;

/* Which buffer of one logical parameter a caller is asking about. COMPUTE is the
 * BF16 weight the forward reads; MASTER is the FP32 weight an optimizer would step;
 * GRAD and OPT_* exist only for a trainable parameter. */
typedef enum {
    TRAIN_SLOT_MASTER = 0,
    TRAIN_SLOT_COMPUTE = 1,
    TRAIN_SLOT_GRAD = 2,
    TRAIN_SLOT_OPT_M = 3,
    TRAIN_SLOT_OPT_V = 4,
    TRAIN_SLOT_COUNT = 5,
} TrainSlot;

/* What a derived copy is: a buffer the engine keeps that is *computed from* a
 * parameter and would silently disagree with it after an update. */
typedef enum {
    TRAIN_DERIVED_BF16_TO_FP32 = 0,  /* gdn_norm_f32: a widened copy of its BF16 source */
} TrainDerivedKind;

/* One parameter as the engine sees it, before tying is resolved. `layer` is -1 for
 * a global role (embedding, LM head, final norm). `elements` is the element count of
 * the checkpoint tensor, not of a rank's shard. */
struct TrainParamSpec {
    int layer;
    int role;
    long long elements;
    int frozen;              /* 1: no master/grad/slot storage is allocated */
    int trainable;           /* 0: a read-only constant (e.g. a bias the trainer fixes) */
};

/* Opaque handles. A caller never sees a CUDA pointer through these. */
typedef struct TrainStore TrainStore;
typedef struct TrainContext TrainContext;
typedef struct TrainStep TrainStep;

/* ------------------------------------------------------------------ */
/* Store                                                              */
/* ------------------------------------------------------------------ */

/* Create a store over `count` parameter specs. Tying is resolved here: two specs
 * with the same (layer, role_template) become one logical parameter, and the specs
 * that map to it are recorded so a publication can write every aliasing buffer.
 *
 * `templates` and `role_names` are parallel to `specs` and must outlive the store
 * (the engine keeps its descriptor for the life of the load). `specs` is copied. */
TrainStore *train_store_create(const struct TrainParamSpec *specs, const char *const *templates,
                               const char *const *role_names, int count);

/* Destroy the store. Returns TRAIN_ERR_BUSY if any context still borrows it or any
 * step is active: a reader must be gone before the memory it reads disappears. */
TrainStatus train_store_destroy(TrainStore *store);

int train_store_param_count(const TrainStore *store);     /* specs, i.e. (layer, role) pairs */
int train_store_logical_count(const TrainStore *store);   /* after tying */

/* The logical parameter a (layer, role) resolves to, or -1. */
int train_store_logical_of(const TrainStore *store, int layer, int role);

/* The spec table, so a caller that owns the buffers can walk it: how many specs were
 * registered, and the (layer, role) of one. A publication visits every spec because a
 * tied parameter has one entry per reader. */
int train_store_spec_count(const TrainStore *store);
int train_store_spec_layer(const TrainStore *store, int index);
int train_store_spec_role(const TrainStore *store, int index);

/* How many (layer, role) specs alias one logical parameter, and which. Used by the
 * tied-role gate: the tied parameter must be one logical thing with two readers. */
int train_store_alias_count(const TrainStore *store, int logical);
int train_store_alias_at(const TrainStore *store, int logical, int index);

/* Element count of a logical parameter, and the role name of its first alias. */
long long train_store_elements(const TrainStore *store, int logical);
const char *train_store_name(const TrainStore *store, int logical);

/* Training state of a logical parameter: a frozen or non-trainable parameter has no
 * master/grad/slot storage, so asking for it is a rejection, not a null read. */
int train_store_is_trainable(const TrainStore *store, int logical);
int train_store_is_frozen(const TrainStore *store, int logical);

/* Opaque buffer slots. The engine sets the real device pointers at load; a CPU test
 * sets host pointers. The store never dereferences them — it only records which
 * slots exist and hands out the ones a caller must fill for a publication. */
TrainStatus train_store_set_slot(TrainStore *store, int logical, TrainSlot slot, void *buffer);
void *train_store_slot(const TrainStore *store, int logical, TrainSlot slot);

/* How many (layer, role) buffers a slot covers: the compute slot of a tied parameter
 * is written for every alias, so a publication has to visit them all. */
int train_store_slot_buffer_count(const TrainStore *store, int logical, TrainSlot slot);

/* Version bookkeeping. The version is a monotonic counter over publications; every
 * context borrow records the version it reads so a later comparison can refuse to
 * mix two versions in one result. */
long long train_store_version(const TrainStore *store);

/* ------------------------------------------------------------------ */
/* Derived copies                                                     */
/* ------------------------------------------------------------------ */

/* Register a derived copy: `derived_logical`'s buffer is computed from
 * `source_logical`'s compute buffer by `kind`. Publication must refresh every
 * registered derived copy of every published parameter. */
TrainStatus train_store_register_derived(TrainStore *store, int source_logical, int derived_logical,
                                         TrainDerivedKind kind);
int train_store_derived_count(const TrainStore *store);

/* The i-th derived copy, and whether it is stale (its source has been published
 * since it was last refreshed). A publication leaves them all stale; refreshing
 * clears them, and the store refuses to publish again with a stale copy of a
 * parameter it just wrote. */
TrainStatus train_store_derived_at(const TrainStore *store, int index, int *source_logical,
                                   int *derived_logical, TrainDerivedKind *kind, int *stale);
TrainStatus train_store_derived_refreshed(TrainStore *store, int index);
int train_store_stale_derived_count(const TrainStore *store);

/* ------------------------------------------------------------------ */
/* Contexts, borrowing and exclusive updates                          */
/* ------------------------------------------------------------------ */

/* Create a context. `is_rollout` distinguishes a synchronous rollout context (which
 * only ever reads a committed version) from a trainer context (which may also own an
 * update). Either way the context borrows the store's current version. */
TrainContext *train_context_create(TrainStore *store, int is_rollout);
TrainStatus train_context_destroy(TrainContext *context);

long long train_context_borrowed_version(const TrainContext *context);

/* Enter and leave an exclusive update. `begin` fails with TRAIN_ERR_BUSY while any
 * context is borrowed, which is how "updating requires exclusive ownership" is
 * enforced rather than documented. The update window is where a caller writes
 * master weights and publishes. */
TrainStatus train_store_begin_update(TrainStore *store);
int train_store_in_update(const TrainStore *store);

/* Publish the update: everything written to a master buffer becomes the compute
 * weights of a new version, and every derived copy is marked stale. Refreshing them
 * is the caller's next step (the engine casts; a test can do it on the host), and
 * `train_store_end_update` refuses while a stale derived copy of a written parameter
 * remains. */
TrainStatus train_store_publish(TrainStore *store);
TrainStatus train_store_end_update(TrainStore *store);

/* The contribution of one micro-batch, recorded so the accumulation schedule is
 * explicit: contributions must arrive in a fixed order and only while no publication
 * is pending. Stage 4's backward fills the gradient buffers; this records the shape
 * of the accumulation the guarantee in the plan refers to. */
TrainStatus train_store_note_microbatch(TrainStore *store, int index, int count);
int train_store_microbatches_seen(const TrainStore *store);

/* ------------------------------------------------------------------ */
/* Cross-device copies                                                */
/* ------------------------------------------------------------------ */

/* A logical parameter that is replicated across devices has one buffer per device
 * and needs an explicit gradient merge before the optimizer step and an explicit
 * update broadcast after it; independent optimizers are exactly what the plan
 * forbids. This records which devices hold a copy. */
TrainStatus train_store_set_replica_devices(TrainStore *store, int logical, int count,
                                            const int *ordinals);
int train_store_replica_count(const TrainStore *store, int logical);

/* The synchronization a caller must run around an update: the gradient merge (sum
 * the replicas) before it, the broadcast after it. The values are *bits*, and a
 * replicated parameter needs both, so a caller tests membership rather than
 * comparing against one value and concluding there is nothing to do. */
typedef enum {
    TRAIN_SYNC_NONE = 0,        /* single device: nothing to merge or broadcast */
    TRAIN_SYNC_GRAD_MERGE = 1,  /* replicas' gradients must be merged in FP32 first */
    TRAIN_SYNC_UPDATE_BCAST = 2,/* the published update must be broadcast to replicas */
} TrainSync;
TrainSync train_store_required_sync(const TrainStore *store, int logical);

/* ------------------------------------------------------------------ */
/* Training step: retained values                                     */
/* ------------------------------------------------------------------ */

/* A value the step keeps for its backward: a forward activation, a statistic, or a
 * GDN chunk-boundary state. `alias_of` is the index of the saved value this one
 * aliases (or -1), which is how the store can refuse to free a buffer twice or to
 * free a buffer another saved value still points at. `free_after` is the step-relative
 * index of the consumer that releases it. */
struct TrainSavedSpec {
    const char *name;
    int layer;
    long long elements;
    int alias_of;
    int free_after;
};

TrainStep *train_step_create(TrainStore *store, const struct TrainSavedSpec *saved, int count);
TrainStatus train_step_destroy(TrainStep *step);

int train_step_saved_count(const TrainStep *step);
TrainStatus train_step_saved_at(const TrainStep *step, int index, const char **name, int *layer,
                                long long *elements, void **buffer);

/* Retain (or hand back) one saved value's buffer. Freeing a buffer that another
 * live saved value aliases is refused. */
TrainStatus train_step_retain(TrainStep *step, int index, void *buffer);
TrainStatus train_step_free(TrainStep *step, int index);
int train_step_live_count(const TrainStep *step);

/* The step's version: a step reads the version its context borrowed, and a
 * publication between the forward and the backward would invalidate it, so the
 * store records the step as an active reader. */
long long train_step_version(const TrainStep *step);
int train_step_active(const TrainStep *step);

/* GDN chunk-boundary states: the plan requires a full-sequence gradient to cross
 * internal chunk boundaries unless truncated BPTT is chosen explicitly, so the step
 * records the schedule it retains. Returning the schedule rather than a boolean is
 * what makes the choice visible. */
typedef enum {
    TRAIN_BPTT_FULL_SEQUENCE = 0,  /* every chunk boundary's state is retained */
    TRAIN_BPTT_TRUNCATED = 1,      /* explicitly truncated (not used by this stage) */
} TrainBptt;
TrainStatus train_step_set_bptt(TrainStep *step, TrainBptt mode, int chunk_count);
TrainBptt train_step_bptt(const TrainStep *step);
int train_step_chunk_count(const TrainStep *step);
/* Elements one layer's retained chunk-boundary state costs under the set schedule. */
long long train_step_gdn_state_elements(const TrainStep *step, int value_heads, int head_dim);

/* ------------------------------------------------------------------ */
/* Teacher forcing                                                    */
/* ------------------------------------------------------------------ */

/* One selected position: the query position whose logits predict `label`, and the
 * label itself. The plan's teacher forcing is "next-token label shift, prompt and
 * padding masks and positions", one sequence at a time; this is that mapping as a
 * pure function so it is testable without a GPU or a model. */
struct TrainForcedPosition {
    int query;      /* position in [0, tokens) whose logits are used */
    int label;      /* the token id predicted at `query` + shift */
    int position;   /* the position the query's logits were computed at */
};

/* Build the selected-position list:
 *   - `shift` is normally 1 (the label of query t is token t+shift);
 *   - `mask` is optional; 0 marks a prompt/padding position and is skipped — the
 *     loss must not include it, so it is not selected;
 *   - the last `shift` positions have no label and are never selected;
 *   - `labels` is optional and parallel to the token ids. When it is null the target
 *     at position `tt` is `token_ids[tt]` (self-supervised next-token prediction);
 *     when it is given the target is `labels[tt]`, so a caller can force a different
 *     target (distillation, a corrected label).
 *
 * Returns the number selected, or a negative TrainStatus. A sequence with no
 * selected position is a valid result (count 0), not an error. */
int train_plan_teacher_forcing(int tokens, const int *token_ids, const int *labels,
                               const uint8_t *mask, const int64_t *positions, int shift,
                               struct TrainForcedPosition *out, int out_capacity);

/* ------------------------------------------------------------------ */
/* Errors                                                             */
/* ------------------------------------------------------------------ */

/* A human-readable description of the last failing call on this thread. The string
 * is owned by the library and valid until the next call on the same thread. */
const char *train_last_error(void);
/* Release the per-thread error string (a no-op on the library side today; present so
 * a caller's cleanup is symmetric). */
void train_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_TRAIN_H */
