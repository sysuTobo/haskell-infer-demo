/**
 * train.c - The trainable runtime's ownership objects (plan Stage 3).
 *
 * See csrc/include/train.h for the contract. This file is CUDA-free on purpose: it
 * is the bookkeeping the plan asks Haskell to drive and C to enforce, and every rule
 * in it (tying, exclusivity, publication, derived-copy refresh, free points) is
 * checked by a CPU test. The buffers stay opaque `void *` slots, so the same code
 * runs against host memory in the test and device memory in the engine.
 *
 * Three of the rules are rejections rather than assertions, because a comment cannot
 * stop a caller:
 *
 *   - `train_store_begin_update` fails while any context borrows the store or any
 *     step is live, so a reader can never observe a half-updated parameter set;
 *   - `train_store_destroy` and `train_step_destroy` fail while a step is live, so
 *     memory a backward still needs cannot be freed underneath it;
 *   - `train_store_end_update` fails while a derived copy of a published parameter is
 *     still stale, so "publishing refreshes `gdn_norm_f32`, not just its BF16 source"
 *     is enforced instead of remembered.
 */
#include "train.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Per-thread error text                                              */
/* ------------------------------------------------------------------ */

static _Thread_local char g_error[256];

const char *train_last_error(void) { return g_error; }

void train_clear_error(void) { g_error[0] = '\0'; }

static TrainStatus fail(TrainStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

/* ------------------------------------------------------------------ */
/* Objects                                                            */
/* ------------------------------------------------------------------ */

struct TrainParamEntry {
    int layer;
    int role;
    int logical;
};

struct TrainLogical {
    char name[64];
    long long elements;
    int frozen;
    int trainable;
    int alias_first;   /* index into store->params */
    int alias_count;
    void *slot[TRAIN_SLOT_COUNT];
    int written;       /* set by publish, cleared by end_update */
};

struct TrainDerived {
    int source_logical;
    int derived_logical;
    TrainDerivedKind kind;
    int stale;
};

struct TrainSaved {
    char name[64];
    int layer;
    long long elements;
    int alias_of;
    int free_after;
    void *buffer;
    int live;
};

struct TrainStore {
    struct TrainParamSpec specs[TRAIN_MAX_PARAMS];
    char templates[TRAIN_MAX_PARAMS][ENGINE_TEMPLATE_MAX];
    char names[TRAIN_MAX_PARAMS][64];
    struct TrainParamEntry params[TRAIN_MAX_PARAMS];
    int param_count;

    struct TrainLogical logical[TRAIN_MAX_LOGICAL];
    int logical_count;

    struct TrainDerived derived[TRAIN_MAX_DERIVED];
    int derived_count;

    /* Replicated parameters: which devices hold a copy. Fixed-size and inline, so
     * the store has no side tables to outlive it. */
    struct {
        int logical;
        int count;
        int ordinals[16];
    } replicas[16];
    int replica_count_total;

    long long version;
    int readers;            /* contexts + live steps */
    int in_update;
    int microbatch_total;   /* -1 until the first contribution of a step is recorded */
    int microbatches;
    int active_steps;
};

struct TrainContext {
    TrainStore *store;
    int is_rollout;
    long long borrowed_version;
    int destroyed;
};

struct TrainStep {
    TrainStore *store;
    struct TrainSaved saved[TRAIN_MAX_SAVED];
    int count;
    TrainBptt bptt;
    int chunk_count;
    long long version;
    int live;
    int destroyed;
};

/* ------------------------------------------------------------------ */
/* Store                                                              */
/* ------------------------------------------------------------------ */

TrainStore *train_store_create(const struct TrainParamSpec *specs, const char *const *templates,
                               const char *const *role_names, int count) {
    if (specs == NULL || templates == NULL || role_names == NULL || count <= 0) {
        fail(TRAIN_ERR_ARG, "train_store_create: null table or empty parameter set");
        return NULL;
    }
    if (count > TRAIN_MAX_PARAMS) {
        fail(TRAIN_ERR_RANGE, "train_store_create: %d parameters exceeds the %d limit", count,
             TRAIN_MAX_PARAMS);
        return NULL;
    }
    TrainStore *store = (TrainStore *)calloc(1, sizeof(TrainStore));
    if (store == NULL) {
        fail(TRAIN_ERR_STATE, "train_store_create: out of memory");
        return NULL;
    }
    store->param_count = count;
    store->microbatch_total = -1;
    for (int i = 0; i < count; ++i) {
        const size_t template_len = strlen(templates[i]);
        if (template_len + 1 > ENGINE_TEMPLATE_MAX) {
            free(store);
            fail(TRAIN_ERR_RANGE, "train_store_create: template %d is too long", i);
            return NULL;
        }
        store->specs[i] = specs[i];
        memcpy(store->templates[i], templates[i], template_len + 1);
        snprintf(store->names[i], sizeof(store->names[i]), "%s", role_names[i]);
        if (specs[i].elements <= 0) {
            free(store);
            fail(TRAIN_ERR_RANGE, "train_store_create: parameter %d has a non-positive size", i);
            return NULL;
        }

        /* Tying: same scope (layer) and the same template text means the same
         * tensor. A global role has layer -1, which is how embed and lmHead tie. */
        int logical = -1;
        for (int j = 0; j < i; ++j) {
            if (store->params[j].layer != specs[i].layer) continue;
            if (strcmp(store->templates[j], store->templates[i]) != 0) continue;
            logical = store->params[j].logical;
            break;
        }
        if (logical < 0) {
            logical = store->logical_count++;
            struct TrainLogical *entry = &store->logical[logical];
            snprintf(entry->name, sizeof(entry->name), "%s", role_names[i]);
            entry->elements = specs[i].elements;
            entry->alias_first = i;
            entry->alias_count = 0;
        } else {
            struct TrainLogical *entry = &store->logical[logical];
            if (entry->elements != specs[i].elements) {
                free(store);
                fail(TRAIN_ERR_ARG,
                     "train_store_create: tied parameters disagree on size (%lld vs %lld)", 
                     entry->elements, specs[i].elements);
                return NULL;
            }
        }
        store->logical[logical].alias_count += 1;
        store->params[i].layer = specs[i].layer;
        store->params[i].role = specs[i].role;
        store->params[i].logical = logical;
    }

    /* A logical parameter is trainable only if all of its aliases agree; a frozen
     * alias of a trainable parameter would be a contradiction. */
    for (int i = 0; i < count; ++i) {
        struct TrainLogical *entry = &store->logical[store->params[i].logical];
        const int trainable = store->specs[i].trainable && !store->specs[i].frozen;
        if (i == entry->alias_first) {
            entry->trainable = trainable;
            entry->frozen = store->specs[i].frozen;
        } else if (entry->trainable != trainable) {
            free(store);
            fail(TRAIN_ERR_ARG, "train_store_create: tied aliases disagree on trainability");
            return NULL;
        }
    }
    return store;
}

static void store_forget_reader(TrainStore *store) {
    if (store->readers > 0) store->readers -= 1;
}

TrainStatus train_store_destroy(TrainStore *store) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_destroy: null store");
    if (store->readers > 0 || store->active_steps > 0) {
        return fail(TRAIN_ERR_BUSY,
                    "train_store_destroy: refused with %d reader(s) and %d active step(s)",
                    store->readers, store->active_steps);
    }
    free(store);
    return TRAIN_OK;
}

int train_store_param_count(const TrainStore *store) { return store == NULL ? -1 : store->param_count; }
int train_store_logical_count(const TrainStore *store) { return store == NULL ? -1 : store->logical_count; }

int train_store_logical_of(const TrainStore *store, int layer, int role) {
    if (store == NULL) return -1;
    for (int i = 0; i < store->param_count; ++i) {
        if (store->params[i].layer == layer && store->params[i].role == role) {
            return store->params[i].logical;
        }
    }
    return -1;
}

int train_store_spec_count(const TrainStore *store) {
    return store == NULL ? -1 : store->param_count;
}

int train_store_spec_layer(const TrainStore *store, int index) {
    if (store == NULL || index < 0 || index >= store->param_count) return -1;
    return store->specs[index].layer;
}

int train_store_spec_role(const TrainStore *store, int index) {
    if (store == NULL || index < 0 || index >= store->param_count) return -1;
    return store->specs[index].role;
}

static int logical_valid(const TrainStore *store, int logical) {
    return store != NULL && logical >= 0 && logical < store->logical_count;
}

int train_store_alias_count(const TrainStore *store, int logical) {
    if (!logical_valid(store, logical)) return -1;
    return store->logical[logical].alias_count;
}

int train_store_alias_at(const TrainStore *store, int logical, int index) {
    if (!logical_valid(store, logical)) return -1;
    const struct TrainLogical *entry = &store->logical[logical];
    if (index < 0 || index >= entry->alias_count) return -1;
    return entry->alias_first + index;
}

long long train_store_elements(const TrainStore *store, int logical) {
    return logical_valid(store, logical) ? store->logical[logical].elements : -1;
}

const char *train_store_name(const TrainStore *store, int logical) {
    return logical_valid(store, logical) ? store->logical[logical].name : NULL;
}

int train_store_is_trainable(const TrainStore *store, int logical) {
    return logical_valid(store, logical) ? store->logical[logical].trainable : 0;
}

int train_store_is_frozen(const TrainStore *store, int logical) {
    return logical_valid(store, logical) ? store->logical[logical].frozen : 0;
}

TrainStatus train_store_set_slot(TrainStore *store, int logical, TrainSlot slot, void *buffer) {
    if (!logical_valid(store, logical)) {
        return fail(TRAIN_ERR_UNKNOWN_PARAM, "train_store_set_slot: no logical parameter %d", logical);
    }
    if (slot < 0 || slot >= TRAIN_SLOT_COUNT) {
        return fail(TRAIN_ERR_ARG, "train_store_set_slot: bad slot %d", (int)slot);
    }
    /* A frozen parameter has no training state: handing it one would let a caller
     * believe an update would land somewhere. */
    if (!store->logical[logical].trainable &&
        (slot == TRAIN_SLOT_MASTER || slot == TRAIN_SLOT_GRAD || slot == TRAIN_SLOT_OPT_M ||
         slot == TRAIN_SLOT_OPT_V)) {
        return fail(TRAIN_ERR_FROZEN,
                    "train_store_set_slot: parameter '%s' is frozen, so it has no %s buffer",
                    store->logical[logical].name,
                    slot == TRAIN_SLOT_MASTER ? "master"
                                              : (slot == TRAIN_SLOT_GRAD ? "gradient" : "optimizer"));
    }
    if (slot == TRAIN_SLOT_COMPUTE && buffer == NULL) {
        return fail(TRAIN_ERR_NO_SLOT, "train_store_set_slot: a compute weight cannot be null");
    }
    store->logical[logical].slot[slot] = buffer;
    return TRAIN_OK;
}

void *train_store_slot(const TrainStore *store, int logical, TrainSlot slot) {
    if (!logical_valid(store, logical) || slot < 0 || slot >= TRAIN_SLOT_COUNT) return NULL;
    return store->logical[logical].slot[slot];
}

int train_store_slot_buffer_count(const TrainStore *store, int logical, TrainSlot slot) {
    if (!logical_valid(store, logical)) return -1;
    /* The compute weight of a tied parameter exists once per alias: a publication
     * has to write the same value into every aliasing buffer, or one reader keeps
     * the old weight and the model quietly has two versions of one parameter. */
    return slot == TRAIN_SLOT_COMPUTE ? store->logical[logical].alias_count : 1;
}

/* ------------------------------------------------------------------ */
/* Derived copies                                                     */
/* ------------------------------------------------------------------ */

TrainStatus train_store_register_derived(TrainStore *store, int source_logical, int derived_logical,
                                         TrainDerivedKind kind) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_register_derived: null store");
    if (!logical_valid(store, source_logical) || !logical_valid(store, derived_logical)) {
        return fail(TRAIN_ERR_UNKNOWN_PARAM, "train_store_register_derived: unknown parameter");
    }
    if (source_logical == derived_logical) {
        return fail(TRAIN_ERR_ARG, "train_store_register_derived: a copy cannot derive from itself");
    }
    if (store->derived_count >= TRAIN_MAX_DERIVED) {
        return fail(TRAIN_ERR_RANGE, "train_store_register_derived: registry is full");
    }
    struct TrainDerived *entry = &store->derived[store->derived_count++];
    entry->source_logical = source_logical;
    entry->derived_logical = derived_logical;
    entry->kind = kind;
    entry->stale = 0;
    return TRAIN_OK;
}

int train_store_derived_count(const TrainStore *store) { return store == NULL ? -1 : store->derived_count; }

TrainStatus train_store_derived_at(const TrainStore *store, int index, int *source_logical,
                                   int *derived_logical, TrainDerivedKind *kind, int *stale) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_derived_at: null store");
    if (index < 0 || index >= store->derived_count) {
        return fail(TRAIN_ERR_RANGE, "train_store_derived_at: index %d out of range", index);
    }
    const struct TrainDerived *entry = &store->derived[index];
    if (source_logical) *source_logical = entry->source_logical;
    if (derived_logical) *derived_logical = entry->derived_logical;
    if (kind) *kind = entry->kind;
    if (stale) *stale = entry->stale;
    return TRAIN_OK;
}

TrainStatus train_store_derived_refreshed(TrainStore *store, int index) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_derived_refreshed: null store");
    if (index < 0 || index >= store->derived_count) {
        return fail(TRAIN_ERR_RANGE, "train_store_derived_refreshed: index %d out of range", index);
    }
    store->derived[index].stale = 0;
    return TRAIN_OK;
}

int train_store_stale_derived_count(const TrainStore *store) {
    if (store == NULL) return -1;
    int count = 0;
    for (int i = 0; i < store->derived_count; ++i) {
        if (store->derived[i].stale) ++count;
    }
    return count;
}

/* ------------------------------------------------------------------ */
/* Contexts, exclusivity, publication                                 */
/* ------------------------------------------------------------------ */

long long train_store_version(const TrainStore *store) { return store == NULL ? -1 : store->version; }

TrainContext *train_context_create(TrainStore *store, int is_rollout) {
    if (store == NULL) {
        fail(TRAIN_ERR_ARG, "train_context_create: null store");
        return NULL;
    }
    if (store->in_update) {
        fail(TRAIN_ERR_EXCLUSIVE, "train_context_create: an update is in progress");
        return NULL;
    }
    TrainContext *context = (TrainContext *)calloc(1, sizeof(TrainContext));
    if (context == NULL) {
        fail(TRAIN_ERR_STATE, "train_context_create: out of memory");
        return NULL;
    }
    context->store = store;
    context->is_rollout = is_rollout ? 1 : 0;
    context->borrowed_version = store->version;
    store->readers += 1;
    return context;
}

TrainStatus train_context_destroy(TrainContext *context) {
    if (context == NULL) return fail(TRAIN_ERR_ARG, "train_context_destroy: null context");
    if (context->destroyed) {
        return fail(TRAIN_ERR_STATE, "train_context_destroy: already destroyed");
    }
    store_forget_reader(context->store);
    context->destroyed = 1;
    free(context);
    return TRAIN_OK;
}

long long train_context_borrowed_version(const TrainContext *context) {
    return context == NULL ? -1 : context->borrowed_version;
}

TrainStatus train_store_begin_update(TrainStore *store) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_begin_update: null store");
    if (store->in_update) {
        return fail(TRAIN_ERR_EXCLUSIVE, "train_store_begin_update: an update is already open");
    }
    if (store->readers > 0) {
        /* This is the plan's "updating requires exclusive ownership": a context that
         * borrowed the version is reading it, and its scratch and sequence state were
         * built from it, so the update waits. */
        return fail(TRAIN_ERR_BUSY,
                    "train_store_begin_update: %d reader(s) still borrow version %lld",
                    store->readers, store->version);
    }
    store->in_update = 1;
    for (int i = 0; i < store->logical_count; ++i) store->logical[i].written = 0;
    return TRAIN_OK;
}

int train_store_in_update(const TrainStore *store) { return store == NULL ? 0 : store->in_update; }

TrainStatus train_store_publish(TrainStore *store) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_publish: null store");
    if (!store->in_update) {
        return fail(TRAIN_ERR_STATE, "train_store_publish: no update is open");
    }
    /* The store cannot know which master buffers a caller actually wrote, so it
     * treats every trainable parameter as published: that is the conservative
     * reading, and it is why end_update insists on refreshing the derived copies. */
    int published = 0;
    for (int i = 0; i < store->logical_count; ++i) {
        if (!store->logical[i].trainable) continue;
        store->logical[i].written = 1;
        ++published;
    }
    store->version += 1;
    for (int i = 0; i < store->derived_count; ++i) {
        if (store->logical[store->derived[i].source_logical].written) {
            store->derived[i].stale = 1;
        }
    }
    if (published == 0) {
        return fail(TRAIN_ERR_STATE, "train_store_publish: this store has no trainable parameter");
    }
    return TRAIN_OK;
}

TrainStatus train_store_end_update(TrainStore *store) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_end_update: null store");
    if (!store->in_update) {
        return fail(TRAIN_ERR_STATE, "train_store_end_update: no update is open");
    }
    const int stale = train_store_stale_derived_count(store);
    if (stale > 0) {
        return fail(TRAIN_ERR_STATE,
                    "train_store_end_update: %d derived copy/copies are still stale after the "
                    "publication", stale);
    }
    store->in_update = 0;
    return TRAIN_OK;
}

TrainStatus train_store_note_microbatch(TrainStore *store, int index, int count) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_note_microbatch: null store");
    if (count <= 0) return fail(TRAIN_ERR_ARG, "train_store_note_microbatch: empty accumulation");
    if (store->in_update) {
        return fail(TRAIN_ERR_STATE, "train_store_note_microbatch: a publication is pending");
    }
    if (store->microbatch_total < 0) {
        store->microbatch_total = count;
        store->microbatches = 0;
    } else if (store->microbatch_total != count) {
        /* The dW guarantee in the plan holds under a fixed schedule; a different
         * microbatch count is a different accumulation, so it needs its own step. */
        return fail(TRAIN_ERR_SCHEDULE,
                    "train_store_note_microbatch: schedule changed from %d to %d micro-batches",
                    store->microbatch_total, count);
    }
    if (index != store->microbatches) {
        return fail(TRAIN_ERR_SCHEDULE,
                    "train_store_note_microbatch: contribution %d arrived out of order (expected %d)",
                    index, store->microbatches);
    }
    store->microbatches += 1;
    if (store->microbatches == store->microbatch_total) store->microbatches = 0;
    return TRAIN_OK;
}

int train_store_microbatches_seen(const TrainStore *store) {
    return store == NULL ? -1 : store->microbatches;
}

/* ------------------------------------------------------------------ */
/* Cross-device replicas                                              */
/* ------------------------------------------------------------------ */

TrainStatus train_store_set_replica_devices(TrainStore *store, int logical, int count,
                                            const int *ordinals) {
    if (store == NULL) return fail(TRAIN_ERR_ARG, "train_store_set_replica_devices: null store");
    if (!logical_valid(store, logical)) {
        return fail(TRAIN_ERR_UNKNOWN_PARAM, "train_store_set_replica_devices: unknown parameter");
    }
    if (count < 1 || count > 16 || ordinals == NULL) {
        return fail(TRAIN_ERR_RANGE, "train_store_set_replica_devices: bad device list");
    }
    /* Replace an existing record rather than duplicating it, so a caller that sets
     * the placement twice is idempotent. */
    int slot = -1;
    for (int i = 0; i < store->replica_count_total; ++i) {
        if (store->replicas[i].logical == logical) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        if (store->replica_count_total >= 16) {
            return fail(TRAIN_ERR_RANGE,
                        "train_store_set_replica_devices: too many replicated parameters");
        }
        slot = store->replica_count_total++;
    }
    store->replicas[slot].logical = logical;
    store->replicas[slot].count = count;
    memcpy(store->replicas[slot].ordinals, ordinals, (size_t)count * sizeof(int));
    return TRAIN_OK;
}

int train_store_replica_count(const TrainStore *store, int logical) {
    if (store == NULL || !logical_valid(store, logical)) return -1;
    for (int i = 0; i < store->replica_count_total; ++i) {
        if (store->replicas[i].logical == logical) return store->replicas[i].count;
    }
    return 1;  /* not recorded is not the same as not replicated, but a single copy is
                * the only thing a caller may assume; the engine records every
                * replica it creates. */
}

TrainSync train_store_required_sync(const TrainStore *store, int logical) {
    if (store == NULL || !logical_valid(store, logical)) return TRAIN_SYNC_NONE;
    if (train_store_replica_count(store, logical) <= 1) return TRAIN_SYNC_NONE;
    /* A replicated parameter needs both, in this order: the replicas' gradients are
     * merged before the optimizer step, and the published update is broadcast after
     * it. Returning the pair (the enum values are bits) is what stops a caller from
     * inferring "nothing to do" from one of them. */
    return (TrainSync)(TRAIN_SYNC_GRAD_MERGE | TRAIN_SYNC_UPDATE_BCAST);
}

/* ------------------------------------------------------------------ */
/* Cross-device replicas (continued)                                  */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Training step                                                      */
/* ------------------------------------------------------------------ */

TrainStep *train_step_create(TrainStore *store, const struct TrainSavedSpec *saved, int count) {
    if (store == NULL || saved == NULL || count <= 0) {
        fail(TRAIN_ERR_ARG, "train_step_create: null store, null table or empty step");
        return NULL;
    }
    if (count > TRAIN_MAX_SAVED) {
        fail(TRAIN_ERR_RANGE, "train_step_create: %d saved values exceeds the %d limit", count,
             TRAIN_MAX_SAVED);
        return NULL;
    }
    if (store->in_update) {
        fail(TRAIN_ERR_EXCLUSIVE, "train_step_create: a publication is in progress");
        return NULL;
    }
    TrainStep *step = (TrainStep *)calloc(1, sizeof(TrainStep));
    if (step == NULL) {
        fail(TRAIN_ERR_STATE, "train_step_create: out of memory");
        return NULL;
    }
    step->store = store;
    step->count = count;
    step->version = store->version;
    step->bptt = TRAIN_BPTT_FULL_SEQUENCE;
    step->chunk_count = 1;
    for (int i = 0; i < count; ++i) {
        if (saved[i].name == NULL || saved[i].elements <= 0) {
            free(step);
            fail(TRAIN_ERR_ARG, "train_step_create: saved value %d is malformed", i);
            return NULL;
        }
        /* An alias must point backwards, or the free-point order it encodes has no
         * meaning. */
        if (saved[i].alias_of >= i || saved[i].alias_of < -1) {
            free(step);
            fail(TRAIN_ERR_ARG,
                 "train_step_create: saved value %d aliases %d, which is not an earlier entry",
                 i, saved[i].alias_of);
            return NULL;
        }
        if (saved[i].free_after < 0) {
            free(step);
            fail(TRAIN_ERR_ARG, "train_step_create: saved value %d has no free point", i);
            return NULL;
        }
        snprintf(step->saved[i].name, sizeof(step->saved[i].name), "%s", saved[i].name);
        step->saved[i].layer = saved[i].layer;
        step->saved[i].elements = saved[i].elements;
        step->saved[i].alias_of = saved[i].alias_of;
        step->saved[i].free_after = saved[i].free_after;
    }
    /* A live step is a reader of its version: its saved activations describe the
     * weights it was computed from, so an update cannot start while it exists. */
    store->readers += 1;
    store->active_steps += 1;
    return step;
}

TrainStatus train_step_destroy(TrainStep *step) {
    if (step == NULL) return fail(TRAIN_ERR_ARG, "train_step_destroy: null step");
    if (step->destroyed) return fail(TRAIN_ERR_STATE, "train_step_destroy: already destroyed");
    if (step->live > 0) {
        return fail(TRAIN_ERR_BUSY,
                    "train_step_destroy: %d saved value(s) are still retained; the backward "
                    "that consumes them has not released them", step->live);
    }
    store_forget_reader(step->store);
    if (step->store->active_steps > 0) step->store->active_steps -= 1;
    step->destroyed = 1;
    free(step);
    return TRAIN_OK;
}

int train_step_saved_count(const TrainStep *step) { return step == NULL ? -1 : step->count; }

TrainStatus train_step_saved_at(const TrainStep *step, int index, const char **name, int *layer,
                                long long *elements, void **buffer) {
    if (step == NULL) return fail(TRAIN_ERR_ARG, "train_step_saved_at: null step");
    if (index < 0 || index >= step->count) {
        return fail(TRAIN_ERR_RANGE, "train_step_saved_at: index %d out of range", index);
    }
    const struct TrainSaved *entry = &step->saved[index];
    if (name) *name = entry->name;
    if (layer) *layer = entry->layer;
    if (elements) *elements = entry->elements;
    if (buffer) *buffer = entry->buffer;
    return TRAIN_OK;
}

TrainStatus train_step_retain(TrainStep *step, int index, void *buffer) {
    if (step == NULL) return fail(TRAIN_ERR_ARG, "train_step_retain: null step");
    if (index < 0 || index >= step->count) {
        return fail(TRAIN_ERR_RANGE, "train_step_retain: index %d out of range", index);
    }
    if (step->saved[index].live) {
        return fail(TRAIN_ERR_STATE, "train_step_retain: '%s' is already retained",
                    step->saved[index].name);
    }
    step->saved[index].buffer = buffer;
    step->saved[index].live = 1;
    step->live += 1;
    return TRAIN_OK;
}

/* Does another live saved value still *point at* this entry's buffer? Releasing a
 * name that points at someone else's buffer is harmless -- the buffer stays alive
 * through the entry it aliases -- but releasing the entry others point at would
 * leave them holding a freed buffer. That asymmetry is also what makes the release
 * order well defined: aliases are released before the entry they alias. */
static int referenced_by_live(const TrainStep *step, int index) {
    for (int j = 0; j < step->count; ++j) {
        if (j == index || !step->saved[j].live) continue;
        if (step->saved[j].alias_of == index) return 1;
    }
    return 0;
}

TrainStatus train_step_free(TrainStep *step, int index) {
    if (step == NULL) return fail(TRAIN_ERR_ARG, "train_step_free: null step");
    if (index < 0 || index >= step->count) {
        return fail(TRAIN_ERR_RANGE, "train_step_free: index %d out of range", index);
    }
    if (!step->saved[index].live) {
        return fail(TRAIN_ERR_STATE, "train_step_free: '%s' is not retained", step->saved[index].name);
    }
    if (referenced_by_live(step, index)) {
        return fail(TRAIN_ERR_BUSY,
                    "train_step_free: '%s' is still referenced by a live saved value that "
                    "aliases it", step->saved[index].name);
    }
    step->saved[index].buffer = NULL;
    step->saved[index].live = 0;
    if (step->live > 0) step->live -= 1;
    return TRAIN_OK;
}

int train_step_live_count(const TrainStep *step) { return step == NULL ? -1 : step->live; }

long long train_step_version(const TrainStep *step) { return step == NULL ? -1 : step->version; }

int train_step_active(const TrainStep *step) { return step == NULL ? 0 : 1; }

TrainStatus train_step_set_bptt(TrainStep *step, TrainBptt mode, int chunk_count) {
    if (step == NULL) return fail(TRAIN_ERR_ARG, "train_step_set_bptt: null step");
    if (mode != TRAIN_BPTT_FULL_SEQUENCE && mode != TRAIN_BPTT_TRUNCATED) {
        return fail(TRAIN_ERR_ARG, "train_step_set_bptt: unknown mode %d", (int)mode);
    }
    if (chunk_count < 1) {
        return fail(TRAIN_ERR_RANGE, "train_step_set_bptt: a sequence has at least one chunk");
    }
    step->bptt = mode;
    step->chunk_count = chunk_count;
    return TRAIN_OK;
}

TrainBptt train_step_bptt(const TrainStep *step) {
    return step == NULL ? TRAIN_BPTT_FULL_SEQUENCE : step->bptt;
}

int train_step_chunk_count(const TrainStep *step) { return step == NULL ? -1 : step->chunk_count; }

long long train_step_gdn_state_elements(const TrainStep *step, int value_heads, int head_dim) {
    if (step == NULL || value_heads <= 0 || head_dim <= 0) return -1;
    /* Under full-sequence BPTT every chunk boundary's state has to be retained, so
     * the cost is the chunk count times one state; a truncated schedule would keep
     * only the window's worth, which is why the mode is recorded rather than
     * assumed. FP32 elements. */
    return (long long)step->chunk_count * value_heads * head_dim * head_dim;
}

/* ------------------------------------------------------------------ */
/* Teacher forcing                                                    */
/* ------------------------------------------------------------------ */

int train_plan_teacher_forcing(int tokens, const int *token_ids, const int *labels,
                               const uint8_t *mask, const int64_t *positions, int shift,
                               struct TrainForcedPosition *out, int out_capacity) {
    if (token_ids == NULL) {
        return -(int)fail(TRAIN_ERR_ARG, "train_plan_teacher_forcing: null token ids");
    }
    if (tokens < 0) {
        return -(int)fail(TRAIN_ERR_RANGE, "train_plan_teacher_forcing: negative token count");
    }
    if (shift < 1) {
        return -(int)fail(TRAIN_ERR_ARG, "train_plan_teacher_forcing: shift must be >= 1");
    }
    if (out_capacity < 0 || (out_capacity > 0 && out == NULL)) {
        return -(int)fail(TRAIN_ERR_ARG, "train_plan_teacher_forcing: bad output buffer");
    }
    int count = 0;
    /* The last `shift` positions predict nothing inside this sequence, so the loop
     * stops there: a next-token label that would come from outside the sequence is
     * not a label for this sequence. */
    for (int query = 0; query + shift < tokens; ++query) {
        const int target = query + shift;
        /* The mask marks the positions whose loss counts (the response tokens); a
         * prompt or padding position is skipped, not zero-weighted, so it cannot
         * contribute a spurious selected position. */
        if (mask != NULL && mask[target] == 0) continue;
        if (count < out_capacity) {
            out[count].query = query;
            out[count].label = labels != NULL ? labels[target] : token_ids[target];
            out[count].position = positions != NULL ? (int)positions[query] : query;
        }
        count += 1;
    }
    if (count > out_capacity && out != NULL) {
        return -(int)fail(TRAIN_ERR_RANGE,
                          "train_plan_teacher_forcing: %d selected positions do not fit %d slots",
                          count, out_capacity);
    }
    return count;
}
