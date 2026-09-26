/**
 * rollout_queue.c - Bounded-staleness rollout admission, protocol half (plan Stage 8).
 * See csrc/include/rollout_queue.h for the contract.
 *
 * CUDA-free: a bounded ring of completed groups plus the consumed-group ledger.
 */
#include "rollout_queue.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Per-thread error text                                              */
/* ------------------------------------------------------------------ */

static _Thread_local char g_error[256];

const char *rollout_queue_last_error(void) { return g_error; }

void rollout_queue_clear_error(void) { g_error[0] = '\0'; }

static RolloutStatus fail(RolloutStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

const char *rollout_status_name(RolloutStatus status) {
    switch (status) {
        case ROLLOUT_OK: return "ok";
        case ROLLOUT_ERR_ARG: return "arg";
        case ROLLOUT_ERR_STATE: return "state";
        case ROLLOUT_ERR_RANGE: return "range";
        case ROLLOUT_ERR_FULL: return "full";
        case ROLLOUT_ERR_EMPTY: return "empty";
        case ROLLOUT_ERR_DUPLICATE: return "duplicate";
        case ROLLOUT_ERR_STALE: return "stale";
        case ROLLOUT_ERR_MIXED_VERSION: return "mixed_version";
        case ROLLOUT_ERR_CONSUMED: return "consumed";
    }
    return "unknown";
}

struct RolloutQueue {
    struct RolloutGroupHeader *groups; /* a ring of `capacity` slots */
    int capacity;
    int head;
    int count;
    long long max_tokens;
    long long tokens;
    int max_behavior_versions;
    long long *consumed; /* the ledger: group ids already trained on */
    int consumed_count;
    int consumed_capacity;
};

/* ------------------------------------------------------------------ */
/* Creation                                                           */
/* ------------------------------------------------------------------ */

RolloutStatus rollout_queue_create(const struct RolloutQueueDims *dims, struct RolloutQueue **out) {
    if (dims == NULL || out == NULL) {
        return fail(ROLLOUT_ERR_ARG, "rollout_queue_create: null argument");
    }
    if (dims->max_groups < 1 || dims->max_tokens < 1 || dims->max_behavior_versions < 1) {
        return fail(ROLLOUT_ERR_RANGE,
                    "rollout_queue_create: the queue must be bounded in at least one group, one "
                    "token and one behavior version (got %d, %lld, %d)",
                    dims->max_groups, dims->max_tokens, dims->max_behavior_versions);
    }
    struct RolloutQueue *queue = (struct RolloutQueue *)calloc(1, sizeof(*queue));
    if (queue == NULL) return fail(ROLLOUT_ERR_STATE, "rollout_queue_create: out of memory");
    queue->groups = (struct RolloutGroupHeader *)calloc((size_t)dims->max_groups,
                                                        sizeof(*queue->groups));
    queue->consumed = (long long *)calloc(16, sizeof(*queue->consumed));
    if (queue->groups == NULL || queue->consumed == NULL) {
        free(queue->groups);
        free(queue->consumed);
        free(queue);
        return fail(ROLLOUT_ERR_STATE, "rollout_queue_create: out of memory");
    }
    queue->capacity = dims->max_groups;
    queue->max_tokens = dims->max_tokens;
    queue->max_behavior_versions = dims->max_behavior_versions;
    queue->consumed_capacity = 16;
    *out = queue;
    return ROLLOUT_OK;
}

RolloutStatus rollout_queue_destroy(struct RolloutQueue *queue) {
    if (queue == NULL) return fail(ROLLOUT_ERR_ARG, "rollout_queue_destroy: null queue");
    free(queue->groups);
    free(queue->consumed);
    free(queue);
    return ROLLOUT_OK;
}

/* ------------------------------------------------------------------ */
/* The ledger                                                         */
/* ------------------------------------------------------------------ */

static int ledger_index(const struct RolloutQueue *queue, long long group_id) {
    for (int i = 0; i < queue->consumed_count; ++i) {
        if (queue->consumed[i] == group_id) return i;
    }
    return -1;
}

static RolloutStatus ledger_append(struct RolloutQueue *queue, long long group_id) {
    if (queue->consumed_count == queue->consumed_capacity) {
        const int grown = queue->consumed_capacity * 2;
        long long *bigger = (long long *)realloc(queue->consumed, (size_t)grown * sizeof(long long));
        if (bigger == NULL) {
            return fail(ROLLOUT_ERR_STATE, "rollout_queue: the consumed ledger could not grow");
        }
        queue->consumed = bigger;
        queue->consumed_capacity = grown;
    }
    queue->consumed[queue->consumed_count++] = group_id;
    return ROLLOUT_OK;
}

int rollout_queue_consumed_count(const struct RolloutQueue *queue) {
    return queue == NULL ? 0 : queue->consumed_count;
}

int rollout_queue_is_consumed(const struct RolloutQueue *queue, long long group_id) {
    return queue == NULL ? 0 : (ledger_index(queue, group_id) >= 0);
}

long long rollout_queue_lag(long long learner_committed_version, long long behavior_version) {
    return learner_committed_version - behavior_version;
}

/* ------------------------------------------------------------------ */
/* Admission                                                          */
/* ------------------------------------------------------------------ */

static const struct RolloutGroupHeader *slot_at(const struct RolloutQueue *queue, int i) {
    return &queue->groups[(queue->head + i) % queue->capacity];
}

static int live_versions(const struct RolloutQueue *queue) {
    long long seen[64];
    int n = 0;
    for (int i = 0; i < queue->count; ++i) {
        const long long v = slot_at(queue, i)->behavior_version;
        int known = 0;
        for (int j = 0; j < n; ++j) {
            if (seen[j] == v) known = 1;
        }
        if (!known && n < 64) seen[n++] = v;
    }
    return n;
}

RolloutStatus rollout_queue_enqueue(struct RolloutQueue *queue,
                                    const struct RolloutGroupHeader *group) {
    if (queue == NULL || group == NULL) {
        return fail(ROLLOUT_ERR_ARG, "rollout_queue_enqueue: null argument");
    }
    if (group->response_versions == NULL) {
        return fail(ROLLOUT_ERR_ARG, "rollout_queue_enqueue: a group without per-response versions");
    }
    /* Whole completed groups only. A group of fewer than two responses has no
     * group-relative advantage, and `admitted` is the verifier's count, so a partially
     * admitted group is refused rather than renormalised. */
    if (group->responses < 2) {
        return fail(ROLLOUT_ERR_STATE,
                    "rollout_queue_enqueue: group %lld has %d responses; a group needs at least 2",
                    group->group_id, group->responses);
    }
    if (group->admitted < 2 || group->admitted > group->responses) {
        return fail(ROLLOUT_ERR_STATE,
                    "rollout_queue_enqueue: group %lld admitted %d of %d responses; a partial "
                    "group is not renormalised", group->group_id, group->admitted, group->responses);
    }
    if (group->tokens < 1) {
        return fail(ROLLOUT_ERR_STATE,
                    "rollout_queue_enqueue: group %lld has no trainable token", group->group_id);
    }
    if (group->sampling_config == NULL || group->reward_version == NULL) {
        return fail(ROLLOUT_ERR_ARG,
                    "rollout_queue_enqueue: group %lld does not identify its sampling "
                    "configuration and verifier", group->group_id);
    }
    /* One group, one behavior version: the plan says not to combine different behavior
     * versions into one group, so the responses and the header must agree. */
    for (int i = 0; i < group->responses; ++i) {
        if (group->response_versions[i] != group->behavior_version) {
            return fail(ROLLOUT_ERR_MIXED_VERSION,
                        "rollout_queue_enqueue: group %lld response %d was generated under "
                        "version %lld but the group declares %lld", group->group_id, i,
                        group->response_versions[i], group->behavior_version);
        }
    }
    if (ledger_index(queue, group->group_id) >= 0) {
        return fail(ROLLOUT_ERR_DUPLICATE,
                    "rollout_queue_enqueue: group %lld was already consumed; a retry must not "
                    "count a response twice", group->group_id);
    }
    for (int i = 0; i < queue->count; ++i) {
        if (slot_at(queue, i)->group_id == group->group_id) {
            return fail(ROLLOUT_ERR_DUPLICATE,
                        "rollout_queue_enqueue: group %lld is already queued", group->group_id);
        }
    }
    if (queue->count == queue->capacity) {
        return fail(ROLLOUT_ERR_FULL,
                    "rollout_queue_enqueue: the queue holds its %d groups; the actor must "
                    "backpressure rather than drop a group", queue->capacity);
    }
    if (queue->tokens + group->tokens > queue->max_tokens) {
        return fail(ROLLOUT_ERR_FULL,
                    "rollout_queue_enqueue: group %lld's %lld tokens would take the queue to %lld "
                    "of its %lld-token bound", group->group_id, group->tokens,
                    queue->tokens + group->tokens, queue->max_tokens);
    }
    /* The bounded allowlist of live behavior versions: publication is blocked while the
     * queue already holds the maximum number of distinct versions. */
    {
        int distinct = live_versions(queue);
        int known = 0;
        for (int i = 0; i < queue->count; ++i) {
            if (slot_at(queue, i)->behavior_version == group->behavior_version) known = 1;
        }
        if (!known) ++distinct;
        if (distinct > queue->max_behavior_versions) {
            return fail(ROLLOUT_ERR_FULL,
                        "rollout_queue_enqueue: group %lld would make %d live behavior versions, "
                        "above the bound of %d", group->group_id, distinct,
                        queue->max_behavior_versions);
        }
    }

    struct RolloutGroupHeader *slot = &queue->groups[(queue->head + queue->count) % queue->capacity];
    *slot = *group;
    slot->anchor_version = -1;
    ++queue->count;
    queue->tokens += group->tokens;
    return ROLLOUT_OK;
}

RolloutStatus rollout_queue_dequeue(struct RolloutQueue *queue, long long learner_committed_version,
                                    int max_lag, struct RolloutGroupHeader *out) {
    if (queue == NULL || out == NULL) {
        return fail(ROLLOUT_ERR_ARG, "rollout_queue_dequeue: null argument");
    }
    if (max_lag < 0) {
        return fail(ROLLOUT_ERR_RANGE, "rollout_queue_dequeue: a negative lag cap is not a cap");
    }
    if (queue->count == 0) {
        return fail(ROLLOUT_ERR_EMPTY, "rollout_queue_dequeue: the queue holds no group");
    }
    const struct RolloutGroupHeader *head = slot_at(queue, 0);
    const long long lag = rollout_queue_lag(learner_committed_version, head->behavior_version);
    if (lag < 0) {
        return fail(ROLLOUT_ERR_STALE,
                    "rollout_queue_dequeue: group %lld was generated under version %lld, ahead of "
                    "the learner's committed version %lld (lag %lld); the behavior version is "
                    "unknown to this learner", head->group_id, head->behavior_version,
                    learner_committed_version, lag);
    }
    if (lag > (long long)max_lag) {
        return fail(ROLLOUT_ERR_STALE,
                    "rollout_queue_dequeue: group %lld's lag is %lld, above the cap of %d; the "
                    "whole group is rejected rather than partially renormalised", head->group_id,
                    lag, max_lag);
    }

    *out = *head;
    out->anchor_version = learner_committed_version;
    const long long consumed_tokens = head->tokens;
    queue->head = (queue->head + 1) % queue->capacity;
    --queue->count;
    queue->tokens -= consumed_tokens;
    const RolloutStatus appended = ledger_append(queue, out->group_id);
    if (appended != ROLLOUT_OK) return appended;
    return ROLLOUT_OK;
}

RolloutStatus rollout_queue_drop_head(struct RolloutQueue *queue,
                                      struct RolloutGroupHeader *out) {
    if (queue == NULL || out == NULL) {
        return fail(ROLLOUT_ERR_ARG, "rollout_queue_drop_head: null argument");
    }
    if (queue->count == 0) {
        return fail(ROLLOUT_ERR_EMPTY, "rollout_queue_drop_head: the queue holds no group");
    }
    const struct RolloutGroupHeader *head = slot_at(queue, 0);
    *out = *head;
    out->anchor_version = -1;
    queue->head = (queue->head + 1) % queue->capacity;
    --queue->count;
    queue->tokens -= head->tokens;
    /* Deliberately not in the consumed ledger: the group was never trained on, so its
     * responses may be regenerated under a fresh attempt identity. */
    return ROLLOUT_OK;
}

RolloutStatus rollout_queue_size(const struct RolloutQueue *queue, int *out_groups,
                                 long long *out_tokens) {
    if (queue == NULL) return fail(ROLLOUT_ERR_ARG, "rollout_queue_size: null queue");
    if (out_groups != NULL) *out_groups = queue->count;
    if (out_tokens != NULL) *out_tokens = queue->tokens;
    return ROLLOUT_OK;
}
