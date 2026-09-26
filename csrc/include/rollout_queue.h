/**
 * rollout_queue.h - Bounded-staleness rollout admission, protocol half (plan Stage 8).
 *
 * docs/plan-numeric-contract.md, Stage 8 is a *proposal* that "starts only after a
 * synchronous algorithm and parameter publication protocol pass all relevant gates", and
 * its own recommendation is the order to build it in: "build A's protocol first, with lag
 * zero, then prototype B". This module is that first step and nothing beyond it - the
 * bounded `LearnerQueue`, the immutable behavior version a group is admitted under, and
 * the lag accounting the plan defines as
 * `learner_committed_version - behavior_version` in optimizer steps.
 *
 * What is deliberately *not* here is the GPU half: there is no weight snapshot, no device
 * cache lease, no publication transfer and no throughput measurement. Those require the
 * actor/learner resource split the plan defers to "before positive-lag async experiments",
 * and claiming them from a queue would be exactly the kind of unmeasured assertion the
 * plan's acceptance discipline rules out. The queue keeps the protocol honest on its own:
 *
 *   - **whole completed groups only.** An empty or partial group is refused, and every
 *     response in a group must carry the same behavior version, because the plan says
 *     "do not train the first k responses to finish" or "combine different behavior
 *     versions into one group" and "do not renormalize a partial failed group".
 *   - **bounded in groups and tokens, and in live behavior versions.** Capacity is
 *     backpressure (a distinct status, not a silent drop), which is what "bound queue
 *     capacity in groups and tokens" and "block/coalesce publication when capacity is
 *     exhausted" need.
 *   - **lag is enforced at admission.** A group whose behavior version is newer than the
 *     learner's committed one is an unknown version, and one further behind than the
 *     declared cap is stale; both refuse the whole group rather than training a subset.
 *   - **consumption is a ledger.** A group can be consumed once: a duplicate enqueue after
 *     consumption is refused, so "retries cannot count a response twice and restart cannot
 *     silently retrain a consumed batch". The ledger survives a queue reset by being
 *     owned by the queue, not by the group.
 *
 * The admission's version relation is also the thing Stage 6 already names: a difference
 * measured with `behavior_version != anchor_version` is a **policy change**, not a
 * numerical mismatch, and the gate reports it that way through `alignment_classify`
 * rather than as a numerical bound.
 *
 * This file is CUDA-free, like train.h and backward.h: it is a bounded container and a
 * version ledger a CPU test drives directly.
 */
#ifndef HASKELL_INFER_ROLLOUT_QUEUE_H
#define HASKELL_INFER_ROLLOUT_QUEUE_H

#ifdef __cplusplus
extern "C" {
#endif

#define ROLLOUT_QUEUE_VERSION 1

typedef enum {
    ROLLOUT_OK = 0,
    ROLLOUT_ERR_ARG = 1,
    ROLLOUT_ERR_STATE = 2,
    ROLLOUT_ERR_RANGE = 3,
    ROLLOUT_ERR_FULL = 4,      /* capacity exhausted: backpressure, not a silent drop */
    ROLLOUT_ERR_EMPTY = 5,
    ROLLOUT_ERR_DUPLICATE = 6, /* the group id is already queued or already consumed */
    ROLLOUT_ERR_STALE = 7,     /* the group's behavior version is unknown or beyond the cap */
    ROLLOUT_ERR_MIXED_VERSION = 8,
    ROLLOUT_ERR_CONSUMED = 9,
} RolloutStatus;

const char *rollout_status_name(RolloutStatus status);
const char *rollout_queue_last_error(void);
void rollout_queue_clear_error(void);

/* One completed group, as the actor hands it to the learner. `response_versions` is the
 * behavior version of each response (`responses` entries); they must all agree. `tokens`
 * is the group's total trainable response tokens, the queue's second bound. `admitted`
 * is how many responses a verifier admitted (>= 2, <= responses). `sampling_config` and
 * `reward_version` identify the generation configuration and the deterministic verifier,
 * so a group from a different sampling rule is not silently mixed in. */
struct RolloutGroupHeader {
    const long long *response_versions;
    const char *sampling_config;
    const char *reward_version;
    int responses;
    int admitted;
    long long group_id;
    long long behavior_version;
    long long tokens;
    /* Written by dequeue: the learner's committed version at admission, the plan's
     * proximal anchor for this batch. */
    long long anchor_version;
};

struct RolloutQueueDims {
    int max_groups;             /* >= 1: the queue's bound in whole groups */
    long long max_tokens;       /* >= 1: the queue's bound in trainable response tokens */
    int max_behavior_versions;  /* >= 1: the bounded allowlist of live behavior versions */
};

struct RolloutQueue;

RolloutStatus rollout_queue_create(const struct RolloutQueueDims *dims, struct RolloutQueue **out);

/* Frees the queue and its ledger. A queue may be destroyed with groups still queued: the
 * plan's restart rule is to discard unconsumed and in-flight groups, while the consumed
 * ledger is what must not be forgotten - so the caller that needs the ledger across a
 * restart keeps the queue, and one that destroys it is discarding the record. */
RolloutStatus rollout_queue_destroy(struct RolloutQueue *queue);

/* Enqueue a whole completed group. Refuses an empty/partial group, a group whose responses
 * disagree on their behavior version, a duplicate group id (queued or already consumed),
 * and a group that would exceed either capacity bound or the live-version allowlist. */
RolloutStatus rollout_queue_enqueue(struct RolloutQueue *queue, const struct RolloutGroupHeader *group);

/* Take the oldest admissible group and stamp it with the learner's committed version as its
 * anchor. Refuses an empty queue (ROLLOUT_ERR_EMPTY) and a group whose lag is outside
 * [0, max_lag] (ROLLOUT_ERR_STALE): a negative lag is a version the learner has not
 * committed, and one above the cap is too stale to train on. The plan's "enforce lag at
 * dispatch/admission" is this check. */
RolloutStatus rollout_queue_dequeue(struct RolloutQueue *queue, long long learner_committed_version,
                                    int max_lag, struct RolloutGroupHeader *out);

/* Discard the oldest group with a recorded reason (a stale or deadline-expired group),
 * which is the plan's "reject the whole group and record the reason" rather than a partial
 * renormalisation. The group is not added to the consumed ledger: it was never trained on. */
RolloutStatus rollout_queue_drop_head(struct RolloutQueue *queue, struct RolloutGroupHeader *out);

RolloutStatus rollout_queue_size(const struct RolloutQueue *queue, int *out_groups,
                                 long long *out_tokens);
int rollout_queue_consumed_count(const struct RolloutQueue *queue);
int rollout_queue_is_consumed(const struct RolloutQueue *queue, long long group_id);

/* The plan's definition, in optimizer steps. Negative means the group was generated under a
 * version the learner has not committed. */
long long rollout_queue_lag(long long learner_committed_version, long long behavior_version);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_ROLLOUT_QUEUE_H */
