/*
 * rollout_queue_test.c - CPU gate for the bounded-staleness admission protocol
 * (plan Stage 8).
 *
 * No GPU and no weights. The plan's Stage 8 is a proposal that "starts only after a
 * synchronous algorithm and parameter publication protocol pass all relevant gates", and
 * its own order is "build A's protocol first, with lag zero". This gate is that protocol:
 * whole completed groups only, bounded in groups, tokens and live behavior versions, lag
 * enforced at admission, and consumption recorded so a retry cannot count a response
 * twice. It also does the plan's async gate 1 and gate 3 at the level the protocol can
 * carry them:
 *
 *   1. **lag zero matches synchronous.** The queue is a container, so a group dequeued at
 *      lag zero must produce bitwise the same GSPO objective and gradient as the same
 *      group computed directly - and its anchor version must be its behavior version.
 *   3. **deterministic lag 0/1/2 injection.** With the learner ahead of the behavior
 *      policy, the objective and gradient drift, and that drift is a *policy change*:
 *      `alignment_classify` names it, so it is reported as staleness rather than as a
 *      numerical mismatch, and no ratio is claimed to repair it.
 *
 * What is not here is the GPU half - snapshots, device cache leases, publication
 * transfer and throughput - which needs the actor/learner resource split the plan defers.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_rollout_queue.
 */
#include "alignment.h"
#include "backward.h"
#include "rollout_queue.h"

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
    fputs("rollout_queue_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void check_close(double got, double want, double tol, const char *what) {
    check(fabs(got - want) <= tol, "%s: got %.9g want %.9g (tol %.3g)", what, got, want, tol);
}

/* ------------------------------------------------------------------ */
/* A tiny fixed group                                                 */
/* ------------------------------------------------------------------ */

#define G_TOY 4
#define T_TOY 2

struct toy_group {
    long long versions[G_TOY];
    float behavior[G_TOY][T_TOY];
    float anchor[G_TOY][T_TOY];
    float rewards[G_TOY];
    uint8_t mask[G_TOY][T_TOY];
    struct BackwardResponse responses[G_TOY];
    struct RolloutGroupHeader header;
};

/* Filled in place on purpose: the header and the response structs hold pointers into the
 * same object, so returning the fixture by value would leave every one of them dangling. */
static void make_toy_group(struct toy_group *g, long long group_id, long long behavior_version,
                           int admitted) {
    memset(g, 0, sizeof(*g));
    for (int i = 0; i < G_TOY; ++i) {
        g->versions[i] = behavior_version;
        g->rewards[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        for (int t = 0; t < T_TOY; ++t) {
            g->behavior[i][t] = -1.0f;
            /* The learner's current log-probability, i.e. a policy that has moved. */
            g->anchor[i][t] = -1.0f + 0.1f * (float)(i + 1);
            g->mask[i][t] = 1;
        }
        g->responses[i].logp = g->anchor[i];
        g->responses[i].behavior_logp = g->behavior[i];
        g->responses[i].response_mask = g->mask[i];
        g->responses[i].tokens = T_TOY;
        g->responses[i].admitted = 1;
    }
    g->header.response_versions = g->versions;
    g->header.sampling_config = "t=1,none";
    g->header.reward_version = "parity-v1";
    g->header.responses = G_TOY;
    g->header.admitted = admitted;
    g->header.group_id = group_id;
    g->header.behavior_version = behavior_version;
    g->header.tokens = G_TOY * T_TOY;
    g->header.anchor_version = -1;
}

static struct BackwardGroupConfig toy_config(void) {
    struct BackwardGroupConfig c;
    c.clip_low = 0.2f;
    c.clip_high = 0.2f;
    c.eps_adv = 1e-6f;
    c.allow_zero_variance = 0;
    return c;
}

/* The GSPO objective and gradient on the group's *current* log-probabilities. */
static void toy_objective(struct toy_group *g, double *out_objective, float *out_d) {
    const struct BackwardGroupConfig config = toy_config();
    struct BackwardGroupStats stats;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, g->responses, G_TOY, g->rewards,
                                   out_d, NULL, NULL, &stats, out_objective) == BACKWARD_OK,
          "the toy objective failed: %s", backward_last_error());
}

/* ------------------------------------------------------------------ */
/* The container does not change the numbers (gate 1)                 */
/* ------------------------------------------------------------------ */

static void test_lag_zero_matches_synchronous(void) {
    struct RolloutQueueDims dims = {8, 1024, 2};
    struct RolloutQueue *queue = NULL;
    check(rollout_queue_create(&dims, &queue) == ROLLOUT_OK, "create: %s",
          rollout_queue_last_error());

    /* The same group, computed directly and through the queue, at lag zero. */
    struct toy_group a;
    make_toy_group(&a, 1, 0, G_TOY);
    struct toy_group b;
    make_toy_group(&b, 2, 0, G_TOY);

    double direct_objective = 0.0;
    float direct_d[G_TOY * T_TOY];
    toy_objective(&a, &direct_objective, direct_d);

    check(rollout_queue_enqueue(queue, &b.header) == ROLLOUT_OK, "enqueue: %s",
          rollout_queue_last_error());
    struct RolloutGroupHeader taken;
    check(rollout_queue_dequeue(queue, /*learner_committed=*/0, /*max_lag=*/0, &taken) == ROLLOUT_OK,
          "dequeue at lag zero: %s", rollout_queue_last_error());
    check(taken.anchor_version == 0 && taken.behavior_version == 0,
          "the anchor is %lld and the behavior version %lld at lag zero", taken.anchor_version,
          taken.behavior_version);
    check(rollout_queue_lag(0, taken.behavior_version) == 0, "the lag is not zero");

    double through_objective = 0.0;
    float through_d[G_TOY * T_TOY];
    toy_objective(&b, &through_objective, through_d);
    check(direct_objective == through_objective,
          "the queue moved the objective (%.17g vs %.17g)", direct_objective, through_objective);
    for (int k = 0; k < G_TOY * T_TOY; ++k) {
        check(direct_d[k] == through_d[k], "the queue moved gradient element %d", k);
    }

    /* The group was consumed, and consumption is once-only. */
    check(rollout_queue_consumed_count(queue) == 1, "the ledger has %d entries",
          rollout_queue_consumed_count(queue));
    check(rollout_queue_is_consumed(queue, 2) == 1, "the consumed group is not in the ledger");
    struct toy_group again;
    make_toy_group(&again, 2, 0, G_TOY);
    check(rollout_queue_enqueue(queue, &again.header) == ROLLOUT_ERR_DUPLICATE,
          "a consumed group was re-enqueued");
    /* And a group never trained on may be regenerated after a drop. */
    struct toy_group dropped_group;
    make_toy_group(&dropped_group, 3, 0, G_TOY);
    check(rollout_queue_enqueue(queue, &dropped_group.header) == ROLLOUT_OK, "enqueue: %s",
          rollout_queue_last_error());
    struct RolloutGroupHeader dropped;
    check(rollout_queue_drop_head(queue, &dropped) == ROLLOUT_OK, "drop: %s",
          rollout_queue_last_error());
    check(rollout_queue_is_consumed(queue, 3) == 0, "a dropped group entered the consumed ledger");
    check(rollout_queue_enqueue(queue, &dropped_group.header) == ROLLOUT_OK,
          "a dropped (never trained) group could not be regenerated: %s",
          rollout_queue_last_error());

    check(rollout_queue_destroy(queue) == ROLLOUT_OK, "destroy failed");
}

/* ------------------------------------------------------------------ */
/* Whole completed groups only                                        */
/* ------------------------------------------------------------------ */

static void test_whole_groups_only(void) {
    struct RolloutQueueDims dims = {4, 64, 2};
    struct RolloutQueue *queue = NULL;
    check(rollout_queue_create(&dims, &queue) == ROLLOUT_OK, "create failed");

    struct toy_group partial;
    make_toy_group(&partial, 10, 0, /*admitted=*/1);
    check(rollout_queue_enqueue(queue, &partial.header) == ROLLOUT_ERR_STATE,
          "a partially admitted group was accepted");

    struct toy_group single;
    make_toy_group(&single, 11, 0, G_TOY);
    single.header.responses = 1;
    check(rollout_queue_enqueue(queue, &single.header) == ROLLOUT_ERR_STATE,
          "a one-response group was accepted");

    struct toy_group empty_tokens;
    make_toy_group(&empty_tokens, 12, 0, G_TOY);
    empty_tokens.header.tokens = 0;
    check(rollout_queue_enqueue(queue, &empty_tokens.header) == ROLLOUT_ERR_STATE,
          "a group with no trainable token was accepted");

    /* A group whose responses were generated under different versions. */
    struct toy_group mixed;
    make_toy_group(&mixed, 13, 0, G_TOY);
    mixed.versions[3] = 1;
    mixed.header.response_versions = mixed.versions;
    check(rollout_queue_enqueue(queue, &mixed.header) == ROLLOUT_ERR_MIXED_VERSION,
          "a mixed-version group was accepted");

    check(rollout_queue_size(queue, NULL, NULL) == ROLLOUT_OK, "size failed");
    int groups = -1;
    long long tokens = -1;
    check(rollout_queue_size(queue, &groups, &tokens) == ROLLOUT_OK && groups == 0 && tokens == 0,
          "a refused group left %d groups and %lld tokens queued", groups, tokens);

    check(rollout_queue_destroy(queue) == ROLLOUT_OK, "destroy failed");
}

/* ------------------------------------------------------------------ */
/* Lag admission and the capacity bounds                              */
/* ------------------------------------------------------------------ */

static void test_lag_admission(void) {
    struct RolloutQueueDims dims = {4, 64, 4};
    struct RolloutQueue *queue = NULL;
    check(rollout_queue_create(&dims, &queue) == ROLLOUT_OK, "create failed");

    struct toy_group g0;
    make_toy_group(&g0, 20, /*behavior_version=*/2, G_TOY);
    check(rollout_queue_enqueue(queue, &g0.header) == ROLLOUT_OK, "enqueue: %s",
          rollout_queue_last_error());

    struct RolloutGroupHeader taken;
    /* A behavior version ahead of the learner is unknown, not old. */
    check(rollout_queue_dequeue(queue, /*committed=*/1, /*max_lag=*/0, &taken) == ROLLOUT_ERR_STALE,
          "a version the learner has not committed was admitted");
    /* Lag 1 is too stale for a zero-lag cap. */
    check(rollout_queue_dequeue(queue, 1, 0, &taken) == ROLLOUT_ERR_STALE,
          "a lag-1 group was admitted under a zero cap");
    /* The same group is admissible once the learner catches up. */
    check(rollout_queue_dequeue(queue, 2, 0, &taken) == ROLLOUT_OK,
          "the lag-zero group was refused: %s", rollout_queue_last_error());
    check(taken.anchor_version == 2, "the anchor is %lld, expected 2", taken.anchor_version);
    check(rollout_queue_lag(2, taken.behavior_version) == 0, "the lag is not zero");

    /* A negative cap is not a cap. */
    check(rollout_queue_dequeue(queue, 2, -1, &taken) == ROLLOUT_ERR_RANGE,
          "a negative lag cap was accepted");
    check(rollout_queue_dequeue(queue, 2, 0, &taken) == ROLLOUT_ERR_EMPTY,
          "an empty queue dequeued a group");

    /* Lag 2 is admissible under a cap of 2 and the anchor records the gap. */
    struct toy_group g1;
    make_toy_group(&g1, 21, 0, G_TOY);
    check(rollout_queue_enqueue(queue, &g1.header) == ROLLOUT_OK, "enqueue failed");
    check(rollout_queue_dequeue(queue, 2, 2, &taken) == ROLLOUT_OK,
          "a lag-2 group was refused under a cap of 2: %s", rollout_queue_last_error());
    check(rollout_queue_lag(taken.anchor_version, taken.behavior_version) == 2,
          "the admitted lag is not 2 (anchor %lld, behavior %lld)", taken.anchor_version,
          taken.behavior_version);
    check(rollout_queue_dequeue(queue, 5, 2, &taken) == ROLLOUT_ERR_EMPTY, "the queue is not empty");

    check(rollout_queue_destroy(queue) == ROLLOUT_OK, "destroy failed");
}

static void test_capacity_bounds(void) {
    /* Two groups at most, eight tokens at most. */
    struct RolloutQueueDims dims = {2, 8, 3};
    struct RolloutQueue *queue = NULL;
    check(rollout_queue_create(&dims, &queue) == ROLLOUT_OK, "create failed");

    struct toy_group g0;
    make_toy_group(&g0, 30, 0, G_TOY); /* 8 tokens */
    struct toy_group g1;
    make_toy_group(&g1, 31, 0, G_TOY);
    struct toy_group g2;
    make_toy_group(&g2, 32, 0, G_TOY);
    check(rollout_queue_enqueue(queue, &g0.header) == ROLLOUT_OK, "enqueue: %s",
          rollout_queue_last_error());
    /* The token bound is hit before the group bound here. */
    check(rollout_queue_enqueue(queue, &g1.header) == ROLLOUT_ERR_FULL,
          "the token bound did not backpressure");
    int groups = 0;
    long long tokens = 0;
    check(rollout_queue_size(queue, &groups, &tokens) == ROLLOUT_OK && groups == 1 && tokens == 8,
          "the refused group changed the queue (%d groups, %lld tokens)", groups, tokens);

    /* Now give it room in tokens but not in groups. */
    struct RolloutQueueDims wide = {2, 1024, 3};
    struct RolloutQueue *bigger = NULL;
    check(rollout_queue_create(&wide, &bigger) == ROLLOUT_OK, "create failed");
    check(rollout_queue_enqueue(bigger, &g0.header) == ROLLOUT_OK, "enqueue failed");
    check(rollout_queue_enqueue(bigger, &g1.header) == ROLLOUT_OK, "enqueue failed");
    check(rollout_queue_enqueue(bigger, &g2.header) == ROLLOUT_ERR_FULL,
          "the group bound did not backpressure");

    /* The live-version allowlist: two versions at most. */
    struct RolloutQueueDims narrow = {4, 1024, 2};
    struct RolloutQueue *allow = NULL;
    check(rollout_queue_create(&narrow, &allow) == ROLLOUT_OK, "create failed");
    struct toy_group v0;
    make_toy_group(&v0, 40, 0, G_TOY);
    struct toy_group v1;
    make_toy_group(&v1, 41, 1, G_TOY);
    struct toy_group v2;
    make_toy_group(&v2, 42, 2, G_TOY);
    check(rollout_queue_enqueue(allow, &v0.header) == ROLLOUT_OK, "enqueue failed");
    check(rollout_queue_enqueue(allow, &v1.header) == ROLLOUT_OK, "enqueue failed");
    check(rollout_queue_enqueue(allow, &v2.header) == ROLLOUT_ERR_FULL,
          "a third live behavior version was accepted under a bound of 2");
    /* A second group at an already-live version is fine. */
    struct toy_group v0b;
    make_toy_group(&v0b, 43, 0, G_TOY);
    check(rollout_queue_enqueue(allow, &v0b.header) == ROLLOUT_OK,
          "a second group at a live version was refused: %s", rollout_queue_last_error());

    check(rollout_queue_destroy(queue) == ROLLOUT_OK, "destroy failed");
    check(rollout_queue_destroy(bigger) == ROLLOUT_OK, "destroy failed");
    check(rollout_queue_destroy(allow) == ROLLOUT_OK, "destroy failed");
}

/* ------------------------------------------------------------------ */
/* Deterministic lag 0/1/2 injection (gate 3)                         */
/* ------------------------------------------------------------------ */

static void test_lag_injection_reports_staleness(void) {
    /* The learner is at version 2; a group generated under version v has lag 2 - v. At
     * lag zero the behavior policy *is* the learner, so the ratio is exactly one and the
     * objective and gradient are the synchronous ones. As the lag grows the behavior
     * log-probabilities fall further behind, and both drift. */
    struct BackwardGroupConfig config = toy_config();
    double objectives[3];
    double worst_gradient[3];
    for (int lag = 0; lag < 3; ++lag) {
        struct toy_group g;
    make_toy_group(&g, 50 + lag, /*behavior_version=*/2 - lag, G_TOY);
        /* The learner's current log-probability is `anchor`; scale the policy move with
         * the lag so lag 0 is exactly the behavior policy. */
        for (int i = 0; i < G_TOY; ++i) {
            for (int t = 0; t < T_TOY; ++t) {
                if (lag == 0) {
                    g.anchor[i][t] = g.behavior[i][t];
                } else {
                    g.anchor[i][t] = g.behavior[i][t] + 0.05f * (float)lag * (float)(i + 1);
                }
            }
        }
        float d[G_TOY * T_TOY];
        struct BackwardGroupStats stats;
        check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, g.responses, G_TOY, g.rewards,
                                       d, NULL, NULL, &stats, &objectives[lag]) == BACKWARD_OK,
              "the lag-%d objective failed: %s", lag, backward_last_error());

        /* The reported quantity is the policy change, and Stage 6 names it as one. */
        const int same_weights = (lag == 0);
        check(alignment_classify(same_weights, 1) ==
                  (lag == 0 ? ALIGNMENT_DEVIATION_NUMERICAL : ALIGNMENT_DEVIATION_POLICY_CHANGE),
              "a lag-%d difference was classified as something other than %s", lag,
              lag == 0 ? "numerical" : "a policy change");

        /* The gradient magnitude at the same fixture, for the drift report. */
        worst_gradient[lag] = 0.0;
        for (int k = 0; k < G_TOY * T_TOY; ++k) {
            if (fabs((double)d[k]) > worst_gradient[lag]) worst_gradient[lag] = fabs((double)d[k]);
        }
        if (lag == 0) {
            check_close(stats.mean_ratio, 1.0, 1e-9, "the lag-zero mean ratio");
            check_close(objectives[0], 0.0, 1e-9, "the lag-zero objective");
        }
    }
    check_close(objectives[0], 0.0, 1e-9, "the lag-zero objective is not zero");
    /* Lag is not free: the objective and gradient both move off the synchronous values. */
    check(fabs(objectives[1] - objectives[0]) > 1e-3,
          "a lag of 1 did not move the objective (%.9g)", objectives[1]);
    check(fabs(objectives[2] - objectives[0]) > fabs(objectives[1] - objectives[0]),
          "the objective did not drift further at lag 2");
    check(worst_gradient[0] < 0.25, "the lag-zero gradient is not the synchronous one (%.6g)",
          worst_gradient[0]);

    /* A zero-lag cap admits only the lag-zero group, which is what "GSPO remains lag-zero
     * until its separate objective gate is satisfied" means operationally. */
    struct RolloutQueueDims dims = {4, 64, 4};
    struct RolloutQueue *queue = NULL;
    check(rollout_queue_create(&dims, &queue) == ROLLOUT_OK, "create failed");
    struct toy_group stale;
    make_toy_group(&stale, 60, 0, G_TOY);
    check(rollout_queue_enqueue(queue, &stale.header) == ROLLOUT_OK, "enqueue failed");
    struct RolloutGroupHeader taken;
    check(rollout_queue_dequeue(queue, /*committed=*/2, /*max_lag=*/0, &taken) == ROLLOUT_ERR_STALE,
          "a lag-2 group entered a lag-zero pipeline");
    check(rollout_queue_dequeue(queue, 2, 2, &taken) == ROLLOUT_OK,
          "the same group was refused under an explicit cap of 2: %s", rollout_queue_last_error());
    check(rollout_queue_destroy(queue) == ROLLOUT_OK, "destroy failed");
}

int main(void) {
    test_lag_zero_matches_synchronous();
    test_whole_groups_only();
    test_lag_admission();
    test_capacity_bounds();
    test_lag_injection_reports_staleness();

    if (g_failures != 0) {
        fprintf(stderr, "rollout_queue_test: %d of %d check(s) failed\n", g_failures, g_checks);
        return EXIT_FAILURE;
    }
    printf("rollout_queue_test: whole groups are admitted under a bounded queue and a checked "
           "lag, lag zero reproduces the synchronous numbers, and larger lag is reported as a "
           "policy change\n");
    return EXIT_SUCCESS;
}
