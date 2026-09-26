/*
 * train_loop_test.c - CPU gate for the synchronous training/rollout baseline's phases,
 * budgets, records and sampler (plan Stage 5).
 *
 * No GPU and no weights. It checks the rules the plan asks for, each of them as a
 * behaviour rather than a note: a phase's budget is what distinguishes it, a phase
 * boundary resets the sequence, a record cannot be read under a version it was not
 * generated under, the ratio is exactly one at unchanged parameters, and the sampler's
 * empirical distribution matches the softmax it samples from to within the finite-sample
 * band. The objective/advantage side of the plan's gate (masks, negative and positive
 * advantages, zero-variance groups) is exercised here at the rollout level on a recorded
 * group, which is where those numbers actually come from.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_train_loop.
 */
#include "backward.h"
#include "train_loop.h"

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
    fputs("train_loop_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void check_close(double got, double want, double tol, const char *what) {
    check(fabs(got - want) <= tol, "%s: got %.9g want %.9g (tol %.3g)", what, got, want, tol);
}

static struct TrainLoopDims tiny_dims(void) {
    struct TrainLoopDims dims;
    memset(&dims, 0, sizeof(dims));
    dims.hidden_size = 128;
    dims.intermediate_size = 256;
    dims.num_layers = 2;
    dims.num_heads = 4;
    dims.num_kv_heads = 2;
    dims.head_dim = 128;
    dims.vocab_size = 1024;
    dims.max_seq_len = 512;
    dims.gdn_conv_dim = 8192;
    dims.gdn_conv_kernel = 4;
    dims.gdn_num_v_heads = 32;
    dims.gdn_head_dim = 128;
    return dims;
}

/* ------------------------------------------------------------------ */
/* Budgets                                                            */
/* ------------------------------------------------------------------ */

static void test_budgets(void) {
    const struct TrainLoopDims dims = tiny_dims();
    struct TrainPhaseBudget sft, rollout;
    check(train_loop_phase_budget(TRAIN_PHASE_SFT, &dims, 8, 700000, 4096, &sft) == TRAIN_OK,
          "the SFT budget failed: %s", train_loop_last_error());
    check(train_loop_phase_budget(TRAIN_PHASE_ROLLOUT, &dims, 8, 700000, 4096, &rollout) ==
              TRAIN_OK,
          "the rollout budget failed: %s", train_loop_last_error());

    /* The phases share the weights and the sequence state, and differ in exactly the
     * three things a rollout does not keep. */
    check(sft.weights_bytes == rollout.weights_bytes && sft.weights_bytes > 0,
          "the two phases disagree about the weights (%lld vs %lld)", sft.weights_bytes,
          rollout.weights_bytes);
    check(sft.kv_cache_bytes == rollout.kv_cache_bytes && sft.kv_cache_bytes > 0,
          "the two phases disagree about the KV cache");
    check(sft.gdn_state_bytes == rollout.gdn_state_bytes && sft.gdn_state_bytes > 0,
          "the two phases disagree about the GDN state");
    check(sft.optimizer_bytes > 0 && rollout.optimizer_bytes == 0,
          "only SFT holds optimizer state (%lld vs %lld)", sft.optimizer_bytes,
          rollout.optimizer_bytes);
    check(sft.activations_bytes > 0 && rollout.activations_bytes == 0,
          "only SFT holds activations for a backward");
    check(sft.retained_bytes > 0 && rollout.retained_bytes == 0,
          "only SFT retains the step's values");
    /* Four FP32 slots per trainable element: master, gradient and two moments. */
    check(sft.optimizer_bytes == 700000LL * 4 * 4, "the optimizer budget is %lld, expected %lld",
          sft.optimizer_bytes, 700000LL * 4 * 4);
    check(sft.retained_bytes == 4096LL * 4, "the retained budget is %lld, expected %lld",
          sft.retained_bytes, 4096LL * 4);
    check(sft.total_bytes == sft.weights_bytes + sft.kv_cache_bytes + sft.gdn_state_bytes +
                                  sft.activations_bytes + sft.retained_bytes +
                                  sft.optimizer_bytes,
          "the SFT total does not add up");
    check(rollout.total_bytes < sft.total_bytes,
          "a rollout must cost less than an SFT step (%lld vs %lld)", rollout.total_bytes,
          sft.total_bytes);

    check(train_loop_phase_budget(TRAIN_PHASE_SFT, NULL, 8, 1, 1, &sft) == TRAIN_ERR_ARG,
          "null dims were accepted");
    check(train_loop_phase_budget(TRAIN_PHASE_SFT, &dims, 0, 1, 1, &sft) == TRAIN_ERR_ARG,
          "zero tokens were accepted");
    check(train_loop_phase_budget(TRAIN_PHASE_SFT, &dims, 8, -1, 1, &sft) == TRAIN_ERR_ARG,
          "a negative element count was accepted");
    check(train_loop_phase_budget((TrainPhase)7, &dims, 8, 1, 1, &sft) == TRAIN_ERR_ARG,
          "an unknown phase was accepted");
}

/* A store with two trainable parameters, so the records and the version rule have a
 * version to bind to. */
static TrainStore *make_store(void) {
    static const char *roles[] = {"embed", "lmHead"};
    static const char *templates[] = {"model.embed_tokens.weight", "model.lm_head.weight"};
    struct TrainParamSpec specs[2];
    memset(specs, 0, sizeof(specs));
    specs[0].layer = -1;
    specs[0].role = 0;
    specs[0].elements = 16;
    specs[0].trainable = 1;
    specs[1].layer = -1;
    specs[1].role = 1;
    specs[1].elements = 16;
    specs[1].trainable = 1;
    return train_store_create(specs, templates, roles, 2);
}

/* ------------------------------------------------------------------ */
/* The phase machine                                                  */
/* ------------------------------------------------------------------ */

static void test_phases(void) {
    TrainStore *store = make_store();
    check(store != NULL, "the store could not be created: %s", train_last_error());
    if (store == NULL) return;
    TrainLoop *loop = train_loop_create(store);
    check(loop != NULL, "the loop could not be created: %s", train_loop_last_error());

    long long version = -1;
    check(train_loop_enter(loop, TRAIN_PHASE_SFT, &version) == TRAIN_OK, "entering SFT failed");
    check(version == train_store_version(store), "the loop borrowed version %lld, the store is at %lld",
          version, train_store_version(store));
    check(train_loop_enter(loop, TRAIN_PHASE_ROLLOUT, &version) == TRAIN_ERR_STATE,
          "a second phase was entered without leaving the first");

    struct TrainSampleRecord record;
    memset(&record, 0, sizeof(record));
    record.version = version;
    record.policy_id = 1;
    record.tokens = 3;
    record.token_ids[0] = 5;
    record.token_ids[1] = 6;
    record.token_ids[2] = 7;
    record.logprobs[0] = -1.0f;
    record.logprobs[1] = -0.5f;
    record.logprobs[2] = -2.0f;
    record.sampled_logprobs[0] = record.logprobs[0];
    record.sampled_logprobs[1] = record.logprobs[1];
    record.sampled_logprobs[2] = record.logprobs[2];
    record.mask[0] = 0; /* the prompt */
    record.mask[1] = 1;
    record.mask[2] = 1;
    record.terminal = TRAIN_TERMINAL_EOS;
    record.reward = 1.0f;

    struct TrainGroup group;
    memset(&group, 0, sizeof(group));
    check(train_loop_record(loop, &group, &record) == TRAIN_OK, "recording failed: %s",
          train_loop_last_error());
    struct TrainSampleRecord stale = record;
    stale.version = version + 1;
    check(train_loop_record(loop, &group, &stale) == TRAIN_ERR_STATE,
          "a record from another version was accepted");

    /* The ratio is exactly one when the parameters have not moved: the same logprobs,
     * bitwise, so the mean log-ratio is zero and exp(0) is one. */
    double ratio = 0.0;
    check(train_loop_ratio(loop, &record, record.logprobs, &ratio) == TRAIN_OK,
          "the ratio failed: %s", train_loop_last_error());
    check(ratio == 1.0, "the ratio at unchanged parameters is %.17g, not exactly 1", ratio);
    float moved[3] = {-1.2f, -0.4f, -2.0f};
    check(train_loop_ratio(loop, &record, moved, &ratio) == TRAIN_OK, "the moved ratio failed");
    check(ratio > 1.0 && ratio < 2.0, "a moved logprob gave a ratio of %.6f", ratio);

    float reward = 0.0f;
    check(train_loop_read_reward(loop, &group, 0, &reward) == TRAIN_OK, "reading the reward failed");
    check(reward == 1.0f, "the reward did not round-trip");
    check(train_loop_read_reward(loop, &group, 1, &reward) == TRAIN_ERR_RANGE,
          "an out-of-range record index was accepted");

    check(train_loop_leave(loop) == TRAIN_OK, "leaving failed");
    check(train_loop_leave(loop) == TRAIN_ERR_STATE, "leaving twice was accepted");

    /* A rollout borrows the version, so an update must be refused while it is open; SFT
     * owns the window, so the same update succeeds. */
    check(train_loop_enter(loop, TRAIN_PHASE_ROLLOUT, &version) == TRAIN_OK, "entering the rollout failed");
    check(train_store_begin_update(store) == TRAIN_ERR_BUSY,
          "an update was accepted while a rollout borrowed the version");
    check(train_loop_leave(loop) == TRAIN_OK, "leaving the rollout failed");
    /* The optimizer state is offloaded during a rollout: the budget's optimizer bytes
     * are zero, and the loop reports that it holds none. */
    check(train_loop_optimizer_offloads(loop) == 0, "a closed loop reports offloaded state");

    /* A phase boundary resets the sequence state, once per real entry: the SFT entry and
     * the rollout entry are two (the refused double-entry is not a reset). */
    check(train_loop_sequence_resets(loop) == 2, "the sequence resets are %d, expected 2",
          train_loop_sequence_resets(loop));

    /* Publishing an update moves the version, and the old record becomes unreadable: the
     * plan's "never reconstruct an old denominator using updated weights". */
    check(train_store_begin_update(store) == TRAIN_OK, "opening the update window failed");
    check(train_store_publish(store) == TRAIN_OK, "publishing failed: %s", train_last_error());
    check(train_loop_read_reward(loop, &group, 0, &reward) == TRAIN_ERR_STATE,
          "a record from a superseded version was read");
    check(train_loop_ratio(loop, &record, record.logprobs, &ratio) == TRAIN_ERR_STATE,
          "a ratio was computed for a superseded record");

    train_loop_destroy(loop);
    check(train_store_destroy(store) == TRAIN_OK, "destroying the store failed");
}

/* ------------------------------------------------------------------ */
/* The rollout's objective side                                       */
/* ------------------------------------------------------------------ */

static void test_rollout_objective(void) {
    /* A group of four completions with rewards whose advantages are a mix of signs, one
     * zero-variance group, and a mask - the plan's synchronous-update gate items. */
    const float rewards[4] = {1.0f, 0.0f, 2.0f, 0.5f};
    float advantages[4];
    double mean = 0.0, std = 0.0;
    check(backward_group_advantage(rewards, NULL, 4, 1e-4f, 0, advantages, &mean, &std) ==
              BACKWARD_OK,
          "the group advantage failed: %s", backward_last_error());
    check(advantages[2] > 0.0f && advantages[1] < 0.0f,
          "the advantages did not separate the good and bad completions");
    check_close(advantages[0] + advantages[1] + advantages[2] + advantages[3], 0.0, 1e-6,
                "the advantages are not centered");

    /* The clipped objective over a recorded completion: the prompt is masked out, the
     * ratio at unchanged parameters is one, and the gradient's sign follows the
     * advantage. */
    const float logp[5] = {-0.5f, -0.7f, -0.9f, -1.1f, -1.3f};
    float old_logp[5];
    memcpy(old_logp, logp, sizeof(old_logp)); /* unchanged parameters */
    float advantage_row[5] = {0.0f, 0.0f, 1.0f, -1.0f, 0.5f};
    const uint8_t mask[5] = {0, 1, 1, 1, 1};
    float d_logp[5];
    float ratio_row[5];
    double objective = 0.0;
    long long selected = 0;
    /* TOKEN mode: each selected row's gradient follows its own advantage. (SEQUENCE mode
     * shares one ratio across the completion, so its gradient is the shared slope rather
     * than the row's own advantage - that is what a sequence-level objective is.) */
    check(backward_clipped_objective(logp, old_logp, advantage_row, mask, 5, 0.2f, 0.2f,
                                     BACKWARD_CLIP_TOKEN, d_logp, ratio_row, &objective,
                                     &selected) == BACKWARD_OK,
          "the clipped objective failed: %s", backward_last_error());
    check(selected == 4, "the masked objective selected %lld rows, expected 4", selected);
    check(ratio_row[2] > 0.999 && ratio_row[2] < 1.001,
          "the ratio at unchanged parameters is %.6f, expected 1", ratio_row[2]);
    check(d_logp[0] == 0.0f, "a masked row has a gradient");
    /* A positive advantage pushes the log-probability up (a negative loss gradient). */
    check(d_logp[2] < 0.0f && d_logp[3] > 0.0f,
          "the gradient's sign does not follow the advantage (%.6f, %.6f)", d_logp[2], d_logp[3]);
}

/* ------------------------------------------------------------------ */
/* The sampler                                                        */
/* ------------------------------------------------------------------ */

static void test_sampler(void) {
    /* A three-way distribution, so the frequency test has something to fail on. */
    const float logits[3] = {1.0f, 0.0f, -1.0f};
    double softmax[3];
    {
        double total = 0.0;
        for (int v = 0; v < 3; ++v) {
            softmax[v] = exp((double)logits[v] - 1.0);
            total += softmax[v];
        }
        for (int v = 0; v < 3; ++v) softmax[v] /= total;
    }
    struct BackwardRng rng;
    backward_rng_seed(&rng, 20260926);
    const int draws = 200000;
    double frequencies[3];
    check(train_loop_sample_frequencies(logits, 3, draws, &rng, frequencies) == TRAIN_OK,
          "the frequency test failed: %s", train_loop_last_error());
    printf("train_loop_test: sampler frequencies %.4f %.4f %.4f vs softmax %.4f %.4f %.4f\n",
           frequencies[0], frequencies[1], frequencies[2], softmax[0], softmax[1], softmax[2]);
    for (int v = 0; v < 3; ++v) {
        /* Three standard deviations of a binomial proportion: the band a finite sample
         * has to fall in, not an exact match. */
        const double sigma = sqrt(softmax[v] * (1.0 - softmax[v]) / draws);
        check(fabs(frequencies[v] - softmax[v]) < 3.0 * sigma,
              "token %d's frequency %.5f is outside 3 sigma (%.5f) of its softmax %.5f", v,
              frequencies[v], 3.0 * sigma, softmax[v]);
    }
    check(fabs(frequencies[0] - frequencies[1]) > 0.05,
          "the sampler did not separate the two most likely tokens");

    /* Seed behaviour: the same seed is the same draw sequence, a different seed is not,
     * and a draw is recoverable from the state alone. */
    struct BackwardRng a, b;
    backward_rng_seed(&a, 7);
    backward_rng_seed(&b, 7);
    int same = 1;
    for (int i = 0; i < 64; ++i) {
        const uint64_t left = backward_rng_next(&a);
        const uint64_t right = backward_rng_next(&b);
        if (left != right) same = 0;
    }
    check(same, "the same sampler seed produced a different draw sequence");
    backward_rng_seed(&a, 7);
    backward_rng_seed(&b, 8);
    check(backward_rng_next(&a) != backward_rng_next(&b), "two seeds produced the same first draw");

    /* The sampled log-probability is the model's own at temperature 1 with no
     * truncation, which is what lets the objective be said to optimize the model's
     * distribution; the gate would catch a transform that made them differ. */
    backward_rng_seed(&rng, 11);
    int token = -1;
    float model_logprob = 0.0f, sampled_logprob = 0.0f;
    check(train_loop_sample_fp64(logits, 3, &rng, &token, &model_logprob, &sampled_logprob) ==
              TRAIN_OK,
          "sampling failed: %s", train_loop_last_error());
    check(token >= 0 && token < 3, "the sampler returned token %d", token);
    check_close((double)model_logprob, log(softmax[token]), 1e-6,
                "the sampled log-probability is not the model's log-softmax");

    /* The plan's explicit validation: the host FP64 sampler's log-probability against
     * the trainer's FP32 one. The two are different functions in their last digits, so
     * the difference is *measured* here rather than assumed away; the trainer's
     * log-softmax is the FP32 formula kernels/logprob.cu uses. */
    {
        double total64 = 0.0;
        for (int v = 0; v < 3; ++v) total64 += exp((double)logits[v] - 1.0);
        const int probe = token;
        const float trainer_logprob =
            (float)((double)logits[probe] -
                    (1.0 + logf((float)total64))); /* an FP32 LSE over the same logits */
        const double gap = fabs((double)trainer_logprob - (double)model_logprob);
        printf("train_loop_test: FP64 sampler log-prob %.9g vs the FP32 trainer's %.9g "
               "(gap %.3e)\n", (double)model_logprob, (double)trainer_logprob, gap);
        check(gap < 1e-5, "the sampler/trainer log-probability gap is %.3e", gap);
        check((double)model_logprob == (double)sampled_logprob,
              "at temperature 1 with no truncation the sampler's log-probability must be the "
              "model's own");
    }

    /* EOS and truncation are distinct terminal reasons, and the mask is what tells a
     * completion's tail from its prompt: both travel in the record. */
    {
        struct TrainSampleRecord finished;
        memset(&finished, 0, sizeof(finished));
        finished.version = 0;
        finished.policy_id = 7;
        finished.tokens = 2;
        finished.token_ids[0] = 1;
        finished.token_ids[1] = 0; /* eos */
        finished.mask[0] = 1;
        finished.mask[1] = 1;
        finished.terminal = TRAIN_TERMINAL_EOS;
        struct TrainSampleRecord cut = finished;
        cut.terminal = TRAIN_TERMINAL_LENGTH;
        cut.mask[1] = 0; /* a truncated completion's last token is not trained on */
        check(finished.terminal != cut.terminal,
              "EOS and truncation are not distinguishable in the record");
        check(finished.policy_id != 0, "the record does not name the policy it came from");
        check(finished.mask[1] == 1 && cut.mask[1] == 0,
              "the mask does not distinguish an EOS completion from a truncated one");
    }

    /* A non-finite logit is a refusal, not a NaN token. */
    const float bad[3] = {0.0f, NAN, 0.0f};
    check(train_loop_sample_fp64(bad, 3, &rng, &token, NULL, NULL) == TRAIN_ERR_STATE,
          "a NaN logit was sampled from");
    check(train_loop_sample_fp64(logits, 0, &rng, &token, NULL, NULL) == TRAIN_ERR_ARG,
          "an empty vocabulary was sampled from");
}

int main(void) {
    test_budgets();
    test_phases();
    test_rollout_objective();
    test_sampler();

    if (g_failures != 0) {
        fprintf(stderr, "train_loop_test: %d of %d check(s) failed\n", g_failures, g_checks);
        return EXIT_FAILURE;
    }
    printf("train_loop_test: the phases, budgets, records, version binding and host sampler "
           "behave as the plan requires\n");
    return EXIT_SUCCESS;
}
