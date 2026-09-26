/*
 * gspo_test.c - CPU gate for the GSPO/GRPO group objective (plan Stage 7).
 *
 * No GPU and no weights. The plan's Stage-7 gate asks for "an independent loss/autograd
 * fixture [that] matches objective and gradient for unequal response lengths, both
 * advantage signs, clip boundaries and masks; compare unclipped analytic gradients; test
 * zero-variance and truncated groups", and for sequence and token clipping statistics to
 * be reported separately. This gate is that fixture.
 *
 * The reference is written independently in FP64 inside this file and the analytic
 * gradient the implementation returns is checked against a central difference of *that*
 * reference, so an implementation and a reference that share a formula cannot certify
 * each other (the trap the Stage-4 clipped objective fell into once, and this stage's
 * min-form fix is the same corner). The fixture is deliberately unequal-length: GSPO
 * weights each response equally and GRPO each token, so the two objectives on the same
 * group are different numbers, which is what "compare GSPO against GRPO on the same
 * fixed rollout groups before comparing online rewards" needs to be checkable at all.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_gspo.
 */
#include "backward.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_G 4
#define MAX_TOK 8
#define FLAT_MAX (MAX_G * MAX_TOK)

static int g_failures = 0;
static int g_checks = 0;

static void check(int condition, const char *fmt, ...) {
    ++g_checks;
    if (condition) return;
    ++g_failures;
    va_list ap;
    va_start(ap, fmt);
    fputs("gspo_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void check_close(double got, double want, double tol, const char *what) {
    check(fabs(got - want) <= tol, "%s: got %.9g want %.9g (tol %.3g)", what, got, want, tol);
}

/* ------------------------------------------------------------------ */
/* The fixture                                                        */
/* ------------------------------------------------------------------ */

struct fixture {
    int g;
    int tokens[MAX_G];
    int admitted[MAX_G];
    float rewards[MAX_G];
    float logp[MAX_G][MAX_TOK];
    float behavior[MAX_G][MAX_TOK];
    uint8_t mask[MAX_G][MAX_TOK]; /* 1 = selected; a fixture may zero a row */
    float clip_low, clip_high, eps_adv;
    int allow_zero_variance;
};

/* The unequal-length group used for the value, gradient and mode-comparison checks.
 * log ratios are [0.6,0.6,0.0], [-0.8,-0.8], [0.1], [0.0,0.0] and the rewards are
 * 1,0,1,0, so the advantages are +1,-1,+1,-1 and responses 0 and 1 fall *outside* the
 * +/-0.2 band on opposite sides (one clipped by the upper clip, one by the lower). */
static struct fixture base_fixture(void) {
    struct fixture f;
    memset(&f, 0, sizeof(f));
    f.g = 4;
    f.tokens[0] = 3; f.tokens[1] = 2; f.tokens[2] = 1; f.tokens[3] = 2;
    for (int i = 0; i < f.g; ++i) {
        f.admitted[i] = 1;
        f.rewards[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        for (int t = 0; t < f.tokens[i]; ++t) {
            f.behavior[i][t] = -1.0f;
            f.mask[i][t] = 1;
        }
    }
    f.logp[0][0] = -0.4f; f.logp[0][1] = -0.4f; f.logp[0][2] = -1.0f;
    f.logp[1][0] = -1.8f; f.logp[1][1] = -1.8f;
    f.logp[2][0] = -0.9f;
    f.logp[3][0] = -1.0f; f.logp[3][1] = -1.0f;
    f.clip_low = 0.2f; f.clip_high = 0.2f; f.eps_adv = 1e-6f;
    return f;
}

static long long fixture_selected(const struct fixture *f, int i) {
    long long n = 0;
    for (int t = 0; t < f->tokens[i]; ++t) {
        if (f->mask[i][t] != 0) ++n;
    }
    return n;
}

static int fixture_admitted(const struct fixture *f) {
    int n = 0;
    for (int i = 0; i < f->g; ++i) {
        if (f->admitted[i]) ++n;
    }
    return n;
}

/* The d_logp layout the implementation uses: response i starts at sum_{j<i} T_j over the
 * admitted responses. */
static long long fixture_offset(const struct fixture *f, int i) {
    long long off = 0;
    for (int j = 0; j < i; ++j) {
        if (f->admitted[j]) off += fixture_selected(f, j);
    }
    return off;
}

static long long fixture_flat_size(const struct fixture *f) {
    long long n = 0;
    for (int i = 0; i < f->g; ++i) {
        if (f->admitted[i]) n += fixture_selected(f, i);
    }
    return n;
}

static void fixture_advantages(const struct fixture *f, double *out_a, double *out_mean,
                              double *out_std) {
    const int admitted = fixture_admitted(f);
    double sum = 0.0;
    for (int i = 0; i < f->g; ++i) {
        if (f->admitted[i]) sum += f->rewards[i];
    }
    const double mean = sum / (double)admitted;
    double var = 0.0;
    for (int i = 0; i < f->g; ++i) {
        if (!f->admitted[i]) continue;
        const double d = (double)f->rewards[i] - mean;
        var += d * d;
    }
    var /= (double)admitted;
    const double std = sqrt(var);
    for (int i = 0; i < f->g; ++i) {
        /* The advantages are already centred; the normalisation epsilon is what makes a
         * zero-variance group a zero advantage rather than 0/0. */
        out_a[i] = f->admitted[i] ? ((double)f->rewards[i] - mean) / (std + f->eps_adv) : 0.0;
    }
    *out_mean = mean;
    *out_std = std;
}

/* The independent FP64 objective. Reads f->logp, so a finite difference can drive it. */
static double ref_group_objective(const struct fixture *f, int mode) {
    if (fixture_admitted(f) < 2) return NAN;
    double a[MAX_G], mean, std;
    fixture_advantages(f, a, &mean, &std);
    if (std == 0.0 && !f->allow_zero_variance) return NAN;

    long long total = 0;
    for (int i = 0; i < f->g; ++i) {
        if (f->admitted[i]) total += fixture_selected(f, i);
    }
    if (total == 0) return NAN;

    double J = 0.0;
    for (int i = 0; i < f->g; ++i) {
        if (!f->admitted[i]) continue;
        const long long T = fixture_selected(f, i);
        double log_s = 0.0;
        for (int t = 0; t < f->tokens[i]; ++t) {
            if (f->mask[i][t] == 0) continue;
            log_s += (double)f->logp[i][t] - (double)f->behavior[i][t];
        }
        log_s /= (double)T;
        const double s = exp(log_s);
        double response_hinge = 0.0;
        for (int t = 0; t < f->tokens[i]; ++t) {
            if (f->mask[i][t] == 0) continue;
            const double log_ratio = (double)f->logp[i][t] - (double)f->behavior[i][t];
            const double r = mode == BACKWARD_GROUP_GSPO ? s : exp(log_ratio);
            const double unclipped = r * a[i];
            const double clamped =
                fmin(fmax(r, 1.0 - (double)f->clip_low), 1.0 + (double)f->clip_high);
            /* GSPO is one term per response (mean_i); GRPO is one term per token. */
            if (mode == BACKWARD_GROUP_GSPO) {
                response_hinge = fmin(unclipped, clamped * a[i]);
            } else {
                J += fmin(unclipped, clamped * a[i]) / (double)total;
            }
        }
        if (mode == BACKWARD_GROUP_GSPO) {
            J += response_hinge / (double)fixture_admitted(f);
        }
    }
    return J;
}

/* Build the ctypes-side array of responses from the fixture. */
static void fixture_responses(const struct fixture *f, struct BackwardResponse *out) {
    for (int i = 0; i < f->g; ++i) {
        out[i].logp = f->logp[i];
        out[i].behavior_logp = f->behavior[i];
        out[i].response_mask = f->mask[i];
        out[i].tokens = f->tokens[i];
        out[i].admitted = f->admitted[i];
    }
}

static struct BackwardGroupConfig fixture_config(const struct fixture *f) {
    struct BackwardGroupConfig c;
    c.clip_low = f->clip_low;
    c.clip_high = f->clip_high;
    c.eps_adv = f->eps_adv;
    c.allow_zero_variance = f->allow_zero_variance;
    return c;
}

/* ------------------------------------------------------------------ */
/* Objective value                                                    */
/* ------------------------------------------------------------------ */

static void test_value_matches_the_reference(void) {
    struct fixture f = base_fixture();
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);

    for (int mode = 0; mode < 2; ++mode) {
        float d[FLAT_MAX];
        float ratio[MAX_G];
        float advantages[MAX_G];
        struct BackwardGroupStats stats;
        double objective = 0.0;
        check(backward_group_objective(&config, (BackwardGroupMode)mode, responses, f.g,
                                       f.rewards, d, ratio, advantages, &stats, &objective) ==
                  BACKWARD_OK,
              "%s: the objective failed: %s", mode ? "GRPO" : "GSPO", backward_last_error());
        check_close(objective, ref_group_objective(&f, mode), 1e-6,
                    mode ? "GRPO objective" : "GSPO objective");

        /* The fixture puts response 0 (A=+1, s=1.4918) above the upper clip and
         * response 1 (A=-1, s=0.4493) below the lower one. GSPO clips whole responses. */
        if (mode == BACKWARD_GROUP_GSPO) {
            check(stats.admitted == 4 && stats.trainable_tokens == 8,
                  "GSPO admitted %d responses over %lld tokens", stats.admitted,
                  stats.trainable_tokens);
            check(stats.clipped_sequences == 2, "GSPO clipped %lld sequences, expected 2",
                  stats.clipped_sequences);
            check(stats.clipped_tokens == 5, "GSPO clipped %lld tokens, expected 5 (3+2)",
                  stats.clipped_tokens);
            check_close(stats.mean_ratio, (exp(0.4) + exp(-0.8) + exp(0.1) + 1.0) / 4.0, 1e-5,
                        "GSPO mean ratio");
            check_close(stats.max_abs_log_ratio, 0.8, 1e-5, "GSPO worst sequence log ratio");
            check_close(stats.max_abs_token_log_ratio, 0.8, 1e-5, "GSPO worst token log ratio");
            check_close(stats.advantage_std, 0.5, 1e-6, "GSPO advantage population std");
            check_close(ratio[0], exp(0.4), 1e-5, "response 0's s_i");
            check_close(ratio[1], exp(-0.8), 1e-5, "response 1's s_i");
        } else {
            /* GRPO clips per token: responses 0 and 1 have 2 clipped tokens each. */
            check(stats.clipped_tokens == 4, "GRPO clipped %lld tokens, expected 4",
                  stats.clipped_tokens);
            check(stats.clipped_sequences == 2, "GRPO touched %lld sequences",
                  stats.clipped_sequences);
        }
        /* Equal-response weighting and equal-token weighting are different objectives on
         * an unequal-length group, which is the whole point of comparing them. */
        if (mode == 1) {
            check(fabs(objective - ref_group_objective(&f, BACKWARD_GROUP_GSPO)) > 1e-4,
                  "GSPO and GRPO gave the same objective on an unequal-length group");
        }
    }
}

/* ------------------------------------------------------------------ */
/* Gradient: independent finite difference                            */
/* ------------------------------------------------------------------ */

static void test_gradient_matches_the_reference(void) {
    struct fixture f = base_fixture();
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    const double h = 1e-4;


    for (int mode = 0; mode < 2; ++mode) {
        float d[FLAT_MAX];
        double objective = 0.0;
        check(backward_group_objective(&config, (BackwardGroupMode)mode, responses, f.g,
                                       f.rewards, d, NULL, NULL, NULL, &objective) == BACKWARD_OK,
              "%s gradient failed: %s", mode ? "GRPO" : "GSPO", backward_last_error());
        check(backward_group_objective(&config, (BackwardGroupMode)mode, responses, f.g,
                                       f.rewards, NULL, NULL, NULL, NULL, NULL) == BACKWARD_OK,
              "%s objective-only failed", mode ? "GRPO" : "GSPO");

        double worst = 0.0;
        for (int i = 0; i < f.g; ++i) {
            if (!f.admitted[i]) continue;
            for (int t = 0; t < f.tokens[i]; ++t) {
                if (f.mask[i][t] == 0) continue;
                const float saved = f.logp[i][t];
                f.logp[i][t] = saved + (float)h;
                const double plus = ref_group_objective(&f, mode);
                f.logp[i][t] = saved - (float)h;
                const double minus = ref_group_objective(&f, mode);
                f.logp[i][t] = saved;
                const double numeric = (plus - minus) / (2.0 * h);
                const double got = (double)d[fixture_offset(&f, i) + t];
                const double tol = fmax(2e-5, fabs(numeric) * 2e-3);
                if (fabs(numeric - got) > tol) {
                    check(0, "%s analytic gradient response %d token %d: got %.9g numeric %.9g "
                             "(plus %.12g minus %.12g T=%lld)",
                          mode ? "GRPO" : "GSPO", i, t, got, numeric, plus, minus,
                          fixture_selected(&f, i));
                }
                worst = fmax(worst, fabs(numeric - got));
            }
        }
        check(worst < 1e-4, "%s gradient worst error %.3e", mode ? "GRPO" : "GSPO", worst);
    }
}

/* The unclipped branch's closed form, checked explicitly rather than only through the
 * finite difference: GSPO's is A_i*s_i/T_i (s_i not detached) and GRPO's is A_i*r_it. */
static void test_unclipped_closed_form(void) {
    struct fixture f = base_fixture();
    /* Everything inside the band: log ratios 0.05 and -0.05, T=2 each. */
    f.g = 2;
    f.tokens[0] = 2; f.tokens[1] = 2;
    f.admitted[0] = 1; f.admitted[1] = 1;
    f.rewards[0] = 1.0f; f.rewards[1] = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int t = 0; t < 2; ++t) {
            f.behavior[i][t] = -1.0f;
            f.mask[i][t] = 1;
            f.logp[i][t] = -1.0f + ((i == 0) ? 0.05f : -0.05f);
        }
    }
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    double a[MAX_G], mean, std;
    fixture_advantages(&f, a, &mean, &std);

    float d[FLAT_MAX];
    double objective = 0.0;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, d,
                                   NULL, NULL, NULL, &objective) == BACKWARD_OK,
          "the unclipped GSPO case failed: %s", backward_last_error());
    for (int i = 0; i < 2; ++i) {
        const double s = exp((i == 0) ? 0.05 : -0.05);
        const double want = (1.0 / 2.0) * a[i] * s / 2.0;
        check_close((double)d[i * 2 + 0], want, 1e-6, "unclipped GSPO gradient (token 0)");
        check_close((double)d[i * 2 + 1], want, 1e-6, "unclipped GSPO gradient (token 1)");
    }

    check(backward_group_objective(&config, BACKWARD_GROUP_GRPO, responses, f.g, f.rewards, d,
                                   NULL, NULL, NULL, &objective) == BACKWARD_OK,
          "the unclipped GRPO case failed: %s", backward_last_error());
    for (int i = 0; i < 2; ++i) {
        const double r = exp((i == 0) ? 0.05 : -0.05);
        const double want = (1.0 / 4.0) * a[i] * r;
        check_close((double)d[i * 2 + 0], want, 1e-6, "unclipped GRPO gradient (token 0)");
        check_close((double)d[i * 2 + 1], want, 1e-6, "unclipped GRPO gradient (token 1)");
    }
}

/* ------------------------------------------------------------------ */
/* The plan's named cases                                             */
/* ------------------------------------------------------------------ */

static void test_at_behavior_s1_is_one(void) {
    struct fixture f = base_fixture();
    /* pi_theta = pi_b: every log ratio is zero, so s_i = 1 and the hinge is A_i. */
    for (int i = 0; i < f.g; ++i) {
        for (int t = 0; t < f.tokens[i]; ++t) f.logp[i][t] = f.behavior[i][t];
    }
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    float d[FLAT_MAX];
    float ratio[MAX_G];
    double objective = 0.0;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, d,
                                   ratio, NULL, NULL, &objective) == BACKWARD_OK,
          "the pi_theta = pi_b case failed: %s", backward_last_error());
    for (int i = 0; i < f.g; ++i) check_close(ratio[i], 1.0, 1e-9, "s_i at pi_theta = pi_b");
    /* The advantages sum to zero, so the objective is zero and every gradient is
     * A_i * 1 / T_i / G. */
    check_close(objective, 0.0, 1e-6, "the objective at pi_theta = pi_b");
    double a[MAX_G], mean, std;
    fixture_advantages(&f, a, &mean, &std);
    for (int i = 0; i < f.g; ++i) {
        const double want = a[i] / (double)f.tokens[i] / 4.0;
        for (int t = 0; t < f.tokens[i]; ++t) {
            check_close((double)d[fixture_offset(&f, i) + t], want, 1e-6,
                        "the gradient at pi_theta = pi_b");
        }
    }
}

static void test_masks(void) {
    struct fixture f = base_fixture();
    /* Mask out response 0's third token, which carries log ratio 0.0: T_0 becomes 2 and
     * s_0 becomes exp(0.6), still outside the upper clip, so the clip statistic alone
     * cannot reveal a mask that was ignored - the token count must change too. */
    f.mask[0][2] = 0;
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    struct BackwardGroupStats stats;
    double objective = 0.0;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL,
                                   NULL, NULL, &stats, &objective) == BACKWARD_OK,
          "the masked case failed: %s", backward_last_error());
    check(stats.trainable_tokens == 7, "the mask did not remove a token (%lld)",
          stats.trainable_tokens);
    check_close(objective, ref_group_objective(&f, BACKWARD_GROUP_GSPO), 1e-6, "masked objective");

    /* The gradient layout packs only the selected rows: response 0 has T_0 = 2, so it
     * occupies flat[0..1], response 1 flat[2..3], response 2 flat[4] and response 3
     * flat[5..6]. Responses 0 and 1 are clipped (s_0 = 1.8221 above the upper clip with
     * A=+1, s_1 = 0.4493 below the lower with A=-1), so only responses 2 and 3 move. */
    float d[FLAT_MAX];
    const float sentinel = -12345.0f;
    for (int k = 0; k < FLAT_MAX; ++k) d[k] = sentinel;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, d,
                                   NULL, NULL, NULL, NULL) == BACKWARD_OK,
          "the masked gradient failed");
    check(d[0] == 0.0f && d[1] == 0.0f && d[2] == 0.0f && d[3] == 0.0f,
          "a clipped response was given a nonzero gradient");
    check_close((double)d[4], (1.0 / 4.0) * 1.0 * exp(0.1) / 1.0, 1e-6,
                "the masked layout's response 2 gradient");
    check_close((double)d[5], (1.0 / 4.0) * -1.0 * 1.0 / 2.0, 1e-6,
                "the masked layout's response 3 gradient");
    check_close((double)d[6], (double)d[5], 1e-6, "response 3's two tokens are not equal");
    check(d[7] == sentinel, "the flat layout wrote past its packed size");
}

static void test_zero_variance(void) {
    struct fixture f = base_fixture();
    for (int i = 0; i < f.g; ++i) f.rewards[i] = 1.0f;
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    double objective = 0.0;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL,
                                   NULL, NULL, NULL, &objective) == BACKWARD_ERR_STATE,
          "a zero-variance group was accepted by default");

    /* GSPO's stated rule is a zero advantage, which the caller admits explicitly. */
    struct fixture waivable = f;
    waivable.allow_zero_variance = 1;
    const struct BackwardGroupConfig waive_config = fixture_config(&waivable);
    float d[FLAT_MAX];
    float advantages[MAX_G];
    check(backward_group_objective(&waive_config, BACKWARD_GROUP_GSPO, responses, waivable.g,
                                   waivable.rewards, d, NULL, advantages, NULL, &objective) ==
              BACKWARD_OK,
          "the waived zero-variance group failed: %s", backward_last_error());
    check_close(objective, 0.0, 1e-9, "the waived zero-variance objective");
    for (int i = 0; i < waivable.g; ++i) {
        check(advantages[i] == 0.0f, "a zero-variance advantage is not zero (%.3e)",
              (double)advantages[i]);
    }
    for (long long k = 0; k < fixture_flat_size(&waivable); ++k) {
        check(d[k] == 0.0f, "a zero-variance gradient is nonzero (%.3e)", (double)d[k]);
    }
}

static void test_truncated_and_partial_groups(void) {
    struct fixture f = base_fixture();
    /* Response 1 hit the length limit: an explicit caller rule drops it. The group is
     * then normalised over the remaining three responses, not over a partial mixture. */
    f.admitted[1] = 0;
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    struct BackwardGroupStats stats;
    double objective = 0.0;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL,
                                   NULL, NULL, &stats, &objective) == BACKWARD_OK,
          "the truncated-group case failed: %s", backward_last_error());
    check(stats.admitted == 3, "the dropped response was counted (%d)", stats.admitted);
    check(stats.trainable_tokens == 6, "the dropped response's tokens were counted (%lld)",
          stats.trainable_tokens);
    check_close(objective, ref_group_objective(&f, BACKWARD_GROUP_GSPO), 1e-6,
                "the truncated-group objective");

    /* The flat layout follows the admitted responses: with response 1 dropped it has
     * T_0 + T_2 + T_3 = 3 + 1 + 2 = 6 slots, and the dropped response's rows are not
     * part of it. */
    float d[FLAT_MAX];
    const float sentinel = -12345.0f;
    for (int k = 0; k < FLAT_MAX; ++k) d[k] = sentinel;
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, d,
                                   NULL, NULL, NULL, NULL) == BACKWARD_OK,
          "the truncated-group gradient failed");
    check(fixture_flat_size(&f) == 6, "the flat size is %lld, expected 6",
          fixture_flat_size(&f));
    /* Response 0 is clipped, response 2 is inside the band, response 3 is inside it. */
    check(d[0] == 0.0f && d[1] == 0.0f && d[2] == 0.0f, "a clipped response moved");
    check(d[6] == sentinel, "the flat layout wrote past its packed size");

    /* A group that would be normalised over one response is refused, not partially
     * normalised. */
    struct fixture one = f;
    one.admitted[2] = 0;
    one.admitted[3] = 0;
    fixture_responses(&one, responses);
    const struct BackwardGroupConfig one_config = fixture_config(&one);
    check(backward_group_objective(&one_config, BACKWARD_GROUP_GSPO, responses, one.g, one.rewards,
                                   NULL, NULL, NULL, NULL, &objective) == BACKWARD_ERR_STATE,
          "a one-response group was partially normalised");

    /* An admitted but zero-length response is invalid, not a quiet zero. */
    struct fixture empty = base_fixture();
    for (int t = 0; t < empty.tokens[2]; ++t) empty.mask[2][t] = 0;
    fixture_responses(&empty, responses);
    const struct BackwardGroupConfig empty_config = fixture_config(&empty);
    check(backward_group_objective(&empty_config, BACKWARD_GROUP_GSPO, responses, empty.g,
                                   empty.rewards, NULL, NULL, NULL, NULL, &objective) ==
              BACKWARD_ERR_STATE,
          "a zero-length admitted response was accepted");
}

static void test_advantages_agree_with_the_reduction(void) {
    struct fixture f = base_fixture();
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    const struct BackwardGroupConfig config = fixture_config(&f);
    float advantages[MAX_G];
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL,
                                   NULL, advantages, NULL, NULL) == BACKWARD_OK,
          "the advantage cross-check failed: %s", backward_last_error());

    /* The same group through the standalone reduction: the objective's advantages must
     * be the reduction's, so the two cannot drift. */
    float standalone[MAX_G];
    double mean = 0.0, std = 0.0;
    check(backward_group_advantage(f.rewards, NULL, f.g, f.eps_adv, 0, standalone, &mean, &std) ==
              BACKWARD_OK,
          "the standalone advantage reduction failed");
    for (int i = 0; i < f.g; ++i) {
        check_close((double)advantages[i], (double)standalone[i], 1e-6,
                    "the objective's advantage against the reduction's");
    }
    check_close(mean, 0.5, 1e-9, "the group mean");
    check_close(std, 0.5, 1e-9, "the group std");
}

static void test_refusals(void) {
    struct fixture f = base_fixture();
    struct BackwardResponse responses[MAX_G];
    fixture_responses(&f, responses);
    double objective = 0.0;

    struct BackwardGroupConfig bad = fixture_config(&f);
    bad.clip_low = -0.1f;
    check(backward_group_objective(&bad, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL, NULL,
                                   NULL, NULL, &objective) == BACKWARD_ERR_RANGE,
          "a negative clip was accepted");
    bad = fixture_config(&f);
    bad.eps_adv = NAN;
    check(backward_group_objective(&bad, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL, NULL,
                                   NULL, NULL, &objective) == BACKWARD_ERR_RANGE,
          "a non-finite advantage epsilon was accepted");

    const struct BackwardGroupConfig config = fixture_config(&f);
    check(backward_group_objective(&config, (BackwardGroupMode)9, responses, f.g, f.rewards, NULL,
                                   NULL, NULL, NULL, &objective) == BACKWARD_ERR_ARG,
          "an unknown group mode was accepted");
    check(backward_group_objective(&config, BACKWARD_GROUP_GSPO, responses, 1, f.rewards, NULL,
                                   NULL, NULL, NULL, &objective) == BACKWARD_ERR_ARG,
          "a one-response group was accepted");
    check(backward_group_objective(NULL, BACKWARD_GROUP_GSPO, responses, f.g, f.rewards, NULL, NULL,
                                   NULL, NULL, &objective) == BACKWARD_ERR_ARG,
          "a null config was accepted");

    struct fixture nan_reward = f;
    nan_reward.rewards[0] = NAN;
    fixture_responses(&nan_reward, responses);
    const struct BackwardGroupConfig nan_config = fixture_config(&nan_reward);
    check(backward_group_objective(&nan_config, BACKWARD_GROUP_GSPO, responses, nan_reward.g,
                                   nan_reward.rewards, NULL, NULL, NULL, NULL, &objective) ==
              BACKWARD_ERR_DIVERGED,
          "a non-finite reward was accepted");

    struct fixture nan_logp = f;
    nan_logp.logp[0][0] = NAN;
    fixture_responses(&nan_logp, responses);
    const struct BackwardGroupConfig nan_logp_config = fixture_config(&nan_logp);
    check(backward_group_objective(&nan_logp_config, BACKWARD_GROUP_GSPO, responses, nan_logp.g,
                                   nan_logp.rewards, NULL, NULL, NULL, NULL, &objective) ==
              BACKWARD_ERR_DIVERGED,
          "a non-finite log-probability was accepted");
}

int main(void) {
    test_value_matches_the_reference();
    test_gradient_matches_the_reference();
    test_unclipped_closed_form();
    test_at_behavior_s1_is_one();
    test_masks();
    test_zero_variance();
    test_truncated_and_partial_groups();
    test_advantages_agree_with_the_reduction();
    test_refusals();

    if (g_failures != 0) {
        fprintf(stderr, "gspo_test: %d of %d check(s) failed\n", g_failures, g_checks);
        return EXIT_FAILURE;
    }
    printf("gspo_test: the GSPO and GRPO objectives match an independent FP64 reference and its "
           "finite difference, with sequence and token clipping reported separately\n");
    return EXIT_SUCCESS;
}
