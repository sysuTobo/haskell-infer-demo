/*
 * alignment_test.c - CPU gate for the Stage-6 numerical-alignment decision
 * (plan Stage 6).
 *
 * No GPU and no weights. It checks that the Stage-6 verdicts are read back out of the
 * committed Stage-1 inventory rather than asserted beside it (every inventory region
 * resolves, the exception regions carry the Stage-2 bound as their declared bound, and
 * the still-unverified regions name the work that would remove them), and that the
 * reporting rule the gate turns on holds: a policy change or a sampler difference is
 * refused as a numerical comparison instead of being scored against a numerical bound,
 * and an exact-by-construction region has no tolerance to be checked against.
 *
 * Stage 6's gate is "region and end-to-end same-weight logprob comparisons across
 * admitted cases, plus gradient/update regression". The region/case *comparisons* are
 * Stage 2's device gates (test_gdn_invariance, test_attention_invariance,
 * test_gemm_invariance, test_attention_lse) and the gradient/update regression is
 * Stage 4's (test_backward, test_backward_kernels); this gate adds the decision those
 * measurements feed and the separation that stops them being relabelled.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_alignment.
 */
#include "alignment.h"
#include "regions.h"

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
    fputs("alignment_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* ------------------------------------------------------------------ */
/* The verdicts are the Stage-1 inventory, read back                  */
/* ------------------------------------------------------------------ */

static void test_verdicts_cover_the_inventory(void) {
    check(alignment_self_check() == ALIGNMENT_OK, "the self-check failed: %s",
          alignment_last_error());

    int count = 0;
    const struct RegionInventoryEntry *inventory = region_inventory(&count);
    check(inventory != NULL && count > 0, "the Stage-1 inventory is empty");

    int exact = 0, pending = 0, excepted = 0;
    for (int i = 0; i < count; ++i) {
        struct AlignmentVerdict verdict;
        check(alignment_verdict(inventory[i].region, &verdict) == ALIGNMENT_OK,
              "'%s' has no Stage-6 verdict: %s", inventory[i].region, alignment_last_error());
        switch (verdict.decision) {
            case ALIGNMENT_EXACT_BY_CONSTRUCTION: ++exact; break;
            case ALIGNMENT_INVARIANT_KERNEL_PENDING: ++pending; break;
            case ALIGNMENT_DECLARED_EXCEPTION: ++excepted; break;
        }
    }
    /* The buckets are asserted so a change in regions.c is visible here rather than
     * silently re-bucketing: the four exception regions, the four regions Stage 2 did
     * not establish, and the rest exact or not-applicable. */
    check(excepted == 4, "expected 4 declared exceptions, got %d", excepted);
    check(pending == 4, "expected 4 invariant-kernel-pending regions, got %d", pending);
    check(exact == count - 8, "expected %d exact-by-construction regions, got %d",
          count - 8, exact);

    /* A region the inventory does not have has no verdict: a Stage-6 claim about a
     * region that does not exist is a claim about nothing. */
    struct AlignmentVerdict bogus;
    check(alignment_verdict("not_a_region", &bogus) == ALIGNMENT_ERR_ARG,
          "a verdict was produced for a region outside the inventory");
}

static void test_known_verdicts(void) {
    struct AlignmentVerdict v;

    /* An exception carries the Stage-2 measurement as its declared bound, and the
     * declared bound is the widest of the region's pairs, never tighter. */
    check(alignment_verdict("attention_core", &v) == ALIGNMENT_OK, "attention_core: %s",
          alignment_last_error());
    check(v.decision == ALIGNMENT_DECLARED_EXCEPTION, "attention_core is %s",
          alignment_decision_name(v.decision));
    check(v.bound_max_abs >= 1.0e-03 && v.bound_max_abs < 2.0e-03,
          "attention_core's declared max_abs is %g, not the Stage-2 1.0e-03", v.bound_max_abs);
    check(v.bound_rms >= 1.0e-04, "attention_core's declared rms is %g, below the Stage-2 1.0e-04",
          v.bound_rms);

    check(alignment_verdict("gemm_bf16", &v) == ALIGNMENT_OK, "gemm_bf16: %s",
          alignment_last_error());
    check(v.decision == ALIGNMENT_DECLARED_EXCEPTION && v.bound_max_abs >= 1.3e-01,
          "gemm_bf16 is %s with bound %g (claim D's worst is 5.4e-3 relative, rounded to 1.3e-01 "
          "absolute)", alignment_decision_name(v.decision), v.bound_max_abs);

    check(alignment_verdict("gdn_core", &v) == ALIGNMENT_OK, "gdn_core: %s",
          alignment_last_error());
    check(v.decision == ALIGNMENT_DECLARED_EXCEPTION && v.bound_max_abs >= 4.3e-04,
          "gdn_core is %s with bound %g", alignment_decision_name(v.decision), v.bound_max_abs);

    /* An exact-by-construction region has no measurement to carry. */
    check(alignment_verdict("embedding", &v) == ALIGNMENT_OK, "embedding: %s",
          alignment_last_error());
    check(v.decision == ALIGNMENT_EXACT_BY_CONSTRUCTION, "embedding is %s",
          alignment_decision_name(v.decision));
    check(isnan(v.bound_max_abs), "an exact-by-construction region carries a bound (%g)",
          v.bound_max_abs);

    /* A region Stage 2 did not establish is pending, and it names the work. */
    check(alignment_verdict("rmsnorm", &v) == ALIGNMENT_OK, "rmsnorm: %s", alignment_last_error());
    check(v.decision == ALIGNMENT_INVARIANT_KERNEL_PENDING, "rmsnorm is %s",
          alignment_decision_name(v.decision));
    check(v.tracked_work != NULL && v.tracked_work[0] != '\0' && strcmp(v.tracked_work, "-") != 0,
          "the pending region names no tracked work");

    check(alignment_verdict("gdn_prepare", &v) == ALIGNMENT_OK, "gdn_prepare: %s",
          alignment_last_error());
    check(v.decision == ALIGNMENT_EXACT_BY_CONSTRUCTION, "gdn_prepare is %s",
          alignment_decision_name(v.decision));
}

/* ------------------------------------------------------------------ */
/* The separation: classify before bounding                           */
/* ------------------------------------------------------------------ */

static void test_separation(void) {
    check(alignment_classify(1, 1) == ALIGNMENT_DEVIATION_NUMERICAL,
          "same weights and same sampler was not classified numerical");
    check(alignment_classify(0, 1) == ALIGNMENT_DEVIATION_POLICY_CHANGE,
          "different weights was not classified a policy change");
    check(alignment_classify(1, 0) == ALIGNMENT_DEVIATION_SAMPLER,
          "a different sampler configuration was not classified a sampler difference");
    check(alignment_classify(0, 0) == ALIGNMENT_DEVIATION_POLICY_CHANGE,
          "different weights and sampler was not classified a policy change");

    int within = -1;
    /* A policy change with a large scalar would read as an alarming numerical mismatch
     * if it were allowed into the numerical column; it is refused instead. This is the
     * Stage-6 rule "report numerical mismatch separately from real policy changes". */
    const double large = 3.0;
    check(alignment_check_observation("attention_core", alignment_classify(0, 1), large, &within) ==
              ALIGNMENT_ERR_STATE,
          "a policy change was scored against a numerical bound");
    check(alignment_check_observation("attention_core", alignment_classify(1, 0), large, &within) ==
              ALIGNMENT_ERR_STATE,
          "a sampler difference was scored against a numerical bound");
    check(within == -1, "a refused comparison wrote its output");

    /* An exact-by-construction region is not checked by a tolerance: exactness is a
     * construction claim, not a bound. */
    check(alignment_check_observation("embedding", ALIGNMENT_DEVIATION_NUMERICAL, 0.0, &within) ==
              ALIGNMENT_ERR_STATE,
          "an exact-by-construction region was checked against a tolerance");

    /* A pending region has no measured bound, so a numerical claim on it is refused
     * rather than compared against an invented number. */
    check(alignment_check_observation("rmsnorm", ALIGNMENT_DEVIATION_NUMERICAL, 1.0e-04, &within) ==
              ALIGNMENT_ERR_MISSING,
          "a numerical claim was made on a region with no measured bound");

    /* A declared exception is scored, and the bound is what was measured. */
    check(alignment_check_observation("attention_core", ALIGNMENT_DEVIATION_NUMERICAL, 5.0e-04,
                                      &within) == ALIGNMENT_OK && within == 1,
          "a difference inside the declared bound was not within it: %s", alignment_last_error());
    check(alignment_check_observation("attention_core", ALIGNMENT_DEVIATION_NUMERICAL, 2.0e-03,
                                      &within) == ALIGNMENT_OK && within == 0,
          "a difference outside the declared bound was reported within it");
    check(alignment_check_observation("gemm_bf16", ALIGNMENT_DEVIATION_NUMERICAL, 1.0e-01,
                                      &within) == ALIGNMENT_OK && within == 1,
          "claim D's 1.0e-01 is not within gemm_bf16's declared bound");

    /* Malformed observations and unknown regions are refusals. */
    check(alignment_check_observation("attention_core", ALIGNMENT_DEVIATION_NUMERICAL, -1.0,
                                      &within) == ALIGNMENT_ERR_RANGE,
          "a negative difference was accepted");
    check(alignment_check_observation("attention_core", ALIGNMENT_DEVIATION_NUMERICAL, NAN,
                                      &within) == ALIGNMENT_ERR_RANGE,
          "a non-finite difference was accepted");
    check(alignment_check_observation("not_a_region", ALIGNMENT_DEVIATION_NUMERICAL, 0.0,
                                      &within) == ALIGNMENT_ERR_ARG,
          "an observation on a region outside the inventory was accepted");
}

int main(void) {
    test_verdicts_cover_the_inventory();
    test_known_verdicts();
    test_separation();

    if (g_failures != 0) {
        fprintf(stderr, "alignment_test: %d of %d check(s) failed\n", g_failures, g_checks);
        return EXIT_FAILURE;
    }
    printf("alignment_test: the Stage-6 verdicts follow the Stage-1 measurements and numerical "
           "mismatch stays separate from policy and sampler differences\n");
    return EXIT_SUCCESS;
}
