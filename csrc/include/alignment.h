/**
 * alignment.h - The Stage-6 numerical-alignment decision (plan Stage 6).
 *
 * docs/plan-numeric-contract.md, Stage 6 is the stage that decides, *from the
 * Stage-2 evidence*, what to do about the cross-case deviations Stage 2 measured.
 * It is deliberately not a mandate to write new kernels: its options are "a fixed
 * reduction-tree kernel, explicitly constrained cuBLAS algorithms/workspaces with a
 * limited tested claim, or a declared numerical exception", and it warns that
 * "pinning an algorithm is not a general proof" and that tolerance-based work "must
 * report their error and learning stability rather than relabel exceptions as
 * bitwise guarantees". Two things follow, and both are objects here rather than
 * prose:
 *
 *   1. `alignment_verdict` - one region's Stage-6 verdict, *derived* from the
 *      committed Stage-1 inventory (`region_inventory`) rather than copied from it,
 *      so the two cannot drift: a region with a measured `exception` pair is a
 *      DECLARED_EXCEPTION, a region whose only gap is an `unverified` pair is an
 *      INVARIANT_KERNEL_PENDING (with the option that would remove it named), and a
 *      region every one of whose pairs is `exact` or `not_applicable` is
 *      EXACT_BY_CONSTRUCTION. A verdict is never invented - it is what Stage 2 left
 *      behind, read back under Stage 6's vocabulary.
 *
 *   2. `alignment_classify` + `alignment_check_observation` - the reporting rule the
 *      Stage-6 gate turns on: "Report numerical mismatch separately from real policy
 *      changes and sampler differences." A single scalar cannot carry all three, so
 *      the kind is decided *before* any bound is applied, and scoring a policy change
 *      or a sampler difference against a numerical bound is refused rather than
 *      producing a number that reads like a numerical mismatch. The same refusal
 *      stops the other mislabel: an EXACT_BY_CONSTRUCTION region has no bound to
 *      compare against, because exactness is not established by a tolerance.
 *
 * This file is CUDA-free, like train.h and backward.h: the verdicts are a committed
 * table over `regions.c` and the checks are string/`double` comparisons a CPU test
 * drives directly.
 *
 * What Stage 6 does **not** do is build the invariant kernels. The verdicts name
 * them as tracked work and the module reports the exception with its measured bound;
 * choosing the exception *is* one of the plan's three options, not an omission.
 */
#ifndef HASKELL_INFER_ALIGNMENT_H
#define HASKELL_INFER_ALIGNMENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ALIGNMENT_ABI_VERSION 1

typedef enum {
    ALIGNMENT_OK = 0,
    ALIGNMENT_ERR_ARG = 1,     /* null or invalid argument (see alignment_last_error) */
    ALIGNMENT_ERR_STATE = 2,   /* the comparison itself is mislabelled */
    ALIGNMENT_ERR_MISSING = 3, /* no measured bound exists for the claim */
    ALIGNMENT_ERR_RANGE = 4,   /* a non-finite observation or bound */
} AlignmentStatus;

/* The plan's Stage-6 options, in its order. These are three named choices rather
 * than an "aligned" flag on purpose: "a declared numerical exception" is a decision,
 * not a failure to decide. */
typedef enum {
    ALIGNMENT_EXACT_BY_CONSTRUCTION = 0,    /* every case pair is elementwise/exact */
    ALIGNMENT_INVARIANT_KERNEL_PENDING = 1, /* a measured gap with no bound yet */
    ALIGNMENT_DECLARED_EXCEPTION = 2,       /* measured and bounded; carried forward */
} AlignmentDecision;

/* Why an observed difference is what it is. Stage 6's gate separates these three; a
 * policy change is genuine staleness, a sampler difference is a numerical-policy
 * difference between two logprob computations, and only the third is a *numerical*
 * mismatch. */
typedef enum {
    ALIGNMENT_DEVIATION_NUMERICAL = 0,
    ALIGNMENT_DEVIATION_POLICY_CHANGE = 1,
    ALIGNMENT_DEVIATION_SAMPLER = 2,
} AlignmentDeviationKind;

const char *alignment_decision_name(AlignmentDecision decision);
const char *alignment_deviation_name(AlignmentDeviationKind kind);

/* One region's Stage-6 verdict. `bound_max_abs`/`bound_rms` are the maximum over the
 * region's pairs that carry a finite measurement (NaN when the region has none), so a
 * declared exception cannot be *tighter* than what was measured. `tracked_work` names
 * the Stage-6 option that would remove a pending deviation, or "-" (an accepted
 * exception needs no further work to be a decision). */
struct AlignmentVerdict {
    const char *region;
    AlignmentDecision decision;
    double bound_max_abs;
    double bound_rms;
    const char *tracked_work;
};

/* The verdict for @region, derived from the Stage-1 inventory. Fails with
 * ALIGNMENT_ERR_ARG for a region the inventory does not contain (a Stage-6 verdict
 * for a region that does not exist would be a claim about nothing). */
AlignmentStatus alignment_verdict(const char *region, struct AlignmentVerdict *out);

/* Classify a comparison before comparing it. Only the two identities decide: a
 * difference measured with different weights is a policy change and one measured with
 * a different sampling configuration (or on the sampler's recorded logprob) is a
 * sampler difference; neither is a numerical mismatch. */
AlignmentDeviationKind alignment_classify(int same_weights, int same_sampler_config);

/* Score an observation against the region's declared bound. Three refusals, each the
 * mislabel the Stage-6 gate exists to prevent:
 *   - @kind is POLICY_CHANGE or SAMPLER (ALIGNMENT_ERR_STATE): the observation is a
 *     different quantity and must be reported in its own column, not scored here;
 *   - the region's verdict is EXACT_BY_CONSTRUCTION (ALIGNMENT_ERR_STATE): an exact
 *     claim has no tolerance to check against;
 *   - the region has no finite measured bound (ALIGNMENT_ERR_MISSING): a numerical
 *     claim needs a measurement, and inventing one is the mislabel.
 * On ALIGNMENT_OK, `*out_within` is 1 when `observed_max_abs <= bound_max_abs`. */
AlignmentStatus alignment_check_observation(const char *region, AlignmentDeviationKind kind,
                                            double observed_max_abs, int *out_within);

/* The self-consistency gate, walked by `test_alignment`: the derivation covers every
 * in-scope Stage-1 region, a DECLARED_EXCEPTION carries a finite bound, and an
 * INVARIANT_KERNEL_PENDING names its tracked work. Returns ALIGNMENT_OK or fails with
 * the first offending region in the error text. */
AlignmentStatus alignment_self_check(void);

const char *alignment_last_error(void);
void alignment_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_ALIGNMENT_H */
