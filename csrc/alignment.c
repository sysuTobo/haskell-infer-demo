/**
 * alignment.c - The Stage-6 numerical-alignment decision (plan Stage 6).
 * See csrc/include/alignment.h for the contract.
 *
 * CUDA-free: the verdicts are read back out of `regions.c` and the separation checks
 * are comparisons a CPU test drives directly.
 */
#include "alignment.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "regions.h"

/* ------------------------------------------------------------------ */
/* Per-thread error text                                              */
/* ------------------------------------------------------------------ */

static _Thread_local char g_error[256];

const char *alignment_last_error(void) { return g_error; }

void alignment_clear_error(void) { g_error[0] = '\0'; }

static AlignmentStatus fail(AlignmentStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

/* ------------------------------------------------------------------ */
/* The tracked work behind a pending verdict                          */
/* ------------------------------------------------------------------ */

/* Stage 6's options for *removing* a deviation, named per region. Only regions whose
 * Stage-1 verdict is `unverified` need an entry: an `exception` is already a measured
 * decision (choosing the exception is the plan's third option), and an exact-by-
 * construction region has no deviation to remove. The gate below refuses a pending
 * verdict whose work is unnamed, so adding an `unverified` pair to `regions.c` forces
 * this table to answer for it. */
static const struct {
    const char *region;
    const char *work;
} kTrackedWork[] = {
    {"rmsnorm",
     "a fixed reduction-tree row-norm kernel, or an explicitly constrained library "
     "configuration carrying a tested claim (Stage 6's GEMM options applied to the "
     "in-row reduction)"},
    {"per_head_norm",
     "a fixed reduction-tree per-head norm kernel, or a constrained library "
     "configuration with a tested claim"},
    {"gdn_conv1d",
     "a fixed scan order for the causal-conv shift register (chunked versus per-token), "
     "or a measurement that establishes the library's scan"},
    {"gdn_gated_norm",
     "a fixed reduction-tree gate kernel over head_dim, or a constrained library "
     "configuration with a tested claim"},
};

static const char *tracked_work_for(const char *region) {
    for (size_t i = 0; i < sizeof(kTrackedWork) / sizeof(kTrackedWork[0]); ++i) {
        if (strcmp(kTrackedWork[i].region, region) == 0) return kTrackedWork[i].work;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Names                                                              */
/* ------------------------------------------------------------------ */

const char *alignment_decision_name(AlignmentDecision decision) {
    switch (decision) {
        case ALIGNMENT_EXACT_BY_CONSTRUCTION: return "exact_by_construction";
        case ALIGNMENT_INVARIANT_KERNEL_PENDING: return "invariant_kernel_pending";
        case ALIGNMENT_DECLARED_EXCEPTION: return "declared_exception";
    }
    return "unknown";
}

const char *alignment_deviation_name(AlignmentDeviationKind kind) {
    switch (kind) {
        case ALIGNMENT_DEVIATION_NUMERICAL: return "numerical";
        case ALIGNMENT_DEVIATION_POLICY_CHANGE: return "policy_change";
        case ALIGNMENT_DEVIATION_SAMPLER: return "sampler";
    }
    return "unknown";
}

/* ------------------------------------------------------------------ */
/* The verdict, derived from the Stage-1 inventory                    */
/* ------------------------------------------------------------------ */

AlignmentStatus alignment_verdict(const char *region, struct AlignmentVerdict *out) {
    if (region == NULL || out == NULL) {
        return fail(ALIGNMENT_ERR_ARG, "alignment_verdict: null argument");
    }
    const struct RegionInventoryEntry *entry = region_inventory_find(region);
    if (entry == NULL) {
        return fail(ALIGNMENT_ERR_ARG,
                    "alignment_verdict: '%s' is not a Stage-1 inventory region", region);
    }

    int saw_exception = 0;
    int saw_unverified = 0;
    double worst_max_abs = NAN;
    double worst_rms = NAN;
    for (int i = 0; i < entry->pair_count; ++i) {
        const struct RegionCasePair *pair = &entry->pairs[i];
        if (strcmp(pair->verdict, REGION_VERDICT_EXCEPTION) == 0) {
            saw_exception = 1;
            /* A declared exception's bound is the maximum over its pairs, so the
             * verdict cannot be tighter than the widest measurement behind it. */
            if (isfinite(pair->max_abs) && (!isfinite(worst_max_abs) || pair->max_abs > worst_max_abs)) {
                worst_max_abs = pair->max_abs;
            }
            if (isfinite(pair->rms) && (!isfinite(worst_rms) || pair->rms > worst_rms)) {
                worst_rms = pair->rms;
            }
        } else if (strcmp(pair->verdict, REGION_VERDICT_UNVERIFIED) == 0) {
            saw_unverified = 1;
        }
    }

    out->region = entry->region;
    out->bound_max_abs = worst_max_abs;
    out->bound_rms = worst_rms;
    if (saw_exception) {
        out->decision = ALIGNMENT_DECLARED_EXCEPTION;
        out->tracked_work = "-";
    } else if (saw_unverified) {
        out->decision = ALIGNMENT_INVARIANT_KERNEL_PENDING;
        const char *work = tracked_work_for(entry->region);
        out->tracked_work = work != NULL ? work : "";
    } else {
        out->decision = ALIGNMENT_EXACT_BY_CONSTRUCTION;
        out->tracked_work = "-";
    }
    return ALIGNMENT_OK;
}

/* ------------------------------------------------------------------ */
/* The reporting rule: classify, then (only then) bound               */
/* ------------------------------------------------------------------ */

AlignmentDeviationKind alignment_classify(int same_weights, int same_sampler_config) {
    if (!same_weights) return ALIGNMENT_DEVIATION_POLICY_CHANGE;
    if (!same_sampler_config) return ALIGNMENT_DEVIATION_SAMPLER;
    return ALIGNMENT_DEVIATION_NUMERICAL;
}

AlignmentStatus alignment_check_observation(const char *region, AlignmentDeviationKind kind,
                                            double observed_max_abs, int *out_within) {
    if (region == NULL || out_within == NULL) {
        return fail(ALIGNMENT_ERR_ARG, "alignment_check_observation: null argument");
    }
    if (kind != ALIGNMENT_DEVIATION_NUMERICAL) {
        return fail(ALIGNMENT_ERR_STATE,
                    "alignment_check_observation: a %s difference is not a numerical "
                    "mismatch and must be reported separately, not scored against the "
                    "numerical bound of '%s'",
                    alignment_deviation_name(kind), region);
    }
    if (!isfinite(observed_max_abs) || observed_max_abs < 0.0) {
        return fail(ALIGNMENT_ERR_RANGE,
                    "alignment_check_observation: '%s' observed a non-finite or negative "
                    "difference (%g)", region, observed_max_abs);
    }
    struct AlignmentVerdict verdict;
    const AlignmentStatus status = alignment_verdict(region, &verdict);
    if (status != ALIGNMENT_OK) return status;
    if (verdict.decision == ALIGNMENT_EXACT_BY_CONSTRUCTION) {
        return fail(ALIGNMENT_ERR_STATE,
                    "alignment_check_observation: '%s' is %s; exactness is not established "
                    "by a tolerance, so there is no bound to compare against",
                    region, alignment_decision_name(verdict.decision));
    }
    if (!isfinite(verdict.bound_max_abs)) {
        return fail(ALIGNMENT_ERR_MISSING,
                    "alignment_check_observation: '%s' is %s and carries no measured bound; "
                    "a numerical claim needs a measurement rather than an invented one",
                    region, alignment_decision_name(verdict.decision));
    }
    *out_within = observed_max_abs <= verdict.bound_max_abs ? 1 : 0;
    return ALIGNMENT_OK;
}

/* ------------------------------------------------------------------ */
/* The gate                                                           */
/* ------------------------------------------------------------------ */

AlignmentStatus alignment_self_check(void) {
    int count = 0;
    const struct RegionInventoryEntry *inventory = region_inventory(&count);
    if (inventory == NULL || count <= 0) {
        return fail(ALIGNMENT_ERR_STATE, "alignment_self_check: the Stage-1 inventory is empty");
    }
    for (int i = 0; i < count; ++i) {
        struct AlignmentVerdict verdict;
        const AlignmentStatus status = alignment_verdict(inventory[i].region, &verdict);
        if (status != ALIGNMENT_OK) return status;
        if (verdict.decision == ALIGNMENT_DECLARED_EXCEPTION) {
            if (!isfinite(verdict.bound_max_abs) || verdict.bound_max_abs <= 0.0) {
                return fail(ALIGNMENT_ERR_STATE,
                            "alignment_self_check: '%s' is a declared exception with no finite "
                            "measured bound; an exception is a measurement, not a default",
                            verdict.region);
            }
        } else if (verdict.decision == ALIGNMENT_INVARIANT_KERNEL_PENDING) {
            if (verdict.tracked_work == NULL || verdict.tracked_work[0] == '\0' ||
                strcmp(verdict.tracked_work, "-") == 0) {
                return fail(ALIGNMENT_ERR_STATE,
                            "alignment_self_check: '%s' is an invariant-kernel-pending region "
                            "that names no tracked work", verdict.region);
            }
        }
    }
    /* The reverse direction: a tracked-work entry for a region the inventory does not
     * carry (or one that is no longer pending) is a stale claim. */
    for (size_t i = 0; i < sizeof(kTrackedWork) / sizeof(kTrackedWork[0]); ++i) {
        struct AlignmentVerdict verdict;
        const AlignmentStatus status = alignment_verdict(kTrackedWork[i].region, &verdict);
        if (status != ALIGNMENT_OK) return status;
        if (verdict.decision != ALIGNMENT_INVARIANT_KERNEL_PENDING) {
            return fail(ALIGNMENT_ERR_STATE,
                        "alignment_self_check: tracked work is named for '%s' but its verdict "
                        "is %s", kTrackedWork[i].region, alignment_decision_name(verdict.decision));
        }
    }
    return ALIGNMENT_OK;
}
