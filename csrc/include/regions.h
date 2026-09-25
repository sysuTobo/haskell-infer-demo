/**
 * regions.h - Stage-1 region inventory of the dense/dense-hybrid forward path.
 *
 * See docs/plan-numeric-contract.md "Stage 1 - Region inventory and cross-case
 * harness". This is a *committed* table, not a runtime fact: it records, per
 * region, what the region reads and writes, which persistent state it owns,
 * which values a backward would need, which execution cases reach it, and -- for
 * each pair of cases it can be reached under -- a verdict:
 *
 *   exact           identical inputs and identical persistent state produce
 *                   bitwise identical outputs and state. The device harness
 *                   re-establishes this (max_abs == 0); it is claimed only where
 *                   the manifest registry already calls the region
 *                   `deterministic` (one elementwise pass, no cross-thread
 *                   reduction, no library tiling decision).
 *   exception       a known, quantified deviation. It carries the tested
 *                   architecture, shapes, max_abs/rms and how persistent state
 *                   was compared, all rounded up to two significant digits so the
 *                   bound has a margin over the observation. It does NOT authorize
 *                   a bitwise whole-model claim. Stage 2 (claims B-D) quantified
 *                   the deviations the Stage-1 fixtures could only report.
 *   unverified      the cases are compared and the difference is measured and
 *                   reported, but invariance is not established (a library
 *                   reduction/tiling order, a cross-device reduction, or a
 *                   decomposition with a different schedule). Stage 2's
 *                   experiments A-E are what would establish it.
 *   not_applicable  the pair cannot be run in this release. The reason says why
 *                   (no trainer API, host-only region, not an independent entry
 *                   point). Requesting such a case fails explicitly rather than
 *                   silently passing.
 *
 * The table is deliberately *not* projected into the execution manifest's hashed
 * numerical policy. That table's `regions` block records what a capture actually
 * ran (implementation + determinism); this inventory records which cases a
 * region is reachable under and what the harness has established about them.
 * Folding test claims into `numerical_policy_id` would make every capture's
 * numerical identity move when a test is added, without strengthening any gate.
 * `test_region_inventory` keeps the two consistent instead.
 *
 * Nothing here is defaulted: a region whose pair was never measured carries NaN
 * for max_abs/rms and says so in `reason`, and an unsupported case is available
 * from no region.
 */

#ifndef HASKELL_INFER_REGIONS_H
#define HASKELL_INFER_REGIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#define REGION_INVENTORY_VERSION 1

/* Verdict vocabulary. Exactly these four strings may appear in a pair. */
#define REGION_VERDICT_EXACT "exact"
#define REGION_VERDICT_EXCEPTION "exception"
#define REGION_VERDICT_UNVERIFIED "unverified"
#define REGION_VERDICT_NOT_APPLICABLE "not_applicable"

/* Case vocabulary. "Execution case" means the way a region is reached, not a
 * model family: the same weights and the same token prefix under a different
 * chunking, a different query count, or a trainer traversal.
 *
 * chunked_prefill     a multi-token forward through the chunked path
 * recurrent_prefill   the same tokens one at a time (recurrent GDN path)
 * tail1               the trailing one-token chunk of a chunked prefill
 * decode              a single-token forward with the cache/state carried
 * train_forward       teacher-forced trainer forward (Stage 3-4; absent)
 * eval_no_autograd    the same forward without saved activations (absent)
 * recompute           activation recomputation for backward (absent)
 * backward            gradient computation (absent)
 *
 * The four absent cases are the trainer traversal the plan's Stages 3-4
 * introduce. Registering them as reachable would claim an API that does not
 * exist. */
#define REGION_CASE_CHUNKED_PREFILL "chunked_prefill"
#define REGION_CASE_RECURRENT_PREFILL "recurrent_prefill"
#define REGION_CASE_TAIL1 "tail1"
#define REGION_CASE_DECODE "decode"
#define REGION_CASE_TRAIN_FORWARD "train_forward"
#define REGION_CASE_EVAL_NO_AUTOGRAD "eval_no_autograd"
#define REGION_CASE_RECOMPUTE "recompute"
#define REGION_CASE_BACKWARD "backward"

/* One ordered pair of cases a region can be reached under, with the registered
 * verdict and whatever evidence the verdict requires. */
struct RegionCasePair {
    const char *left;
    const char *right;
    const char *verdict;  /* REGION_VERDICT_* */
    const char *arch;     /* tested architecture, or "-" when not measured */
    const char *shapes;   /* tested shapes, or "-" when not measured */
    double max_abs;       /* NaN when not measured; 0 for an established exact pair */
    double rms;           /* NaN when not measured */
    const char *state;    /* how persistent state was compared, or "-" */
    const char *reason;
};

/* One in-scope forward region. */
struct RegionInventoryEntry {
    const char *region;              /* the name the manifest registry uses */
    int in_trainer_allowlist;        /* 1 = first trainer allowlist, 0 = inference only */
    const char *family;              /* families this region exists for */
    const char *impl;                /* file: symbol that runs it */
    const char *inputs;              /* dtypes, layout, what the case varies */
    const char *outputs;
    const char *persistent_state;    /* mutated in place across calls, or "none" */
    const char *saved_for_backward;  /* what a backward needs; "nothing yet" when none is produced */
    const char *cases;               /* comma-separated available cases, or "none" */
    const char *note;                /* region-level status a pair cannot carry */
    int pair_count;                  /* 0 when no case pair is reachable */
    const struct RegionCasePair *pairs;
};

/* A region that is deliberately outside this inventory, with the reason. */
struct RegionExclusion {
    const char *region;
    const char *reason;
};

int region_inventory_version(void);

/* The committed inventory and the explicit exclusions. */
const struct RegionInventoryEntry *region_inventory(int *count);
const struct RegionExclusion *region_exclusions(int *count);

/* The entry for @region, or NULL when it is not in the inventory. */
const struct RegionInventoryEntry *region_inventory_find(const char *region);

/* 1 when @case_name is in the Stage-1 case vocabulary. */
int region_case_known(const char *case_name);

/* 1 when @region exists and its `cases` list contains @case_name. An unsupported
 * shape or case is then rejected by the region's own entry point (a thrown
 * std::invalid_argument or a non-zero status), never by a silent no-op. */
int region_case_available(const char *region, const char *case_name);

/* 1 when the pair (left, right) is registered in either order. */
const struct RegionCasePair *region_find_pair(const char *region,
                                              const char *left, const char *right);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_REGIONS_H */
