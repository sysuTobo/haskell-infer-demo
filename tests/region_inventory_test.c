/*
 * region_inventory_test.c - CPU gate for the Stage-1 region inventory.
 *
 * No GPU and no weights. It checks the two things a reader cannot: that the
 * inventory covers the plan's in-scope forward operations and nothing else, and
 * that every registered verdict is one the device harness can enforce.
 *
 *   - every inventory region is a region the manifest registry names, and every
 *     manifest region is either inventoried or explicitly excluded, so the two
 *     tables cannot drift apart;
 *   - `exact` is claimed only where the manifest registry already says
 *     `deterministic`, and it carries 0.0, not a rounded measurement;
 *   - `unverified` carries no number, `exception` carries all of one, and
 *     `not_applicable` carries a reason;
 *   - the trainer traversal cases (train_forward, eval_no_autograd, recompute,
 *     backward) are available from no region: registering them would claim an
 *     API the plan's Stages 3-4 have not built;
 *   - the case list is matched as whole tokens, so "decode" cannot match
 *     "recurrent_prefill".
 *
 * It also prints the inventory, because the table is otherwise only readable by
 * opening the source. Run: ctest --test-dir csrc/build-libs -R test_region_inventory.
 */
#include "manifest.h"
#include "regions.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

static void check(int condition, const char *fmt, ...) {
    if (condition) return;
    ++g_failures;
    va_list ap;
    va_start(ap, fmt);
    fputs("region_inventory_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* The plan's Stage-1 bullet list, as region names. Keeping it here (rather than
 * deriving it from the inventory) is the point: the inventory cannot certify its
 * own coverage. */
static const char *const kRequiredRegions[] = {
    /* Embedding gather, norms, GEMMs, residual, LM head, log-softmax/gather, losses */
    "embedding", "rmsnorm", "per_head_norm", "gemm_bf16", "gemm_fp32_lmhead",
    "residual_add", "logits_softmax_gather", "masked_loss",
    /* Attention: Q/gate split, RoPE, KV write, core, output gate */
    "q_gate_split", "rope", "kv_write", "attention_core", "attention_output_gate",
    /* GDN: conv1d, conv SiLU, prepare, core, gated norm */
    "gdn_conv1d", "conv_silu", "gdn_prepare", "gdn_core", "gdn_gated_norm",
    /* Dense MLP */
    "silu_mul",
    /* The proposed generation-only sampler, registered separately from the
     * trainer's differentiable loss. */
    "sampler_softmax_cdf",
    /* The trainer's backward path. */
    "backward",
};

static int is_required(const char *region) {
    for (size_t i = 0; i < sizeof(kRequiredRegions) / sizeof(kRequiredRegions[0]); ++i) {
        if (strcmp(kRequiredRegions[i], region) == 0) return 1;
    }
    return 0;
}

static const struct ManifestRegion *manifest_region_find(const char *name) {
    int count = 0;
    const struct ManifestRegion *regions = manifest_default_regions(&count);
    for (int i = 0; i < count; ++i) {
        if (strcmp(regions[i].region, name) == 0) return &regions[i];
    }
    return NULL;
}

static int inventory_has(const char *name) { return region_inventory_find(name) != NULL; }

static int exclusion_has(const char *name) {
    int count = 0;
    const struct RegionExclusion *exclusions = region_exclusions(&count);
    for (int i = 0; i < count; ++i) {
        if (strcmp(exclusions[i].region, name) == 0) return 1;
    }
    return 0;
}

static int verdict_is_known(const char *verdict) {
    return strcmp(verdict, REGION_VERDICT_EXACT) == 0 ||
           strcmp(verdict, REGION_VERDICT_EXCEPTION) == 0 ||
           strcmp(verdict, REGION_VERDICT_UNVERIFIED) == 0 ||
           strcmp(verdict, REGION_VERDICT_NOT_APPLICABLE) == 0;
}

static int nonempty(const char *text) { return text != NULL && text[0] != '\0'; }

/* One pair must carry exactly the evidence its verdict needs: no less (an
 * exception with no numbers is not quantified) and no more (an unverified pair
 * with a number would look established). */
static void check_pair(const struct RegionInventoryEntry *entry, const struct RegionCasePair *p) {
    const char *where = entry->region;
    check(nonempty(p->left) && nonempty(p->right) && strcmp(p->left, p->right) != 0,
          "%s: a pair needs two distinct case names", where);
    check(region_case_known(p->left) && region_case_known(p->right),
          "%s: pair %s/%s uses a case outside the vocabulary", where, p->left, p->right);
    check(verdict_is_known(p->verdict), "%s: unknown verdict '%s'", where, p->verdict);
    check(nonempty(p->reason), "%s: pair %s/%s has no reason", where, p->left, p->right);
    check(nonempty(p->arch) && nonempty(p->shapes) && nonempty(p->state),
          "%s: pair %s/%s leaves an evidence field empty", where, p->left, p->right);

    const int measured = isfinite(p->max_abs) || isfinite(p->rms);
    if (strcmp(p->verdict, REGION_VERDICT_EXACT) == 0) {
        check(p->max_abs == 0.0 && p->rms == 0.0,
              "%s: an exact pair must claim exactly 0.0 (got %g/%g)", where, p->max_abs, p->rms);
        const struct ManifestRegion *registry = manifest_region_find(where);
        check(registry != NULL && strcmp(registry->determinism, "deterministic") == 0,
              "%s: 'exact' needs the manifest registry to say deterministic", where);
        /* Both cases must be reachable, or the fixture cannot run. */
        check(region_case_available(where, p->left) && region_case_available(where, p->right),
              "%s: pair %s/%s is not reachable from the declared cases '%s'", where,
              p->left, p->right, entry->cases);
    } else if (strcmp(p->verdict, REGION_VERDICT_UNVERIFIED) == 0) {
        check(!measured, "%s: an unverified pair must not carry a number", where);
        const struct ManifestRegion *registry = manifest_region_find(where);
        check(registry != NULL && strcmp(registry->determinism, "unverified") == 0,
              "%s: 'unverified' expects the manifest registry's conservative verdict", where);
        check(region_case_available(where, p->left) && region_case_available(where, p->right),
              "%s: pair %s/%s is not reachable from the declared cases '%s'", where,
              p->left, p->right, entry->cases);
    } else if (strcmp(p->verdict, REGION_VERDICT_EXCEPTION) == 0) {
        check(isfinite(p->max_abs) && isfinite(p->rms) && (p->max_abs > 0.0 || p->rms > 0.0),
              "%s: a quantified exception needs a measured deviation", where);
        check(strcmp(p->arch, "-") != 0 && strcmp(p->shapes, "-") != 0 &&
                  strcmp(p->state, "-") != 0,
              "%s: a quantified exception needs arch, shapes and the state comparison", where);
    } else if (strcmp(p->verdict, REGION_VERDICT_NOT_APPLICABLE) == 0) {
        check(!measured, "%s: a not_applicable pair must not carry a number", where);
    }
}

static void check_entry(const struct RegionInventoryEntry *entry) {
    check(nonempty(entry->region), "an entry has no region name");
    check(nonempty(entry->family) && nonempty(entry->impl) && nonempty(entry->inputs) &&
              nonempty(entry->outputs) && nonempty(entry->persistent_state) &&
              nonempty(entry->saved_for_backward) && nonempty(entry->cases),
          "%s: an inventory field is empty", entry->region);
    check(entry->in_trainer_allowlist == 0 || entry->in_trainer_allowlist == 1,
          "%s: the trainer-allowlist flag is not a boolean", entry->region);
    check(manifest_region_find(entry->region) != NULL,
          "%s: the inventory names a region the manifest registry does not", entry->region);
    check(!exclusion_has(entry->region),
          "%s: a region cannot be both inventoried and excluded", entry->region);
    check(entry->pair_count >= 0 && (entry->pair_count == 0) == (entry->pairs == NULL),
          "%s: pair_count and the pair table disagree", entry->region);
    if (strcmp(entry->cases, "none") != 0 && entry->pair_count == 0)
        check(0, "%s: declares reachable cases but registers no pair", entry->region);

    int all_not_applicable = entry->pair_count > 0;
    for (int i = 0; i < entry->pair_count; ++i) {
        check_pair(entry, &entry->pairs[i]);
        if (strcmp(entry->pairs[i].verdict, REGION_VERDICT_NOT_APPLICABLE) != 0)
            all_not_applicable = 0;
    }
    /* A region that can be reached at all must have at least one verdict the
     * device harness enforces, unless every pair is genuinely unreachable. */
    if (!all_not_applicable && strcmp(entry->cases, "none") != 0) {
        int enforceable = 0;
        for (int i = 0; i < entry->pair_count; ++i) {
            const char *v = entry->pairs[i].verdict;
            if (strcmp(v, REGION_VERDICT_EXACT) == 0 ||
                strcmp(v, REGION_VERDICT_UNVERIFIED) == 0 ||
                strcmp(v, REGION_VERDICT_EXCEPTION) == 0)
                enforceable = 1;
        }
        check(enforceable, "%s: no pair carries a verdict the harness can enforce", entry->region);
    }
    /* The forward `cases` column never names a traversal. Stage 3 added a
     * single-sequence teacher-forced forward but not a per-step schedule, so
     * train_forward and eval_no_autograd still have no API; recompute is an attention
     * backward option rather than a per-region forward case; and the backward itself
     * now exists (Stage 4) but is registered in the backward inventory, not as a
     * per-region forward case. */
    check(!region_case_available(entry->region, REGION_CASE_TRAIN_FORWARD),
          "%s: registers train_forward, which has no trainer API", entry->region);
    check(!region_case_available(entry->region, REGION_CASE_EVAL_NO_AUTOGRAD),
          "%s: registers eval_no_autograd, which has no trainer API", entry->region);
    check(!region_case_available(entry->region, REGION_CASE_RECOMPUTE),
          "%s: registers recompute, which has no trainer API", entry->region);
    check(!region_case_available(entry->region, REGION_CASE_BACKWARD),
          "%s: registers backward as a forward case, which is not what it is", entry->region);
}

static void print_inventory(void) {
    int count = 0;
    const struct RegionInventoryEntry *inventory = region_inventory(&count);
    int exclusion_count = 0;
    const struct RegionExclusion *exclusions = region_exclusions(&exclusion_count);
    printf("Stage-1 region inventory v%d: %d regions, %d excluded\n",
           region_inventory_version(), count, exclusion_count);
    for (int i = 0; i < count; ++i) {
        const struct RegionInventoryEntry *e = &inventory[i];
        printf("\n%-22s trainer=%s family=%s cases=%s\n", e->region,
               e->in_trainer_allowlist ? "yes" : "no", e->family, e->cases);
        printf("  impl    %s\n", e->impl);
        printf("  in      %s\n", e->inputs);
        printf("  out     %s\n", e->outputs);
        printf("  state   %s\n", e->persistent_state);
        printf("  saved   %s\n", e->saved_for_backward);
        if (strcmp(e->note, "none") != 0) printf("  note    %s\n", e->note);
        for (int j = 0; j < e->pair_count; ++j) {
            const struct RegionCasePair *p = &e->pairs[j];
            printf("  pair    %-18s %-18s %-14s", p->left, p->right, p->verdict);
            if (isfinite(p->max_abs)) printf(" max_abs=%.9g rms=%.9g", p->max_abs, p->rms);
            printf("\n            %s\n", p->reason);
        }
    }
    int n = exclusion_count;
    printf("\nOutside this inventory:\n");
    for (int i = 0; i < n; ++i) printf("  %-22s %s\n", exclusions[i].region, exclusions[i].reason);
}

int main(void) {
    check(region_inventory_version() == REGION_INVENTORY_VERSION,
          "the inventory version does not match the header");

    int count = 0;
    const struct RegionInventoryEntry *inventory = region_inventory(&count);
    check(count > 0 && inventory != NULL, "the inventory is empty");

    /* Unique names. */
    for (int i = 0; i < count; ++i) {
        check_entry(&inventory[i]);
        for (int j = i + 1; j < count; ++j) {
            check(strcmp(inventory[i].region, inventory[j].region) != 0,
                  "duplicate region '%s'", inventory[i].region);
        }
    }

    /* Both directions of the manifest link: nothing invented, nothing dropped. */
    int manifest_count = 0;
    const struct ManifestRegion *manifest_regions = manifest_default_regions(&manifest_count);
    for (int i = 0; i < manifest_count; ++i) {
        const char *name = manifest_regions[i].region;
        check(inventory_has(name) || exclusion_has(name),
              "manifest region '%s' is neither inventoried nor excluded", name);
    }
    /* ...and they must agree that the trainer traversal does not exist. A manifest
     * row that lists train_forward/eval_no_autograd/recompute/backward would
     * advertise coverage no region can be exercised for, and the row's digest is
     * part of the numerical policy. */
    static const char *const kAbsentCases[] = {
        REGION_CASE_TRAIN_FORWARD, REGION_CASE_EVAL_NO_AUTOGRAD, REGION_CASE_RECOMPUTE,
        REGION_CASE_BACKWARD};
    for (int i = 0; i < manifest_count; ++i) {
        for (size_t c = 0; c < sizeof(kAbsentCases) / sizeof(kAbsentCases[0]); ++c) {
            check(strstr(manifest_regions[i].cases, kAbsentCases[c]) == NULL,
                  "manifest region '%s' claims the unavailable case '%s' in its cases column",
                  manifest_regions[i].region, kAbsentCases[c]);
        }
    }
    int exclusion_count = 0;
    const struct RegionExclusion *exclusions = region_exclusions(&exclusion_count);
    for (int i = 0; i < exclusion_count; ++i) {
        check(manifest_region_find(exclusions[i].region) != NULL,
              "excluded region '%s' is not a manifest region", exclusions[i].region);
        check(!inventory_has(exclusions[i].region),
              "region '%s' is both excluded and inventoried", exclusions[i].region);
        check(nonempty(exclusions[i].reason), "excluded region '%s' has no reason",
              exclusions[i].region);
    }

    /* The plan's bullet list, checked one region at a time. */
    for (size_t i = 0; i < sizeof(kRequiredRegions) / sizeof(kRequiredRegions[0]); ++i) {
        check(inventory_has(kRequiredRegions[i]),
              "the plan's in-scope region '%s' is missing from the inventory",
              kRequiredRegions[i]);
    }
    /* ...and no invented extras. */
    for (int i = 0; i < count; ++i) {
        check(is_required(inventory[i].region),
              "region '%s' is inventoried but the plan does not list it", inventory[i].region);
    }

    /* Case lookup is whole-token and vocabulary-bounded. */
    check(region_case_known(REGION_CASE_DECODE) && region_case_known(REGION_CASE_TAIL1),
          "the case vocabulary is missing a case");
    check(!region_case_known("prefill"), "'prefill' is not a Stage-1 case name");
    check(!region_case_known(""), "the empty string is not a case");
    check(region_case_available("rope", REGION_CASE_DECODE),
          "rope should be reachable from decode");
    check(region_case_available("gdn_core", REGION_CASE_RECURRENT_PREFILL),
          "gdn_core should be reachable from recurrent_prefill");
    check(!region_case_available("rope", "prefill"),
          "'prefill' must not match any region's case list");
    check(!region_case_available("logits_softmax_gather", REGION_CASE_DECODE),
          "the host selector has no device case");
    check(!region_case_available("nonexistent_region", REGION_CASE_DECODE),
          "an unknown region must not report a case as available");

    /* Pair lookup is order-free. */
    check(region_find_pair("rope", REGION_CASE_CHUNKED_PREFILL,
                           REGION_CASE_RECURRENT_PREFILL) != NULL &&
              region_find_pair("rope", REGION_CASE_RECURRENT_PREFILL,
                               REGION_CASE_CHUNKED_PREFILL) != NULL,
          "pair lookup must not depend on the argument order");
    check(region_find_pair("rope", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE) == NULL,
          "an unregistered pair must not be found");

    if (getenv("REGION_INVENTORY_QUIET") == NULL) print_inventory();

    if (g_failures != 0) {
        fprintf(stderr, "region_inventory_test: %d check(s) failed\n", g_failures);
        return EXIT_FAILURE;
    }
    printf("region_inventory_test: the Stage-1 inventory covers the plan's in-scope path, "
           "matches the manifest registry, and every registered verdict is enforceable\n");
    return EXIT_SUCCESS;
}
