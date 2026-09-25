/*
 * train_test.c - CPU gate for the trainable runtime's ownership objects (Stage 3).
 *
 * No GPU and no weights: the parameter store's buffers are opaque slots, so tree
 * bookkeeping (tying, frozen parameters, versions, readers, derived copies, the
 * teacher-forcing plan) is testable exactly as it runs in the engine. The tying
 * case is the *real* Qwen3-4B descriptor, whose `embed` and `lmHead` roles share
 * one template - the gate's "tied roles stay tied" is checked against the model
 * that actually has the tie, not a synthetic pair.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_train [-- descriptors/qwen3-4b.json]
 */
#include "model_desc.h"
#include "train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond, ...)                                                                     \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            ++g_failures;                                                                    \
            fprintf(stderr, "train_test: FAIL: ");                                           \
            fprintf(stderr, __VA_ARGS__);                                                    \
            fprintf(stderr, "  (train_last_error: %s)\n", train_last_error());               \
        }                                                                                    \
    } while (0)

/* A store with three parameters, two of which tie (the 'model.embed_tokens.weight'
 * template in the global scope) and one frozen: the shape the tying and frozen rules
 * need, without a model. */
static TrainStore *tiny_store(int *embed_logical, int *lm_head_logical, int *frozen_logical) {
    struct TrainParamSpec specs[3];
    const char *templates[3] = {"model.embed_tokens.weight", "model.embed_tokens.weight",
                                "model.norm.weight"};
    const char *names[3] = {"embed", "lmHead", "finalNorm"};
    specs[0] = (struct TrainParamSpec){-1, ROLE_EMBED, 100, 0, 1};
    specs[1] = (struct TrainParamSpec){-1, ROLE_LM_HEAD, 100, 0, 1};
    specs[2] = (struct TrainParamSpec){-1, ROLE_FINAL_NORM, 100, 1, 1};  /* frozen */
    TrainStore *store = train_store_create(specs, templates, names, 3);
    if (store == NULL) return NULL;
    *embed_logical = train_store_logical_of(store, -1, ROLE_EMBED);
    *lm_head_logical = train_store_logical_of(store, -1, ROLE_LM_HEAD);
    *frozen_logical = train_store_logical_of(store, -1, ROLE_FINAL_NORM);
    return store;
}

static void test_tying(void) {
    int embed = -1, lm_head = -1, frozen = -1;
    TrainStore *store = tiny_store(&embed, &lm_head, &frozen);
    CHECK(store != NULL, "tiny store creation");
    if (store == NULL) return;

    CHECK(train_store_param_count(store) == 3, "three (layer, role) parameters");
    CHECK(train_store_logical_count(store) == 2, "two logical parameters after tying");
    CHECK(embed == lm_head, "embed and lmHead resolve to one logical parameter (%d vs %d)", embed,
          lm_head);
    CHECK(train_store_alias_count(store, embed) == 2, "the tied parameter has two aliases");
    /* A publication has to write the compute weight once per alias, or one reader
     * keeps the old weight. */
    CHECK(train_store_slot_buffer_count(store, embed, TRAIN_SLOT_COMPUTE) == 2,
          "the tied compute weight exists once per alias");
    CHECK(train_store_slot_buffer_count(store, embed, TRAIN_SLOT_MASTER) == 1,
          "the tied parameter has one master weight, not two");
    CHECK(train_store_elements(store, embed) == 100, "tied element count");
    CHECK(strcmp(train_store_name(store, embed), "embed") == 0, "the first alias names the parameter");

    /* Frozen parameters are not trainable and have no training state. */
    CHECK(train_store_is_frozen(store, frozen), "finalNorm is frozen");
    CHECK(!train_store_is_trainable(store, frozen), "a frozen parameter is not trainable");
    void *host = malloc(64);
    CHECK(train_store_set_slot(store, frozen, TRAIN_SLOT_MASTER, host) == TRAIN_ERR_FROZEN,
          "a frozen parameter refuses a master buffer");
    CHECK(strstr(train_last_error(), "frozen") != NULL, "the refusal names the reason");
    free(host);
    CHECK(train_store_destroy(store) == TRAIN_OK, "destroy a store with no readers");
}

static void test_lifetime_and_publication(void) {
    int embed = -1, lm_head = -1, frozen = -1;
    TrainStore *store = tiny_store(&embed, &lm_head, &frozen);
    CHECK(store != NULL, "store creation");
    if (store == NULL) return;

    /* A derived copy: the FP32 GDN-norm case, here as an arbitrary second buffer.
     * It derives from the tied parameter into the frozen one, so the source and the
     * copy are genuinely different logical parameters. */
    CHECK(train_store_register_derived(store, embed, frozen, TRAIN_DERIVED_BF16_TO_FP32) ==
              TRAIN_OK,
          "register a derived copy");
    CHECK(train_store_derived_count(store) == 1, "one derived copy registered");
    CHECK(train_store_register_derived(store, embed, embed, TRAIN_DERIVED_BF16_TO_FP32) ==
              TRAIN_ERR_ARG,
          "a copy cannot derive from itself");

    /* Borrow: a context is a reader. */
    TrainContext *trainer = train_context_create(store, 0);
    TrainContext *rollout = train_context_create(store, 1);
    CHECK(trainer != NULL && rollout != NULL, "two contexts borrow the store");
    CHECK(train_context_borrowed_version(rollout) == 0, "a context borrows the current version");

    CHECK(train_store_begin_update(store) == TRAIN_ERR_BUSY,
          "an update is refused while a context borrows the version");
    CHECK(strstr(train_last_error(), "reader") != NULL, "the refusal counts the readers");

    /* A step is also a reader: its saved activations describe the weights it read. */
    struct TrainSavedSpec saved[2] = {
        {"attn_out", 0, 16, -1, 1},
        {"attn_out_alias", 0, 16, 0, 1},  /* aliases entry 0 */
    };
    TrainStep *step = train_step_create(store, saved, 2);
    CHECK(step != NULL, "step creation");
    CHECK(train_step_version(step) == 0, "the step records the version it read");
    CHECK(train_context_destroy(rollout) == TRAIN_OK, "release one context");
    CHECK(train_store_begin_update(store) == TRAIN_ERR_BUSY, "the trainer context still blocks");
    CHECK(train_context_destroy(trainer) == TRAIN_OK, "release the trainer context");
    CHECK(train_store_begin_update(store) == TRAIN_ERR_BUSY,
          "a live step still blocks an update, even with no contexts");

    /* Retain and free the step's saved values, including the alias rule. */
    CHECK(train_step_saved_count(step) == 2, "two saved values");
    CHECK(train_step_retain(step, 0, (void *)0x1000) == TRAIN_OK, "retain the first");
    CHECK(train_step_retain(step, 1, (void *)0x1000) == TRAIN_OK, "retain the alias");
    CHECK(train_step_live_count(step) == 2, "two live saved values");
    CHECK(train_step_free(step, 0) == TRAIN_ERR_BUSY,
          "freeing a buffer another live saved value aliases is refused");
    CHECK(train_step_destroy(step) == TRAIN_ERR_BUSY, "destroying a step with live values is refused");
    CHECK(train_step_free(step, 1) == TRAIN_OK, "free the alias first");
    CHECK(train_step_free(step, 0) == TRAIN_OK, "then the original");
    CHECK(train_step_free(step, 0) == TRAIN_ERR_STATE, "double free is refused");

    /* GDN chunk-boundary retention: the schedule is recorded, and its cost follows. */
    CHECK(train_step_set_bptt(step, TRAIN_BPTT_FULL_SEQUENCE, 4) == TRAIN_OK, "set full BPTT");
    CHECK(train_step_bptt(step) == TRAIN_BPTT_FULL_SEQUENCE, "the schedule is reported back");
    CHECK(train_step_gdn_state_elements(step, 48, 128) == 4LL * 48 * 128 * 128,
          "full-sequence retention keeps every chunk boundary's state");

    CHECK(train_step_destroy(step) == TRAIN_OK, "destroy the step once its values are released");

    /* The update window. */
    CHECK(train_store_begin_update(store) == TRAIN_OK, "an update opens with no readers");
    CHECK(train_store_begin_update(store) == TRAIN_ERR_EXCLUSIVE, "a second update is refused");
    CHECK(train_store_in_update(store), "the update window is open");
    CHECK(train_context_create(store, 0) == NULL, "a context cannot be created during an update");
    CHECK(train_store_publish(store) == TRAIN_OK, "publish bumps the version");
    CHECK(train_store_version(store) == 1, "version counters are monotonic");
    CHECK(train_store_stale_derived_count(store) == 1,
          "the publication marks the derived copy stale");
    CHECK(train_store_end_update(store) == TRAIN_ERR_STATE,
          "the update cannot end while a derived copy is stale");
    int source = -1, target = -1;
    TrainDerivedKind kind = TRAIN_DERIVED_BF16_TO_FP32;
    int stale = 0;
    CHECK(train_store_derived_at(store, 0, &source, &target, &kind, &stale) == TRAIN_OK &&
              stale == 1,
          "the stale copy is visible with its source and kind");
    CHECK(train_store_derived_refreshed(store, 0) == TRAIN_OK, "refresh the derived copy");
    CHECK(train_store_stale_derived_count(store) == 0, "nothing is stale after the refresh");
    CHECK(train_store_end_update(store) == TRAIN_OK, "the update ends once the copies are fresh");

    /* A new reader sees the new version. */
    TrainContext *after = train_context_create(store, 1);
    CHECK(train_context_borrowed_version(after) == 1, "a new context borrows the new version");
    CHECK(train_context_destroy(after) == TRAIN_OK, "release it");
    CHECK(train_store_destroy(store) == TRAIN_OK, "destroy");
}

static void test_schedule_and_replicas(void) {
    TrainStore *store = train_store_create(NULL, NULL, NULL, 0);
    CHECK(store == NULL, "an empty parameter set is refused");
    int embed = -1, lm_head = -1, frozen = -1;
    store = tiny_store(&embed, &lm_head, &frozen);
    if (store == NULL) return;

    /* The accumulation schedule is fixed: the same count, in order. */
    CHECK(train_store_note_microbatch(store, 0, 3) == TRAIN_OK, "first contribution");
    CHECK(train_store_note_microbatch(store, 1, 3) == TRAIN_OK, "second contribution");
    CHECK(train_store_note_microbatch(store, 3, 3) == TRAIN_ERR_SCHEDULE,
          "an out-of-order contribution is refused");
    CHECK(train_store_note_microbatch(store, 2, 4) == TRAIN_ERR_SCHEDULE,
          "a changed microbatch count is refused");
    CHECK(train_store_note_microbatch(store, 2, 3) == TRAIN_OK, "the schedule completes");

    /* Replicas: one copy needs nothing; more than one needs the merge *and* the
     * broadcast, and the answer is a pair of bits so neither can be skipped. */
    CHECK(train_store_required_sync(store, embed) == TRAIN_SYNC_NONE, "one copy needs no sync");
    const int devices[2] = {0, 1};
    CHECK(train_store_set_replica_devices(store, embed, 2, devices) == TRAIN_OK, "record replicas");
    CHECK(train_store_replica_count(store, embed) == 2, "two replicas");
    const TrainSync sync = train_store_required_sync(store, embed);
    CHECK((sync & TRAIN_SYNC_GRAD_MERGE) && (sync & TRAIN_SYNC_UPDATE_BCAST),
          "a replicated parameter needs both the gradient merge and the broadcast (got %d)",
          (int)sync);
    CHECK(train_store_set_replica_devices(store, embed, 2, devices) == TRAIN_OK &&
              train_store_replica_count(store, embed) == 2,
          "recording the placement twice is idempotent");
    CHECK(train_store_destroy(store) == TRAIN_OK, "destroy");
}

static void test_teacher_forcing(void) {
    /* A 6-token sequence: two prompt tokens, four response tokens. The mask marks the
     * response; the label of query t is token t+1. */
    const int tokens[6] = {10, 11, 12, 13, 14, 15};
    const uint8_t mask[6] = {0, 0, 1, 1, 1, 1};
    const int64_t positions[6] = {100, 101, 102, 103, 104, 105};
    struct TrainForcedPosition out[8];

    const int count = train_plan_teacher_forcing(6, tokens, NULL, mask, positions, 1, out, 8);
    CHECK(count == 4, "four response positions are selected (got %d)", count);
    /* Query 1 predicts token 2, which is the first response token; query 5 predicts
     * nothing inside the sequence. */
    CHECK(out[0].query == 1 && out[0].label == 12 && out[0].position == 101,
          "the first selected position is the first response token (q=%d l=%d p=%d)", out[0].query,
          out[0].label, out[0].position);
    CHECK(out[3].query == 4 && out[3].label == 15, "the last selected position predicts the last token");

    /* Without a mask every position but the last `shift` is selected. */
    const int all = train_plan_teacher_forcing(6, tokens, NULL, NULL, positions, 1, out, 8);
    CHECK(all == 5, "without a mask, tokens-shift positions are selected (got %d)", all);
    CHECK(out[0].query == 0 && out[0].label == 11, "the shift is applied at the start too");

    /* A forced label overrides the self-supervised target. */
    const int labels[6] = {0, 0, 77, 0, 0, 0};
    const int forced = train_plan_teacher_forcing(6, tokens, labels, mask, NULL, 1, out, 8);
    CHECK(forced == 4, "forced labels do not change the selection (got %d)", forced);
    CHECK(out[0].label == 77, "the forced label is used at the selected position (got %d)",
          out[0].label);
    CHECK(out[0].position == 1, "without positions the query position is used");

    /* A different shift changes which labels exist. */
    struct TrainForcedPosition shifted[8];
    const int two = train_plan_teacher_forcing(6, tokens, NULL, mask, NULL, 2, shifted, 8);
    /* shift=2 leaves queries 0..3; their targets are 2..5, all of them response
     * positions, so all four are selected. */
    CHECK(two == 4, "shift=2 selects four positions (got %d)", two);
    CHECK(shifted[0].query == 0 && shifted[0].label == 12, "shift=2 labels query 0 with token 2");

    /* A caller can ask for the count without a buffer, and is refused a short one. */
    CHECK(train_plan_teacher_forcing(6, tokens, NULL, mask, NULL, 1, NULL, 0) == 4,
          "a null output buffer reports the required count");
    CHECK(train_plan_teacher_forcing(6, tokens, NULL, mask, NULL, 1, out, 2) < 0,
          "a short output buffer is refused rather than truncated");
    CHECK(train_plan_teacher_forcing(6, tokens, NULL, mask, NULL, 0, out, 8) < 0,
          "shift must be at least one");
}

int main(int argc, char **argv) {
    test_tying();
    test_lifetime_and_publication();
    test_schedule_and_replicas();
    test_teacher_forcing();

    /* The tying gate against the real model that has the tie: Qwen3-4B maps its
     * lmHead role onto the embedding tensor, so one logical parameter has two
     * readers and a publication must reach both. */
    const char *descriptor_path = argc > 1 ? argv[1] : "descriptors/qwen3-4b.json";
    FILE *file = fopen(descriptor_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "train_test: cannot open %s\n", descriptor_path);
        return EXIT_FAILURE;
    }
    static char text[1 << 20];
    const size_t bytes = fread(text, 1, sizeof(text) - 1, file);
    fclose(file);
    text[bytes] = '\0';
    struct ModelDesc desc;
    char err[256];
    if (model_desc_parse(text, &desc, err, sizeof(err)) != 0) {
        fprintf(stderr, "train_test: descriptor parse failed: %s\n", err);
        return EXIT_FAILURE;
    }
    /* A real model has layers x roles parameters (Qwen3-4B: 3 global + 11 per-layer
     * roles x 36 layers = 399), so these are static rather than a 64-entry stack
     * array -- an undersized buffer here is a silent overwrite, and the check below
     * would then be testing garbage. */
    static struct TrainParamSpec specs[TRAIN_MAX_PARAMS];
    static const char *templates[TRAIN_MAX_PARAMS];
    static const char *names[TRAIN_MAX_PARAMS];
    int count = 0;
    for (int i = 0; i < desc.role_count; ++i) {
        const int role = desc.role_ids[i];
        const int global = role == ROLE_EMBED || role == ROLE_LM_HEAD || role == ROLE_FINAL_NORM;
        /* A global role is one parameter; a per-layer role exists once per layer. */
        const int layers = global ? 1 : desc.num_layers;
        for (int layer = 0; layer < layers; ++layer) {
            specs[count] = (struct TrainParamSpec){global ? -1 : layer, role, 1024, 0, 1};
            templates[count] = desc.role_templates[i];
            names[count] = model_desc_role_name(role);
            ++count;
        }
    }
    TrainStore *store = train_store_create(specs, templates, names, count);
    CHECK(store != NULL, "the real descriptor builds a store (%d parameters)", count);
    if (store != NULL) {
        const int embed = train_store_logical_of(store, -1, ROLE_EMBED);
        const int lm_head = train_store_logical_of(store, -1, ROLE_LM_HEAD);
        CHECK(embed >= 0 && lm_head >= 0, "both global roles are present");
        CHECK(embed == lm_head, "Qwen3-4B's lmHead ties to its embedding (logical %d vs %d)", embed,
              lm_head);
        CHECK(train_store_alias_count(store, embed) == 2, "the tie has two aliases");
        CHECK(train_store_slot_buffer_count(store, embed, TRAIN_SLOT_COMPUTE) == 2,
              "a publication writes both readers of the tied weight");
        /* The per-layer roles are not tied to each other. */
        const int q0 = train_store_logical_of(store, 0, ROLE_ATTN_Q);
        const int q1 = train_store_logical_of(store, 1, ROLE_ATTN_Q);
        CHECK(q0 >= 0 && q1 >= 0 && q0 != q1, "the same role in two layers is two parameters");
        const int expected_logical = count - 1;
        CHECK(train_store_logical_count(store) == expected_logical,
              "tying removes exactly one logical parameter (%d vs %d)",
              train_store_logical_count(store), expected_logical);
        CHECK(train_store_destroy(store) == TRAIN_OK, "destroy the descriptor store");
    }

    if (g_failures != 0) {
        fprintf(stderr, "train_test: %d check(s) failed\n", g_failures);
        return EXIT_FAILURE;
    }
    printf("train_test: tying, frozen parameters, borrow/update lifetime, derived-copy "
           "publication, the accumulation schedule, replicas and the teacher-forcing plan "
           "all hold\n");
    return EXIT_SUCCESS;
}
