/**
 * quant_manifest_test.c - the CPU gate for the `weights.manifest.json` reader (plan Q2's
 * "a loader can validate the artifact before a GPU sees it").
 *
 * Two halves, and the second half is the one that matters: a reader is only a validator if it
 * can fail, so beside a valid fixture every refusal gets its own case - a version it does not
 * speak, a format block that is not the frozen format, a group axis that is not K, a K that is
 * not a whole number of groups, extents that disagree with the shape or with the pair's
 * members, a malformed digest, a duplicate key, a truncated document and a precision map that
 * does not match the entries. Each mutant is produced by textual surgery on the fixture, and a
 * mutant whose anchor is not found is itself a failure - otherwise a renamed fixture would turn
 * the whole set of refusals into silent passes.
 *
 * The fixture is hand-written rather than produced by the converter, so the reader is tested
 * against the *schema* and not against one writer's current output. `--manifest PATH [--root
 * DIR]` additionally reads a real sidecar (and, with `--root`, verifies one recorded artifact's
 * digest), which is how tests/test_quantization_converter.py ties the two together.
 *
 * Run: quant_manifest_test [--manifest weights.manifest.json [--root DIR]]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "quant_manifest.h"
#include "sha256.h"

static int g_failures = 0;

static int check(int condition, const char *message) {
    if (condition) return 1;
    printf("quant_manifest_test: FAIL: %s\n", message);
    ++g_failures;
    return 0;
}

/* The fixture: 2-space indent and sorted keys, the way the converter's encoder writes it. */
static const char *kFixture =
    "{\n"
    "  \"converter\": {\n"
    "    \"name\": \"scripts/quantize_weights.py\",\n"
    "    \"version\": 1\n"
    "  },\n"
    "  \"descriptor\": {\n"
    "    \"path\": \"/tmp/qwen3-dense-synth.json\",\n"
    "    \"sha256\": \"1111111111111111111111111111111111111111111111111111111111111111\"\n"
    "  },\n"
    "  \"entries\": [\n"
    "    {\n"
    "      \"checkpoint_tensor\": \"model.layers.0.mlp.gate_proj.weight\",\n"
    "      \"group\": 128,\n"
    "      \"group_axis\": \"k\",\n"
    "      \"layer\": 0,\n"
    "      \"local_shape\": [\n"
    "        256,\n"
    "        256\n"
    "      ],\n"
    "      \"logical_shape\": [\n"
    "        256,\n"
    "        256\n"
    "      ],\n"
    "      \"packed\": {\n"
    "        \"bytes\": 32768,\n"
    "        \"dtype\": \"u8\",\n"
    "        \"file\": \"artifacts/mlpGate_layer0.packed\",\n"
    "        \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
    "      },\n"
    "      \"role\": \"mlpGate\",\n"
    "      \"role_index\": 3,\n"
    "      \"scales\": {\n"
    "        \"bytes\": 1024,\n"
    "        \"count\": 512,\n"
    "        \"dtype\": \"bf16\",\n"
    "        \"file\": \"artifacts/mlpGate_layer0.scales\",\n"
    "        \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
    "      }\n"
    "    },\n"
    "    {\n"
    "      \"checkpoint_tensor\": \"model.layers.0.mlp.up_proj.weight\",\n"
    "      \"group\": 128,\n"
    "      \"group_axis\": \"k\",\n"
    "      \"layer\": 0,\n"
    "      \"local_shape\": [\n"
    "        128,\n"
    "        256\n"
    "      ],\n"
    "      \"logical_shape\": [\n"
    "        128,\n"
    "        256\n"
    "      ],\n"
    "      \"packed\": {\n"
    "        \"bytes\": 16384,\n"
    "        \"dtype\": \"u8\",\n"
    "        \"file\": \"artifacts/mlpUp_layer0.packed\",\n"
    "        \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
    "      },\n"
    "      \"role\": \"mlpUp\",\n"
    "      \"role_index\": 4,\n"
    "      \"scales\": {\n"
    "        \"bytes\": 512,\n"
    "        \"count\": 256,\n"
    "        \"dtype\": \"bf16\",\n"
    "        \"file\": \"artifacts/mlpUp_layer0.scales\",\n"
    "        \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
    "      }\n"
    "    }\n"
    "  ],\n"
    "  \"format\": {\n"
    "    \"abi_version\": 1,\n"
    "    \"arithmetic\": \"csrc/linear_weight.cpp (the Q0 reference)\",\n"
    "    \"group\": 128,\n"
    "    \"invalid_code\": -8,\n"
    "    \"name\": \"int4_symmetric_group_bf16_scale\",\n"
    "    \"packed_dtype\": \"u8\",\n"
    "    \"packing\": \"two_twos_complement_nibbles_per_byte_lower_k_index_in_low_nibble\",\n"
    "    \"qmax\": 7,\n"
    "    \"qmin\": -7,\n"
    "    \"scale_dtype\": \"bf16\",\n"
    "    \"zero_point\": 0\n"
    "  },\n"
    "  \"manifest_version\": 1,\n"
    "  \"pairs\": [\n"
    "    {\n"
    "      \"group\": 128,\n"
    "      \"layer\": 0,\n"
    "      \"logical_shape\": [\n"
    "        384,\n"
    "        256\n"
    "      ],\n"
    "      \"members\": [\n"
    "        {\n"
    "          \"role\": \"mlpGate\",\n"
    "          \"role_index\": 3\n"
    "        },\n"
    "        {\n"
    "          \"role\": \"mlpUp\",\n"
    "          \"role_index\": 4\n"
    "        }\n"
    "      ],\n"
    "      \"name\": \"mlpGateUp\",\n"
    "      \"packed\": {\n"
    "        \"bytes\": 49152,\n"
    "        \"dtype\": \"u8\",\n"
    "        \"file\": \"artifacts/mlpGateUp_layer0.packed\",\n"
    "        \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
    "      },\n"
    "      \"rule\": \"row concatenation: gate rows then up rows\",\n"
    "      \"scales\": {\n"
    "        \"bytes\": 1536,\n"
    "        \"count\": 768,\n"
    "        \"dtype\": \"bf16\",\n"
    "        \"file\": \"artifacts/mlpGateUp_layer0.scales\",\n"
    "        \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
    "      }\n"
    "    }\n"
    "  ],\n"
    "  \"precision_map\": [\n"
    "    {\n"
    "      \"checkpoint_tensor\": \"model.layers.0.mlp.gate_proj.weight\",\n"
    "      \"layer\": 0,\n"
    "      \"precision\": \"int4\",\n"
    "      \"role\": \"mlpGate\"\n"
    "    },\n"
    "    {\n"
    "      \"checkpoint_tensor\": \"model.layers.0.mlp.up_proj.weight\",\n"
    "      \"layer\": 0,\n"
    "      \"precision\": \"int4\",\n"
    "      \"role\": \"mlpUp\"\n"
    "    },\n"
    "    {\n"
    "      \"checkpoint_tensor\": \"model.embed_tokens.weight\",\n"
    "      \"layer\": 0,\n"
    "      \"precision\": \"bf16\",\n"
    "      \"role\": \"embed\"\n"
    "    }\n"
    "  ],\n"
    "  \"quantization_error\": {\n"
    "    \"max_abs\": 0.00622559,\n"
    "    \"note\": \"measured against the source\",\n"
    "    \"rms\": 0.00235134\n"
    "  },\n"
    "  \"source\": {\n"
    "    \"model_dir\": \"/tmp/synth-qwen3-dense\",\n"
    "    \"files\": []\n"
    "  }\n"
    "}\n";

/* Replace the first occurrence of `needle`, or return NULL when it is not there. A missing
 * anchor is a failure of the *test*, never a silently unapplied mutation. */
static char *replace_first(const char *src, const char *needle, const char *replacement) {
    const char *at = strstr(src, needle);
    if (at == NULL) return NULL;
    const size_t head = (size_t)(at - src);
    const size_t tail_len = strlen(at + strlen(needle));
    char *out = (char *)malloc(head + strlen(replacement) + tail_len + 1);
    if (out == NULL) return NULL;
    memcpy(out, src, head);
    memcpy(out + head, replacement, strlen(replacement));
    memcpy(out + head + strlen(replacement), at + strlen(needle), tail_len);
    out[head + strlen(replacement) + tail_len] = '\0';
    return out;
}

static void expect_status(const char *label, const char *json, QuantManifestStatus want) {
    struct QuantManifest manifest;
    char err[512];
    const QuantManifestStatus got = quant_manifest_parse(json, &manifest, err, sizeof(err));
    if (got == want) {
        printf("  PASS %s -> %d\n", label, (int)got);
        return;
    }
    printf("  FAIL %s -> %d (wanted %d): %s\n", label, (int)got, (int)want,
           quant_manifest_last_error());
    ++g_failures;
}

static void expect_mutant(const char *label, const char *needle, const char *replacement,
                          QuantManifestStatus want) {
    char *mutant = replace_first(kFixture, needle, replacement);
    if (mutant == NULL) {
        printf("  FAIL %s: the anchor %s is not in the fixture\n", label, needle);
        ++g_failures;
        return;
    }
    expect_status(label, mutant, want);
    free(mutant);
}

/* Corrupt, in place, the digest recorded for the artifact whose file is `file` - one character,
 * same length - so the mutant tests "not hex" rather than "not 64 bytes". The anchor is the
 * artifact's own file name because the document holds several `sha256` fields (the descriptor's
 * among them) and corrupting the wrong one would test nothing. */
static char *corrupt_digest_of(const char *src, const char *file, char replacement) {
    const char *artifact = strstr(src, file);
    if (artifact == NULL) return NULL;
    const char *anchor = strstr(artifact, "\"sha256\": \"");
    if (anchor == NULL) return NULL;
    const size_t at = (size_t)(anchor - src) + strlen("\"sha256\": \"");
    char *out = (char *)malloc(strlen(src) + 1);
    if (out == NULL) return NULL;
    memcpy(out, src, strlen(src) + 1);
    out[at] = replacement;
    return out;
}

static void test_fixture(void) {
    printf("=== the fixture parses and its claims are re-derived ===\n");
    struct QuantManifest manifest;
    char err[512];
    const QuantManifestStatus status =
        quant_manifest_parse(kFixture, &manifest, err, sizeof(err));
    if (!check(status == QUANT_MANIFEST_OK, quant_manifest_last_error())) return;
    check(manifest.manifest_version == 1, "manifest version");
    check(manifest.group == 128 && manifest.qmin == -7 && manifest.qmax == 7,
          "the format block");
    check(manifest.zero_point == 0 && manifest.invalid_code == -8, "zero point and reserved code");
    check(strcmp(manifest.converter_name, "scripts/quantize_weights.py") == 0 &&
              manifest.converter_version == 1,
          "the converter identity");
    check(manifest.entry_count == 2, "two entries");
    check(manifest.pair_count == 1, "one pair");
    check(manifest.precision_int4_cells == 2 && manifest.precision_bf16_cells == 1,
          "the precision map's cells");

    const int gate = quant_manifest_find_entry(&manifest, 0, "mlpGate");
    check(gate >= 0, "mlpGate is found by (layer, role)");
    check(quant_manifest_find_entry(&manifest, 1, "mlpGate") < 0, "a missing layer is not found");
    const int up = quant_manifest_find_entry(&manifest, 0, "mlpUp");
    check(up >= 0, "mlpUp is found");
    if (gate >= 0) {
        struct LinearWeightLayout layout;
        check(quant_manifest_layout(&manifest.entries[gate], &layout, err, sizeof(err)) ==
                  QUANT_MANIFEST_OK,
              "the gate entry has a layout");
        check(layout.packed_bytes == 256ll * 256ll / 2 && layout.scale_count == 256ll * 2,
              "the extents come from the format, not from the manifest's byte count");
        check(manifest.entries[gate].role_index == 3, "the role index is read");
    }
    if (up >= 0) {
        check(manifest.entries[up].n == 128, "the up entry's rows");
    }
    const int pair = quant_manifest_find_pair(&manifest, "mlpGateUp", 0);
    check(pair >= 0, "the F1 pair is found by (name, layer)");
    if (pair >= 0) {
        check(manifest.pairs[pair].n == 384 && manifest.pairs[pair].k == 256,
              "the pair's rows are its members' rows");
        check(strcmp(manifest.pairs[pair].member_role[0], "mlpGate") == 0 &&
                  strcmp(manifest.pairs[pair].member_role[1], "mlpUp") == 0,
              "the pair's member order is recorded");
    }
    check(quant_manifest_find_pair(&manifest, "mlpGateUp", 7) < 0, "a missing pair layer is absent");
}

static void test_refusals(void) {
    printf("=== the refusals ===\n");
    /* The format block must be the frozen format, and the manifest version must be one this
     * reader speaks. */
    expect_mutant("a version this reader does not speak", "\"manifest_version\": 1",
                  "\"manifest_version\": 2", QUANT_MANIFEST_ERR_FORMAT);
    expect_mutant("a qmin that is not the frozen one", "\"qmin\": -7", "\"qmin\": -6",
                  QUANT_MANIFEST_ERR_FORMAT);
    /* The layout refuses a foreign group, a K that is not a whole number of groups and a row
     * count that is not a whole number of the tile - through linear_layout_init, the same
     * function the quantizer uses, so these are the format's errors rather than this reader's
     * opinion. */
    expect_mutant("a foreign group", "\"group\": 128,\n      \"group_axis\"",
                  "\"group\": 64,\n      \"group_axis\"", QUANT_MANIFEST_ERR_SHAPE);
    expect_mutant("a K that is not a whole number of groups", "\"logical_shape\": [\n        256,\n        256\n      ]",
                  "\"logical_shape\": [\n        256,\n        192\n      ]", QUANT_MANIFEST_ERR_SHAPE);
    expect_mutant("a group axis that is not K", "\"group_axis\": \"k\"", "\"group_axis\": \"n\"",
                  QUANT_MANIFEST_ERR_FORMAT);
    /* A byte count or a scale count that disagrees with the shape is not corrected here. */
    expect_mutant("packed bytes that disagree with the shape", "\"bytes\": 32768",
                  "\"bytes\": 32769", QUANT_MANIFEST_ERR_SHAPE);
    expect_mutant("a scale count that disagrees with the shape", "\"count\": 512",
                  "\"count\": 511", QUANT_MANIFEST_ERR_SHAPE);
    /* The digests are the only tie to bytes this reader does not re-derive, so a malformed one
     * is refused rather than carried. */
    {
        char *digest_mutant = corrupt_digest_of(kFixture, "artifacts/mlpGate_layer0.packed", 'z');
        if (digest_mutant == NULL) {
            printf("  FAIL a malformed digest: no digest in the fixture\n");
            ++g_failures;
        } else {
            expect_status("a malformed digest", digest_mutant, QUANT_MANIFEST_ERR_FORMAT);
            free(digest_mutant);
        }
    }
    expect_mutant("a packed dtype that is not u8",
                  "\"dtype\": \"u8\"", "\"dtype\": \"f16\"", QUANT_MANIFEST_ERR_FORMAT);
    /* The precision map and the entries have to describe the same cells. */
    expect_mutant("an entry with no int4 precision cell", "\"precision\": \"int4\"",
                  "\"precision\": \"bf16\"", QUANT_MANIFEST_ERR_SHAPE);
    /* A pair is only trustworthy if it is its members' rows. */
    expect_mutant("a pair whose rows are not its members'", "\"logical_shape\": [\n        384,\n        256\n      ]",
                  "\"logical_shape\": [\n        383,\n        256\n      ]", QUANT_MANIFEST_ERR_SHAPE);
    expect_mutant("a pair member that is not an entry", "\"role\": \"mlpUp\"",
                  "\"role\": \"mlpDown\"", QUANT_MANIFEST_ERR_SHAPE);
    /* The document has to be the JSON the schema requires. */
    expect_mutant("a duplicate key", "\"manifest_version\": 1",
                  "\"manifest_version\": 1,\n  \"manifest_version\": 1",
                  QUANT_MANIFEST_ERR_SYNTAX);
    expect_mutant("a missing required section", "\"entries\":", "\"entries_x\":",
                  QUANT_MANIFEST_ERR_SYNTAX);

    char *truncated = (char *)malloc(strlen(kFixture) - 20 + 1);
    if (truncated == NULL) {
        printf("  FAIL a truncated document: out of memory\n");
        ++g_failures;
    } else {
        memcpy(truncated, kFixture, strlen(kFixture) - 20);
        truncated[strlen(kFixture) - 20] = '\0';
        expect_status("a truncated document", truncated, QUANT_MANIFEST_ERR_SYNTAX);
        free(truncated);
    }
    expect_status("an empty document", "", QUANT_MANIFEST_ERR_SYNTAX);
    expect_status("a document that is not an object", "[]", QUANT_MANIFEST_ERR_SYNTAX);
}

/* The reader has to tie the bytes to the digest the manifest recorded, and refuse a file that
 * is the wrong size, the wrong dtype or the wrong content. */
static void test_artifact_read(void) {
    printf("=== an artifact is read, sized and hashed ===\n");
    char dir[] = "/tmp/quant-manifest-XXXXXX";
    if (mkdtemp(dir) == NULL) {
        printf("  FAIL cannot create a temporary directory\n");
        ++g_failures;
        return;
    }
    uint8_t payload[64];
    for (size_t i = 0; i < sizeof(payload); ++i) payload[i] = (uint8_t)(i * 7u + 1u);
    char path[256];
    snprintf(path, sizeof(path), "%s/one.packed", dir);
    FILE *handle = fopen(path, "wb");
    if (handle == NULL) {
        printf("  FAIL cannot write %s\n", path);
        ++g_failures;
        return;
    }
    fwrite(payload, 1, sizeof(payload), handle);
    fclose(handle);

    struct QuantArtifact artifact;
    memset(&artifact, 0, sizeof(artifact));
    strcpy(artifact.file, "one.packed");
    strcpy(artifact.dtype, "u8");
    artifact.bytes = (long long)sizeof(payload);
    artifact.count = -1;
    sha256_hex(payload, sizeof(payload), artifact.sha256);

    uint8_t *got = NULL;
    size_t got_len = 0;
    char err[512];
    QuantManifestStatus status =
        quant_artifact_read(dir, &artifact, "u8", &got, &got_len, err, sizeof(err));
    check(status == QUANT_MANIFEST_OK, quant_manifest_last_error());
    if (status == QUANT_MANIFEST_OK) {
        check(got_len == sizeof(payload) && memcmp(got, payload, sizeof(payload)) == 0,
              "the payload is the file's bytes");
        free(got);
    }

    /* A corrupted file is a hash failure, not silently accepted. */
    payload[3] ^= 0xFFu;
    handle = fopen(path, "wb");
    fwrite(payload, 1, sizeof(payload), handle);
    fclose(handle);
    got = NULL;
    got_len = 0;
    status = quant_artifact_read(dir, &artifact, "u8", &got, &got_len, err, sizeof(err));
    check(status == QUANT_MANIFEST_ERR_HASH, "a corrupted payload is refused");
    check(got == NULL, "a refused read hands back no buffer");

    /* A size the manifest did not record is refused before the digest is even computed. */
    struct QuantArtifact wrong = artifact;
    wrong.bytes = (long long)sizeof(payload) + 1;
    status = quant_artifact_read(dir, &wrong, "u8", &got, &got_len, err, sizeof(err));
    check(status == QUANT_MANIFEST_ERR_IO, "a size the manifest did not record is refused");

    /* And the dtype has to be the artifact's. */
    status = quant_artifact_read(dir, &artifact, "bf16", &got, &got_len, err, sizeof(err));
    check(status == QUANT_MANIFEST_ERR_FORMAT, "a different expected dtype is refused");

    /* A missing file is an I/O error rather than an empty artifact. */
    struct QuantArtifact absent = artifact;
    strcpy(absent.file, "two.packed");
    status = quant_artifact_read(dir, &absent, "u8", &got, &got_len, err, sizeof(err));
    check(status == QUANT_MANIFEST_ERR_IO, "a missing file is refused");

    snprintf(path, sizeof(path), "%s/one.packed", dir);
    remove(path);
    rmdir(dir);
}

/* The real sidecar, when one is pointed at: this is how the converter's gate ties its output to
 * this reader, so a manifest the converter writes has to be one the engine's reader accepts. */
static int test_real_manifest(const char *path, const char *root) {
    printf("=== a sidecar written by the converter ===\n");
    FILE *handle = fopen(path, "rb");
    if (handle == NULL) {
        printf("quant_manifest_test: FAIL: cannot open %s\n", path);
        return ++g_failures;
    }
    fseek(handle, 0, SEEK_END);
    const long size = ftell(handle);
    fseek(handle, 0, SEEK_SET);
    char *json = (char *)malloc((size_t)size + 1);
    if (json == NULL || fread(json, 1, (size_t)size, handle) != (size_t)size) {
        fclose(handle);
        free(json);
        printf("quant_manifest_test: FAIL: cannot read %s\n", path);
        return ++g_failures;
    }
    fclose(handle);
    json[size] = '\0';

    struct QuantManifest manifest;
    char err[512];
    const QuantManifestStatus status = quant_manifest_parse(json, &manifest, err, sizeof(err));
    if (!check(status == QUANT_MANIFEST_OK, quant_manifest_last_error())) {
        free(json);
        return g_failures;
    }
    printf("  parsed: %d entries, %d pairs, %d int4 cells, %d bf16 cells\n", manifest.entry_count,
           manifest.pair_count, manifest.precision_int4_cells, manifest.precision_bf16_cells);
    check(manifest.entry_count > 0, "a real sidecar has entries");
    check(manifest.precision_int4_cells == manifest.entry_count,
          "every entry is an int4 cell of the precision map");

    if (root != NULL) {
        uint8_t *bytes = NULL;
        size_t length = 0;
        const QuantManifestStatus read =
            quant_artifact_read(root, &manifest.entries[0].packed, "u8", &bytes, &length, err,
                                sizeof(err));
        if (check(read == QUANT_MANIFEST_OK, quant_manifest_last_error())) {
            printf("  verified %s against its recorded digest (%zu bytes)\n",
                   manifest.entries[0].packed.file, length);
            free(bytes);
        }
    }
    free(json);
    return g_failures;
}

int main(int argc, char **argv) {
    const char *manifest = NULL;
    const char *root = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) manifest = argv[++i];
        else if (strcmp(argv[i], "--root") == 0 && i + 1 < argc) root = argv[++i];
    }
    test_fixture();
    test_refusals();
    test_artifact_read();
    if (manifest != NULL) test_real_manifest(manifest, root);
    if (g_failures != 0) {
        printf("quant_manifest_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("quant_manifest_test: PASS (the fixture parses and its claims are re-derived, every "
           "refusal fires, and an artifact is tied to its digest)\n");
    return 0;
}
