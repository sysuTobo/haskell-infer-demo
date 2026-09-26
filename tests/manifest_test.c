/**
 * manifest_test.c - CPU gate for the execution manifest.
 *
 * Three things are pinned here:
 *
 *   1. the SHA-256 implementation, against the published FIPS 180-4 vectors
 *      (including the two-block padding case and a 1 MB stream);
 *   2. canonical format determinism: the same inputs format to identical bytes,
 *      and the three identities are stable across repeated queries;
 *   3. the identity matrix the plan's Stage 0 gate asks for: a semantic or
 *      numerical change moves the appropriate id and nothing else, a
 *      deployment-only change moves only deployment_id, and a provenance or
 *      weight change moves no identity at all.
 *
 * The Python companion (tests/test_manifest_hashes.py) re-derives every digest
 * from the parsed JSON with hashlib and re-checks the same matrix, so the
 * canonical encoding is verified by a second implementation rather than
 * self-reported. `--emit` / `--emit-variant` exist only to hand documents to
 * that companion.
 */
#include "manifest.h"
#include "model_desc.h"
#include "sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            ++failures;                                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);          \
            printf(__VA_ARGS__);                                \
            printf("\n");                                       \
        }                                                       \
    } while (0)

/* A small hybrid dense model: layer 0 GDN, layer 1 full attention, dense MLP.
 * embed and lmHead deliberately share a template so the tied-parameter relation
 * the semantic identity must record is exercised. */
static const char kDescriptor[] =
    "{"
    "\"desc_version\":1,"
    "\"family\":\"test_hybrid\","
    "\"model_type\":\"test_hybrid_text\","
    "\"num_layers\":2,"
    "\"hidden_size\":256,"
    "\"intermediate_size\":512,"
    "\"vocab_size\":100,"
    "\"rms_eps\":1e-06,"
    "\"max_position_embeddings\":1024,"
    "\"max_seq_len\":512,"
    "\"num_heads\":4,"
    "\"num_kv_heads\":2,"
    "\"head_dim\":64,"
    "\"rotary_dim\":32,"
    "\"rotary_theta\":1000000.0,"
    "\"norm_style\":\"gemma\","
    "\"attn_qk_norm\":true,"
    "\"attn_output_gate\":true,"
    "\"q_gate_interleave\":true,"
    "\"gdn_conv_dim\":512,"
    "\"gdn_value_dim\":256,"
    "\"gdn_num_v_heads\":8,"
    "\"gdn_num_k_heads\":4,"
    "\"gdn_head_dim\":32,"
    "\"gdn_conv_kernel\":4,"
    "\"fla_chunk_size\":64,"
    "\"max_chunk\":64,"
    "\"moe_num_experts\":0,"
    "\"moe_top_k\":0,"
    "\"moe_intermediate_size\":0,"
    "\"moe_router_scoring\":\"softmax\","
    "\"moe_norm_topk_prob\":false,"
    "\"moe_num_shared_experts\":0,"
    "\"moe_shared_intermediate_size\":0,"
    "\"moe_routed_scaling_factor\":1.0,"
    "\"moe_shared_gate_scalar\":false,"
    "\"eos_tokens\":[2,3],"
    "\"layer_mixers\":[\"gdn\",\"full_attn\"],"
    "\"layer_ffns\":[\"dense\",\"dense\"],"
    "\"role_names\":[\"embed\",\"lmHead\",\"finalNorm\",\"inputNorm\",\"postNorm\","
    "\"mlpGate\",\"mlpUp\",\"mlpDown\",\"attnQ\",\"attnK\",\"attnV\",\"attnO\","
    "\"attnQNorm\",\"attnKNorm\",\"gdnQkv\",\"gdnZ\",\"gdnA\",\"gdnB\",\"gdnConv1d\","
    "\"gdnDtBias\",\"gdnALog\",\"gdnOut\",\"gdnNorm\"],"
    "\"role_templates\":[\"embed.weight\",\"embed.weight\",\"final_norm.weight\","
    "\"layers.%d.input_norm.weight\",\"layers.%d.post_norm.weight\","
    "\"layers.%d.mlp.gate.weight\",\"layers.%d.mlp.up.weight\",\"layers.%d.mlp.down.weight\","
    "\"layers.%d.attn.q.weight\",\"layers.%d.attn.k.weight\",\"layers.%d.attn.v.weight\","
    "\"layers.%d.attn.o.weight\",\"layers.%d.attn.q_norm.weight\","
    "\"layers.%d.attn.k_norm.weight\",\"layers.%d.gdn.qkv.weight\",\"layers.%d.gdn.z.weight\","
    "\"layers.%d.gdn.a.weight\",\"layers.%d.gdn.b.weight\",\"layers.%d.gdn.conv.weight\","
    "\"layers.%d.gdn.dt_bias\",\"layers.%d.gdn.a_log\",\"layers.%d.gdn.out.weight\","
    "\"layers.%d.gdn.norm.weight\"]"
    "}";

static struct ModelDesc g_desc;
static struct ManifestBuildInfo g_build;
static struct ManifestDevice g_devices[2];
static struct ManifestWeights g_weights;
static struct ManifestInputs g_in;
static int g_layer_device[2] = {0, 1};
static int g_device_ordinals[2] = {0, 1};
static char g_descriptor_sha256[SHA256_HEX_LEN];
static char g_desc_text[4096];

static void set_hex(char out[SHA256_HEX_LEN], char fill) {
    for (int i = 0; i < 64; ++i) out[i] = fill;
    out[64] = '\0';
}

static void setup(void) {
    char err[256];
    if (model_desc_parse(kDescriptor, &g_desc, err, sizeof(err)) != 0) {
        printf("FAIL: the test descriptor does not parse: %s\n", err);
        exit(2);
    }
    if (model_desc_validate(&g_desc, err, sizeof(err)) != 0) {
        printf("FAIL: the test descriptor does not validate: %s\n", err);
        exit(2);
    }
    const int desc_bytes = model_desc_format(&g_desc, g_desc_text, sizeof(g_desc_text));
    if (desc_bytes <= 0) {
        printf("FAIL: cannot format the test descriptor\n");
        exit(2);
    }
    sha256_hex(g_desc_text, (size_t)desc_bytes, g_descriptor_sha256);

    memset(&g_build, 0, sizeof(g_build));
    snprintf(g_build.engine_version, sizeof(g_build.engine_version), "0.1.0.0-test");
    snprintf(g_build.git_commit, sizeof(g_build.git_commit), "0123456789abcdef0123456789abcdef01234567");
    snprintf(g_build.cuda_toolkit, sizeof(g_build.cuda_toolkit), "12.9");
    snprintf(g_build.cuda_archs, sizeof(g_build.cuda_archs), "86;89;90a;90-virtual");
    snprintf(g_build.triton_archs, sizeof(g_build.triton_archs), "86;89;90");
    snprintf(g_build.triton_version, sizeof(g_build.triton_version), "3.4.0");
    snprintf(g_build.fla_version, sizeof(g_build.fla_version), "0.5.2");
    set_hex(g_build.flashinfer_header_sha256, 'a');
    set_hex(g_build.nvcc_flags_sha256, 'b');
    set_hex(g_build.generated_kernels_sha256, 'c');

    memset(g_devices, 0, sizeof(g_devices));
    g_devices[0].cuda_ordinal = 0;
    snprintf(g_devices[0].name, sizeof(g_devices[0].name), "Test Device Zero");
    snprintf(g_devices[0].compute_capability, sizeof(g_devices[0].compute_capability), "8.6");
    snprintf(g_devices[0].uuid, sizeof(g_devices[0].uuid), "GPU-00000000-0000-0000-0000-000000000001");
    snprintf(g_devices[0].kernel_path, sizeof(g_devices[0].kernel_path), "sass");
    snprintf(g_devices[0].triton_cubin_arch, sizeof(g_devices[0].triton_cubin_arch), "86");
    g_devices[1].cuda_ordinal = 1;
    snprintf(g_devices[1].name, sizeof(g_devices[1].name), "Test Device One");
    snprintf(g_devices[1].compute_capability, sizeof(g_devices[1].compute_capability), "8.9");
    snprintf(g_devices[1].uuid, sizeof(g_devices[1].uuid), "GPU-00000000-0000-0000-0000-000000000002");
    snprintf(g_devices[1].kernel_path, sizeof(g_devices[1].kernel_path), "sass");
    snprintf(g_devices[1].triton_cubin_arch, sizeof(g_devices[1].triton_cubin_arch), "89");

    memset(&g_weights, 0, sizeof(g_weights));
    set_hex(g_weights.parameter_manifest_sha256, 'd');
    g_weights.tensor_count = 42;
    set_hex(g_weights.shards_sha256, 'e');
    g_weights.content_hash_present = 0;

    memset(&g_in, 0, sizeof(g_in));
    g_in.desc = &g_desc;
    g_in.descriptor_sha256 = g_descriptor_sha256;
    g_in.descriptor_bytes = desc_bytes;
    g_in.build = &g_build;
    g_in.cuda_runtime_version = "12.9";
    g_in.cuda_driver_version = "580.65.06";
    g_in.cublas_version = "12.9.0";
    g_in.devices = g_devices;
    g_in.device_count = 2;
    g_in.weights = &g_weights;
    g_in.replicated = 0;
    g_in.ep_size = 1;
    g_in.declared_tp_rank = 0;
    g_in.declared_ep_rank = 0;
    g_in.device_ordinals = g_device_ordinals;
    g_in.layer_device = g_layer_device;
    g_in.num_layers = 2;
    g_in.regions = NULL;     /* the committed registry */
    g_in.region_count = 0;
    g_in.sampling = manifest_default_sampling();   /* the committed greedy policy */
}

/* ------------------------------------------------------------------ */
/* SHA-256 vectors                                                    */
/* ------------------------------------------------------------------ */

static void test_sha256(void) {
    char hex[SHA256_HEX_LEN];
    sha256_hex("", 0, hex);
    CHECK(strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0,
          "sha256(\"\") = %s", hex);
    sha256_hex("abc", 3, hex);
    CHECK(strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0,
          "sha256(\"abc\") = %s", hex);
    const char *two_block = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    sha256_hex(two_block, strlen(two_block), hex);
    CHECK(strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") == 0,
          "sha256(two-block FIPS vector) = %s", hex);

    /* A 1 MB stream exercises the incremental path across many blocks; the
     * expected digest is the FIPS vector for one million 'a' characters. */
    struct sha256_ctx ctx;
    unsigned char digest[32];
    sha256_init(&ctx);
    char chunk[1000];
    memset(chunk, 'a', sizeof(chunk));
    for (int i = 0; i < 1000; ++i) sha256_update(&ctx, chunk, sizeof(chunk));
    sha256_final(&ctx, digest);
    sha256_to_hex(digest, hex);
    CHECK(strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") == 0,
          "sha256(1e6 * 'a') = %s", hex);
}

/* ------------------------------------------------------------------ */
/* Identity matrix                                                    */
/* ------------------------------------------------------------------ */

struct Identities {
    char semantic[SHA256_HEX_LEN];
    char numerical[SHA256_HEX_LEN];
    char deployment[SHA256_HEX_LEN];
};

static void format_manifest(const struct ManifestInputs *in, char *buf, int buf_len) {
    const int written = manifest_format(in, buf, buf_len);
    if (written < 0) {
        printf("FAIL: manifest_format returned %d\n", written);
        ++failures;
        buf[0] = '\0';
    }
}

static void read_identities(const char *buf, struct Identities *out) {
    const char *keys[3] = {"\"semantic_id\":\"", "\"numerical_policy_id\":\"",
                           "\"deployment_id\":\""};
    char *slots[3] = {out->semantic, out->numerical, out->deployment};
    for (int i = 0; i < 3; ++i) {
        const char *at = strstr(buf, keys[i]);
        if (at == NULL) {
            printf("FAIL: %s is missing from the manifest\n", keys[i]);
            ++failures;
            continue;
        }
        at += strlen(keys[i]);
        memcpy(slots[i], at, 64);
        slots[i][64] = '\0';
    }
}

static void test_identity_matrix(void) {
    static char baseline_text[ENGINE_MANIFEST_MAX];
    struct Identities base;
    format_manifest(&g_in, baseline_text, sizeof(baseline_text));
    read_identities(baseline_text, &base);
    /* The document's own version is what lets a consumer tell a manifest from a
     * legacy capture; omitting it made the engine's manifest unparseable once. */
    CHECK(strstr(baseline_text, "\"manifest_version\":1") != NULL,
          "the manifest does not carry its own version");
    /* The contract fixes the type of every region flag and reference: an integer
     * where a boolean belongs made the Haskell consumer refuse the manifest. */
    CHECK(strstr(baseline_text, "\"rng_dependency\":false") != NULL,
          "the region rng_dependency flag is not a JSON boolean");
    CHECK(strstr(baseline_text, "\"desc_version\":1") != NULL,
          "the descriptor reference does not carry its version");

    /* Determinism: the same inputs must format to identical bytes twice. */
    static char again_text[ENGINE_MANIFEST_MAX];
    format_manifest(&g_in, again_text, sizeof(again_text));
    CHECK(strcmp(baseline_text, again_text) == 0,
          "two identical queries produced different bytes");
    {
        /* A too-small buffer must be refused (distinctly, so the caller can
         * size it) rather than truncated into an unverifiable document. */
        char small[1024];
        CHECK(manifest_format(&g_in, small, (int)sizeof(small)) == -2,
              "a too-small buffer was not refused as a capacity error");
    }

    static char variant_text[ENGINE_MANIFEST_MAX];
    struct Identities variant;

    /* Semantic change: an architecture dimension. */
    g_desc.hidden_size = 320;
    format_manifest(&g_in, variant_text, sizeof(variant_text));
    read_identities(variant_text, &variant);
    CHECK(strcmp(base.semantic, variant.semantic) != 0, "hidden_size did not move semantic_id");
    CHECK(strcmp(base.numerical, variant.numerical) == 0, "hidden_size moved numerical_policy_id");
    CHECK(strcmp(base.deployment, variant.deployment) == 0, "hidden_size moved deployment_id");
    g_desc.hidden_size = 256;

    /* A constant that describes both the function and its realization is
     * projected into both identities; that is deliberate, not a leak. */
    g_desc.rms_eps = 1e-05;
    format_manifest(&g_in, variant_text, sizeof(variant_text));
    read_identities(variant_text, &variant);
    CHECK(strcmp(base.semantic, variant.semantic) != 0, "rms_eps did not move semantic_id");
    CHECK(strcmp(base.numerical, variant.numerical) != 0,
          "rms_eps did not move numerical_policy_id (the projection is broken)");
    CHECK(strcmp(base.deployment, variant.deployment) == 0, "rms_eps moved deployment_id");
    g_desc.rms_eps = 1e-06;

    /* Numerical change: the chunking schedule. */
    g_desc.max_chunk = 32;
    format_manifest(&g_in, variant_text, sizeof(variant_text));
    read_identities(variant_text, &variant);
    CHECK(strcmp(base.semantic, variant.semantic) == 0, "max_chunk moved semantic_id");
    CHECK(strcmp(base.numerical, variant.numerical) != 0, "max_chunk did not move numerical_policy_id");
    CHECK(strcmp(base.deployment, variant.deployment) == 0, "max_chunk moved deployment_id");
    g_desc.max_chunk = 64;

    /* Numerical change: the sampler's transform. The concrete temperature and seed
     * are per-request replay data, but which transform and arithmetic ran is
     * numerical policy, so a capture taken under a different one must not compare
     * equal. */
    {
        const struct ManifestSampling *saved = g_in.sampling;
        struct ManifestSampling edited;
        memcpy(&edited, manifest_default_sampling(), sizeof(edited));
        edited.transform = "categorical_softmax_cdf";
        g_in.sampling = &edited;
        format_manifest(&g_in, variant_text, sizeof(variant_text));
        read_identities(variant_text, &variant);
        CHECK(strcmp(base.semantic, variant.semantic) == 0, "a sampler change moved semantic_id");
        CHECK(strcmp(base.numerical, variant.numerical) != 0,
              "a sampler change did not move numerical_policy_id");
        CHECK(strcmp(base.deployment, variant.deployment) == 0,
              "a sampler change moved deployment_id");
        g_in.sampling = saved;
    }

    /* Numerical change: weight-only INT4 FFN operands (plan Q2). A packed weight set is a
     * different *realization* of the same function on the same placement, so only the
     * numerical policy may move - if this moved semantic_id, a quantized model would look like a
     * different architecture, and if it moved deployment_id, it would look like a different
     * placement. */
    {
        const int saved = g_in.weight_only_int4;
        g_in.weight_only_int4 = 1;
        format_manifest(&g_in, variant_text, sizeof(variant_text));
        read_identities(variant_text, &variant);
        CHECK(strcmp(base.semantic, variant.semantic) == 0,
              "weight-only INT4 moved semantic_id");
        CHECK(strcmp(base.numerical, variant.numerical) != 0,
              "weight-only INT4 did not move numerical_policy_id");
        CHECK(strcmp(base.deployment, variant.deployment) == 0,
              "weight-only INT4 moved deployment_id");
        g_in.weight_only_int4 = saved;
    }

    /* Numerical change: the region binding table. */
    {
        int count = 0;
        const struct ManifestRegion *committed = manifest_default_regions(&count);
        struct ManifestRegion *edited = (struct ManifestRegion *)malloc(
            sizeof(struct ManifestRegion) * (size_t)count);
        memcpy(edited, committed, sizeof(struct ManifestRegion) * (size_t)count);
        edited[0].determinism = "unverified";
        const struct ManifestRegion *saved = g_in.regions;
        const int saved_count = g_in.region_count;
        g_in.regions = edited;
        g_in.region_count = count;
        format_manifest(&g_in, variant_text, sizeof(variant_text));
        read_identities(variant_text, &variant);
        CHECK(strcmp(base.semantic, variant.semantic) == 0, "a region edit moved semantic_id");
        CHECK(strcmp(base.numerical, variant.numerical) != 0,
              "a region edit did not move numerical_policy_id");
        CHECK(strcmp(base.deployment, variant.deployment) == 0, "a region edit moved deployment_id");
        g_in.regions = saved;
        g_in.region_count = saved_count;
        free(edited);
    }

    /* Deployment-only change: device ordinals, the layer assignment and the
     * shard plan are placement. None of them changes what arithmetic runs, so
     * neither identity may move. */
    g_device_ordinals[0] = 4;
    g_device_ordinals[1] = 5;
    g_layer_device[0] = 1;
    g_layer_device[1] = 0;
    g_desc.role_shards[8] = ENGINE_SHARD_OUT_HEADS;   /* attnQ */
    format_manifest(&g_in, variant_text, sizeof(variant_text));
    read_identities(variant_text, &variant);
    CHECK(strcmp(base.semantic, variant.semantic) == 0, "a deployment change moved semantic_id");
    CHECK(strcmp(base.numerical, variant.numerical) == 0,
          "a deployment change moved numerical_policy_id");
    CHECK(strcmp(base.deployment, variant.deployment) != 0,
          "a deployment change did not move deployment_id");
    g_device_ordinals[0] = 0;
    g_device_ordinals[1] = 1;
    g_layer_device[0] = 0;
    g_layer_device[1] = 1;
    g_desc.role_shards[8] = ENGINE_SHARD_NONE;

    /* Replicated placement is *not* a deployment-only variation: splitting the
     * heads and reducing across devices changes the reduction order, which is
     * why TP/EP results are not bit-identical to the layer split. Both the
     * numerical policy and the deployment must move. */
    g_in.replicated = 1;
    g_in.ep_size = 2;
    format_manifest(&g_in, variant_text, sizeof(variant_text));
    read_identities(variant_text, &variant);
    CHECK(strcmp(base.semantic, variant.semantic) == 0, "replication moved semantic_id");
    CHECK(strcmp(base.numerical, variant.numerical) != 0,
          "replication did not move numerical_policy_id (the collective and the "
          "expert-merge reduction are part of the numerical policy)");
    CHECK(strcmp(base.deployment, variant.deployment) != 0,
          "replication did not move deployment_id");
    g_in.replicated = 0;
    g_in.ep_size = 1;

    /* Provenance change: a different build and runtime is not a different
     * identity, it is a different capture that strict admission must refuse. */
    {
        char saved_commit[64];
        const char *saved_runtime = g_in.cuda_runtime_version;
        memcpy(saved_commit, g_build.git_commit, sizeof(saved_commit));
        snprintf(g_build.git_commit, sizeof(g_build.git_commit), "ffffffffffffffffffffffffffffffffffffffff");
        g_in.cuda_runtime_version = "13.0";
        format_manifest(&g_in, variant_text, sizeof(variant_text));
        read_identities(variant_text, &variant);
        CHECK(strcmp(base.semantic, variant.semantic) == 0, "provenance moved semantic_id");
        CHECK(strcmp(base.numerical, variant.numerical) == 0, "provenance moved numerical_policy_id");
        CHECK(strcmp(base.deployment, variant.deployment) == 0, "provenance moved deployment_id");
        CHECK(strstr(variant_text, "ffffffffffffffffffffffffffffffffffffffff") != NULL,
              "the changed git commit is not in the manifest");
        memcpy(g_build.git_commit, saved_commit, sizeof(saved_commit));
        g_in.cuda_runtime_version = saved_runtime;
    }

    /* A weight change moves no identity: weights are their own identity. */
    {
        const char *saved = g_weights.parameter_manifest_sha256;
        char saved_copy[SHA256_HEX_LEN];
        memcpy(saved_copy, saved, sizeof(saved_copy));
        set_hex(g_weights.parameter_manifest_sha256, '9');
        format_manifest(&g_in, variant_text, sizeof(variant_text));
        read_identities(variant_text, &variant);
        CHECK(strcmp(base.semantic, variant.semantic) == 0, "a weight change moved semantic_id");
        CHECK(strcmp(base.numerical, variant.numerical) == 0, "a weight change moved numerical_policy_id");
        CHECK(strcmp(base.deployment, variant.deployment) == 0, "a weight change moved deployment_id");
        CHECK(strstr(variant_text, "9999999999999999999999999999999999999999999999999999999999999999") != NULL,
              "the changed weight digest is not in the manifest");
        memcpy(g_weights.parameter_manifest_sha256, saved_copy, sizeof(saved_copy));
    }

    /* Unprintable input is refused instead of being escaped inconsistently. */
    {
        static char small[ENGINE_MANIFEST_MAX];
        const char *saved = g_devices[0].name;
        snprintf(g_devices[0].name, sizeof(g_devices[0].name), "bad\tname");
        CHECK(manifest_format(&g_in, small, (int)sizeof(small)) < 0,
              "a non-printable-ASCII string was not refused");
        snprintf(g_devices[0].name, sizeof(g_devices[0].name), "%s", saved);
    }
}

/* ------------------------------------------------------------------ */
/* Emitting documents for the cross-language verifier                 */
/* ------------------------------------------------------------------ */

static int apply_variant(const char *name) {
    if (strcmp(name, "semantic") == 0) { g_desc.hidden_size = 320; return 0; }
    if (strcmp(name, "projected_constant") == 0) { g_desc.rms_eps = 1e-05; return 0; }
    if (strcmp(name, "numerical") == 0) { g_desc.max_chunk = 32; return 0; }
    if (strcmp(name, "sampling") == 0) {
        static struct ManifestSampling edited;
        memcpy(&edited, manifest_default_sampling(), sizeof(edited));
        edited.transform = "categorical_softmax_cdf";
        g_in.sampling = &edited;
        return 0;
    }
    if (strcmp(name, "regions") == 0) return 1;   /* needs the copied registry */
    if (strcmp(name, "deployment") == 0) {
        g_device_ordinals[0] = 4;
        g_device_ordinals[1] = 5;
        g_layer_device[0] = 1;
        g_layer_device[1] = 0;
        g_desc.role_shards[8] = ENGINE_SHARD_OUT_HEADS;
        return 0;
    }
    if (strcmp(name, "provenance") == 0) {
        snprintf(g_build.git_commit, sizeof(g_build.git_commit), "ffffffffffffffffffffffffffffffffffffffff");
        g_in.cuda_runtime_version = "13.0";
        return 0;
    }
    if (strcmp(name, "weights") == 0) {
        set_hex(g_weights.parameter_manifest_sha256, '9');
        return 0;
    }
    return -1;
}

static int emit(const char *path, const char *variant) {
    struct ManifestRegion *edited = NULL;
    if (variant != NULL) {
        if (strcmp(variant, "baseline") != 0) {
            if (apply_variant(variant) != 0) {
                int count = 0;
                const struct ManifestRegion *committed = manifest_default_regions(&count);
                edited = (struct ManifestRegion *)malloc(
                    sizeof(struct ManifestRegion) * (size_t)count);
                memcpy(edited, committed, sizeof(struct ManifestRegion) * (size_t)count);
                edited[0].determinism = "unverified";
                g_in.regions = edited;
                g_in.region_count = count;
            }
        }
    }
    static char text[ENGINE_MANIFEST_MAX];
    const int written = manifest_format(&g_in, text, sizeof(text));
    free(edited);
    if (written < 0) {
        fprintf(stderr, "cannot format the manifest (%d)\n", written);
        return 1;
    }
    FILE *handle = fopen(path, "wb");
    if (handle == NULL) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    const size_t wrote = fwrite(text, 1, (size_t)written, handle);
    fclose(handle);
    if (wrote != (size_t)written) {
        fprintf(stderr, "short write to %s\n", path);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    setup();
    if (argc >= 3 && strcmp(argv[1], "--emit") == 0) return emit(argv[2], NULL);
    if (argc >= 4 && strcmp(argv[1], "--emit-variant") == 0) return emit(argv[2], argv[3]);

    test_sha256();
    test_identity_matrix();

    if (failures != 0) {
        printf("manifest_test: %d failure(s)\n", failures);
        return 1;
    }
    printf("manifest_test: SHA-256 vectors, canonical determinism and the identity matrix pass\n");
    return 0;
}
