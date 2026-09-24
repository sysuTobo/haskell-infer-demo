/**
 * manifest.c - Canonical execution manifest: identities, provenance, regions.
 *
 * See csrc/include/manifest.h for the contract and docs/manifest-contract.md for
 * the field ownership table. Two properties matter for correctness:
 *
 *   - the encoding is canonical (keys sorted by byte value, no whitespace,
 *     integers bare, non-integer constants as decimal strings, printable ASCII
 *     only), so a consumer in another language can re-derive every digest from
 *     the parsed document by re-serializing it the same way instead of trusting
 *     this emitter; and
 *   - nothing is defaulted silently. A fact the build or runtime does not know
 *     is reported as "unknown"/"unavailable"/"unsupported" and a strict
 *     comparison has to refuse it, because an unestablished numerical setting
 *     cannot support an exact claim.
 */
#include "manifest.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MJ_BLOCK_MAX 65536
#define MJ_ERR_MAX 128

/* ------------------------------------------------------------------ */
/* Canonical JSON writer                                              */
/* ------------------------------------------------------------------ */

struct MjBuf {
    char *buf;
    int cap;
    int used;
    int failed;
    char err[MJ_ERR_MAX];
};

static void mj_fail(struct MjBuf *b, const char *msg) {
    if (b->failed) return;
    b->failed = 1;
    snprintf(b->err, MJ_ERR_MAX, "%s", msg);
}

static void mj_put(struct MjBuf *b, const char *text) {
    if (b->failed) return;
    size_t n = strlen(text);
    if (b->used + (int)n + 1 > b->cap) {
        mj_fail(b, "manifest block buffer is too small");
        return;
    }
    memcpy(b->buf + b->used, text, n);
    b->used += (int)n;
    b->buf[b->used] = '\0';
}

static void mj_putf(struct MjBuf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int written = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (written < 0 || written >= (int)sizeof(tmp)) {
        mj_fail(b, "manifest field does not fit its format");
        return;
    }
    mj_put(b, tmp);
}

static void mj_init(struct MjBuf *b, char *buf, int cap) {
    memset(b, 0, sizeof(*b));
    b->buf = buf;
    b->cap = cap;
    buf[0] = '\0';
}

/* Strings are restricted to printable ASCII, so this encoder and any other
 * JSON writer agree byte for byte on the only escaping that can occur. */
static void mj_str(struct MjBuf *b, const char *value) {
    mj_put(b, "\"");
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (*p < 0x20 || *p > 0x7e) {
            mj_fail(b, "manifest strings must be printable ASCII");
            return;
        }
        if (*p == '"' || *p == '\\') {
            char escaped[3] = {'\\', (char)*p, '\0'};
            mj_put(b, escaped);
        } else {
            char one[2] = {(char)*p, '\0'};
            mj_put(b, one);
        }
    }
    mj_put(b, "\"");
}

static void mj_raw(struct MjBuf *b, const char *text) { mj_put(b, text); }

static void mj_int(struct MjBuf *b, long long value) { mj_putf(b, "%lld", value); }

static void mj_bool(struct MjBuf *b, int value) { mj_put(b, value ? "true" : "false"); }

static void mj_null(struct MjBuf *b) { mj_put(b, "null"); }

/* A non-integer constant is emitted as a JSON *string*: decimal-to-JSON float
 * formatting is the one place two languages legitimately disagree, and an
 * identity must not depend on a printf convention. Rejects non-finite values
 * rather than emitting a literal another parser would refuse. */
static void mj_numstr(struct MjBuf *b, double value) {
    if (!isfinite(value)) {
        mj_fail(b, "manifest constant is not finite");
        return;
    }
    mj_putf(b, "\"%.17g\"", value);
}

/* Emits the separating comma unless this is the first member of an object. */
static void mj_member(struct MjBuf *b, int *first, const char *key) {
    if (!*first) mj_put(b, ",");
    *first = 0;
    mj_str(b, key);
    mj_put(b, ":");
}

static void mj_kv_int(struct MjBuf *b, int *first, const char *key, long long value) {
    mj_member(b, first, key);
    mj_int(b, value);
}

static void mj_kv_str(struct MjBuf *b, int *first, const char *key, const char *value) {
    mj_member(b, first, key);
    mj_str(b, value);
}

static void mj_kv_bool(struct MjBuf *b, int *first, const char *key, int value) {
    mj_member(b, first, key);
    mj_bool(b, value);
}

static void mj_kv_numstr(struct MjBuf *b, int *first, const char *key, double value) {
    mj_member(b, first, key);
    mj_numstr(b, value);
}

static void mj_kv_int_array(struct MjBuf *b, int *first, const char *key,
                            const int *values, int count) {
    mj_member(b, first, key);
    mj_put(b, "[");
    for (int i = 0; i < count; ++i) {
        if (i) mj_put(b, ",");
        mj_int(b, values[i]);
    }
    mj_put(b, "]");
}

static void mj_kv_str_array(struct MjBuf *b, int *first, const char *key,
                            const char *const *values, int count) {
    mj_member(b, first, key);
    mj_put(b, "[");
    for (int i = 0; i < count; ++i) {
        if (i) mj_put(b, ",");
        mj_str(b, values[i]);
    }
    mj_put(b, "]");
}

/* ------------------------------------------------------------------ */
/* Small helpers                                                      */
/* ------------------------------------------------------------------ */

const char *manifest_or_unknown(const char *value) {
    return (value == NULL || value[0] == '\0') ? MANIFEST_UNKNOWN : value;
}

/* The effective FP32 constant, which is what the kernels actually read. The
 * declared double is a semantic field; this is the numerical realization. */
static double effective_float(double value) { return (double)(float)value; }

static int desc_has_layer(const struct ModelDesc *d, int predicate, int mixer) {
    for (int i = 0; i < d->num_layers; ++i)
        if (d->layer_mixers[i] == mixer) return 1;
    (void)predicate;
    return 0;
}

static int desc_has_ffn(const struct ModelDesc *d, int ffn) {
    for (int i = 0; i < d->num_layers; ++i)
        if (d->layer_ffns[i] == ffn) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Region registry                                                    */
/* ------------------------------------------------------------------ */

/* The committed binding table is part of the numerical identity: a region whose
 * implementation, case coverage or determinism status changed is a different
 * numerical policy even when every dimension is unchanged.
 *
 * determinism is deliberately conservative. "deterministic" is claimed only
 * where the implementation is a single elementwise pass with no cross-thread
 * reduction to reorder. Anything involving a library tiling decision (FlashInfer,
 * cuBLAS, the AOT FLA cubins, the cross-device collective) is "unverified": the
 * whole-model golden capture shows bitwise repeatability for one fixed
 * configuration, which is not the same as an established reduction order, and
 * the plan's Stage 2 experiments are what would establish it. Backward regions do
 * not exist yet, so they are recorded as not_implemented rather than assumed. */
static const struct ManifestRegion kRegions[] = {
    {"embedding", "prefill,decode,train_forward", "kernels/embedding.cu row gather",
     "deterministic", "one thread per output element, no reduction", 0},
    {"rmsnorm", "prefill,decode,train_forward",
     "FlashInfer RMSNorm (plain or gemma weight+1)", "unverified",
     "library-internal reduction tree not established", 0},
    {"per_head_norm", "prefill,decode", "FlashInfer per-head Q/K RMSNorm", "unverified",
     "library-internal reduction tree not established", 0},
    {"rope", "prefill,decode,train_forward", "FlashInfer RoPE position kernel",
     "deterministic", "per-element rotation, no cross-thread reduction", 0},
    {"q_gate_split", "prefill,decode", "engine de-interleave of the fused Q/gate projection",
     "deterministic", "elementwise permutation", 0},
    {"attention_core", "prefill,decode,train_forward,recompute",
     "FlashInfer single prefill, split-KV disabled, LSE not returned", "unverified",
     "null workspace fixes the no-split-KV mode only, not the internal tiling", 0},
    {"attention_output_gate", "prefill,decode", "sigmoid gate multiply", "deterministic",
     "elementwise", 0},
    {"gemm_bf16", "prefill,decode,backward", "cuBLAS BF16 in, FP32 accumulate, BF16 out",
     "unverified", "cuBLAS algorithm and workspace selection is not pinned", 0},
    {"gemm_fp32_lmhead", "prefill,decode", "cuBLAS BF16 in, FP32 out", "unverified",
     "cuBLAS algorithm and workspace selection is not pinned", 0},
    {"residual_add", "prefill,decode,train_forward", "engine residual addition",
     "deterministic", "elementwise", 0},
    {"silu_mul", "prefill,decode,train_forward", "fused SiLU(gate) * up", "deterministic",
     "elementwise", 0},
    {"gdn_conv1d", "prefill,decode", "AOT causal_conv1d kernel", "unverified",
     "library scan order not established", 0},
    {"gdn_prepare", "prefill,decode", "engine Q/K L2 norm, head expansion, a/b/A_log/dt_bias",
     "unverified", "cross-thread reduction over the head dimension", 0},
    {"gdn_core", "chunked_prefill,recurrent_prefill,decode,tail1",
     "AOT FLA chunkwise cubin; recurrent path for tokens==1", "unverified",
     "the chunked and recurrent decompositions do not share a reduction schedule", 0},
    {"gdn_gated_norm", "prefill,decode", "fused FLA RMSNormGated (raw weight)", "unverified",
     "in-block reduction over the head dimension", 0},
    {"moe_router", "prefill,decode", "engine top-k router (softmax or sigmoid)", "deterministic",
     "per-row scoring with a fixed top-k scan", 0},
    {"moe_experts", "prefill,decode", "per-expert cuBLAS GEMM", "unverified",
     "cuBLAS algorithm selection per expert shape", 0},
    {"moe_combine", "prefill,decode", "FP32 accumulation of routed outputs, one rounding",
     "deterministic", "fixed token-order accumulation in FP32", 0},
    {"moe_ep_merge", "ep_gt_1", "FP32 routed partial, single all-reduce", "unverified",
     "cross-device reduction order not established", 0},
    {"mla_core", "prefill,decode", "engine tiled MLA attention, shared-memory bounded",
     "unverified", "in-block reduction over the KV length", 0},
    {"collective_allreduce", "replicated_tp,replicated_ep",
     "engine all-reduce (BF16 activations, FP32 routed partial)", "unverified",
     "cross-device tree order not established", 0},
    {"logits_softmax_gather", "prefill,decode", "host FP32 argmax over the final row",
     "deterministic", "exact host comparison, no reduction reorder", 0},
    {"sampler_softmax_cdf", "not_implemented",
     "proposed host binary64 softmax/CDF (plan T0-T4)", "not_implemented",
     "one RNG word per sampled token once implemented", 1},
    {"backward", "not_implemented", "no backward region is implemented", "not_implemented",
     "the trainer plan stages 3-4 define this", 0},
};

const struct ManifestRegion *manifest_default_regions(int *count) {
    if (count != NULL) *count = (int)(sizeof(kRegions) / sizeof(kRegions[0]));
    return kRegions;
}

static const struct ManifestSampling kSampling = {
    "greedy", "argmax_lowest_token_id", 1, "host_fp32_logits_exact_compare", "none"};

const struct ManifestSampling *manifest_default_sampling(void) { return &kSampling; }

/* ------------------------------------------------------------------ */
/* Identity blocks                                                    */
/* ------------------------------------------------------------------ */

/* semantic: what the architecture *means*. Keys are sorted by byte value; see
 * docs/manifest-contract.md for why each field is here and not in the numerical
 * policy or the deployment. */
static void emit_semantic(struct MjBuf *b, const struct ModelDesc *d) {
    int first = 1;
    mj_put(b, "{");
    mj_kv_bool(b, &first, "attn_output_gate", d->attn_output_gate);
    mj_kv_bool(b, &first, "attn_qk_norm", d->attn_qk_norm);
    mj_kv_int_array(b, &first, "eos_tokens", d->eos_tokens, d->eos_count);
    mj_kv_str(b, &first, "family", d->family);
    mj_kv_int(b, &first, "gdn_conv_dim", d->gdn_conv_dim);
    mj_kv_int(b, &first, "gdn_conv_kernel", d->gdn_conv_kernel);
    mj_kv_str(b, &first, "gdn_decay_order", "decay_before_prediction");
    mj_kv_int(b, &first, "gdn_head_dim", d->gdn_head_dim);
    mj_kv_int(b, &first, "gdn_num_k_heads", d->gdn_num_k_heads);
    mj_kv_int(b, &first, "gdn_num_v_heads", d->gdn_num_v_heads);
    mj_kv_str(b, &first, "gdn_q_scale", "one_over_sqrt_head_dim");
    mj_kv_int(b, &first, "gdn_value_dim", d->gdn_value_dim);
    mj_kv_int(b, &first, "head_dim", d->head_dim);
    mj_kv_int(b, &first, "hidden_size", d->hidden_size);
    mj_kv_int(b, &first, "intermediate_size", d->intermediate_size);
    {
        const char *names[ENGINE_MAX_LAYERS];
        for (int i = 0; i < d->num_layers; ++i)
            names[i] = model_desc_ffn_name(d->layer_ffns[i]);
        mj_kv_str_array(b, &first, "layer_ffns", names, d->num_layers);
        for (int i = 0; i < d->num_layers; ++i)
            names[i] = model_desc_mixer_name(d->layer_mixers[i]);
        mj_kv_str_array(b, &first, "layer_mixers", names, d->num_layers);
    }
    mj_kv_int(b, &first, "max_position_embeddings", d->max_position_embeddings);
    mj_kv_int(b, &first, "max_seq_len", d->max_seq_len);
    mj_kv_int(b, &first, "mla_kv_lora_rank", d->mla_kv_lora_rank);
    mj_kv_int(b, &first, "mla_qk_nope_head_dim", d->mla_qk_nope_head_dim);
    mj_kv_int(b, &first, "mla_qk_rope_head_dim", d->mla_qk_rope_head_dim);
    mj_kv_int(b, &first, "mla_v_head_dim", d->mla_v_head_dim);
    mj_kv_str(b, &first, "model_type", d->model_type);
    mj_kv_int(b, &first, "moe_intermediate_size", d->moe_intermediate_size);
    mj_kv_bool(b, &first, "moe_norm_topk_prob", d->moe_norm_topk_prob);
    mj_kv_int(b, &first, "moe_num_experts", d->moe_num_experts);
    mj_kv_int(b, &first, "moe_num_shared_experts", d->moe_num_shared_experts);
    mj_kv_numstr(b, &first, "moe_routed_scaling_factor", d->moe_routed_scaling_factor);
    mj_kv_str(b, &first, "moe_router_scoring", d->moe_router_scoring);
    mj_kv_bool(b, &first, "moe_shared_gate_scalar", d->moe_shared_gate_scalar);
    mj_kv_int(b, &first, "moe_shared_intermediate_size", d->moe_shared_intermediate_size);
    mj_kv_int(b, &first, "moe_top_k", d->moe_top_k);
    mj_kv_str(b, &first, "norm_style",
              strcmp(d->norm_style, "gemma") == 0 ? "gemma_weight_plus_one" : "plain_rms");
    mj_kv_int(b, &first, "num_heads", d->num_heads);
    mj_kv_int(b, &first, "num_kv_heads", d->num_kv_heads);
    mj_kv_int(b, &first, "num_layers", d->num_layers);
    mj_kv_bool(b, &first, "q_gate_interleave", d->q_gate_interleave);
    mj_kv_numstr(b, &first, "rms_eps_declared", d->rms_eps);
    {
        const char *templates[ENGINE_MAX_ROLES];
        const char *names[ENGINE_MAX_ROLES];
        char tied_buf[ENGINE_MAX_ROLES][64];
        const char *tied[ENGINE_MAX_ROLES];
        int tied_count = 0;
        for (int i = 0; i < d->role_count; ++i) {
            templates[i] = d->role_templates[i];
            names[i] = model_desc_role_name(d->role_ids[i]);
        }
        for (int i = 0; i < d->role_count; ++i) {
            for (int j = i + 1; j < d->role_count; ++j) {
                if (strcmp(d->role_templates[i], d->role_templates[j]) != 0) continue;
                snprintf(tied_buf[tied_count], sizeof(tied_buf[0]), "%s=%s",
                         names[i], names[j]);
                tied[tied_count] = tied_buf[tied_count];
                ++tied_count;
            }
        }
        mj_kv_str_array(b, &first, "role_templates", templates, d->role_count);
        mj_kv_str_array(b, &first, "roles", names, d->role_count);
        mj_kv_str(b, &first, "rope_convention", "partial_split_half");
        mj_kv_int(b, &first, "rotary_dim", d->rotary_dim);
        mj_kv_numstr(b, &first, "rotary_theta", d->rotary_theta);
        mj_kv_str_array(b, &first, "tied_roles", tied, tied_count);
    }
    mj_kv_int(b, &first, "vocab_size", d->vocab_size);
    mj_put(b, "}");
}

/* numerical_policy: how the semantics are realized. */
static void emit_numerical(struct MjBuf *b, const struct ManifestInputs *in,
                           const char *regions_sha256) {
    const struct ModelDesc *d = in->desc;
    const int has_gdn = desc_has_layer(d, 0, ENGINE_MIXER_GDN);
    const int has_mla = desc_has_layer(d, 0, ENGINE_MIXER_MLA);
    const int has_moe = desc_has_ffn(d, ENGINE_FFN_MOE);
    int first = 1;
    mj_put(b, "{");
    mj_kv_str(b, &first, "attention_split_kv", "disabled_null_workspace");
    mj_kv_str(b, &first, "attention_workspace", "flashinfer_single_prefill_no_lse");
    mj_kv_str(b, &first, "backward", "not_implemented");
    mj_kv_str(b, &first, "cast_boundaries",
              "bf16_storage_and_activations_fp32_accumulation");
    mj_kv_str(b, &first, "collective",
              in->replicated ? "engine_allreduce_bf16_activation" : "not_used");
    mj_kv_numstr(b, &first, "effective_rms_eps", effective_float(d->rms_eps));
    mj_kv_numstr(b, &first, "effective_rope_theta", effective_float(d->rotary_theta));
    mj_kv_int(b, &first, "fla_chunk_size", has_gdn ? d->fla_chunk_size : 0);
    mj_kv_str(b, &first, "fusion",
              "flashinfer_attention;silu_mul;gdn_gated_norm;dense_gate_up_separate");
    mj_kv_str(b, &first, "gdn_recurrent_impl",
              has_gdn ? "fla_aot_chunkwise_cubin_plus_recurrent_tokens1" : "not_used");
    mj_kv_str(b, &first, "gemm_algorithm_policy", "cublas_default_heuristic_unpinned");
    mj_kv_str(b, &first, "gemm_output_type", "bf16_default_fp32_for_lm_head");
    mj_kv_str(b, &first, "lm_head_output_type", "fp32");
    mj_kv_int(b, &first, "max_chunk", d->max_chunk);
    mj_kv_str(b, &first, "mla_impl",
              has_mla ? "engine_tiled_mla_shared_memory_bounded" : "not_used");
    mj_kv_str(b, &first, "moe_combine",
              has_moe ? "fp32_accumulate_single_rounding" : "not_used");
    mj_kv_str(b, &first, "moe_ep_merge",
              in->ep_size > 1 ? "fp32_partial_single_allreduce" : "not_used");
    mj_kv_str(b, &first, "norm_impl",
              strcmp(d->norm_style, "gemma") == 0 ? "flashinfer_gemma_rmsnorm"
                                                 : "flashinfer_rmsnorm");
    mj_kv_str(b, &first, "regions_sha256", manifest_or_unknown(regions_sha256));
    mj_put(b, "}");
}

/* deployment: where it ran and how memory moved. */
static void emit_deployment(struct MjBuf *b, const struct ManifestInputs *in) {
    int first = 1;
    char transfer[128];
    const char *shards[ENGINE_MAX_ROLES];
    for (int i = 0; i < in->desc->role_shard_count; ++i)
        shards[i] = model_desc_shard_name(in->desc->role_shards[i]);
    snprintf(transfer, sizeof(transfer), "cross_device_copies_with_completion_events%s%s",
             in->replicated ? ";sub_layer_allreduce" : "",
             in->ep_size > 1 ? ";expert_partial_fp32_merge" : "");
    mj_put(b, "{");
    mj_kv_str(b, &first, "allocations",
              "engine_owned_bf16_weights;fp32_derived_gdn_norm;per_request_kv_gdn_state");
    mj_kv_int(b, &first, "declared_ep_rank", in->declared_ep_rank);
    mj_kv_int(b, &first, "declared_tp_rank", in->declared_tp_rank);
    mj_kv_int_array(b, &first, "devices", in->device_ordinals, in->device_count);
    mj_kv_int(b, &first, "ep_size", in->ep_size);
    mj_kv_int_array(b, &first, "layer_device", in->layer_device, in->num_layers);
    mj_kv_str(b, &first, "placement", in->replicated ? "replicated_tp_ep" : "layer_split");
    mj_kv_str_array(b, &first, "role_shards", shards, in->desc->role_shard_count);
    mj_kv_str(b, &first, "transfer", transfer);
    mj_put(b, "}");
}

/* The region binding table. Its digest enters the numerical policy, so a change
 * to a binding is a numerical change even for an unchanged model. */
static void emit_regions(struct MjBuf *b, const struct ManifestRegion *regions,
                         int count) {
    mj_put(b, "[");
    for (int i = 0; i < count; ++i) {
        const struct ManifestRegion *r = &regions[i];
        int first = 1;
        if (i) mj_put(b, ",");
        mj_put(b, "{");
        mj_kv_str(b, &first, "cases", r->cases);
        mj_kv_str(b, &first, "determinism", r->determinism);
        mj_kv_str(b, &first, "implementation", r->implementation);
        mj_kv_str(b, &first, "mechanism", r->mechanism);
        mj_kv_str(b, &first, "region", r->region);
        /* A boolean, not 0/1: the contract fixes the type, and a consumer that
         * reads it as a boolean must not be handed an integer. */
        mj_kv_bool(b, &first, "rng_dependency", r->rng_dependency);
        mj_put(b, "}");
    }
    mj_put(b, "]");
}

static void emit_weights(struct MjBuf *b, const struct ManifestWeights *w) {
    int first = 1;
    mj_put(b, "{");
    mj_member(b, &first, "content_sha256");
    if (w->content_hash_present) mj_str(b, w->content_sha256);
    else mj_null(b);
    mj_kv_str(b, &first, "parameter_manifest_sha256",
              manifest_or_unknown(w->parameter_manifest_sha256));
    mj_kv_str(b, &first, "shards_sha256", manifest_or_unknown(w->shards_sha256));
    mj_kv_int(b, &first, "tensor_count", w->tensor_count);
    mj_put(b, "}");
}

static void emit_build(struct MjBuf *b, const struct ManifestBuildInfo *info) {
    int first = 1;
    struct ManifestBuildInfo unknown;
    memset(&unknown, 0, sizeof(unknown));
    if (info == NULL) info = &unknown;
    mj_put(b, "{");
    mj_kv_str(b, &first, "cuda_archs", manifest_or_unknown(info->cuda_archs));
    mj_kv_str(b, &first, "cuda_toolkit", manifest_or_unknown(info->cuda_toolkit));
    mj_kv_str(b, &first, "engine_build", manifest_or_unknown(info->engine_version));
    mj_kv_str(b, &first, "fla_version", manifest_or_unknown(info->fla_version));
    mj_kv_str(b, &first, "flashinfer_header_sha256",
              manifest_or_unknown(info->flashinfer_header_sha256));
    mj_kv_str(b, &first, "generated_kernels_sha256",
              manifest_or_unknown(info->generated_kernels_sha256));
    mj_kv_str(b, &first, "git_commit", manifest_or_unknown(info->git_commit));
    mj_kv_str(b, &first, "nvcc_flags_sha256",
              manifest_or_unknown(info->nvcc_flags_sha256));
    mj_kv_str(b, &first, "triton_archs", manifest_or_unknown(info->triton_archs));
    mj_kv_str(b, &first, "triton_version", manifest_or_unknown(info->triton_version));
    mj_put(b, "}");
}

static void emit_runtime(struct MjBuf *b, const struct ManifestInputs *in) {
    int first = 1;
    mj_put(b, "{");
    mj_kv_str(b, &first, "cublas_version", manifest_or_unknown(in->cublas_version));
    mj_kv_str(b, &first, "cuda_driver_version",
              manifest_or_unknown(in->cuda_driver_version));
    mj_kv_str(b, &first, "cuda_runtime_version",
              manifest_or_unknown(in->cuda_runtime_version));
    mj_member(b, &first, "devices");
    mj_put(b, "[");
    for (int i = 0; i < in->device_count; ++i) {
        const struct ManifestDevice *dev = &in->devices[i];
        int inner = 1;
        if (i) mj_put(b, ",");
        mj_put(b, "{");
        mj_kv_str(b, &inner, "compute_capability", manifest_or_unknown(dev->compute_capability));
        mj_kv_int(b, &inner, "cuda_ordinal", dev->cuda_ordinal);
        mj_kv_str(b, &inner, "kernel_path", manifest_or_unknown(dev->kernel_path));
        mj_kv_str(b, &inner, "name", manifest_or_unknown(dev->name));
        mj_kv_str(b, &inner, "triton_cubin_arch",
                  manifest_or_unknown(dev->triton_cubin_arch));
        mj_kv_str(b, &inner, "uuid", manifest_or_unknown(dev->uuid));
        mj_put(b, "}");
    }
    mj_put(b, "]");
    mj_put(b, "}");
}

static void emit_sampling(struct MjBuf *b, const struct ManifestSampling *s) {
    static const struct ManifestSampling fallback = {
        "unspecified", "unspecified", 0, "unspecified", "unspecified"};
    int first = 1;
    if (s == NULL) s = &fallback;
    mj_put(b, "{");
    mj_kv_str(b, &first, "arithmetic", s->arithmetic);
    mj_kv_str(b, &first, "mode", s->mode);
    mj_member(b, &first, "rng");
    if (s->rng == NULL) mj_null(b);
    else mj_str(b, s->rng);
    mj_kv_str(b, &first, "transform", s->transform);
    mj_kv_int(b, &first, "transform_version", s->transform_version);
    mj_put(b, "}");
}

/* ------------------------------------------------------------------ */
/* Assembly                                                           */
/* ------------------------------------------------------------------ */

static int finish_block(struct MjBuf *b, const char *what, char *err, size_t err_len) {
    if (!b->failed) return 0;
    snprintf(err, err_len, "%s: %s", what, b->err);
    return -1;
}

int manifest_format(const struct ManifestInputs *in, char *buf, int buf_len) {
    if (in == NULL || in->desc == NULL || in->weights == NULL || buf == NULL ||
        buf_len <= 0) {
        return -1;
    }
    if (in->device_count < 0 || in->device_count > ENGINE_MANIFEST_MAX_DEVICES) {
        return -1;
    }
    if ((in->device_count > 0 && (in->devices == NULL || in->device_ordinals == NULL)) ||
        (in->num_layers > 0 && in->layer_device == NULL) ||
        (in->region_count > 0 && in->regions == NULL)) {
        return -1;
    }

    char *semantic = (char *)malloc(MJ_BLOCK_MAX);
    char *numerical = (char *)malloc(MJ_BLOCK_MAX);
    char *deployment = (char *)malloc(MJ_BLOCK_MAX);
    char *region_text = (char *)malloc(MJ_BLOCK_MAX);
    char err[256];
    if (semantic == NULL || numerical == NULL || deployment == NULL || region_text == NULL) {
        free(semantic);
        free(numerical);
        free(deployment);
        free(region_text);
        return -1;
    }

    struct MjBuf b;
    int status = -1;

    /* Regions first: the numerical policy carries their digest. Passing zero
     * regions means "the committed registry", so the default is explicit rather
     * than an empty table. */
    int default_region_count = 0;
    const struct ManifestRegion *default_regions = manifest_default_regions(&default_region_count);
    const struct ManifestRegion *regions =
        in->region_count > 0 ? in->regions : default_regions;
    const int used_region_count = in->region_count > 0 ? in->region_count : default_region_count;
    manifest_hex_t regions_sha256;
    mj_init(&b, region_text, MJ_BLOCK_MAX);
    emit_regions(&b, regions, used_region_count);
    if (finish_block(&b, "regions", err, sizeof(err)) != 0) goto done;
    sha256_hex(region_text, (size_t)b.used, regions_sha256);

    manifest_hex_t semantic_id, numerical_id, deployment_id;

    mj_init(&b, semantic, MJ_BLOCK_MAX);
    emit_semantic(&b, in->desc);
    if (finish_block(&b, "semantic", err, sizeof(err)) != 0) goto done;
    sha256_hex(semantic, (size_t)b.used, semantic_id);

    mj_init(&b, numerical, MJ_BLOCK_MAX);
    emit_numerical(&b, in, regions_sha256);
    if (finish_block(&b, "numerical_policy", err, sizeof(err)) != 0) goto done;
    sha256_hex(numerical, (size_t)b.used, numerical_id);

    mj_init(&b, deployment, MJ_BLOCK_MAX);
    emit_deployment(&b, in);
    if (finish_block(&b, "deployment", err, sizeof(err)) != 0) goto done;
    sha256_hex(deployment, (size_t)b.used, deployment_id);

    /* Top level. Every nested object is already canonical; the enclosing keys
     * are written in byte order too, so the whole document re-serializes
     * unchanged in another language. */
    mj_init(&b, buf, buf_len);
    mj_put(&b, "{");
    mj_put(&b, "\"deployment\":{\"deployment_id\":\"");
    mj_raw(&b, deployment_id);
    mj_put(&b, "\",\"fields\":");
    mj_raw(&b, deployment);
    mj_put(&b, "},");
    mj_put(&b, "\"descriptor\":{\"bytes\":");
    mj_int(&b, in->descriptor_bytes);
    mj_put(&b, ",\"desc_version\":");
    mj_int(&b, in->desc->version);
    mj_put(&b, ",\"sha256\":\"");
    mj_raw(&b, manifest_or_unknown(in->descriptor_sha256));
    mj_put(&b, "\"},");
    /* The document's own version. A consumer has to be able to tell a version-1
     * manifest from a legacy capture before it interprets anything else. */
    mj_put(&b, "\"manifest_version\":");
    mj_int(&b, ENGINE_MANIFEST_VERSION);
    mj_put(&b, ",");
    mj_put(&b, "\"numerical_policy\":{\"fields\":");
    mj_raw(&b, numerical);
    mj_put(&b, ",\"numerical_policy_id\":\"");
    mj_raw(&b, numerical_id);
    mj_put(&b, "\"},");
    mj_put(&b, "\"provenance\":{\"build\":");
    {
        char build_text[4096];
        struct MjBuf sub;
        mj_init(&sub, build_text, (int)sizeof(build_text));
        emit_build(&sub, in->build);
        if (finish_block(&sub, "provenance.build", err, sizeof(err)) != 0) goto done;
        mj_raw(&b, build_text);
    }
    mj_put(&b, ",\"runtime\":");
    {
        char runtime_text[8192];
        struct MjBuf sub;
        mj_init(&sub, runtime_text, (int)sizeof(runtime_text));
        emit_runtime(&sub, in);
        if (finish_block(&sub, "provenance.runtime", err, sizeof(err)) != 0) goto done;
        mj_raw(&b, runtime_text);
    }
    mj_put(&b, "},");
    mj_put(&b, "\"regions\":");
    mj_raw(&b, region_text);
    mj_put(&b, ",\"sampling\":");
    {
        char sampling_text[1024];
        struct MjBuf sub;
        mj_init(&sub, sampling_text, (int)sizeof(sampling_text));
        emit_sampling(&sub, in->sampling);
        if (finish_block(&sub, "sampling", err, sizeof(err)) != 0) goto done;
        mj_raw(&b, sampling_text);
    }
    mj_put(&b, ",\"semantic\":{\"fields\":");
    mj_raw(&b, semantic);
    mj_put(&b, ",\"semantic_id\":\"");
    mj_raw(&b, semantic_id);
    mj_put(&b, "\"},");
    mj_put(&b, "\"weights\":");
    {
        char weights_text[1024];
        struct MjBuf sub;
        mj_init(&sub, weights_text, (int)sizeof(weights_text));
        emit_weights(&sub, in->weights);
        if (finish_block(&sub, "weights", err, sizeof(err)) != 0) goto done;
        mj_raw(&b, weights_text);
    }
    mj_put(&b, "}");

    if (b.failed) {
        /* Report the buffer limit distinctly so a caller can size its buffer
         * instead of misreading a truncation as a bad input. */
        if (strstr(b.err, "too small") != NULL) status = -2;
        goto done;
    }
    status = b.used;

done:
    free(semantic);
    free(numerical);
    free(deployment);
    free(region_text);
    return status;
}
