/**
 * model_desc.h - Model descriptor: the architecture data the engine is built from.
 *
 * The descriptor is a *flat* JSON document produced by the Haskell side
 * (see src/Infer/Descriptor.hs). C parses it strictly: unknown keys, missing
 * keys and wrong types are all hard errors. Nothing here is family-specific --
 * families differ only in the descriptor contents.
 */
#ifndef HASKELL_INFER_MODEL_DESC_H
#define HASKELL_INFER_MODEL_DESC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ENGINE_DESC_VERSION 1
#define ENGINE_MAX_LAYERS 512
#define ENGINE_MAX_ROLES 64
#define ENGINE_MAX_EOS 16
#define ENGINE_TEMPLATE_MAX 256
#define ENGINE_FAMILY_MAX 64

/* Layer mixer kinds (mirrors Infer.Descriptor.MixerKind). */
#define ENGINE_MIXER_FULL_ATTN 0
#define ENGINE_MIXER_GDN 1
#define ENGINE_MIXER_MLA 2

/* Feed-forward kinds (mirrors Infer.Descriptor.FfnKind). */
#define ENGINE_FFN_DENSE 0
#define ENGINE_FFN_MOE 1

/* Tensor-parallel sharding rules for one weight role (mirrors
 * Infer.Descriptor.ShardKind). The vocabulary is deliberately small so the
 * engine can load shards without any family knowledge:
 *
 *   NONE      - replicated: every rank loads the whole tensor;
 *   OUT_HEADS - output dimension split by head (rows are heads*head_dim; each
 *               rank takes a contiguous block of heads);
 *   OUT_DIM   - output dimension (rows of an [out, in] weight) split into
 *               tp_size contiguous blocks;
 *   IN_DIM    - input dimension (columns) split into tp_size contiguous blocks.
 *   OUT_EXPERTS - the expert dimension: rank r keeps experts
 *               [r * E/ep_size, (r + 1) * E/ep_size) (expert parallelism; the
 *               router and the shared experts stay replicated). The rule names
 *               the tensor's *role* rather than a dimension: each expert's own
 *               tensor is loaded whole, the loader just picks the local range.
 *
 * With tp_size == 1 / ep_size == 1 every rule is a no-op. */
#define ENGINE_SHARD_NONE 0
#define ENGINE_SHARD_OUT_HEADS 1
#define ENGINE_SHARD_OUT_DIM 2
#define ENGINE_SHARD_IN_DIM 3
#define ENGINE_SHARD_OUT_EXPERTS 4

/* Weight roles (mirrors Infer.Descriptor.Role, same order). */
enum {
    ROLE_EMBED = 0,
    ROLE_LM_HEAD,
    ROLE_FINAL_NORM,
    ROLE_INPUT_NORM,
    ROLE_POST_NORM,
    ROLE_MLP_GATE,
    ROLE_MLP_UP,
    ROLE_MLP_DOWN,
    ROLE_ATTN_Q,
    ROLE_ATTN_K,
    ROLE_ATTN_V,
    ROLE_ATTN_O,
    ROLE_ATTN_Q_NORM,
    ROLE_ATTN_K_NORM,
    ROLE_GDN_QKV,
    ROLE_GDN_Z,
    ROLE_GDN_A,
    ROLE_GDN_B,
    ROLE_GDN_CONV1D,
    ROLE_GDN_DT_BIAS,
    ROLE_GDN_A_LOG,
    ROLE_GDN_OUT,
    ROLE_GDN_NORM,
    ROLE_GDN_QKVZ,        /* fused q+k+v+z projection (Qwen3-Next style) */
    ROLE_GDN_BA,          /* fused b+a projection */
    ROLE_MOE_ROUTER,
    ROLE_MOE_ROUTER_BIAS,
    ROLE_MOE_EXPERT_GATE,
    ROLE_MOE_EXPERT_UP,
    ROLE_MOE_EXPERT_DOWN,
    ROLE_MOE_SHARED_GATE,
    ROLE_MOE_SHARED_UP,
    ROLE_MOE_SHARED_DOWN,
    ROLE_MOE_SHARED_GATE_SCALAR,
    ROLE_MLA_Q,           /* MLA query projection */
    ROLE_MLA_KV_A,        /* latent KV + shared RoPE key projection */
    ROLE_MLA_KV_A_NORM,   /* RMSNorm over the latent */
    ROLE_MLA_KV_B,        /* latent -> per-head k_nope and v */
    ROLE_MLA_O,           /* output projection */
    ROLE_COUNT
};

struct ModelDesc {
    int version;
    char family[ENGINE_FAMILY_MAX];
    char model_type[ENGINE_FAMILY_MAX];

    int num_layers;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    double rms_eps;
    int max_position_embeddings;
    int max_seq_len;

    int num_heads;
    int num_kv_heads;
    int head_dim;
    int rotary_dim;
    double rotary_theta;
    char norm_style[16];        /* "gemma" (weight + 1) or "plain" */
    int attn_qk_norm;           /* attention has per-head q/k RMSNorm */
    int attn_output_gate;
    int q_gate_interleave;

    int gdn_conv_dim;
    int gdn_value_dim;
    int gdn_num_v_heads;
    int gdn_num_k_heads;
    int gdn_head_dim;
    int gdn_conv_kernel;
    int fla_chunk_size;
    int max_chunk;              /* prefill batch size the engine chunks to */

    /* Mixture-of-experts feed-forward (used when a layer's ffn kind is moe). */
    int moe_num_experts;
    int moe_top_k;
    int moe_intermediate_size;
    char moe_router_scoring[16];   /* "softmax" or "sigmoid" */
    int moe_norm_topk_prob;
    int moe_num_shared_experts;
    int moe_shared_intermediate_size;
    double moe_routed_scaling_factor;
    int moe_shared_gate_scalar;

    /* Multi-head latent attention (used when a layer's mixer kind is MLA).
     * 0 means "no MLA layers in this model". */
    int mla_kv_lora_rank;       /* compressed KV width the cache holds */
    int mla_qk_nope_head_dim;   /* per-head q/k width carried explicitly */
    int mla_qk_rope_head_dim;   /* per-head q/k width carrying RoPE */
    int mla_v_head_dim;         /* per-head value width */

    /* Tensor parallelism (defaults: tp_size 1, tp_rank 0 = no sharding).
     * tp_size is the number of ranks; tp_rank identifies this rank. role_shards
     * is parallel to role_ids/role_templates and holds one ENGINE_SHARD_* rule
     * per role, so the loader knows which dimension (if any) to split. When the
     * wire document omits the key the table is all ENGINE_SHARD_NONE. */
    int tp_size;
    int tp_rank;

    /* Expert parallelism (defaults: ep_size 1, ep_rank 0 = whole experts per
     * rank). Rank r holds experts [r * E/ep_size, (r + 1) * E/ep_size); the
     * router and the shared experts stay replicated, so every rank computes the
     * same top-k and the routed partial is all-reduced. */
    int ep_size;
    int ep_rank;
    int role_shard_count;
    int role_shards[ENGINE_MAX_ROLES];

    int eos_count;
    int eos_tokens[ENGINE_MAX_EOS];

    int layer_mixers[ENGINE_MAX_LAYERS];
    int layer_ffns[ENGINE_MAX_LAYERS];

    int role_count;
    int role_ids[ENGINE_MAX_ROLES];
    char role_templates[ENGINE_MAX_ROLES][ENGINE_TEMPLATE_MAX];
};

/* Parse a descriptor document. Returns 0 on success, -1 on failure with a
 * message in err (truncated to err_len). */
int model_desc_parse(const char *json, struct ModelDesc *out, char *err, size_t err_len);

/* Structural validation (lengths, ranges, required roles per layer kind). */
int model_desc_validate(const struct ModelDesc *desc, char *err, size_t err_len);

/* Runtime capability validation: rejects combinations the AOT kernels were not
 * built for (currently the GDN head layout). Structural validation does not
 * cover this because such a descriptor is still a faithful checkpoint
 * description. Returns 0 on success, -1 with a message in err. */
int model_desc_check_runtime_support(const struct ModelDesc *desc, char *err, size_t err_len);

/* Index of a role in the descriptor's role table, or -1 if absent. */
int model_desc_role_index(const struct ModelDesc *desc, int role);

/* The slice of a role's tensor one tensor-parallel rank holds. With tp_size 1
 * (or an ENGINE_SHARD_NONE rule) the view is the whole tensor. */
struct ShardView {
    long long row_off;   /* first row of the rank's shard */
    long long rows;      /* rows the rank holds */
    long long col_off;   /* first column of the rank's shard */
    long long cols;      /* columns the rank holds */
};

/* Compute @rank@'s slice of a role whose checkpoint shape is [global_rows,
 * global_cols]. The rank is explicit because one engine process holds every
 * rank (device index == rank); @desc->tp_rank@ stays 0 there. Returns 0 on
 * success, -1 with a message in err when the rule does not fit the role or the
 * split is not even. The loader combines the view with the role's expected
 * shape (engine.cu). */
int model_desc_shard_view(const struct ModelDesc *desc, int role,
                          long long global_rows, long long global_cols, int rank,
                          struct ShardView *out, char *err, size_t err_len);

/* Canonical JSON echo of the parsed descriptor. Returns the number of bytes
 * written (excluding the terminator) or -1 when the buffer is too small. */
int model_desc_format(const struct ModelDesc *desc, char *buf, int buf_len);

/* Canonical wire spellings for the mixer/ffn/role/shard vocabularies. They are
 * shared by the parser, the echo and the execution manifest so the three cannot
 * drift apart. Return NULL for a value outside the vocabulary. */
const char *model_desc_mixer_name(int kind);
const char *model_desc_ffn_name(int kind);
const char *model_desc_role_name(int role);
const char *model_desc_shard_name(int rule);

/* Expand a role template: %d -> layer index, %e -> expert index. */
void model_desc_expand(const char *templ, int layer, int expert, char *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_MODEL_DESC_H */
