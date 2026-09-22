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
    ROLE_MOE_ROUTER,
    ROLE_MOE_ROUTER_BIAS,
    ROLE_MOE_EXPERT_GATE,
    ROLE_MOE_EXPERT_UP,
    ROLE_MOE_EXPERT_DOWN,
    ROLE_MOE_SHARED_GATE,
    ROLE_MOE_SHARED_UP,
    ROLE_MOE_SHARED_DOWN,
    ROLE_MOE_SHARED_GATE_SCALAR,
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
    int attn_output_gate;
    int q_gate_interleave;

    int gdn_conv_dim;
    int gdn_value_dim;
    int gdn_num_v_heads;
    int gdn_num_k_heads;
    int gdn_head_dim;
    int gdn_conv_kernel;
    int fla_chunk_size;

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

/* Index of a role in the descriptor's role table, or -1 if absent. */
int model_desc_role_index(const struct ModelDesc *desc, int role);

/* Canonical JSON echo of the parsed descriptor. Returns the number of bytes
 * written (excluding the terminator) or -1 when the buffer is too small. */
int model_desc_format(const struct ModelDesc *desc, char *buf, int buf_len);

/* Expand a role template: %d -> layer index, %e -> expert index. */
void model_desc_expand(const char *templ, int layer, int expert, char *out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_MODEL_DESC_H */
