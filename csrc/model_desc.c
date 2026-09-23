/**
 * model_desc.c - Strict parser and validator for the flat model descriptor.
 *
 * The wire format is a single flat JSON object (no nesting), so a small
 * hand-rolled scanner suffices and keeps the C side dependency-free. Parsing is
 * strict by design: every key must be known, every required key must be present
 * and have the right type. Typos and drift between Haskell and C fail loudly.
 */
#include "model_desc.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Key table                                                          */
/* ------------------------------------------------------------------ */

enum {
    KV_INT,
    KV_DOUBLE,
    KV_BOOL,
    KV_TEXT,
    KV_INT_ARRAY,
    KV_TEXT_ARRAY
};

enum {
    K_DESC_VERSION = 0, K_FAMILY, K_MODEL_TYPE, K_NUM_LAYERS, K_HIDDEN_SIZE,
    K_INTERMEDIATE_SIZE, K_VOCAB_SIZE, K_RMS_EPS, K_MAX_POSITION_EMBEDDINGS,
    K_MAX_SEQ_LEN, K_NUM_HEADS, K_NUM_KV_HEADS, K_HEAD_DIM, K_ROTARY_DIM,
    K_ROTARY_THETA, K_NORM_STYLE, K_ATTN_QK_NORM, K_ATTN_OUTPUT_GATE, K_Q_GATE_INTERLEAVE,
    K_GDN_CONV_DIM,
    K_GDN_VALUE_DIM, K_GDN_NUM_V_HEADS, K_GDN_NUM_K_HEADS, K_GDN_HEAD_DIM,
    K_GDN_CONV_KERNEL, K_FLA_CHUNK_SIZE, K_MAX_CHUNK, K_MOE_NUM_EXPERTS, K_MOE_TOP_K,
    K_MOE_INTERMEDIATE_SIZE, K_MOE_ROUTER_SCORING, K_MOE_NORM_TOPK_PROB,
    K_MOE_NUM_SHARED_EXPERTS, K_MOE_SHARED_INTERMEDIATE_SIZE, K_MOE_ROUTED_SCALING_FACTOR,
    K_MOE_SHARED_GATE_SCALAR,
    K_MLA_KV_LORA_RANK, K_MLA_QK_NOPE_HEAD_DIM, K_MLA_QK_ROPE_HEAD_DIM, K_MLA_V_HEAD_DIM,
    K_EOS_TOKENS, K_LAYER_MIXERS,
    K_LAYER_FFNS, K_ROLE_NAMES, K_ROLE_TEMPLATES,
    /* Optional keys: absent means the historical single-rank behaviour. */
    K_TP_SIZE, K_TP_RANK, K_EP_SIZE, K_EP_RANK, K_ROLE_SHARDS,
    K_COUNT
};

static const struct {
    const char *name;
    int type;
    int optional;   /* optional keys may be absent and take their default */
} kKeys[K_COUNT] = {
    {"desc_version", KV_INT, 0},
    {"family", KV_TEXT, 0},
    {"model_type", KV_TEXT, 0},
    {"num_layers", KV_INT, 0},
    {"hidden_size", KV_INT, 0},
    {"intermediate_size", KV_INT, 0},
    {"vocab_size", KV_INT, 0},
    {"rms_eps", KV_DOUBLE, 0},
    {"max_position_embeddings", KV_INT, 0},
    {"max_seq_len", KV_INT, 0},
    {"num_heads", KV_INT, 0},
    {"num_kv_heads", KV_INT, 0},
    {"head_dim", KV_INT, 0},
    {"rotary_dim", KV_INT, 0},
    {"rotary_theta", KV_DOUBLE, 0},
    {"norm_style", KV_TEXT, 0},
    {"attn_qk_norm", KV_BOOL, 0},
    {"attn_output_gate", KV_BOOL, 0},
    {"q_gate_interleave", KV_BOOL, 0},
    {"gdn_conv_dim", KV_INT, 0},
    {"gdn_value_dim", KV_INT, 0},
    {"gdn_num_v_heads", KV_INT, 0},
    {"gdn_num_k_heads", KV_INT, 0},
    {"gdn_head_dim", KV_INT, 0},
    {"gdn_conv_kernel", KV_INT, 0},
    {"fla_chunk_size", KV_INT, 0},
    {"max_chunk", KV_INT, 0},
    {"moe_num_experts", KV_INT, 0},
    {"moe_top_k", KV_INT, 0},
    {"moe_intermediate_size", KV_INT, 0},
    {"moe_router_scoring", KV_TEXT, 0},
    {"moe_norm_topk_prob", KV_BOOL, 0},
    {"moe_num_shared_experts", KV_INT, 0},
    {"moe_shared_intermediate_size", KV_INT, 0},
    {"moe_routed_scaling_factor", KV_DOUBLE, 0},
    {"moe_shared_gate_scalar", KV_BOOL, 0},
    {"mla_kv_lora_rank", KV_INT, 1},
    {"mla_qk_nope_head_dim", KV_INT, 1},
    {"mla_qk_rope_head_dim", KV_INT, 1},
    {"mla_v_head_dim", KV_INT, 1},
    {"eos_tokens", KV_INT_ARRAY, 0},
    {"layer_mixers", KV_TEXT_ARRAY, 0},
    {"layer_ffns", KV_TEXT_ARRAY, 0},
    {"role_names", KV_TEXT_ARRAY, 0},
    {"role_templates", KV_TEXT_ARRAY, 0},
    {"tp_size", KV_INT, 1},
    {"tp_rank", KV_INT, 1},
    {"ep_size", KV_INT, 1},
    {"ep_rank", KV_INT, 1},
    {"role_shards", KV_TEXT_ARRAY, 1},
};

static int key_id(const char *name) {
    for (int i = 0; i < K_COUNT; ++i)
        if (strcmp(kKeys[i].name, name) == 0) return i;
    return -1;
}

/* Shard rules, in the same order as the Haskell ShardKind enum. */
static const char *kShardNames[] = {"none", "out_heads", "out_dim", "in_dim", "out_experts"};
#define SHARD_KIND_COUNT ((int)(sizeof(kShardNames) / sizeof(kShardNames[0])))

/* Role names, in the same order as the Haskell Role enum. */
static const char *kRoleNames[ROLE_COUNT] = {
    "embed", "lmHead", "finalNorm", "inputNorm", "postNorm",
    "mlpGate", "mlpUp", "mlpDown",
    "attnQ", "attnK", "attnV", "attnO", "attnQNorm", "attnKNorm",
    "gdnQkv", "gdnZ", "gdnA", "gdnB", "gdnConv1d", "gdnDtBias", "gdnALog",
    "gdnOut", "gdnNorm", "gdnQkvz", "gdnBa",
    "moeRouter", "moeRouterBias",
    "moeExpertGate", "moeExpertUp", "moeExpertDown",
    "moeSharedGate", "moeSharedUp", "moeSharedDown", "moeSharedGateScalar",
    "mlaQ", "mlaKvA", "mlaKvANorm", "mlaKvB", "mlaO",
};

/* ------------------------------------------------------------------ */
/* Scanner                                                            */
/* ------------------------------------------------------------------ */

struct Cursor {
    const char *p;
};

static void fail(char *err, size_t err_len, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static void skip_ws(struct Cursor *c) {
    while (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r') ++c->p;
}

static int parse_string(struct Cursor *c, char *out, size_t out_len) {
    skip_ws(c);
    if (*c->p != '"') return -1;
    ++c->p;
    size_t n = 0;
    while (*c->p && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\') {
            char esc = *c->p++;
            switch (esc) {
            case 'n': ch = '\n'; break;
            case 't': ch = '\t'; break;
            case 'r': ch = '\r'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case '/': ch = '/'; break;
            case '\\': ch = '\\'; break;
            case '"': ch = '"'; break;
            case 'u': /* descriptors are ASCII/ASCII-escaped; reject \\u */
                return -1;
            default: return -1;
            }
        }
        if (n + 1 >= out_len) return -1;
        out[n++] = ch;
    }
    if (*c->p != '"') return -1;
    ++c->p;
    out[n] = '\0';
    return 0;
}

static int parse_number(struct Cursor *c, double *out) {
    skip_ws(c);
    char *end = NULL;
    double value = strtod(c->p, &end);
    if (end == c->p) return -1;
    c->p = end;
    *out = value;
    return 0;
}

static int parse_bool(struct Cursor *c, int *out) {
    skip_ws(c);
    if (strncmp(c->p, "true", 4) == 0) { c->p += 4; *out = 1; return 0; }
    if (strncmp(c->p, "false", 5) == 0) { c->p += 5; *out = 0; return 0; }
    return -1;
}

static int expect_char(struct Cursor *c, char expected) {
    skip_ws(c);
    if (*c->p != expected) return -1;
    ++c->p;
    return 0;
}

static int mixer_from_name(const char *name) {
    if (strcmp(name, "full_attn") == 0) return ENGINE_MIXER_FULL_ATTN;
    if (strcmp(name, "gdn") == 0) return ENGINE_MIXER_GDN;
    if (strcmp(name, "mla") == 0) return ENGINE_MIXER_MLA;
    return -1;
}

static int ffn_from_name(const char *name) {
    if (strcmp(name, "dense") == 0) return ENGINE_FFN_DENSE;
    if (strcmp(name, "moe") == 0) return ENGINE_FFN_MOE;
    return -1;
}

static int role_from_name(const char *name) {
    for (int i = 0; i < ROLE_COUNT; ++i)
        if (strcmp(kRoleNames[i], name) == 0) return i;
    return -1;
}

static int shard_from_name(const char *name) {
    for (int i = 0; i < SHARD_KIND_COUNT; ++i)
        if (strcmp(kShardNames[i], name) == 0) return i;
    return -1;
}

static int parse_int_array(struct Cursor *c, int *out, int max, int *count,
                           char *err, size_t err_len) {
    if (expect_char(c, '[') != 0) return -1;
    int n = 0;
    skip_ws(c);
    if (*c->p == ']') { ++c->p; *count = 0; return 0; }
    for (;;) {
        double value = 0;
        if (parse_number(c, &value) != 0) {
            fail(err, err_len, "expected a number in array");
            return -1;
        }
        if (n >= max) {
            fail(err, err_len, "array has more than %d entries", max);
            return -1;
        }
        out[n++] = (int)value;
        skip_ws(c);
        if (*c->p == ',') { ++c->p; continue; }
        if (*c->p == ']') { ++c->p; break; }
        fail(err, err_len, "expected ',' or ']' in array");
        return -1;
    }
    *count = n;
    return 0;
}

static int parse_text_array(struct Cursor *c, int max, int *count, char *err,
                            size_t err_len, int (*map)(const char *),
                            int store[ENGINE_MAX_ROLES], char store_text[ENGINE_MAX_ROLES][ENGINE_TEMPLATE_MAX]) {
    if (expect_char(c, '[') != 0) return -1;
    int n = 0;
    skip_ws(c);
    if (*c->p == ']') { ++c->p; *count = 0; return 0; }
    for (;;) {
        char item[ENGINE_TEMPLATE_MAX];
        if (parse_string(c, item, sizeof(item)) != 0) {
            fail(err, err_len, "expected a string in array");
            return -1;
        }
        if (n >= max) {
            fail(err, err_len, "array has more than %d entries", max);
            return -1;
        }
        if (store != NULL) {
            int id = map != NULL ? map(item) : -1;
            if (id < 0) {
                fail(err, err_len, "unknown name in array: %s", item);
                return -1;
            }
            store[n] = id;
        }
        if (store_text != NULL) {
            snprintf(store_text[n], ENGINE_TEMPLATE_MAX, "%s", item);
        }
        ++n;
        skip_ws(c);
        if (*c->p == ',') { ++c->p; continue; }
        if (*c->p == ']') { ++c->p; break; }
        fail(err, err_len, "expected ',' or ']' in array");
        return -1;
    }
    *count = n;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Parse                                                              */
/* ------------------------------------------------------------------ */

int model_desc_parse(const char *json, struct ModelDesc *out, char *err, size_t err_len) {
    if (json == NULL || out == NULL) {
        fail(err, err_len, "null descriptor");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    /* Defaults for the optional parallel keys: one rank, no sharding. */
    out->tp_size = 1;
    out->tp_rank = 0;
    out->ep_size = 1;
    out->ep_rank = 0;
    unsigned char seen[K_COUNT] = {0};
    int eos_count = 0, mixer_count = 0, ffn_count = 0, role_count = 0, template_count = 0;
    int shard_count = 0;
    char (*templates)[ENGINE_TEMPLATE_MAX] = out->role_templates;

    struct Cursor c = {json};
    if (expect_char(&c, '{') != 0) {
        fail(err, err_len, "descriptor must be a JSON object");
        return -1;
    }
    skip_ws(&c);
    if (*c.p == '}') {
        fail(err, err_len, "descriptor is empty");
        return -1;
    }
    for (;;) {
        char key[128];
        if (parse_string(&c, key, sizeof(key)) != 0) {
            fail(err, err_len, "expected a key name at offset %ld", (long)(c.p - json));
            return -1;
        }
        int id = key_id(key);
        if (id < 0) {
            fail(err, err_len, "unknown descriptor key: %s", key);
            return -1;
        }
        if (seen[id]) {
            fail(err, err_len, "duplicate descriptor key: %s", key);
            return -1;
        }
        seen[id] = 1;
        if (expect_char(&c, ':') != 0) {
            fail(err, err_len, "expected ':' after key %s", key);
            return -1;
        }
        double number = 0;
        int boolean = 0;
        switch (id) {
        case K_DESC_VERSION: case K_NUM_LAYERS: case K_HIDDEN_SIZE:
        case K_INTERMEDIATE_SIZE: case K_VOCAB_SIZE: case K_MAX_POSITION_EMBEDDINGS:
        case K_MAX_SEQ_LEN: case K_NUM_HEADS: case K_NUM_KV_HEADS: case K_HEAD_DIM:
        case K_ROTARY_DIM: case K_GDN_CONV_DIM: case K_GDN_VALUE_DIM:
        case K_GDN_NUM_V_HEADS: case K_GDN_NUM_K_HEADS: case K_GDN_HEAD_DIM:
        case K_GDN_CONV_KERNEL: case K_FLA_CHUNK_SIZE: case K_MAX_CHUNK:
        case K_TP_SIZE: case K_TP_RANK:
        case K_EP_SIZE: case K_EP_RANK:
        case K_MOE_NUM_EXPERTS: case K_MOE_TOP_K: case K_MOE_INTERMEDIATE_SIZE:
        case K_MOE_NUM_SHARED_EXPERTS: case K_MOE_SHARED_INTERMEDIATE_SIZE:
        case K_MLA_KV_LORA_RANK: case K_MLA_QK_NOPE_HEAD_DIM:
        case K_MLA_QK_ROPE_HEAD_DIM: case K_MLA_V_HEAD_DIM:
            if (parse_number(&c, &number) != 0) {
                fail(err, err_len, "key %s must be a number", key);
                return -1;
            }
            break;
        case K_RMS_EPS: case K_ROTARY_THETA: case K_MOE_ROUTED_SCALING_FACTOR:
            if (parse_number(&c, &number) != 0) {
                fail(err, err_len, "key %s must be a number", key);
                return -1;
            }
            break;
        case K_ATTN_QK_NORM: case K_ATTN_OUTPUT_GATE: case K_Q_GATE_INTERLEAVE:
        case K_MOE_NORM_TOPK_PROB: case K_MOE_SHARED_GATE_SCALAR:
            if (parse_bool(&c, &boolean) != 0) {
                fail(err, err_len, "key %s must be a boolean", key);
                return -1;
            }
            break;
        case K_NORM_STYLE:
            if (parse_string(&c, out->norm_style, sizeof(out->norm_style)) != 0) {
                fail(err, err_len, "key %s must be a string", key);
                return -1;
            }
            break;
        case K_MOE_ROUTER_SCORING:
            if (parse_string(&c, out->moe_router_scoring, sizeof(out->moe_router_scoring)) != 0) {
                fail(err, err_len, "key %s must be a string", key);
                return -1;
            }
            break;
        case K_FAMILY: case K_MODEL_TYPE:
            if (parse_string(&c, id == K_FAMILY ? out->family : out->model_type,
                             ENGINE_FAMILY_MAX) != 0) {
                fail(err, err_len, "key %s must be a string", key);
                return -1;
            }
            break;
        case K_EOS_TOKENS:
            if (parse_int_array(&c, out->eos_tokens, ENGINE_MAX_EOS, &eos_count, err, err_len) != 0)
                return -1;
            break;
        case K_LAYER_MIXERS:
            if (parse_text_array(&c, ENGINE_MAX_LAYERS, &mixer_count, err, err_len,
                                 mixer_from_name,
                                 out->layer_mixers, NULL) != 0)
                return -1;
            break;
        case K_LAYER_FFNS:
            if (parse_text_array(&c, ENGINE_MAX_LAYERS, &ffn_count, err, err_len,
                                 ffn_from_name,
                                 out->layer_ffns, NULL) != 0)
                return -1;
            break;
        case K_ROLE_NAMES:
            if (parse_text_array(&c, ENGINE_MAX_ROLES, &role_count, err, err_len,
                                 role_from_name,
                                 out->role_ids, NULL) != 0)
                return -1;
            break;
        case K_ROLE_TEMPLATES:
            if (parse_text_array(&c, ENGINE_MAX_ROLES, &template_count, err, err_len,
                                 NULL, NULL, templates) != 0)
                return -1;
            break;
        case K_ROLE_SHARDS:
            if (parse_text_array(&c, ENGINE_MAX_ROLES, &shard_count, err, err_len,
                                 shard_from_name, out->role_shards, NULL) != 0)
                return -1;
            break;
        default:
            fail(err, err_len, "unhandled descriptor key: %s", key);
            return -1;
        }
        /* Store scalars after the switch to keep the cases uniform. */
        switch (id) {
        case K_DESC_VERSION: out->version = (int)number; break;
        case K_NUM_LAYERS: out->num_layers = (int)number; break;
        case K_HIDDEN_SIZE: out->hidden_size = (int)number; break;
        case K_INTERMEDIATE_SIZE: out->intermediate_size = (int)number; break;
        case K_VOCAB_SIZE: out->vocab_size = (int)number; break;
        case K_MAX_POSITION_EMBEDDINGS: out->max_position_embeddings = (int)number; break;
        case K_MAX_SEQ_LEN: out->max_seq_len = (int)number; break;
        case K_NUM_HEADS: out->num_heads = (int)number; break;
        case K_NUM_KV_HEADS: out->num_kv_heads = (int)number; break;
        case K_HEAD_DIM: out->head_dim = (int)number; break;
        case K_ROTARY_DIM: out->rotary_dim = (int)number; break;
        case K_RMS_EPS: out->rms_eps = number; break;
        case K_ROTARY_THETA: out->rotary_theta = number; break;
        case K_ATTN_QK_NORM: out->attn_qk_norm = boolean; break;
        case K_ATTN_OUTPUT_GATE: out->attn_output_gate = boolean; break;
        case K_Q_GATE_INTERLEAVE: out->q_gate_interleave = boolean; break;
        case K_GDN_CONV_DIM: out->gdn_conv_dim = (int)number; break;
        case K_GDN_VALUE_DIM: out->gdn_value_dim = (int)number; break;
        case K_GDN_NUM_V_HEADS: out->gdn_num_v_heads = (int)number; break;
        case K_GDN_NUM_K_HEADS: out->gdn_num_k_heads = (int)number; break;
        case K_GDN_HEAD_DIM: out->gdn_head_dim = (int)number; break;
        case K_GDN_CONV_KERNEL: out->gdn_conv_kernel = (int)number; break;
        case K_FLA_CHUNK_SIZE: out->fla_chunk_size = (int)number; break;
        case K_MAX_CHUNK: out->max_chunk = (int)number; break;
        case K_TP_SIZE: out->tp_size = (int)number; break;
        case K_TP_RANK: out->tp_rank = (int)number; break;
        case K_EP_SIZE: out->ep_size = (int)number; break;
        case K_EP_RANK: out->ep_rank = (int)number; break;
        case K_MOE_NUM_EXPERTS: out->moe_num_experts = (int)number; break;
        case K_MOE_TOP_K: out->moe_top_k = (int)number; break;
        case K_MOE_INTERMEDIATE_SIZE: out->moe_intermediate_size = (int)number; break;
        case K_MOE_ROUTED_SCALING_FACTOR: out->moe_routed_scaling_factor = number; break;
        case K_MOE_NUM_SHARED_EXPERTS: out->moe_num_shared_experts = (int)number; break;
        case K_MLA_KV_LORA_RANK: out->mla_kv_lora_rank = (int)number; break;
        case K_MLA_QK_NOPE_HEAD_DIM: out->mla_qk_nope_head_dim = (int)number; break;
        case K_MLA_QK_ROPE_HEAD_DIM: out->mla_qk_rope_head_dim = (int)number; break;
        case K_MLA_V_HEAD_DIM: out->mla_v_head_dim = (int)number; break;
        case K_MOE_SHARED_INTERMEDIATE_SIZE: out->moe_shared_intermediate_size = (int)number; break;
        case K_MOE_NORM_TOPK_PROB: out->moe_norm_topk_prob = boolean; break;
        case K_MOE_SHARED_GATE_SCALAR: out->moe_shared_gate_scalar = boolean; break;
        case K_EOS_TOKENS: out->eos_count = eos_count; break;
        default: break;
        }
        skip_ws(&c);
        if (*c.p == ',') { ++c.p; continue; }
        if (*c.p == '}') { ++c.p; break; }
        fail(err, err_len, "expected ',' or '}' after key %s", key);
        return -1;
    }
    skip_ws(&c);
    if (*c.p != '\0') {
        fail(err, err_len, "trailing content after the descriptor object");
        return -1;
    }
    if (role_count != template_count) {
        fail(err, err_len, "role_names (%d) and role_templates (%d) differ in length",
             role_count, template_count);
        return -1;
    }
    /* role_shards is parallel to the role table; absent means all-replicated
     * (the struct was zeroed, and ENGINE_SHARD_NONE is 0). */
    out->role_shard_count = seen[K_ROLE_SHARDS] ? shard_count : role_count;
    if (out->role_shard_count != role_count) {
        fail(err, err_len, "role_shards (%d) and role_names (%d) differ in length",
             out->role_shard_count, role_count);
        return -1;
    }
    out->role_count = role_count;
    for (int i = 0; i < K_COUNT; ++i) {
        if (!seen[i] && !kKeys[i].optional) {
            fail(err, err_len, "descriptor key is missing: %s", kKeys[i].name);
            return -1;
        }
    }
    if (out->version != ENGINE_DESC_VERSION) {
        fail(err, err_len, "descriptor version %d is not supported (expected %d)",
             out->version, ENGINE_DESC_VERSION);
        return -1;
    }
    return model_desc_validate(out, err, err_len);
}

int model_desc_role_index(const struct ModelDesc *desc, int role) {
    for (int i = 0; i < desc->role_count; ++i)
        if (desc->role_ids[i] == role) return i;
    return -1;
}

/* Head count a role's output rows are grouped by (0 = the role is not
 * head-grouped). Mirrors the engine's expected_shape for these roles. */
static int role_heads(const struct ModelDesc *desc, int role) {
    switch (role) {
    case ROLE_ATTN_Q: return desc->num_heads;
    case ROLE_ATTN_K:
    case ROLE_ATTN_V: return desc->num_kv_heads;
    default: return 0;
    }
}

/* Which shard rules a role's tensor layout can carry; mirrors the Haskell
 * validation so a hand-edited descriptor fails on both sides. */
static int shard_rule_fits_role(int role, int rule) {
    switch (rule) {
    case ENGINE_SHARD_NONE: return 1;
    case ENGINE_SHARD_OUT_HEADS:
        return role == ROLE_ATTN_Q || role == ROLE_ATTN_K || role == ROLE_ATTN_V;
    case ENGINE_SHARD_OUT_DIM: return role == ROLE_MLP_GATE || role == ROLE_MLP_UP;
    case ENGINE_SHARD_IN_DIM: return role == ROLE_ATTN_O || role == ROLE_MLP_DOWN;
    case ENGINE_SHARD_OUT_EXPERTS:
        return role == ROLE_MOE_EXPERT_GATE || role == ROLE_MOE_EXPERT_UP ||
               role == ROLE_MOE_EXPERT_DOWN;
    default: return 0;
    }
}

int model_desc_shard_view(const struct ModelDesc *desc, int role,
                          long long global_rows, long long global_cols, int rank,
                          struct ShardView *out, char *err, size_t err_len) {
    if (!desc || !out) {
        fail(err, err_len, "model_desc_shard_view: null argument");
        return -1;
    }
    out->row_off = 0;
    out->rows = global_rows;
    out->col_off = 0;
    out->cols = global_cols;
    const int tp = desc->tp_size > 0 ? desc->tp_size : 1;
    if (tp == 1) return 0;
    if (rank < 0 || rank >= tp) {
        fail(err, err_len, "shard view rank %d is outside [0, %d)", rank, tp);
        return -1;
    }
    const int slot = model_desc_role_index(desc, role);
    const int rule = slot >= 0 && slot < desc->role_shard_count
                         ? desc->role_shards[slot] : ENGINE_SHARD_NONE;
    switch (rule) {
    case ENGINE_SHARD_NONE:
        return 0;
    case ENGINE_SHARD_OUT_EXPERTS:
        /* The expert dimension is not a tensor dimension: each expert keeps its
         * own tensor whole, and the loader picks the local expert range. */
        return 0;
    case ENGINE_SHARD_OUT_HEADS: {
        const int heads = role_heads(desc, role);
        if (heads <= 0) {
            fail(err, err_len, "role %s cannot be split by heads", kRoleNames[role]);
            return -1;
        }
        const long long per_head = global_rows / heads;
        if (per_head * heads != global_rows) {
            fail(err, err_len, "role %s: rows %lld are not a whole number of heads",
                 kRoleNames[role], global_rows);
            return -1;
        }
        if (heads % tp != 0) {
            fail(err, err_len, "role %s: tp_size %d does not divide %d heads",
                 kRoleNames[role], tp, heads);
            return -1;
        }
        out->rows = per_head * (long long)(heads / tp);
        out->row_off = out->rows * rank;
        return 0;
    }
    case ENGINE_SHARD_OUT_DIM:
        if (global_rows % tp != 0) {
            fail(err, err_len, "role %s: tp_size %d does not divide %lld rows",
                 kRoleNames[role], tp, global_rows);
            return -1;
        }
        out->rows = global_rows / tp;
        out->row_off = out->rows * rank;
        return 0;
    case ENGINE_SHARD_IN_DIM:
        if (global_cols % tp != 0) {
            fail(err, err_len, "role %s: tp_size %d does not divide %lld columns",
                 kRoleNames[role], tp, global_cols);
            return -1;
        }
        out->cols = global_cols / tp;
        out->col_off = out->cols * rank;
        return 0;
    default:
        fail(err, err_len, "role %s has an unknown shard rule %d", kRoleNames[role], rule);
        return -1;
    }
}

/* ------------------------------------------------------------------ */
/* Validation                                                         */
/* ------------------------------------------------------------------ */

static int require_role(const struct ModelDesc *d, int role, const char *why,
                        char *err, size_t err_len) {
    if (model_desc_role_index(d, role) < 0) {
        fail(err, err_len, "descriptor has no '%s' template (%s)", kRoleNames[role], why);
        return -1;
    }
    return 0;
}

/* The rule a role carries, or -1 when the descriptor does not use the role. */
static int role_rule(const struct ModelDesc *d, int role) {
    const int slot = model_desc_role_index(d, role);
    return slot >= 0 && slot < d->role_shard_count ? d->role_shards[slot] : -1;
}

static int require_role_rule(const struct ModelDesc *d, int role, int rule,
                             char *err, size_t err_len) {
    const int found = role_rule(d, role);
    if (found < 0 || found == rule) return 0;
    fail(err, err_len, "tp_size %d: role %s must carry the '%s' shard rule (it carries '%s')",
         d->tp_size, kRoleNames[role], kShardNames[rule], kShardNames[found]);
    return -1;
}

/* With tp_size > 1 every rank works with the *divided* dimensions (attention
 * heads, KV heads, the dense MLP), so the roles whose shapes those dimensions
 * describe have to carry the matching shard rule. A role left at 'none' under
 * divided dimensions makes the forward read the wrong rows of the tensor -- and
 * for an output projection it reads past the end of the destination buffer --
 * with no error anywhere. GDN roles are not divided, so they stay free. */
static int validate_tp_roles(const struct ModelDesc *d, char *err, size_t err_len) {
    if (d->tp_size <= 1) return 0;
    for (int i = 0; i < d->num_layers; ++i) {
        if (d->layer_mixers[i] == ENGINE_MIXER_FULL_ATTN) {
            if (require_role_rule(d, ROLE_ATTN_Q, ENGINE_SHARD_OUT_HEADS, err, err_len) != 0 ||
                require_role_rule(d, ROLE_ATTN_K, ENGINE_SHARD_OUT_HEADS, err, err_len) != 0 ||
                require_role_rule(d, ROLE_ATTN_V, ENGINE_SHARD_OUT_HEADS, err, err_len) != 0 ||
                require_role_rule(d, ROLE_ATTN_O, ENGINE_SHARD_IN_DIM, err, err_len) != 0)
                return -1;
        } else if (d->layer_mixers[i] == ENGINE_MIXER_MLA) {
            if (require_role_rule(d, ROLE_MLA_Q, ENGINE_SHARD_OUT_HEADS, err, err_len) != 0 ||
                require_role_rule(d, ROLE_MLA_KV_B, ENGINE_SHARD_OUT_HEADS, err, err_len) != 0 ||
                require_role_rule(d, ROLE_MLA_O, ENGINE_SHARD_IN_DIM, err, err_len) != 0)
                return -1;
        }
        if (d->layer_ffns[i] == ENGINE_FFN_DENSE) {
            if (require_role_rule(d, ROLE_MLP_GATE, ENGINE_SHARD_OUT_DIM, err, err_len) != 0 ||
                require_role_rule(d, ROLE_MLP_UP, ENGINE_SHARD_OUT_DIM, err, err_len) != 0 ||
                require_role_rule(d, ROLE_MLP_DOWN, ENGINE_SHARD_IN_DIM, err, err_len) != 0)
                return -1;
        }
    }
    return 0;
}

int model_desc_validate(const struct ModelDesc *d, char *err, size_t err_len) {
    if (d->num_layers <= 0 || d->num_layers > ENGINE_MAX_LAYERS) {
        fail(err, err_len, "num_layers %d out of range", d->num_layers);
        return -1;
    }
    if (d->hidden_size <= 0 || d->vocab_size <= 0 || d->intermediate_size <= 0) {
        fail(err, err_len, "hidden/intermediate/vocab sizes must be positive");
        return -1;
    }
    if (d->max_seq_len <= 0 || d->max_seq_len > d->max_position_embeddings) {
        fail(err, err_len, "max_seq_len %d out of range (max_position_embeddings %d)",
             d->max_seq_len, d->max_position_embeddings);
        return -1;
    }
    if (d->num_heads <= 0 || d->num_kv_heads <= 0 || d->num_heads % d->num_kv_heads != 0) {
        fail(err, err_len, "invalid head configuration (%d heads, %d kv heads)",
             d->num_heads, d->num_kv_heads);
        return -1;
    }
    if (d->head_dim <= 0 || d->rotary_dim < 0 || d->rotary_dim > d->head_dim ||
        d->rotary_dim % 2 != 0) {
        fail(err, err_len, "invalid rotary_dim %d for head_dim %d", d->rotary_dim, d->head_dim);
        return -1;
    }
    if (d->rms_eps <= 0 || d->rotary_theta <= 0) {
        fail(err, err_len, "rms_eps and rotary_theta must be positive");
        return -1;
    }
    if (d->max_chunk < 1 || d->max_chunk > 128) {
        fail(err, err_len, "max_chunk %d is outside the supported range [1,128]", d->max_chunk);
        return -1;
    }
    if (d->tp_size < 1) {
        fail(err, err_len, "tp_size %d must be at least 1", d->tp_size);
        return -1;
    }
    if (d->tp_rank < 0 || d->tp_rank >= d->tp_size) {
        fail(err, err_len, "tp_rank %d is outside [0, tp_size=%d)", d->tp_rank, d->tp_size);
        return -1;
    }
    if (d->ep_size < 1) {
        fail(err, err_len, "ep_size %d must be at least 1", d->ep_size);
        return -1;
    }
    if (d->ep_rank < 0 || d->ep_rank >= d->ep_size) {
        fail(err, err_len, "ep_rank %d is outside [0, ep_size=%d)", d->ep_rank, d->ep_size);
        return -1;
    }
    if (d->role_shard_count != d->role_count) {
        fail(err, err_len, "role_shards (%d) and role_names (%d) differ in length",
             d->role_shard_count, d->role_count);
        return -1;
    }
    for (int i = 0; i < d->role_shard_count; ++i) {
        if (d->role_shards[i] < 0 || d->role_shards[i] >= SHARD_KIND_COUNT) {
            fail(err, err_len, "role_shards entry %d is not a known shard rule", i);
            return -1;
        }
        if (!shard_rule_fits_role(d->role_ids[i], d->role_shards[i])) {
            fail(err, err_len, "role_shards uses a rule role %s cannot carry",
                 kRoleNames[d->role_ids[i]]);
            return -1;
        }
    }
    /* The shard rules have to cover the dimensions tensor parallelism divides. */
    if (validate_tp_roles(d, err, err_len) != 0) return -1;
    if (strcmp(d->norm_style, "gemma") != 0 && strcmp(d->norm_style, "plain") != 0) {
        fail(err, err_len, "norm_style must be gemma or plain");
        return -1;
    }
    if (d->eos_count <= 0) {
        fail(err, err_len, "eos_tokens must not be empty");
        return -1;
    }
    for (int i = 0; i < d->eos_count; ++i) {
        if (d->eos_tokens[i] < 0 || d->eos_tokens[i] >= d->vocab_size) {
            fail(err, err_len, "eos token %d is outside the vocabulary", d->eos_tokens[i]);
            return -1;
        }
    }
    /* Role names must be unique. */
    for (int i = 0; i < d->role_count; ++i) {
        for (int j = i + 1; j < d->role_count; ++j) {
            if (d->role_ids[i] == d->role_ids[j]) {
                fail(err, err_len, "duplicate role in descriptor: %s", kRoleNames[d->role_ids[i]]);
                return -1;
            }
        }
        if (d->role_templates[i][0] == '\0') {
            fail(err, err_len, "empty weight template for role %s", kRoleNames[d->role_ids[i]]);
            return -1;
        }
    }
    /* Roles every layer needs, regardless of kind. */
    const int global_roles[] = {ROLE_EMBED, ROLE_LM_HEAD, ROLE_FINAL_NORM,
                               ROLE_INPUT_NORM, ROLE_POST_NORM};
    for (size_t i = 0; i < sizeof(global_roles) / sizeof(global_roles[0]); ++i) {
        if (require_role(d, global_roles[i], "required by every layer", err, err_len) != 0)
            return -1;
    }
    int has_full = 0, has_gdn = 0, has_moe = 0, has_dense = 0, has_mla = 0;
    for (int i = 0; i < d->num_layers; ++i) {
        int mixer = d->layer_mixers[i];
        if (mixer == ENGINE_MIXER_FULL_ATTN) has_full = 1;
        else if (mixer == ENGINE_MIXER_GDN) has_gdn = 1;
        else if (mixer == ENGINE_MIXER_MLA) has_mla = 1;
        else {
            fail(err, err_len, "layer %d has an unsupported mixer kind %d", i, mixer);
            return -1;
        }
        if (d->layer_ffns[i] == ENGINE_FFN_MOE) has_moe = 1;
        else if (d->layer_ffns[i] == ENGINE_FFN_DENSE) has_dense = 1;
        else {
            fail(err, err_len, "layer %d has an unsupported ffn kind %d", i, d->layer_ffns[i]);
            return -1;
        }
    }
    if (has_dense) {
        const int dense_roles[] = {ROLE_MLP_GATE, ROLE_MLP_UP, ROLE_MLP_DOWN};
        for (size_t i = 0; i < sizeof(dense_roles) / sizeof(dense_roles[0]); ++i) {
            if (require_role(d, dense_roles[i], "needed by dense layers", err, err_len) != 0)
                return -1;
        }
    }
    if (has_full) {
        const int attn_roles[] = {ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_O};
        for (size_t i = 0; i < sizeof(attn_roles) / sizeof(attn_roles[0]); ++i) {
            if (require_role(d, attn_roles[i], "needed by full_attn layers", err, err_len) != 0)
                return -1;
        }
        if (d->attn_qk_norm) {
            const int norm_roles[] = {ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM};
            for (size_t i = 0; i < sizeof(norm_roles) / sizeof(norm_roles[0]); ++i) {
                if (require_role(d, norm_roles[i], "attn_qk_norm is enabled", err, err_len) != 0)
                    return -1;
            }
        }
        if (d->num_heads % d->num_kv_heads != 0) {
            fail(err, err_len, "num_heads must be a multiple of num_kv_heads");
            return -1;
        }
    }
    if (has_mla) {
        const int mla_roles[] = {ROLE_MLA_Q, ROLE_MLA_KV_A, ROLE_MLA_KV_A_NORM,
                                 ROLE_MLA_KV_B, ROLE_MLA_O};
        for (size_t i = 0; i < sizeof(mla_roles) / sizeof(mla_roles[0]); ++i) {
            if (require_role(d, mla_roles[i], "needed by mla layers", err, err_len) != 0)
                return -1;
        }
        if (d->mla_kv_lora_rank <= 0 || d->mla_qk_nope_head_dim <= 0 ||
            d->mla_qk_rope_head_dim <= 0 || d->mla_v_head_dim <= 0) {
            fail(err, err_len, "MLA layers need positive kv_lora_rank/qk_nope/qk_rope/v head dims");
            return -1;
        }
        if (d->attn_output_gate) {
            fail(err, err_len, "MLA layers carry no attention output gate");
            return -1;
        }
    }
    if (has_gdn) {
        const int gdn_common[] = {ROLE_GDN_CONV1D, ROLE_GDN_DT_BIAS, ROLE_GDN_A_LOG,
                                  ROLE_GDN_OUT, ROLE_GDN_NORM};
        for (size_t i = 0; i < sizeof(gdn_common) / sizeof(gdn_common[0]); ++i) {
            if (require_role(d, gdn_common[i], "needed by gdn layers", err, err_len) != 0)
                return -1;
        }
        /* Either the projections are separate (Qwen3.5) or fused into qkvz+ba
         * (Qwen3-Next); both layouts are complete. */
        const int separate = model_desc_role_index(d, ROLE_GDN_QKV) >= 0 &&
                              model_desc_role_index(d, ROLE_GDN_Z) >= 0 &&
                              model_desc_role_index(d, ROLE_GDN_A) >= 0 &&
                              model_desc_role_index(d, ROLE_GDN_B) >= 0;
        const int fused = model_desc_role_index(d, ROLE_GDN_QKVZ) >= 0 &&
                           model_desc_role_index(d, ROLE_GDN_BA) >= 0;
        if (!separate && !fused) {
            fail(err, err_len, "gdn layers need either qkv/z/a/b or the fused qkvz/ba roles");
            return -1;
        }
        if (d->gdn_head_dim <= 0 || d->gdn_num_v_heads <= 0 || d->gdn_num_k_heads <= 0 ||
            d->gdn_num_v_heads % d->gdn_num_k_heads != 0) {
            fail(err, err_len, "invalid GDN head configuration");
            return -1;
        }
        if (d->gdn_value_dim != d->gdn_num_v_heads * d->gdn_head_dim) {
            fail(err, err_len, "gdn_value_dim %d != gdn_num_v_heads * gdn_head_dim (%d)",
                 d->gdn_value_dim, d->gdn_num_v_heads * d->gdn_head_dim);
            return -1;
        }
        if (d->gdn_conv_dim != 2 * d->gdn_num_k_heads * d->gdn_head_dim + d->gdn_value_dim) {
            fail(err, err_len, "gdn_conv_dim %d is inconsistent with the GDN head layout",
                 d->gdn_conv_dim);
            return -1;
        }
        if (d->gdn_conv_kernel <= 0) {
            fail(err, err_len, "gdn_conv_kernel must be positive");
            return -1;
        }
        if (d->fla_chunk_size < 1 || d->fla_chunk_size > 128) {
            fail(err, err_len, "fla_chunk_size %d is outside the supported range [1,128]",
                 d->fla_chunk_size);
            return -1;
        }
    }
    if (has_moe) {
        const int moe_roles[] = {ROLE_MOE_ROUTER, ROLE_MOE_EXPERT_GATE,
                                 ROLE_MOE_EXPERT_UP, ROLE_MOE_EXPERT_DOWN};
        for (size_t i = 0; i < sizeof(moe_roles) / sizeof(moe_roles[0]); ++i) {
            if (require_role(d, moe_roles[i], "needed by moe layers", err, err_len) != 0)
                return -1;
        }
        if (d->moe_num_experts <= 0) {
            fail(err, err_len, "moe_num_experts must be positive");
            return -1;
        }
        if (d->moe_top_k < 1 || d->moe_top_k > d->moe_num_experts) {
            fail(err, err_len, "moe_top_k %d is outside [1, %d]", d->moe_top_k, d->moe_num_experts);
            return -1;
        }
        if (d->moe_intermediate_size <= 0) {
            fail(err, err_len, "moe_intermediate_size must be positive");
            return -1;
        }
        if (strcmp(d->moe_router_scoring, "softmax") != 0 &&
            strcmp(d->moe_router_scoring, "sigmoid") != 0) {
            fail(err, err_len, "moe_router_scoring must be softmax or sigmoid");
            return -1;
        }
        if (d->moe_num_shared_experts < 0 || d->moe_shared_intermediate_size < 0 ||
            (d->moe_num_shared_experts > 0 && d->moe_shared_intermediate_size <= 0)) {
            fail(err, err_len, "inconsistent shared-expert configuration");
            return -1;
        }
        if (d->moe_routed_scaling_factor <= 0) {
            fail(err, err_len, "moe_routed_scaling_factor must be positive");
            return -1;
        }
        if (d->moe_shared_gate_scalar && d->moe_num_shared_experts == 0) {
            fail(err, err_len, "moe_shared_gate_scalar needs at least one shared expert");
            return -1;
        }
        if (d->moe_num_shared_experts > 0) {
            const int shared_roles[] = {ROLE_MOE_SHARED_GATE, ROLE_MOE_SHARED_UP,
                                        ROLE_MOE_SHARED_DOWN};
            for (size_t i = 0; i < sizeof(shared_roles) / sizeof(shared_roles[0]); ++i) {
                if (require_role(d, shared_roles[i], "needed by shared experts", err, err_len) != 0)
                    return -1;
            }
            if (d->moe_shared_gate_scalar &&
                model_desc_role_index(d, ROLE_MOE_SHARED_GATE_SCALAR) < 0) {
                fail(err, err_len, "descriptor has no 'moeSharedGateScalar' template "
                                   "(moe_shared_gate_scalar is enabled)");
                return -1;
            }
        }
    }
    return 0;
}

/* Engine-runtime capability checks: a descriptor can be structurally valid yet
 * describe a model the AOT kernels cannot execute. engine_create calls this right
 * after model_desc_validate so an unsupported layout is rejected before any GPU
 * memory is allocated, instead of silently indexing out of bounds in a kernel. */
int model_desc_check_runtime_support(const struct ModelDesc *d, char *err, size_t err_len) {
    int has_gdn = 0;
    for (int i = 0; i < d->num_layers; ++i)
        if (d->layer_mixers[i] == ENGINE_MIXER_GDN) has_gdn = 1;
    if (has_gdn) {
        /* The FLA kernels are AOT-compiled for a fixed 128-wide head and 16 key
         * heads, and recurrent (decode) kernels only exist for 32 and 48 value
         * heads; see csrc/triton/build_aot.py and csrc/kernels/fla_gdn.cu. A
         * different layout would have the kernels read past their buffers. */
        if (d->gdn_head_dim != 128) {
            fail(err, err_len, "gdn_head_dim %d is not supported: the AOT FLA kernels are built for 128",
                 d->gdn_head_dim);
            return -1;
        }
        if (d->gdn_num_k_heads != 16) {
            fail(err, err_len, "gdn_num_k_heads %d is not supported: the AOT FLA kernels are built for 16",
                 d->gdn_num_k_heads);
            return -1;
        }
        if (d->gdn_num_v_heads != 32 && d->gdn_num_v_heads != 48) {
            fail(err, err_len, "gdn_num_v_heads %d is not supported: the AOT recurrent kernels cover 32 and 48",
                 d->gdn_num_v_heads);
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Canonical echo                                                     */
/* ------------------------------------------------------------------ */

static int append(char *buf, int buf_len, int *used, const char *fmt, ...) {
    if (*used >= buf_len) return -1;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *used, (size_t)(buf_len - *used), fmt, ap);
    va_end(ap);
    if (n < 0 || *used + n >= buf_len) return -1;
    *used += n;
    return 0;
}

int model_desc_format(const struct ModelDesc *d, char *buf, int buf_len) {
    int used = 0;
    if (append(buf, buf_len, &used, "{") != 0) return -1;
    if (append(buf, buf_len, &used, "\"desc_version\":%d,", d->version) != 0) return -1;
    if (append(buf, buf_len, &used, "\"family\":\"%s\",", d->family) != 0) return -1;
    if (append(buf, buf_len, &used, "\"model_type\":\"%s\",", d->model_type) != 0) return -1;
    if (append(buf, buf_len, &used, "\"num_layers\":%d,", d->num_layers) != 0) return -1;
    if (append(buf, buf_len, &used, "\"hidden_size\":%d,", d->hidden_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"intermediate_size\":%d,", d->intermediate_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"vocab_size\":%d,", d->vocab_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"rms_eps\":%.17g,", d->rms_eps) != 0) return -1;
    if (append(buf, buf_len, &used, "\"max_position_embeddings\":%d,", d->max_position_embeddings) != 0) return -1;
    if (append(buf, buf_len, &used, "\"max_seq_len\":%d,", d->max_seq_len) != 0) return -1;
    if (append(buf, buf_len, &used, "\"num_heads\":%d,", d->num_heads) != 0) return -1;
    if (append(buf, buf_len, &used, "\"num_kv_heads\":%d,", d->num_kv_heads) != 0) return -1;
    if (append(buf, buf_len, &used, "\"head_dim\":%d,", d->head_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"rotary_dim\":%d,", d->rotary_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"rotary_theta\":%.17g,", d->rotary_theta) != 0) return -1;
    if (append(buf, buf_len, &used, "\"norm_style\":\"%s\",", d->norm_style) != 0) return -1;
    if (append(buf, buf_len, &used, "\"attn_qk_norm\":%s,", d->attn_qk_norm ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"attn_output_gate\":%s,", d->attn_output_gate ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"q_gate_interleave\":%s,", d->q_gate_interleave ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_conv_dim\":%d,", d->gdn_conv_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_value_dim\":%d,", d->gdn_value_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_num_v_heads\":%d,", d->gdn_num_v_heads) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_num_k_heads\":%d,", d->gdn_num_k_heads) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_head_dim\":%d,", d->gdn_head_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_conv_kernel\":%d,", d->gdn_conv_kernel) != 0) return -1;
    if (append(buf, buf_len, &used, "\"fla_chunk_size\":%d,", d->fla_chunk_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"max_chunk\":%d,", d->max_chunk) != 0) return -1;
    if (append(buf, buf_len, &used, "\"tp_size\":%d,", d->tp_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"tp_rank\":%d,", d->tp_rank) != 0) return -1;
    if (append(buf, buf_len, &used, "\"ep_size\":%d,", d->ep_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"ep_rank\":%d,", d->ep_rank) != 0) return -1;
    if (append(buf, buf_len, &used, "\"mla_kv_lora_rank\":%d,", d->mla_kv_lora_rank) != 0) return -1;
    if (append(buf, buf_len, &used, "\"mla_qk_nope_head_dim\":%d,", d->mla_qk_nope_head_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"mla_qk_rope_head_dim\":%d,", d->mla_qk_rope_head_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"mla_v_head_dim\":%d,", d->mla_v_head_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_num_experts\":%d,", d->moe_num_experts) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_top_k\":%d,", d->moe_top_k) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_intermediate_size\":%d,", d->moe_intermediate_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_router_scoring\":\"%s\",", d->moe_router_scoring) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_norm_topk_prob\":%s,", d->moe_norm_topk_prob ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_num_shared_experts\":%d,", d->moe_num_shared_experts) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_shared_intermediate_size\":%d,", d->moe_shared_intermediate_size) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_routed_scaling_factor\":%.17g,", d->moe_routed_scaling_factor) != 0) return -1;
    if (append(buf, buf_len, &used, "\"moe_shared_gate_scalar\":%s,", d->moe_shared_gate_scalar ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"eos_tokens\":[") != 0) return -1;
    for (int i = 0; i < d->eos_count; ++i) {
        if (append(buf, buf_len, &used, "%s%d", i ? "," : "", d->eos_tokens[i]) != 0) return -1;
    }
    if (append(buf, buf_len, &used, "],") != 0) return -1;
    if (append(buf, buf_len, &used, "\"layer_mixers\":[") != 0) return -1;
    for (int i = 0; i < d->num_layers; ++i) {
        const char *name = d->layer_mixers[i] == ENGINE_MIXER_FULL_ATTN ? "full_attn"
                         : d->layer_mixers[i] == ENGINE_MIXER_GDN ? "gdn" : "mla";
        if (append(buf, buf_len, &used, "%s\"%s\"", i ? "," : "", name) != 0) return -1;
    }
    if (append(buf, buf_len, &used, "],") != 0) return -1;
    if (append(buf, buf_len, &used, "\"layer_ffns\":[") != 0) return -1;
    for (int i = 0; i < d->num_layers; ++i) {
        const char *name = d->layer_ffns[i] == ENGINE_FFN_DENSE ? "dense" : "moe";
        if (append(buf, buf_len, &used, "%s\"%s\"", i ? "," : "", name) != 0) return -1;
    }
    if (append(buf, buf_len, &used, "],") != 0) return -1;
    if (append(buf, buf_len, &used, "\"role_names\":[") != 0) return -1;
    for (int i = 0; i < d->role_count; ++i) {
        if (append(buf, buf_len, &used, "%s\"%s\"", i ? "," : "", kRoleNames[d->role_ids[i]]) != 0) return -1;
    }
    if (append(buf, buf_len, &used, "],") != 0) return -1;
    if (append(buf, buf_len, &used, "\"role_templates\":[") != 0) return -1;
    for (int i = 0; i < d->role_count; ++i) {
        if (append(buf, buf_len, &used, "%s\"%s\"", i ? "," : "", d->role_templates[i]) != 0) return -1;
    }
    if (append(buf, buf_len, &used, "],") != 0) return -1;
    if (append(buf, buf_len, &used, "\"role_shards\":[") != 0) return -1;
    for (int i = 0; i < d->role_shard_count; ++i) {
        if (append(buf, buf_len, &used, "%s\"%s\"", i ? "," : "",
                   kShardNames[d->role_shards[i]]) != 0) return -1;
    }
    if (append(buf, buf_len, &used, "]}") != 0) return -1;
    return used;
}

void model_desc_expand(const char *templ, int layer, int expert, char *out, size_t out_len) {
    size_t n = 0;
    for (const char *p = templ; *p; ++p) {
        if (*p == '%' && (p[1] == 'd' || p[1] == 'e')) {
            char digits[24];
            snprintf(digits, sizeof(digits), "%d", p[1] == 'd' ? layer : expert);
            for (const char *q = digits; *q && n + 1 < out_len; ++q) out[n++] = *q;
            ++p;
        } else if (n + 1 < out_len) {
            out[n++] = *p;
        }
    }
    if (out_len > 0) out[n < out_len ? n : out_len - 1] = '\0';
}
