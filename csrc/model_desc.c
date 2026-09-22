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
    K_ROTARY_THETA, K_ATTN_OUTPUT_GATE, K_Q_GATE_INTERLEAVE, K_GDN_CONV_DIM,
    K_GDN_VALUE_DIM, K_GDN_NUM_V_HEADS, K_GDN_NUM_K_HEADS, K_GDN_HEAD_DIM,
    K_GDN_CONV_KERNEL, K_FLA_CHUNK_SIZE, K_EOS_TOKENS, K_LAYER_MIXERS,
    K_LAYER_FFNS, K_ROLE_NAMES, K_ROLE_TEMPLATES,
    K_COUNT
};

static const struct {
    const char *name;
    int type;
} kKeys[K_COUNT] = {
    {"desc_version", KV_INT},
    {"family", KV_TEXT},
    {"model_type", KV_TEXT},
    {"num_layers", KV_INT},
    {"hidden_size", KV_INT},
    {"intermediate_size", KV_INT},
    {"vocab_size", KV_INT},
    {"rms_eps", KV_DOUBLE},
    {"max_position_embeddings", KV_INT},
    {"max_seq_len", KV_INT},
    {"num_heads", KV_INT},
    {"num_kv_heads", KV_INT},
    {"head_dim", KV_INT},
    {"rotary_dim", KV_INT},
    {"rotary_theta", KV_DOUBLE},
    {"attn_output_gate", KV_BOOL},
    {"q_gate_interleave", KV_BOOL},
    {"gdn_conv_dim", KV_INT},
    {"gdn_value_dim", KV_INT},
    {"gdn_num_v_heads", KV_INT},
    {"gdn_num_k_heads", KV_INT},
    {"gdn_head_dim", KV_INT},
    {"gdn_conv_kernel", KV_INT},
    {"fla_chunk_size", KV_INT},
    {"eos_tokens", KV_INT_ARRAY},
    {"layer_mixers", KV_TEXT_ARRAY},
    {"layer_ffns", KV_TEXT_ARRAY},
    {"role_names", KV_TEXT_ARRAY},
    {"role_templates", KV_TEXT_ARRAY},
};

static int key_id(const char *name) {
    for (int i = 0; i < K_COUNT; ++i)
        if (strcmp(kKeys[i].name, name) == 0) return i;
    return -1;
}

/* Role names, in the same order as the Haskell Role enum. */
static const char *kRoleNames[ROLE_COUNT] = {
    "embed", "lmHead", "finalNorm", "inputNorm", "postNorm",
    "mlpGate", "mlpUp", "mlpDown",
    "attnQ", "attnK", "attnV", "attnO", "attnQNorm", "attnKNorm",
    "gdnQkv", "gdnZ", "gdnA", "gdnB", "gdnConv1d", "gdnDtBias", "gdnALog",
    "gdnOut", "gdnNorm",
    "moeRouter", "moeRouterBias",
    "moeExpertGate", "moeExpertUp", "moeExpertDown",
    "moeSharedGate", "moeSharedUp", "moeSharedDown", "moeSharedGateScalar",
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
    unsigned char seen[K_COUNT] = {0};
    int eos_count = 0, mixer_count = 0, ffn_count = 0, role_count = 0, template_count = 0;
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
        case K_GDN_CONV_KERNEL: case K_FLA_CHUNK_SIZE:
            if (parse_number(&c, &number) != 0) {
                fail(err, err_len, "key %s must be a number", key);
                return -1;
            }
            break;
        case K_RMS_EPS: case K_ROTARY_THETA:
            if (parse_number(&c, &number) != 0) {
                fail(err, err_len, "key %s must be a number", key);
                return -1;
            }
            break;
        case K_ATTN_OUTPUT_GATE: case K_Q_GATE_INTERLEAVE:
            if (parse_bool(&c, &boolean) != 0) {
                fail(err, err_len, "key %s must be a boolean", key);
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
        case K_ATTN_OUTPUT_GATE: out->attn_output_gate = boolean; break;
        case K_Q_GATE_INTERLEAVE: out->q_gate_interleave = boolean; break;
        case K_GDN_CONV_DIM: out->gdn_conv_dim = (int)number; break;
        case K_GDN_VALUE_DIM: out->gdn_value_dim = (int)number; break;
        case K_GDN_NUM_V_HEADS: out->gdn_num_v_heads = (int)number; break;
        case K_GDN_NUM_K_HEADS: out->gdn_num_k_heads = (int)number; break;
        case K_GDN_HEAD_DIM: out->gdn_head_dim = (int)number; break;
        case K_GDN_CONV_KERNEL: out->gdn_conv_kernel = (int)number; break;
        case K_FLA_CHUNK_SIZE: out->fla_chunk_size = (int)number; break;
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
    out->role_count = role_count;
    for (int i = 0; i < K_COUNT; ++i) {
        if (!seen[i]) {
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
    /* Global roles are always required. */
    const int global_roles[] = {ROLE_EMBED, ROLE_LM_HEAD, ROLE_FINAL_NORM,
                               ROLE_INPUT_NORM, ROLE_POST_NORM, ROLE_MLP_GATE,
                               ROLE_MLP_UP, ROLE_MLP_DOWN};
    for (size_t i = 0; i < sizeof(global_roles) / sizeof(global_roles[0]); ++i) {
        if (require_role(d, global_roles[i], "required by every layer", err, err_len) != 0)
            return -1;
    }
    int has_full = 0, has_gdn = 0, has_moe = 0;
    for (int i = 0; i < d->num_layers; ++i) {
        int mixer = d->layer_mixers[i];
        if (mixer == ENGINE_MIXER_FULL_ATTN) has_full = 1;
        else if (mixer == ENGINE_MIXER_GDN) has_gdn = 1;
        else if (mixer != ENGINE_MIXER_MLA) {
            fail(err, err_len, "layer %d has an unsupported mixer kind %d", i, mixer);
            return -1;
        }
        if (d->layer_ffns[i] == ENGINE_FFN_MOE) has_moe = 1;
        else if (d->layer_ffns[i] != ENGINE_FFN_DENSE) {
            fail(err, err_len, "layer %d has an unsupported ffn kind %d", i, d->layer_ffns[i]);
            return -1;
        }
    }
    if (has_full) {
        const int attn_roles[] = {ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_O,
                                  ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM};
        for (size_t i = 0; i < sizeof(attn_roles) / sizeof(attn_roles[0]); ++i) {
            if (require_role(d, attn_roles[i], "needed by full_attn layers", err, err_len) != 0)
                return -1;
        }
        if (d->num_heads % d->num_kv_heads != 0) {
            fail(err, err_len, "num_heads must be a multiple of num_kv_heads");
            return -1;
        }
    }
    if (has_gdn) {
        const int gdn_roles[] = {ROLE_GDN_QKV, ROLE_GDN_Z, ROLE_GDN_A, ROLE_GDN_B,
                                 ROLE_GDN_CONV1D, ROLE_GDN_DT_BIAS, ROLE_GDN_A_LOG,
                                 ROLE_GDN_OUT, ROLE_GDN_NORM};
        for (size_t i = 0; i < sizeof(gdn_roles) / sizeof(gdn_roles[0]); ++i) {
            if (require_role(d, gdn_roles[i], "needed by gdn layers", err, err_len) != 0)
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
        fail(err, err_len, "moe layers are not implemented yet");
        return -1;
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
    if (append(buf, buf_len, &used, "\"attn_output_gate\":%s,", d->attn_output_gate ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"q_gate_interleave\":%s,", d->q_gate_interleave ? "true" : "false") != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_conv_dim\":%d,", d->gdn_conv_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_value_dim\":%d,", d->gdn_value_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_num_v_heads\":%d,", d->gdn_num_v_heads) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_num_k_heads\":%d,", d->gdn_num_k_heads) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_head_dim\":%d,", d->gdn_head_dim) != 0) return -1;
    if (append(buf, buf_len, &used, "\"gdn_conv_kernel\":%d,", d->gdn_conv_kernel) != 0) return -1;
    if (append(buf, buf_len, &used, "\"fla_chunk_size\":%d,", d->fla_chunk_size) != 0) return -1;
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
