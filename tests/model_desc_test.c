/* Unit tests for the descriptor parser: accepts a well-formed flat descriptor,
 * rejects unknown/missing/duplicated keys and inconsistent dimensions, and
 * round-trips through the canonical echo. Runs without a GPU (part of ctest). */
#include "model_desc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char *what) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

/* A minimal descriptor exercising the whole schema (not a real checkpoint).
 * GDN consistency: value_dim = v_heads * head_dim = 4, conv_dim = 2*k*d + v = 12. */
static const char *kGoodDesc =
    "{"
    "\"desc_version\":1,"
    "\"family\":\"test\","
    "\"model_type\":\"test_text\","
    "\"num_layers\":2,"
    "\"hidden_size\":8,"
    "\"intermediate_size\":16,"
    "\"vocab_size\":32,"
    "\"rms_eps\":1e-06,"
    "\"max_position_embeddings\":128,"
    "\"max_seq_len\":64,"
    "\"num_heads\":2,"
    "\"num_kv_heads\":1,"
    "\"head_dim\":4,"
    "\"rotary_dim\":2,"
    "\"rotary_theta\":10000000,"
    "\"attn_output_gate\":true,"
    "\"q_gate_interleave\":true,"
    "\"gdn_conv_dim\":12,"
    "\"gdn_value_dim\":4,"
    "\"gdn_num_v_heads\":1,"
    "\"gdn_num_k_heads\":1,"
    "\"gdn_head_dim\":4,"
    "\"gdn_conv_kernel\":4,"
    "\"fla_chunk_size\":64,"
    "\"max_chunk\":128,"
    "\"eos_tokens\":[7,9],"
    "\"layer_mixers\":[\"full_attn\",\"gdn\"],"
    "\"layer_ffns\":[\"dense\",\"dense\"],"
    "\"role_names\":[\"embed\",\"lmHead\",\"finalNorm\",\"inputNorm\",\"postNorm\","
    "\"mlpGate\",\"mlpUp\",\"mlpDown\",\"attnQ\",\"attnK\",\"attnV\",\"attnO\","
    "\"attnQNorm\",\"attnKNorm\",\"gdnQkv\",\"gdnZ\",\"gdnA\",\"gdnB\",\"gdnConv1d\","
    "\"gdnDtBias\",\"gdnALog\",\"gdnOut\",\"gdnNorm\"],"
    "\"role_templates\":[\"e.w\",\"h.w\",\"fn.w\",\"in.w\",\"pn.w\",\"mg.w\",\"mu.w\","
    "\"md.w\",\"q.w\",\"k.w\",\"v.w\",\"o.w\",\"qn.w\",\"kn.w\",\"gq.w\",\"gz.w\","
    "\"ga.w\",\"gb.w\",\"gc.w\",\"gd.w\",\"ge.w\",\"go.w\",\"gn.w\"]"
    "}";

#define TEST_ROLE_COUNT 23

static void copy_desc(char *buf, size_t buf_len) {
    snprintf(buf, buf_len, "%s", kGoodDesc);
}

/* Replace the first occurrence of a literal, shifting the tail. */
static void set_literal(char *buf, const char *old, const char *replacement) {
    char *pos = strstr(buf, old);
    if (pos == NULL) {
        fprintf(stderr, "FAIL: test setup could not find %s\n", old);
        ++failures;
        return;
    }
    size_t old_len = strlen(old), new_len = strlen(replacement);
    memmove(pos + new_len, pos + old_len, strlen(pos + old_len) + 1);
    memcpy(pos, replacement, new_len);
}

static void test_good(void) {
    struct ModelDesc desc;
    char err[256] = {0};
    int rc = model_desc_parse(kGoodDesc, &desc, err, sizeof(err));
    check(rc == 0, err[0] ? err : "well-formed descriptor is rejected");
    check(desc.num_layers == 2, "num_layers parsed");
    check(desc.hidden_size == 8, "hidden_size parsed");
    check(desc.attn_output_gate == 1, "attn_output_gate parsed");
    check(desc.max_seq_len == 64, "max_seq_len parsed");
    check(desc.max_chunk == 128, "max_chunk parsed");
    check(desc.layer_mixers[0] == ENGINE_MIXER_FULL_ATTN, "layer 0 mixer");
    check(desc.layer_mixers[1] == ENGINE_MIXER_GDN, "layer 1 mixer");
    check(desc.eos_count == 2 && desc.eos_tokens[1] == 9, "eos tokens parsed");
    check(desc.role_count == TEST_ROLE_COUNT, "role count");
    check(model_desc_role_index(&desc, ROLE_GDN_NORM) == TEST_ROLE_COUNT - 1, "role index lookup");
    check(model_desc_role_index(&desc, ROLE_MOE_ROUTER) == -1, "absent role lookup");

    char expanded[ENGINE_TEMPLATE_MAX];
    model_desc_expand("model.layers.%d.attn.q_proj.weight", 7, 0, expanded, sizeof(expanded));
    check(strcmp(expanded, "model.layers.7.attn.q_proj.weight") == 0, "template %d expansion");
    model_desc_expand("experts.%e.w", 3, 5, expanded, sizeof(expanded));
    check(strcmp(expanded, "experts.5.w") == 0, "template %e expansion");

    /* Canonical echo must be parseable again and preserve the fields. */
    char echo[8192];
    int written = model_desc_format(&desc, echo, sizeof(echo));
    check(written > 0, "canonical echo produced");
    struct ModelDesc again;
    rc = model_desc_parse(echo, &again, err, sizeof(err));
    check(rc == 0, err[0] ? err : "canonical echo does not re-parse");
    check(again.num_layers == desc.num_layers && again.hidden_size == desc.hidden_size &&
              again.role_count == desc.role_count &&
              strcmp(again.model_type, desc.model_type) == 0 &&
              again.layer_mixers[1] == desc.layer_mixers[1] &&
              again.max_seq_len == desc.max_seq_len &&
              again.eos_tokens[0] == desc.eos_tokens[0],
          "round-trip preserves the descriptor");
    /* The echo must fit the documented buffer size used by engine_describe. */
    check(written < (int)sizeof(echo), "canonical echo fits its buffer");
}

static void test_rejects(char *json, const char *needle, const char *what) {
    struct ModelDesc desc;
    char err[256] = {0};
    int rc = model_desc_parse(json, &desc, err, sizeof(err));
    if (rc == 0) {
        fprintf(stderr, "FAIL: %s (accepted)\n", what);
        ++failures;
        return;
    }
    if (needle != NULL && strstr(err, needle) == NULL) {
        fprintf(stderr, "FAIL: %s (error was '%s', expected to mention '%s')\n", what, err, needle);
        ++failures;
    }
}

static void test_errors(void) {
    char buf[8192];
    test_rejects("{\"desc_version\":1}", "missing", "descriptor with missing keys is rejected");
    test_rejects("{", "key name", "truncated descriptor is rejected");

    /* Unknown key. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"rms_eps\":1e-06,", "\"rms_eps\":1e-06,\"surprise\":1,");
    test_rejects(buf, "unknown descriptor key", "unknown key is rejected");

    /* Duplicate key. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"num_layers\":2,", "\"num_layers\":2,\"num_layers\":3,");
    test_rejects(buf, "duplicate", "duplicate key is rejected");

    /* Inconsistent GDN dimensions. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"gdn_conv_dim\":12", "\"gdn_conv_dim\":13");
    test_rejects(buf, "gdn_conv_dim", "inconsistent GDN dims are rejected");

    /* max_seq_len beyond max_position_embeddings. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"max_seq_len\":64", "\"max_seq_len\":999");
    test_rejects(buf, "max_seq_len", "max_seq_len beyond the position limit is rejected");

    /* EOS outside the vocabulary. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"eos_tokens\":[7,9]", "\"eos_tokens\":[7,999]");
    test_rejects(buf, "outside the vocabulary", "out-of-vocab EOS is rejected");

    /* max_chunk beyond the kernels' limit. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"max_chunk\":128", "\"max_chunk\":256");
    test_rejects(buf, "max_chunk", "max_chunk beyond the kernel limit is rejected");

    /* Wrong type for a scalar. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"num_heads\":2", "\"num_heads\":\"two\"");
    test_rejects(buf, "must be a number", "wrong scalar type is rejected");

    /* Missing role template for a layer kind in use: drop gdnNorm while a GDN
     * layer exists. Extra unused roles are allowed; missing ones are not. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"gdnOut\",\"gdnNorm\"]", "\"gdnOut\"]");
    set_literal(buf, "\"go.w\",\"gn.w\"]", "\"go.w\"]");
    test_rejects(buf, "gdnNorm", "descriptor without GDN roles is rejected when GDN layers exist");

    /* Unsupported descriptor version. */
    copy_desc(buf, sizeof(buf));
    set_literal(buf, "\"desc_version\":1", "\"desc_version\":2");
    test_rejects(buf, "version", "unsupported descriptor version is rejected");
}

int main(void) {
    test_good();
    test_errors();
    if (failures == 0) {
        printf("model_desc tests passed\n");
        return 0;
    }
    fprintf(stderr, "%d model_desc test(s) failed\n", failures);
    return 1;
}
