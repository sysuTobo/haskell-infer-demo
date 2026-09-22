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
    "\"norm_style\":\"gemma\","
    "\"attn_qk_norm\":true,"
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
    "\"moe_num_experts\":0,"
    "\"moe_top_k\":0,"
    "\"moe_intermediate_size\":0,"
    "\"moe_router_scoring\":\"softmax\","
    "\"moe_norm_topk_prob\":false,"
    "\"moe_num_shared_experts\":0,"
    "\"moe_shared_intermediate_size\":0,"
    "\"moe_routed_scaling_factor\":1.0,"
    "\"moe_shared_gate_scalar\":false,"
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
    /* The tensor-parallel keys are optional: absent means one replicated rank. */
    check(desc.tp_size == 1 && desc.tp_rank == 0, "tp_size/tp_rank default to one rank");
    check(desc.role_shard_count == desc.role_count, "role_shards defaults to the role count");
    check(desc.role_shards[0] == ENGINE_SHARD_NONE &&
              desc.role_shards[TEST_ROLE_COUNT - 1] == ENGINE_SHARD_NONE,
          "absent role_shards means every role is replicated");

    char expanded[ENGINE_TEMPLATE_MAX];
    model_desc_expand("model.layers.%d.attn.q_proj.weight", 7, 0, expanded, sizeof(expanded));
    check(strcmp(expanded, "model.layers.7.attn.q_proj.weight") == 0, "template %d expansion");
    model_desc_expand("experts.%e.w", 3, 5, expanded, sizeof(expanded));
    check(strcmp(expanded, "experts.5.w") == 0, "template %e expansion");

    /* Canonical echo must be parseable again and preserve the fields. */
    char echo[8192];
    int written = model_desc_format(&desc, echo, sizeof(echo));
    check(written > 0, "canonical echo produced");
    check(strstr(echo, "\"tp_size\":1") != NULL, "canonical echo carries tp_size");
    check(strstr(echo, "\"tp_rank\":0") != NULL, "canonical echo carries tp_rank");
    check(strstr(echo, "\"role_shards\":[\"none\"") != NULL,
          "canonical echo carries role_shards");
    struct ModelDesc again;
    rc = model_desc_parse(echo, &again, err, sizeof(err));
    check(rc == 0, err[0] ? err : "canonical echo does not re-parse");
    check(again.num_layers == desc.num_layers && again.hidden_size == desc.hidden_size &&
              again.role_count == desc.role_count &&
              strcmp(again.model_type, desc.model_type) == 0 &&
              again.layer_mixers[1] == desc.layer_mixers[1] &&
              again.max_seq_len == desc.max_seq_len &&
              again.eos_tokens[0] == desc.eos_tokens[0] &&
              again.tp_size == desc.tp_size && again.tp_rank == desc.tp_rank &&
              again.role_shard_count == desc.role_shard_count &&
              again.role_shards[0] == desc.role_shards[0],
          "round-trip preserves the descriptor");
    /* The echo must fit the documented buffer size used by engine_describe. */
    check(written < (int)sizeof(echo), "canonical echo fits its buffer");
}

/* The good descriptor with tensor parallelism enabled (tp_size 2, rank 1) and a
 * role_shards table exercising the whole rule vocabulary on the roles that can
 * carry it (q/k/v by heads, o/down by input dim, gate/up by output dim; the
 * rest replicated). The table is parallel to role_names (23 entries). */
static void build_tp2(char *buf, size_t buf_len) {
    copy_desc(buf, buf_len);
    set_literal(buf, "\"num_layers\":2,",
        "\"num_layers\":2,"
        "\"tp_size\":2,\"tp_rank\":1,"
        "\"role_shards\":[\"none\",\"none\",\"none\",\"none\",\"none\","
        "\"out_dim\",\"out_dim\",\"in_dim\",\"out_heads\",\"out_heads\","
        "\"out_heads\",\"in_dim\",\"none\",\"none\",\"none\",\"none\","
        "\"none\",\"none\",\"none\",\"none\",\"none\",\"none\",\"none\"],");
}

static void test_tp(void) {
    char buf[8192];
    struct ModelDesc desc;
    char err[256] = {0};
    build_tp2(buf, sizeof(buf));
    int rc = model_desc_parse(buf, &desc, err, sizeof(err));
    check(rc == 0, err[0] ? err : "well-formed TP2 descriptor is rejected");
    check(desc.tp_size == 2 && desc.tp_rank == 1, "tp_size/tp_rank parsed");
    check(desc.role_shard_count == desc.role_count, "role_shards length matches the roles");
    check(desc.role_shards[model_desc_role_index(&desc, ROLE_EMBED)] == ENGINE_SHARD_NONE,
          "replicated role parsed");
    check(desc.role_shards[model_desc_role_index(&desc, ROLE_ATTN_Q)] == ENGINE_SHARD_OUT_HEADS,
          "out_heads rule parsed");
    check(desc.role_shards[model_desc_role_index(&desc, ROLE_MLP_UP)] == ENGINE_SHARD_OUT_DIM,
          "out_dim rule parsed");
    check(desc.role_shards[model_desc_role_index(&desc, ROLE_MLP_DOWN)] == ENGINE_SHARD_IN_DIM,
          "in_dim rule parsed");

    /* The echo must reproduce the rules, not just the scalar keys. */
    char echo[8192];
    int written = model_desc_format(&desc, echo, sizeof(echo));
    check(written > 0 && strstr(echo, "\"tp_size\":2") != NULL &&
              strstr(echo, "\"tp_rank\":1") != NULL &&
              strstr(echo, "\"out_heads\"") != NULL &&
              strstr(echo, "\"in_dim\"") != NULL,
          "canonical echo carries the tensor-parallel keys");
    struct ModelDesc again;
    rc = model_desc_parse(echo, &again, err, sizeof(err));
    check(rc == 0, err[0] ? err : "TP2 echo does not re-parse");
    check(again.tp_size == 2 && again.tp_rank == 1 &&
              again.role_shard_count == desc.role_shard_count,
          "TP2 echo round-trips");

    /* tp_rank must be inside [0, tp_size). */
    snprintf(buf, sizeof(buf), "%s", kGoodDesc);
    set_literal(buf, "\"num_layers\":2,", "\"num_layers\":2,\"tp_size\":2,\"tp_rank\":2,");
    test_rejects(buf, "tp_rank", "tp_rank >= tp_size is rejected");
    snprintf(buf, sizeof(buf), "%s", kGoodDesc);
    set_literal(buf, "\"num_layers\":2,", "\"num_layers\":2,\"tp_size\":2,\"tp_rank\":-1,");
    test_rejects(buf, "tp_rank", "negative tp_rank is rejected");
    build_tp2(buf, sizeof(buf));
    set_literal(buf, "\"tp_size\":2,", "\"tp_size\":0,");
    test_rejects(buf, "tp_size", "tp_size below 1 is rejected");

    /* role_shards must be parallel to the role tables. */
    build_tp2(buf, sizeof(buf));
    set_literal(buf, "\"role_shards\":[\"none\",", "\"role_shards\":[");
    test_rejects(buf, "role_shards", "short role_shards table is rejected");
    snprintf(buf, sizeof(buf), "%s", kGoodDesc);
    set_literal(buf, "\"num_layers\":2,", "\"num_layers\":2,\"role_shards\":[],");
    test_rejects(buf, "role_shards", "empty role_shards table is rejected");

    /* Unknown shard name. */
    build_tp2(buf, sizeof(buf));
    set_literal(buf, "\"out_heads\"", "\"out_heds\"");
    test_rejects(buf, "unknown name", "unknown shard rule name is rejected");

    /* Rules the role's layout cannot carry. */
    build_tp2(buf, sizeof(buf));
    set_literal(buf, "\"role_shards\":[\"none\",\"none\"",
                    "\"role_shards\":[\"out_dim\",\"none\"");
    test_rejects(buf, "cannot carry", "out_dim on the embedding is rejected");
    snprintf(buf, sizeof(buf), "%s", kGoodDesc);
    set_literal(buf, "\"num_layers\":2,",
        "\"num_layers\":2,\"tp_size\":2,\"tp_rank\":0,\"role_shards\":["
        "\"none\",\"none\",\"none\",\"none\",\"none\",\"none\",\"none\",\"none\","
        "\"out_heads\",\"out_heads\",\"out_heads\",\"in_dim\",\"none\",\"none\","
        "\"out_heads\",\"none\",\"none\",\"none\",\"none\",\"none\",\"none\","
        "\"none\",\"none\"],");
    test_rejects(buf, "cannot carry", "out_heads on a GDN projection is rejected");
}

/* The rank's slice of a tensor, as the loader computes it. With tp_size 2:
 * q rows = num_heads * head_dim * 2 (fused gate) = 16 -> 8 rows per rank;
 * gate/up [16,8] -> 8 rows per rank; down [8,16] -> 8 columns per rank; every
 * replicated role keeps the whole tensor. Both ranks are checked because the
 * slice must come from the explicit rank, not from desc.tp_rank (one engine
 * process holds every rank and keeps tp_rank at 0). */
static void test_shard_view(void) {
    char buf[8192];
    struct ModelDesc desc;
    char err[256] = {0};
    struct ShardView view;
    build_tp2(buf, sizeof(buf));
    int rc = model_desc_parse(buf, &desc, err, sizeof(err));
    check(rc == 0, err[0] ? err : "TP2 descriptor is rejected");
    check(desc.tp_rank == 1, "the fixture still carries a non-zero tp_rank");

    rc = model_desc_shard_view(&desc, ROLE_ATTN_Q, 16, 8, 0, &view, err, sizeof(err));
    check(rc == 0 && view.row_off == 0 && view.rows == 8 && view.col_off == 0 && view.cols == 8,
          "rank 0 keeps the first q head block");
    rc = model_desc_shard_view(&desc, ROLE_ATTN_Q, 16, 8, 1, &view, err, sizeof(err));
    check(rc == 0 && view.row_off == 8 && view.rows == 8 && view.col_off == 0 && view.cols == 8,
          "rank 1 keeps the second q head block");

    rc = model_desc_shard_view(&desc, ROLE_ATTN_K, 4, 8, 1, &view, err, sizeof(err));
    check(rc != 0 && strstr(err, "does not divide") != NULL,
          "a head count that does not divide tp_size is rejected");

    rc = model_desc_shard_view(&desc, ROLE_MLP_UP, 16, 8, 1, &view, err, sizeof(err));
    check(rc == 0 && view.row_off == 8 && view.rows == 8 && view.col_off == 0 && view.cols == 8,
          "mlpUp shard is the rank's row block");

    rc = model_desc_shard_view(&desc, ROLE_MLP_DOWN, 8, 16, 1, &view, err, sizeof(err));
    check(rc == 0 && view.row_off == 0 && view.rows == 8 && view.col_off == 8 && view.cols == 8,
          "mlpDown shard is the rank's column block");

    rc = model_desc_shard_view(&desc, ROLE_GDN_QKV, 12, 8, 1, &view, err, sizeof(err));
    check(rc == 0 && view.row_off == 0 && view.rows == 12 && view.col_off == 0 && view.cols == 8,
          "a replicated role keeps the whole tensor");

    rc = model_desc_shard_view(&desc, ROLE_ATTN_Q, 16, 8, 2, &view, err, sizeof(err));
    check(rc != 0 && strstr(err, "rank") != NULL, "a rank outside [0, tp_size) is rejected");

    copy_desc(buf, sizeof(buf));
    rc = model_desc_parse(buf, &desc, err, sizeof(err));
    check(rc == 0, err[0] ? err : "single-rank descriptor is rejected");
    rc = model_desc_shard_view(&desc, ROLE_ATTN_Q, 16, 8, 0, &view, err, sizeof(err));
    check(rc == 0 && view.row_off == 0 && view.rows == 16 && view.cols == 8,
          "tp_size 1 yields the whole tensor");
}


/* A dense-attention + MoE descriptor (Mixtral/Qwen3-MoE shape): every layer is
 * full attention with an MoE feed-forward and one shared expert. */
static const char *kMoeDesc =
    "{"
    "\"desc_version\":1,"
    "\"family\":\"mixtral\","
    "\"model_type\":\"mixtral\","
    "\"num_layers\":2,"
    "\"hidden_size\":8,"
    "\"intermediate_size\":16,"
    "\"vocab_size\":32,"
    "\"rms_eps\":1e-05,"
    "\"max_position_embeddings\":128,"
    "\"max_seq_len\":64,"
    "\"num_heads\":2,"
    "\"num_kv_heads\":1,"
    "\"head_dim\":4,"
    "\"rotary_dim\":4,"
    "\"rotary_theta\":1000000,"
    "\"norm_style\":\"plain\","
    "\"attn_qk_norm\":false,"
    "\"attn_output_gate\":false,"
    "\"q_gate_interleave\":false,"
    "\"gdn_conv_dim\":0,"
    "\"gdn_value_dim\":0,"
    "\"gdn_num_v_heads\":0,"
    "\"gdn_num_k_heads\":0,"
    "\"gdn_head_dim\":0,"
    "\"gdn_conv_kernel\":0,"
    "\"fla_chunk_size\":64,"
    "\"max_chunk\":128,"
    "\"moe_num_experts\":4,"
    "\"moe_top_k\":2,"
    "\"moe_intermediate_size\":6,"
    "\"moe_router_scoring\":\"softmax\","
    "\"moe_norm_topk_prob\":true,"
    "\"moe_num_shared_experts\":1,"
    "\"moe_shared_intermediate_size\":6,"
    "\"moe_routed_scaling_factor\":1.0,"
    "\"moe_shared_gate_scalar\":false,"
    "\"eos_tokens\":[2],"
    "\"layer_mixers\":[\"full_attn\",\"full_attn\"],"
    "\"layer_ffns\":[\"moe\",\"moe\"],"
    "\"role_names\":[\"embed\",\"lmHead\",\"finalNorm\",\"inputNorm\",\"postNorm\","
    "\"attnQ\",\"attnK\",\"attnV\",\"attnO\",\"attnQNorm\",\"attnKNorm\","
    "\"moeRouter\",\"moeExpertGate\",\"moeExpertUp\",\"moeExpertDown\","
    "\"moeSharedGate\",\"moeSharedUp\",\"moeSharedDown\"],"
    "\"role_templates\":[\"e.w\",\"h.w\",\"fn.w\",\"in.w\",\"pn.w\",\"q.w\",\"k.w\","
    "\"v.w\",\"o.w\",\"qn.w\",\"kn.w\",\"r.w\",\"eg.%e.w\",\"eu.%e.w\",\"ed.%e.w\","
    "\"sg.w\",\"su.w\",\"sd.w\"]"
    "}";

static void test_moe(void) {
    char err[256] = {0};
    struct ModelDesc desc;
    int rc = model_desc_parse(kMoeDesc, &desc, err, sizeof(err));
    check(rc == 0, err[0] ? err : "well-formed MoE descriptor is rejected");
    check(desc.moe_num_experts == 4 && desc.moe_top_k == 2, "MoE routing parsed");
    check(desc.moe_num_shared_experts == 1 && desc.moe_shared_intermediate_size == 6,
          "shared expert parsed");
    check(strcmp(desc.moe_router_scoring, "softmax") == 0, "router scoring parsed");
    check(desc.moe_norm_topk_prob == 1, "norm_topk_prob parsed");
    check(model_desc_role_index(&desc, ROLE_MOE_ROUTER) == 11, "MoE router role lookup");
    /* Templates expand per expert. */
    char expanded[ENGINE_TEMPLATE_MAX];
    model_desc_expand(desc.role_templates[model_desc_role_index(&desc, ROLE_MOE_EXPERT_UP)],
                      3, 7, expanded, sizeof(expanded));
    check(strcmp(expanded, "eu.7.w") == 0, "expert template expansion");

    char buf[8192];
    /* top_k beyond the expert count. */
    copy_desc(buf, sizeof(buf));
    snprintf(buf, sizeof(buf), "%s", kMoeDesc);
    set_literal(buf, "\"moe_top_k\":2", "\"moe_top_k\":9");
    test_rejects(buf, "moe_top_k", "top_k beyond num_experts is rejected");
    /* Unknown scoring function. */
    snprintf(buf, sizeof(buf), "%s", kMoeDesc);
    set_literal(buf, "\"moe_router_scoring\":\"softmax\"", "\"moe_router_scoring\":\"topk\"");
    test_rejects(buf, "moe_router_scoring", "unknown router scoring is rejected");
    /* Missing expert role while MoE layers exist. */
    snprintf(buf, sizeof(buf), "%s", kMoeDesc);
    set_literal(buf, "\"moeSharedGate\",\"moeSharedUp\",\"moeSharedDown\"]", "\"moeSharedGate\"]");
    set_literal(buf, "\"sg.w\",\"su.w\",\"sd.w\"]", "\"sg.w\"]");
    test_rejects(buf, "moeSharedUp", "missing shared-expert roles are rejected");
    /* Dense MLP roles are not required for an all-MoE model (the fixture above
     * proves it) but must be present when a dense layer exists. */
    snprintf(buf, sizeof(buf), "%s", kMoeDesc);
    set_literal(buf, "\"layer_ffns\":[\"moe\",\"moe\"]", "\"layer_ffns\":[\"moe\",\"dense\"]");
    test_rejects(buf, "mlpGate", "dense layer without MLP roles is rejected");
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

    /* Fused GDN projections: replacing the four separate roles with qkvz+ba
     * describes the same layer, so the descriptor must still validate. */
    snprintf(buf, sizeof(buf), "%s", kGoodDesc);
    set_literal(buf, "\"gdnQkv\",\"gdnZ\",\"gdnA\",\"gdnB\",", "\"gdnQkvz\",\"gdnBa\",");
    set_literal(buf, "\"gq.w\",\"gz.w\",\"ga.w\",\"gb.w\",", "\"gqz.w\",\"gba.w\",");
    {
        struct ModelDesc fused;
        char err_fused[256] = {0};
        int rc = model_desc_parse(buf, &fused, err_fused, sizeof(err_fused));
        check(rc == 0, err_fused[0] ? err_fused : "fused GDN roles are rejected");
        check(model_desc_role_index(&fused, ROLE_GDN_QKVZ) >= 0, "fused qkvz role present");
    }

    /* Neither layout complete: the layer has no usable projections. */
    snprintf(buf, sizeof(buf), "%s", kGoodDesc);
    set_literal(buf, "\"gdnQkv\",\"gdnZ\",\"gdnA\",\"gdnB\",", "\"gdnQkv\",");
    set_literal(buf, "\"gq.w\",\"gz.w\",\"ga.w\",\"gb.w\",", "\"gq.w\",");
    test_rejects(buf, "qkv/z/a/b or the fused", "incomplete GDN projections are rejected");

    /* Gate enabled without a shared-expert gate template. */
    snprintf(buf, sizeof(buf), "%s", kMoeDesc);
    set_literal(buf, "\"moe_shared_gate_scalar\":false", "\"moe_shared_gate_scalar\":true");
    test_rejects(buf, "moeSharedGateScalar", "shared gate without its template is rejected");

    /* Gate enabled without any shared expert to scale. */
    snprintf(buf, sizeof(buf), "%s", kMoeDesc);
    set_literal(buf, "\"moe_shared_gate_scalar\":false", "\"moe_shared_gate_scalar\":true");
    set_literal(buf, "\"moe_num_shared_experts\":1", "\"moe_num_shared_experts\":0");
    test_rejects(buf, "at least one shared expert", "shared gate without shared experts is rejected");

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
    test_tp();
    test_shard_view();
    test_moe();
    test_errors();
    if (failures == 0) {
        printf("model_desc tests passed\n");
        return 0;
    }
    fprintf(stderr, "%d model_desc test(s) failed\n", failures);
    return 1;
}
