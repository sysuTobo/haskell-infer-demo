/**
 * layer_dispatch.cu - Per-layer forward sequence and kind dispatch.
 *
 * A layer is (token mixer) -> residual -> (feed-forward) -> residual, with an
 * RMSNorm in front of each sublayer. Which mixer and which feed-forward kind a
 * layer uses comes from its plan, so introducing a kind means adding a row to
 * the tables below plus its kernel file -- the engine loop stays untouched.
 */
#include "layers.h"
#include "kernels.h"
#include "model_desc.h"

#include <cstdio>
#include <stdexcept>
#include <string>

/* Fold a pending CUDA error into a kernel status so callers see one code. */
static int with_cuda_error(int status) {
    if (status != 0) return status;
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : (int)error;
}

static int run_mixer(const LayerContext *ctx, const struct LayerWeights *w,
                     const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out);
static int run_ffn(const LayerContext *ctx, const struct LayerWeights *w,
                   const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out);

int forward_mixer(const LayerContext *ctx, const struct LayerWeights *w,
                  const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out) {
    int status = with_cuda_error(run_mixer(ctx, w, residual, layer_out));
    if (status != 0) {
        fprintf(stderr, "[engine] layer %d mixer kind %d failed (status %d)\n",
                ctx->layer_index, w->plan.mixer, status);
        return status;
    }
    tap_dump_rows(ctx->taps, "mixer", ctx->layer_index, ctx->device, ctx->stream, layer_out,
                  ctx->tokens, ctx->dims->hidden_size);
    return 0;
}

int forward_ffn(const LayerContext *ctx, const struct LayerWeights *w,
                const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out) {
    int status = with_cuda_error(run_ffn(ctx, w, residual, layer_out));
    if (status != 0) {
        fprintf(stderr, "[engine] layer %d ffn kind %d failed (status %d)\n",
                ctx->layer_index, w->plan.ffn, status);
        return status;
    }
    tap_dump_rows(ctx->taps, "ffn", ctx->layer_index, ctx->device, ctx->stream, layer_out,
                  ctx->tokens, ctx->dims->hidden_size);
    return 0;
}

static int run_mixer(const LayerContext *ctx, const struct LayerWeights *w,
                     const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out) {
    switch (w->plan.mixer) {
    case ENGINE_MIXER_FULL_ATTN: {
        AttentionWeights aw{};
        aw.q_proj_w = w->q_proj_w;
        aw.k_proj_w = w->k_proj_w;
        aw.v_proj_w = w->v_proj_w;
        aw.o_proj_w = w->o_proj_w;
        aw.q_norm_w = w->q_norm_w;
        aw.k_norm_w = w->k_norm_w;
        aw.input_norm_w = w->input_norm_w;
        return forward_attention_layer(ctx->cublas, ctx->stream, residual, ctx->workspace,
                                       layer_out, &aw, w->kv_cache, ctx->positions,
                                       ctx->tokens, ctx->seq_len, ctx->dims);
    }
    case ENGINE_MIXER_MLA: {
        MlaWeights mw{};
        mw.q_proj_w = w->mla_q_proj_w;
        mw.kv_a_proj_w = w->mla_kv_a_proj_w;
        mw.kv_a_norm_w = w->mla_kv_a_norm_w;
        mw.kv_b_proj_w = w->mla_kv_b_proj_w;
        mw.o_proj_w = w->mla_o_proj_w;
        mw.input_norm_w = w->input_norm_w;
        const GdnTapSites sites{ctx->taps, ctx->layer_index, ctx->device};
        return forward_mla_layer(ctx->cublas, ctx->stream, residual, ctx->workspace,
                                 layer_out, &mw, w->mla_cache, ctx->mla_scratch,
                                 ctx->positions, ctx->tokens, ctx->seq_len, ctx->dims,
                                 &sites);
    }
    case ENGINE_MIXER_GDN: {
        GdnWeights gw{};
        gw.in_proj_qkv_w = w->in_proj_qkv_w;
        gw.in_proj_z_w = w->in_proj_z_w;
        gw.in_proj_a_w = w->in_proj_a_w;
        gw.in_proj_b_w = w->in_proj_b_w;
        gw.conv1d_w = w->conv1d_w;
        gw.conv1d_bias = ctx->conv_bias_zero;
        gw.dt_bias = w->dt_bias;
        gw.A_log = w->A_log;
        gw.out_proj_w = w->gdn_out_proj_w;
        gw.gdn_norm_w = w->gdn_norm_f32;
        gw.input_norm_w = w->input_norm_w;
        const GdnTapSites sites{ctx->taps, ctx->layer_index, ctx->device};
        return forward_gdn_layer(ctx->cublas, ctx->stream, residual, ctx->workspace,
                                 layer_out, &gw, w->conv_state, w->ssm_state,
                                 ctx->fla_scratch, ctx->tokens, ctx->dims, &sites);
    }
    default:
        throw std::runtime_error("unsupported mixer kind " + std::to_string(w->plan.mixer));
    }
}

static int run_ffn(const LayerContext *ctx, const struct LayerWeights *w,
                   const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out) {
    switch (w->plan.ffn) {
    case ENGINE_FFN_DENSE: {
        MlpWeights mw{w->gate_proj_w, w->up_proj_w, w->down_proj_w, w->post_norm_w};
        return forward_mlp(ctx->cublas, ctx->stream, residual, ctx->workspace, layer_out,
                           &mw, ctx->tokens, ctx->dims);
    }
    case ENGINE_FFN_MOE: {
        MoeScratch scratch{ctx->moe_scratch, 0};
        MoeConfig config = w->moe_config;
        if (ctx->reduce == nullptr) {
            return forward_moe_ffn(ctx->cublas, ctx->stream, residual, layer_out, &w->moe,
                                   &config, scratch, ctx->tokens, ctx->dims);
        }
        /* Expert parallelism: the routed experts are split across ranks, so the
         * routed part is only this rank's partial sum. The caller computes every
         * rank's routed half first, reduces, and only then asks for the
         * (replicated) shared experts -- adding them per rank would otherwise
         * count them once per rank. */
        if (ctx->split_phase == 0) {
            return forward_moe_routed(ctx->cublas, ctx->stream, residual, layer_out,
                                      &w->moe, &config, scratch, ctx->tokens, ctx->dims);
        }
        return forward_moe_shared(ctx->cublas, ctx->stream, residual, layer_out, &w->moe,
                                  &config, scratch, ctx->tokens, ctx->dims);
    }
    default:
        throw std::runtime_error("unsupported ffn kind " + std::to_string(w->plan.ffn));
    }
}

int forward_layer(const LayerContext *ctx, const struct LayerWeights *w,
                  const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out) {
    const size_t elements = (size_t)ctx->tokens * ctx->dims->hidden_size;

    int status = forward_mixer(ctx, w, residual, layer_out);
    if (status != 0) return status;
    kernel_residual_add(const_cast<__nv_bfloat16 *>(residual), layer_out, elements, ctx->stream);
    status = with_cuda_error(0);
    if (status != 0) {
        fprintf(stderr, "[engine] layer %d mixer residual failed (status %d)\n",
                ctx->layer_index, status);
        return status;
    }

    status = forward_ffn(ctx, w, residual, layer_out);
    if (status != 0) return status;
    kernel_residual_add(const_cast<__nv_bfloat16 *>(residual), layer_out, elements, ctx->stream);
    status = with_cuda_error(0);
    if (status != 0) {
        fprintf(stderr, "[engine] layer %d ffn residual failed (status %d)\n",
                ctx->layer_index, status);
        return status;
    }
    return 0;
}
