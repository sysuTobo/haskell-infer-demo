/** Batched attention, GDN and MLP forwards with caller-owned scratch. */
#include "kernels.h"
#include "layers.h"
#include "flashinfer_ops.h"
#include "fla_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

/* Input/post/q/k norms come in two flavours: Gemma-style (weight + 1) and the
 * plain RMSNorm that Qwen3-MoE/Mixtral use. The descriptor picks; a family that
 * has no such norm (e.g. no q/k norm) passes a null weight and skips it. */
static void layer_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                       const __nv_bfloat16 *weight, int cols, int rows,
                       const ModelDims *dims, cudaStream_t stream) {
    if (weight == nullptr) return;
    if (dims->norm_style == 1) {
        kernel_rms_norm_plain(out, x, weight, cols, rows, dims->rms_eps, stream);
    } else {
        kernel_gemma_rms_norm(out, x, weight, cols, rows, dims->rms_eps, stream);
    }
}

static void check_tokens(int tokens, const ModelDims *dims) {
    const int limit = dims != nullptr && dims->max_chunk > 0 ? dims->max_chunk : ENGINE_MAX_CHUNK;
    if (tokens < 1 || tokens > limit)
        throw std::invalid_argument("Layer token count exceeds the model's max_chunk");
}

static void check_launch() {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Layer CUDA: ") + cudaGetErrorString(status));
}

static void bind_stream(cublasHandle_t cublas, cudaStream_t stream) {
    cublasStatus_t status = cublasSetStream(cublas, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSetStream failed: " + std::to_string(status));
}

static void checked_gemm(cublasHandle_t cublas, __nv_bfloat16 *out,
                         const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                         int tokens, int cols, int inner) {
    // The legacy GEMM wrapper clears pending launch errors itself.
    check_launch();
    int status = gemm_bf16(cublas, out, x, weight, tokens, cols, inner);
    if (status != 0)
        throw std::runtime_error("BF16 GEMM failed: " + std::to_string(status));
    check_launch();
}

size_t layer_workspace_size(int tokens, const ModelDims *dims) {
    check_tokens(tokens, dims);
    const size_t H = dims->hidden_size;
    const size_t Q = (size_t)dims->num_heads * dims->head_dim;
    const size_t KV = (size_t)dims->num_kv_heads * dims->head_dim;
    const size_t V = (size_t)dims->gdn_num_v_heads * dims->gdn_head_dim;
    const size_t attention = H + 6 * Q + 2 * KV;
    const size_t gdn = H + 2 * (size_t)dims->gdn_conv_dim + 3 * V +
                       2 * (size_t)dims->gdn_num_v_heads;
    const size_t mlp = H + 3 * (size_t)dims->intermediate_size;
    return (size_t)tokens * std::max({attention, gdn, mlp}) * sizeof(__nv_bfloat16);
}

// Flattening (token, head) preserves the per-head Q/gate interleave.
__global__ void deinterleave_qg_kernel(__nv_bfloat16 *__restrict__ q,
                                      __nv_bfloat16 *__restrict__ gate,
                                      const __nv_bfloat16 *__restrict__ raw,
                                      int total, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int head = i / hd;
    int d = i % hd;
    q[i] = raw[head * 2 * hd + d];
    gate[i] = raw[head * 2 * hd + hd + d];
}

__global__ void silu_inplace_kernel(__nv_bfloat16 *__restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(v / (1.0f + expf(-v)));
}

int forward_mlp(cublasHandle_t cublas, cudaStream_t stream,
                const __nv_bfloat16 *residual, __nv_bfloat16 *ws,
                __nv_bfloat16 *layer_out, const MlpWeights *w,
                int tokens, const ModelDims *dims) {
    check_tokens(tokens, dims);
    bind_stream(cublas, stream);
    const int H = dims->hidden_size;
    const int I = dims->intermediate_size;
    const size_t TI = (size_t)tokens * I;
    __nv_bfloat16 *normed = ws;
    __nv_bfloat16 *gate = normed + (size_t)tokens * H;
    // Entire [T,I] gate matrix precedes the entire [T,I] up matrix.
    __nv_bfloat16 *up = gate + TI;
    __nv_bfloat16 *mlp_act = up + TI;

    layer_norm(normed, residual, w->post_norm_w, H, tokens, dims, stream);
    checked_gemm(cublas, gate, normed, w->gate_proj_w, tokens, I, H);
    checked_gemm(cublas, up, normed, w->up_proj_w, tokens, I, H);
    kernel_silu_mul(mlp_act, gate, up, tokens * I, stream);
    checked_gemm(cublas, layer_out, mlp_act, w->down_proj_w, tokens, H, I);
    return 0;
}

int forward_attention_layer(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const AttentionWeights *w, __nv_bfloat16 *kv_cache,
    const int64_t *positions, int tokens, int seq_len, const ModelDims *dims) {
    check_tokens(tokens, dims);
    if (!positions || seq_len < tokens || seq_len > dims->max_seq_len)
        throw std::invalid_argument("Invalid attention positions or sequence length");
    bind_stream(cublas, stream);
    const int H = dims->hidden_size;
    const int nH = dims->num_heads;
    const int nKV = dims->num_kv_heads;
    const int hd = dims->head_dim;
    const int Q = nH * hd;
    const int KV = nKV * hd;
    const size_t T = tokens;

    __nv_bfloat16 *normed = ws;
    __nv_bfloat16 *q_raw = normed + T * H;
    __nv_bfloat16 *k_out = q_raw + T * 2 * Q;
    __nv_bfloat16 *v_out = k_out + T * KV;
    __nv_bfloat16 *q = v_out + T * KV;
    __nv_bfloat16 *gate = q + T * Q;
    __nv_bfloat16 *attn_out = gate + T * Q;
    __nv_bfloat16 *gated = attn_out + T * Q;

    layer_norm(normed, residual, w->input_norm_w, H, tokens, dims, stream);
    /* With a fused gate the projection lands in q_raw and is split below;
     * without one it is written straight into the query buffer. */
    __nv_bfloat16 *q_dst = dims->attn_output_gate ? q_raw : q;
    checked_gemm(cublas, q_dst, normed, w->q_proj_w, tokens,
                 dims->attn_output_gate ? 2 * Q : Q, H);
    checked_gemm(cublas, k_out, normed, w->k_proj_w, tokens, KV, H);
    checked_gemm(cublas, v_out, normed, w->v_proj_w, tokens, KV, H);

    if (dims->attn_output_gate) {
        const int total = tokens * Q;
        deinterleave_qg_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            q, gate, q_raw, total, hd);
        check_launch();
    }
    layer_norm(q, q, w->q_norm_w, hd, tokens * nH, dims, stream);
    layer_norm(k_out, k_out, w->k_norm_w, hd, tokens * nKV, dims, stream);
    kernel_flashinfer_rope(q, k_out, positions, tokens, nH, nKV, hd,
                           dims->rotary_dim, dims->rope_theta, stream);
    check_launch();

    const int seq_start = seq_len - tokens;
    kernel_kv_cache_write(kv_cache, k_out, v_out, seq_start, tokens,
                          nKV, hd, dims->max_seq_len, stream);
    check_launch();
    kernel_attention(attn_out, q, kv_cache, seq_start, tokens, seq_len,
                      nH, nKV, hd, 1.0f / sqrtf((float)hd), dims->max_seq_len, stream);
    check_launch();
    if (dims->attn_output_gate) {
        kernel_sigmoid_mul(gated, attn_out, gate, Q, tokens, Q, 0, stream);
        checked_gemm(cublas, layer_out, gated, w->o_proj_w, tokens, H, Q);
    } else {
        checked_gemm(cublas, layer_out, attn_out, w->o_proj_w, tokens, H, Q);
    }
    return 0;
}

int forward_gdn_layer(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const GdnWeights *w, __nv_bfloat16 *conv_state,
    float *ssm_state, void *fla_scratch, int tokens, const ModelDims *dims) {
    check_tokens(tokens, dims);
    bind_stream(cublas, stream);
    const int H = dims->hidden_size;
    const int C = dims->gdn_conv_dim;
    const int nVH = dims->gdn_num_v_heads;
    const int nKH = dims->gdn_num_k_heads;
    const int hd = dims->gdn_head_dim;
    const int V = nVH * hd;
    const size_t T = tokens;

    __nv_bfloat16 *normed = ws;
    __nv_bfloat16 *qkv = normed + T * H;
    __nv_bfloat16 *z = qkv + T * C;
    __nv_bfloat16 *a = z + T * V;
    __nv_bfloat16 *b = a + T * nVH;
    __nv_bfloat16 *conv_out = b + T * nVH;
    __nv_bfloat16 *delta_out = conv_out + T * C;
    __nv_bfloat16 *norm_delta = delta_out + T * V;

    kernel_gemma_rms_norm(normed, residual, w->input_norm_w,
                          H, tokens, dims->rms_eps, stream);
    checked_gemm(cublas, qkv, normed, w->in_proj_qkv_w, tokens, C, H);
    checked_gemm(cublas, z, normed, w->in_proj_z_w, tokens, V, H);
    checked_gemm(cublas, a, normed, w->in_proj_a_w, tokens, nVH, H);
    checked_gemm(cublas, b, normed, w->in_proj_b_w, tokens, nVH, H);

    kernel_causal_conv1d(conv_out, qkv, w->conv1d_w, w->conv1d_bias,
                         conv_state, C, tokens, dims->gdn_conv_kernel, stream);
    check_launch();
    // Convolution returns unfused BF16; preserve its rounding before SiLU.
    silu_inplace_kernel<<<(tokens * C + 255) / 256, 256, 0, stream>>>(conv_out, tokens * C);
    check_launch();
    kernel_fla_gdn(delta_out, conv_out, a, b, w->A_log, w->dt_bias,
                    ssm_state, fla_scratch, tokens, nKH, nVH, stream);
    check_launch();
    kernel_gdn_gated_norm(norm_delta, delta_out, z, w->gdn_norm_w,
                          hd, tokens * nVH, dims->rms_eps, stream);
    checked_gemm(cublas, layer_out, norm_delta, w->out_proj_w, tokens, H, V);
    return 0;
}
