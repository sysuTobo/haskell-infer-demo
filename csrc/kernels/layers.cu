/**
 * layers.cu - Forward pass for attention and GDN layers.
 *
 * Each function processes a single token (M=1) through one layer.
 * Buffers are pre-allocated in the workspace.
 */

#include "kernels.h"
#include "layers.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstdio>

/* Declared in gemm.cu */
extern int gemm_bf16(cublasHandle_t, __nv_bfloat16*, const __nv_bfloat16*,
                     const __nv_bfloat16*, int M, int N, int K);

/* Helper: set cuBLAS stream before GEMM calls */
static inline void sync_cublas_stream(cublasHandle_t cublas, cudaStream_t stream) {
    cublasSetStream(cublas, stream);
}

/* ------------------------------------------------------------------ */
/*  Inline kernels for attention layer                                */
/* ------------------------------------------------------------------ */

/**
 * Deinterleave Q and gate from q_proj output.
 * Input layout:  [head0_Q(hd), head0_gate(hd), head1_Q(hd), head1_gate(hd), ...]
 * Output: q_ext = [head0_Q, head1_Q, ...] = [nH, hd]
 *         gate  = [head0_gate, head1_gate, ...] = [nH, hd]
 */
__global__ void deinterleave_qg_kernel(__nv_bfloat16 *__restrict__ q_ext,
                                       __nv_bfloat16 *__restrict__ gate_ext,
                                       const __nv_bfloat16 *__restrict__ q_raw,
                                       int nH, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nH * hd;
    if (idx >= total) return;
    int h = idx / hd;   // head index
    int d = idx % hd;   // dim within head
    // Source: q_raw[h * 2*hd + d] for Q, q_raw[h * 2*hd + hd + d] for gate
    q_ext[idx] = q_raw[h * 2 * hd + d];
    gate_ext[idx] = q_raw[h * 2 * hd + hd + d];
}

/**
 * Contiguous sigmoid gate: out[i] = attn[i] * sigmoid(gate[i])
 */
__global__ void sigmoid_mul_contiguous_kernel(__nv_bfloat16 *__restrict__ out,
                                              const __nv_bfloat16 *__restrict__ attn,
                                              const __nv_bfloat16 *__restrict__ gate,
                                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = __bfloat162float(attn[i]);
    float g = __bfloat162float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = __float2bfloat16(a * sig);
}

/**
 * Compute GDN alpha (decay) and beta (update gate) from projections.
 * Formula (Mamba-style):
 *   dt = softplus(a_out + dt_bias)
 *   alpha = exp(-dt * exp(A_log))
 *   beta = sigmoid(b_out)
 * Grid: (1), Block: (num_v_heads=48)
 */
__global__ void gdn_compute_alpha_beta_kernel(
    float *__restrict__ alpha,          // [nVH] output
    float *__restrict__ beta,           // [nVH] output
    const __nv_bfloat16 *__restrict__ a_out,  // [nVH] BF16
    const __nv_bfloat16 *__restrict__ b_out,  // [nVH] BF16
    const __nv_bfloat16 *__restrict__ A_log,  // [nVH] BF16
    const __nv_bfloat16 *__restrict__ dt_bias,// [nVH] BF16
    int nVH) {
    int h = threadIdx.x;
    if (h >= nVH) return;

    float a = __bfloat162float(a_out[h]);
    float b = __bfloat162float(b_out[h]);
    float a_log = __bfloat162float(A_log[h]);
    float bias = __bfloat162float(dt_bias[h]);

    // dt = softplus(a + bias) = log(1 + exp(a + bias))
    float x = a + bias;
    float dt = (x > 20.0f) ? x : log1pf(expf(x));  // numerically stable softplus

    // alpha = exp(-dt * exp(A_log))
    alpha[h] = expf(-dt * expf(a_log));

    // beta = sigmoid(b)
    beta[h] = 1.0f / (1.0f + expf(-b));
}

/* ------------------------------------------------------------------ */
/*  MLP: gate_up → SiLU×mul → down                                    */
/* ------------------------------------------------------------------ */

int forward_mlp(cublasHandle_t cublas, cudaStream_t stream,
                __nv_bfloat16 *residual,       // [1, hidden] in/out
                __nv_bfloat16 *normed,         // [1, hidden] scratch (post-norm input)
                __nv_bfloat16 *gate_up_out,    // [1, 2*intermediate] scratch
                __nv_bfloat16 *mlp_act,        // [1, intermediate] scratch
                __nv_bfloat16 *mlp_down_out,   // [1, hidden] scratch
                const __nv_bfloat16 *post_norm_w_p1, // [hidden] f32 (weight+1)
                const __nv_bfloat16 *gate_proj_w,    // [intermediate, hidden]
                const __nv_bfloat16 *up_proj_w,      // [intermediate, hidden]
                const __nv_bfloat16 *down_proj_w,    // [hidden, intermediate]
                const float *post_norm_w_p1_f32,     // [hidden] f32
                int hidden, int intermediate, float eps) {
    // 1. Post-attention RMSNorm (fused add: residual += normed_input already done)
    // normed is already computed by caller

    // 2. gate_proj: [1, hidden] @ [intermediate, hidden]^T → [1, intermediate]
    if (gemm_bf16(cublas, gate_up_out, normed, gate_proj_w, 1, intermediate, hidden) != 0)
        return -1;

    // 3. up_proj: [1, hidden] @ [intermediate, hidden]^T → [1, intermediate]
    if (gemm_bf16(cublas, gate_up_out + intermediate, normed, up_proj_w, 1, intermediate, hidden) != 0)
        return -1;

    // 4. SiLU(gate) * up → mlp_act [1, intermediate]
    kernel_silu_mul(mlp_act, gate_up_out, gate_up_out + intermediate,
                    intermediate, stream);

    // 5. down_proj: [1, intermediate] @ [hidden, intermediate]^T → [1, hidden]
    if (gemm_bf16(cublas, mlp_down_out, mlp_act, down_proj_w, 1, hidden, intermediate) != 0)
        return -1;

    // 6. Residual add: residual += mlp_down_out (done via fused_add_rms_norm in next layer)
    // For now, just add in-place using a simple kernel
    // We'll use the fused_add_rms_norm at the start of the next layer
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Full Attention Layer                                              */
/* ------------------------------------------------------------------ */

int forward_attention_layer(
    cublasHandle_t cublas, cudaStream_t stream,
    __nv_bfloat16 *residual,        // [1, hidden] in/out (residual stream)
    __nv_bfloat16 *ws,              // workspace pointer
    __nv_bfloat16 *layer_out,       // [1, hidden] caller-provided output buffer
    const AttentionWeights *w,
    __nv_bfloat16 *kv_cache,        // per-layer KV cache
    __nv_bfloat16 *cos_cache,       // RoPE cos table
    __nv_bfloat16 *sin_cache,       // RoPE sin table
    int64_t *d_position,            // device position scalar
    int seq_len,                    // current sequence length (after this token)
    const ModelDims *dims) {

    int H = dims->hidden_size;
    int nH = dims->num_heads;       // 24
    int nKV = dims->num_kv_heads;   // 4
    int hd = dims->head_dim;        // 256
    int rot = dims->rotary_dim;     // 64

    // Bind cuBLAS to the same stream as our kernels
    sync_cublas_stream(cublas, stream);

    // Workspace layout:
    // q_proj output is [1, nH*2*hd] = [1, 12288] (interleaved Q+gate per head)
    // We deinterleave into Q[24,256] and gate[24,256]
    __nv_bfloat16 *normed = ws;                       // [1, H=5120]
    __nv_bfloat16 *q_raw = normed + H;                // [1, 12288] raw q_proj output
    __nv_bfloat16 *k_out = q_raw + nH * 2 * hd;      // [1, 1024]
    __nv_bfloat16 *v_out = k_out + nKV * hd;          // [1, 1024]
    __nv_bfloat16 *q_ext = v_out + nKV * hd;          // [24, 256] extracted Q
    __nv_bfloat16 *gate_ext = q_ext + nH * hd;        // [24, 256] extracted gate
    __nv_bfloat16 *attn_out = gate_ext + nH * hd;     // [24, 256] = [1, 6144]
    __nv_bfloat16 *gated = attn_out + nH * hd;        // [1, 6144]

    // 1. Input RMSNorm
    kernel_rms_norm(normed, residual, w->input_norm_w_p1, H, 1, dims->rms_eps, stream);

    // 2. Q projection: [1, H] @ [nH*2*hd, H]^T -> [1, 12288]
    gemm_bf16(cublas, q_raw, normed, w->q_proj_w, 1, nH * 2 * hd, H);

    // 3. K projection: [1, H] @ [nKV*hd, H]^T -> [1, 1024]
    gemm_bf16(cublas, k_out, normed, w->k_proj_w, 1, nKV * hd, H);

    // 4. V projection: [1, H] @ [nKV*hd, H]^T -> [1, 1024]
    gemm_bf16(cublas, v_out, normed, w->v_proj_w, 1, nKV * hd, H);

    // 5. Deinterleave Q and gate from q_raw
    // q_raw layout: [head0_Q(256), head0_gate(256), head1_Q(256), head1_gate(256), ...]
    // q_ext layout: [head0_Q(256), head1_Q(256), ...] = [24, 256]
    // gate_ext layout: [head0_gate(256), head1_gate(256), ...] = [24, 256]
    // Simple copy kernel: each thread copies one element
    {
        int total = nH * hd;  // 6144
        // Launch a simple deinterleave kernel
        // For demo: use cudaMemcpy2D or a custom kernel
        // Quick approach: one kernel that does both copies
        deinterleave_qg_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            q_ext, gate_ext, q_raw, nH, hd);
    }

    // 6. Per-head QK norm (RMSNorm with head_dim=256, Gemma variant weight+1)
    // q_ext is [24, 256] contiguous -> rms_norm with cols=256, rows=24
    if (w->q_norm_w_p1) {
        kernel_rms_norm(q_ext, q_ext, w->q_norm_w_p1, hd, nH, dims->rms_eps, stream);
    }
    // k_out is [4, 256] contiguous -> rms_norm with cols=256, rows=4
    if (w->k_norm_w_p1) {
        kernel_rms_norm(k_out, k_out, w->k_norm_w_p1, hd, nKV, dims->rms_eps, stream);
    }

    // 7. RoPE on Q and K (partial: first 64 dims of each 256-dim head)
    if (d_position != nullptr && cos_cache != nullptr) {
        // Q: 24 heads, head_dim=256, rotary_dim=64, stride=256 (contiguous)
        kernel_rope(q_ext, cos_cache, sin_cache, d_position,
                    1, nH, hd, rot, nH * hd, 0, stream);
        // K: 4 heads, head_dim=256, rotary_dim=64, stride=256
        kernel_rope(k_out, cos_cache, sin_cache, d_position,
                    1, nKV, hd, rot, nKV * hd, 0, stream);
    }

    // 8. Write K, V to cache
    kernel_kv_cache_write(kv_cache, k_out, v_out, seq_len - 1, 1,
                          nKV, hd, dims->max_seq_len, stream);

    // 9. Attention: Q[24, 256] x K_cache -> attn_out[24, 256]
    kernel_attention(attn_out, q_ext, kv_cache, seq_len - 1, 1, seq_len,
                     nH, nKV, hd, 1.0f / sqrtf((float)hd), dims->max_seq_len, stream);

    // 10. Output gate: gated = attn_out * sigmoid(gate_ext)
    // gate_ext is now contiguous [24, 256] = [1, 6144]
    {
        int dim = nH * hd;  // 6144
        sigmoid_mul_contiguous_kernel<<<1, 256, 0, stream>>>(
            gated, attn_out, gate_ext, dim);
    }

    // 11. O projection: [1, nH*hd] @ [H, nH*hd]^T -> [1, H]
    gemm_bf16(cublas, layer_out, gated, w->o_proj_w, 1, H, nH * hd);

    return 0;
}

/* ------------------------------------------------------------------ */
/*  GDN Layer                                                         */
/* ------------------------------------------------------------------ */

int forward_gdn_layer(
    cublasHandle_t cublas, cudaStream_t stream,
    __nv_bfloat16 *residual,        // [1, hidden] in/out
    __nv_bfloat16 *ws,              // workspace
    __nv_bfloat16 *layer_out,       // [1, hidden] caller-provided output buffer
    const GdnWeights *w,
    __nv_bfloat16 *conv_state,      // [conv_dim, kernel_size-1] BF16
    float *ssm_state,               // [nVH, k_hd, v_hd] F32
    const ModelDims *dims) {

    int H = dims->hidden_size;
    int conv_dim = dims->gdn_conv_dim;      // 10240
    int nVH = dims->gdn_num_v_heads;        // 48
    int nKH = dims->gdn_num_k_heads;        // 16
    int k_hd = dims->gdn_head_dim;          // 128
    int v_hd = dims->gdn_head_dim;          // 128
    int v_dim = nVH * v_hd;                 // 6144
    int qk_dim = nKH * k_hd;               // 2048

    // Bind cuBLAS to the same stream as our kernels
    sync_cublas_stream(cublas, stream);

    // Workspace layout:
    __nv_bfloat16 *normed = ws;                  // [1, H]
    __nv_bfloat16 *qkv_out = normed + H;         // [1, conv_dim=10240]
    __nv_bfloat16 *z_out = qkv_out + conv_dim;   // [1, v_dim=6144]
    __nv_bfloat16 *a_out = z_out + v_dim;        // [1, 48]
    __nv_bfloat16 *b_out = a_out + 48;           // [1, 48]
    __nv_bfloat16 *conv_out = b_out + 48;        // [1, conv_dim]
    __nv_bfloat16 *delta_out = conv_out + conv_dim; // [1, v_dim]
    __nv_bfloat16 *norm_delta = delta_out + v_dim;  // [1, v_dim]
    __nv_bfloat16 *o_out = norm_delta + v_dim;      // [1, H]
    float *alpha_f32 = (float *)(o_out + H);        // [48] f32
    float *beta_f32 = alpha_f32 + 48;               // [48] f32

    // 1. Input RMSNorm
    fprintf(stderr, "[gdn_layer] rms_norm: normed=%p residual=%p w_p1=%p H=%d\n",
            (void*)normed, (void*)residual, (void*)w->input_norm_w_p1, H);
    kernel_rms_norm(normed, residual, w->input_norm_w_p1, H, 1, dims->rms_eps, stream);
    cudaError_t rn_err = cudaDeviceSynchronize();
    if (rn_err != cudaSuccess) {
        fprintf(stderr, "[gdn_layer] rms_norm FAILED: %s\n", cudaGetErrorString(rn_err));
        return -1;
    }

    // 2. in_proj_qkv: [1, H] @ [conv_dim, H]^T → [1, conv_dim]
    gemm_bf16(cublas, qkv_out, normed, w->in_proj_qkv_w, 1, conv_dim, H);

    // 3. in_proj_z: [1, H] @ [v_dim, H]^T → [1, v_dim]
    gemm_bf16(cublas, z_out, normed, w->in_proj_z_w, 1, v_dim, H);

    // 4. in_proj_a: [1, H] @ [48, H]^T → [1, 48]
    gemm_bf16(cublas, a_out, normed, w->in_proj_a_w, 1, nVH, H);

    // 5. in_proj_b: [1, H] @ [48, H]^T → [1, 48]
    gemm_bf16(cublas, b_out, normed, w->in_proj_b_w, 1, nVH, H);

    // 6. Causal conv1d on qkv_out (updates conv_state in-place)
    kernel_causal_conv1d(conv_out, qkv_out, w->conv1d_w, w->conv1d_bias,
                         conv_state, conv_dim, 1, dims->gdn_conv_kernel, stream);

    // 7. Compute alpha and beta from a_out, b_out, A_log, dt_bias
    // Formula (Mamba-style gated delta rule):
    //   dt = softplus(a_out + dt_bias)
    //   alpha = exp(-dt * exp(A_log))   -- decay in (0, 1)
    //   beta = sigmoid(b_out)           -- update gate in (0, 1)
    gdn_compute_alpha_beta_kernel<<<1, nVH, 0, stream>>>(
        alpha_f32, beta_f32, a_out, b_out,
        w->A_log, w->dt_bias, nVH);

    // 8. Split conv_out into q[0:qk_dim], k[qk_dim:2*qk_dim], v[2*qk_dim:conv_dim]
    const __nv_bfloat16 *q_ptr = conv_out;
    const __nv_bfloat16 *k_ptr = conv_out + qk_dim;
    const __nv_bfloat16 *v_ptr = conv_out + 2 * qk_dim;

    // 9. Gated delta rule (recurrent, single token)
    kernel_gdn_delta_rule_decode(delta_out, q_ptr, k_ptr, v_ptr,
                                 alpha_f32, beta_f32, ssm_state,
                                 nKH, nVH, k_hd, v_hd, stream);

    // 10. Gated RMSNorm: norm_delta = rmsnorm(delta_out) * sigmoid(z_out)
    kernel_gdn_gated_norm(norm_delta, delta_out, z_out, w->gdn_norm_w_p1,
                          v_dim, 1, dims->rms_eps, stream);

    // 11. out_proj: [1, v_dim] @ [H, v_dim]^T → [1, H]
    gemm_bf16(cublas, layer_out, norm_delta, w->out_proj_w, 1, H, v_dim);

    return 0;
}
