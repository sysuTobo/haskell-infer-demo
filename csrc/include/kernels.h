/**
 * kernels.h - Internal declarations for all CUDA kernels.
 *
 * Each kernel function follows the convention:
 *   void kernel_<name>(<params>, cudaStream_t stream);
 *
 * All kernels operate on BF16 (__nv_bfloat16) unless noted.
 * GEMM operations use cuBLAS and are declared in engine.cu directly.
 */

#ifndef HASKELL_INFER_KERNELS_H
#define HASKELL_INFER_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  RMSNorm (Gemma variant: weight+1, f32 accumulation)               */
/* ------------------------------------------------------------------ */

/**
 * GemmaRMSNorm: out = (x * f32(weight + 1)) * rsqrt(mean(x^2) + eps)
 * Computed in f32, output cast to BF16.
 *
 * @param out       [rows, cols] BF16 output
 * @param x         [rows, cols] BF16 input
 * @param weight_p1 [cols] F32 weight (already +1)
 * @param cols      feature dimension (e.g. 5120)
 * @param rows      number of tokens
 * @param eps       epsilon (1e-6)
 * @param stream    CUDA stream
 */
void kernel_rms_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                     const float *weight_p1, int cols, int rows,
                     float eps, cudaStream_t stream);

/**
 * Fused add + RMSNorm: residual += x; out = rmsnorm(residual)
 * Used for post-attention and post-MLP normalization.
 *
 * @param out       [rows, cols] BF16 normalized output
 * @param residual  [rows, cols] BF16 residual (updated in-place)
 * @param x         [rows, cols] BF16 input to add
 * @param weight_p1 [cols] F32 weight+1
 * @param cols      feature dimension
 * @param rows      number of tokens
 * @param eps       epsilon
 * @param stream    CUDA stream
 */
void kernel_fused_add_rms_norm(__nv_bfloat16 *out, __nv_bfloat16 *residual,
                               const __nv_bfloat16 *x, const float *weight_p1,
                               int cols, int rows, float eps,
                               cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  RoPE (partial rotary: first rotary_dim of head_dim)                */
/* ------------------------------------------------------------------ */

/**
 * Apply rotary position embedding in-place.
 * Only the first rotary_dim elements of each head are rotated.
 *
 * @param qkv         [tokens, qkv_dim] BF16, q/k/v concatenated
 * @param cos_cache   [max_pos, rotary_dim/2] BF16
 * @param sin_cache   [max_pos, rotary_dim/2] BF16
 * @param positions   [tokens] int64 position indices
 * @param tokens      number of tokens
 * @param num_heads   number of q heads (or kv heads)
 * @param head_dim    full head dimension (256)
 * @param rotary_dim  dimensions to rotate (64)
 * @param qkv_stride  stride between tokens in the qkv buffer
 * @param head_offset byte offset of this head group within qkv_stride
 * @param stream      CUDA stream
 */
void kernel_rope(__nv_bfloat16 *qkv, const __nv_bfloat16 *cos_cache,
                 const __nv_bfloat16 *sin_cache, const int64_t *positions,
                 int tokens, int num_heads, int head_dim, int rotary_dim,
                 int qkv_stride, int head_offset, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Attention (naive, with KV cache and GQA)                          */
/* ------------------------------------------------------------------ */

/**
 * Write new K/V into the KV cache.
 *
 * @param kv_cache  [2, max_seq, num_kv_heads, head_dim] BF16
 * @param k_new     [tokens, num_kv_heads, head_dim] BF16
 * @param v_new     [tokens, num_kv_heads, head_dim] BF16
 * @param seq_start position offset in the cache
 * @param tokens    number of new tokens
 * @param num_kv_heads
 * @param head_dim
 * @param stream
 */
void kernel_kv_cache_write(__nv_bfloat16 *kv_cache,
                           const __nv_bfloat16 *k_new,
                           const __nv_bfloat16 *v_new,
                           int seq_start, int tokens,
                           int num_kv_heads, int head_dim,
                           int max_seq_len, cudaStream_t stream);

/**
 * Naive scaled-dot-product attention with causal mask and GQA.
 *
 * @param out        [tokens, num_heads * head_dim] BF16
 * @param q          [tokens, num_heads, head_dim] BF16
 * @param kv_cache   [2, max_seq, num_kv_heads, head_dim] BF16
 * @param seq_start  start position of current tokens in cache
 * @param tokens     number of query tokens
 * @param seq_len    total sequence length (seq_start + tokens)
 * @param num_heads  number of query heads (24)
 * @param num_kv_heads number of KV heads (4)
 * @param head_dim   head dimension (256)
 * @param scale      attention scale (1/sqrt(head_dim))
 * @param stream
 */
void kernel_attention(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                      const __nv_bfloat16 *kv_cache,
                      int seq_start, int tokens, int seq_len,
                      int num_heads, int num_kv_heads, int head_dim,
                      float scale, int max_seq_len, cudaStream_t stream);

/**
 * Gated attention output: out = attn * sigmoid(gate)
 *
 * @param out    [tokens, dim] BF16
 * @param attn   [tokens, dim] BF16
 * @param gate   [tokens, dim] BF16 (strided view from qkv projection)
 * @param dim    feature dimension (6144)
 * @param tokens number of tokens
 * @param gate_stride stride between tokens in gate buffer
 * @param gate_offset offset to gate data within the stride
 * @param stream
 */
void kernel_sigmoid_mul(__nv_bfloat16 *out, const __nv_bfloat16 *attn,
                        const __nv_bfloat16 *gate, int dim, int tokens,
                        int gate_stride, int gate_offset, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: Causal Conv1d                                                */
/* ------------------------------------------------------------------ */

/**
 * Causal 1D convolution with kernel_size=4, maintaining conv state.
 * conv_state is updated in-place (shift left, append new input).
 *
 * @param out        [tokens, conv_dim] BF16
 * @param x          [tokens, conv_dim] BF16 input
 * @param weight     [conv_dim, 1, kernel_size] BF16 conv weight
 * @param bias       [conv_dim] BF16
 * @param conv_state [conv_dim, kernel_size-1] BF16 (updated in-place)
 * @param conv_dim   channel dimension (10240)
 * @param tokens     number of tokens to process
 * @param kernel_size convolution kernel size (4)
 * @param stream
 */
void kernel_causal_conv1d(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                          const __nv_bfloat16 *weight,
                          const __nv_bfloat16 *bias,
                          __nv_bfloat16 *conv_state,
                          int conv_dim, int tokens, int kernel_size,
                          cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: Gated Delta Rule (recurrent, single token)                   */
/* ------------------------------------------------------------------ */

/**
 * Recurrent gated delta rule update for a single token (decode path).
 *
 * For each v-head h:
 *   S[h] = alpha[h] * S[h] + beta[h] * k[h]^T @ (v[h] - k[h] @ S[h])
 *   o[h] = q[h] @ S[h]
 *
 * @param out        [num_v_heads, v_head_dim] BF16 output
 * @param q          [num_k_heads, k_head_dim] BF16 query
 * @param k          [num_k_heads, k_head_dim] BF16 key
 * @param v          [num_v_heads, v_head_dim] BF16 value
 * @param alpha      [num_v_heads] F32 decay gate (from a_log)
 * @param beta       [num_v_heads] F32 update gate
 * @param ssm_state  [num_v_heads, k_head_dim, v_head_dim] F32 (updated in-place)
 * @param num_k_heads  number of key heads (16)
 * @param num_v_heads  number of value heads (48)
 * @param k_head_dim   key head dimension (128)
 * @param v_head_dim   value head dimension (128)
 * @param stream
 */
void kernel_gdn_delta_rule_decode(
    __nv_bfloat16 *out, const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const float *alpha, const float *beta,
    float *ssm_state, int num_k_heads, int num_v_heads,
    int k_head_dim, int v_head_dim, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: RMSNormGated (l2norm + z gate)                               */
/* ------------------------------------------------------------------ */

/**
 * Gated RMS normalization: out = rmsnorm(x) * sigmoid(z)
 * Used after the delta rule output, before out_proj.
 *
 * @param out    [tokens, dim] BF16
 * @param x      [tokens, dim] BF16
 * @param z      [tokens, dim] BF16 gate
 * @param weight [dim] F32 (weight+1)
 * @param dim    feature dimension (6144)
 * @param tokens number of tokens
 * @param eps    epsilon
 * @param stream
 */
void kernel_gdn_gated_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *z, const float *weight,
                           int dim, int tokens, float eps,
                           cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  MLP: SiLU gate                                                    */
/* ------------------------------------------------------------------ */

/**
 * Fused SiLU(gate) * up for the MLP.
 *
 * @param out  [tokens, intermediate_size] BF16
 * @param gate [tokens, intermediate_size] BF16
 * @param up   [tokens, intermediate_size] BF16
 * @param n    total elements (tokens * intermediate_size)
 * @param stream
 */
void kernel_silu_mul(__nv_bfloat16 *out, const __nv_bfloat16 *gate,
                     const __nv_bfloat16 *up, int n, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Embedding                                                         */
/* ------------------------------------------------------------------ */

/**
 * Token embedding lookup: out[i] = table[token_ids[i]]
 *
 * @param out       [tokens, hidden_size] BF16
 * @param table     [vocab_size, hidden_size] BF16
 * @param token_ids [tokens] int64
 * @param hidden_size
 * @param tokens
 * @param stream
 */
void kernel_embedding(__nv_bfloat16 *out, const __nv_bfloat16 *table,
                      const int64_t *token_ids, int hidden_size, int tokens,
                      cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  ArgMax                                                            */
/* ------------------------------------------------------------------ */

/**
 * Find the index of the maximum value in a float array.
 *
 * @param result  host pointer to store the argmax index
 * @param logits  [vocab_size] F32 on device
 * @param vocab_size
 * @param stream
 */
void kernel_argmax(int64_t *result, const float *logits, int vocab_size,
                   cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Weight preparation                                                */
/* ------------------------------------------------------------------ */

/**
 * Compute weight_p1 = float(weight) + 1.0 for GemmaRMSNorm.
 *
 * @param out    [n] F32
 * @param weight [n] BF16
 * @param n      number of elements
 * @param stream
 */
void kernel_weight_p1(float *out, const __nv_bfloat16 *weight, int n,
                      cudaStream_t stream);

/**
 * Generate RoPE cos/sin tables.
 * cos[pos, i] = cos(pos / theta^(2i/dim))
 * sin[pos, i] = sin(pos / theta^(2i/dim))
 *
 * @param cos_out   [max_pos, rotary_dim/2] BF16
 * @param sin_out   [max_pos, rotary_dim/2] BF16
 * @param max_pos   maximum position
 * @param rotary_dim number of rotary dimensions (64)
 * @param theta     base frequency (1e7)
 * @param stream
 */
void kernel_rope_table(__nv_bfloat16 *cos_out, __nv_bfloat16 *sin_out,
                       int max_pos, int rotary_dim, float theta,
                       cudaStream_t stream);

/**
 * BF16 to F32 cast.
 */
void kernel_cast_bf16_f32(float *out, const __nv_bfloat16 *in, int n,
                          cudaStream_t stream);

/**
 * Fill a float buffer with a constant.
 */
void kernel_fill_f32(float *out, int n, float value, cudaStream_t stream);

/**
 * Residual add: dst[i] += src[i] (BF16, element-wise).
 */
void kernel_residual_add(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                         int n, cudaStream_t stream);

#endif /* HASKELL_INFER_KERNELS_H */
