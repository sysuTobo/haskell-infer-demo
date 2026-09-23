/**
 * kernels.h - Native CUDA kernel declarations still owned by this project.
 *
 * Convention: void kernel_<name>(<params>, cudaStream_t stream); BF16 unless
 * noted. Library-backed operators live in flashinfer_ops.h (norm/RoPE/
 * attention/SiLU) and fla_ops.h (GatedDeltaNet); cuBLAS GEMM in layers.h.
 */

#ifndef HASKELL_INFER_KERNELS_H
#define HASKELL_INFER_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Attention: KV cache + output gate                                 */
/* ------------------------------------------------------------------ */

/**
 * Write new K/V into the KV cache at [seq_start, seq_start+tokens).
 * kv_cache layout: [2, max_seq, num_kv_heads, head_dim] BF16.
 */
void kernel_kv_cache_write(__nv_bfloat16 *kv_cache,
                           const __nv_bfloat16 *k_new,
                           const __nv_bfloat16 *v_new,
                           int seq_start, int tokens,
                           int num_kv_heads, int head_dim,
                           int max_seq_len, cudaStream_t stream);

/**
 * Causal GQA attention over the KV cache (FlashInfer-backed).
 * Requires head_dim=256 and seq_len == seq_start + tokens.
 */
void kernel_attention(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                      const __nv_bfloat16 *kv_cache,
                      int seq_start, int tokens, int seq_len,
                      int num_heads, int num_kv_heads, int head_dim,
                      float scale, int max_seq_len, cudaStream_t stream);

/**
 * Gated attention output: out = attn * sigmoid(gate), BF16-rounded sigmoid.
 */
void kernel_sigmoid_mul(__nv_bfloat16 *out, const __nv_bfloat16 *attn,
                        const __nv_bfloat16 *gate, int dim, int tokens,
                        int gate_stride, int gate_offset, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: Causal Conv1d (causal-conv1d backed)                         */
/* ------------------------------------------------------------------ */

/**
 * Causal 1D convolution, width 4, conv_state updated in place.
 * Returns unfused BF16 output; caller applies SiLU separately.
 */
void kernel_causal_conv1d(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                          const __nv_bfloat16 *weight,
                          const __nv_bfloat16 *bias,
                          __nv_bfloat16 *conv_state,
                          int conv_dim, int tokens, int kernel_size,
                          cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: RMSNormGated                                                 */
/* ------------------------------------------------------------------ */

/**
 * Gated RMS norm matching Transformers dtype boundaries:
 * norm(FP32)->BF16, *weight(BF16)->BF16, *SiLU(z)(FP32)->BF16.
 * weight is the effective per-head FP32 value (no +1).
 */
void kernel_gdn_gated_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *z, const float *weight,
                           int dim, int tokens, float eps,
                           cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  MLP: SiLU gate (FlashInfer-backed)                                */
/* ------------------------------------------------------------------ */

/**
 * Fused SiLU(gate) * up. Requires up == gate + n (contiguous [gate, up]).
 */
void kernel_silu_mul(__nv_bfloat16 *out, const __nv_bfloat16 *gate,
                     const __nv_bfloat16 *up, int n, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Embedding                                                         */
/* ------------------------------------------------------------------ */

/** Token embedding lookup: out[i] = table[token_ids[i]]. */
void kernel_embedding(__nv_bfloat16 *out, const __nv_bfloat16 *table,
                      const int64_t *token_ids, int hidden_size, int tokens,
                      cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Small helpers                                                     */
/* ------------------------------------------------------------------ */

/** BF16 to F32 cast. */
void kernel_cast_bf16_f32(float *out, const __nv_bfloat16 *in, int n,
                          cudaStream_t stream);

/** F32 to BF16 cast: the single rounding that turns a merged FP32 partial into
 *  the activation it belongs to. */
void kernel_cast_f32_bf16(__nv_bfloat16 *out, const float *in, int n,
                          cudaStream_t stream);

/** Residual add: dst[i] += src[i] (BF16, element-wise). */
void kernel_residual_add(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                         int n, cudaStream_t stream);

#endif /* HASKELL_INFER_KERNELS_H */
