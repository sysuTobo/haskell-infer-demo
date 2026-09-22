#ifndef HASKELL_INFER_FLASHINFER_OPS_H
#define HASKELL_INFER_FLASHINFER_OPS_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Contiguous BF16 [rows, cols]; raw_weight is BF16, without +1. out == x is supported.
void kernel_gemma_rms_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *raw_weight, int cols, int rows,
                           float eps, cudaStream_t stream);

// In-place contiguous Q/K [tokens, heads, head_dim], split-half RoPE; device-local positions.
void kernel_flashinfer_rope(__nv_bfloat16 *q, __nv_bfloat16 *k,
                            const int64_t *positions, int tokens,
                            int q_heads, int kv_heads, int head_dim,
                            int rotary_dim, float theta, cudaStream_t stream);

#endif  // HASKELL_INFER_FLASHINFER_OPS_H
