/**
 * rope.cu - Partial Rotary Position Embedding (SPLIT-HALF variant).
 *
 * Qwen3.8-27B uses partial rotary: only the first rotary_dim (64) of each
 * head_dim (256) are rotated. theta = 1e7.
 *
 * Rotation formula (split-half, as in Llama/Qwen):
 *   For i in 0..half-1 (half = rotary_dim/2 = 32):
 *     out[i]      = x[i] * cos[i] - x[i + half] * sin[i]
 *     out[i+half] = x[i + half] * cos[i] + x[i] * sin[i]
 *   Dims [rotary_dim..head_dim-1] pass through unchanged.
 *
 * cos/sin tables: [max_pos, rotary_dim/2] BF16.
 */

#include "kernels.h"
#include <cuda_bf16.h>

/**
 * Apply RoPE in-place to q or k heads (split-half variant).
 * Grid: (tokens * num_heads), Block: (rotary_dim / 2)
 * Each thread handles one pair (x[i], x[i+half]).
 */
__global__ void rope_kernel(__nv_bfloat16 *__restrict__ qkv,
                            const __nv_bfloat16 *__restrict__ cos_cache,
                            const __nv_bfloat16 *__restrict__ sin_cache,
                            const int64_t *__restrict__ positions,
                            int tokens, int num_heads, int head_dim,
                            int rotary_dim, int qkv_stride, int head_offset) {
    int idx = blockIdx.x;
    int token = idx / num_heads;
    int head = idx % num_heads;
    int half = rotary_dim / 2;  // 32
    int i = threadIdx.x;        // 0..31

    if (token >= tokens || i >= half) return;

    int64_t pos = positions[token];
    long long base = (long long)token * qkv_stride + head_offset + head * head_dim;

    float cos_val = __bfloat162float(cos_cache[(long long)pos * half + i]);
    float sin_val = __bfloat162float(sin_cache[(long long)pos * half + i]);

    // Split-half rotation: x0 = x[i], x1 = x[i + half]
    float x0 = __bfloat162float(qkv[base + i]);
    float x1 = __bfloat162float(qkv[base + half + i]);

    qkv[base + i]        = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    qkv[base + half + i] = __float2bfloat16(x1 * cos_val + x0 * sin_val);
}

void kernel_rope(__nv_bfloat16 *qkv, const __nv_bfloat16 *cos_cache,
                 const __nv_bfloat16 *sin_cache, const int64_t *positions,
                 int tokens, int num_heads, int head_dim, int rotary_dim,
                 int qkv_stride, int head_offset, cudaStream_t stream) {
    int half = rotary_dim / 2;
    dim3 grid(tokens * num_heads);
    dim3 block(half);
    rope_kernel<<<grid, block, 0, stream>>>(
        qkv, cos_cache, sin_cache, positions,
        tokens, num_heads, head_dim, rotary_dim, qkv_stride, head_offset);
}

/**
 * Generate RoPE cos/sin tables.
 * Grid: (max_pos), Block: (rotary_dim/2)
 */
__global__ void rope_table_kernel(__nv_bfloat16 *__restrict__ cos_out,
                                  __nv_bfloat16 *__restrict__ sin_out,
                                  int max_pos, int half_dim, float theta) {
    int pos = blockIdx.x;
    int i = threadIdx.x;
    if (pos >= max_pos || i >= half_dim) return;

    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)(2 * half_dim));
    float angle = (float)pos * freq;

    cos_out[(long long)pos * half_dim + i] = __float2bfloat16(cosf(angle));
    sin_out[(long long)pos * half_dim + i] = __float2bfloat16(sinf(angle));
}

void kernel_rope_table(__nv_bfloat16 *cos_out, __nv_bfloat16 *sin_out,
                       int max_pos, int rotary_dim, float theta,
                       cudaStream_t stream) {
    int half = rotary_dim / 2;
    rope_table_kernel<<<max_pos, half, 0, stream>>>(cos_out, sin_out, max_pos, half, theta);
}
