/**
 * rope.cu - Partial Rotary Position Embedding.
 *
 * Qwen3.8-27B uses partial rotary: only the first rotary_dim (64) of each
 * head_dim (256) are rotated. theta = 1e7.
 *
 * cos/sin tables are precomputed: [max_pos, rotary_dim/2].
 * Rotation: for each pair (x[2i], x[2i+1]):
 *   out[2i]   = x[2i] * cos - x[2i+1] * sin
 *   out[2i+1] = x[2i] * sin + x[2i+1] * cos
 */

#include "kernels.h"
#include <cuda_bf16.h>

/**
 * Apply RoPE in-place to q or k heads.
 * Grid: (tokens * num_heads), Block: (rotary_dim / 2)
 * Each thread handles one (cos, sin) pair.
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
    int pair = threadIdx.x;     // 0..31

    if (token >= tokens || pair >= half) return;

    int64_t pos = positions[token];
    // Base pointer for this head within the qkv buffer
    long long base = (long long)token * qkv_stride + head_offset + head * head_dim;

    float cos_val = __bfloat162float(cos_cache[(long long)pos * half + pair]);
    float sin_val = __bfloat162float(sin_cache[(long long)pos * half + pair]);

    float x0 = __bfloat162float(qkv[base + 2 * pair]);
    float x1 = __bfloat162float(qkv[base + 2 * pair + 1]);

    qkv[base + 2 * pair]     = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    qkv[base + 2 * pair + 1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
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
