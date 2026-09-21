/**
 * attention.cu - Naive scaled-dot-product attention with KV cache and GQA.
 *
 * Not optimized (no flash attention). Correctness-first for the demo.
 * Supports Grouped Query Attention: 24 q-heads share 4 kv-heads (6:1 ratio).
 *
 * KV cache layout: [2, max_seq_len, num_kv_heads, head_dim] BF16
 *   - cache[0] = K, cache[1] = V
 */

#include "kernels.h"
#include <cuda_bf16.h>
#include <math.h>

/**
 * Write new K/V into the cache at positions [seq_start, seq_start+tokens).
 * Grid: (tokens), Block: (num_kv_heads * head_dim / 4)
 */
__global__ void kv_cache_write_kernel(__nv_bfloat16 *__restrict__ kv_cache,
                                      const __nv_bfloat16 *__restrict__ k_new,
                                      const __nv_bfloat16 *__restrict__ v_new,
                                      int seq_start, int tokens,
                                      int num_kv_heads, int head_dim,
                                      int max_seq_len) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    int kv_dim = num_kv_heads * head_dim;
    int pos = seq_start + t;

    // Each thread copies 4 elements (2 BF16 = 4 bytes)
    for (int i = threadIdx.x * 4; i < kv_dim; i += blockDim.x * 4) {
        // K
        kv_cache[(long long)0 * max_seq_len * kv_dim + pos * kv_dim + i]     = k_new[(long long)t * kv_dim + i];
        kv_cache[(long long)0 * max_seq_len * kv_dim + pos * kv_dim + i + 1] = k_new[(long long)t * kv_dim + i + 1];
        kv_cache[(long long)0 * max_seq_len * kv_dim + pos * kv_dim + i + 2] = k_new[(long long)t * kv_dim + i + 2];
        kv_cache[(long long)0 * max_seq_len * kv_dim + pos * kv_dim + i + 3] = k_new[(long long)t * kv_dim + i + 3];
        // V
        long long v_base = (long long)1 * max_seq_len * kv_dim;
        kv_cache[v_base + pos * kv_dim + i]     = v_new[(long long)t * kv_dim + i];
        kv_cache[v_base + pos * kv_dim + i + 1] = v_new[(long long)t * kv_dim + i + 1];
        kv_cache[v_base + pos * kv_dim + i + 2] = v_new[(long long)t * kv_dim + i + 2];
        kv_cache[v_base + pos * kv_dim + i + 3] = v_new[(long long)t * kv_dim + i + 3];
    }
}

void kernel_kv_cache_write(__nv_bfloat16 *kv_cache,
                           const __nv_bfloat16 *k_new,
                           const __nv_bfloat16 *v_new,
                           int seq_start, int tokens,
                           int num_kv_heads, int head_dim,
                           int max_seq_len, cudaStream_t stream) {
    int kv_dim = num_kv_heads * head_dim;
    int block = min(256, (kv_dim + 3) / 4);
    kv_cache_write_kernel<<<tokens, block, 0, stream>>>(
        kv_cache, k_new, v_new, seq_start, tokens,
        num_kv_heads, head_dim, max_seq_len);
}

/**
 * Naive attention: for each query token and head, compute softmax(Q*K^T/sqrt(d))*V.
 * Grid: (tokens, num_heads), Block: (head_dim)
 *
 * GQA: q_head h maps to kv_head (h / (num_heads / num_kv_heads)).
 * Causal mask: query at position p can only attend to positions <= p.
 */
__global__ void attention_kernel(__nv_bfloat16 *__restrict__ out,
                                 const __nv_bfloat16 *__restrict__ q,
                                 const __nv_bfloat16 *__restrict__ kv_cache,
                                 int seq_start, int tokens, int seq_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, int max_seq_len) {
    int t = blockIdx.x;    // query token index (0..tokens-1)
    int h = blockIdx.y;    // query head index (0..num_heads-1)
    if (t >= tokens) return;

    int kv_h = h / (num_heads / num_kv_heads);  // GQA mapping
    int query_pos = seq_start + t;

    // Load query vector for this head
    extern __shared__ float s_data[];
    float *q_vec = s_data;  // [head_dim]
    float *scores = s_data + head_dim;  // [max_seq_len] - but we limit to seq_len

    const __nv_bfloat16 *q_ptr = q + ((long long)t * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_vec[d] = __bfloat162float(q_ptr[d]);
    }
    __syncthreads();

    // Compute attention scores against all cached K
    // K cache: kv_cache[0, pos, kv_h, d]
    long long k_base = (long long)kv_h * head_dim;
    long long kv_stride = (long long)num_kv_heads * head_dim;

    // Each thread computes scores for a subset of positions
    for (int p = threadIdx.x; p <= query_pos; p += blockDim.x) {
        const __nv_bfloat16 *k_ptr = kv_cache + (long long)p * kv_stride + k_base;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_vec[d] * __bfloat162float(k_ptr[d]);
        }
        scores[p] = dot * scale;
    }
    __syncthreads();

    // Softmax (thread 0 does it serially for simplicity in this demo)
    if (threadIdx.x == 0) {
        float max_score = -1e30f;
        for (int p = 0; p <= query_pos; p++) {
            if (scores[p] > max_score) max_score = scores[p];
        }
        float sum = 0.0f;
        for (int p = 0; p <= query_pos; p++) {
            scores[p] = expf(scores[p] - max_score);
            sum += scores[p];
        }
        float inv_sum = 1.0f / (sum + 1e-9f);
        for (int p = 0; p <= query_pos; p++) {
            scores[p] *= inv_sum;
        }
    }
    __syncthreads();

    // Weighted sum of V
    long long v_base = (long long)max_seq_len * kv_stride + (long long)kv_h * head_dim;
    __nv_bfloat16 *out_ptr = out + ((long long)t * num_heads + h) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p <= query_pos; p++) {
            const __nv_bfloat16 *v_ptr = kv_cache + v_base + (long long)p * kv_stride + d;
            acc += scores[p] * __bfloat162float(*v_ptr);
        }
        out_ptr[d] = __float2bfloat16(acc);
    }
}

void kernel_attention(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                      const __nv_bfloat16 *kv_cache,
                      int seq_start, int tokens, int seq_len,
                      int num_heads, int num_kv_heads, int head_dim,
                      float scale, int max_seq_len, cudaStream_t stream) {
    dim3 grid(tokens, num_heads);
    dim3 block(min(head_dim, 256));
    size_t smem = (head_dim + seq_len) * sizeof(float);
    attention_kernel<<<grid, block, smem, stream>>>(
        out, q, kv_cache, seq_start, tokens, seq_len,
        num_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

/**
 * Sigmoid gate multiplication: out = attn * sigmoid(gate)
 * Grid: (tokens), Block: (256)
 */
__global__ void sigmoid_mul_kernel(__nv_bfloat16 *__restrict__ out,
                                   const __nv_bfloat16 *__restrict__ attn,
                                   const __nv_bfloat16 *__restrict__ gate,
                                   int dim, int tokens,
                                   int gate_stride, int gate_offset) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    const __nv_bfloat16 *attn_row = attn + (long long)t * dim;
    const __nv_bfloat16 *gate_row = gate + (long long)t * gate_stride + gate_offset;
    __nv_bfloat16 *out_row = out + (long long)t * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = __bfloat162float(attn_row[i]);
        float g = __bfloat162float(gate_row[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        out_row[i] = __float2bfloat16(a * sig);
    }
}

void kernel_sigmoid_mul(__nv_bfloat16 *out, const __nv_bfloat16 *attn,
                        const __nv_bfloat16 *gate, int dim, int tokens,
                        int gate_stride, int gate_offset, cudaStream_t stream) {
    sigmoid_mul_kernel<<<tokens, 256, 0, stream>>>(
        out, attn, gate, dim, tokens, gate_stride, gate_offset);
}
