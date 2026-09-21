/**
 * gdn_delta_rule.cu - Gated Delta Rule recurrent update for GDN layers.
 *
 * Decode path (single token):
 *   For each v-head h (48 total):
 *     k_head = h / 3  (16 k-heads, 48 v-heads, ratio 3:1)
 *     delta = v[h] - k[k_head] @ S[h]
 *     S[h] = alpha[h] * S[h] + beta[h] * outer(k[k_head], delta)
 *     o[h] = q[k_head] @ S[h]
 *
 * SSM state layout: [num_v_heads, k_head_dim, v_head_dim] F32
 *   S[h][j][i] where j = k-dim index, i = v-dim index
 *
 * Prefill path: calls decode repeatedly for each token (first version).
 *
 * Kernel design: one block per v-head, 128 threads.
 * Thread i handles column i of S[h] (128 elements: S[h][0..127][i]).
 * No inter-thread communication needed.
 */

#include "kernels.h"
#include <cuda_bf16.h>
#include <math.h>

#define K_HEAD_DIM 128
#define V_HEAD_DIM 128

__global__ void gdn_delta_rule_decode_kernel(
    __nv_bfloat16 *__restrict__ out,        // [num_v_heads, v_head_dim] BF16
    const __nv_bfloat16 *__restrict__ q,    // [num_k_heads, k_head_dim] BF16
    const __nv_bfloat16 *__restrict__ k,    // [num_k_heads, k_head_dim] BF16
    const __nv_bfloat16 *__restrict__ v,    // [num_v_heads, v_head_dim] BF16
    const float *__restrict__ alpha,        // [num_v_heads] F32
    const float *__restrict__ beta,         // [num_v_heads] F32
    float *__restrict__ ssm_state,          // [num_v_heads, k_head_dim, v_head_dim] F32
    int num_k_heads, int num_v_heads,
    int k_head_dim, int v_head_dim) {

    int h = blockIdx.x;  // v-head index (0..47)
    int i = threadIdx.x; // v-dim index (0..127)
    if (h >= num_v_heads || i >= v_head_dim) return;

    int kh = h / (num_v_heads / num_k_heads);  // k-head index (0..15)
    float a = alpha[h];
    float b = beta[h];

    // Pointers
    float *S = ssm_state + (long long)h * k_head_dim * v_head_dim;
    const __nv_bfloat16 *q_ptr = q + (long long)kh * k_head_dim;
    const __nv_bfloat16 *k_ptr = k + (long long)kh * k_head_dim;
    const __nv_bfloat16 *v_ptr = v + (long long)h * v_head_dim;

    // Step 1: o[i] = (q @ S_old)[i] -- OUTPUT BEFORE STATE UPDATE
    // The GatedDeltaNet convention: o_t = q_t @ S_t (state BEFORE update)
    float o = 0.0f;
    for (int j = 0; j < k_head_dim; j++) {
        o += __bfloat162float(q_ptr[j]) * S[j * v_head_dim + i];
    }
    out[(long long)h * v_head_dim + i] = __float2bfloat16(o);

    // Step 2: compute (k @ S_old)[i]
    float kS = 0.0f;
    for (int j = 0; j < k_head_dim; j++) {
        kS += __bfloat162float(k_ptr[j]) * S[j * v_head_dim + i];
    }

    // Step 3: delta[i] = v[i] - kS
    float delta = __bfloat162float(v_ptr[i]) - kS;

    // Step 4: S[j][i] = alpha * S[j][i] + beta * k[j] * delta (STATE UPDATE)
    for (int j = 0; j < k_head_dim; j++) {
        float kj = __bfloat162float(k_ptr[j]);
        S[j * v_head_dim + i] = a * S[j * v_head_dim + i] + b * kj * delta;
    }
}

void kernel_gdn_delta_rule_decode(
    __nv_bfloat16 *out, const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const float *alpha, const float *beta,
    float *ssm_state, int num_k_heads, int num_v_heads,
    int k_head_dim, int v_head_dim, cudaStream_t stream) {

    // One block per v-head, v_head_dim threads per block
    gdn_delta_rule_decode_kernel<<<num_v_heads, v_head_dim, 0, stream>>>(
        out, q, k, v, alpha, beta, ssm_state,
        num_k_heads, num_v_heads, k_head_dim, v_head_dim);
}
