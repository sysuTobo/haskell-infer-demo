/**
 * gdn_conv.cu - Causal 1D convolution for GatedDeltaNet layers.
 *
 * Kernel size = 4, causal (only looks at current and past tokens).
 * Maintains a conv_state buffer of shape [conv_dim, kernel_size-1] that
 * holds the last (kernel_size-1) inputs for the next call.
 *
 * For prefill (multiple tokens), processes sequentially to maintain state.
 * For decode (single token), one state update.
 */

#include "kernels.h"
#include <cuda_bf16.h>

/**
 * Causal conv1d for a single token (decode path).
 * Grid: (conv_dim / 256), Block: (256)
 *
 * conv_state layout: [conv_dim, kernel_size-1] BF16
 * On entry: conv_state[d, 0..2] = x[t-3], x[t-2], x[t-1] for channel d
 * On exit:  conv_state[d, 0..2] = x[t-2], x[t-1], x[t]
 *
 * out[d] = bias[d] + sum_{k=0}^{3} weight[d, k] * input_history[d, k]
 * where input_history = [conv_state[d,0], conv_state[d,1], conv_state[d,2], x[t,d]]
 */
__global__ void causal_conv1d_decode_kernel(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,  // [conv_dim, kernel_size]
    const __nv_bfloat16 *__restrict__ bias,    // [conv_dim]
    __nv_bfloat16 *__restrict__ conv_state,    // [conv_dim, kernel_size-1]
    int conv_dim, int kernel_size) {

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim) return;

    int ks = kernel_size;       // 4
    int state_len = ks - 1;     // 3

    // Load history from conv_state
    float history[4];  // kernel_size = 4
    for (int k = 0; k < state_len; k++) {
        history[k] = __bfloat162float(conv_state[d * state_len + k]);
    }
    // Current input
    history[state_len] = __bfloat162float(x[d]);

    // Compute convolution
    float acc = __bfloat162float(bias[d]);
    for (int k = 0; k < ks; k++) {
        acc += __bfloat162float(weight[d * ks + k]) * history[k];
    }
    out[d] = __float2bfloat16(acc);

    // Update conv_state: shift left, append current
    for (int k = 0; k < state_len - 1; k++) {
        conv_state[d * state_len + k] = conv_state[d * state_len + k + 1];
    }
    conv_state[d * state_len + state_len - 1] = x[d];
}

/**
 * Causal conv1d for multiple tokens (prefill path).
 * Processes tokens sequentially to maintain causal state.
 * Grid: (conv_dim / 256), Block: (256)
 * Each block handles one channel range across all tokens.
 */
__global__ void causal_conv1d_prefill_kernel(
    __nv_bfloat16 *__restrict__ out,          // [tokens, conv_dim]
    const __nv_bfloat16 *__restrict__ x,      // [tokens, conv_dim]
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ bias,
    __nv_bfloat16 *__restrict__ conv_state,   // [conv_dim, kernel_size-1]
    int conv_dim, int kernel_size, int tokens) {

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim) return;

    int ks = kernel_size;
    int state_len = ks - 1;

    // Load initial state into registers
    float history[4];
    for (int k = 0; k < state_len; k++) {
        history[k] = __bfloat162float(conv_state[d * state_len + k]);
    }

    float w[4];
    for (int k = 0; k < ks; k++) {
        w[k] = __bfloat162float(weight[d * ks + k]);
    }
    float b = __bfloat162float(bias[d]);

    // Process each token sequentially
    for (int t = 0; t < tokens; t++) {
        float cur = __bfloat162float(x[(long long)t * conv_dim + d]);
        history[state_len] = cur;

        float acc = b;
        for (int k = 0; k < ks; k++) {
            acc += w[k] * history[k];
        }
        out[(long long)t * conv_dim + d] = __float2bfloat16(acc);

        // Shift history
        for (int k = 0; k < state_len; k++) {
            history[k] = history[k + 1];
        }
    }

    // Write back final state
    for (int k = 0; k < state_len; k++) {
        conv_state[d * state_len + k] = __float2bfloat16(history[k]);
    }
}

void kernel_causal_conv1d(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                          const __nv_bfloat16 *weight,
                          const __nv_bfloat16 *bias,
                          __nv_bfloat16 *conv_state,
                          int conv_dim, int tokens, int kernel_size,
                          cudaStream_t stream) {
    int block = 256;
    int grid = (conv_dim + block - 1) / block;

    if (tokens == 1) {
        causal_conv1d_decode_kernel<<<grid, block, 0, stream>>>(
            out, x, weight, bias, conv_state, conv_dim, kernel_size);
    } else {
        causal_conv1d_prefill_kernel<<<grid, block, 0, stream>>>(
            out, x, weight, bias, conv_state, conv_dim, kernel_size, tokens);
    }
}
