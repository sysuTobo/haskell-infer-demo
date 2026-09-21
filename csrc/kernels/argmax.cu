/**
 * argmax.cu - Find index of maximum value in a float array.
 * Uses a two-pass reduction: block-level then atomic.
 */
#include "kernels.h"

__global__ void argmax_kernel(const float *__restrict__ logits,
                              int vocab_size, int64_t *__restrict__ result) {
    __shared__ float s_max;
    __shared__ int s_idx;

    float local_max = -1e30f;
    int local_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    __shared__ float warp_max[32];
    __shared__ int warp_idx[32];
    if (lane == 0) {
        warp_max[warp] = local_max;
        warp_idx[warp] = local_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best = -1e30f;
        int best_idx = 0;
        int num_warps = blockDim.x / 32;
        for (int w = 0; w < num_warps; w++) {
            if (warp_max[w] > best) {
                best = warp_max[w];
                best_idx = warp_idx[w];
            }
        }
        *result = (int64_t)best_idx;
    }
}

void kernel_argmax(int64_t *result, const float *logits, int vocab_size,
                   cudaStream_t stream) {
    // Single block with 1024 threads for vocab_size=248320
    argmax_kernel<<<1, 1024, 0, stream>>>(logits, vocab_size, result);
}
