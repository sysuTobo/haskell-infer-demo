/**
 * rmsnorm.cu - GemmaRMSNorm kernel.
 *
 * Gemma variant: weight is stored as (w + 1) in f32, computation is:
 *   out = (x_f32 * weight_p1) * rsqrt(mean(x_f32^2) + eps)
 * cast back to BF16.
 *
 * One block per row (token). Block size 512 threads.
 * For hidden_size=5120, each thread handles 10 elements.
 */

#include "kernels.h"
#include <cuda_bf16.h>
#include <math.h>

#define BLOCK_SIZE 512
#define EPS 1e-6f

/**
 * Plain RMSNorm (no residual).
 * Grid: (rows), Block: (BLOCK_SIZE)
 */
__global__ void rms_norm_kernel(__nv_bfloat16 *__restrict__ out,
                                const __nv_bfloat16 *__restrict__ x,
                                const float *__restrict__ weight_p1,
                                int cols, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16 *x_row = x + (long long)row * cols;
    __nv_bfloat16 *out_row = out + (long long)row * cols;

    // Phase 1: compute sum of squares in f32
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = __bfloat162float(x_row[i]);
        sum_sq += val * val;
    }

    // Block-level reduction
    __shared__ float shared[BLOCK_SIZE / 32];
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();

    // First warp reduces the partial sums
    if (warp == 0) {
        sum_sq = (lane < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    float mean_sq = shared[0] / (float)cols;
    float scale = rsqrtf(mean_sq + eps);

    // Phase 2: normalize and scale
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = __bfloat162float(x_row[i]);
        float w = weight_p1[i];
        out_row[i] = __float2bfloat16(val * scale * w);
    }
}

/**
 * Fused add + RMSNorm: residual += x; out = rmsnorm(residual).
 * Grid: (rows), Block: (BLOCK_SIZE)
 */
__global__ void fused_add_rms_norm_kernel(__nv_bfloat16 *__restrict__ out,
                                          __nv_bfloat16 *__restrict__ residual,
                                          const __nv_bfloat16 *__restrict__ x,
                                          const float *__restrict__ weight_p1,
                                          int cols, float eps) {
    int row = blockIdx.x;
    long long offset = (long long)row * cols;

    // Phase 1: add x to residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float r = __bfloat162float(residual[offset + i]);
        float v = __bfloat162float(x[offset + i]);
        float new_r = r + v;
        residual[offset + i] = __float2bfloat16(new_r);
        sum_sq += new_r * new_r;
    }

    // Block reduction (same as above)
    __shared__ float shared[BLOCK_SIZE / 32];
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
        if (lane == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(shared[0] / (float)cols + eps);

    // Phase 2: normalize
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float r = __bfloat162float(residual[offset + i]);
        out[offset + i] = __float2bfloat16(r * scale * weight_p1[i]);
    }
}

/* ------------------------------------------------------------------ */
/*  Host wrappers                                                     */
/* ------------------------------------------------------------------ */

void kernel_rms_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                     const float *weight_p1, int cols, int rows,
                     float eps, cudaStream_t stream) {
    rms_norm_kernel<<<rows, BLOCK_SIZE, 0, stream>>>(out, x, weight_p1, cols, eps);
}

void kernel_fused_add_rms_norm(__nv_bfloat16 *out, __nv_bfloat16 *residual,
                               const __nv_bfloat16 *x, const float *weight_p1,
                               int cols, int rows, float eps,
                               cudaStream_t stream) {
    fused_add_rms_norm_kernel<<<rows, BLOCK_SIZE, 0, stream>>>(
        out, residual, x, weight_p1, cols, eps);
}
