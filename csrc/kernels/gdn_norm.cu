/**
 * gdn_norm.cu - Gated RMS normalization for GDN layers.
 *
 * After the delta rule output, apply:
 *   out = rmsnorm(x) * sigmoid(z)
 *
 * where rmsnorm uses the Gemma variant (weight+1 in f32).
 * This is the "RMSNormGated" from the GDN architecture.
 *
 * Also includes the l2norm + gating preparation step
 * (fused_post_conv_prep from vLLM).
 */

#include "kernels.h"
#include <cuda_bf16.h>
#include <math.h>

#define BLOCK 256

/**
 * Gated RMSNorm: out = rmsnorm(x, weight) * sigmoid(z)
 * Grid: (tokens), Block: (BLOCK)
 */
__global__ void gdn_gated_norm_kernel(__nv_bfloat16 *__restrict__ out,
                                      const __nv_bfloat16 *__restrict__ x,
                                      const __nv_bfloat16 *__restrict__ z,
                                      const float *__restrict__ weight_p1,
                                      int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16 *x_row = x + (long long)row * dim;
    const __nv_bfloat16 *z_row = z + (long long)row * dim;
    __nv_bfloat16 *out_row = out + (long long)row * dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK) {
        float val = __bfloat162float(x_row[i]);
        sum_sq += val * val;
    }

    // Block reduction
    __shared__ float shared[BLOCK / 32];
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < BLOCK / 32) ? shared[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
        if (lane == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(shared[0] / (float)dim + eps);

    // Normalize and gate with swish (config: output_gate_type=swish)
    // swish(z) = z * sigmoid(z)
    for (int i = threadIdx.x; i < dim; i += BLOCK) {
        float val = __bfloat162float(x_row[i]);
        float gate = __bfloat162float(z_row[i]);
        float swish = gate / (1.0f + expf(-gate));
        float normalized = __bfloat162float(__float2bfloat16(val * scale));
        float weighted = __bfloat162float(__float2bfloat16(normalized * weight_p1[i]));
        out_row[i] = __float2bfloat16(weighted * swish);
    }
}

void kernel_gdn_gated_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *z, const float *weight_p1,
                           int dim, int tokens, float eps,
                           cudaStream_t stream) {
    gdn_gated_norm_kernel<<<tokens, BLOCK, 0, stream>>>(
        out, x, z, weight_p1, dim, eps);
}

/* ------------------------------------------------------------------ */
/*  Small helpers                                                     */
/* ------------------------------------------------------------------ */

/** Cast BF16 to F32. */
__global__ void cast_bf16_f32_kernel(float *__restrict__ out,
                                     const __nv_bfloat16 *__restrict__ in,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(in[i]);
}

void kernel_cast_bf16_f32(float *out, const __nv_bfloat16 *in, int n,
                          cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cast_bf16_f32_kernel<<<grid, block, 0, stream>>>(out, in, n);
}

/** Cast F32 to BF16: one rounding, for a merged FP32 partial becoming an
 *  activation. */
__global__ void cast_f32_bf16_kernel(__nv_bfloat16 *__restrict__ out,
                                     const float *__restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16_rn(in[i]);
}

void kernel_cast_f32_bf16(__nv_bfloat16 *out, const float *in, int n,
                          cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cast_f32_bf16_kernel<<<grid, block, 0, stream>>>(out, in, n);
}

/* ------------------------------------------------------------------ */
/*  Residual add: dst[i] += src[i] (BF16)                             */
/* ------------------------------------------------------------------ */

__global__ void residual_add_kernel(__nv_bfloat16 *__restrict__ dst,
                                    const __nv_bfloat16 *__restrict__ src,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = __bfloat162float(dst[i]);
    float s = __bfloat162float(src[i]);
    dst[i] = __float2bfloat16(d + s);
}

void kernel_residual_add(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                         int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    residual_add_kernel<<<grid, block, 0, stream>>>(dst, src, n);
}
