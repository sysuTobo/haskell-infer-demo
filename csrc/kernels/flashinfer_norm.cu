#include "flashinfer_ops.h"

#include <flashinfer/trtllm/common/cudaFp8Utils.h>
#include <flashinfer/norm.cuh>
#include <flashinfer/pos_enc.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

void kernel_gemma_rms_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *raw_weight, int cols, int rows,
                           float eps, cudaStream_t stream) {
    if (rows < 0 || cols <= 0 || !std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("kernel_gemma_rms_norm: invalid shape or epsilon");
    }
    if (rows == 0) return;
    if (!out || !x || !raw_weight) {
        throw std::runtime_error("kernel_gemma_rms_norm: null buffer");
    }

    cudaError_t status;
    try {
        status = flashinfer::norm::GemmaRMSNorm<__nv_bfloat16>(
            const_cast<__nv_bfloat16 *>(x), const_cast<__nv_bfloat16 *>(raw_weight),
            out, rows, cols, /*stride_input=*/cols, /*stride_output=*/cols,
            eps, /*enable_pdl=*/false, stream);
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string("kernel_gemma_rms_norm: ") + error.what());
    }
    check_cuda(status, "kernel_gemma_rms_norm");
    check_cuda(cudaGetLastError(), "kernel_gemma_rms_norm launch");
}

void kernel_rms_norm_plain(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *raw_weight, int cols, int rows,
                           float eps, cudaStream_t stream) {
    if (rows < 0 || cols <= 0 || !std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("kernel_rms_norm_plain: invalid shape or epsilon");
    }
    if (rows == 0) return;
    if (!out || !x || !raw_weight) {
        throw std::runtime_error("kernel_rms_norm_plain: null buffer");
    }
    cudaError_t status;
    try {
        status = flashinfer::norm::RMSNorm<__nv_bfloat16>(
            const_cast<__nv_bfloat16 *>(x), const_cast<__nv_bfloat16 *>(raw_weight),
            out, rows, cols, /*stride_input=*/cols, /*stride_output=*/cols,
            eps, /*enable_pdl=*/false, stream);
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string("kernel_rms_norm_plain: ") + error.what());
    }
    check_cuda(status, "kernel_rms_norm_plain");
    check_cuda(cudaGetLastError(), "kernel_rms_norm_plain launch");
}

void kernel_flashinfer_rope(__nv_bfloat16 *q, __nv_bfloat16 *k,
                            const int64_t *positions, int tokens,
                            int q_heads, int kv_heads, int head_dim,
                            int rotary_dim, float theta, cudaStream_t stream) {
    const int vec_size = std::max(8, head_dim / 32);
    if (tokens < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % (2 * vec_size) != 0 ||
        !std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("kernel_flashinfer_rope: invalid shape, rotary alignment or theta");
    }
    if (tokens == 0) return;
    if (!q || !k || !positions) {
        throw std::runtime_error("kernel_flashinfer_rope: null buffer");
    }

    const size_t q_stride = static_cast<size_t>(q_heads) * head_dim;
    const size_t k_stride = static_cast<size_t>(kv_heads) * head_dim;
    cudaError_t status;
    try {
        status = flashinfer::BatchQKApplyRotaryPosIds<__nv_bfloat16, int64_t>(
            q, k, q, k, const_cast<int64_t *>(positions), tokens,
            q_heads, kv_heads, rotary_dim, head_dim,
            q_stride, head_dim, k_stride, head_dim,
            q_stride, head_dim, k_stride, head_dim,
            /*interleave=*/false, /*rope_scale=*/1.0f, theta, stream);
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string("kernel_flashinfer_rope: ") + error.what());
    }
    check_cuda(status, "kernel_flashinfer_rope");
    check_cuda(cudaGetLastError(), "kernel_flashinfer_rope launch");
}

/* ------------------------------------------------------------------ */
/*  F2: the residual update and the following norm, in one pass       */
/* ------------------------------------------------------------------ */

/* The plan's F2 asks for two outputs from one pass: the updated BF16 residual
 * `r = BF16(old_r + sublayer_out)` and the normalized activation. The rounding is the whole
 * point - the norm has to read the *rounded* r, because normalizing an unrounded FP32 sum is a
 * different function from normalizing what the next sublayer will read - so the update is
 * stored into the residual stream and the reduction reads it back from there. The weight
 * convention is a parameter: a Gemma norm uses `weight + 1` and a plain RMSNorm uses the weight
 * as stored, and the plan requires both to be retained. */
__global__ void residual_norm_kernel(__nv_bfloat16 *__restrict__ normed,
                                     __nv_bfloat16 *__restrict__ residual,
                                     const __nv_bfloat16 *__restrict__ sublayer,
                                     const __nv_bfloat16 *__restrict__ raw_weight,
                                     int hidden, float eps, int gemma) {
    const int row = blockIdx.x;
    const long long base = (long long)row * hidden;
    __nv_bfloat16 *res_row = residual + base;
    const __nv_bfloat16 *sub_row = sublayer + base;
    __nv_bfloat16 *out_row = normed + base;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float updated = __bfloat162float(res_row[i]) + __bfloat162float(sub_row[i]);
        const __nv_bfloat16 rounded = __float2bfloat16(updated);
        res_row[i] = rounded;
        const float value = __bfloat162float(rounded);
        sum_sq = fmaf(value, value, sum_sq);
    }

    __shared__ float shared[32];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();
    const int warps = (int)(blockDim.x / 32);
    if (warp == 0) {
        float total = (lane < warps) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) shared[0] = total;
    }
    __syncthreads();

    const float scale = rsqrtf(shared[0] / (float)hidden + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float value = __bfloat162float(res_row[i]);
        float weight = __bfloat162float(raw_weight[i]);
        if (gemma) weight += 1.0f;
        out_row[i] = __float2bfloat16(value * scale * weight);
    }
}

void kernel_residual_norm(__nv_bfloat16 *normed, __nv_bfloat16 *residual,
                          const __nv_bfloat16 *sublayer, const __nv_bfloat16 *raw_weight,
                          int hidden, int rows, float eps, int gemma, cudaStream_t stream) {
    if (hidden <= 0 || rows < 0 || !std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("kernel_residual_norm: invalid shape or epsilon");
    }
    if (rows == 0) return;
    if (!normed || !residual || !sublayer || !raw_weight) {
        throw std::runtime_error("kernel_residual_norm: null buffer");
    }
    /* A power-of-two block, because the warp reduction below assumes it: the plan's F gates
     * require any shared-memory tree reduction to handle its launch shape rather than assuming
     * multiples of 32 are powers of two. */
    const int block = 256;
    residual_norm_kernel<<<rows, block, 0, stream>>>(normed, residual, sublayer, raw_weight,
                                                     hidden, eps, gemma ? 1 : 0);
    check_cuda(cudaGetLastError(), "kernel_residual_norm launch");
}
