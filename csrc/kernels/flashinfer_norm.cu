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
