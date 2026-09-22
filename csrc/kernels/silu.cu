#include "kernels.h"

#include <flashinfer/activation.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

// Scalar callback required by FlashInfer's native activation template (as in its JIT).
__device__ __forceinline__ float SiLU(const float &value) {
    return value / (1.0f + __expf(-value));
}

}  // namespace

void kernel_silu_mul(__nv_bfloat16 *out, const __nv_bfloat16 *gate,
                     const __nv_bfloat16 *up, int n, cudaStream_t stream) {
    if (n < 0) {
        throw std::runtime_error("kernel_silu_mul: negative element count");
    }
    if (n == 0) return;
    if (!out || !gate || !up || up != gate + n) {
        throw std::runtime_error("kernel_silu_mul: expected contiguous [gate[n], up[n]] input");
    }
    constexpr int vec_size = 16 / sizeof(__nv_bfloat16);
    if (n >= vec_size && (n % vec_size != 0 ||
        reinterpret_cast<uintptr_t>(gate) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(out) % 16 != 0)) {
        throw std::runtime_error("kernel_silu_mul: vectorized input/output must be 16-byte aligned and n divisible by 8");
    }

    const int block = std::max(1, std::min(n / vec_size, 1024));
    flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, SiLU>
        <<<1, block, 0, stream>>>(out, gate, n);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("kernel_silu_mul: ") + cudaGetErrorString(status));
    }
}
