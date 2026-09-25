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

// Elementwise SiLU in place, for callers that hold the pre-activation in a
// buffer of their own (the GDN conv output). Separate from kernel_silu_mul: it
// takes one buffer, not a contiguous [gate, up] pair.
__global__ void silu_inplace_kernel(__nv_bfloat16 *__restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(v / (1.0f + expf(-v)));
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

void kernel_silu_inplace(__nv_bfloat16 *x, int n, cudaStream_t stream) {
    if (n < 0) {
        throw std::runtime_error("kernel_silu_inplace: negative element count");
    }
    if (n == 0) return;
    if (!x) {
        throw std::runtime_error("kernel_silu_inplace: null buffer");
    }
    silu_inplace_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, n);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("kernel_silu_inplace: ") + cudaGetErrorString(status));
    }
}
