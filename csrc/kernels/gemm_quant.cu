/**
 * gemm_quant.cu - The weight-only INT4 GEMM (plan "Q - Weight-only quantization", Q2).
 *
 * `C = A * B^T` with A a BF16 [M, K] activation, B an INT4-packed [N, K] weight in the Q0
 * format (see csrc/include/linear_weight.h: two two's-complement nibbles per byte, lower K
 * index in the low nibble, BF16 scales per 128-wide K group) and a BF16 [M, N] output
 * accumulated in FP32. There is no cuBLAS here on purpose: a 4-bit weight has no BF16 GEMM to
 * call, and the point of the kernel is the *bytes it reads*, not the arithmetic it does.
 *
 * The plan's Q2 says what not to do, and this is the alternative: "Reject full-weight
 * dequantization into a BF16 temporary on every forward: it restores weight traffic and can
 * erase the optimization. A one-time full BF16 expansion also does not retain device-weight
 * compression." So the weights are unpacked and scaled **inside the thread**: one 32-bit load
 * carries eight codes, the group's scale is hoisted out of the inner loop, and the only memory
 * traffic per output tile is the packed bytes plus the scales - a quarter of a BF16 read plus
 * 2/128 of a scale read, which is the axis F0's baseline identified as the decode bottleneck.
 *
 * Numerical contract: the accumulate is FP32 and each product is `code * scale * x` with the
 * scale BF16-widened, i.e. exactly the Q0 reference's `w = code * scale` multiplied by the
 * activation - so a kernel/output comparison against a *dequantized* weight reference is a
 * comparison of accumulation orders, not of formulas. That is what the gate checks.
 *
 * What this kernel is not, yet: it is a first SIMT implementation (one thread per output
 * element, no shared-memory tiling of the activation, no tensor cores). Its purpose in Q2's
 * order is to be admitted against the reference on synthetic shapes - "Start from pack/unpack
 * and synthetic GEMM fixtures, then route only admitted FFN roles to a kernel that
 * unpacks/scales inside its register/shared-memory tiles." Routing it into the dense FFN is
 * the next step, not this one.
 */
#include "layers.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

/* The eighth code of a 32-bit word, sign-extended from the format's four-bit two's complement
 * field. -8 is the format's reserved invalid value and the quantizer never writes it; a reader
 * that met one here would be reading a corrupt artifact, which is the loader's check (Q1), not
 * the kernel's. */
__device__ __forceinline__ int int4_code(uint32_t word, int j) {
    const int nibble = (int)((word >> (4 * j)) & 0xFu);
    return nibble >= 8 ? nibble - 16 : nibble;
}

/* One thread per output element. The K loop walks the packed row eight codes (one 32-bit load)
 * at a time with the group's scale hoisted, so nothing is dequantized into memory. */
__global__ void gemm_int4_kernel(const __nv_bfloat16 *__restrict__ a,
                                 const uint8_t *__restrict__ packed,
                                 const __nv_bfloat16 *__restrict__ scales,
                                 __nv_bfloat16 *__restrict__ out, int m, int n, int k,
                                 int group) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  /* output column, over N */
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  /* output row, over M */
    if (row >= n || col >= m) return;

    const __nv_bfloat16 *a_row = a + (size_t)col * (size_t)k;
    const uint8_t *w_row = packed + (size_t)row * (size_t)(k / 2);
    const __nv_bfloat16 *s_row = scales + (size_t)row * (size_t)(k / group);
    const int groups = k / group;

    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const float scale = __bfloat162float(s_row[g]);
        const uint8_t *w_group = w_row + (size_t)g * (size_t)(group / 2);
        const __nv_bfloat16 *a_group = a_row + (size_t)g * group;
        for (int i = 0; i < group; i += 8) {
            /* One aligned 32-bit load carries eight codes: the group is a multiple of 8 so
             * `i/2` is a multiple of 4 and the payload's start is 4-byte aligned. */
            const uint32_t word = *(const uint32_t *)(w_group + i / 2);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float weight = (float)int4_code(word, j) * scale;
                acc = fmaf(weight, __bfloat162float(a_group[i + j]), acc);
            }
        }
    }
    out[(size_t)col * (size_t)n + row] = __float2bfloat16(acc);
}

}  // namespace

int gemm_int4_bf16(const __nv_bfloat16 *a, const uint8_t *packed, const __nv_bfloat16 *scales,
                   __nv_bfloat16 *out, int M, int N, int K, int group, cudaStream_t stream) {
    /* Unsupported shapes and null buffers fail *here*, at creation time, rather than midway
     * through a request (plan Q2). */
    if (M < 0 || N < 0 || K <= 0 || group <= 0) {
        throw std::invalid_argument("gemm_int4_bf16: negative extent or nonpositive group");
    }
    if (M == 0 || N == 0) return 0;
    if (!a || !packed || !scales || !out) {
        throw std::invalid_argument("gemm_int4_bf16: null buffer");
    }
    if (K % group != 0) {
        throw std::invalid_argument("gemm_int4_bf16: K=" + std::to_string(K) +
                                    " is not a whole number of " + std::to_string(group) +
                                    "-wide groups");
    }
    if (group % 8 != 0 || (group / 2) % 4 != 0) {
        throw std::invalid_argument("gemm_int4_bf16: a group must be a multiple of 8 so that "
                                    "one 32-bit load carries eight codes; got " +
                                    std::to_string(group));
    }
    const dim3 block(16, 16);
    const dim3 grid((unsigned)((M + 15) / 16), (unsigned)((N + 15) / 16));
    gemm_int4_kernel<<<grid, block, 0, stream>>>(a, packed, scales, out, M, N, K, group);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemm_int4_bf16: ") + cudaGetErrorString(status));
    }
    return 0;
}
