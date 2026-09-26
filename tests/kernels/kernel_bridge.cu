#include "fla_ops.h"
#include "kernels.h"
#include "flashinfer_ops.h"
#include <cstdio>
#include <exception>

extern "C" int test_fla(void *out, const void *qkv, const void *a, const void *b,
                         const void *A_log, const void *bias, float *state, int tokens) {
    void *workspace = nullptr;
    try {
        cudaError_t status = cudaMalloc(&workspace, kernel_fla_workspace_size(tokens, 48));
        if (status != cudaSuccess) return int(status);
        cudaMemset(workspace, 0xff, kernel_fla_workspace_size(tokens, 48));
        kernel_fla_gdn(static_cast<__nv_bfloat16 *>(out),
                        static_cast<const __nv_bfloat16 *>(qkv),
                        static_cast<const __nv_bfloat16 *>(a),
                        static_cast<const __nv_bfloat16 *>(b),
                        static_cast<const __nv_bfloat16 *>(A_log),
                        static_cast<const __nv_bfloat16 *>(bias), state,
                        workspace, tokens, 16, 48, nullptr);
        status = cudaDeviceSynchronize();
        cudaFree(workspace);
        return int(status);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "test_fla: %s\n", error.what());
        if (workspace) cudaFree(workspace);
        return -1;
    }
}

extern "C" int test_norm(void *out, const void *x, const void *weight, int rows, int cols) {
    try {
        kernel_gemma_rms_norm(static_cast<__nv_bfloat16 *>(out),
                              static_cast<const __nv_bfloat16 *>(x),
                              static_cast<const __nv_bfloat16 *>(weight), cols, rows, 1e-6f, nullptr);
        return int(cudaDeviceSynchronize());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "test_norm: %s\n", error.what());
        return -1;
    }
}

extern "C" int test_rope(void *q, void *k, const int64_t *positions, int tokens) {
    try {
        kernel_flashinfer_rope(static_cast<__nv_bfloat16 *>(q),
                               static_cast<__nv_bfloat16 *>(k), positions,
                               tokens, 24, 4, 256, 64, 1e7f, nullptr);
        return int(cudaDeviceSynchronize());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "test_rope: %s\n", error.what());
        return -1;
    }
}

extern "C" int test_silu_mul(void *out, const void *gate, int n) {
    try {
        kernel_silu_mul(static_cast<__nv_bfloat16 *>(out),
                        static_cast<const __nv_bfloat16 *>(gate),
                        static_cast<const __nv_bfloat16 *>(gate) + n, n, nullptr);
        return int(cudaDeviceSynchronize());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "test_silu_mul: %s\n", error.what());
        return -1;
    }
}

/* The row-interleaved contract of the single gate/up GEMM (plan F1). */
extern "C" int test_silu_mul_packed(void *out, const void *packed, int tokens,
                                    int intermediate) {
    try {
        kernel_silu_mul_packed(static_cast<__nv_bfloat16 *>(out),
                               static_cast<const __nv_bfloat16 *>(packed), tokens,
                               intermediate, nullptr);
        return int(cudaDeviceSynchronize());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "test_silu_mul_packed: %s\n", error.what());
        return -1;
    }
}

/* The weight-only INT4 GEMM (plan Q2): the activation in BF16, the weight packed in the Q0
 * format, the output in BF16. */
extern "C" int test_gemm_int4(void *out, const void *a, const void *packed, const void *scales,
                              int M, int N, int K, int group) {
    try {
        gemm_int4_bf16(static_cast<const __nv_bfloat16 *>(a),
                       static_cast<const uint8_t *>(packed),
                       static_cast<const __nv_bfloat16 *>(scales),
                       static_cast<__nv_bfloat16 *>(out), M, N, K, group, nullptr);
        return int(cudaDeviceSynchronize());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "test_gemm_int4: %s\n", error.what());
        return -1;
    }
}
