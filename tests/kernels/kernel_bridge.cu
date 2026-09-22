#include "fla_ops.h"
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
