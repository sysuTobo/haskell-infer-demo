/**
 * gemm.cu - cuBLAS BF16 GEMM wrapper.
 *
 * All linear layers use: out = x @ W^T
 * where x is [M, K] BF16, W is [N, K] BF16, out is [M, N] BF16.
 * cuBLAS column-major: C = alpha * op(A) * op(B) + beta * C
 * For row-major out[M,N] = x[M,K] @ W^T[K,N]:
 *   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...)
 *   with A=W (N,K row-major = K,N col-major, op=T → N,K), B=x (M,K row-major = K,M col-major)
 */

#include "kernels.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>

/**
 * BF16 GEMM: out[M,N] = x[M,K] @ W[N,K]^T
 * All matrices in row-major BF16.
 */
int gemm_bf16(cublasHandle_t handle,
              __nv_bfloat16 *out,        // [M, N] row-major
              const __nv_bfloat16 *x,    // [M, K] row-major
              const __nv_bfloat16 *W,    // [N, K] row-major (weight)
              int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;

    // Clear any pending CUDA error that might poison cuBLAS
    cudaError_t prior = cudaGetLastError();
    if (prior != cudaSuccess) {
        fprintf(stderr, "[gemm] WARNING: clearing prior CUDA error: %s\n",
                cudaGetErrorString(prior));
    }

    // Diagnostic: validate pointers
    cudaPointerAttributes attr_W, attr_x, attr_out;
    cudaError_t e1 = cudaPointerGetAttributes(&attr_W, W);
    cudaError_t e2 = cudaPointerGetAttributes(&attr_x, x);
    cudaError_t e3 = cudaPointerGetAttributes(&attr_out, out);
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        fprintf(stderr, "GEMM ptr invalid: W=%p(%d) x=%p(%d) out=%p(%d) errs=%d/%d/%d\n",
                W, attr_W.type, x, attr_x.type, out, attr_out.type, e1, e2, e3);
        return -2;
    }
    if (attr_W.device != attr_x.device || attr_W.device != attr_out.device) {
        fprintf(stderr, "GEMM device mismatch: W@%d x@%d out@%d\n",
                attr_W.device, attr_x.device, attr_out.device);
        return -3;
    }

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        x, CUDA_R_16BF, K,
        &beta,
        out, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM failed: %d (M=%d N=%d K=%d) W=%p x=%p out=%d dev=%d\n",
                status, M, N, K, W, x, attr_out.device);
        return -1;
    }
    return 0;
}

/**
 * BF16 GEMM with F32 output (for lm_head → logits).
 * out[M,N] f32 = x[M,K] bf16 @ W[N,K]^T bf16
 */
int gemm_bf16_f32out(cublasHandle_t handle,
                     float *out,                    // [M, N] row-major f32
                     const __nv_bfloat16 *x,        // [M, K] row-major bf16
                     const __nv_bfloat16 *W,        // [N, K] row-major bf16
                     int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        x, CUDA_R_16BF, K,
        &beta,
        out, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM f32out failed: %d\n", status);
        return -1;
    }
    return 0;
}
