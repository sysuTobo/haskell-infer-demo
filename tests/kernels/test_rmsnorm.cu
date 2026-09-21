/**
 * test_rmsnorm.cu - Unit test for GemmaRMSNorm kernel.
 * Compares GPU output against a CPU reference implementation.
 */
#include "kernels.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// CPU reference: GemmaRMSNorm
void cpu_rms_norm(std::vector<float> &out, const std::vector<float> &x,
                  const std::vector<float> &weight_p1, int cols, int rows, float eps) {
    for (int r = 0; r < rows; r++) {
        float sum_sq = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = x[r * cols + c];
            sum_sq += v * v;
        }
        float scale = 1.0f / sqrtf(sum_sq / cols + eps);
        for (int c = 0; c < cols; c++) {
            out[r * cols + c] = x[r * cols + c] * scale * weight_p1[c];
        }
    }
}

int main() {
    const int cols = 5120, rows = 4;
    const float eps = 1e-6f;

    // Allocate host data
    std::vector<float> h_x(cols * rows), h_w(cols), h_out_ref(cols * rows);
    srand(42);
    for (auto &v : h_x) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (auto &v : h_w) v = ((float)rand() / RAND_MAX - 0.5f) + 1.0f;  // weight+1

    // CPU reference
    cpu_rms_norm(h_out_ref, h_x, h_w, cols, rows, eps);

    // Convert to BF16 for GPU
    std::vector<__nv_bfloat16> h_x_bf16(cols * rows), h_w_bf16(cols);
    // Note: weight_p1 is f32 on GPU
    for (int i = 0; i < cols * rows; i++) h_x_bf16[i] = __float2bfloat16(h_x[i]);

    // Allocate device memory
    __nv_bfloat16 *d_x, *d_out;
    float *d_w;
    cudaMalloc(&d_x, cols * rows * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, cols * rows * sizeof(__nv_bfloat16));
    cudaMalloc(&d_w, cols * sizeof(float));

    cudaMemcpy(d_x, h_x_bf16.data(), cols * rows * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    kernel_rms_norm(d_out, d_x, d_w, cols, rows, eps, 0);
    cudaDeviceSynchronize();

    // Copy back and compare
    std::vector<__nv_bfloat16> h_out_bf16(cols * rows);
    cudaMemcpy(h_out_bf16.data(), d_out, cols * rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < cols * rows; i++) {
        float gpu_val = __bfloat162float(h_out_bf16[i]);
        float ref_val = h_out_ref[i];
        float diff = fabsf(gpu_val - ref_val);
        if (diff > max_diff) max_diff = diff;
        // BF16 has ~3 decimal digits of precision
        if (diff > 0.01f * fabsf(ref_val) + 0.001f) errors++;
    }

    printf("test_rmsnorm: max_diff=%.6f errors=%d/%d\n", max_diff, errors, cols * rows);
    if (errors > 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");

    cudaFree(d_x); cudaFree(d_out); cudaFree(d_w);
    return 0;
}
