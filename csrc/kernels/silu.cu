/**
 * silu.cu - Fused SiLU(gate) * up for MLP.
 * out[i] = gate[i] * sigmoid(gate[i]) * up[i]
 */
#include "kernels.h"
#include <cuda_bf16.h>

__global__ void silu_mul_kernel(__nv_bfloat16 *__restrict__ out,
                                const __nv_bfloat16 *__restrict__ gate,
                                const __nv_bfloat16 *__restrict__ up,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float s = g / (1.0f + expf(-g));  // SiLU
    out[i] = __float2bfloat16(s * u);
}

void kernel_silu_mul(__nv_bfloat16 *out, const __nv_bfloat16 *gate,
                     const __nv_bfloat16 *up, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block, 0, stream>>>(out, gate, up, n);
}
