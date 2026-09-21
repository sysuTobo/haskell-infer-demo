/**
 * embedding.cu - Token embedding lookup.
 * out[token_i] = table[token_ids[token_i]]
 */
#include "kernels.h"
#include <cuda_bf16.h>

__global__ void embedding_kernel(__nv_bfloat16 *__restrict__ out,
                                 const __nv_bfloat16 *__restrict__ table,
                                 const int64_t *__restrict__ token_ids,
                                 int hidden_size, int tokens) {
    int t = blockIdx.x;
    if (t >= tokens) return;
    int64_t tid = token_ids[t];
    const __nv_bfloat16 *src = table + tid * hidden_size;
    __nv_bfloat16 *dst = out + (long long)t * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void kernel_embedding(__nv_bfloat16 *out, const __nv_bfloat16 *table,
                      const int64_t *token_ids, int hidden_size, int tokens,
                      cudaStream_t stream) {
    embedding_kernel<<<tokens, 256, 0, stream>>>(out, table, token_ids, hidden_size, tokens);
}
