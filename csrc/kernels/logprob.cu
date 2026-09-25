/**
 * logprob.cu - FP32 log-softmax + gather for the teacher-forced loss.
 *
 * The trainer needs `log P(label | prefix)` for selected positions and must not
 * retain a [tokens, vocab] logits tensor (docs/plan-numeric-contract.md, Stage 3:
 * "use chunked/fused LM-head/loss evaluation where needed to avoid retaining [T,V]
 * logits"). This kernel is the fused half: it consumes one row of FP32 logits on the
 * device and returns one FP32 log-probability, so only the scalar crosses back.
 *
 * The value is a natural-log log-softmax over the FP32 logits, matching the
 * trainer's FP32 differentiable loss region rather than the sampler's binary64
 * transform: `logits[label] - (max + log(sum(exp(logits - max))))`, with the max
 * subtracted for stability. It is deliberately not the attention LSE convention
 * (that one is base 2); the two live in different regions and must not be confused.
 */
#include "kernels.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

/* One block per row. Two strided passes (max, then the exponential sum) and a
 * direct read of the label's logit: the label is a known index, so no third pass
 * and no gather are needed. */
__global__ void logprob_gather_kernel(float *__restrict__ out, const float *__restrict__ logits,
                                      const int *__restrict__ labels, int rows, int vocab) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float *row_logits = logits + (size_t)row * vocab;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[i]);
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = local_max;
    __syncthreads();
    // The block size is fixed at 256, a power of two, so the tree reduction has no
    // unpaired lanes (the same rule the block-reduce kernels follow).
    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + step]);
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_sum += expf(row_logits[i] - row_max);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) shared[threadIdx.x] += shared[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int label = labels[row];
        // A label outside the vocabulary is a caller error the host validates; the
        // kernel still refuses to read out of bounds.
        out[row] = (label >= 0 && label < vocab)
                       ? row_logits[label] - (row_max + logf(shared[0]))
                       : NAN;
    }
}

}  // namespace

void kernel_logprob_gather(float *out, const float *logits, const int *labels, int rows, int vocab,
                           cudaStream_t stream) {
    if (rows < 0 || vocab <= 0) {
        throw std::invalid_argument("kernel_logprob_gather: invalid shape");
    }
    if (rows == 0) return;
    if (out == nullptr || logits == nullptr || labels == nullptr) {
        throw std::invalid_argument("kernel_logprob_gather: null buffer");
    }
    logprob_gather_kernel<<<rows, 256, 0, stream>>>(out, logits, labels, rows, vocab);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("kernel_logprob_gather: ") +
                                 cudaGetErrorString(status));
    }
}
