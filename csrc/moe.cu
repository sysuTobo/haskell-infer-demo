/**
 * moe.cu - Sparse feed-forward execution: routing, permute, expert GEMMs, combine.
 *
 * Correctness-first, matching the reference implementation's arithmetic order:
 * router logits in BF16 (as the reference linear layer produces), scoring in
 * FP32, then top-k selection with a deterministic lowest-index tie break.
 * Experts are executed one at a time over their packed token slices and empty
 * experts are skipped, which keeps single-request decode cheap (at most top_k
 * experts are non-empty). A capacity-padded batched GEMM would be the next step
 * if throughput mattered more than transparency.
 */
#include "moe.h"

#include "flashinfer_ops.h"
#include "kernels.h"
#include "layers.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;

/* Align a byte offset so every scratch section starts on a wide boundary. */
size_t align_up(size_t value, size_t alignment = 256) {
    return (value + alignment - 1) / alignment * alignment;
}

void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(status));
}

}  // namespace

/* ----------------------------------------------------------------------- */
/* Router                                                                  */
/* ----------------------------------------------------------------------- */

/* One block per token, one thread per block: E is at most a few hundred and this
 * runs once per layer, so the simple sequential form is the right trade. Ties
 * resolve to the lowest expert index because the scan uses a strict > test. */
__global__ void moe_router_topk_kernel(const __nv_bfloat16 *logits, int experts,
                                       int top_k, int norm_topk, int sigmoid_scoring,
                                       float scaling, int *ids, float *weights) {
    const int token = blockIdx.x;
    const __nv_bfloat16 *row = logits + (size_t)token * experts;
    float *selected = weights + (size_t)token * top_k;
    int *chosen = ids + (size_t)token * top_k;

    float max_score = -CUDART_INF_F;
    float sum = 0.0f;
    if (!sigmoid_scoring) {
        for (int e = 0; e < experts; ++e)
            max_score = fmaxf(max_score, __bfloat162float(row[e]));
        for (int e = 0; e < experts; ++e)
            sum += __expf(__bfloat162float(row[e]) - max_score);
    }

    for (int k = 0; k < top_k; ++k) {
        float best = -CUDART_INF_F;
        int best_index = -1;
        for (int e = 0; e < experts; ++e) {
            bool taken = false;
            for (int prior = 0; prior < k; ++prior)
                if (chosen[prior] == e) taken = true;
            if (taken) continue;
            const float value = __bfloat162float(row[e]);
            if (value > best) {
                best = value;
                best_index = e;
            }
        }
        chosen[k] = best_index;
        float score;
        if (sigmoid_scoring) {
            score = 1.0f / (1.0f + __expf(-best));
        } else {
            score = __expf(best - max_score) / sum;
        }
        selected[k] = score;
    }
    if (norm_topk && top_k > 0) {
        float total = 0.0f;
        for (int k = 0; k < top_k; ++k) total += selected[k];
        if (total > 0.0f)
            for (int k = 0; k < top_k; ++k) selected[k] /= total;
    }
    for (int k = 0; k < top_k; ++k) selected[k] *= scaling;
    /* The reference casts the routing weights to the model dtype before scaling
     * the expert outputs; matching that keeps the comparison tight. */
    for (int k = 0; k < top_k; ++k)
        selected[k] = __bfloat162float(__float2bfloat16_rn(selected[k]));
}

void kernel_moe_router_topk(const __nv_bfloat16 *logits, int tokens, int experts,
                            int top_k, int norm_topk, int scoring_sigmoid,
                            float scaling, int *ids, float *weights,
                            cudaStream_t stream) {
    moe_router_topk_kernel<<<tokens, 1, 0, stream>>>(
        logits, experts, top_k, norm_topk, scoring_sigmoid, scaling, ids, weights);
}

/* ----------------------------------------------------------------------- */
/* Permute                                                                 */
/* ----------------------------------------------------------------------- */

__global__ void moe_count_kernel(const int *ids, int entries, int experts, int *counts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < entries;
         i += gridDim.x * blockDim.x) {
        const int expert = ids[i];
        if (expert >= 0 && expert < experts) atomicAdd(counts + expert, 1);
    }
}

void kernel_moe_count(const int *ids, int entries, int experts, int *counts,
                      cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(counts, 0, sizeof(int) * experts, stream), "moe count zero");
    const int blocks = (entries + kThreads - 1) / kThreads;
    moe_count_kernel<<<blocks, kThreads, 0, stream>>>(ids, entries, experts, counts);
}

__global__ void moe_offsets_kernel(const int *counts, int experts, int *offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int running = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e] = running;
        running += counts[e];
    }
    offsets[experts] = running;
}

void kernel_moe_offsets(const int *counts, int experts, int *offsets, cudaStream_t stream) {
    moe_offsets_kernel<<<1, 1, 0, stream>>>(counts, experts, offsets);
}

__global__ void moe_permute_kernel(const int *ids, int tokens, int top_k, int experts,
                                   const int *offsets, int *cursor, int *slot_of,
                                   int *token_of_slot) {
    const int entries = tokens * top_k;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < entries;
         i += gridDim.x * blockDim.x) {
        const int expert = ids[i];
        if (expert < 0 || expert >= experts) continue;
        const int slot = offsets[expert] + atomicAdd(cursor + expert, 1);
        slot_of[i] = slot;
        token_of_slot[slot] = i / top_k;
    }
}

void kernel_moe_permute(const int *ids, int tokens, int top_k, int experts,
                        const int *offsets, int *cursor, int *slot_of,
                        int *token_of_slot, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(cursor, 0, sizeof(int) * experts, stream), "moe cursor zero");
    const int entries = tokens * top_k;
    const int blocks = (entries + kThreads - 1) / kThreads;
    moe_permute_kernel<<<blocks, kThreads, 0, stream>>>(
        ids, tokens, top_k, experts, offsets, cursor, slot_of, token_of_slot);
}

__global__ void moe_gather_kernel(const __nv_bfloat16 *input, const int *token_of_slot,
                                  __nv_bfloat16 *packed, int rows, int width) {
    const size_t total = (size_t)rows * width;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const int row = (int)(i / width);
        const int column = (int)(i % width);
        packed[i] = input[(size_t)token_of_slot[row] * width + column];
    }
}

void kernel_moe_gather(const __nv_bfloat16 *input, const int *token_of_slot,
                       __nv_bfloat16 *packed, int rows, int width, cudaStream_t stream) {
    const size_t total = (size_t)rows * width;
    const int blocks = (int)((total + kThreads - 1) / kThreads);
    moe_gather_kernel<<<blocks, kThreads, 0, stream>>>(input, token_of_slot, packed,
                                                       rows, width);
}

/* ----------------------------------------------------------------------- */
/* Combine                                                                 */
/* ----------------------------------------------------------------------- */

/* Store one combined value. The activation path rounds into BF16 here; the
 * expert-parallel partial is kept in FP32 because it is merged across ranks
 * before anything rounds, so rounding it early would cost a precision step the
 * single-rank path does not pay. */
__device__ inline void moe_store(__nv_bfloat16 *out, size_t index, float value) {
    out[index] = __float2bfloat16_rn(value);
}

__device__ inline void moe_store(float *out, size_t index, float value) {
    out[index] = value;
}

template <typename Output>
__global__ void moe_combine_kernel(const __nv_bfloat16 *expert_out, const int *slot_of,
                                   const float *weights, int top_k, int width,
                                   Output *out) {
    const int token = blockIdx.x;
    const size_t total = (size_t)width;
    for (size_t j = threadIdx.x; j < total; j += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int slot = slot_of[token * top_k + k];
            /* Negative slots are entries routed to another rank's experts. */
            if (slot < 0) continue;
            acc += weights[token * top_k + k] *
                   __bfloat162float(expert_out[(size_t)slot * width + j]);
        }
        moe_store(out, (size_t)token * width + j, acc);
    }
}

void kernel_moe_combine(const __nv_bfloat16 *expert_out, const int *slot_of,
                        const float *weights, int tokens, int top_k, int width,
                        __nv_bfloat16 *out, cudaStream_t stream) {
    moe_combine_kernel<<<tokens, kThreads, 0, stream>>>(expert_out, slot_of, weights,
                                                        top_k, width, out);
}

void kernel_moe_combine_f32(const __nv_bfloat16 *expert_out, const int *slot_of,
                            const float *weights, int tokens, int top_k, int width,
                            float *out, cudaStream_t stream) {
    moe_combine_kernel<<<tokens, kThreads, 0, stream>>>(expert_out, slot_of, weights,
                                                        top_k, width, out);
}

/* Pick the combine flavour from the destination type. */
inline void kernel_moe_combine_into(const __nv_bfloat16 *expert_out, const int *slot_of,
                                    const float *weights, int tokens, int top_k, int width,
                                    __nv_bfloat16 *out, cudaStream_t stream) {
    kernel_moe_combine(expert_out, slot_of, weights, tokens, top_k, width, out, stream);
}

inline void kernel_moe_combine_into(const __nv_bfloat16 *expert_out, const int *slot_of,
                                    const float *weights, int tokens, int top_k, int width,
                                    float *out, cudaStream_t stream) {
    kernel_moe_combine_f32(expert_out, slot_of, weights, tokens, top_k, width, out, stream);
}

__global__ void moe_localize_kernel(int *ids, int entries, int offset, int local) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < entries;
         i += gridDim.x * blockDim.x) {
        const int expert = ids[i];
        ids[i] = (expert >= offset && expert < offset + local) ? expert - offset : -1;
    }
}

void kernel_moe_localize_ids(int *ids, int entries, int offset, int local,
                             cudaStream_t stream) {
    if (entries <= 0 || local <= 0) return;
    const int blocks = (entries + kThreads - 1) / kThreads;
    moe_localize_kernel<<<blocks, kThreads, 0, stream>>>(ids, entries, offset, local);
}

__global__ void moe_scale_add_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *extra,
                                     const float *gate, int width) {
    const int token = blockIdx.x;
    const float scale = gate != nullptr ? 1.0f / (1.0f + __expf(-gate[token])) : 1.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const float value = __bfloat162float(extra[(size_t)token * width + j]);
        const float current = __bfloat162float(out[(size_t)token * width + j]);
        out[(size_t)token * width + j] = __float2bfloat16_rn(current + scale * value);
    }
}

void kernel_moe_scale_add(__nv_bfloat16 *out, const __nv_bfloat16 *extra,
                          const float *gate, int tokens, int width, cudaStream_t stream) {
    moe_scale_add_kernel<<<tokens, kThreads, 0, stream>>>(out, extra, gate, width);
}

/* ----------------------------------------------------------------------- */
/* Forward                                                                 */
/* ----------------------------------------------------------------------- */

/* Experts this rank holds (expert parallelism); <= 0 means all of them. */
static int moe_local_experts(const MoeConfig *moe) {
    return moe->num_local_experts > 0 ? moe->num_local_experts : moe->num_experts;
}

size_t moe_workspace_size(int tokens, const ModelDims *dims, const MoeConfig *moe) {
    if (tokens < 1) throw std::invalid_argument("MoE token count must be positive");
    const size_t t = (size_t)tokens;
    const size_t k = (size_t)moe->top_k;
    const size_t experts = (size_t)moe->num_experts;
    const size_t local = (size_t)moe_local_experts(moe);
    const size_t hidden = (size_t)dims->hidden_size;
    const size_t inner = (size_t)moe->intermediate_size;
    size_t bytes = 0;
    bytes += align_up(t * hidden * sizeof(__nv_bfloat16));    // post-normed input
    bytes += align_up(t * experts * sizeof(__nv_bfloat16));   // router logits
    bytes += align_up(t * k * sizeof(int));                   // expert ids
    bytes += align_up(t * k * sizeof(float));                 // routing weights
    bytes += align_up(local * sizeof(int));                   // counts (local experts)
    bytes += align_up((local + 1) * sizeof(int));             // offsets
    bytes += align_up(local * sizeof(int));                   // cursor
    bytes += align_up(t * k * sizeof(int));                   // slot per entry
    bytes += align_up(t * k * sizeof(int));                   // token per slot
    bytes += align_up(t * k * hidden * sizeof(__nv_bfloat16));  // packed input
    bytes += align_up(2 * t * k * inner * sizeof(__nv_bfloat16));  // [gate; up] per slice
    bytes += align_up(t * k * inner * sizeof(__nv_bfloat16));   // activated
    bytes += align_up(t * k * hidden * sizeof(__nv_bfloat16));  // expert output
    const size_t s = (size_t)moe->num_shared_experts;
    const size_t sis = (size_t)moe->shared_intermediate_size;
    if (s > 0) {
        bytes += align_up(t * hidden * sizeof(__nv_bfloat16));      // shared output
        bytes += align_up(2 * t * sis * sizeof(__nv_bfloat16));     // shared [gate; up]
        bytes += align_up(t * sis * sizeof(__nv_bfloat16));         // shared activated
    }
    if (moe->shared_gate_scalar) bytes += align_up(t * sizeof(float));  // gate scalar
    return bytes;
}

/* The routed partial, written either straight into the activation (BF16) or into
 * an FP32 buffer that expert parallelism merges across ranks first. */
template <typename Output>
static int moe_routed_forward(cublasHandle_t cublas, cudaStream_t stream,
                              const __nv_bfloat16 *normed, Output *out,
                              const MoeWeights *w, const MoeConfig *moe,
                              MoeScratch scratch, int tokens, const ModelDims *dims) {
    try {
        if (tokens < 1 || tokens > dims->max_chunk)
            throw std::invalid_argument("MoE token count exceeds max_chunk");
        if (moe->top_k < 1 || moe->top_k > moe->num_experts)
            throw std::invalid_argument("MoE top_k must be within [1, num_experts]");
        /* The activation kernel consumes [gate[n]; up[n]] and vectorizes in units
         * of 8 bf16 elements, so the per-expert width must be a multiple of 8. */
        if (moe->intermediate_size % 8 != 0)
            throw std::invalid_argument("MoE expert width must be a multiple of 8");

        const int hidden = dims->hidden_size;
        const int inner = moe->intermediate_size;
        const int experts = moe->num_experts;     /* the router covers them all */
        const int local = moe_local_experts(moe); /* this rank's experts */
        const int expert_offset = moe->expert_offset;
        const int top_k = moe->top_k;
        const int entries = tokens * top_k;

        /* Carve the scratch buffer in the same order as moe_workspace_size. */
        char *base = (char *)scratch.base;
        size_t used = 0;
        auto take = [&](size_t bytes) {
            char *result = base + used;
            used += align_up(bytes);
            return result;
        };
        __nv_bfloat16 *normed_input = (__nv_bfloat16 *)take((size_t)tokens * hidden * 2);
        __nv_bfloat16 *router_logits = (__nv_bfloat16 *)take((size_t)tokens * experts * 2);
        int *ids = (int *)take((size_t)entries * sizeof(int));
        float *weights = (float *)take((size_t)entries * sizeof(float));
        int *counts = (int *)take((size_t)local * sizeof(int));
        int *offsets = (int *)take((size_t)(local + 1) * sizeof(int));
        int *cursor = (int *)take((size_t)local * sizeof(int));
        int *slot_of = (int *)take((size_t)entries * sizeof(int));
        int *token_of_slot = (int *)take((size_t)entries * sizeof(int));
        __nv_bfloat16 *packed = (__nv_bfloat16 *)take((size_t)entries * hidden * 2);
        __nv_bfloat16 *gate_up = (__nv_bfloat16 *)take((size_t)entries * 2 * inner * 2);
        __nv_bfloat16 *activated = (__nv_bfloat16 *)take((size_t)entries * inner * 2);
        __nv_bfloat16 *expert_out = (__nv_bfloat16 *)take((size_t)entries * hidden * 2);

        if (w->post_norm_w != nullptr) {
            if (dims->norm_style == 1) {
                kernel_rms_norm_plain(normed_input, normed, w->post_norm_w, hidden, tokens,
                                      dims->rms_eps, stream);
            } else {
                kernel_gemma_rms_norm(normed_input, normed, w->post_norm_w, hidden, tokens,
                                      dims->rms_eps, stream);
            }
        } else {
            check_cuda(cudaMemcpyAsync(normed_input, normed, (size_t)tokens * hidden * 2,
                                       cudaMemcpyDeviceToDevice, stream),
                       "MoE input copy");
        }

        /* Routing: BF16 logits as the reference linear produces, scored in FP32. */
        const int gemm_status = gemm_bf16(cublas, router_logits, normed_input, w->router_w,
                                          tokens, experts, hidden);
        if (gemm_status != 0) return gemm_status;
        kernel_moe_router_topk(router_logits, tokens, experts, top_k,
                               moe->norm_topk_prob, moe->scoring_sigmoid,
                               moe->routed_scaling_factor, ids, weights, stream);

        /* Expert parallelism: keep only this rank's experts; the entries for
         * other ranks' experts become -1 and are skipped downstream. */
        if (local != experts || expert_offset != 0) {
            kernel_moe_localize_ids(ids, entries, expert_offset, local, stream);
        }
        kernel_moe_count(ids, entries, local, counts, stream);
        kernel_moe_offsets(counts, local, offsets, stream);
        check_cuda(cudaMemsetAsync(slot_of, 0xff, (size_t)entries * sizeof(int), stream),
                   "MoE slot init");
        kernel_moe_permute(ids, tokens, top_k, local, offsets, cursor, slot_of,
                           token_of_slot, stream);

        /* Expert loop: one plain GEMM per non-empty expert over its packed slice.
         * The per-expert token counts are device data, so this pulls the offsets
         * back once per MoE layer. That is one small sync per layer; the
         * capacity-padded batched-GEMM variant would remove it, at the cost of
         * padding every expert to the same capacity. */
        std::vector<int> host_offsets((size_t)local + 1);
        check_cuda(cudaMemcpyAsync(host_offsets.data(), offsets,
                                   (local + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream),
                   "MoE offsets download");
        check_cuda(cudaStreamSynchronize(stream), "MoE offsets sync");
        /* Only this rank's packed prefix is populated; the tail's slots are -1. */
        const int packed_rows = host_offsets[local];
        if (packed_rows > 0) {
            kernel_moe_gather(normed_input, token_of_slot, packed, packed_rows, hidden, stream);
        }

        for (int e = 0; e < local; ++e) {
            const int count = host_offsets[e + 1] - host_offsets[e];
            if (count == 0) continue;
            const size_t slice = (size_t)host_offsets[e];
            /* Expert e owns [offsets[e], offsets[e+1]) rows, so its gate/up slice
             * is [gate(count,I); up(count,I)] inside that region. */
            __nv_bfloat16 *gate = gate_up + slice * 2 * inner;
            __nv_bfloat16 *up = gate + (size_t)count * inner;
            const int status_gate = gemm_bf16(cublas, gate, packed + slice * hidden,
                                              w->experts_gate + (size_t)e * inner * hidden,
                                              count, inner, hidden);
            if (status_gate != 0) return status_gate;
            const int status_up = gemm_bf16(cublas, up, packed + slice * hidden,
                                            w->experts_up + (size_t)e * inner * hidden,
                                            count, inner, hidden);
            if (status_up != 0) return status_up;
            kernel_silu_mul(activated + slice * inner, gate, up, count * inner, stream);
            const int status_down = gemm_bf16(cublas, expert_out + slice * hidden,
                                              activated + slice * inner,
                                              w->experts_down + (size_t)e * hidden * inner,
                                              count, hidden, inner);
            if (status_down != 0) return status_down;
        }

        kernel_moe_combine_into(expert_out, slot_of, weights, tokens, top_k, hidden, out,
                                stream);
        check_cuda(cudaGetLastError(), "MoE routed forward");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "MoE routed forward failed: %s\n", e.what());
        return -1;
    }
}

int forward_moe_routed(cublasHandle_t cublas, cudaStream_t stream,
                       const __nv_bfloat16 *normed, __nv_bfloat16 *out,
                       const MoeWeights *w, const MoeConfig *moe,
                       MoeScratch scratch, int tokens, const ModelDims *dims) {
    return moe_routed_forward(cublas, stream, normed, out, w, moe, scratch, tokens, dims);
}

int forward_moe_routed_f32(cublasHandle_t cublas, cudaStream_t stream,
                           const __nv_bfloat16 *normed, float *out_f32,
                           const MoeWeights *w, const MoeConfig *moe,
                           MoeScratch scratch, int tokens, const ModelDims *dims) {
    return moe_routed_forward(cublas, stream, normed, out_f32, w, moe, scratch, tokens,
                              dims);
}

int forward_moe_shared(cublasHandle_t cublas, cudaStream_t stream,
                       const __nv_bfloat16 *normed, __nv_bfloat16 *out,
                       const MoeWeights *w, const MoeConfig *moe,
                       MoeScratch scratch, int tokens, const ModelDims *dims) {
    try {
        if (tokens < 1 || tokens > dims->max_chunk)
            throw std::invalid_argument("MoE token count exceeds max_chunk");
        const int shared = moe->num_shared_experts;
        if (shared <= 0) return 0;
        if (moe->shared_intermediate_size % 8 != 0)
            throw std::invalid_argument("shared expert width must be a multiple of 8");
        if (w->shared_gate == nullptr || w->shared_up == nullptr ||
            w->shared_down == nullptr)
            throw std::invalid_argument("shared expert weights are missing");

        const int hidden = dims->hidden_size;
        const size_t shared_inner = (size_t)moe->shared_intermediate_size;

        char *base = (char *)scratch.base;
        size_t used = 0;
        auto take = [&](size_t bytes) {
            char *result = base + used;
            used += align_up(bytes);
            return result;
        };
        __nv_bfloat16 *normed_input = (__nv_bfloat16 *)take((size_t)tokens * hidden * 2);
        __nv_bfloat16 *shared_out = (__nv_bfloat16 *)take((size_t)tokens * hidden * 2);
        /* [gate; up] adjacency is what the activation kernel consumes. */
        __nv_bfloat16 *shared_gate = (__nv_bfloat16 *)take(2 * (size_t)tokens * shared_inner * 2);
        __nv_bfloat16 *shared_up = shared_gate + (size_t)tokens * shared_inner;
        __nv_bfloat16 *shared_act = (__nv_bfloat16 *)take((size_t)tokens * shared_inner * 2);
        float *shared_scale = moe->shared_gate_scalar
            ? (float *)take((size_t)tokens * sizeof(float)) : nullptr;

        if (w->post_norm_w != nullptr) {
            if (dims->norm_style == 1) {
                kernel_rms_norm_plain(normed_input, normed, w->post_norm_w, hidden, tokens,
                                      dims->rms_eps, stream);
            } else {
                kernel_gemma_rms_norm(normed_input, normed, w->post_norm_w, hidden, tokens,
                                      dims->rms_eps, stream);
            }
        } else {
            check_cuda(cudaMemcpyAsync(normed_input, normed, (size_t)tokens * hidden * 2,
                                       cudaMemcpyDeviceToDevice, stream),
                       "MoE input copy");
        }

        /* Always-on experts: a dense MLP (or several, summed) on the same normed
         * input, optionally scaled per token by sigmoid(x @ w). The gate is a
         * single projection over the hidden state, so compute it first. */
        if (moe->shared_gate_scalar && w->shared_gate_scalar_w != nullptr) {
            const int status = gemm_bf16_f32out(cublas, shared_scale, normed_input,
                                                w->shared_gate_scalar_w, tokens, 1, hidden);
            if (status != 0) return status;
        }
        for (int e = 0; e < shared; ++e) {
            const int status_gate = gemm_bf16(cublas, shared_gate, normed_input,
                                              w->shared_gate + (size_t)e * shared_inner * hidden,
                                              tokens, (int)shared_inner, hidden);
            if (status_gate != 0) return status_gate;
            const int status_up = gemm_bf16(cublas, shared_up, normed_input,
                                            w->shared_up + (size_t)e * shared_inner * hidden,
                                            tokens, (int)shared_inner, hidden);
            if (status_up != 0) return status_up;
            kernel_silu_mul(shared_act, shared_gate, shared_up, tokens * (int)shared_inner, stream);
            const int status_down = gemm_bf16(cublas, shared_out, shared_act,
                                              w->shared_down + (size_t)e * hidden * shared_inner,
                                              tokens, hidden, (int)shared_inner);
            if (status_down != 0) return status_down;
            if (e == 0) {
                kernel_moe_scale_add(out, shared_out, shared_scale, tokens, hidden, stream);
            } else {
                kernel_moe_scale_add(out, shared_out, nullptr, tokens, hidden, stream);
            }
        }
        check_cuda(cudaGetLastError(), "MoE shared forward");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "MoE shared forward failed: %s\n", e.what());
        return -1;
    }
}

int forward_moe_ffn(cublasHandle_t cublas, cudaStream_t stream,
                    const __nv_bfloat16 *normed, __nv_bfloat16 *out,
                    const MoeWeights *w, const MoeConfig *moe,
                    MoeScratch scratch, int tokens, const ModelDims *dims) {
    const int status = forward_moe_routed(cublas, stream, normed, out, w, moe, scratch,
                                          tokens, dims);
    if (status != 0) return status;
    return forward_moe_shared(cublas, stream, normed, out, w, moe, scratch, tokens, dims);
}
