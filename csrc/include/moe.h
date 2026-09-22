/**
 * moe.h - Mixture-of-experts feed-forward: weights, scratch and entry point.
 *
 * Layout: routed expert tensors are stored fused, one contiguous block per
 * expert, so the expert loop can run a plain GEMM on each slice:
 *   experts_gate [E, I, H]  (up the same shape, down [E, H, I])
 *
 * Routing: router logits are produced in BF16 (matching the reference), then
 * scored in FP32 -- softmax or sigmoid -- top-k selected, optionally
 * renormalized, and scaled. Selected (expert, weight) pairs are permuted into a
 * packed token list per expert; experts with no tokens are skipped, which is what
 * single-request decode almost always hits.
 */
#ifndef HASKELL_INFER_MOE_H
#define HASKELL_INFER_MOE_H

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>

/* Defined in layers.h; kept as a forward declaration because layers.h owns the
 * MoE weight fields and therefore includes this header. */
struct ModelDims;

/* Routing configuration (from the descriptor's moe_* fields). */
typedef struct {
    int num_experts;
    int top_k;
    int intermediate_size;      /* per-expert FFN width */
    int norm_topk_prob;         /* renormalize the selected weights */
    int scoring_sigmoid;        /* 0 = softmax, 1 = sigmoid */
    float routed_scaling_factor;
} MoeConfig;

/* Per-layer routed-expert weights, fused per expert (owned by the layer). */
typedef struct {
    const __nv_bfloat16 *router_w;    // [E, H]
    const __nv_bfloat16 *experts_gate; // [E, I, H]
    const __nv_bfloat16 *experts_up;   // [E, I, H]
    const __nv_bfloat16 *experts_down; // [E, H, I]
} MoeWeights;

/* Workspace for one MoE forward, sized by moe_workspace_size(). */
typedef struct {
    void *base;
    size_t bytes;
} MoeScratch;

/* Bytes needed for `tokens` tokens at the model's MoE configuration. */
size_t moe_workspace_size(int tokens, const struct ModelDims *dims, const MoeConfig *moe);

/* MoE feed-forward: out[T, H] = combine(top_k experts of normed[T, H]).
 * `normed` must already be the post-attention RMSNorm output. */
int forward_moe_ffn(cublasHandle_t cublas, cudaStream_t stream,
                    const __nv_bfloat16 *normed, __nv_bfloat16 *out,
                    const MoeWeights *w, const MoeConfig *moe,
                    MoeScratch scratch, int tokens, const struct ModelDims *dims);

/* ----------------------------------------------------------------------- */
/* Kernels (exposed for the differential tests)                            */
/* ----------------------------------------------------------------------- */

/* Router: bf16 logits [T, E] -> fp32 scores -> top-k ids [T, K] + weights. */
void kernel_moe_router_topk(const __nv_bfloat16 *logits, int tokens, int experts,
                            int top_k, int norm_topk, int scoring_sigmoid,
                            float scaling, int *ids, float *weights,
                            cudaStream_t stream);

/* Histogram of expert ids into counts[E]. */
void kernel_moe_count(const int *ids, int entries, int experts, int *counts,
                      cudaStream_t stream);

/* Exclusive prefix sum of counts[E] into offsets[E+1] (single thread). */
void kernel_moe_offsets(const int *counts, int experts, int *offsets,
                        cudaStream_t stream);

/* Place (token, slot) entries: packed slot per entry and source token per slot. */
void kernel_moe_permute(const int *ids, int tokens, int top_k, int experts,
                        const int *offsets, int *cursor, int *slot_of,
                        int *token_of_slot, cudaStream_t stream);

/* Gather rows: packed[slot, :] = input[token_of_slot[slot], :]. */
void kernel_moe_gather(const __nv_bfloat16 *input, const int *token_of_slot,
                       __nv_bfloat16 *packed, int rows, int width,
                       cudaStream_t stream);

/* Combine: out[t, :] += sum_k weights[t, k] * expert_out[slot_of[t, k], :]. */
void kernel_moe_combine(const __nv_bfloat16 *expert_out, const int *slot_of,
                        const float *weights, int tokens, int top_k, int width,
                        __nv_bfloat16 *out, cudaStream_t stream);

#endif /* HASKELL_INFER_MOE_H */
