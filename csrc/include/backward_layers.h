/**
 * backward_layers.h - The model-level backward: one layer's regions, wired in reverse
 * (plan Stage 5, "first bring up SFT after Stages 3-4").
 *
 * Stage 4 landed the backward of every region as a kernel with its own gate. Stage 5
 * has to chain them into an actual training step, and that is what this file is: the
 * layer-level traversal that consumes the values Stage 3's step retained
 * (`mixerOut`, `ffnOut`, `residual`), recomputes the fine-grained intermediates the
 * reverse pass needs, accumulates every parameter's FP32 gradient, and hands the
 * residual stream's gradient to the next layer down.
 *
 * Three decisions are worth stating, because each trades one resource for another:
 *
 *   - **The fine-grained activations are recomputed, not retained.** The step keeps the
 *     sublayer boundaries; a sublayer's backward re-runs its own forward into the layer
 *     workspace (the same buffers, so no new allocation) and then walks its regions in
 *     reverse. That doubles the arithmetic per sublayer and keeps the retained set at
 *     three values per layer. Retaining instead is the other side of the same trade,
 *     and the plan's "saved or recomputed statistics" is what makes it a choice.
 *   - **The operands are the values the forward read.** A GEMM's dW and dX consume the
 *     FP32 *widening* of the BF16 weight the forward used (not the FP32 master, which
 *     differs from it by the publication rounding), and the widened activation. A cast
 *     is identity for gradient propagation, so the faithful operand is the rounded one.
 *   - **The residual's identity branch is an explicit add.** `r1 = r0 + m` and
 *     `r2 = r1 + f` are BF16 adds, so their gradient is the identity in FP32: the
 *     incoming gradient flows to both branches, and the sublayer's internal backward
 *     *adds* into the same running gradient. The two are kept in separate buffers
 *     rather than aliased, because an aliased accumulate would depend on kernel
 *     ordering inside a sublayer for its correctness.
 *
 * The wiring covers the dense path of the trainer allowlist: the full-attention mixer
 * (QKV projection with the fused output gate, Q/K per-head norm, partial RoPE, the KV
 * write, the attention core from its saved base-2 LSE, the output gate and the output
 * projection) and the dense MLP feed-forward, with the residual adds and all four
 * norms. The GDN mixer and the sparse FFN are outside this file's first cut and are
 * named in the stage's Status block as the remaining wiring.
 */
#ifndef HASKELL_INFER_BACKWARD_LAYERS_H
#define HASKELL_INFER_BACKWARD_LAYERS_H

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "layers.h"
#include "model_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One trainable role of one layer: the BF16 compute weight the forward read (and the
 * backward widens as an operand) and the FP32 gradient accumulator the optimizer later
 * consumes. A role the layer does not have has both pointers null. */
struct TrainRole {
    const __nv_bfloat16 *compute;
    float *grad;
};

/* A layer's roles, named as the descriptor's roles are. `grad` is null for a frozen
 * role (the GDN A_log), which is why the gradient is accumulated through an explicit
 * pointer rather than "the role exists, so it trains". */
struct LayerBackwardWeights {
    struct TrainRole input_norm;   /* the mixer's input norm (weight+1 for gemma) */
    struct TrainRole post_norm;    /* the feed-forward's input norm */
    struct TrainRole q_proj;       /* fused Q (and, with a gate, Q+gate interleaved) */
    struct TrainRole k_proj;
    struct TrainRole v_proj;
    struct TrainRole o_proj;
    struct TrainRole q_norm;       /* per-head, over [tokens * num_heads, head_dim] */
    struct TrainRole k_norm;
    struct TrainRole gate_proj;
    struct TrainRole up_proj;
    struct TrainRole down_proj;
};

/* Everything one layer's backward needs. `workspace` is the layer's forward scratch,
 * reused for the recompute; `cast_f32` is an FP32 widening buffer at least
 * `cast_f32_elements` long; `grad_f32` is an FP32 scratch at least two hidden-state
 * rows long (the running residual gradients). */
struct LayerBackwardCtx {
    cublasHandle_t cublas;
    cudaStream_t stream;
    const ModelDims *dims;
    int tokens;
    int layer;
    int64_t seq_len;               /* the attention's causal range; == tokens for a
                                    * full-sequence step (Stage 4 requires it) */
    int mixer;                     /* ENGINE_MIXER_* */
    int ffn;                       /* ENGINE_FFN_* */
    struct LayerBackwardWeights w;
    __nv_bfloat16 *workspace;
    /* Where the backward's own BF16 tail starts inside `workspace`: the forward's
     * scratch layout ends here, and the backward needs one hidden-state-sized buffer
     * beyond it (the narrowed residual). `layer_backward_scratch` reports the total. */
    size_t ws_backward_offset;
    float *cast_f32;
    size_t cast_f32_elements;
    float *grad_f32;               /* >= 2 * tokens * hidden_size floats */
    size_t grad_f32_elements;
    const int64_t *positions;
    __nv_bfloat16 *kv_cache;

    /* Retained by the step (FP32 widenings of the BF16 values the forward read, so the
     * identity branches are exact and the residual stream can be rebuilt). */
    const float *residual_in;      /* the layer's input, pre-mixer */
    const float *mixer_out;        /* the mixer's output, pre-add */
};

/* What one layer's backward needs, in elements. The engine sizes the three buffers from
 * this so a too-small pool is a named refusal inside the backward rather than an
 * overwrite of another value. */
struct LayerBackwardScratch {
    long long workspace_elements;  /* total, including the backward's BF16 tail */
    long long cast_elements;
    long long grad_elements;
};

void layer_backward_scratch(const ModelDims *dims, int tokens, int mixer_kind, int ffn_kind,
                            struct LayerBackwardScratch *out);

/* Reverse one layer. `d_out` is the gradient of the layer's output (the running
 * gradient of the reverse walk, FP32 [tokens, hidden]); `d_residual_io` is the
 * gradient of the layer's *input*, accumulated in place ([tokens, hidden] FP32).
 *
 * The layer's identity branches are the caller's: a layer is
 * `r1 = r0 + mixer(r0)`, `r2 = r1 + ffn(r1)`, so `dL/dr1` and `dL/dr0` need the
 * incoming gradient added before each sublayer's internal backward contributes. The
 * function does that, so a caller only has to seed `d_residual_io` from the next
 * layer up and pass the same tensor on. */
void backward_layer(const struct LayerBackwardCtx *ctx, const float *d_out, float *d_residual_io);

/* The per-row inverse RMS the norm backwards consume. The forward's norms (FlashInfer)
 * do not return it, so the backward recomputes it from the same FP32 widening of the
 * BF16 input with the same formula (mean of squares + eps, then rsqrt). Exported
 * because the engine also needs it for the final norm before the LM head. */
void kernel_rms_inv(float *out, const float *x, int cols, int rows, float eps,
                    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_BACKWARD_LAYERS_H */
