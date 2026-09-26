/**
 * backward_layers.cu - The layer-level backward (plan Stage 5). See
 * csrc/include/backward_layers.h for the design and the three trade-offs it takes.
 *
 * Every function here mirrors its forward counterpart in csrc/kernels/layers.cu, in
 * reverse, using the Stage-4 kernels. That is deliberate: a backward whose op order
 * drifts from the forward's is the gradient of a different function, and the device
 * gate (`tests/test_sft.py`) compares a whole training step against a torch run on the
 * same weights and data, which is what catches a drift.
 *
 * Three disciplines keep the wiring honest and are worth naming, because each one is a
 * bug class this project has already paid for:
 *
 *   - **Scratch is taken from sized pools, never from hand-computed offsets.** The
 *     engine sizes `cast_f32` and `grad_f32` through `layer_backward_scratch`, and each
 *     use takes a slot from a bump allocator that throws when the pool is exhausted -
 *     so an under-sized buffer is a refusal, not a silent overwrite of another value.
 *   - **A host loop never touches device memory.** Every copy is a `cudaMemcpyAsync` on
 *     the layer's stream. (A host `for` over a device pointer compiles and then
 *     segfaults; it appeared three times in Stage 4.)
 *   - **A gradient with more than one consumer accumulates.** The gate/up pair and the
 *     Q/K/V projections share one input, so their `gemm_backward_dx` calls pass
 *     accumulate=1; the attention core's dK/dV are pre-zeroed because that kernel
 *     accumulates with atomics across the query axis.
 */
#include "backward_layers.h"

#include "flashinfer_ops.h"
#include "kernels.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void check_launch(const char *op) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

void check_cuda(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

/* A bump allocator over one FP32 pool. `take` throws rather than wrapping, so a pool
 * that is too small is a named refusal. */
struct Pool {
    float *base;
    size_t elements;
    size_t cursor;

    float *take(size_t n, const char *what) {
        if (cursor + n > elements) {
            throw std::runtime_error(std::string("backward_layers: ") + what +
                                     ": pool exhausted (needs " + std::to_string(cursor + n) +
                                     " of " + std::to_string(elements) + ")");
        }
        float *p = base + cursor;
        cursor += n;
        return p;
    }
};

struct Buffers {
    cublasHandle_t cublas;
    cudaStream_t stream;
    __nv_bfloat16 *ws;      /* the layer workspace (the forward's, reused) */
    __nv_bfloat16 *ws_tail; /* TH of BF16 for the narrowed residual */
    Pool cast;
    Pool grad;
};

void with_stream(cublasHandle_t cublas, cudaStream_t stream) {
    if (cublasSetStream(cublas, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("backward_layers: cublasSetStream failed");
    }
}

void gemm_fwd(Buffers *b, __nv_bfloat16 *out, const __nv_bfloat16 *x, const __nv_bfloat16 *w,
              int M, int N, int K, const char *what) {
    if (gemm_bf16(b->cublas, out, x, w, M, N, K) != 0) {
        throw std::runtime_error(std::string("backward_layers: gemm_bf16 failed in ") + what);
    }
}

/* The FP32 widening of a BF16 tensor: the operand a gradient consumes, because a cast
 * is identity for gradient propagation and the faithful derivative is taken at the
 * rounded value the forward read. */
float *widen(Buffers *b, const __nv_bfloat16 *src, size_t elements, const char *what) {
    if (src == nullptr) throw std::runtime_error(std::string("backward_layers: ") + what + " null");
    float *dst = b->cast.take(elements, what);
    kernel_cast_bf16_f32(dst, src, (int)elements, b->stream);
    check_launch(what);
    return dst;
}

float *widen_weight(Buffers *b, const struct TrainRole *role, size_t elements, const char *what) {
    if (role->compute == nullptr) {
        throw std::runtime_error(std::string("backward_layers: ") + what +
                                 " has no compute weight");
    }
    return widen(b, role->compute, elements, what);
}

/* The weight gradient's target: a frozen role has no slot, so the accumulation lands in
 * a pool slot whose value is discarded. Passing the weight itself (a widening) would
 * silently corrupt the operand. */
float *weight_grad(Buffers *b, const struct TrainRole *role, size_t elements, const char *what) {
    if (role->grad != nullptr) return role->grad;
    float *discard = b->cast.take(elements, what);
    for (size_t i = 0; i < elements; ++i) discard[i] = 0.0f;
    return discard;
}

/* The norm's input as BF16: the retained boundary is FP32 (an exact widening of the
 * BF16 the forward read), so narrowing back is lossless and gives the forward's own
 * value for the recompute. */
void narrow_to_bf16(cudaStream_t stream, __nv_bfloat16 *out, const float *src, size_t elements,
                    const char *what) {
    kernel_cast_f32_bf16(out, src, (int)elements, stream);
    check_launch(what);
}

void layer_norm_forward(cudaStream_t stream, __nv_bfloat16 *out, const __nv_bfloat16 *x,
                        const __nv_bfloat16 *weight, int cols, int rows, const ModelDims *dims,
                        const char *what) {
    if (weight == nullptr) {
        /* A family without this norm: the sublayer's input passes through unchanged. */
        kernel_cast_f32_bf16(out, nullptr, 0, stream);
        return;
    }
    if (dims->norm_style == 1) {
        kernel_rms_norm_plain(out, x, weight, cols, rows, dims->rms_eps, stream);
    } else {
        kernel_gemma_rms_norm(out, x, weight, cols, rows, dims->rms_eps, stream);
    }
    check_launch(what);
}

/* One row-block norm's backward: d_x is written (or accumulated into) at FP32, the
 * weight gradient accumulates into the role's slot, and the chain through the
 * reciprocal square root uses the recomputed inverse RMS. */
void norm_backward(Buffers *b, float *d_x, const struct TrainRole *role, const float *d_out,
                   const float *x_f32, const float *inv_rms, int cols, int rows, int accumulate,
                   const ModelDims *dims, const char *what) {
    if (role->compute == nullptr) {
        /* No norm: the gradient passes through, which is a copy when the caller owns a
         * fresh buffer and a no-op when it already accumulates in place. */
        const long long n = (long long)rows * cols;
        if (!accumulate) {
            check_cuda(cudaMemcpyAsync(d_x, d_out, (size_t)n * sizeof(float),
                                       cudaMemcpyDeviceToDevice, b->stream),
                       what);
        } else {
            kernel_f32_accumulate(d_x, d_out, n, b->stream);
            check_launch(what);
        }
        return;
    }
    float *w_f32 = widen_weight(b, role, (size_t)cols, what);
    float *dw = weight_grad(b, role, (size_t)cols, what);
    const int gemma = dims->norm_style == 0 ? 1 : 0;
    kernel_rmsnorm_backward(d_x, dw, d_out, x_f32, w_f32, inv_rms, cols, rows, gemma, accumulate,
                            b->stream);
    check_launch(what);
}

/* ------------------------------------------------ dense MLP ------------------ */

/* Mirrors forward_mlp: norm -> gate/up -> silu_mul -> down. */
void backward_mlp(const struct LayerBackwardCtx *ctx, Buffers *b, const float *residual_f32,
                  const float *d_out, float *d_res) {
    const ModelDims *dims = ctx->dims;
    const int H = dims->hidden_size;
    const int I = dims->intermediate_size;
    const size_t T = (size_t)ctx->tokens;
    const size_t TI = T * I;
    const size_t TH = T * H;

    __nv_bfloat16 *normed = b->ws;
    __nv_bfloat16 *gate = normed + TH;
    __nv_bfloat16 *up = gate + TI;
    __nv_bfloat16 *act = up + TI;

    /* ---- recompute the sublayer's forward ---- */
    narrow_to_bf16(b->stream, b->ws_tail, residual_f32, TH, "mlp residual narrowing");
    layer_norm_forward(b->stream, normed, b->ws_tail, ctx->w.post_norm.compute, H, ctx->tokens,
                       dims, "mlp norm recompute");
    gemm_fwd(b, gate, normed, ctx->w.gate_proj.compute, ctx->tokens, I, H, "mlp gate recompute");
    gemm_fwd(b, up, normed, ctx->w.up_proj.compute, ctx->tokens, I, H, "mlp up recompute");
    kernel_silu_mul(act, gate, up, (int)TI, b->stream);
    check_launch("mlp silu recompute");

    /* ---- reverse ---- */
    float *d_act = b->grad.take(TI, "mlp d_act");
    {
        float *w = widen_weight(b, &ctx->w.down_proj, (size_t)H * I, "down weight widen");
        if (gemm_backward_dx(b->cublas, d_act, d_out, w, ctx->tokens, H, I, 0) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dx failed in mlp down");
        }
        float *act_f32 = widen(b, act, TI, "mlp act widen");
        if (gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.down_proj, (size_t)H * I, "down grad"),
                             act_f32, d_out, ctx->tokens, H, I, 1) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dw failed in mlp down");
        }
    }
    float *d_gate = b->grad.take(TI, "mlp d_gate");
    float *d_up = b->grad.take(TI, "mlp d_up");
    kernel_silu_mul_backward(d_gate, d_up, d_act, gate, up, (int)TI, 0, b->stream);
    check_launch("silu_mul backward");
    float *d_normed = b->grad.take(TH, "mlp d_normed");
    {
        float *w_gate = widen_weight(b, &ctx->w.gate_proj, (size_t)I * H, "gate weight widen");
        float *w_up = widen_weight(b, &ctx->w.up_proj, (size_t)I * H, "up weight widen");
        /* One input, two projections: the second call accumulates. */
        if (gemm_backward_dx(b->cublas, d_normed, d_gate, w_gate, ctx->tokens, I, H, 0) != 0 ||
            gemm_backward_dx(b->cublas, d_normed, d_up, w_up, ctx->tokens, I, H, 1) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dx failed in gate/up");
        }
        float *x_f32 = widen(b, normed, TH, "mlp normed widen");
        if (gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.gate_proj, (size_t)I * H, "gate grad"),
                             x_f32, d_gate, ctx->tokens, I, H, 1) != 0 ||
            gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.up_proj, (size_t)I * H, "up grad"),
                             x_f32, d_up, ctx->tokens, I, H, 1) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dw failed in gate/up");
        }
    }
    float *inv = b->grad.take(T, "mlp inv");
    kernel_rms_inv(inv, residual_f32, H, ctx->tokens, dims->rms_eps, b->stream);
    check_launch("mlp inv");
    norm_backward(b, d_res, &ctx->w.post_norm, d_normed, residual_f32, inv, H, ctx->tokens, 1, dims,
                  "mlp input norm backward");
}

/* ------------------------------------------------ full attention ------------- */

/* Mirrors forward_attention_layer. */
void backward_attention(const struct LayerBackwardCtx *ctx, Buffers *b, const float *residual_f32,
                        const float *d_out, float *d_res) {
    const ModelDims *dims = ctx->dims;
    const int H = dims->hidden_size;
    const int nH = dims->num_heads;
    const int nKV = dims->num_kv_heads;
    const int hd = dims->head_dim;
    const int Q = nH * hd;
    const int KV = nKV * hd;
    const int gated = dims->attn_output_gate;
    const size_t T = (size_t)ctx->tokens;
    const size_t TH = T * H;

    __nv_bfloat16 *normed = b->ws;
    __nv_bfloat16 *q_raw = normed + TH;
    __nv_bfloat16 *k_out = q_raw + T * 2 * Q;
    __nv_bfloat16 *v_out = k_out + T * KV;
    __nv_bfloat16 *q = v_out + T * KV;
    __nv_bfloat16 *gate = q + T * Q;
    __nv_bfloat16 *attn_out = gate + T * Q;
    __nv_bfloat16 *gated_out = attn_out + T * Q;

    float *lse = b->grad.take(T * nH, "attn lse");
    float *inv_q = b->grad.take(T * nH, "attn inv q");
    float *inv_k = b->grad.take(T * nKV, "attn inv k");
    float *inv_h = b->grad.take(T, "attn inv h");
    float *d_gated = b->grad.take(T * Q, "attn d_gated");
    float *d_attn = b->grad.take(T * Q, "attn d_attn");
    float *d_rq = b->grad.take(T * Q, "attn d_q_roped");
    float *d_rk = b->grad.take(T * KV, "attn d_k_roped");
    float *d_v = b->grad.take(T * KV, "attn d_v");
    float *d_q_pre = b->grad.take(T * Q, "attn d_q_pre");
    float *d_k_pre = b->grad.take(T * KV, "attn d_k_pre");
    float *d_q_norm = b->grad.take(T * Q, "attn d_q_norm");
    float *d_k_norm = b->grad.take(T * KV, "attn d_k_norm");
    float *d_q_raw = b->grad.take(T * 2 * Q, "attn d_q_raw");
    float *d_normed = b->grad.take(TH, "attn d_normed");
    float *q_pre_f32 = b->grad.take(T * Q, "attn q pre-norm value");
    float *k_pre_f32 = b->grad.take(T * KV, "attn k pre-norm value");

    /* ---- recompute ---- */
    narrow_to_bf16(b->stream, b->ws_tail, residual_f32, TH, "attn residual narrowing");
    layer_norm_forward(b->stream, normed, b->ws_tail, ctx->w.input_norm.compute, H, ctx->tokens,
                       dims, "attn input norm recompute");
    gemm_fwd(b, gated ? q_raw : q, normed, ctx->w.q_proj.compute, ctx->tokens, gated ? 2 * Q : Q, H,
             "attn q recompute");
    gemm_fwd(b, k_out, normed, ctx->w.k_proj.compute, ctx->tokens, KV, H, "attn k recompute");
    gemm_fwd(b, v_out, normed, ctx->w.v_proj.compute, ctx->tokens, KV, H, "attn v recompute");
    if (gated) {
        kernel_q_gate_split(q, gate, q_raw, ctx->tokens * Q, hd, b->stream);
        check_launch("attn q/gate split recompute");
    }
    /* The per-head norm's backward needs the *pre-norm* q and k, and the forward
     * overwrites them in place, so they are widened before the norm runs. */
    kernel_cast_bf16_f32(q_pre_f32, q, (int)(T * Q), b->stream);
    kernel_cast_bf16_f32(k_pre_f32, k_out, (int)(T * KV), b->stream);
    check_launch("attn pre-norm snapshots");
    if (ctx->w.q_norm.compute != nullptr) {
        layer_norm_forward(b->stream, q, q, ctx->w.q_norm.compute, hd, ctx->tokens * nH, dims,
                           "attn q per-head norm recompute");
        layer_norm_forward(b->stream, k_out, k_out, ctx->w.k_norm.compute, hd, ctx->tokens * nKV,
                           dims, "attn k per-head norm recompute");
    }
    kernel_flashinfer_rope(q, k_out, ctx->positions, ctx->tokens, nH, nKV, hd, dims->rotary_dim,
                           dims->rope_theta, b->stream);
    check_launch("attn rope recompute");
    const int seq_start = (int)ctx->seq_len - ctx->tokens;
    kernel_kv_cache_write(ctx->kv_cache, k_out, v_out, seq_start, ctx->tokens, nKV, hd,
                          dims->max_seq_len, b->stream);
    check_launch("attn kv write recompute");
    kernel_attention_lse(attn_out, lse, q, ctx->kv_cache, seq_start, ctx->tokens, (int)ctx->seq_len,
                         nH, nKV, hd, 1.0f / sqrtf((float)hd), dims->max_seq_len, b->stream);
    check_launch("attn core recompute");
    if (gated) {
        kernel_sigmoid_mul(gated_out, attn_out, gate, Q, ctx->tokens, Q, 0, b->stream);
        check_launch("attn output-gate recompute");
    }

    /* ---- reverse ---- */
    const int gemma = dims->norm_style == 0 ? 1 : 0;
    /* 1. the output projection: out = gated . W_o^T. */
    {
        float *w_o = widen_weight(b, &ctx->w.o_proj, (size_t)H * Q, "o weight widen");
        float *gated_f32 = widen(b, gated ? gated_out : attn_out, T * Q, "attn gated widen");
        if (gemm_backward_dx(b->cublas, d_gated, d_out, w_o, ctx->tokens, H, Q, 0) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dx failed in o_proj");
        }
        if (gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.o_proj, (size_t)H * Q, "o grad"),
                             gated_f32, d_out, ctx->tokens, H, Q, 1) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dw failed in o_proj");
        }
    }
    /* 2. the output gate: out = attn * sigmoid(gate_bf16). */
    if (gated) {
        kernel_sigmoid_mul_backward(d_attn, d_q_raw, d_gated, attn_out, gate, Q, ctx->tokens, Q, 0,
                                    0, b->stream);
    } else {
        check_cuda(cudaMemcpyAsync(d_attn, d_gated, T * Q * sizeof(float),
                                   cudaMemcpyDeviceToDevice, b->stream),
                   "attn d_attn copy");
    }
    check_launch("attn output-gate backward");
    /* 3. the attention core, from the recomputed base-2 LSE. The probabilities are
     * rounded to BF16 (round_probabilities=1) because that is what the forward's PV
     * product does: the pairing is with the forward that ran. */
    {
        if (ctx->seq_len != ctx->tokens) {
            throw std::runtime_error(
                "backward_layers: the training step's attention is a full sequence "
                "(seq_len == tokens); a cached prefix is not wired yet");
        }
        const float scale = 1.0f / sqrtf((float)hd);
        check_cuda(cudaMemsetAsync(d_rk, 0, T * KV * sizeof(float), b->stream), "attn dk zero");
        check_cuda(cudaMemsetAsync(d_v, 0, T * KV * sizeof(float), b->stream), "attn dv zero");
        kernel_attention_backward(d_rq, d_rk, d_v, d_attn, q, ctx->kv_cache, lse, ctx->tokens,
                                  (int)ctx->seq_len, dims->max_seq_len, nH, nKV, hd, scale, 1, 0,
                                  b->stream);
        check_launch("attention backward");
    }
    /* 4. the transposed rotation. */
    kernel_rope_backward(d_q_norm, d_k_norm, d_rq, d_rk, ctx->positions, ctx->tokens, nH, nKV, hd,
                         dims->rotary_dim, dims->rope_theta, 0, b->stream);
    check_launch("rope backward");
    /* 5. the per-head norms. */
    if (ctx->w.q_norm.compute != nullptr) {
        kernel_rms_inv(inv_q, q_pre_f32, hd, ctx->tokens * nH, dims->rms_eps, b->stream);
        kernel_rms_inv(inv_k, k_pre_f32, hd, ctx->tokens * nKV, dims->rms_eps, b->stream);
        check_launch("attn per-head inv");
        norm_backward(b, d_q_pre, &ctx->w.q_norm, d_q_norm, q_pre_f32, inv_q, hd, ctx->tokens * nH,
                      0, dims, "attn q norm backward");
        norm_backward(b, d_k_pre, &ctx->w.k_norm, d_k_norm, k_pre_f32, inv_k, hd, ctx->tokens * nKV,
                      0, dims, "attn k norm backward");
    } else {
        check_cuda(cudaMemcpyAsync(d_q_pre, d_q_norm, T * Q * sizeof(float),
                                   cudaMemcpyDeviceToDevice, b->stream),
                   "attn d_q_pre copy");
        check_cuda(cudaMemcpyAsync(d_k_pre, d_k_norm, T * KV * sizeof(float),
                                   cudaMemcpyDeviceToDevice, b->stream),
                   "attn d_k_pre copy");
    }
    /* 6. the fused projection's re-interleave (or the plain query path). */
    if (gated) {
        /* The gate's gradient (in d_q_raw's second half, from step 2) and the query's
         * are interleaved back into the projection's layout. */
        kernel_qgate_merge_backward(d_q_raw, d_q_pre, d_q_raw, ctx->tokens * Q, hd, b->stream);
        check_launch("q/gate merge backward");
    } else {
        check_cuda(cudaMemcpyAsync(d_q_raw, d_q_pre, T * Q * sizeof(float),
                                   cudaMemcpyDeviceToDevice, b->stream),
                   "attn d_q_raw copy");
    }
    /* 7. the QKV projections: one input (normed), three consumers. */
    {
        const int q_out = gated ? 2 * Q : Q;
        float *w_q = widen_weight(b, &ctx->w.q_proj, (size_t)q_out * H, "q weight widen");
        float *w_k = widen_weight(b, &ctx->w.k_proj, (size_t)KV * H, "k weight widen");
        float *w_v = widen_weight(b, &ctx->w.v_proj, (size_t)KV * H, "v weight widen");
        if (gemm_backward_dx(b->cublas, d_normed, d_q_raw, w_q, ctx->tokens, q_out, H, 0) != 0 ||
            gemm_backward_dx(b->cublas, d_normed, d_k_pre, w_k, ctx->tokens, KV, H, 1) != 0 ||
            gemm_backward_dx(b->cublas, d_normed, d_v, w_v, ctx->tokens, KV, H, 1) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dx failed in qkv");
        }
        float *x_f32 = widen(b, normed, TH, "attn normed widen");
        if (gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.q_proj, (size_t)q_out * H, "q grad"),
                             x_f32, d_q_raw, ctx->tokens, q_out, H, 1) != 0 ||
            gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.k_proj, (size_t)KV * H, "k grad"),
                             x_f32, d_k_pre, ctx->tokens, KV, H, 1) != 0 ||
            gemm_backward_dw(b->cublas, weight_grad(b, &ctx->w.v_proj, (size_t)KV * H, "v grad"),
                             x_f32, d_v, ctx->tokens, KV, H, 1) != 0) {
            throw std::runtime_error("backward_layers: gemm_backward_dw failed in qkv");
        }
    }
    /* 8. the input norm. */
    kernel_rms_inv(inv_h, residual_f32, H, ctx->tokens, dims->rms_eps, b->stream);
    check_launch("attn inv hidden");
    norm_backward(b, d_res, &ctx->w.input_norm, d_normed, residual_f32, inv_h, H, ctx->tokens, 1,
                  dims, "attn input norm backward");
}

}  // namespace

/* ------------------------------------------------------------------ */
/* The per-row inverse RMS                                            */
/* ------------------------------------------------------------------ */

namespace {

/* One block per row: mean of squares, then rsqrt. The forward's library norms do not
 * return this, and the chain through the reciprocal square root is analytic but needs
 * the *value* the forward used; recomputing it from the same FP32 widening with the
 * same formula is what keeps the two consistent. */
__global__ void rms_inv_kernel(float *__restrict__ out, const float *__restrict__ x, int cols,
                               float eps) {
    const int row = blockIdx.x;
    const float *row_x = x + (size_t)row * cols;
    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local += row_x[c] * row_x[c];
    __shared__ float shared[256];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) shared[threadIdx.x] += shared[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = rsqrtf(shared[0] / (float)cols + eps);
}

}  // namespace

void kernel_rms_inv(float *out, const float *x, int cols, int rows, float eps,
                    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || out == nullptr || x == nullptr) {
        throw std::runtime_error("kernel_rms_inv: invalid shape or null buffer");
    }
    rms_inv_kernel<<<rows, 256, 0, stream>>>(out, x, cols, eps);
    check_launch("kernel_rms_inv");
}

/* ------------------------------------------------------------------ */
/* What one layer's backward needs                                    */
/* ------------------------------------------------------------------ */

void layer_backward_scratch(const ModelDims *dims, int tokens, int mixer_kind, int ffn_kind,
                            struct LayerBackwardScratch *out) {
    if (dims == nullptr || out == nullptr || tokens <= 0) {
        throw std::runtime_error("layer_backward_scratch: invalid arguments");
    }
    const size_t T = (size_t)tokens;
    const int H = dims->hidden_size;
    const int I = dims->intermediate_size;
    const int Q = dims->num_heads * dims->head_dim;
    const int KV = dims->num_kv_heads * dims->head_dim;
    const size_t TH = T * H;

    size_t ws_forward = 0;
    size_t cast = 0;
    size_t grad = 0;
    if (mixer_kind == ENGINE_MIXER_FULL_ATTN) {
        /* The forward's own layout, reserving the gated sizes even when the model does
         * not use the gate: one workspace per layer serves every layer. */
        ws_forward = std::max(ws_forward, (size_t)TH + T * 2 * Q + 2 * T * KV + 3 * T * Q);
        /* The widenings: the gated activation, the four weights, the normed input and
         * the two per-head weights, the input norm's weight, and one discard slot for a
         * frozen role's weight gradient. */
        const size_t discard = (size_t)2 * Q * H;
        cast = std::max(cast, (size_t)T * Q + (size_t)H * Q + 2 * (size_t)Q * H +
                                   2 * (size_t)KV * H + TH + 2 * dims->head_dim + H + discard);
        /* The gradients: the LSE and three inverse-RMS rows, the six query-sized
         * gradients (the fused projection's raw gradient is twice a query), the four
         * KV-sized ones, the two pre-norm snapshots, and the normed input's. */
        grad = std::max(grad, (size_t)(2 * T * dims->num_heads + T * dims->num_kv_heads + T) +
                                  8 * T * Q + 5 * T * KV + TH);
    }
    if (ffn_kind == ENGINE_FFN_DENSE) {
        ws_forward = std::max(ws_forward, (size_t)TH + 3 * T * I);
        cast = std::max(cast, (size_t)TH + T * I + 3 * (size_t)I * H + H + (size_t)I * H);
        grad = std::max(grad, (size_t)T + 3 * T * I + TH);
    }
    /* The layer walk itself holds three hidden-state-sized buffers (the post-mixer
     * residual's gradient, the mixer's output gradient and the incoming one). */
    const size_t layer_walk = 3 * TH;
    if (out != nullptr) {
        out->workspace_elements = ws_forward + TH; /* + the narrowed residual's tail */
        /* A quarter more than the count above, so the pools refuse only a genuinely
         * wrong size rather than tracking every buffer by hand: a pool that is exactly
         * the sum of its takes would turn any future edit into a failure at run time. */
        out->cast_elements = (long long)(cast + cast / 4 + 16);
        out->grad_elements = (long long)(grad + layer_walk + (grad + layer_walk) / 4 + 16);
    }
}

/* ------------------------------------------------------------------ */
/* The layer walk                                                     */
/* ------------------------------------------------------------------ */

void backward_layer(const struct LayerBackwardCtx *ctx, const float *d_out, float *d_residual_io) {
    if (ctx == nullptr || d_out == nullptr || d_residual_io == nullptr) {
        throw std::runtime_error("backward_layer: null context or gradient");
    }
    if (ctx->mixer != ENGINE_MIXER_FULL_ATTN) {
        throw std::runtime_error(
            "backward_layer: only the full-attention mixer is wired in this stage");
    }
    if (ctx->ffn != ENGINE_FFN_DENSE) {
        throw std::runtime_error("backward_layer: only the dense feed-forward is wired");
    }
    with_stream(ctx->cublas, ctx->stream);
    const size_t TH = (size_t)ctx->tokens * ctx->dims->hidden_size;

    /* r2 = r1 + ffn(r1): the layer's output is the post-ffn residual, so the incoming
     * gradient *is* the post-mixer residual's gradient before any sublayer internal
     * backward adds to it. Seeding it here is what makes `d_residual_io` an accumulator
     * a caller can hand in uninitialised; a version that only accumulated the sublayer
     * terms into whatever the buffer held produced a growing loss, which is how this
     * was found. */
    check_cuda(cudaMemcpyAsync(d_residual_io, d_out, TH * sizeof(float),
                               cudaMemcpyDeviceToDevice, ctx->stream),
               "layer seed the residual gradient");

    /* The layer's own pool region: the post-mixer residual's gradient, then the
     * sublayers' scratch. */
    Pool layer_grad{ctx->grad_f32, ctx->grad_f32_elements, 0};
    float *d_mid = layer_grad.take(TH, "layer d_mid");
    Buffers b{ctx->cublas, ctx->stream, ctx->workspace, ctx->workspace + ctx->ws_backward_offset,
              {ctx->cast_f32, ctx->cast_f32_elements, 0},
              {ctx->grad_f32 + TH, ctx->grad_f32_elements > TH ? ctx->grad_f32_elements - TH : 0,
               0}};

    /* r2 = r1 + ffn(r1): the identity branch puts d_out into the post-mixer residual's
     * gradient, and the feed-forward's internal backward adds its norm's contribution
     * to the same tensor. `d_mid` is a separate buffer from `d_out` so the feed-forward
     * can read its output gradient while accumulating into the residual's. */
    check_cuda(cudaMemcpyAsync(d_mid, d_out, TH * sizeof(float), cudaMemcpyDeviceToDevice,
                               ctx->stream),
               "layer d_mid copy");
    /* The retentions are FP32; the mixer/ffn recomputes need the FP32 residual for the
     * norms and narrow it themselves. For the feed-forward the residual input is the
     * post-mixer stream, which is `residual_in + mixer_out` reconstructed from the
     * retained boundaries below. */
    float *post_mixer = b.grad.take(TH, "layer post-mixer residual");
    kernel_f32_add(post_mixer, ctx->residual_in, ctx->mixer_out, (long long)TH, ctx->stream);
    check_launch("layer post-mixer residual");
    backward_mlp(ctx, &b, post_mixer, d_mid, d_residual_io);

    float *d_mixer_out = b.grad.take(TH, "layer d_mixer_out");
    check_cuda(cudaMemcpyAsync(d_mixer_out, d_residual_io, TH * sizeof(float),
                               cudaMemcpyDeviceToDevice, ctx->stream),
               "layer d_mixer_out copy");
    backward_attention(ctx, &b, ctx->residual_in, d_mixer_out, d_residual_io);
}
