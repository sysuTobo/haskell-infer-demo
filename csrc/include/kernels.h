/**
 * kernels.h - Native CUDA kernel declarations still owned by this project.
 *
 * Convention: void kernel_<name>(<params>, cudaStream_t stream); BF16 unless
 * noted. Library-backed operators live in flashinfer_ops.h (norm/RoPE/
 * attention/SiLU) and fla_ops.h (GatedDeltaNet); cuBLAS GEMM in layers.h.
 */

#ifndef HASKELL_INFER_KERNELS_H
#define HASKELL_INFER_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Attention: KV cache + output gate                                 */
/* ------------------------------------------------------------------ */

/**
 * Write new K/V into the KV cache at [seq_start, seq_start+tokens).
 * kv_cache layout: [2, max_seq, num_kv_heads, head_dim] BF16.
 */
void kernel_kv_cache_write(__nv_bfloat16 *kv_cache,
                           const __nv_bfloat16 *k_new,
                           const __nv_bfloat16 *v_new,
                           int seq_start, int tokens,
                           int num_kv_heads, int head_dim,
                           int max_seq_len, cudaStream_t stream);

/**
 * Causal GQA attention over the KV cache (FlashInfer-backed).
 * Requires head_dim=256 and seq_len == seq_start + tokens.
 */
void kernel_attention(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                      const __nv_bfloat16 *kv_cache,
                      int seq_start, int tokens, int seq_len,
                      int num_heads, int num_kv_heads, int head_dim,
                      float scale, int max_seq_len, cudaStream_t stream);

/**
 * Split the fused Q / output-gate projection. `total` counts Q *elements*
 * (rows * head_dim, where a row is a (token, head) pair), matching the real
 * caller's `tokens * num_heads * head_dim`; raw holds 2 * total elements with Q
 * and the gate interleaved per row, so q[i] = raw[row*2*head_dim + d] and
 * gate[i] = raw[row*2*head_dim + head_dim + d] for i = row*head_dim + d. A
 * per-token caller passes the token's slice of raw and the token-local element
 * count. Exported so the region harness can drive it without a whole attention
 * layer.
 */
void kernel_q_gate_split(__nv_bfloat16 *q, __nv_bfloat16 *gate,
                         const __nv_bfloat16 *raw, int total, int head_dim,
                         cudaStream_t stream);

/**
 * Gated attention output: out = attn * sigmoid(gate), BF16-rounded sigmoid.
 */
void kernel_sigmoid_mul(__nv_bfloat16 *out, const __nv_bfloat16 *attn,
                        const __nv_bfloat16 *gate, int dim, int tokens,
                        int gate_stride, int gate_offset, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: Causal Conv1d (causal-conv1d backed)                         */
/* ------------------------------------------------------------------ */

/**
 * Causal 1D convolution, width 4, conv_state updated in place.
 * Returns unfused BF16 output; caller applies SiLU separately.
 */
void kernel_causal_conv1d(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                          const __nv_bfloat16 *weight,
                          const __nv_bfloat16 *bias,
                          __nv_bfloat16 *conv_state,
                          int conv_dim, int tokens, int kernel_size,
                          cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  GDN: RMSNormGated                                                 */
/* ------------------------------------------------------------------ */

/**
 * Gated RMS norm matching Transformers dtype boundaries:
 * norm(FP32)->BF16, *weight(BF16)->BF16, *SiLU(z)(FP32)->BF16.
 * weight is the effective per-head FP32 value (no +1).
 */
void kernel_gdn_gated_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                           const __nv_bfloat16 *z, const float *weight,
                           int dim, int tokens, float eps,
                           cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  MLP: SiLU gate (FlashInfer-backed)                                */
/* ------------------------------------------------------------------ */

/**
 * Fused SiLU(gate) * up. Requires up == gate + n (contiguous [gate, up]).
 */
void kernel_silu_mul(__nv_bfloat16 *out, const __nv_bfloat16 *gate,
                     const __nv_bfloat16 *up, int n, cudaStream_t stream);

/**
 * Elementwise SiLU in place. This is the GDN conv activation (a separate region
 * from the dense MLP's fused gate*up) and is exported so the region harness can
 * drive it independently of the layer that happens to call it.
 */
void kernel_silu_inplace(__nv_bfloat16 *x, int n, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Embedding                                                         */
/* ------------------------------------------------------------------ */

/** Token embedding lookup: out[i] = table[token_ids[i]]. */
void kernel_embedding(__nv_bfloat16 *out, const __nv_bfloat16 *table,
                      const int64_t *token_ids, int hidden_size, int tokens,
                      cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Small helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * Natural-log log-softmax of one row plus the label's log-probability:
 * out[row] = logits[row*label] - logsumexp(logits[row, :]), computed in FP32 on the
 * device so a teacher-forced loss never materialises a [rows, vocab] tensor.
 */
void kernel_logprob_gather(float *out, const float *logits, const int *labels, int rows,
                           int vocab, cudaStream_t stream);

/** BF16 to F32 cast. */
void kernel_cast_bf16_f32(float *out, const __nv_bfloat16 *in, int n,
                          cudaStream_t stream);

/** F32 to BF16 cast: the single rounding that turns a merged FP32 partial into
 *  the activation it belongs to. */
void kernel_cast_f32_bf16(__nv_bfloat16 *out, const float *in, int n,
                          cudaStream_t stream);

/** Residual add: dst[i] += src[i] (BF16, element-wise). */
void kernel_residual_add(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                         int n, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Backward (plan Stage 4)                                           */
/* ------------------------------------------------------------------ */

/* Gradient buffers are FP32 everywhere: the plan's convention is FP32 accumulation,
 * and every entry point below writes the gradient of its forward *region*, not of a
 * layer. `accumulate` selects between assigning (0, the caller owns a fresh buffer)
 * and adding (1, the branch-reduction case, where two consumers write one gradient).
 * A reversed (transposed) operation is the same math with the operands swapped, which
 * is why the RoPE and Q/gate entries are separate from their forwards rather than
 * sharing a "mode" flag. */

/* out = attn * sigmoid(gate); d_attn = d_out * sigmoid_bf16, d_gate = d_out*attn*s*(1-s)
 * with s the BF16-rounded sigmoid the forward actually multiplied by. */
void kernel_sigmoid_mul_backward(float *d_attn, float *d_gate, const float *d_out,
                                 const __nv_bfloat16 *attn, const __nv_bfloat16 *gate,
                                 int dim, int tokens, int gate_stride, int gate_offset,
                                 int accumulate, cudaStream_t stream);

/* out = silu(gate) * up. The forward's operands are a contiguous [gate[n], up[n]] pair
 * in one buffer; the backward reads both and writes two separate gradient halves. */
void kernel_silu_mul_backward(float *d_gate, float *d_up, const float *d_out,
                              const __nv_bfloat16 *gate, const __nv_bfloat16 *up, int n,
                              int accumulate, cudaStream_t stream);

/* In-place SiLU: d_x = d_out * silu'(pre_activation), over `n` elements. */
void kernel_silu_inplace_backward(float *d_x, const float *d_out, const float *pre_activation,
                                  int n, int accumulate, cudaStream_t stream);

/* y = a + b: both branches receive d_out. */
void kernel_branch_backward(float *d_a, float *d_b, const float *d_out, long long n,
                            int accumulate, cudaStream_t stream);

/* y_j = x_j * inv_rms * (w_j or w_j + 1), one BF16 rounding at the end.
 *   d_x_j = inv_rms*w'_j*d_out_j - (inv_rms^3/cols) * x_j * sum_k(w'_k x_k d_out_k)
 *   d_w_j = d_out_j * inv_rms * x_j
 * `inv_rms` is the saved statistic (Stage 1: "x and the inverse RMS"): the chain
 * through the reciprocal square root is analytic, but its *value* has to be the one
 * the forward rounded with, or the pairing is a different function. */
void kernel_rmsnorm_backward(float *d_x, float *d_weight, const float *d_out, const float *x,
                             const float *raw_weight, const float *inv_rms, int cols, int rows,
                             int gemma, int accumulate, cudaStream_t stream);

/* GDN Q/K L2 normalisation: y = x * rsqrt(sum_j x_j^2 + eps), no mean, eps inside.
 *   d_x_j = r*d_out_j - r^3 * x_j * sum_k(d_out_k x_k),  r = the saved rsqrt value. */
void kernel_l2norm_backward(float *d_x, const float *d_out, const float *x, const float *inv_norm,
                            int cols, int rows, int accumulate, cudaStream_t stream);

/* GDN RMSNormGated, mirroring the forward's three BF16 boundaries:
 *   s = rsqrt(mean(x^2)+eps);  n = bf16(x*s);  y = bf16(bf16(n*w) * swish(z))
 * The backward passes the gradient through the casts (the convention) and consumes
 * the rounded values, so d_w uses the rounded n and d_z the exact swish derivative. */
void kernel_gdn_gated_norm_backward(float *d_x, float *d_weight, float *d_z, const float *d_out,
                                    const float *x, const float *z, const float *weight,
                                    const float *inv_rms, int dim, int rows, int accumulate,
                                    cudaStream_t stream);

/* Embedding gather backward: the table's gradient, accumulated with atomics so a
 * token id that appears more than once sums rather than overwrites (the plan's
 * "repeated embedding IDs"). d_table is FP32 [vocab, hidden] and is *added* to. */
void kernel_embedding_backward(float *d_table, const float *d_out, const int64_t *token_ids,
                               int hidden, int tokens, cudaStream_t stream);

/* Split-half RoPE backward: the transposed rotation, with the same tables the forward
 * used, recomputed from the positions. */
void kernel_rope_backward(float *d_q, float *d_k, const float *d_out_q, const float *d_out_k,
                          const int64_t *positions, int tokens, int q_heads, int kv_heads,
                          int head_dim, int rotary_dim, float theta, int accumulate,
                          cudaStream_t stream);

/* The inverse of kernel_q_gate_split: scatter the two gradients back into the
 * interleaved layout of the fused projection's output. */
void kernel_qgate_merge_backward(float *d_raw, const float *d_q, const float *d_gate,
                                 int total, int head_dim, cudaStream_t stream);

/* out[M,N] = x[M,K] @ W[N,K]^T with BF16 operands and FP32 accumulation (the same
 * accumulate type the forward used).
 *
 * The gradient operands are **FP32**: d_out is a gradient already, and the weight and
 * activation the other product needs are the FP32 widenings of the forward's BF16
 * values, which the caller produces with kernel_cast_bf16_f32. cuBLAS does not accept
 * mixed A/B operand types, so widening once is also the only way to get one call per
 * product; and it is the honest contract, because a BF16-by-BF16 product inside the
 * backward would be a third rounding that the forward's own MMA does not perform.
 * Pairing dX/dW with the forward's BF16 MMA is therefore a Stage-6 alignment question,
 * recorded in csrc/backward.c. */
int gemm_backward_dx(cublasHandle_t handle, float *d_x, const float *d_out, const float *W_fp32,
                     int M, int N, int K);
int gemm_backward_dw(cublasHandle_t handle, float *d_w, const float *x_fp32, const float *d_out,
                     int M, int N, int K);

/* Log-softmax + gather backward for one row at a time, the fused counterpart of
 * kernel_logprob_gather: d_logits[row] = d_out[row] * (softmax - onehot(label)),
 * zeroed when the row's mask byte is 0. */
void kernel_logprob_gather_backward(float *d_logits, const float *d_out, const float *logits,
                                    const int *labels, const uint8_t *mask, int rows, int vocab,
                                    cudaStream_t stream);

/* AdamW on the device, the same order backward.c documents. `bf16_out` may be null. */
void kernel_adamw(float *master, const float *grad, float *m_slot, float *v_slot, long long n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, int step,
                  __nv_bfloat16 *bf16_out, cudaStream_t stream);

/* Causal conv1d backward (width `kernel_size`, the state oldest-first):
 *   out[t,c] = bias[c] + sum_j w[c,j] * xv_j,  xv_j = x[t-(k-1)+j]  (state when < 0).
 * Writes d_x, the weight and bias gradients (both summed over tokens) and
 * `d_state_in`, the gradient of the *incoming* state -- which a full-sequence backward
 * needs even though the forward overwrote it. `x_in` is the conv's BF16 input, which
 * the weight gradient has to read; each (channel, position) has exactly one owner, so
 * accumulation is a plain add rather than an atomic. */
void kernel_causal_conv1d_backward(float *d_x, float *d_weight, float *d_bias,
                                   float *d_state_in, const float *d_out,
                                   const __nv_bfloat16 *x_in, const __nv_bfloat16 *weight,
                                   const __nv_bfloat16 *conv_state_in, int conv_dim, int tokens,
                                   int kernel_size, int accumulate, cudaStream_t stream);

/* GDN prepare backward. The forward turns the conv output into the core's inputs:
 *   conv_out[t, :K*D] and [t, K*D:2*K*D] -> q,k (L2-normalised per key head, then
 *   BF16-rounded and expanded to the value heads); conv_out[t, 2*K*D:] -> v (a copy)
 *   a,b [tokens, value_heads] and A_log,dt_bias [value_heads] -> log_decay and beta,
 *   with beta stored as the FP32 value of the BF16-rounded sigmoid.
 * The upstream gradients of the prepared (head-expanded) q/k/v, of the log-decay and
 * of `beta` come in; the gradients of conv_out (reduced over the duplicated key
 * heads), a, b, A_log and dt_bias go out. */
void kernel_gdn_prepare_backward(float *d_conv_out, float *d_a, float *d_b, float *d_A_log,
                                 float *d_dt_bias, const float *d_q, const float *d_k,
                                 const float *d_v, const float *d_log_decay, const float *d_beta,
                                 const __nv_bfloat16 *conv_out, const __nv_bfloat16 *a,
                                 const __nv_bfloat16 *b, const __nv_bfloat16 *A_log,
                                 const __nv_bfloat16 *dt_bias, int tokens, int key_heads,
                                 int value_heads, int head_dim, int accumulate,
                                 cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Backward: the two paired regions (plan Stage 4)                   */
/* ------------------------------------------------------------------ */

/* Causal GQA attention with the base-2 LSE exported, bitwise identical in `out` to
 * kernel_attention (Stage 2 measured that asking for the LSE does not perturb the
 * output). `lse` is [tokens, num_heads] FP32, row-major, and holds log2(sum_j e^{s_j})
 * over the causal scores. */
void kernel_attention_lse(__nv_bfloat16 *out, float *lse, const __nv_bfloat16 *q,
                          const __nv_bfloat16 *kv_cache, int seq_start, int tokens, int seq_len,
                          int num_heads, int num_kv_heads, int head_dim, float scale,
                          int max_seq_len, cudaStream_t stream);

/* The paired attention backward, from the saved (q, k/v cache, base-2 LSE) with P
 * *recomputed* rather than retained -- the resource trade Stage 2 asked for.
 *
 *   P_ts = exp2(s_ts * log2(e) - L_t)     (the kernel's own probability form)
 *   dV_s = sum_t P_ts * dO_t
 *   dP_ts = dO_t . V_s ;  row_t = sum_s dP_ts * P_ts
 *   dS_ts = P_ts * (dP_ts - row_t) * ln 2         (the ln-2 the base-2 LSE needs)
 *   dQ_t = scale * sum_s dS_ts * K_s ;  dK_s = scale * sum_t dS_ts * Q_t
 *
 * `round_probabilities_to_bf16` mirrors the kernel's PV MMA, which rounds P to BF16
 * before the product; whether a trainer wants that rounding is a Stage-6 pairing
 * decision, so the entry point makes it a parameter rather than a hidden assumption.
 * dk/dv are [tokens, num_kv_heads, head_dim] FP32: the gradient of the K/V *values
 * the forward wrote, which sum over the query tokens and over the GQA group. */
void kernel_attention_backward(float *d_q, float *d_k, float *d_v, const float *d_out,
                               const __nv_bfloat16 *q, const __nv_bfloat16 *kv_cache,
                               const float *lse, int tokens, int seq_len, int max_seq_len,
                               int num_heads, int num_kv_heads, int head_dim, float scale,
                               int round_probabilities_to_bf16, int accumulate,
                               cudaStream_t stream);

/* GDN core backward: the reverse of the decay-before-prediction recurrence, in FP32,
 * started from the retained chunk-boundary states so the gradient crosses every
 * internal boundary (the plan's full-sequence BPTT).
 *
 *   alpha_t = exp(g_t) ; D_t = alpha_t * S_{t-1} ; pred_t = k_t D_t
 *   delta_t = v_t - pred_t ; S_t = D_t + outer(k_t, beta_t * delta_t)
 *   o_t = scale * q_t S_t
 *
 * Every per-head state is [key_dim, value_dim] FP32. `chunk_state` holds the state at
 * the start of each chunk ([n_chunks, value_heads, key_dim, value_dim]), and the
 * per-chunk kernel *recomputes* the states inside the chunk from that retained value
 * rather than keeping a [tokens, ...] tensor. `d_state_in` is the upstream gradient of
 * the *final* state (may be null for a zero gradient); `d_state_start` receives the
 * gradient of the *initial* state, which is what a previous step's state needs.
 *
 * The gradient is of the mathematical recurrence. The chunkwise cubin this model runs
 * decomposes the same recurrence with (I + A)^{-1} and BF16 intermediate MMAs, so the
 * two agree to that rounding and no closer; pairing them is Stage 6's alignment work,
 * and the header of csrc/backward.c records that split.
 *
 * `q`/`k` are [tokens, value_heads, key_dim] and `v` [tokens, value_heads, value_dim]
 * FP32 (head-expanded), `log_decay`/`beta` [tokens, value_heads] FP32, `chunk_state`
 * [n_chunks, value_heads, key_dim, value_dim] FP32. `workspace` must be at least
 * kernel_gdn_core_workspace_bytes(...); `d_log_decay`, `d_beta`, `d_q`, `d_k` are
 * accumulated into, and `d_v` is written (accumulated only when `accumulate` is set). */
size_t kernel_gdn_core_workspace_bytes(int tokens, int value_heads, int key_dim, int value_dim,
                                       int chunk);
void kernel_gdn_core_backward(float *d_q, float *d_k, float *d_v, float *d_log_decay,
                              float *d_beta, float *d_state_start, const float *d_out,
                              const float *d_state_in, const float *q, const float *k,
                              const float *v, const float *log_decay, const float *beta,
                              const float *chunk_state, int tokens, int value_heads,
                              int key_heads, int key_dim, int value_dim, int chunk, float scale,
                              int accumulate, void *workspace, size_t workspace_bytes,
                              cudaStream_t stream);

#endif /* HASKELL_INFER_KERNELS_H */
