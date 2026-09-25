/**
 * regions.c - The Stage-1 region inventory of the dense/dense-hybrid forward
 * path. See csrc/include/regions.h for the contract and
 * docs/plan-numeric-contract.md for the stage.
 *
 * Two rules are worth stating because they are what a reviewer should check:
 *
 *   - `exact` is claimed only for a region the manifest registry already calls
 *     `deterministic` -- a single elementwise pass, a row gather or a
 *     permutation, with no cross-thread reduction and no library tiling
 *     decision. Nothing whose reduction order comes from a library (FlashInfer,
 *     cuBLAS, the AOT FLA cubins) is called exact here, even where a measurement
 *     would show 0; the manifest's determinism column is the conservative
 *     source and the plan's Stage 2 is what would establish the rest.
 *   - an unmeasured number is NaN and the reason says the pair is unverified.
 *     Nothing is defaulted to "close enough".
 */
#include "regions.h"

#include <math.h>
#include <string.h>

/* An unmeasured pair, or a pair whose verdict needs no numbers. max_abs and rms
 * are NaN: the CPU gate refuses a finite number under any verdict other than a
 * measured `exception`, so a number here has to be paid for by an experiment. */
#define PAIR(l, r, v, a, s, st, why) {l, r, v, a, s, NAN, NAN, st, why}

/* An established bitwise pair. The claim is stated numerically (0.0, not "not
 * measured") because it is the one verdict the device harness re-derives. */
#define PAIR_EXACT(l, r, st, why) \
    {l, r, REGION_VERDICT_EXACT, "-", "-", 0.0, 0.0, st, why}

/* A measured, quantified deviation: the worst the Stage-2 experiment saw for this
 * pair over the tested architecture and shapes, rounded *up* to two significant
 * digits so the bound has a margin over the observation it came from (a bound that
 * the measurement reproduces exactly is a bad bound: the last digit trips it).
 * max_abs/rms are absolute in the region's own output units and are conservative
 * -- for a region with persistent state they are the worse of the output and the
 * state comparison, over both the Stage-2 matrix and the Stage-1 fixture. */
#define PAIR_EXCEPTION(l, r, arch, shapes, mx, rm, st, why) \
    {l, r, REGION_VERDICT_EXCEPTION, arch, shapes, mx, rm, st, why}

/* ------------------------------------------------------------------ */
/* Case-pair tables                                                   */
/* ------------------------------------------------------------------ */

static const struct RegionCasePair kEmbeddingPairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, "-",
               "one thread per output element with no reduction, so neither the "
               "chunk length nor the query count can change a copied row"),
};

static const struct RegionCasePair kRmsnormPairs[] = {
    PAIR(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, REGION_VERDICT_UNVERIFIED,
         "-", "-", "-",
         "the row width is case-independent but FlashInfer's reduction tree over "
         "it is not established (Stage 2 claim C covers attention, not the norm)"),
};

static const struct RegionCasePair kPerHeadNormPairs[] = {
    PAIR(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, REGION_VERDICT_UNVERIFIED,
         "-", "-", "-",
         "per-head rows are normalized independently, but the library's "
         "in-row reduction order is not established"),
};

static const struct RegionCasePair kRopePairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, "-",
               "per-element rotation with no cross-thread reduction; the case "
               "only changes which positions are passed in"),
};

static const struct RegionCasePair kQGateSplitPairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, "-",
               "an elementwise permutation of the fused projection output; the "
               "token count does not enter the mapping"),
};

static const struct RegionCasePair mKvWritePairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE,
               "the whole KV cache is compared elementwise",
               "a contiguous copy at a caller-given offset; the only "
               "case-dependent input is that offset"),
};

static const struct RegionCasePair kAttentionCorePairs[] = {
    PAIR_EXCEPTION(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE,
                   "sm_86 (2x A40)",
                   "head_dim 128/256; GQA 24x4, 24x8, 8x4; kv_len 1..269",
                   1.0e-03, 1.0e-04,
                   "the KV cache is compared elementwise and is unchanged by any arm",
                   "the query-block tiling changes the softmax reduction: one call "
                   "with all L queries versus one call per query. 202 of 264 measured "
                   "tilings are bitwise identical and the worst is 2.0e-3 relative "
                   "(about half a BF16 ULP), at head_dim 128 / 24x4 / L=63. Stage 2 "
                   "claim C"),
    PAIR_EXCEPTION(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_TAIL1,
                   "sm_86 (2x A40)",
                   "head_dim 128/256; GQA 24x4, 24x8, 8x4; kv_len 1..269; split boundaries 1, L/2, 64, 128",
                   1.0e-03, 4.2e-05,
                   "the KV cache is compared elementwise and is unchanged by any arm",
                   "a prefix+suffix prefill against the single-call prefill over the "
                   "same cache: worst 2.0e-3 relative, and the deviations cluster at "
                   "the boundaries that do not align with the query tile. Stage 2 "
                   "claim C"),
};

static const struct RegionCasePair kAttentionOutputGatePairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, "-",
               "elementwise multiply by a BF16-rounded sigmoid; the gate stride "
               "and offset are the same layout at any token count"),
};

static const struct RegionCasePair kGemmBf16Pairs[] = {
    PAIR_EXCEPTION(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE,
                   "sm_86 (2x A40)",
                   "real projections N x K: 1024..248320 x 5120, 5120 x 6144, "
                   "5120 x 17408; M 1..434; per-row and prefix/suffix arms",
                   1.3e-01, 1.2e-02,
                   "no persistent state: the output matrix is the comparison",
                   "cuBLAS selects its algorithm and workspace by shape, so the "
                   "accumulation order changes with M: an M-row call against one call "
                   "per row (decode's M=1). 33 of 251 measurements are bitwise and the "
                   "worst is 5.4e-3 relative, below one BF16 ULP of the output, at the "
                   "widest K. Stage 2 claim D"),
};

static const struct RegionCasePair kGemmFp32LmHeadPairs[] = {
    PAIR_EXCEPTION(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE,
                   "sm_86 (2x A40)",
                   "the same projections with an FP32 output; the engine enters the "
                   "LM head with M=1, so the fixture also drives M=tokens",
                   8.0e-04, 1.3e-04,
                   "no persistent state: the FP32 output matrix is the comparison",
                   "an FP32 output has no output rounding to hide behind, so the "
                   "deviation is the accumulation order alone: worst 3.0e-5 relative "
                   "at K=17408, and 22 of 222 measurements are bitwise. Stage 2 claim D"),
};

static const struct RegionCasePair kResidualAddPairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, "-",
               "elementwise accumulation of two BF16 buffers"),
};

static const struct RegionCasePair kSiluMulPairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, "-",
               "elementwise SiLU(gate) * up over a contiguous pair of buffers"),
};

static const struct RegionCasePair kConvSiluPairs[] = {
    PAIR_EXACT(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, "-",
               "elementwise activation in place; the token count does not enter it"),
};

static const struct RegionCasePair kGdnConv1dPairs[] = {
    PAIR(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, REGION_VERDICT_UNVERIFIED,
         "-", "-", "the shift register is compared elementwise after the run",
         "the chunked call and the per-token calls must carry the same shift "
         "register, but the library scan order is not established (Stage 2 "
         "claim B)"),
};

static const struct RegionCasePair kGdnPreparePairs[] = {
    PAIR(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, REGION_VERDICT_NOT_APPLICABLE,
         "-", "-", "-",
         "not an independent entry point in this release: it is a stage inside "
         "kernel_fla_gdn, reachable only through the gdn_core pair"),
};

static const struct RegionCasePair kGdnCorePairs[] = {
    PAIR_EXCEPTION(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL,
                   "sm_86 (2x A40)",
                   "conv 10240, 48 v-heads x head_dim 128; lengths 2..128; splits at "
                   "L-1, L/2 and 64; nonzero initial state",
                   4.3e-04, 4.8e-05,
                   "the FP32 ssm_state is compared elementwise after the run",
                   "the recurrent path (one token per call) against one chunkwise "
                   "call. prepare is bitwise invariant across every arm (51/51 "
                   "measurements), so the difference is the core's: the output stays "
                   "within 9.2e-5 (7.2e-3 relative, about one BF16 ULP) and the FP32 "
                   "ssm_state within 4.2e-4 (4.0e-3 relative). A split aligned to the "
                   "FLA chunk size (64+64 at L=128) is bitwise identical. Stage 2 "
                   "claim B"),
    PAIR_EXCEPTION(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_TAIL1,
                   "sm_86 (2x A40)",
                   "conv 10240, 48 v-heads x head_dim 128; lengths 2..128; a one-token "
                   "tail after an L-1 chunk; nonzero initial state",
                   3.8e-04, 4.3e-05,
                   "the FP32 ssm_state is compared elementwise after the run",
                   "a one-token tail takes the recurrent path inside a chunked "
                   "prefill: the output stays within 3.1e-5 and the ssm_state within "
                   "3.8e-4 (4.0e-3 relative). The registered numbers are the maximum "
                   "over the Stage-2 matrix and the Stage-1 fixture, so the smaller "
                   "rms of the two does not become an accidental ceiling. Stage 2 "
                   "claim B"),
};

static const struct RegionCasePair kGdnGatedNormPairs[] = {
    PAIR(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, REGION_VERDICT_UNVERIFIED,
         "-", "-", "-",
         "an in-block reduction over head_dim; the row width is "
         "case-independent but the reduction order is not established"),
};

static const struct RegionCasePair kLogitsGatherPairs[] = {
    PAIR(REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, REGION_VERDICT_NOT_APPLICABLE,
         "-", "-", "-",
         "a host-Haskell region with no CUDA entry point to drive; the model-only "
         "suite (cabal test infer-generation-tests) covers the selector and its "
         "lowest-token-id tie rule"),
};

static const struct RegionCasePair kMaskedLossPairs[] = {
    PAIR(REGION_CASE_TRAIN_FORWARD, REGION_CASE_BACKWARD, REGION_VERDICT_NOT_APPLICABLE,
         "-", "-", "-",
         "no loss exists in this release: Stage 3 implemented the per-position "
         "log-probability a loss needs, while the masked reduction and the backward "
         "that consumes it are Stage 4"),
};

static const struct RegionCasePair kBackwardPairs[] = {
    PAIR(REGION_CASE_TRAIN_FORWARD, REGION_CASE_BACKWARD, REGION_VERDICT_NOT_APPLICABLE,
         "-", "-", "-",
         "no backward region exists; the plan's Stages 3-4 define it, and the "
         "manifest registry records it as not_implemented rather than assuming a "
         "determinism verdict"),
};

/* ------------------------------------------------------------------ */
/* The inventory                                                      */
/* ------------------------------------------------------------------ */

#define N_PAIRS(a) ((int)(sizeof(a) / sizeof((a)[0])))

static const struct RegionInventoryEntry kInventory[] = {
    {"embedding", 1, "all families", "kernels/embedding.cu: kernel_embedding",
     "int64 token_ids[tokens]; bf16 table[vocab_size,hidden_size]",
     "bf16 out[tokens,hidden_size], a row-exact copy of the selected table row",
     "none",
     "token_ids; the table is a parameter, not an activation",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kEmbeddingPairs), kEmbeddingPairs},

    {"rmsnorm", 1, "all families",
     "FlashInfer RMSNorm (kernels/flashinfer_norm.cu): gemma weight+1 or plain",
     "bf16 x[rows,cols]; bf16 raw_weight[cols] (no +1); fp32 effective eps; out may alias x",
     "bf16 out[rows,cols]",
     "none",
     "x and the inverse RMS; the entry point returns only out",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kRmsnormPairs), kRmsnormPairs},

    {"per_head_norm", 1, "families with attn_qk_norm",
     "FlashInfer per-head Q/K RMSNorm, driven by layers.cu layer_norm over "
     "[tokens*heads,head_dim] rows",
     "bf16 q or k[tokens*heads,head_dim]; bf16 raw_weight[head_dim]; fp32 effective eps",
     "bf16 q/k normalized in place",
     "none",
     "the pre-norm q/k and the inverse RMS",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kPerHeadNormPairs), kPerHeadNormPairs},

    {"rope", 1, "all families",
     "FlashInfer RoPE (kernels/flashinfer_norm.cu): split-half, rotary_dim of head_dim",
     "bf16 q[tokens,heads,head_dim] and k[tokens,kv_heads,head_dim] in place; "
     "int64 positions[tokens]",
     "bf16 q,k rotated in place",
     "none: the position is an input, not stored here",
     "the pre-rotation q,k; a backward can re-rotate from positions",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kRopePairs), kRopePairs},

    {"q_gate_split", 1, "families with attn_output_gate",
     "kernels/layers.cu: deinterleave_qg_kernel",
     "bf16 raw[tokens*heads,2*head_dim] with Q and gate interleaved per row, "
     "where a row is the flattened (token, head) pair",
     "bf16 q[tokens*heads,head_dim] and gate[tokens*heads,head_dim]",
     "none",
     "nothing: raw is recoverable from q and gate",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kQGateSplitPairs), kQGateSplitPairs},

    {"kv_write", 0, "families with a KV cache",
     "kernels/attention.cu: kernel_kv_cache_write",
     "bf16 k_new,v_new[tokens,kv_heads,head_dim]; int seq_start; int max_seq_len",
     "bf16 kv_cache[2,max_seq,kv_heads,head_dim] at [seq_start,seq_start+tokens)",
     "the KV cache is mutated in place and read by attention_core",
     "the written k/v; the cache is a copy a backward can read",
     "chunked_prefill,tail1,decode",
     "absent from a full-sequence trainer: a full-sequence forward needs no cache",
     N_PAIRS(mKvWritePairs), mKvWritePairs},

    {"attention_core", 1, "families with full attention",
     "FlashInfer single prefill (kernels/attention.cu: kernel_attention); "
     "split-KV disabled by a null workspace; the entry point does not return the "
     "LSE, although the kernel can (Stage 2 claim E)",
     "bf16 q[tokens,heads,head_dim]; bf16 kv_cache; seq_start,tokens,seq_len; "
     "scale = 1/sqrt(head_dim)",
     "bf16 out[tokens,heads,head_dim]",
     "reads the KV cache; writes no persistent state",
     "nothing usable today, and Stage 2 showed what a backward would need: this "
     "entry point passes lse=nullptr, while the same FlashInfer dispatcher writes "
     "a base-2 LSE of layout [qo_len, num_heads] f32 when asked, leaving the "
     "output bitwise unchanged (Stage 2 claim E)",
     "chunked_prefill,tail1,decode", "none",
     N_PAIRS(kAttentionCorePairs), kAttentionCorePairs},

    {"attention_output_gate", 1, "families with attn_output_gate",
     "kernels/attention.cu: kernel_sigmoid_mul (BF16-rounded sigmoid)",
     "bf16 attn[tokens,heads,head_dim]; bf16 gate with a per-token stride and offset",
     "bf16 attn * sigmoid(gate)",
     "none",
     "nothing extra: both inputs are still live",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kAttentionOutputGatePairs), kAttentionOutputGatePairs},

    {"gemm_bf16", 1, "all families",
     "kernels/gemm.cu: gemm_bf16 (cuBLAS, BF16 in, FP32 accumulate, BF16 out)",
     "bf16 x[M,K]; bf16 W[N,K]; M = tokens",
     "bf16 out[M,N]",
     "none",
     "x (for dW) and W (for dx); the engine keeps neither",
     "chunked_prefill,recurrent_prefill,decode",
     "every projection in a dense layer: Q/K/V, gate/up/down, the GDN in/out "
     "projections",
     N_PAIRS(kGemmBf16Pairs), kGemmBf16Pairs},

    {"gemm_fp32_lmhead", 1, "all families",
     "kernels/gemm.cu: gemm_bf16_f32out, the LM head in engine.cu compute_logits",
     "bf16 x[1,hidden_size] (the final row only); bf16 W[vocab_size,hidden_size]",
     "fp32 logits[vocab_size]",
     "none",
     "x and W; the projection is over the final row only",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kGemmFp32LmHeadPairs), kGemmFp32LmHeadPairs},

    {"residual_add", 1, "all families",
     "kernels/layers.cu: kernel_residual_add (dst[i] += src[i], BF16)",
     "bf16 dst[n] in place; bf16 src[n]; n = tokens * hidden_size",
     "bf16 dst updated in place",
     "the residual stream itself is the persistent state across layers",
     "nothing: the residual stream is live for the next layer",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kResidualAddPairs), kResidualAddPairs},

    {"silu_mul", 1, "families with a dense MLP",
     "kernels/silu.cu: kernel_silu_mul (FlashInfer act_and_mul)",
     "bf16 [gate[n], up[n]] contiguous in one buffer; n = tokens * intermediate_size",
     "bf16 out[n]",
     "none",
     "gate and up; the entry point returns only the product",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kSiluMulPairs), kSiluMulPairs},

    {"conv_silu", 1, "families with GDN",
     "kernels/silu.cu: kernel_silu_inplace, the activation of the GDN conv output",
     "bf16 x[n] in place; n = tokens * conv_dim",
     "bf16 x, activated in place",
     "none",
     "the pre-activation conv output",
     "chunked_prefill,recurrent_prefill,decode",
     "the conv1d activation was previously an unexported kernel in layers.cu, so "
     "no region harness could reach it",
     N_PAIRS(kConvSiluPairs), kConvSiluPairs},

    {"gdn_conv1d", 1, "families with GDN",
     "kernels/gdn_conv.cu: kernel_causal_conv1d (AOT causal_conv1d, width from "
     "the descriptor)",
     "bf16 x[tokens,conv_dim]; bf16 weight[conv_dim,1,kernel]; bf16 bias; conv_state",
     "bf16 out[tokens,conv_dim], unfused (the activation is conv_silu)",
     "conv_state[conv_dim,kernel-1] is a shift register mutated in place",
     "x and conv_state; Stage 2 claim B pairs prepare, core and backward",
     "chunked_prefill,recurrent_prefill,tail1,decode", "none",
     N_PAIRS(kGdnConv1dPairs), kGdnConv1dPairs},

    {"gdn_prepare", 1, "families with GDN",
     "kernels/fla_gdn.cu: the prepare stage of kernel_fla_gdn (Q/K L2 norm, "
     "head expansion, a/b/A_log/dt_bias transforms)",
     "bf16 conv_out[tokens,conv_dim]; bf16 a,b[tokens,v_heads]; bf16 A_log,dt_bias[v_heads]",
     "prepared q,k,v inside the FLA workspace; no separate output buffer",
     "none of its own",
     "the prepared q,k,v; a backward pairs it with the core (Stage 2 claim B)",
     "chunked_prefill,recurrent_prefill,tail1,decode", "none",
     N_PAIRS(kGdnPreparePairs), kGdnPreparePairs},

    {"gdn_core", 1, "families with GDN",
     "kernels/fla_gdn.cu: kernel_fla_gdn; AOT FLA chunkwise cubin, the recurrent "
     "path when tokens == 1",
     "bf16 conv_out[tokens,conv_dim]; bf16 a,b[tokens,v_heads]; bf16 A_log,dt_bias; "
     "fp32 ssm_state",
     "bf16 delta_out[tokens,v_dim]",
     "ssm_state[v_heads,head_dim,head_dim] in FP32, mutated in place",
     "the chunkwise states and the per-step outputs (Stage 2 claim B)",
     "chunked_prefill,recurrent_prefill,tail1,decode", "none",
     N_PAIRS(kGdnCorePairs), kGdnCorePairs},

    {"gdn_gated_norm", 1, "families with GDN",
     "kernels/gdn_norm.cu: kernel_gdn_gated_norm (FLA RMSNormGated, raw weight, "
     "Transformers BF16 boundaries)",
     "bf16 x[tokens*v_heads,head_dim]; bf16 z (the gate); fp32 effective weight",
     "bf16 out[tokens*v_heads,head_dim]",
     "none",
     "x, z and the inverse RMS",
     "chunked_prefill,recurrent_prefill,decode", "none",
     N_PAIRS(kGdnGatedNormPairs), kGdnGatedNormPairs},

    {"logits_softmax_gather", 0, "all families",
     "Haskell: src/Infer/Generation.hs argmax over the final row (lowest token id "
     "on ties); the FP32 logits come from gemm_fp32_lmhead",
     "fp32 logits[vocab_size] for the final row only",
     "int token_id",
     "none",
     "nothing: the trainer's differentiable log-softmax and loss is masked_loss, "
     "not this selector",
     "none", "none",
     N_PAIRS(kLogitsGatherPairs), kLogitsGatherPairs},

    {"masked_loss", 1, "all families (proposed)",
     "the FP32 log-softmax and gather landed in Stage 3 (kernels/logprob.cu, one row at "
     "a time so no [tokens, vocab] tensor is materialised); the masked reduction and "
     "its backward are Stage 4",
     "proposed: fp32 logits[tokens,vocab_size] and int64 targets[tokens] with a loss mask",
     "proposed: fp32 per-token log-softmax and the masked mean loss",
     "none",
     "proposed: the log-softmax probabilities (Stage 4)",
     "none",
     "the trainer's FP32 differentiable loss region; deliberately not the same "
     "region as sampler_softmax_cdf. Stage 3 gave it its first implementation (the "
     "natural-log log-softmax of a selected row), which is why it is no longer "
     "registered as absent: a region the forward runs cannot be recorded as "
     "not_implemented",
     N_PAIRS(kMaskedLossPairs), kMaskedLossPairs},

    {"backward", 1, "all families (proposed)",
     "not implemented: no backward region exists",
     "proposed: the values each forward region would have to save",
     "proposed: dW and dx per region",
     "proposed: an FP32 gradient accumulator per parameter",
     "the forward regions must save what it needs; none of them does today",
     "none",
     "the plan's Stages 3-4 define this",
     N_PAIRS(kBackwardPairs), kBackwardPairs},

    {"sampler_softmax_cdf", 0, "generation only (proposed)",
     "proposed host binary64 softmax/CDF with a request-owned RNG (plan T0-T4)",
     "fp32 logits[vocab_size]; temperature; the request's PRNG state",
     "sampled int token_id and its log-probability",
     "a request-owned PRNG state",
     "nothing: generation only",
     "none",
     "unimplemented: greedy is the only sampler in this release, and the plan "
     "registers this as a separate generation-only region rather than equating "
     "it with masked_loss",
     0, NULL},
};

static const struct RegionExclusion kExclusions[] = {
    {"mla_core",
     "Multi-head latent attention needs its own regions (latent norm, compressed "
     "KV write, tiled core) and is outside the first trainer allowlist. Its "
     "inference coverage stays in ctest test_mla and the DeepSeek-V2-Lite "
     "end-to-end run; no training case is registered for it."},
    {"moe_router",
     "MoE routing is outside the first trainer allowlist (the first target is a "
     "small dense or dense-hybrid model). Inference coverage stays in ctest "
     "test_moe and the MoE end-to-end runs."},
    {"moe_experts",
     "Per-expert GEMMs are outside the first trainer allowlist for the same "
     "reason as moe_router."},
    {"moe_combine",
     "The FP32 routed combine is outside the first trainer allowlist for the "
     "same reason as moe_router."},
    {"moe_ep_merge",
     "Expert parallelism is a placement, not a training region; it is outside "
     "the first trainer allowlist and gated by test_tp.py --ep 2."},
    {"collective_allreduce",
     "The cross-device collective is a placement primitive; its regression is "
     "the TP/EP equivalence gate, and no training case is registered for it."},
};

int region_inventory_version(void) { return REGION_INVENTORY_VERSION; }

const struct RegionInventoryEntry *region_inventory(int *count) {
    if (count != NULL) *count = (int)(sizeof(kInventory) / sizeof(kInventory[0]));
    return kInventory;
}

const struct RegionExclusion *region_exclusions(int *count) {
    if (count != NULL) *count = (int)(sizeof(kExclusions) / sizeof(kExclusions[0]));
    return kExclusions;
}

const struct RegionInventoryEntry *region_inventory_find(const char *region) {
    if (region == NULL) return NULL;
    int count = 0;
    const struct RegionInventoryEntry *table = region_inventory(&count);
    for (int i = 0; i < count; ++i) {
        if (strcmp(table[i].region, region) == 0) return &table[i];
    }
    return NULL;
}

/* The case vocabulary, in one place so `region_case_known` and the gate cannot
 * disagree with the inventory's `cases` strings. */
static const char *const kCases[] = {
    REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, REGION_CASE_TAIL1,
    REGION_CASE_DECODE, REGION_CASE_TRAIN_FORWARD, REGION_CASE_EVAL_NO_AUTOGRAD,
    REGION_CASE_RECOMPUTE, REGION_CASE_BACKWARD,
};

int region_case_known(const char *case_name) {
    if (case_name == NULL) return 0;
    for (size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        if (strcmp(kCases[i], case_name) == 0) return 1;
    }
    return 0;
}

/* A comma-separated list, matched as whole tokens so "decode" cannot match
 * "recurrent_prefill". */
static int case_list_contains(const char *list, const char *case_name) {
    if (list == NULL || case_name == NULL) return 0;
    const size_t n = strlen(case_name);
    for (const char *p = list; *p != '\0';) {
        const char *comma = strchr(p, ',');
        const size_t len = comma == NULL ? strlen(p) : (size_t)(comma - p);
        if (len == n && strncmp(p, case_name, n) == 0) return 1;
        if (comma == NULL) break;
        p = comma + 1;
    }
    return 0;
}

int region_case_available(const char *region, const char *case_name) {
    const struct RegionInventoryEntry *entry = region_inventory_find(region);
    if (entry == NULL) return 0;
    return case_list_contains(entry->cases, case_name);
}

const struct RegionCasePair *region_find_pair(const char *region,
                                              const char *left, const char *right) {
    const struct RegionInventoryEntry *entry = region_inventory_find(region);
    if (entry == NULL || left == NULL || right == NULL) return NULL;
    for (int i = 0; i < entry->pair_count; ++i) {
        const struct RegionCasePair *pair = &entry->pairs[i];
        if ((strcmp(pair->left, left) == 0 && strcmp(pair->right, right) == 0) ||
            (strcmp(pair->left, right) == 0 && strcmp(pair->right, left) == 0)) {
            return pair;
        }
    }
    return NULL;
}
