/**
 * layers.h - Layer weight structures and forward function declarations.
 */
#ifndef HASKELL_INFER_LAYERS_H
#define HASKELL_INFER_LAYERS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "moe.h"
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

/* Model dimensions (shared across all layers). Filled from the model descriptor;
 * no per-family constants live on the C side. Named so moe.h can forward-declare
 * it without including this header (the two are mutually dependent). */
typedef struct ModelDims {
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int rotary_dim;
    int norm_style;           /* 0 = gemma (weight + 1), 1 = plain */
    int attn_output_gate;     /* q_proj carries a fused output gate */
    int q_gate_interleave;    /* the fused Q/gate rows are interleaved per head */
    int num_layers;
    int vocab_size;
    float rms_eps;
    float rope_theta;
    int max_seq_len;
    // GDN
    int gdn_conv_dim;
    int gdn_value_dim;
    int gdn_num_v_heads;
    int gdn_num_k_heads;
    int gdn_head_dim;
    int gdn_conv_kernel;
    // MLA (multi-head latent attention); all zero when the model has none
    int mla_kv_lora_rank;
    int mla_qk_nope_head_dim;
    int mla_qk_rope_head_dim;
    int mla_v_head_dim;
    int max_chunk;            /* prefill batch size (<= ENGINE_MAX_CHUNK) */
} ModelDims;

/* Hard limit of the kernels (FLA chunk pipeline and layer scratch). Descriptors
 * ask for a max_chunk <= this; the engine chunks prefill accordingly. */
static constexpr int ENGINE_MAX_CHUNK = 128;

/* Per-layer MLP weights (shared between attention and GDN layers) */
typedef struct {
    const __nv_bfloat16 *gate_proj_w;   // [intermediate, hidden]
    const __nv_bfloat16 *up_proj_w;     // [intermediate, hidden]
    const __nv_bfloat16 *down_proj_w;   // [hidden, intermediate]
    const __nv_bfloat16 *post_norm_w;   // [hidden], raw Gemma weight
} MlpWeights;

/* Attention layer weights */
typedef struct {
    const __nv_bfloat16 *q_proj_w;      // [num_heads*head_dim*2, hidden]
    const __nv_bfloat16 *k_proj_w;      // [num_kv_heads*head_dim, hidden]
    const __nv_bfloat16 *v_proj_w;      // [num_kv_heads*head_dim, hidden]
    const __nv_bfloat16 *o_proj_w;      // [hidden, num_heads*head_dim]
    const __nv_bfloat16 *q_norm_w;      // [head_dim], raw Gemma weight
    const __nv_bfloat16 *k_norm_w;      // [head_dim], raw Gemma weight
    const __nv_bfloat16 *input_norm_w;  // [hidden], raw Gemma weight
} AttentionWeights;

/* GDN layer weights */
typedef struct {
    const __nv_bfloat16 *in_proj_qkv_w; // [conv_dim, hidden]
    const __nv_bfloat16 *in_proj_z_w;   // [v_dim, hidden]
    const __nv_bfloat16 *in_proj_a_w;   // [num_v_heads, hidden]
    const __nv_bfloat16 *in_proj_b_w;   // [num_v_heads, hidden]
    const __nv_bfloat16 *conv1d_w;      // [conv_dim, 1, kernel_size]
    const __nv_bfloat16 *conv1d_bias;   // [conv_dim], device-local zero for this model
    const __nv_bfloat16 *dt_bias;       // [num_v_heads]
    const __nv_bfloat16 *A_log;         // [num_v_heads]
    const __nv_bfloat16 *out_proj_w;    // [hidden, v_dim]
    const float *gdn_norm_w;           // [head_dim], effective FP32 weight (no +1)
    const __nv_bfloat16 *input_norm_w;  // [hidden], raw Gemma weight
} GdnWeights;

/* MLA layer weights (DeepSeek-V2 style). */
typedef struct {
    const __nv_bfloat16 *q_proj_w;      // [H*(nope+rope), hidden]
    const __nv_bfloat16 *kv_a_proj_w;   // [kv_lora_rank + rope, hidden]
    const __nv_bfloat16 *kv_a_norm_w;   // [kv_lora_rank], plain RMSNorm weight
    const __nv_bfloat16 *kv_b_proj_w;   // [H*(nope+v), kv_lora_rank]
    const __nv_bfloat16 *o_proj_w;      // [hidden, H*v]
    const __nv_bfloat16 *input_norm_w;  // [hidden]
} MlaWeights;

/* ------------------------------------------------------------------ */
/* Layer plan and dispatch                                            */
/* ------------------------------------------------------------------ */

/* What a layer does: which token mixer and which feed-forward sublayer. Kinds
 * come from the model descriptor (ENGINE_MIXER_* / ENGINE_FFN_* in
 * model_desc.h). Adding a kind means one enum value, one kernel file and one row
 * in the dispatch table in layer_dispatch.cu -- the engine's layer loop does not
 * change. */
typedef struct {
    int mixer;
    int ffn;
} LayerPlan;

/* Per-layer weights. The owned/state registries are filled while loading, so the
 * lifecycle (destroy frees, reset zeroes) needs no per-kind bookkeeping. */
struct LayerWeights {
    LayerPlan plan;
    /* MoE feed-forward (used when plan.ffn is moe). */
    MoeWeights moe;
    MoeConfig moe_config;
    __nv_bfloat16 *input_norm_w;
    __nv_bfloat16 *post_norm_w;
    __nv_bfloat16 *gate_proj_w;
    __nv_bfloat16 *up_proj_w;
    __nv_bfloat16 *down_proj_w;
    __nv_bfloat16 *q_proj_w;
    __nv_bfloat16 *k_proj_w;
    __nv_bfloat16 *v_proj_w;
    __nv_bfloat16 *o_proj_w;
    __nv_bfloat16 *q_norm_w;
    __nv_bfloat16 *k_norm_w;
    __nv_bfloat16 *in_proj_qkv_w;
    __nv_bfloat16 *in_proj_z_w;
    __nv_bfloat16 *in_proj_a_w;
    __nv_bfloat16 *in_proj_b_w;
    __nv_bfloat16 *conv1d_w;
    __nv_bfloat16 *dt_bias;
    __nv_bfloat16 *A_log;
    __nv_bfloat16 *gdn_out_proj_w;
    __nv_bfloat16 *gdn_norm_w;
    float *gdn_norm_f32;
    __nv_bfloat16 *mla_q_proj_w;
    __nv_bfloat16 *mla_kv_a_proj_w;
    __nv_bfloat16 *mla_kv_a_norm_w;
    __nv_bfloat16 *mla_kv_b_proj_w;
    __nv_bfloat16 *mla_o_proj_w;
    __nv_bfloat16 *kv_cache;
    __nv_bfloat16 *mla_cache;   /* latent KV: [max_seq, kv_lora_rank + rope] */
    __nv_bfloat16 *conv_state;
    float *ssm_state;

    /* Weight/inner-state buffers owned by this layer (freed on destroy). */
    std::vector<void *> owned;
    /* Buffers cleared by engine_reset (KV cache / GDN conv and SSM state). */
    std::vector<std::pair<void *, size_t>> reset_zero;
    /* Set when tensor parallelism actually split this sublayer's weights: the
     * caller then all-reduces the sublayer's partial output across ranks. */
    bool mixer_sharded = false;
    bool ffn_sharded = false;
};

/* Debug taps: which layers get their activations dumped, and where. Enabled by
 * INFER_TAP_LAYERS=0,1,2 and INFER_TAP_DIR=<dir>; shared by the engine's layer
 * loop (kind "layer", the residual stream) and the sub-layer sites in
 * layer_dispatch.cu (kind "mixer" / "ffn"). */
struct TapConfig {
    std::vector<int> layers;
    std::string dir;
};

void tap_parse_env(TapConfig *taps);
/* Dump one activation as raw float32 [tokens, cols] for comparison against the
 * reference (tests/synth/taps.py). Row width is explicit (GDN stages are wider
 * than the hidden size) and the producing stream is synchronized first, so a tap
 * cannot race the kernels that wrote it (compute streams are nonblocking). */
void tap_dump_rows(const TapConfig *taps, const char *kind, int layer, int device,
                   cudaStream_t stream, const __nv_bfloat16 *data, int tokens, int cols);

/* Where a GDN sub-layer's stage taps belong: the layer index and device are only
 * known to forward_gdn_layer's caller. */
struct GdnTapSites {
    const TapConfig *config;
    int layer;
    int device;
};

/* Optional cross-rank reduce wired in by the caller for sublayers whose output
 * is a partial sum (expert parallelism: the routed experts live on different
 * ranks). It reduces the rank's sublayer output buffer across the group; the
 * engine owns those buffers, so only the element count crosses the boundary.
 * Returns 0 or a transport error status. */
typedef int (*LayerReduceFn)(void *opaque, size_t elements);

/* Per-invocation context: device handles, scratch and sequence position. */
typedef struct {
    cublasHandle_t cublas;
    cudaStream_t stream;
    __nv_bfloat16 *workspace;
    __nv_bfloat16 *conv_bias_zero;
    const int64_t *positions;
    void *fla_scratch;
    void *moe_scratch;
    /* Expert parallelism only: this rank's routed partial in FP32. The merge
     * across ranks happens here and the activation is rounded once afterwards,
     * so the partial must not be rounded on the way out. Null otherwise. */
    float *moe_partial_f32;
    void *mla_scratch;
    int tokens;
    int seq_len;              /* sequence length including the current tokens */
    int layer_index;          /* for diagnostics only */
    int device;               /* CUDA ordinal, for diagnostics only */
    const ModelDims *dims;
    const TapConfig *taps;
    /* Non-null when a sublayer's parts are split across ranks. The caller then
     * drives the sublayer in two passes: split_phase 0 computes the local part,
     * the caller reduces, and split_phase 1 finishes the sublayer. */
    LayerReduceFn reduce;
    void *reduce_opaque;
    int split_phase;
} LayerContext;

/* norm -> mixer -> residual -> norm -> ffn -> residual, dispatched on the plan.
 * Returns 0 on success or a non-zero kernel error status. */
int forward_layer(const LayerContext *ctx, const struct LayerWeights *w,
                  const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out);

/* The two halves of a layer, for callers that synchronize between them (tensor
 * parallelism all-reduces the mixer/ffn output before the residual add). The
 * partial results are written to layer_out; the residual stream is untouched. */
int forward_mixer(const LayerContext *ctx, const struct LayerWeights *w,
                  const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out);
int forward_ffn(const LayerContext *ctx, const struct LayerWeights *w,
                const __nv_bfloat16 *residual, __nv_bfloat16 *layer_out);

/* ------------------------------------------------------------------ */
/* Cross-device primitives (collective.cu)                            */
/* ------------------------------------------------------------------ */

/* Probe (and optionally enable) peer access for every ordered device pair;
 * returns the number of reachable pairs and logs the outcome. */
int peer_probe_all(const int *devices, int count, int enable);

/* Copy [src] on from_device to [dst] on to_device, ordered by an event recorded
 * on the producer stream (no host synchronization). */
int copy_across_devices(int from_device, cudaStream_t from_stream, cudaEvent_t *from_event,
                        int to_device, cudaStream_t to_stream,
                        void *dst, const void *src, size_t bytes);

/* Element type of the buffers a cross-device primitive moves. The type belongs
 * to the *data*, not to the transport: the caller states what its buffer holds
 * and every copy is sized from it. The reduction accumulates in FP32 and rounds
 * once when storing, whatever the element type is. */
typedef enum {
    COLLECTIVE_F32 = 0,       /* float */
    COLLECTIVE_F16 = 1,       /* __half */
    COLLECTIVE_BF16 = 2,      /* __nv_bfloat16 */
    COLLECTIVE_FP8_E4M3 = 3,  /* __nv_fp8_e4m3 */
    COLLECTIVE_FP8_E5M2 = 4,  /* __nv_fp8_e5m2 */
} CollectiveDtype;

/* Bytes per element of @dtype, or 0 when the tag is unknown. */
static inline size_t collective_element_bytes(CollectiveDtype dtype) {
    switch (dtype) {
    case COLLECTIVE_F32: return sizeof(float);
    case COLLECTIVE_F16:
    case COLLECTIVE_BF16: return 2;
    case COLLECTIVE_FP8_E4M3:
    case COLLECTIVE_FP8_E5M2: return 1;
    default: return 0;
    }
}

/* In-place sum of one buffer per device (elementwise), via leader staging on
 * devices[0]. `buffers[i]` and `leader_staging` each hold @elements values of
 * @dtype, which is what every cross-device copy is sized from.
 *
 * events[i] is the producer event of stream i: it orders rank i's data before
 * the leader reads it. done_events[i] is recorded on stream i once its incoming
 * broadcast copy has finished reading devices[0]'s buffer; the leader waits on
 * all of them, so any later reuse of that buffer is ordered after every read.
 * Both arrays and streams are index-aligned with devices. The caller
 * synchronizes afterwards. Returns 0, an unknown-type error, or a transport
 * status. */
int allreduce_sum(const int *devices, cudaStream_t *streams, cudaEvent_t *events,
                  cudaEvent_t *done_events, int count, void *const *buffers,
                  void *leader_staging, size_t elements, CollectiveDtype dtype);

/* Bytes for reusable attention/GDN/MLP workspace; excludes FLA scratch and
 * the independent residual/layer_out buffers. tokens must be in [1, max_chunk]. */
size_t layer_workspace_size(int tokens, const ModelDims *dims);

/* residual and layer_out are separate [tokens, hidden_size] BF16 buffers,
 * disjoint from ws. Caller adds layer_out to residual after each call.
 * All pointers and the cuBLAS handle belong to the stream's current device.
 * Attention positions is int64[tokens]; seq_len includes these tokens.
 * GDN scratch needs kernel_fla_workspace_size(tokens, gdn_num_v_heads) bytes;
 * ssm_state stays FP32 [gdn_num_v_heads, gdn_head_dim, gdn_head_dim].
 * Internal C++ entry points may throw std::exception on backend errors. */
int forward_attention_layer(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const AttentionWeights *w, __nv_bfloat16 *kv_cache,
    const int64_t *positions, int tokens, int seq_len, const ModelDims *dims);

int forward_gdn_layer(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const GdnWeights *w, __nv_bfloat16 *conv_state,
    float *ssm_state, void *fla_scratch, int tokens, const ModelDims *dims,
    const GdnTapSites *taps = nullptr);

int forward_mlp(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const MlpWeights *w, int tokens, const ModelDims *dims);

/* Multi-head latent attention (DeepSeek-V2 style). latent_cache is
 * [max_seq_len, kv_lora_rank + qk_rope_head_dim] BF16 and holds the compressed
 * KV; scratch needs kernel_mla_scratch_size(max_seq_len, dims) bytes. Positions
 * are int64[tokens]; seq_len includes these tokens. */
int forward_mla_layer(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const MlaWeights *w, __nv_bfloat16 *latent_cache, void *scratch,
    const int64_t *positions, int tokens, int seq_len, const ModelDims *dims,
    const GdnTapSites *taps = nullptr);

/* Bytes for the MLA decoder scratch (decompressed K/V + repacked latent). */
size_t kernel_mla_scratch_size(int max_seq, const ModelDims *dims);

/* Longest sequence the MLA attention kernel can decode on the *current* device,
 * given the per-block shared-memory ceiling it does not opt out of. The engine
 * refuses a cache longer than this at initialization, and forward_mla_layer
 * refuses a longer call. */
int kernel_mla_max_seq_len(void);

/* GEMM declarations (gemm.cu) */
int gemm_bf16(cublasHandle_t handle, __nv_bfloat16 *out,
              const __nv_bfloat16 *x, const __nv_bfloat16 *W,
              int M, int N, int K);
int gemm_bf16_f32out(cublasHandle_t handle, float *out,
                     const __nv_bfloat16 *x, const __nv_bfloat16 *W,
                     int M, int N, int K);

/* Weight-only INT4 GEMM (plan "Q - Weight-only quantization", Q2): `out[M, N] = a[M, K] *
 * packed[N, K]^T` with the activation in BF16, the weight in the Q0 format
 * (csrc/include/linear_weight.h) and the output in BF16, accumulated in FP32. `scales` is
 * [N, K/group] BF16 in row-major (row, group) order, exactly what linear_quantize_int4 writes.
 *
 * There is no cuBLAS handle because there is no BF16 GEMM for a 4-bit weight: the kernel
 * unpacks and scales inside its own threads, which is the whole point (reading a quarter of a
 * BF16 weight's bytes). A shape this format cannot represent - K not a whole number of groups,
 * a group that is not a multiple of 8 - is refused here, at creation time, not midway through
 * a request. */
int gemm_int4_bf16(const __nv_bfloat16 *a, const uint8_t *packed, const __nv_bfloat16 *scales,
                   __nv_bfloat16 *out, int M, int N, int K, int group, cudaStream_t stream);

/* Safetensors loader: TensorInfo and the parsing/validation entry points live
 * in safetensors.h (CUDA-free, shared with the CPU test). */
#include "safetensors.h"

#endif /* HASKELL_INFER_LAYERS_H */
