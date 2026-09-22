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
    __nv_bfloat16 *kv_cache;
    __nv_bfloat16 *conv_state;
    float *ssm_state;

    /* Weight/inner-state buffers owned by this layer (freed on destroy). */
    std::vector<void *> owned;
    /* Buffers cleared by engine_reset (KV cache / GDN conv and SSM state). */
    std::vector<std::pair<void *, size_t>> reset_zero;
};

/* Per-invocation context: device handles, scratch and sequence position. */
typedef struct {
    cublasHandle_t cublas;
    cudaStream_t stream;
    __nv_bfloat16 *workspace;
    __nv_bfloat16 *conv_bias_zero;
    const int64_t *positions;
    void *fla_scratch;
    void *moe_scratch;
    int tokens;
    int seq_len;              /* sequence length including the current tokens */
    int layer_index;          /* for diagnostics only */
    const ModelDims *dims;
} LayerContext;

/* norm -> mixer -> residual -> norm -> ffn -> residual, dispatched on the plan.
 * Returns 0 on success or a non-zero kernel error status. */
int forward_layer(const LayerContext *ctx, const struct LayerWeights *w,
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

/* In-place sum of one bf16 buffer per device (elementwise, bf16 arithmetic), via
 * leader staging on devices[0]. events/streams are index-aligned with devices;
 * the caller synchronizes afterwards. */
int allreduce_sum_bf16(const int *devices, cudaStream_t *streams, cudaEvent_t *events,
                       int count, __nv_bfloat16 **buffers, __nv_bfloat16 *leader_staging,
                       size_t elements);

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
    float *ssm_state, void *fla_scratch, int tokens, const ModelDims *dims);

int forward_mlp(cublasHandle_t cublas, cudaStream_t stream,
    const __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const MlpWeights *w, int tokens, const ModelDims *dims);

/* GEMM declarations (gemm.cu) */
int gemm_bf16(cublasHandle_t handle, __nv_bfloat16 *out,
              const __nv_bfloat16 *x, const __nv_bfloat16 *W,
              int M, int N, int K);
int gemm_bf16_f32out(cublasHandle_t handle, float *out,
                     const __nv_bfloat16 *x, const __nv_bfloat16 *W,
                     int M, int N, int K);

/* Safetensors loader declarations */
#include <map>

struct TensorInfo {
    std::string name;
    int dtype;
    int shape[4];
    int ndim;
    long long data_start;
    long long data_end;
    std::string file_path;
    long long file_data_offset;
};

int safetensors_scan_dir(const char *model_dir, std::map<std::string, TensorInfo> &index);
int safetensors_load_tensor(const TensorInfo &ti, void *dst, int device);

#endif /* HASKELL_INFER_LAYERS_H */
