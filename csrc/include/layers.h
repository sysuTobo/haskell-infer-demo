/**
 * layers.h - Layer weight structures and forward function declarations.
 */
#ifndef HASKELL_INFER_LAYERS_H
#define HASKELL_INFER_LAYERS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdint.h>

/* Model dimensions (shared across all layers) */
typedef struct {
    int hidden_size;          // 5120
    int intermediate_size;    // 17408
    int num_heads;            // 24
    int num_kv_heads;         // 4
    int head_dim;             // 256
    int rotary_dim;           // 64
    int num_layers;           // 64
    int full_attn_interval;   // 4
    int vocab_size;           // 248320
    float rms_eps;            // 1e-6
    float rope_theta;         // 1e7
    int max_seq_len;          // 4096
    // GDN
    int gdn_conv_dim;         // 10240
    int gdn_num_v_heads;      // 48
    int gdn_num_k_heads;      // 16
    int gdn_head_dim;         // 128
    int gdn_conv_kernel;      // 4
} ModelDims;

/* Per-layer MLP weights (shared between attention and GDN layers) */
typedef struct {
    const __nv_bfloat16 *gate_proj_w;   // [intermediate, hidden]
    const __nv_bfloat16 *up_proj_w;     // [intermediate, hidden]
    const __nv_bfloat16 *down_proj_w;   // [hidden, intermediate]
    const float *post_norm_w_p1;        // [hidden] f32 (weight+1)
} MlpWeights;

/* Attention layer weights */
typedef struct {
    const __nv_bfloat16 *q_proj_w;      // [num_heads*head_dim*2, hidden] = [12288, 5120]
    const __nv_bfloat16 *k_proj_w;      // [num_kv_heads*head_dim, hidden] = [1024, 5120]
    const __nv_bfloat16 *v_proj_w;      // [num_kv_heads*head_dim, hidden] = [1024, 5120]
    const __nv_bfloat16 *o_proj_w;      // [hidden, num_heads*head_dim] = [5120, 6144]
    const __nv_bfloat16 *q_norm_w;      // [head_dim] = [256]
    const __nv_bfloat16 *k_norm_w;      // [head_dim] = [256]
    const float *input_norm_w_p1;       // [hidden] f32
    MlpWeights mlp;
} AttentionWeights;

/* GDN layer weights */
typedef struct {
    const __nv_bfloat16 *in_proj_qkv_w; // [conv_dim, hidden] = [10240, 5120]
    const __nv_bfloat16 *in_proj_z_w;   // [v_dim, hidden] = [6144, 5120]
    const __nv_bfloat16 *in_proj_a_w;   // [num_v_heads, hidden] = [48, 5120]
    const __nv_bfloat16 *in_proj_b_w;   // [num_v_heads, hidden] = [48, 5120]
    const __nv_bfloat16 *conv1d_w;      // [conv_dim, 1, kernel_size] = [10240, 1, 4]
    const __nv_bfloat16 *conv1d_bias;   // [conv_dim] or NULL (model has no conv bias)
    const __nv_bfloat16 *dt_bias;       // [num_v_heads] = [48]
    const __nv_bfloat16 *A_log;         // [num_v_heads] = [48]
    const __nv_bfloat16 *out_proj_w;    // [hidden, v_dim] = [5120, 6144]
    const float *gdn_norm_w_p1;         // [head_dim] f32 = [128]
    const float *input_norm_w_p1;       // [hidden] f32
    MlpWeights mlp;
} GdnWeights;

/* Forward function declarations.
 * layer_out: caller-provided buffer [1, hidden_size] for the layer's output
 * (before residual add). The caller does: residual += layer_out. */
int forward_attention_layer(cublasHandle_t cublas, cudaStream_t stream,
    __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const AttentionWeights *w, __nv_bfloat16 *kv_cache,
    __nv_bfloat16 *cos_cache, __nv_bfloat16 *sin_cache,
    int64_t *d_position, int seq_len, const ModelDims *dims);

int forward_gdn_layer(cublasHandle_t cublas, cudaStream_t stream,
    __nv_bfloat16 *residual, __nv_bfloat16 *ws, __nv_bfloat16 *layer_out,
    const GdnWeights *w, __nv_bfloat16 *conv_state,
    float *ssm_state, const ModelDims *dims);

int forward_mlp(cublasHandle_t cublas, cudaStream_t stream,
    __nv_bfloat16 *residual, __nv_bfloat16 *normed,
    __nv_bfloat16 *gate_up_out, __nv_bfloat16 *mlp_act,
    __nv_bfloat16 *mlp_down_out, const __nv_bfloat16 *post_norm_w_p1,
    const __nv_bfloat16 *gate_proj_w, const __nv_bfloat16 *up_proj_w,
    const __nv_bfloat16 *down_proj_w, const float *post_norm_w_p1_f32,
    int hidden, int intermediate, float eps);

/* GEMM declarations (gemm.cu) */
int gemm_bf16(cublasHandle_t handle, __nv_bfloat16 *out,
              const __nv_bfloat16 *x, const __nv_bfloat16 *W,
              int M, int N, int K);
int gemm_bf16_f32out(cublasHandle_t handle, float *out,
                     const __nv_bfloat16 *x, const __nv_bfloat16 *W,
                     int M, int N, int K);

/* Safetensors loader declarations */
#include <string>
#include <vector>
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
