/**
 * engine.cu - Full engine implementation: weight loading + multi-GPU forward.
 *
 * Lifecycle: engine_create loads weights from safetensors, allocates caches.
 * Forward: embed → 64 layers (attention/GDN + MLP) → norm → lm_head → logits.
 * Multi-GPU: layers split across devices, activation copied at boundaries.
 */

#include "engine.h"
#include "kernels.h"
#include "layers.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <vector>
#include <map>

/* ------------------------------------------------------------------ */
/*  Error handling                                                    */
/* ------------------------------------------------------------------ */

static thread_local char g_error_buf[512] = {0};

static void set_error(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(g_error_buf, sizeof(g_error_buf), fmt, ap);
    va_end(ap);
}

const char *engine_last_error(void) { return g_error_buf; }

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    set_error("CUDA %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return ENGINE_ERR_CUDA; } } while(0)

/* ------------------------------------------------------------------ */
/*  Hello-world (Phase 1)                                             */
/* ------------------------------------------------------------------ */

__global__ void hello_kernel(int *buf, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *buf = value;
}

int engine_hello_gpu(int device, int value) {
    CUDA_OK(cudaSetDevice(device));
    int *d_buf = nullptr;
    CUDA_OK(cudaMalloc(&d_buf, sizeof(int)));
    hello_kernel<<<1,1>>>(d_buf, value);
    CUDA_OK(cudaDeviceSynchronize());
    int h = 0;
    CUDA_OK(cudaMemcpy(&h, d_buf, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_buf));
    return h;
}

/* ------------------------------------------------------------------ */
/*  Internal structures                                               */
/* ------------------------------------------------------------------ */

static const int MAX_LAYERS = 64;
static const int HIDDEN = 5120;
static const int INTERMEDIATE = 17408;
static const int VOCAB = 248320;
static const int NUM_HEADS = 24;
static const int NUM_KV_HEADS = 4;
static const int HEAD_DIM = 256;
static const int ROTARY_DIM = 64;
static const int CONV_DIM = 10240;
static const int GDN_V_DIM = 6144;
static const int GDN_VH = 48;
static const int GDN_KH = 16;
static const int GDN_HD = 128;
static const int ATTN_INTERVAL = 4;

/* Per-layer weight pointers (device-local) */
struct LayerWeights {
    bool is_attention;
    // Shared
    __nv_bfloat16 *input_norm_w;    // raw BF16 [hidden]
    float *input_norm_w_p1;         // derived f32 [hidden]
    __nv_bfloat16 *post_norm_w;
    float *post_norm_w_p1;
    __nv_bfloat16 *gate_proj_w;     // [intermediate, hidden]
    __nv_bfloat16 *up_proj_w;
    __nv_bfloat16 *down_proj_w;     // [hidden, intermediate]
    // Attention-specific
    __nv_bfloat16 *q_proj_w;        // [12288, hidden]
    __nv_bfloat16 *k_proj_w;        // [1024, hidden]
    __nv_bfloat16 *v_proj_w;        // [1024, hidden]
    __nv_bfloat16 *o_proj_w;        // [hidden, 6144]
    __nv_bfloat16 *q_norm_w;        // [256]
    __nv_bfloat16 *k_norm_w;        // [256]
    float *q_norm_w_p1;             // derived f32 [256] (weight+1)
    float *k_norm_w_p1;             // derived f32 [256] (weight+1)
    // GDN-specific
    __nv_bfloat16 *in_proj_qkv_w;   // [10240, hidden]
    __nv_bfloat16 *in_proj_z_w;     // [6144, hidden]
    __nv_bfloat16 *in_proj_a_w;     // [48, hidden]
    __nv_bfloat16 *in_proj_b_w;     // [48, hidden]
    __nv_bfloat16 *conv1d_w;        // [10240, 1, 4]
    __nv_bfloat16 *dt_bias;         // [48]
    __nv_bfloat16 *A_log;           // [48]
    __nv_bfloat16 *gdn_out_proj_w;  // [hidden, 6144]
    __nv_bfloat16 *gdn_norm_w;      // [128]
    float *gdn_norm_w_p1;           // derived f32 [128]
    float *gdn_norm_w_p1_tiled;     // derived f32 [6144] (tiled 48x for kernel)
};

struct DeviceCtx {
    int device_id;
    cublasHandle_t cublas;
    cudaStream_t stream;
    __nv_bfloat16 *residual;    // [1, hidden] activation buffer
    __nv_bfloat16 *workspace;   // scratch space for layer computation
    size_t ws_size;
};

struct EngineHandle {
    ModelDims dims;
    int num_devices;
    std::vector<int> devices;
    std::vector<int> layer_device;  // per-layer device index
    DeviceCtx *ctx;                 // array[num_devices]
    LayerWeights *layers;           // array[MAX_LAYERS]

    // Embedding (first device) and lm_head (last device)
    __nv_bfloat16 *embed_w;         // [vocab, hidden] on devices[0]
    __nv_bfloat16 *lm_head_w;       // [vocab, hidden] on last device
    __nv_bfloat16 *final_norm_w;
    float *final_norm_w_p1;

    // RoPE tables (on each device that has attention layers)
    __nv_bfloat16 *cos_cache;       // [max_seq, rotary_dim/2]
    __nv_bfloat16 *sin_cache;

    // Per-device KV cache and GDN state
    std::vector<__nv_bfloat16*> kv_caches;   // per attention layer
    std::vector<__nv_bfloat16*> conv_states; // per GDN layer
    std::vector<float*> ssm_states;          // per GDN layer
    __nv_bfloat16 *conv_bias_zero;           // zero buffer for conv1d (no bias in model)

    // Output logits buffer (host-pinned)
    float *d_logits;                // [vocab] on last device
    int64_t *d_position;            // [1] position scalar on device 0 (for RoPE)
    int seq_len;
};

/* ------------------------------------------------------------------ */
/*  Weight loading helpers                                            */
/* ------------------------------------------------------------------ */

static bool is_attention_layer(int i) { return (i + 1) % ATTN_INTERVAL == 0; }

// Allocate and load a tensor to a specific device
static int load_weight(const std::map<std::string, TensorInfo> &idx,
                       const std::string &name, void **dst, int device) {
    auto it = idx.find(name);
    if (it == idx.end()) {
        set_error("Tensor not found: %s", name.c_str());
        return ENGINE_ERR_WEIGHTS;
    }
    const TensorInfo &ti = it->second;
    long long bytes = ti.data_end - ti.data_start;
    cudaSetDevice(device);
    cudaError_t err = cudaMalloc(dst, bytes);
    if (err != cudaSuccess) return ENGINE_ERR_ALLOC;
    return safetensors_load_tensor(ti, *dst, device);
}

/* ------------------------------------------------------------------ */
/*  engine_create                                                     */
/* ------------------------------------------------------------------ */

EngineHandle *engine_create(const char *model_dir, const EngineConfig *config) {
    EngineHandle *eng = new EngineHandle();
    eng->num_devices = config->num_devices;
    eng->devices.assign(config->devices, config->devices + config->num_devices);
    eng->layer_device.assign(config->layer_devices, config->layer_devices + config->num_layers);
    eng->seq_len = 0;

    // Set model dims
    eng->dims = {HIDDEN, INTERMEDIATE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
                 ROTARY_DIM, MAX_LAYERS, ATTN_INTERVAL, VOCAB, 1e-6f, 1e7f,
                 config->max_seq_len, CONV_DIM, GDN_VH, GDN_KH, GDN_HD, 4};

    // Scan safetensors
    std::map<std::string, TensorInfo> index;
    int nt = safetensors_scan_dir(model_dir, index);
    if (nt <= 0) { set_error("No safetensors found in %s", model_dir); return nullptr; }
    fprintf(stderr, "[engine] Scanned %d tensors from %s\n", nt, model_dir);

    // Enable peer access between devices
    for (int i = 0; i < eng->num_devices; i++)
        for (int j = 0; j < eng->num_devices; j++)
            if (i != j) {
                cudaSetDevice(eng->devices[i]);
                cudaDeviceEnablePeerAccess(eng->devices[j], 0);
            }

    // Initialize device contexts
    eng->ctx = new DeviceCtx[eng->num_devices];
    for (int d = 0; d < eng->num_devices; d++) {
        cudaSetDevice(eng->devices[d]);
        cublasCreate(&eng->ctx[d].cublas);
        eng->ctx[d].stream = 0;  // Use NULL stream for simplicity (demo)
        eng->ctx[d].device_id = eng->devices[d];
        // Residual buffer
        cudaMalloc(&eng->ctx[d].residual, HIDDEN * sizeof(__nv_bfloat16));
        // Workspace: enough for the largest intermediate (gate_up = 2*intermediate)
        eng->ctx[d].ws_size = (size_t)(2 * INTERMEDIATE + 4 * HIDDEN + 2 * CONV_DIM +
                              2 * GDN_V_DIM + NUM_HEADS * HEAD_DIM * 4 + 1024) * sizeof(__nv_bfloat16);
        cudaMalloc(&eng->ctx[d].workspace, eng->ctx[d].ws_size);
    }

    // Load embedding (first device) and lm_head (last device)
    std::string P = "model.language_model.";
    load_weight(index, P + "embed_tokens.weight", (void**)&eng->embed_w, eng->devices[0]);
    load_weight(index, "lm_head.weight", (void**)&eng->lm_head_w, eng->devices[eng->num_devices-1]);
    load_weight(index, P + "norm.weight", (void**)&eng->final_norm_w, eng->devices[eng->num_devices-1]);

    // Derive final_norm weight+1
    int last_dev = eng->devices[eng->num_devices - 1];
    cudaSetDevice(last_dev);
    cudaMalloc(&eng->final_norm_w_p1, HIDDEN * sizeof(float));
    kernel_weight_p1(eng->final_norm_w_p1, eng->final_norm_w, HIDDEN, eng->ctx[eng->num_devices-1].stream);

    // Allocate logits buffer and position buffer
    cudaMalloc(&eng->d_logits, VOCAB * sizeof(float));
    cudaSetDevice(eng->devices[0]);
    cudaMalloc(&eng->d_position, sizeof(int64_t));

    // Generate RoPE tables on each device
    int rope_half = ROTARY_DIM / 2;  // 32
    for (int d = 0; d < eng->num_devices; d++) {
        cudaSetDevice(eng->devices[d]);
        cudaMalloc(&eng->cos_cache, config->max_seq_len * rope_half * sizeof(__nv_bfloat16));
        // Note: for multi-device we'd need per-device cos/sin. Simplified: allocate on each.
        // For v1 with 2 devices, just allocate on device 0 and 1 separately.
    }
    // Actually, let's just do it on device 0 for now and copy if needed
    cudaSetDevice(eng->devices[0]);
    cudaFree(eng->cos_cache);
    cudaMalloc(&eng->cos_cache, config->max_seq_len * rope_half * sizeof(__nv_bfloat16));
    cudaMalloc(&eng->sin_cache, config->max_seq_len * rope_half * sizeof(__nv_bfloat16));
    kernel_rope_table(eng->cos_cache, eng->sin_cache, config->max_seq_len,
                      ROTARY_DIM, 1e7f, eng->ctx[0].stream);

    // Allocate zero conv1d bias on device 0 (model has no conv bias)
    cudaSetDevice(eng->devices[0]);
    cudaMalloc(&eng->conv_bias_zero, CONV_DIM * sizeof(__nv_bfloat16));
    cudaMemset(eng->conv_bias_zero, 0, CONV_DIM * sizeof(__nv_bfloat16));

    // Load per-layer weights
    eng->layers = new LayerWeights[MAX_LAYERS];
    eng->kv_caches.clear();
    eng->conv_states.clear();
    eng->ssm_states.clear();

    for (int i = 0; i < MAX_LAYERS; i++) {
        int dev = eng->devices[eng->layer_device[i]];
        cudaSetDevice(dev);
        LayerWeights &lw = eng->layers[i];
        memset(&lw, 0, sizeof(lw));
        lw.is_attention = is_attention_layer(i);

        std::string lp = P + "layers." + std::to_string(i) + ".";

        // Shared: norms + MLP
        load_weight(index, lp + "input_layernorm.weight", (void**)&lw.input_norm_w, dev);
        cudaMalloc(&lw.input_norm_w_p1, HIDDEN * sizeof(float));
        kernel_weight_p1(lw.input_norm_w_p1, lw.input_norm_w, HIDDEN, 0);

        load_weight(index, lp + "post_attention_layernorm.weight", (void**)&lw.post_norm_w, dev);
        cudaMalloc(&lw.post_norm_w_p1, HIDDEN * sizeof(float));
        kernel_weight_p1(lw.post_norm_w_p1, lw.post_norm_w, HIDDEN, 0);

        load_weight(index, lp + "mlp.gate_proj.weight", (void**)&lw.gate_proj_w, dev);
        load_weight(index, lp + "mlp.up_proj.weight", (void**)&lw.up_proj_w, dev);
        load_weight(index, lp + "mlp.down_proj.weight", (void**)&lw.down_proj_w, dev);

        if (lw.is_attention) {
            load_weight(index, lp + "self_attn.q_proj.weight", (void**)&lw.q_proj_w, dev);
            load_weight(index, lp + "self_attn.k_proj.weight", (void**)&lw.k_proj_w, dev);
            load_weight(index, lp + "self_attn.v_proj.weight", (void**)&lw.v_proj_w, dev);
            load_weight(index, lp + "self_attn.o_proj.weight", (void**)&lw.o_proj_w, dev);
            load_weight(index, lp + "self_attn.q_norm.weight", (void**)&lw.q_norm_w, dev);
            load_weight(index, lp + "self_attn.k_norm.weight", (void**)&lw.k_norm_w, dev);
            // Derive f32 weight+1 for QK norm
            cudaMalloc(&lw.q_norm_w_p1, HEAD_DIM * sizeof(float));
            kernel_weight_p1(lw.q_norm_w_p1, lw.q_norm_w, HEAD_DIM, 0);
            cudaMalloc(&lw.k_norm_w_p1, HEAD_DIM * sizeof(float));
            kernel_weight_p1(lw.k_norm_w_p1, lw.k_norm_w, HEAD_DIM, 0);
            // KV cache: [2, max_seq, num_kv_heads, head_dim]
            size_t kv_bytes = 2 * config->max_seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(__nv_bfloat16);
            __nv_bfloat16 *kv;
            cudaMalloc(&kv, kv_bytes);
            cudaMemset(kv, 0, kv_bytes);
            eng->kv_caches.push_back(kv);
        } else {
            load_weight(index, lp + "linear_attn.in_proj_qkv.weight", (void**)&lw.in_proj_qkv_w, dev);
            load_weight(index, lp + "linear_attn.in_proj_z.weight", (void**)&lw.in_proj_z_w, dev);
            load_weight(index, lp + "linear_attn.in_proj_a.weight", (void**)&lw.in_proj_a_w, dev);
            load_weight(index, lp + "linear_attn.in_proj_b.weight", (void**)&lw.in_proj_b_w, dev);
            load_weight(index, lp + "linear_attn.conv1d.weight", (void**)&lw.conv1d_w, dev);
            load_weight(index, lp + "linear_attn.dt_bias", (void**)&lw.dt_bias, dev);
            load_weight(index, lp + "linear_attn.A_log", (void**)&lw.A_log, dev);
            load_weight(index, lp + "linear_attn.out_proj.weight", (void**)&lw.gdn_out_proj_w, dev);
            load_weight(index, lp + "linear_attn.norm.weight", (void**)&lw.gdn_norm_w, dev);
            cudaMalloc(&lw.gdn_norm_w_p1, GDN_HD * sizeof(float));
            kernel_cast_bf16_f32(lw.gdn_norm_w_p1, lw.gdn_norm_w, GDN_HD, 0);
            // Tile norm weight from [128] to [6144] (repeat 48 times for v_dim)
            cudaMalloc(&lw.gdn_norm_w_p1_tiled, GDN_V_DIM * sizeof(float));
            {
                // Simple tiling: copy 128 floats 48 times
                float *h_tiled = (float*)malloc(GDN_V_DIM * sizeof(float));
                float *h_src = (float*)malloc(GDN_HD * sizeof(float));
                cudaMemcpy(h_src, lw.gdn_norm_w_p1, GDN_HD * sizeof(float), cudaMemcpyDeviceToHost);
                for (int t = 0; t < GDN_VH; t++)
                    memcpy(h_tiled + t * GDN_HD, h_src, GDN_HD * sizeof(float));
                cudaMemcpy(lw.gdn_norm_w_p1_tiled, h_tiled, GDN_V_DIM * sizeof(float), cudaMemcpyHostToDevice);
                free(h_tiled); free(h_src);
            }
            // Conv state: [conv_dim, kernel-1]
            __nv_bfloat16 *cs;
            size_t cs_bytes = CONV_DIM * 3 * sizeof(__nv_bfloat16);
            cudaMalloc(&cs, cs_bytes);
            cudaMemset(cs, 0, cs_bytes);
            eng->conv_states.push_back(cs);
            // SSM state: [48, 128, 128] f32
            float *ss;
            size_t ss_bytes = GDN_VH * GDN_HD * GDN_HD * sizeof(float);
            cudaMalloc(&ss, ss_bytes);
            cudaMemset(ss, 0, ss_bytes);
            eng->ssm_states.push_back(ss);
        }

        if (i % 16 == 0) fprintf(stderr, "[engine] Loaded layer %d/%d\n", i, MAX_LAYERS);
    }

    // Synchronize ALL devices to ensure weight_p1 kernels completed
    for (int d = 0; d < eng->num_devices; d++) {
        cudaSetDevice(eng->devices[d]);
        cudaDeviceSynchronize();
    }
    fprintf(stderr, "[engine] Model loaded successfully on %d device(s)\n", eng->num_devices);
    return eng;
}

/* ------------------------------------------------------------------ */
/*  Forward pass (single token)                                       */
/* ------------------------------------------------------------------ */

static int forward_token(EngineHandle *eng, int64_t token_id, float *h_logits) {
    int attn_idx = 0, gdn_idx = 0;
    int cur_dev_idx = 0;  // index into eng->devices

    // 1. Embedding on first device
    int dev0 = eng->devices[0];
    cudaSetDevice(dev0);
    __nv_bfloat16 *act = eng->ctx[0].residual;
    int64_t *d_token;
    cudaMalloc(&d_token, sizeof(int64_t));
    cudaMemcpy(d_token, &token_id, sizeof(int64_t), cudaMemcpyHostToDevice);
    kernel_embedding(act, eng->embed_w, d_token, HIDDEN, 1, eng->ctx[0].stream);
    cudaFree(d_token);

    // Update position buffer for RoPE (position = current seq_len, 0-based)
    int64_t pos = (int64_t)eng->seq_len;
    cudaMemcpy(eng->d_position, &pos, sizeof(int64_t), cudaMemcpyHostToDevice);
    // Sync and check for errors after embedding
    cudaDeviceSynchronize();
    cudaError_t emb_err = cudaGetLastError();
    if (emb_err != cudaSuccess) {
        set_error("Embedding kernel failed: %s", cudaGetErrorString(emb_err));
        return ENGINE_ERR_CUDA;
    }

    // Diagnostic: print first 4 embedding values (only for first token)
    if (eng->seq_len == 0) {
        __nv_bfloat16 h_emb[4];
        cudaMemcpy(h_emb, act, 4 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[diag] embed[0:4] = %.6f %.6f %.6f %.6f (token=%lld)\n",
                __bfloat162float(h_emb[0]), __bfloat162float(h_emb[1]),
                __bfloat162float(h_emb[2]), __bfloat162float(h_emb[3]),
                (long long)token_id);
    }

    // 2. Layers
    for (int i = 0; i < MAX_LAYERS; i++) {
        int dev_idx = eng->layer_device[i];
        int dev = eng->devices[dev_idx];

        // Cross-device activation copy
        if (dev_idx != cur_dev_idx) {
            cudaSetDevice(dev);
            cudaMemcpyPeerAsync(eng->ctx[dev_idx].residual, dev,
                               act, eng->devices[cur_dev_idx],
                               HIDDEN * sizeof(__nv_bfloat16),
                               eng->ctx[dev_idx].stream);
            cudaStreamSynchronize(eng->ctx[dev_idx].stream);
            act = eng->ctx[dev_idx].residual;
            cur_dev_idx = dev_idx;
        }

        cudaSetDevice(dev);
        cublasHandle_t cublas = eng->ctx[dev_idx].cublas;
        cudaStream_t stream = eng->ctx[dev_idx].stream;
        __nv_bfloat16 *ws = eng->ctx[dev_idx].workspace;
        LayerWeights &lw = eng->layers[i];

        // layer_out buffer: dedicated space for the layer's output before residual add
        // Place it after the MLP workspace to avoid conflicts
        __nv_bfloat16 *layer_out = ws + HIDDEN + 2 * INTERMEDIATE + INTERMEDIATE + HIDDEN;

        if (lw.is_attention) {
            AttentionWeights aw;
            aw.q_proj_w = lw.q_proj_w; aw.k_proj_w = lw.k_proj_w;
            aw.v_proj_w = lw.v_proj_w; aw.o_proj_w = lw.o_proj_w;
            aw.q_norm_w = lw.q_norm_w; aw.k_norm_w = lw.k_norm_w;
            aw.q_norm_w_p1 = lw.q_norm_w_p1; aw.k_norm_w_p1 = lw.k_norm_w_p1;
            aw.input_norm_w_p1 = lw.input_norm_w_p1;
            forward_attention_layer(cublas, stream, act, ws, layer_out, &aw,
                                   eng->kv_caches[attn_idx], eng->cos_cache,
                                   eng->sin_cache, eng->d_position, eng->seq_len + 1, &eng->dims);
            attn_idx++;
        } else {
            GdnWeights gw;
            gw.in_proj_qkv_w = lw.in_proj_qkv_w; gw.in_proj_z_w = lw.in_proj_z_w;
            gw.in_proj_a_w = lw.in_proj_a_w; gw.in_proj_b_w = lw.in_proj_b_w;
            gw.conv1d_w = lw.conv1d_w; gw.conv1d_bias = eng->conv_bias_zero;
            gw.dt_bias = lw.dt_bias; gw.A_log = lw.A_log;
            gw.out_proj_w = lw.gdn_out_proj_w; gw.gdn_norm_w_p1 = lw.gdn_norm_w_p1;
            gw.input_norm_w_p1 = lw.input_norm_w_p1;
            forward_gdn_layer(cublas, stream, act, ws, layer_out, &gw,
                             eng->conv_states[gdn_idx], eng->ssm_states[gdn_idx], &eng->dims);
            gdn_idx++;
        }

        // Residual add: act += layer_out
        kernel_residual_add(act, layer_out, HIDDEN, stream);

        // MLP: post_norm → gate_up → silu → down → residual add
        __nv_bfloat16 *normed = ws;
        __nv_bfloat16 *gate_up = ws + HIDDEN;
        __nv_bfloat16 *mlp_act = gate_up + 2 * INTERMEDIATE;
        __nv_bfloat16 *mlp_out = mlp_act + INTERMEDIATE;

        kernel_rms_norm(normed, act, lw.post_norm_w_p1, HIDDEN, 1, eng->dims.rms_eps, stream);
        gemm_bf16(cublas, gate_up, normed, lw.gate_proj_w, 1, INTERMEDIATE, HIDDEN);
        gemm_bf16(cublas, gate_up + INTERMEDIATE, normed, lw.up_proj_w, 1, INTERMEDIATE, HIDDEN);
        kernel_silu_mul(mlp_act, gate_up, gate_up + INTERMEDIATE, INTERMEDIATE, stream);
        gemm_bf16(cublas, mlp_out, mlp_act, lw.down_proj_w, 1, HIDDEN, INTERMEDIATE);

        // Residual add: act += mlp_out
        kernel_residual_add(act, mlp_out, HIDDEN, stream);
        // Diagnostic: print residual after layer 0 (first token only)
        if (i == 0 && eng->seq_len == 0) {
            __nv_bfloat16 h_act[4];
            cudaMemcpy(h_act, act, 4 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[diag] after_layer0[0:4] = %.6f %.6f %.6f %.6f\n",
                    __bfloat162float(h_act[0]), __bfloat162float(h_act[1]),
                    __bfloat162float(h_act[2]), __bfloat162float(h_act[3]));
        }
    }

    // 3. Final norm + lm_head on last device
    int last_idx = eng->num_devices - 1;
    int last_dev = eng->devices[last_idx];
    if (cur_dev_idx != last_idx) {
        cudaSetDevice(last_dev);
        cudaMemcpyPeerAsync(eng->ctx[last_idx].residual, last_dev,
                           act, eng->devices[cur_dev_idx],
                           HIDDEN * sizeof(__nv_bfloat16),
                           eng->ctx[last_idx].stream);
        cudaStreamSynchronize(eng->ctx[last_idx].stream);
        act = eng->ctx[last_idx].residual;
    }

    cudaSetDevice(last_dev);
    __nv_bfloat16 *normed_final = eng->ctx[last_idx].workspace;
    kernel_rms_norm(normed_final, act, eng->final_norm_w_p1, HIDDEN, 1,
                    eng->dims.rms_eps, eng->ctx[last_idx].stream);

    // lm_head: [1, hidden] @ [vocab, hidden]^T → [1, vocab] f32
    gemm_bf16_f32out(eng->ctx[last_idx].cublas, eng->d_logits,
                     normed_final, eng->lm_head_w, 1, VOCAB, HIDDEN);

    // Copy logits to host
    cudaMemcpy(h_logits, eng->d_logits, VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
    eng->seq_len++;
    return ENGINE_OK;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

int engine_prefill(EngineHandle *eng, const int64_t *token_ids,
                   int num_tokens, float *out_logits) {
    if (!eng) return ENGINE_ERR_STATE;
    // Per-token prefill: process each token sequentially
    for (int t = 0; t < num_tokens; t++) {
        int rc = forward_token(eng, token_ids[t], out_logits);
        if (rc != ENGINE_OK) return rc;
    }
    return ENGINE_OK;
}

int engine_decode(EngineHandle *eng, int64_t token_id, float *out_logits) {
    if (!eng) return ENGINE_ERR_STATE;
    return forward_token(eng, token_id, out_logits);
}

void engine_reset(EngineHandle *eng) {
    if (!eng) return;
    eng->seq_len = 0;
    // Zero KV caches and GDN states
    for (size_t i = 0; i < eng->kv_caches.size(); i++) {
        size_t bytes = 2 * eng->dims.max_seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(__nv_bfloat16);
        cudaMemset(eng->kv_caches[i], 0, bytes);
    }
    for (size_t i = 0; i < eng->conv_states.size(); i++)
        cudaMemset(eng->conv_states[i], 0, CONV_DIM * 3 * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < eng->ssm_states.size(); i++)
        cudaMemset(eng->ssm_states[i], 0, GDN_VH * GDN_HD * GDN_HD * sizeof(float));
}

void engine_destroy(EngineHandle *eng) {
    if (!eng) return;
    // TODO: free all GPU memory
    delete[] eng->ctx;
    delete[] eng->layers;
    delete eng;
}

int engine_vocab_size(const EngineHandle *eng) { return eng ? eng->dims.vocab_size : 0; }
int engine_seq_len(const EngineHandle *eng) { return eng ? eng->seq_len : 0; }
