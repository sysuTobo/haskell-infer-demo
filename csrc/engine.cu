/** Weight loading and chunked, layer-partitioned multi-GPU inference. */
#include "engine.h"
#include "kernels.h"
#include "layers.h"
#include "flashinfer_ops.h"
#include "fla_ops.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local char g_error_buf[512] = {0};

static void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error_buf, sizeof(g_error_buf), fmt, ap);
    va_end(ap);
}

const char *engine_last_error(void) { return g_error_buf; }

class EngineError : public std::runtime_error {
public:
    int code;
    EngineError(int code, const std::string &message)
        : std::runtime_error(message), code(code) {}
};

static void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw EngineError(status == cudaErrorMemoryAllocation ? ENGINE_ERR_ALLOC : ENGINE_ERR_CUDA,
                          std::string(operation) + ": " + cudaGetErrorString(status));
}

static void check_cublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw EngineError(ENGINE_ERR_CUDA, std::string(operation) + ": " + std::to_string(status));
}

static void check_forward(int status, const char *operation) {
    if (status != 0)
        throw EngineError(ENGINE_ERR_CUDA, std::string(operation) + ": " + std::to_string(status));
    check_cuda(cudaGetLastError(), operation);
}

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    set_error("CUDA %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return ENGINE_ERR_CUDA; } } while(0)

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

struct LayerWeights {
    bool is_attention;
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
};

struct DeviceCtx {
    int device_id = -1;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    __nv_bfloat16 *residual = nullptr;   // [128, hidden]
    __nv_bfloat16 *layer_out = nullptr;  // Never aliases layer/MLP scratch.
    __nv_bfloat16 *workspace = nullptr;
    __nv_bfloat16 *conv_bias_zero = nullptr;
    int64_t *positions = nullptr;
    void *fla_scratch = nullptr;
    size_t ws_size = 0;
    size_t fla_size = 0;
};

struct EngineHandle {
    ModelDims dims{};
    int num_devices = 0;
    std::vector<int> devices;
    std::vector<int> layer_device;      // Internal indices, not CUDA ordinals.
    DeviceCtx *ctx = nullptr;
    LayerWeights *layers = nullptr;
    __nv_bfloat16 *embed_w = nullptr;   // First configured device.
    int64_t *token_ids = nullptr;
    __nv_bfloat16 *lm_head_w = nullptr; // Last configured device.
    __nv_bfloat16 *final_norm_w = nullptr;
    float *d_logits = nullptr;
    int seq_len = 0;
    bool state_valid = true;
};

static bool is_attention_layer(int i) { return (i + 1) % ATTN_INTERVAL == 0; }

static size_t kv_cache_bytes(const EngineHandle *eng) {
    return (size_t)2 * eng->dims.max_seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(__nv_bfloat16);
}

static size_t conv_state_bytes() { return (size_t)CONV_DIM * 3 * sizeof(__nv_bfloat16); }
static size_t ssm_state_bytes() { return (size_t)GDN_VH * GDN_HD * GDN_HD * sizeof(float); }

static void load_weight(const std::map<std::string, TensorInfo> &index,
                        const std::string &name, __nv_bfloat16 **dst,
                        int device, size_t elements) {
    auto it = index.find(name);
    if (it == index.end())
        throw EngineError(ENGINE_ERR_WEIGHTS, "Tensor not found: " + name);
    const TensorInfo &ti = it->second;
    const size_t bytes = elements * sizeof(__nv_bfloat16);
    if (ti.dtype != 0 || ti.data_start < 0 || ti.data_end < ti.data_start ||
        (unsigned long long)(ti.data_end - ti.data_start) != bytes)
        throw EngineError(ENGINE_ERR_WEIGHTS, "Unexpected tensor dtype/size: " + name);
    check_cuda(cudaSetDevice(device), "Set weight device");
    check_cuda(cudaMalloc(dst, bytes), "Allocate weight");
    int status = safetensors_load_tensor(ti, *dst, device);
    check_cuda(cudaGetLastError(), "Upload weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, "Cannot load tensor: " + name);
    // The loader uploads on the default stream; compute streams are nonblocking.
    check_cuda(cudaStreamSynchronize(nullptr), "Finish weight upload");
}

EngineHandle *engine_create(const char *model_dir, const EngineConfig *config) {
    g_error_buf[0] = '\0';
    EngineHandle *eng = nullptr;
    try {
        if (!model_dir || !*model_dir || !config || config->num_layers != MAX_LAYERS ||
            config->num_devices < 1 || !config->devices || !config->layer_devices ||
            config->max_seq_len < 1)
            throw EngineError(ENGINE_ERR_CONFIG, "Invalid engine configuration");
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "Get device count");
        if (config->num_devices > device_count)
            throw EngineError(ENGINE_ERR_CONFIG, "Too many configured devices");

        eng = new EngineHandle();
        eng->num_devices = config->num_devices;
        eng->devices.assign(config->devices, config->devices + config->num_devices);
        for (int d = 0; d < eng->num_devices; ++d) {
            int ordinal = eng->devices[d];
            if (ordinal < 0 || ordinal >= device_count ||
                std::find(eng->devices.begin(), eng->devices.begin() + d, ordinal) !=
                    eng->devices.begin() + d)
                throw EngineError(ENGINE_ERR_CONFIG, "Invalid or duplicate CUDA device ordinal");
        }
        eng->layer_device.resize(MAX_LAYERS);
        for (int i = 0; i < MAX_LAYERS; ++i) {
            // Haskell's layer_devices contains CUDA ordinals, including for [1,0].
            auto it = std::find(eng->devices.begin(), eng->devices.end(), config->layer_devices[i]);
            if (it == eng->devices.end())
                throw EngineError(ENGINE_ERR_CONFIG, "Layer assigned to an unconfigured device");
            eng->layer_device[i] = (int)(it - eng->devices.begin());
        }
        eng->dims = {HIDDEN, INTERMEDIATE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
                     ROTARY_DIM, MAX_LAYERS, ATTN_INTERVAL, VOCAB, 1e-6f, 1e7f,
                     config->max_seq_len, CONV_DIM, GDN_VH, GDN_KH, GDN_HD, 4};

        std::map<std::string, TensorInfo> index;
        int nt = safetensors_scan_dir(model_dir, index);
        if (nt <= 0)
            throw EngineError(ENGINE_ERR_WEIGHTS, std::string("No safetensors found in ") + model_dir);
        fprintf(stderr, "[engine] Scanned %d tensors from %s\n", nt, model_dir);

        eng->ctx = new DeviceCtx[eng->num_devices]();
        eng->layers = new LayerWeights[MAX_LAYERS]();
        const size_t activation_bytes = (size_t)ENGINE_BATCH_TOKENS * HIDDEN * sizeof(__nv_bfloat16);
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            ctx.device_id = eng->devices[d];
            check_cuda(cudaSetDevice(ctx.device_id), "Initialize device");
            check_cuda(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "Create stream");
            check_cublas(cublasCreate(&ctx.cublas), "Create cuBLAS");
            check_cublas(cublasSetStream(ctx.cublas, ctx.stream), "Set cuBLAS stream");
            check_cuda(cudaMalloc(&ctx.residual, activation_bytes), "Allocate residual");
            check_cuda(cudaMalloc(&ctx.layer_out, activation_bytes), "Allocate layer output");
            ctx.ws_size = layer_workspace_size(ENGINE_BATCH_TOKENS, &eng->dims);
            check_cuda(cudaMalloc(&ctx.workspace, ctx.ws_size), "Allocate layer workspace");
            ctx.fla_size = kernel_fla_workspace_size(ENGINE_BATCH_TOKENS, GDN_VH);
            check_cuda(cudaMalloc(&ctx.fla_scratch, ctx.fla_size), "Allocate FLA scratch");
            check_cuda(cudaMalloc(&ctx.positions, ENGINE_BATCH_TOKENS * sizeof(int64_t)), "Allocate positions");
            check_cuda(cudaMalloc(&ctx.conv_bias_zero, CONV_DIM * sizeof(__nv_bfloat16)), "Allocate conv bias");
            check_cuda(cudaMemsetAsync(ctx.conv_bias_zero, 0, CONV_DIM * sizeof(__nv_bfloat16), ctx.stream),
                       "Zero conv bias");
        }

        const std::string P = "model.language_model.";
        load_weight(index, P + "embed_tokens.weight", &eng->embed_w,
                    eng->devices.front(), (size_t)VOCAB * HIDDEN);
        check_cuda(cudaMalloc(&eng->token_ids, ENGINE_BATCH_TOKENS * sizeof(int64_t)), "Allocate token IDs");
        load_weight(index, "lm_head.weight", &eng->lm_head_w,
                    eng->devices.back(), (size_t)VOCAB * HIDDEN);
        load_weight(index, P + "norm.weight", &eng->final_norm_w, eng->devices.back(), HIDDEN);
        check_cuda(cudaMalloc(&eng->d_logits, VOCAB * sizeof(float)), "Allocate logits");

        for (int i = 0; i < MAX_LAYERS; ++i) {
            DeviceCtx &ctx = eng->ctx[eng->layer_device[i]];
            check_cuda(cudaSetDevice(ctx.device_id), "Load layer device");
            LayerWeights &lw = eng->layers[i];
            lw.is_attention = is_attention_layer(i);
            const std::string lp = P + "layers." + std::to_string(i) + ".";
            auto load = [&](const std::string &suffix, __nv_bfloat16 **dst, size_t elements) {
                load_weight(index, lp + suffix, dst, ctx.device_id, elements);
            };
            load("input_layernorm.weight", &lw.input_norm_w, HIDDEN);
            load("post_attention_layernorm.weight", &lw.post_norm_w, HIDDEN);
            load("mlp.gate_proj.weight", &lw.gate_proj_w, (size_t)INTERMEDIATE * HIDDEN);
            load("mlp.up_proj.weight", &lw.up_proj_w, (size_t)INTERMEDIATE * HIDDEN);
            load("mlp.down_proj.weight", &lw.down_proj_w, (size_t)HIDDEN * INTERMEDIATE);

            if (lw.is_attention) {
                load("self_attn.q_proj.weight", &lw.q_proj_w, (size_t)2 * NUM_HEADS * HEAD_DIM * HIDDEN);
                load("self_attn.k_proj.weight", &lw.k_proj_w, (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN);
                load("self_attn.v_proj.weight", &lw.v_proj_w, (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN);
                load("self_attn.o_proj.weight", &lw.o_proj_w, (size_t)HIDDEN * NUM_HEADS * HEAD_DIM);
                load("self_attn.q_norm.weight", &lw.q_norm_w, HEAD_DIM);
                load("self_attn.k_norm.weight", &lw.k_norm_w, HEAD_DIM);
                check_cuda(cudaMalloc(&lw.kv_cache, kv_cache_bytes(eng)), "Allocate KV cache");
                check_cuda(cudaMemsetAsync(lw.kv_cache, 0, kv_cache_bytes(eng), ctx.stream), "Zero KV cache");
            } else {
                load("linear_attn.in_proj_qkv.weight", &lw.in_proj_qkv_w, (size_t)CONV_DIM * HIDDEN);
                load("linear_attn.in_proj_z.weight", &lw.in_proj_z_w, (size_t)GDN_V_DIM * HIDDEN);
                load("linear_attn.in_proj_a.weight", &lw.in_proj_a_w, (size_t)GDN_VH * HIDDEN);
                load("linear_attn.in_proj_b.weight", &lw.in_proj_b_w, (size_t)GDN_VH * HIDDEN);
                load("linear_attn.conv1d.weight", &lw.conv1d_w, (size_t)CONV_DIM * 4);
                load("linear_attn.dt_bias", &lw.dt_bias, GDN_VH);
                load("linear_attn.A_log", &lw.A_log, GDN_VH);
                load("linear_attn.out_proj.weight", &lw.gdn_out_proj_w, (size_t)HIDDEN * GDN_V_DIM);
                load("linear_attn.norm.weight", &lw.gdn_norm_w, GDN_HD);
                check_cuda(cudaMalloc(&lw.gdn_norm_f32, GDN_HD * sizeof(float)), "Allocate GDN norm");
                kernel_cast_bf16_f32(lw.gdn_norm_f32, lw.gdn_norm_w, GDN_HD, ctx.stream);
                check_cuda(cudaGetLastError(), "Convert GDN norm");
                check_cuda(cudaMalloc(&lw.conv_state, conv_state_bytes()), "Allocate conv state");
                check_cuda(cudaMemsetAsync(lw.conv_state, 0, conv_state_bytes(), ctx.stream), "Zero conv state");
                check_cuda(cudaMalloc(&lw.ssm_state, ssm_state_bytes()), "Allocate SSM state");
                check_cuda(cudaMemsetAsync(lw.ssm_state, 0, ssm_state_bytes(), ctx.stream), "Zero SSM state");
            }
            if (i % 16 == 0) fprintf(stderr, "[engine] Loaded layer %d/%d\n", i, MAX_LAYERS);
        }
        for (int d = 0; d < eng->num_devices; ++d) {
            check_cuda(cudaSetDevice(eng->devices[d]), "Synchronize loaded device");
            check_cuda(cudaStreamSynchronize(eng->ctx[d].stream), "Finish loading weights");
        }
        fprintf(stderr, "[engine] Model loaded successfully on %d device(s)\n", eng->num_devices);
        return eng;
    } catch (const std::exception &e) {
        set_error("engine_create: %s", e.what());
    } catch (...) {
        set_error("engine_create: unknown failure");
    }
    engine_destroy(eng);
    return nullptr;
}

static __nv_bfloat16 *move_activation(EngineHandle *eng, int from, int to, int tokens) {
    if (from != to) {
        DeviceCtx &src = eng->ctx[from];
        DeviceCtx &dst = eng->ctx[to];
        // The destination stream alone does not order reads after source kernels.
        check_cuda(cudaSetDevice(src.device_id), "Select activation source");
        check_cuda(cudaStreamSynchronize(src.stream), "Finish activation source");
        check_cuda(cudaSetDevice(dst.device_id), "Select activation destination");
        check_cuda(cudaMemcpyPeerAsync(dst.residual, dst.device_id, src.residual, src.device_id,
                                      (size_t)tokens * HIDDEN * sizeof(__nv_bfloat16), dst.stream),
                   "Copy activation between devices");
    } else {
        check_cuda(cudaSetDevice(eng->devices[to]), "Select layer device");
    }
    return eng->ctx[to].residual;
}

static void forward_tokens(EngineHandle *eng, const int64_t *token_ids,
                           int tokens, float *h_logits) {
    int64_t positions[ENGINE_BATCH_TOKENS];
    for (int t = 0; t < tokens; ++t) positions[t] = (int64_t)eng->seq_len + t;
    for (int d = 0; d < eng->num_devices; ++d) {
        DeviceCtx &ctx = eng->ctx[d];
        check_cuda(cudaSetDevice(ctx.device_id), "Select position device");
        check_cuda(cudaMemcpyAsync(ctx.positions, positions, tokens * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, ctx.stream), "Upload positions");
        // Finish staging host arrays before any subsequent operation can throw.
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish position upload");
    }
    DeviceCtx &first = eng->ctx[0];
    check_cuda(cudaSetDevice(first.device_id), "Select embedding device");
    check_cuda(cudaMemcpyAsync(eng->token_ids, token_ids, tokens * sizeof(int64_t),
                               cudaMemcpyHostToDevice, first.stream), "Upload token IDs");
    check_cuda(cudaStreamSynchronize(first.stream), "Finish token upload");
    kernel_embedding(first.residual, eng->embed_w, eng->token_ids, HIDDEN, tokens, first.stream);
    check_cuda(cudaGetLastError(), "Embedding");

    int current = 0;
    for (int i = 0; i < MAX_LAYERS; ++i) {
        const int dev_idx = eng->layer_device[i];
        __nv_bfloat16 *act = move_activation(eng, current, dev_idx, tokens);
        current = dev_idx;
        DeviceCtx &ctx = eng->ctx[dev_idx];
        LayerWeights &lw = eng->layers[i];
        if (lw.is_attention) {
            AttentionWeights aw{};
            aw.q_proj_w = lw.q_proj_w;
            aw.k_proj_w = lw.k_proj_w;
            aw.v_proj_w = lw.v_proj_w;
            aw.o_proj_w = lw.o_proj_w;
            aw.q_norm_w = lw.q_norm_w;
            aw.k_norm_w = lw.k_norm_w;
            aw.input_norm_w = lw.input_norm_w;
            check_forward(forward_attention_layer(ctx.cublas, ctx.stream, act, ctx.workspace,
                          ctx.layer_out, &aw, lw.kv_cache, ctx.positions, tokens,
                          eng->seq_len + tokens, &eng->dims), "Attention layer");
        } else {
            GdnWeights gw{};
            gw.in_proj_qkv_w = lw.in_proj_qkv_w;
            gw.in_proj_z_w = lw.in_proj_z_w;
            gw.in_proj_a_w = lw.in_proj_a_w;
            gw.in_proj_b_w = lw.in_proj_b_w;
            gw.conv1d_w = lw.conv1d_w;
            gw.conv1d_bias = ctx.conv_bias_zero;
            gw.dt_bias = lw.dt_bias;
            gw.A_log = lw.A_log;
            gw.out_proj_w = lw.gdn_out_proj_w;
            gw.gdn_norm_w = lw.gdn_norm_f32;
            gw.input_norm_w = lw.input_norm_w;
            check_forward(forward_gdn_layer(ctx.cublas, ctx.stream, act, ctx.workspace,
                          ctx.layer_out, &gw, lw.conv_state, lw.ssm_state, ctx.fla_scratch,
                          tokens, &eng->dims), "GDN layer");
        }
        kernel_residual_add(act, ctx.layer_out, tokens * HIDDEN, ctx.stream);
        check_cuda(cudaGetLastError(), "Layer residual add");
        MlpWeights mw{lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w, lw.post_norm_w};
        check_forward(forward_mlp(ctx.cublas, ctx.stream, act, ctx.workspace, ctx.layer_out,
                      &mw, tokens, &eng->dims), "MLP");
        kernel_residual_add(act, ctx.layer_out, tokens * HIDDEN, ctx.stream);
        check_cuda(cudaGetLastError(), "MLP residual add");
    }

    if (h_logits) {
        const int last_idx = eng->num_devices - 1;
        __nv_bfloat16 *act = move_activation(eng, current, last_idx, tokens);
        current = last_idx;
        DeviceCtx &last = eng->ctx[last_idx];
        // Public API returns only the final row; avoid [128,vocab] logits.
        kernel_gemma_rms_norm(last.workspace, act + (size_t)(tokens - 1) * HIDDEN,
                              eng->final_norm_w, HIDDEN, 1, eng->dims.rms_eps, last.stream);
        check_cuda(cudaGetLastError(), "Final norm");
        check_forward(gemm_bf16_f32out(last.cublas, eng->d_logits, last.workspace,
                      eng->lm_head_w, 1, VOCAB, HIDDEN), "LM head");
        check_cuda(cudaMemcpyAsync(h_logits, eng->d_logits, VOCAB * sizeof(float),
                                   cudaMemcpyDeviceToHost, last.stream), "Download logits");
    }
    check_cuda(cudaSetDevice(eng->devices[current]), "Select final forward device");
    check_cuda(cudaStreamSynchronize(eng->ctx[current].stream), "Finish forward");
    eng->seq_len += tokens;
}

static int validate_inputs(const EngineHandle *eng, const int64_t *ids, int tokens,
                            const float *logits) {
    if (!eng || !eng->state_valid) {
        set_error("Engine is null or requires engine_reset after a failed forward");
        return ENGINE_ERR_STATE;
    }
    if (!ids || !logits || tokens < 1) {
        set_error("Token IDs/logits must be non-null and token count must be positive");
        return ENGINE_ERR_CONFIG;
    }
    if (tokens > eng->dims.max_seq_len - eng->seq_len) {
        set_error("Sequence capacity exceeded");
        return ENGINE_ERR_SEQ_FULL;
    }
    for (int t = 0; t < tokens; ++t) {
        if (ids[t] < 0 || ids[t] >= VOCAB) {
            set_error("Token ID at offset %d is outside [0, %d)", t, VOCAB);
            return ENGINE_ERR_CONFIG;
        }
    }
    return ENGINE_OK;
}

static int forward_error(EngineHandle *eng, const std::exception &e) {
    // A failed kernel may already have updated recurrent state; require explicit reset.
    if (eng) eng->state_valid = false;
    set_error("Forward failed: %s", e.what());
    if (const auto *error = dynamic_cast<const EngineError *>(&e)) return error->code;
    if (dynamic_cast<const std::bad_alloc *>(&e)) return ENGINE_ERR_ALLOC;
    return ENGINE_ERR_CUDA;
}

int engine_prefill(EngineHandle *eng, const int64_t *token_ids,
                   int num_tokens, float *out_logits) {
    g_error_buf[0] = '\0';
    try {
        int status = validate_inputs(eng, token_ids, num_tokens, out_logits);
        if (status != ENGINE_OK) return status;
        for (int offset = 0; offset < num_tokens;) {
            int tokens = std::min(ENGINE_BATCH_TOKENS, num_tokens - offset);
            forward_tokens(eng, token_ids + offset, tokens,
                           offset + tokens == num_tokens ? out_logits : nullptr);
            offset += tokens;
        }
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    } catch (...) {
        if (eng) eng->state_valid = false;
        set_error("Prefill failed: unknown failure");
        return ENGINE_ERR_CUDA;
    }
}

int engine_decode(EngineHandle *eng, int64_t token_id, float *out_logits) {
    g_error_buf[0] = '\0';
    try {
        int status = validate_inputs(eng, &token_id, 1, out_logits);
        if (status != ENGINE_OK) return status;
        forward_tokens(eng, &token_id, 1, out_logits);
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    } catch (...) {
        if (eng) eng->state_valid = false;
        set_error("Decode failed: unknown failure");
        return ENGINE_ERR_CUDA;
    }
}

void engine_reset(EngineHandle *eng) {
    g_error_buf[0] = '\0';
    if (!eng) return;
    eng->state_valid = false;
    try {
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            check_cuda(cudaSetDevice(ctx.device_id), "Select reset device");
            for (int i = 0; i < MAX_LAYERS; ++i) {
                if (eng->layer_device[i] != d) continue;
                LayerWeights &lw = eng->layers[i];
                if (lw.is_attention) {
                    check_cuda(cudaMemsetAsync(lw.kv_cache, 0, kv_cache_bytes(eng), ctx.stream), "Reset KV cache");
                } else {
                    check_cuda(cudaMemsetAsync(lw.conv_state, 0, conv_state_bytes(), ctx.stream), "Reset conv state");
                    check_cuda(cudaMemsetAsync(lw.ssm_state, 0, ssm_state_bytes(), ctx.stream), "Reset SSM state");
                }
            }
            check_cuda(cudaStreamSynchronize(ctx.stream), "Finish reset");
        }
        eng->seq_len = 0;
        eng->state_valid = true;
    } catch (const std::exception &e) {
        set_error("engine_reset: %s", e.what());
    } catch (...) {
        set_error("engine_reset: unknown failure");
    }
}

static void cleanup_cuda(cudaError_t status) {
    if (status != cudaSuccess && !g_error_buf[0])
        set_error("engine_destroy: %s", cudaGetErrorString(status));
}

void engine_destroy(EngineHandle *eng) {
    if (!eng) return;
    // Drain every stream before freeing buffers that peer copies may still read.
    if (eng->ctx) {
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            if (ctx.device_id < 0) continue;
            cleanup_cuda(cudaSetDevice(ctx.device_id));
            if (ctx.stream) cleanup_cuda(cudaStreamSynchronize(ctx.stream));
        }
    }
    if (eng->layers) {
        for (int i = 0; i < MAX_LAYERS; ++i) {
            cleanup_cuda(cudaSetDevice(eng->devices[eng->layer_device[i]]));
            LayerWeights &lw = eng->layers[i];
            void *buffers[] = {lw.input_norm_w, lw.post_norm_w, lw.gate_proj_w, lw.up_proj_w,
                lw.down_proj_w, lw.q_proj_w, lw.k_proj_w, lw.v_proj_w, lw.o_proj_w,
                lw.q_norm_w, lw.k_norm_w, lw.in_proj_qkv_w, lw.in_proj_z_w,
                lw.in_proj_a_w, lw.in_proj_b_w, lw.conv1d_w, lw.dt_bias, lw.A_log,
                lw.gdn_out_proj_w, lw.gdn_norm_w, lw.gdn_norm_f32,
                lw.kv_cache, lw.conv_state, lw.ssm_state};
            for (void *ptr : buffers) if (ptr) cleanup_cuda(cudaFree(ptr));
        }
    }
    if (eng->embed_w || eng->token_ids) {
        cleanup_cuda(cudaSetDevice(eng->devices.front()));
        if (eng->embed_w) cleanup_cuda(cudaFree(eng->embed_w));
        if (eng->token_ids) cleanup_cuda(cudaFree(eng->token_ids));
    }
    if (eng->lm_head_w || eng->final_norm_w || eng->d_logits) {
        cleanup_cuda(cudaSetDevice(eng->devices.back()));
        if (eng->lm_head_w) cleanup_cuda(cudaFree(eng->lm_head_w));
        if (eng->final_norm_w) cleanup_cuda(cudaFree(eng->final_norm_w));
        if (eng->d_logits) cleanup_cuda(cudaFree(eng->d_logits));
    }
    if (eng->ctx) {
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            if (ctx.device_id < 0) continue;
            cleanup_cuda(cudaSetDevice(ctx.device_id));
            void *buffers[] = {ctx.residual, ctx.layer_out, ctx.workspace, ctx.conv_bias_zero,
                               ctx.positions, ctx.fla_scratch};
            for (void *ptr : buffers) if (ptr) cleanup_cuda(cudaFree(ptr));
            if (ctx.cublas) {
                cublasStatus_t status = cublasDestroy(ctx.cublas);
                if (status != CUBLAS_STATUS_SUCCESS && !g_error_buf[0])
                    set_error("engine_destroy: cuBLAS status %d", (int)status);
            }
            if (ctx.stream) cleanup_cuda(cudaStreamDestroy(ctx.stream));
        }
    }
    delete[] eng->ctx;
    delete[] eng->layers;
    delete eng;
}

int engine_vocab_size(const EngineHandle *eng) { return eng ? eng->dims.vocab_size : 0; }
int engine_seq_len(const EngineHandle *eng) { return eng ? eng->seq_len : 0; }
