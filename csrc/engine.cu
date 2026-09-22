/** Weight loading and chunked, layer-partitioned multi-GPU inference. */
#include "engine.h"
#include "kernels.h"
#include "layers.h"
#include "model_desc.h"
#include "moe.h"
#include "flashinfer_ops.h"
#include "fla_ops.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
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

/* ------------------------------------------------------------------ */
/* Weight roles                                                       */
/* ------------------------------------------------------------------ */

/* Where a role's tensor lives: on the layer's own device (0), the first
 * configured device (1, embedding) or the last one (2, output head). The
 * convention is family-independent, so it lives here rather than in the
 * descriptor. */
enum RoleOwner { OWNER_LAYER = 0, OWNER_FIRST = 1, OWNER_LAST = 2 };

/* Expected tensor shape per role, in terms of the model dimensions. */
struct ExpectedShape {
    int ndim;
    long long dims[3];
};

static ExpectedShape expected_shape(int role, const struct ModelDesc &d) {
    const long long hidden = d.hidden_size;
    const long long q_rows = (long long)d.num_heads * d.head_dim *
                             (d.attn_output_gate ? 2 : 1);
    const long long kv_rows = (long long)d.num_kv_heads * d.head_dim;
    switch (role) {
    case ROLE_EMBED: case ROLE_LM_HEAD:
        return {2, {d.vocab_size, hidden, 0}};
    case ROLE_FINAL_NORM: case ROLE_INPUT_NORM: case ROLE_POST_NORM:
        return {1, {hidden, 0, 0}};
    case ROLE_MLP_GATE: case ROLE_MLP_UP:
        return {2, {d.intermediate_size, hidden, 0}};
    case ROLE_MLP_DOWN:
        return {2, {hidden, d.intermediate_size, 0}};
    case ROLE_ATTN_Q:
        return {2, {q_rows, hidden, 0}};
    case ROLE_ATTN_K: case ROLE_ATTN_V:
        return {2, {kv_rows, hidden, 0}};
    case ROLE_ATTN_O:
        return {2, {hidden, (long long)d.num_heads * d.head_dim, 0}};
    case ROLE_ATTN_Q_NORM: case ROLE_ATTN_K_NORM:
        return {1, {d.head_dim, 0, 0}};
    case ROLE_GDN_QKV:
        return {2, {d.gdn_conv_dim, hidden, 0}};
    case ROLE_GDN_Z:
        return {2, {d.gdn_value_dim, hidden, 0}};
    case ROLE_GDN_A: case ROLE_GDN_B:
        return {2, {d.gdn_num_v_heads, hidden, 0}};
    case ROLE_GDN_CONV1D:
        return {3, {d.gdn_conv_dim, 1, d.gdn_conv_kernel}};
    case ROLE_GDN_DT_BIAS: case ROLE_GDN_A_LOG:
        return {1, {d.gdn_num_v_heads, 0, 0}};
    case ROLE_GDN_OUT:
        return {2, {hidden, d.gdn_value_dim, 0}};
    case ROLE_GDN_NORM:
        return {1, {d.gdn_head_dim, 0, 0}};
    case ROLE_MOE_ROUTER:
        return {2, {d.moe_num_experts, hidden, 0}};
    case ROLE_MOE_EXPERT_GATE: case ROLE_MOE_EXPERT_UP:
        return {2, {d.moe_intermediate_size, hidden, 0}};
    case ROLE_MOE_EXPERT_DOWN:
        return {2, {hidden, d.moe_intermediate_size, 0}};
    default:
        return {0, {0, 0, 0}};
    }
}

static int role_owner(int role) {
    switch (role) {
    case ROLE_EMBED: return OWNER_FIRST;
    case ROLE_LM_HEAD: case ROLE_FINAL_NORM: return OWNER_LAST;
    default: return OWNER_LAYER;
    }
}

/* Roles a layer needs, given its mixer and feed-forward kinds. The feed-forward
 * roles are mutually exclusive: a dense layer has no router/expert tensors and an
 * MoE layer has no dense MLP tensors. */
static bool role_used_by_layer(int role, int mixer, int ffn) {
    if (role == ROLE_INPUT_NORM || role == ROLE_POST_NORM)
        return true;
    if (role == ROLE_MLP_GATE || role == ROLE_MLP_UP || role == ROLE_MLP_DOWN)
        return ffn == ENGINE_FFN_DENSE;
    if (role == ROLE_MOE_ROUTER || role == ROLE_MOE_EXPERT_GATE ||
        role == ROLE_MOE_EXPERT_UP || role == ROLE_MOE_EXPERT_DOWN)
        return ffn == ENGINE_FFN_MOE;
    if (mixer == ENGINE_MIXER_FULL_ATTN)
        return role == ROLE_ATTN_Q || role == ROLE_ATTN_K || role == ROLE_ATTN_V ||
               role == ROLE_ATTN_O || role == ROLE_ATTN_Q_NORM || role == ROLE_ATTN_K_NORM;
    if (mixer == ENGINE_MIXER_GDN)
        return role == ROLE_GDN_QKV || role == ROLE_GDN_Z || role == ROLE_GDN_A ||
               role == ROLE_GDN_B || role == ROLE_GDN_CONV1D || role == ROLE_GDN_DT_BIAS ||
               role == ROLE_GDN_A_LOG || role == ROLE_GDN_OUT || role == ROLE_GDN_NORM;
    return false;
}

struct DeviceCtx {
    int device_id = -1;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t copy_event = nullptr;   // orders cross-device copies
    __nv_bfloat16 *residual = nullptr;   // [128, hidden]
    __nv_bfloat16 *layer_out = nullptr;  // Never aliases layer/MLP scratch.
    __nv_bfloat16 *workspace = nullptr;
    __nv_bfloat16 *conv_bias_zero = nullptr;
    int64_t *positions = nullptr;
    void *fla_scratch = nullptr;
    void *moe_scratch = nullptr;
    size_t ws_size = 0;
    size_t fla_size = 0;
    size_t moe_ws_size = 0;
};

struct EngineHandle {
    ModelDims dims{};
    struct ModelDesc desc{};
    int num_layers = 0;      // layers actually allocated (0 before parsing succeeds)
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

static size_t kv_cache_bytes(const EngineHandle *eng) {
    return (size_t)2 * eng->dims.max_seq_len * eng->dims.num_kv_heads *
           eng->dims.head_dim * sizeof(__nv_bfloat16);
}

static size_t conv_state_bytes(const EngineHandle *eng) {
    return (size_t)eng->dims.gdn_conv_dim * (eng->dims.gdn_conv_kernel - 1) *
           sizeof(__nv_bfloat16);
}

static size_t ssm_state_bytes(const EngineHandle *eng) {
    return (size_t)eng->dims.gdn_num_v_heads * eng->dims.gdn_head_dim *
           eng->dims.gdn_head_dim * sizeof(float);
}

/* Load a tensor by role: expand the template, validate the shape, upload. */
static void load_role(const std::map<std::string, TensorInfo> &index,
                      const struct ModelDesc &desc, int role, int layer,
                      int device, __nv_bfloat16 **dst) {
    int slot = model_desc_role_index(&desc, role);
    if (slot < 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Descriptor has no template for role ") +
                              std::to_string(role));
    char name[ENGINE_TEMPLATE_MAX];
    model_desc_expand(desc.role_templates[slot], layer, 0, name, sizeof(name));
    auto it = index.find(name);
    if (it == index.end())
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Tensor not found: ") + name);
    const TensorInfo &ti = it->second;
    if (ti.dtype != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Tensor is not BF16: ") + name);
    ExpectedShape want = expected_shape(role, desc);
    bool shape_ok = ti.ndim == want.ndim;
    for (int i = 0; shape_ok && i < want.ndim; ++i) shape_ok = ti.shape[i] == want.dims[i];
    if (!shape_ok)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Unexpected shape for ") + name + " (" +
                              std::to_string(ti.ndim) + " dims)");
    long long elements = 1;
    for (int i = 0; i < ti.ndim; ++i) elements *= ti.shape[i];
    const size_t bytes = (size_t)elements * sizeof(__nv_bfloat16);
    check_cuda(cudaSetDevice(device), "Set weight device");
    check_cuda(cudaMalloc(dst, bytes), "Allocate weight");
    int status = safetensors_load_tensor(ti, *dst, device);
    check_cuda(cudaGetLastError(), "Upload weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Cannot load tensor: ") + name);
    // The loader uploads on the default stream; compute streams are nonblocking.
    check_cuda(cudaStreamSynchronize(nullptr), "Finish weight upload");
}

/* Load one expert's tensor straight into its slot of a fused [E, ...] buffer. */
static void load_expert_role(const std::map<std::string, TensorInfo> &index,
                             const struct ModelDesc &desc, int role, int layer,
                             int expert, int device, __nv_bfloat16 *dst) {
    int slot = model_desc_role_index(&desc, role);
    if (slot < 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Descriptor has no template for role ") +
                              std::to_string(role));
    char name[ENGINE_TEMPLATE_MAX];
    model_desc_expand(desc.role_templates[slot], layer, expert, name, sizeof(name));
    auto it = index.find(name);
    if (it == index.end())
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Tensor not found: ") + name);
    const TensorInfo &ti = it->second;
    if (ti.dtype != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Tensor is not BF16: ") + name);
    ExpectedShape want = expected_shape(role, desc);
    bool shape_ok = ti.ndim == want.ndim;
    for (int i = 0; shape_ok && i < want.ndim; ++i) shape_ok = ti.shape[i] == want.dims[i];
    if (!shape_ok)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Unexpected shape for ") + name);
    long long elements = 1;
    for (int i = 0; i < ti.ndim; ++i) elements *= ti.shape[i];
    check_cuda(cudaSetDevice(device), "Set expert weight device");
    int status = safetensors_load_tensor(ti, dst, device);
    check_cuda(cudaGetLastError(), "Upload expert weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Cannot load tensor: ") + name);
}

/* Field that receives a role's tensor for the given layer kind. */
static __nv_bfloat16 **role_target(LayerWeights &lw, int role) {
    switch (role) {
    case ROLE_INPUT_NORM: return &lw.input_norm_w;
    case ROLE_POST_NORM: return &lw.post_norm_w;
    case ROLE_MLP_GATE: return &lw.gate_proj_w;
    case ROLE_MLP_UP: return &lw.up_proj_w;
    case ROLE_MLP_DOWN: return &lw.down_proj_w;
    case ROLE_ATTN_Q: return &lw.q_proj_w;
    case ROLE_ATTN_K: return &lw.k_proj_w;
    case ROLE_ATTN_V: return &lw.v_proj_w;
    case ROLE_ATTN_O: return &lw.o_proj_w;
    case ROLE_ATTN_Q_NORM: return &lw.q_norm_w;
    case ROLE_ATTN_K_NORM: return &lw.k_norm_w;
    case ROLE_GDN_QKV: return &lw.in_proj_qkv_w;
    case ROLE_GDN_Z: return &lw.in_proj_z_w;
    case ROLE_GDN_A: return &lw.in_proj_a_w;
    case ROLE_GDN_B: return &lw.in_proj_b_w;
    case ROLE_GDN_CONV1D: return &lw.conv1d_w;
    case ROLE_GDN_DT_BIAS: return &lw.dt_bias;
    case ROLE_GDN_A_LOG: return &lw.A_log;
    case ROLE_GDN_OUT: return &lw.gdn_out_proj_w;
    case ROLE_GDN_NORM: return &lw.gdn_norm_w;
    default: return nullptr;
    }
}

/* Roles whose tensors are loaded once, outside the layer loop. */
struct GlobalRole {
    int role;
    __nv_bfloat16 **target;   /* points into EngineHandle */
};

static void fill_dims(ModelDims &dims, const struct ModelDesc &desc) {
    dims = ModelDims{};
    dims.hidden_size = desc.hidden_size;
    dims.intermediate_size = desc.intermediate_size;
    dims.num_heads = desc.num_heads;
    dims.num_kv_heads = desc.num_kv_heads;
    dims.head_dim = desc.head_dim;
    dims.rotary_dim = desc.rotary_dim;
    dims.norm_style = strcmp(desc.norm_style, "plain") == 0 ? 1 : 0;
    dims.attn_output_gate = desc.attn_output_gate;
    dims.q_gate_interleave = desc.q_gate_interleave;
    dims.num_layers = desc.num_layers;
    dims.vocab_size = desc.vocab_size;
    dims.rms_eps = (float)desc.rms_eps;
    dims.rope_theta = (float)desc.rotary_theta;
    dims.max_seq_len = desc.max_seq_len;
    dims.max_chunk = desc.max_chunk;
    dims.gdn_conv_dim = desc.gdn_conv_dim;
    dims.gdn_value_dim = desc.gdn_value_dim;
    dims.gdn_num_v_heads = desc.gdn_num_v_heads;
    dims.gdn_num_k_heads = desc.gdn_num_k_heads;
    dims.gdn_head_dim = desc.gdn_head_dim;
    dims.gdn_conv_kernel = desc.gdn_conv_kernel;
}

EngineHandle *engine_create(const char *model_dir, const char *descriptor_json,
                            int num_devices, const int *devices,
                            const int *layer_devices) {
    g_error_buf[0] = '\0';
    EngineHandle *eng = nullptr;
    char desc_error[256] = {0};
    try {
        if (!model_dir || !*model_dir || !descriptor_json || num_devices < 1 ||
            !devices || !layer_devices)
            throw EngineError(ENGINE_ERR_CONFIG, "Invalid engine configuration");

        eng = new EngineHandle();
        if (model_desc_parse(descriptor_json, &eng->desc, desc_error, sizeof(desc_error)) != 0)
            throw EngineError(ENGINE_ERR_CONFIG, std::string("Bad descriptor: ") + desc_error);
        fill_dims(eng->dims, eng->desc);
        eng->num_layers = eng->desc.num_layers;

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "Get device count");
        if (num_devices > device_count)
            throw EngineError(ENGINE_ERR_CONFIG, "Too many configured devices");
        eng->num_devices = num_devices;
        eng->devices.assign(devices, devices + num_devices);
        for (int d = 0; d < num_devices; ++d) {
            int ordinal = eng->devices[d];
            if (ordinal < 0 || ordinal >= device_count ||
                std::find(eng->devices.begin(), eng->devices.begin() + d, ordinal) !=
                    eng->devices.begin() + d)
                throw EngineError(ENGINE_ERR_CONFIG, "Invalid or duplicate CUDA device ordinal");
        }
        const int num_layers = eng->desc.num_layers;
        for (int i = 0; i < num_layers; ++i) {
            if (eng->desc.layer_mixers[i] == ENGINE_MIXER_MLA)
                throw EngineError(ENGINE_ERR_CONFIG,
                                  "mla mixer layers are not implemented in this build");
        }
        eng->layer_device.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            auto it = std::find(eng->devices.begin(), eng->devices.end(), layer_devices[i]);
            if (it == eng->devices.end())
                throw EngineError(ENGINE_ERR_CONFIG, "Layer assigned to an unconfigured device");
            eng->layer_device[i] = (int)(it - eng->devices.begin());
        }

        peer_probe_all(eng->devices.data(), eng->num_devices, 1);

        std::map<std::string, TensorInfo> index;
        int nt = safetensors_scan_dir(model_dir, index);
        if (nt <= 0)
            throw EngineError(ENGINE_ERR_WEIGHTS, std::string("No safetensors found in ") + model_dir);
        fprintf(stderr, "[engine] Scanned %d tensors from %s\n", nt, model_dir);

        const int hidden = eng->dims.hidden_size;
        eng->ctx = new DeviceCtx[eng->num_devices]();
        eng->layers = new LayerWeights[num_layers]();
        const int max_chunk = eng->dims.max_chunk;
        bool has_gdn = false;
        for (int i = 0; i < num_layers; ++i)
            has_gdn = has_gdn || eng->desc.layer_mixers[i] == ENGINE_MIXER_GDN;
        const size_t activation_bytes = (size_t)max_chunk * hidden * sizeof(__nv_bfloat16);
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            ctx.device_id = eng->devices[d];
            check_cuda(cudaSetDevice(ctx.device_id), "Initialize device");
            check_cuda(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "Create stream");
            check_cuda(cudaEventCreateWithFlags(&ctx.copy_event, cudaEventDisableTiming),
                       "Create copy event");
            check_cublas(cublasCreate(&ctx.cublas), "Create cuBLAS");
            check_cublas(cublasSetStream(ctx.cublas, ctx.stream), "Set cuBLAS stream");
            check_cuda(cudaMalloc(&ctx.residual, activation_bytes), "Allocate residual");
            check_cuda(cudaMalloc(&ctx.layer_out, activation_bytes), "Allocate layer output");
            ctx.ws_size = layer_workspace_size(max_chunk, &eng->dims);
            check_cuda(cudaMalloc(&ctx.workspace, ctx.ws_size), "Allocate layer workspace");
                    ctx.moe_scratch = nullptr;
            if (has_gdn) {
                ctx.fla_size = kernel_fla_workspace_size(max_chunk, eng->dims.gdn_num_v_heads);
                check_cuda(cudaMalloc(&ctx.fla_scratch, ctx.fla_size), "Allocate FLA scratch");
            }
            check_cuda(cudaMalloc(&ctx.positions, max_chunk * sizeof(int64_t)), "Allocate positions");
            if (eng->dims.gdn_conv_dim > 0) {
                check_cuda(cudaMalloc(&ctx.conv_bias_zero,
                                      eng->dims.gdn_conv_dim * sizeof(__nv_bfloat16)),
                           "Allocate conv bias");
                check_cuda(cudaMemsetAsync(ctx.conv_bias_zero, 0,
                                           eng->dims.gdn_conv_dim * sizeof(__nv_bfloat16),
                                           ctx.stream),
                           "Zero conv bias");
            }
        }

        load_role(index, eng->desc, ROLE_EMBED, 0, eng->devices.front(), &eng->embed_w);
        check_cuda(cudaMalloc(&eng->token_ids, max_chunk * sizeof(int64_t)), "Allocate token IDs");
        load_role(index, eng->desc, ROLE_LM_HEAD, 0, eng->devices.back(), &eng->lm_head_w);
        load_role(index, eng->desc, ROLE_FINAL_NORM, 0, eng->devices.back(), &eng->final_norm_w);
        check_cuda(cudaMalloc(&eng->d_logits, eng->dims.vocab_size * sizeof(float)), "Allocate logits");

        for (int i = 0; i < num_layers; ++i) {
            DeviceCtx &ctx = eng->ctx[eng->layer_device[i]];
            check_cuda(cudaSetDevice(ctx.device_id), "Load layer device");
            LayerWeights &lw = eng->layers[i];
            lw.plan.mixer = eng->desc.layer_mixers[i];
            lw.plan.ffn = eng->desc.layer_ffns[i];
            for (int role = 0; role < ROLE_COUNT; ++role) {
                if (!role_used_by_layer(role, lw.plan.mixer, lw.plan.ffn)) continue;
                __nv_bfloat16 **target = role_target(lw, role);
                if (target == nullptr) continue;
                load_role(index, eng->desc, role, i, ctx.device_id, target);
                lw.owned.push_back(*target);
            }
            if (lw.plan.ffn == ENGINE_FFN_MOE) {
                const int experts = eng->desc.moe_num_experts;
                const int inner = eng->desc.moe_intermediate_size;
                lw.moe_config = MoeConfig{experts, eng->desc.moe_top_k, inner,
                                          eng->desc.moe_norm_topk_prob,
                                          strcmp(eng->desc.moe_router_scoring, "sigmoid") == 0,
                                          (float)eng->desc.moe_routed_scaling_factor};
                __nv_bfloat16 *router = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
                load_role(index, eng->desc, ROLE_MOE_ROUTER, i, ctx.device_id, &router);
                const size_t gate_bytes = (size_t)experts * inner * hidden * sizeof(__nv_bfloat16);
                const size_t down_bytes = (size_t)experts * hidden * inner * sizeof(__nv_bfloat16);
                check_cuda(cudaMalloc(&gate, gate_bytes), "Allocate expert gate weights");
                check_cuda(cudaMalloc(&up, gate_bytes), "Allocate expert up weights");
                check_cuda(cudaMalloc(&down, down_bytes), "Allocate expert down weights");
                for (int e = 0; e < experts; ++e) {
                    load_expert_role(index, eng->desc, ROLE_MOE_EXPERT_GATE, i, e,
                                     ctx.device_id, gate + (size_t)e * inner * hidden);
                    load_expert_role(index, eng->desc, ROLE_MOE_EXPERT_UP, i, e,
                                     ctx.device_id, up + (size_t)e * inner * hidden);
                    load_expert_role(index, eng->desc, ROLE_MOE_EXPERT_DOWN, i, e,
                                     ctx.device_id, down + (size_t)e * hidden * inner);
                }
                check_cuda(cudaStreamSynchronize(nullptr), "Finish expert weight upload");
                lw.moe.post_norm_w = lw.post_norm_w;
                lw.moe.router_w = router;
                lw.moe.experts_gate = gate;
                lw.moe.experts_up = up;
                lw.moe.experts_down = down;
                lw.owned.push_back(router);
                lw.owned.push_back(gate);
                lw.owned.push_back(up);
                lw.owned.push_back(down);
                if (ctx.moe_scratch == nullptr) {
                    const size_t bytes = moe_workspace_size(max_chunk, &eng->dims, &lw.moe_config);
                    check_cuda(cudaMalloc(&ctx.moe_scratch, bytes), "Allocate MoE scratch");
                    ctx.moe_ws_size = bytes;
                }
            }
            if (lw.plan.mixer == ENGINE_MIXER_FULL_ATTN) {
                check_cuda(cudaMalloc(&lw.kv_cache, kv_cache_bytes(eng)), "Allocate KV cache");
                check_cuda(cudaMemsetAsync(lw.kv_cache, 0, kv_cache_bytes(eng), ctx.stream), "Zero KV cache");
                lw.owned.push_back(lw.kv_cache);
                lw.reset_zero.emplace_back(lw.kv_cache, kv_cache_bytes(eng));
            } else if (lw.plan.mixer == ENGINE_MIXER_GDN) {
                check_cuda(cudaMalloc(&lw.gdn_norm_f32, eng->dims.gdn_head_dim * sizeof(float)),
                           "Allocate GDN norm");
                kernel_cast_bf16_f32(lw.gdn_norm_f32, lw.gdn_norm_w, eng->dims.gdn_head_dim, ctx.stream);
                check_cuda(cudaGetLastError(), "Convert GDN norm");
                lw.owned.push_back(lw.gdn_norm_f32);
                check_cuda(cudaMalloc(&lw.conv_state, conv_state_bytes(eng)), "Allocate conv state");
                check_cuda(cudaMemsetAsync(lw.conv_state, 0, conv_state_bytes(eng), ctx.stream),
                           "Zero conv state");
                check_cuda(cudaMalloc(&lw.ssm_state, ssm_state_bytes(eng)), "Allocate SSM state");
                check_cuda(cudaMemsetAsync(lw.ssm_state, 0, ssm_state_bytes(eng), ctx.stream),
                           "Zero SSM state");
                lw.owned.push_back(lw.conv_state);
                lw.owned.push_back(lw.ssm_state);
                lw.reset_zero.emplace_back(lw.conv_state, conv_state_bytes(eng));
                lw.reset_zero.emplace_back(lw.ssm_state, ssm_state_bytes(eng));
            } else {
                throw EngineError(ENGINE_ERR_CONFIG,
                                  "Layer kind is not implemented in this build");
            }
            if (i % 16 == 0) fprintf(stderr, "[engine] Loaded layer %d/%d\n", i, num_layers);
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

int engine_desc_version(void) { return ENGINE_DESC_VERSION; }

int engine_describe(const EngineHandle *eng, char *buf, int buf_len) {
    if (!eng || !buf || buf_len <= 0) {
        set_error("engine_describe: invalid arguments");
        return ENGINE_ERR_CONFIG;
    }
    return model_desc_format(&eng->desc, buf, buf_len);
}

static __nv_bfloat16 *move_activation(EngineHandle *eng, int from, int to, int tokens) {
    if (from != to) {
        DeviceCtx &src = eng->ctx[from];
        DeviceCtx &dst = eng->ctx[to];
        check_cuda((cudaError_t)copy_across_devices(
                       src.device_id, src.stream, &src.copy_event,
                       dst.device_id, dst.stream,
                       dst.residual, src.residual,
                       (size_t)tokens * eng->dims.hidden_size * sizeof(__nv_bfloat16)),
                   "Copy activation between devices");
        check_cuda(cudaSetDevice(dst.device_id), "Select activation destination");
    } else {
        check_cuda(cudaSetDevice(eng->devices[to]), "Select layer device");
    }
    return eng->ctx[to].residual;
}

static void forward_tokens(EngineHandle *eng, const int64_t *token_ids,
                           int tokens, float *h_logits) {
    std::vector<int64_t> positions(eng->dims.max_chunk);
    for (int t = 0; t < tokens; ++t) positions[t] = (int64_t)eng->seq_len + t;
    for (int d = 0; d < eng->num_devices; ++d) {
        DeviceCtx &ctx = eng->ctx[d];
        check_cuda(cudaSetDevice(ctx.device_id), "Select position device");
        check_cuda(cudaMemcpyAsync(ctx.positions, positions.data(), tokens * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, ctx.stream), "Upload positions");
        // Finish staging host arrays before any subsequent operation can throw.
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish position upload");
    }
    DeviceCtx &first = eng->ctx[0];
    check_cuda(cudaSetDevice(first.device_id), "Select embedding device");
    check_cuda(cudaMemcpyAsync(eng->token_ids, token_ids, tokens * sizeof(int64_t),
                               cudaMemcpyHostToDevice, first.stream), "Upload token IDs");
    check_cuda(cudaStreamSynchronize(first.stream), "Finish token upload");
    kernel_embedding(first.residual, eng->embed_w, eng->token_ids, eng->dims.hidden_size,
                     tokens, first.stream);
    check_cuda(cudaGetLastError(), "Embedding");

    int current = 0;
    for (int i = 0; i < eng->num_layers; ++i) {
        const int dev_idx = eng->layer_device[i];
        __nv_bfloat16 *act = move_activation(eng, current, dev_idx, tokens);
        current = dev_idx;
        DeviceCtx &ctx = eng->ctx[dev_idx];
        LayerContext lctx;
        lctx.cublas = ctx.cublas;
        lctx.stream = ctx.stream;
        lctx.workspace = ctx.workspace;
        lctx.conv_bias_zero = ctx.conv_bias_zero;
        lctx.positions = ctx.positions;
        lctx.fla_scratch = ctx.fla_scratch;
        lctx.moe_scratch = ctx.moe_scratch;
        lctx.tokens = tokens;
        lctx.seq_len = eng->seq_len + tokens;
        lctx.layer_index = i;
        lctx.dims = &eng->dims;
        check_forward(forward_layer(&lctx, &eng->layers[i], act, ctx.layer_out), "Layer forward");
    }

    if (h_logits) {
        const int last_idx = eng->num_devices - 1;
        __nv_bfloat16 *act = move_activation(eng, current, last_idx, tokens);
        current = last_idx;
        DeviceCtx &last = eng->ctx[last_idx];
        // Public API returns only the final row; avoid [128,vocab] logits.
        __nv_bfloat16 *final_row = act + (size_t)(tokens - 1) * eng->dims.hidden_size;
        if (eng->dims.norm_style == 1) {
            kernel_rms_norm_plain(last.workspace, final_row, eng->final_norm_w,
                                  eng->dims.hidden_size, 1, eng->dims.rms_eps, last.stream);
        } else {
            kernel_gemma_rms_norm(last.workspace, final_row, eng->final_norm_w,
                                  eng->dims.hidden_size, 1, eng->dims.rms_eps, last.stream);
        }
        check_cuda(cudaGetLastError(), "Final norm");
        check_forward(gemm_bf16_f32out(last.cublas, eng->d_logits, last.workspace,
                      eng->lm_head_w, 1, eng->dims.vocab_size, eng->dims.hidden_size), "LM head");
        check_cuda(cudaMemcpyAsync(h_logits, eng->d_logits, eng->dims.vocab_size * sizeof(float),
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
        if (ids[t] < 0 || ids[t] >= eng->dims.vocab_size) {
            set_error("Token ID at offset %d is outside [0, %d)", t, eng->dims.vocab_size);
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
            int tokens = std::min(eng->dims.max_chunk, num_tokens - offset);
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
            for (int i = 0; i < eng->num_layers; ++i) {
                if (eng->layer_device[i] != d) continue;
                for (const auto &buffer : eng->layers[i].reset_zero) {
                    check_cuda(cudaMemsetAsync(buffer.first, 0, buffer.second, ctx.stream),
                               "Reset layer state");
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
        for (int i = 0; i < eng->num_layers; ++i) {
            cleanup_cuda(cudaSetDevice(eng->devices[eng->layer_device[i]]));
            for (void *ptr : eng->layers[i].owned)
                if (ptr) cleanup_cuda(cudaFree(ptr));
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
                               ctx.positions, ctx.fla_scratch, ctx.moe_scratch};
            for (void *ptr : buffers) if (ptr) cleanup_cuda(cudaFree(ptr));
            if (ctx.cublas) {
                cublasStatus_t status = cublasDestroy(ctx.cublas);
                if (status != CUBLAS_STATUS_SUCCESS && !g_error_buf[0])
                    set_error("engine_destroy: cuBLAS status %d", (int)status);
            }
            if (ctx.copy_event) cleanup_cuda(cudaEventDestroy(ctx.copy_event));
            if (ctx.stream) cleanup_cuda(cudaStreamDestroy(ctx.stream));
        }
    }
    delete[] eng->ctx;
    delete[] eng->layers;
    delete eng;
}

int engine_vocab_size(const EngineHandle *eng) { return eng ? eng->dims.vocab_size : 0; }
int engine_seq_len(const EngineHandle *eng) { return eng ? eng->seq_len : 0; }
