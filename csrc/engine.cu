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

/* Allocate a device buffer and hand it to its owner *immediately*.
 *
 * engine_destroy frees exactly what a layer registered, and everything a loader
 * does after the allocation (the upload, the next allocation, growing the list)
 * can throw; allocating first and registering later would leave every failure
 * path holding an unreachable buffer. The registration itself is the one step
 * that is not a CUDA call, so its failure frees the buffer on its own device and
 * rethrows. */
static void *alloc_owned(std::vector<void *> &owned, size_t bytes, int device,
                         const char *what) {
    check_cuda(cudaSetDevice(device), "Select weight device");
    void *pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, bytes), what);
    try {
        owned.push_back(pointer);
    } catch (...) {
        cudaSetDevice(device);
        cudaFree(pointer);
        throw;
    }
    return pointer;
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
    case ROLE_GDN_QKVZ:
        return {2, {d.gdn_conv_dim + d.gdn_value_dim, hidden, 0}};
    case ROLE_GDN_BA:
        return {2, {2 * d.gdn_num_v_heads, hidden, 0}};
    case ROLE_MOE_ROUTER:
        return {2, {d.moe_num_experts, hidden, 0}};
    case ROLE_MOE_EXPERT_GATE: case ROLE_MOE_EXPERT_UP:
        return {2, {d.moe_intermediate_size, hidden, 0}};
    case ROLE_MOE_EXPERT_DOWN:
        return {2, {hidden, d.moe_intermediate_size, 0}};
    case ROLE_MOE_SHARED_GATE: case ROLE_MOE_SHARED_UP:
        return {2, {d.moe_shared_intermediate_size, hidden, 0}};
    case ROLE_MOE_SHARED_DOWN:
        return {2, {hidden, d.moe_shared_intermediate_size, 0}};
    case ROLE_MOE_SHARED_GATE_SCALAR:
        return {2, {1, hidden, 0}};
    case ROLE_MLA_Q:
        return {2, {(long long)d.num_heads *
                        (d.mla_qk_nope_head_dim + d.mla_qk_rope_head_dim), hidden, 0}};
    case ROLE_MLA_KV_A:
        return {2, {d.mla_kv_lora_rank + d.mla_qk_rope_head_dim, hidden, 0}};
    case ROLE_MLA_KV_A_NORM:
        return {1, {d.mla_kv_lora_rank, 0, 0}};
    case ROLE_MLA_KV_B:
        return {2, {(long long)d.num_heads * (d.mla_qk_nope_head_dim + d.mla_v_head_dim),
                    d.mla_kv_lora_rank, 0}};
    case ROLE_MLA_O:
        return {2, {hidden, (long long)d.num_heads * d.mla_v_head_dim, 0}};
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
    if (mixer == ENGINE_MIXER_MLA)
        return role == ROLE_MLA_Q || role == ROLE_MLA_KV_A || role == ROLE_MLA_KV_A_NORM ||
               role == ROLE_MLA_KV_B || role == ROLE_MLA_O;
    return false;
}

struct DeviceCtx {
    int device_id = -1;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t copy_event = nullptr;   // orders cross-device copies
    cudaEvent_t read_done_event = nullptr;  // this rank finished reading the leader's buffer
    __nv_bfloat16 *residual = nullptr;   // [max_chunk, hidden]
    __nv_bfloat16 *layer_out = nullptr;  // Never aliases layer/MLP scratch.
    __nv_bfloat16 *workspace = nullptr;
    __nv_bfloat16 *embed_w = nullptr;       // replicated: one copy per rank
    __nv_bfloat16 *lm_head_w = nullptr;     // logits device only
    __nv_bfloat16 *final_norm_w = nullptr;  // logits device only
    float *d_logits = nullptr;              // logits device only
    int64_t *token_ids = nullptr;
    __nv_bfloat16 *conv_bias_zero = nullptr;
    int64_t *positions = nullptr;
    void *fla_scratch = nullptr;
    void *moe_scratch = nullptr;
    void *mla_scratch = nullptr;
    __nv_bfloat16 *reduce_staging = nullptr;  // rank-0 all-reduce scratch
    size_t ws_size = 0;
    size_t fla_size = 0;
    size_t moe_ws_size = 0;
    size_t mla_size = 0;
};

struct EngineHandle {
    ModelDims dims{};            // full (checkpoint) dimensions
    ModelDims local_dims{};      // rank dimensions: heads and the dense MLP are
                                 // divided by tp_size; everything else is global
    struct ModelDesc desc{};
    int num_layers = 0;      // layers actually allocated (0 before parsing succeeds)
    int num_devices = 0;
    bool replicated = false; // tp_size > 1 || ep_size > 1: every rank holds every layer
    int tp_size = 1;
    int ep_size = 1;         // > 1: experts are split across the ranks
    bool expert_parallel = false;
    std::vector<int> devices;
    std::vector<int> layer_device;      // Internal indices, not CUDA ordinals.
    DeviceCtx *ctx = nullptr;
    LayerWeights *layers = nullptr;     // num_layers entries, or num_layers * ranks
                                        // when replicated (layer-major)
    int seq_len = 0;
    bool state_valid = true;
    /* Debug taps: INFER_TAP_LAYERS / INFER_TAP_DIR, see tap.cu. */
    TapConfig taps;
};

/* Weights of `layer` on device index `dev` (rank index when replicated). */
static LayerWeights &layer_weights(EngineHandle *eng, int layer, int dev) {
    return eng->replicated ? eng->layers[layer * eng->num_devices + dev]
                           : eng->layers[layer];
}

static size_t kv_cache_bytes(const ModelDims &dims) {
    return (size_t)2 * dims.max_seq_len * dims.num_kv_heads *
           dims.head_dim * sizeof(__nv_bfloat16);
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
static const TensorInfo &find_role(const std::map<std::string, TensorInfo> &index,
                                   const struct ModelDesc &desc, int role, int layer) {
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
    return ti;
}

/* `owned` is the layer's ownership list when the buffer belongs to a layer, and
 * null for a device-context buffer (the engine frees those through their own
 * ctx fields). */
static void load_role(const std::map<std::string, TensorInfo> &index,
                      const struct ModelDesc &desc, int role, int layer,
                      int device, __nv_bfloat16 **dst,
                      std::vector<void *> *owned = nullptr) {
    const TensorInfo &ti = find_role(index, desc, role, layer);
    long long elements = 1;
    for (int i = 0; i < ti.ndim; ++i) elements *= ti.shape[i];
    const size_t bytes = (size_t)elements * sizeof(__nv_bfloat16);
    if (owned != nullptr) {
        *dst = (__nv_bfloat16 *)alloc_owned(*owned, bytes, device, "Allocate weight");
    } else {
        check_cuda(cudaSetDevice(device), "Set weight device");
        check_cuda(cudaMalloc(dst, bytes), "Allocate weight");
    }
    int status = safetensors_load_tensor(ti, *dst, (int64_t)bytes, device);
    check_cuda(cudaGetLastError(), "Upload weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Cannot load tensor: ") + ti.name + ": " +
                              safetensors_last_error());
    // The loader uploads on the default stream; compute streams are nonblocking.
    check_cuda(cudaStreamSynchronize(nullptr), "Finish weight upload");
}

/* Load a role honouring the descriptor's shard rule: this rank keeps only its
 * slice of the checkpoint tensor. `sharded` reports whether the rule actually
 * split the tensor -- the sublayer then all-reduces its output across ranks. */
static void load_role_view(const std::map<std::string, TensorInfo> &index,
                           const struct ModelDesc &desc, int role, int layer,
                           int device, int rank, __nv_bfloat16 **dst, bool *sharded,
                           std::vector<void *> *owned) {
    *sharded = false;
    const int slot = model_desc_role_index(&desc, role);
    const int rule = slot >= 0 && slot < desc.role_shard_count
                         ? desc.role_shards[slot] : ENGINE_SHARD_NONE;
    if (desc.tp_size <= 1 || rule == ENGINE_SHARD_NONE) {
        load_role(index, desc, role, layer, device, dst, owned);
        return;
    }
    const TensorInfo &ti = find_role(index, desc, role, layer);
    if (ti.ndim != 2)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Cannot shard a non-2-D tensor: ") + ti.name);
    struct ShardView view;
    char err[256] = {0};
    if (model_desc_shard_view(&desc, role, ti.shape[0], ti.shape[1], rank, &view, err,
                              sizeof(err)) != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Shard view: ") + err);
    const size_t shard_bytes = (size_t)(view.rows * view.cols) * sizeof(__nv_bfloat16);
    *dst = (__nv_bfloat16 *)alloc_owned(*owned, shard_bytes, device, "Allocate weight shard");
    int status = safetensors_load_tensor_slice(ti, *dst, (int64_t)shard_bytes,
                                               device, view.row_off, view.rows,
                                               view.col_off, view.cols);
    check_cuda(cudaGetLastError(), "Upload weight shard");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Cannot load tensor slice: ") + ti.name + ": " +
                              safetensors_last_error());
    check_cuda(cudaStreamSynchronize(nullptr), "Finish weight upload");
    *sharded = true;
}

/* Load a role's rows gathered into the order `order` names (see the fused GDN
 * layout below). The destination holds exactly order.size() rows. */
static void load_role_rows(const std::map<std::string, TensorInfo> &index,
                           const struct ModelDesc &desc, int role, int layer, int device,
                           const std::vector<int> &order, __nv_bfloat16 **dst,
                           std::vector<void *> &owned) {
    const TensorInfo &ti = find_role(index, desc, role, layer);
    const long long row_bytes = (long long)ti.shape[ti.ndim - 1] * sizeof(__nv_bfloat16);
    const long long rows = (long long)order.size();
    *dst = (__nv_bfloat16 *)alloc_owned(owned, (size_t)(rows * row_bytes), device,
                                        "Allocate weight");
    int status = safetensors_load_tensor_rows(ti, *dst, (int64_t)(rows * row_bytes), device,
                                              order.data(), rows, row_bytes);
    check_cuda(cudaGetLastError(), "Upload weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Cannot load tensor: ") + ti.name + ": " +
                              safetensors_last_error());
    check_cuda(cudaStreamSynchronize(nullptr), "Finish weight upload");
}

/* Row orders that turn a fused qkvz/ba checkpoint into the views the kernels
 * want. The reference derives its q/k/v/z from the fused rows with
 * `fix_query_key_value_ordering`: each key head owns a [q, k, v, z] block, with
 * the block's v/z split over its `repeat = num_v_heads / num_k_heads` value
 * heads, and in_proj_ba holds [b, a] pairs per key head. The kernels want plain
 * [q | k | v] plus [z] (and b/a) blocks in value-head order, so the rows are
 * gathered while loading. */
static void fused_gdn_row_orders(const ModelDims &dims, std::vector<int> &qkv_order,
                                 std::vector<int> &z_order, std::vector<int> &b_order,
                                 std::vector<int> &a_order) {
    const int key_heads = dims.gdn_num_k_heads;
    const int repeat = dims.gdn_num_v_heads / key_heads;
    const int head_dim = dims.gdn_head_dim;
    const int group = 2 * head_dim + 2 * repeat * head_dim;
    qkv_order.clear();
    z_order.clear();
    b_order.clear();
    a_order.clear();
    for (int head = 0; head < key_heads; ++head)
        for (int d = 0; d < head_dim; ++d)
            qkv_order.push_back(head * group + d);                                    // all q
    for (int head = 0; head < key_heads; ++head)
        for (int d = 0; d < head_dim; ++d)
            qkv_order.push_back(head * group + head_dim + d);                         // all k
    for (int head = 0; head < key_heads; ++head)
        for (int d = 0; d < repeat * head_dim; ++d)
            qkv_order.push_back(head * group + 2 * head_dim + d);                     // all v
    for (int head = 0; head < key_heads; ++head) {
        const int base = head * group;
        for (int d = 0; d < repeat * head_dim; ++d)
            z_order.push_back(base + 2 * head_dim + repeat * head_dim + d);            // z
        for (int r = 0; r < repeat; ++r) b_order.push_back(head * 2 * repeat + r);
        for (int r = 0; r < repeat; ++r) a_order.push_back(head * 2 * repeat + repeat + r);
    }
}

/* Load one expert's tensor straight into its slot of a fused [E, ...] buffer. */
static void load_expert_role(const std::map<std::string, TensorInfo> &index,
                             const struct ModelDesc &desc, int role, int layer,
                             int expert, int device, __nv_bfloat16 *dst) {
    /* The slot already belongs to a registered fused buffer, so nothing is
     * allocated (or registered) here. */
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
    /* The slot is the caller's slice of the fused expert buffer; its capacity is
     * exactly this tensor's byte count. */
    int status = safetensors_load_tensor(ti, dst,
                                        (int64_t)elements * (int64_t)sizeof(__nv_bfloat16),
                                        device);
    check_cuda(cudaGetLastError(), "Upload expert weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Cannot load tensor: ") + name + ": " +
                              safetensors_last_error());
}

/* MoE weights of one layer on one device. Under expert parallelism the rank
 * loads only its slice of the expert tensors ([expert_offset, expert_offset +
 * local_experts)); the router and the shared experts stay whole. */
static void load_moe_weights(const std::map<std::string, TensorInfo> &index,
                             EngineHandle *eng, LayerWeights &lw, int layer, int device,
                             int expert_offset, int local_experts) {
    const int hidden = eng->dims.hidden_size;
    const int experts = eng->desc.moe_num_experts;
    const int inner = eng->desc.moe_intermediate_size;
    const int shared = eng->desc.moe_num_shared_experts;
    lw.moe_config = MoeConfig{experts, eng->desc.moe_top_k, inner,
                              eng->desc.moe_norm_topk_prob,
                              strcmp(eng->desc.moe_router_scoring, "sigmoid") == 0,
                              (float)eng->desc.moe_routed_scaling_factor,
                              shared, eng->desc.moe_shared_intermediate_size,
                              eng->desc.moe_shared_gate_scalar,
                              expert_offset, local_experts};
    __nv_bfloat16 *router = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
    load_role(index, eng->desc, ROLE_MOE_ROUTER, layer, device, &router, &lw.owned);
    const size_t gate_bytes = (size_t)local_experts * inner * hidden * sizeof(__nv_bfloat16);
    const size_t down_bytes = (size_t)local_experts * hidden * inner * sizeof(__nv_bfloat16);
    /* Registered as they are allocated: the expert loop below loads hundreds of
     * tensors and any of them can fail. */
    gate = (__nv_bfloat16 *)alloc_owned(lw.owned, gate_bytes, device, "Allocate expert gate weights");
    up = (__nv_bfloat16 *)alloc_owned(lw.owned, gate_bytes, device, "Allocate expert up weights");
    down = (__nv_bfloat16 *)alloc_owned(lw.owned, down_bytes, device, "Allocate expert down weights");
    for (int e = 0; e < local_experts; ++e) {
        load_expert_role(index, eng->desc, ROLE_MOE_EXPERT_GATE, layer, expert_offset + e,
                         device, gate + (size_t)e * inner * hidden);
        load_expert_role(index, eng->desc, ROLE_MOE_EXPERT_UP, layer, expert_offset + e,
                         device, up + (size_t)e * inner * hidden);
        load_expert_role(index, eng->desc, ROLE_MOE_EXPERT_DOWN, layer, expert_offset + e,
                         device, down + (size_t)e * hidden * inner);
    }
    check_cuda(cudaStreamSynchronize(nullptr), "Finish expert weight upload");
    if (shared > 0) {
        const int shared_inner = eng->desc.moe_shared_intermediate_size;
        const size_t shared_gate_bytes =
            (size_t)shared * shared_inner * hidden * sizeof(__nv_bfloat16);
        const size_t shared_down_bytes =
            (size_t)shared * hidden * shared_inner * sizeof(__nv_bfloat16);
        __nv_bfloat16 *sgate = (__nv_bfloat16 *)alloc_owned(lw.owned, shared_gate_bytes,
                                                            device, "Allocate shared gate");
        __nv_bfloat16 *sup = (__nv_bfloat16 *)alloc_owned(lw.owned, shared_gate_bytes,
                                                          device, "Allocate shared up");
        __nv_bfloat16 *sdown = (__nv_bfloat16 *)alloc_owned(lw.owned, shared_down_bytes,
                                                            device, "Allocate shared down");
        for (int e = 0; e < shared; ++e) {
            load_expert_role(index, eng->desc, ROLE_MOE_SHARED_GATE, layer, e, device,
                             sgate + (size_t)e * shared_inner * hidden);
            load_expert_role(index, eng->desc, ROLE_MOE_SHARED_UP, layer, e, device,
                             sup + (size_t)e * shared_inner * hidden);
            load_expert_role(index, eng->desc, ROLE_MOE_SHARED_DOWN, layer, e, device,
                             sdown + (size_t)e * hidden * shared_inner);
        }
        lw.moe.shared_gate = sgate;
        lw.moe.shared_up = sup;
        lw.moe.shared_down = sdown;
        if (eng->desc.moe_shared_gate_scalar) {
            __nv_bfloat16 *scalar_w = nullptr;
            load_role(index, eng->desc, ROLE_MOE_SHARED_GATE_SCALAR, layer, device, &scalar_w,
                      &lw.owned);
            lw.moe.shared_gate_scalar_w = scalar_w;
        }
    }
    lw.moe.post_norm_w = lw.post_norm_w;
    lw.moe.router_w = router;
    lw.moe.experts_gate = gate;
    lw.moe.experts_up = up;
    lw.moe.experts_down = down;
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
    case ROLE_MLA_Q: return &lw.mla_q_proj_w;
    case ROLE_MLA_KV_A: return &lw.mla_kv_a_proj_w;
    case ROLE_MLA_KV_A_NORM: return &lw.mla_kv_a_norm_w;
    case ROLE_MLA_KV_B: return &lw.mla_kv_b_proj_w;
    case ROLE_MLA_O: return &lw.mla_o_proj_w;
    /* The fused layout has no single destination field; it is expanded below. */
    case ROLE_GDN_QKVZ: case ROLE_GDN_BA: return nullptr;
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
    dims.mla_kv_lora_rank = desc.mla_kv_lora_rank;
    dims.mla_qk_nope_head_dim = desc.mla_qk_nope_head_dim;
    dims.mla_qk_rope_head_dim = desc.mla_qk_rope_head_dim;
    dims.mla_v_head_dim = desc.mla_v_head_dim;
}

/* Rank dimensions: the sharded roles divide the attention heads and the dense
 * MLP; everything else (GDN, norms, embeddings) stays whole. */
static void fill_local_dims(ModelDims &dims, const ModelDims &global, int tp) {
    dims = global;
    if (tp > 1) {
        dims.num_heads = global.num_heads / tp;
        dims.num_kv_heads = global.num_kv_heads / tp;
        dims.intermediate_size = global.intermediate_size / tp;
    }
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
        tap_parse_env(&eng->taps);
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
        /* tp_size > 1 (tensor parallel) or ep_size > 1 (expert parallel) select
         * the replicated policy: every rank holds every layer with its shard of
         * the weights, all activations are replicated and a sublayer whose
         * weights were split all-reduces its output. */
        eng->tp_size = eng->desc.tp_size > 1 ? eng->desc.tp_size : 1;
        eng->ep_size = eng->desc.ep_size > 1 ? eng->desc.ep_size : 1;
        eng->expert_parallel = eng->ep_size > 1;
        eng->replicated = eng->desc.tp_size > 1 || eng->desc.ep_size > 1;
        if (eng->replicated) {
            if (eng->desc.tp_rank != 0 || eng->desc.ep_rank != 0)
                throw EngineError(ENGINE_ERR_CONFIG,
                                  "this engine process holds every rank; tp_rank and ep_rank must be 0");
            if (num_devices != eng->tp_size * eng->ep_size)
                throw EngineError(ENGINE_ERR_CONFIG,
                                  "replicated placement needs exactly tp_size * ep_size devices");
            if (eng->tp_size > 1 && eng->expert_parallel)
                throw EngineError(ENGINE_ERR_CONFIG,
                                  "combined tensor and expert parallelism is not implemented");
            if (eng->dims.num_heads % eng->tp_size != 0 ||
                eng->dims.num_kv_heads % eng->tp_size != 0)
                throw EngineError(ENGINE_ERR_CONFIG, "tp_size does not divide the attention heads");
            if (eng->dims.intermediate_size % eng->tp_size != 0)
                throw EngineError(ENGINE_ERR_CONFIG, "tp_size does not divide intermediate_size");
            for (int i = 0; i < num_layers; ++i) {
                if (eng->desc.layer_ffns[i] == ENGINE_FFN_MOE && eng->tp_size > 1)
                    throw EngineError(ENGINE_ERR_CONFIG,
                                      "MoE layers under tensor parallel are not implemented (use ep_size)");
            }
            if (eng->expert_parallel) {
                if (eng->desc.moe_num_experts <= 0)
                    throw EngineError(ENGINE_ERR_CONFIG, "ep_size > 1 needs MoE layers");
                if (eng->desc.moe_num_experts % eng->ep_size != 0)
                    throw EngineError(ENGINE_ERR_CONFIG, "ep_size does not divide the expert count");
                const int expert_roles[] = {ROLE_MOE_EXPERT_GATE, ROLE_MOE_EXPERT_UP,
                                            ROLE_MOE_EXPERT_DOWN};
                for (size_t r = 0; r < sizeof(expert_roles) / sizeof(expert_roles[0]); ++r) {
                    const int slot = model_desc_role_index(&eng->desc, expert_roles[r]);
                    if (slot >= 0 && eng->desc.role_shards[slot] != ENGINE_SHARD_OUT_EXPERTS)
                        throw EngineError(ENGINE_ERR_CONFIG,
                                          "expert roles must carry the out_experts rule when ep_size > 1");
                }
            }
            fill_local_dims(eng->local_dims, eng->dims, eng->tp_size);
        } else {
            eng->local_dims = eng->dims;
        }
        eng->layer_device.resize(num_layers);
        if (eng->replicated) {
            /* Every device runs every layer; the per-layer owner array is unused. */
            for (int i = 0; i < num_layers; ++i) eng->layer_device[i] = 0;
        } else {
            for (int i = 0; i < num_layers; ++i) {
                auto it = std::find(eng->devices.begin(), eng->devices.end(), layer_devices[i]);
                if (it == eng->devices.end())
                    throw EngineError(ENGINE_ERR_CONFIG, "Layer assigned to an unconfigured device");
                eng->layer_device[i] = (int)(it - eng->devices.begin());
            }
        }

        peer_probe_all(eng->devices.data(), eng->num_devices, 1);

        std::map<std::string, TensorInfo> index;
        if (safetensors_scan_dir(model_dir, index) != 0)
            throw EngineError(ENGINE_ERR_WEIGHTS,
                              std::string("Cannot read weights from ") + model_dir + ": " +
                                  safetensors_last_error());
        fprintf(stderr, "[engine] Scanned %zu tensors from %s\n", index.size(), model_dir);

        const int hidden = eng->dims.hidden_size;
        eng->ctx = new DeviceCtx[eng->num_devices]();
        const int layer_slots = eng->replicated ? num_layers * eng->num_devices : num_layers;
        eng->layers = new LayerWeights[layer_slots]();
        const int max_chunk = eng->dims.max_chunk;
        bool has_gdn = false, has_mla = false;
        for (int i = 0; i < num_layers; ++i) {
            has_gdn = has_gdn || eng->desc.layer_mixers[i] == ENGINE_MIXER_GDN;
            has_mla = has_mla || eng->desc.layer_mixers[i] == ENGINE_MIXER_MLA;
        }
        /* Refuse an MLA cache the attention kernel could never decode, before
         * the per-layer caches are allocated. The kernel's shared-memory budget,
         * not the model's position limit, is what bounds this. */
        if (has_mla) {
            for (int d = 0; d < eng->num_devices; ++d) {
                check_cuda(cudaSetDevice(eng->devices[d]), "Select device for the MLA limit");
                const int limit = kernel_mla_max_seq_len();
                if (eng->dims.max_seq_len > limit)
                    throw EngineError(ENGINE_ERR_CONFIG,
                                      "max_seq_len " + std::to_string(eng->dims.max_seq_len) +
                                          " exceeds the MLA attention kernel's "
                                          "shared-memory limit of " +
                                          std::to_string(limit) + " tokens on device " +
                                          std::to_string(eng->devices[d]));
            }
            check_cuda(cudaSetDevice(eng->devices[0]), "Restore the first device");
        }
        const size_t activation_bytes = (size_t)max_chunk * hidden * sizeof(__nv_bfloat16);
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            ctx.device_id = eng->devices[d];
            check_cuda(cudaSetDevice(ctx.device_id), "Initialize device");
            check_cuda(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "Create stream");
            check_cuda(cudaEventCreateWithFlags(&ctx.copy_event, cudaEventDisableTiming),
                       "Create copy event");
            check_cuda(cudaEventCreateWithFlags(&ctx.read_done_event, cudaEventDisableTiming),
                       "Create read-completion event");
            check_cublas(cublasCreate(&ctx.cublas), "Create cuBLAS");
            check_cublas(cublasSetStream(ctx.cublas, ctx.stream), "Set cuBLAS stream");
            check_cuda(cudaMalloc(&ctx.residual, activation_bytes), "Allocate residual");
            check_cuda(cudaMalloc(&ctx.layer_out, activation_bytes), "Allocate layer output");
            ctx.ws_size = layer_workspace_size(max_chunk, &eng->local_dims);
            check_cuda(cudaMalloc(&ctx.workspace, ctx.ws_size), "Allocate layer workspace");
            check_cuda(cudaMalloc(&ctx.token_ids, max_chunk * sizeof(int64_t)),
                       "Allocate token IDs");
            if (eng->replicated && d == 0) {
                check_cuda(cudaMalloc(&ctx.reduce_staging, activation_bytes),
                           "Allocate reduce staging");
            }
                    ctx.moe_scratch = nullptr;
            if (has_gdn) {
                ctx.fla_size = kernel_fla_workspace_size(max_chunk, eng->dims.gdn_num_v_heads);
                check_cuda(cudaMalloc(&ctx.fla_scratch, ctx.fla_size), "Allocate FLA scratch");
            }
            if (has_mla) {
                ctx.mla_size = kernel_mla_scratch_size(eng->dims.max_seq_len, &eng->dims);
                check_cuda(cudaMalloc(&ctx.mla_scratch, ctx.mla_size), "Allocate MLA scratch");
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

        /* The embedding is replicated on every rank; the logits side of the
         * model (final norm, LM head, logits buffer) stays on the last device. */
        for (int d = 0; d < eng->num_devices; ++d) {
            if (!eng->replicated && d > 0) break;
            load_role(index, eng->desc, ROLE_EMBED, 0, eng->devices[d], &eng->ctx[d].embed_w);
        }
        {
            DeviceCtx &last = eng->ctx[eng->num_devices - 1];
            load_role(index, eng->desc, ROLE_LM_HEAD, 0, last.device_id, &last.lm_head_w);
            load_role(index, eng->desc, ROLE_FINAL_NORM, 0, last.device_id, &last.final_norm_w);
            check_cuda(cudaMalloc(&last.d_logits, eng->dims.vocab_size * sizeof(float)),
                       "Allocate logits");
        }

        const int ranks = eng->replicated ? eng->num_devices : 1;
        for (int i = 0; i < num_layers; ++i) {
            for (int r = 0; r < ranks; ++r) {
                const int dev_idx = eng->replicated ? r : eng->layer_device[i];
                DeviceCtx &ctx = eng->ctx[dev_idx];
                check_cuda(cudaSetDevice(ctx.device_id), "Load layer device");
                LayerWeights &lw = layer_weights(eng, i, dev_idx);
                lw.plan.mixer = eng->desc.layer_mixers[i];
                lw.plan.ffn = eng->desc.layer_ffns[i];
                bool mixer_sharded = false, ffn_sharded = false;
                for (int role = 0; role < ROLE_COUNT; ++role) {
                    if (!role_used_by_layer(role, lw.plan.mixer, lw.plan.ffn)) continue;
                    if (model_desc_role_index(&eng->desc, role) < 0) continue;
                    __nv_bfloat16 **target = role_target(lw, role);
                    if (target == nullptr) continue;
                    bool sharded = false;
                    /* The buffer is registered with the layer by the loader
                     * itself, as soon as it is allocated. */
                    load_role_view(index, eng->desc, role, i, ctx.device_id, r, target,
                                   &sharded, &lw.owned);
                    if (sharded) {
                        if (role == ROLE_ATTN_Q || role == ROLE_ATTN_K ||
                            role == ROLE_ATTN_V || role == ROLE_ATTN_O)
                            mixer_sharded = true;
                        else if (role == ROLE_MLP_GATE || role == ROLE_MLP_UP ||
                                 role == ROLE_MLP_DOWN)
                            ffn_sharded = true;
                    }
                }
                lw.mixer_sharded = mixer_sharded;
                lw.ffn_sharded = ffn_sharded;
                if (lw.plan.ffn == ENGINE_FFN_MOE) {
                    const int local_experts = eng->expert_parallel
                        ? eng->desc.moe_num_experts / eng->ep_size
                        : eng->desc.moe_num_experts;
                    const int expert_offset = eng->expert_parallel ? r * local_experts : 0;
                    load_moe_weights(index, eng, lw, i, ctx.device_id, expert_offset,
                                     local_experts);
                    if (ctx.moe_scratch == nullptr) {
                        const size_t bytes = moe_workspace_size(max_chunk, &eng->local_dims,
                                                                &lw.moe_config);
                        check_cuda(cudaMalloc(&ctx.moe_scratch, bytes), "Allocate MoE scratch");
                        ctx.moe_ws_size = bytes;
                    }
                }
                /* A fused qkvz/ba checkpoint stores the reference's own row grouping,
                 * so its rows are gathered into the kernels' contiguous views. */
                const int qkvz_slot = model_desc_role_index(&eng->desc, ROLE_GDN_QKVZ);
                if (qkvz_slot >= 0 && lw.plan.mixer == ENGINE_MIXER_GDN) {
                    std::vector<int> qkv_order, z_order, b_order, a_order;
                    fused_gdn_row_orders(eng->dims, qkv_order, z_order, b_order, a_order);
                    load_role_rows(index, eng->desc, ROLE_GDN_QKVZ, i, ctx.device_id, qkv_order,
                                   &lw.in_proj_qkv_w, lw.owned);
                    load_role_rows(index, eng->desc, ROLE_GDN_QKVZ, i, ctx.device_id, z_order,
                                   &lw.in_proj_z_w, lw.owned);
                    load_role_rows(index, eng->desc, ROLE_GDN_BA, i, ctx.device_id, b_order,
                                   &lw.in_proj_b_w, lw.owned);
                    load_role_rows(index, eng->desc, ROLE_GDN_BA, i, ctx.device_id, a_order,
                                   &lw.in_proj_a_w, lw.owned);
                }
                if (lw.plan.mixer == ENGINE_MIXER_FULL_ATTN) {
                    const size_t kv_bytes = kv_cache_bytes(eng->local_dims);
                    lw.kv_cache = (__nv_bfloat16 *)alloc_owned(lw.owned, kv_bytes,
                                                               ctx.device_id, "Allocate KV cache");
                    check_cuda(cudaMemsetAsync(lw.kv_cache, 0, kv_bytes, ctx.stream),
                               "Zero KV cache");
                    lw.reset_zero.emplace_back(lw.kv_cache, kv_bytes);
                } else if (lw.plan.mixer == ENGINE_MIXER_MLA) {
                    const size_t mla_bytes = (size_t)eng->dims.max_seq_len *
                        (eng->dims.mla_kv_lora_rank + eng->dims.mla_qk_rope_head_dim) *
                        sizeof(__nv_bfloat16);
                    lw.mla_cache = (__nv_bfloat16 *)alloc_owned(lw.owned, mla_bytes,
                                                                ctx.device_id, "Allocate MLA cache");
                    check_cuda(cudaMemsetAsync(lw.mla_cache, 0, mla_bytes, ctx.stream),
                               "Zero MLA cache");
                    lw.reset_zero.emplace_back(lw.mla_cache, mla_bytes);
                } else if (lw.plan.mixer == ENGINE_MIXER_GDN) {
                    const size_t norm_bytes = eng->dims.gdn_head_dim * sizeof(float);
                    lw.gdn_norm_f32 = (float *)alloc_owned(lw.owned, norm_bytes, ctx.device_id,
                                                           "Allocate GDN norm");
                    kernel_cast_bf16_f32(lw.gdn_norm_f32, lw.gdn_norm_w, eng->dims.gdn_head_dim, ctx.stream);
                    check_cuda(cudaGetLastError(), "Convert GDN norm");
                    const size_t conv_bytes = conv_state_bytes(eng);
                    lw.conv_state = (__nv_bfloat16 *)alloc_owned(lw.owned, conv_bytes,
                                                                 ctx.device_id, "Allocate conv state");
                    check_cuda(cudaMemsetAsync(lw.conv_state, 0, conv_bytes, ctx.stream),
                               "Zero conv state");
                    const size_t ssm_bytes = ssm_state_bytes(eng);
                    lw.ssm_state = (float *)alloc_owned(lw.owned, ssm_bytes, ctx.device_id,
                                                        "Allocate SSM state");
                    check_cuda(cudaMemsetAsync(lw.ssm_state, 0, ssm_bytes, ctx.stream),
                               "Zero SSM state");
                    lw.reset_zero.emplace_back(lw.conv_state, conv_bytes);
                    lw.reset_zero.emplace_back(lw.ssm_state, ssm_bytes);
                } else {
                    throw EngineError(ENGINE_ERR_CONFIG,
                                      "Layer kind is not implemented in this build");
                }
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

static LayerContext make_layer_context(EngineHandle *eng, int layer, int dev_idx,
                                       int tokens, bool with_taps) {
    DeviceCtx &ctx = eng->ctx[dev_idx];
    LayerContext lctx;
    lctx.cublas = ctx.cublas;
    lctx.stream = ctx.stream;
    lctx.workspace = ctx.workspace;
    lctx.conv_bias_zero = ctx.conv_bias_zero;
    lctx.positions = ctx.positions;
    lctx.fla_scratch = ctx.fla_scratch;
    lctx.moe_scratch = ctx.moe_scratch;
    lctx.mla_scratch = ctx.mla_scratch;
    lctx.tokens = tokens;
    lctx.seq_len = eng->seq_len + tokens;
    lctx.layer_index = layer;
    lctx.device = ctx.device_id;
    lctx.dims = eng->replicated ? &eng->local_dims : &eng->dims;
    lctx.taps = with_taps ? &eng->taps : nullptr;
    lctx.reduce = nullptr;
    lctx.reduce_opaque = nullptr;
    lctx.split_phase = 0;
    return lctx;
}

static void residual_add_device(EngineHandle *eng, int dev_idx, size_t elements) {
    DeviceCtx &ctx = eng->ctx[dev_idx];
    check_cuda(cudaSetDevice(ctx.device_id), "Select residual device");
    kernel_residual_add(ctx.residual, ctx.layer_out, (int)elements, ctx.stream);
    check_cuda(cudaGetLastError(), "Residual add");
}

/* Sum every rank's layer_out in place (leader reduce + broadcast). */
static int allreduce_layer_out(EngineHandle *eng, size_t elements) {
    /* The activations this reduces are BF16, so that is the element type the
     * collective is told to move; the transport itself is type-agnostic. */
    std::vector<void *> buffers(eng->num_devices);
    std::vector<cudaStream_t> streams(eng->num_devices);
    std::vector<cudaEvent_t> events(eng->num_devices);
    std::vector<cudaEvent_t> done_events(eng->num_devices);
    for (int d = 0; d < eng->num_devices; ++d) {
        buffers[d] = eng->ctx[d].layer_out;
        streams[d] = eng->ctx[d].stream;
        events[d] = eng->ctx[d].copy_event;
        done_events[d] = eng->ctx[d].read_done_event;
    }
    return allreduce_sum(eng->devices.data(), streams.data(), events.data(),
                         done_events.data(), eng->num_devices, buffers.data(),
                         eng->ctx[0].reduce_staging, elements, COLLECTIVE_BF16);
}

/* LayerContext hook for sublayers whose parts live on different ranks (expert
 * parallelism: the routed experts). Reduces the per-rank layer_out buffers. */
static int engine_reduce_activation(void *opaque, size_t elements) {
    return allreduce_layer_out((EngineHandle *)opaque, elements);
}

/* Final norm + LM head for the last row of `act`, downloaded to h_logits. */
static void compute_logits(EngineHandle *eng, DeviceCtx &last, const __nv_bfloat16 *act,
                           int tokens, float *h_logits) {
    // Public API returns only the final row; avoid [max_chunk,vocab] logits.
    const __nv_bfloat16 *final_row = act + (size_t)(tokens - 1) * eng->dims.hidden_size;
    if (eng->dims.norm_style == 1) {
        kernel_rms_norm_plain(last.workspace, final_row, last.final_norm_w,
                              eng->dims.hidden_size, 1, eng->dims.rms_eps, last.stream);
    } else {
        kernel_gemma_rms_norm(last.workspace, final_row, last.final_norm_w,
                              eng->dims.hidden_size, 1, eng->dims.rms_eps, last.stream);
    }
    check_cuda(cudaGetLastError(), "Final norm");
    check_forward(gemm_bf16_f32out(last.cublas, last.d_logits, last.workspace,
                  last.lm_head_w, 1, eng->dims.vocab_size, eng->dims.hidden_size), "LM head");
    check_cuda(cudaMemcpyAsync(h_logits, last.d_logits, eng->dims.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, last.stream), "Download logits");
}

/* Layer-wise placement: the residual hops device-to-device once per layer. */
static void forward_pipelined(EngineHandle *eng, const int64_t *token_ids, int tokens,
                              float *h_logits) {
    DeviceCtx &first = eng->ctx[0];
    check_cuda(cudaSetDevice(first.device_id), "Select embedding device");
    check_cuda(cudaMemcpyAsync(first.token_ids, token_ids, tokens * sizeof(int64_t),
                               cudaMemcpyHostToDevice, first.stream), "Upload token IDs");
    check_cuda(cudaStreamSynchronize(first.stream), "Finish token upload");
    kernel_embedding(first.residual, first.embed_w, first.token_ids, eng->dims.hidden_size,
                     tokens, first.stream);
    check_cuda(cudaGetLastError(), "Embedding");

    int current = 0;
    for (int i = 0; i < eng->num_layers; ++i) {
        const int dev_idx = eng->layer_device[i];
        __nv_bfloat16 *act = move_activation(eng, current, dev_idx, tokens);
        current = dev_idx;
        DeviceCtx &ctx = eng->ctx[dev_idx];
        LayerContext lctx = make_layer_context(eng, i, dev_idx, tokens, true);
        check_forward(forward_layer(&lctx, &eng->layers[i], act, ctx.layer_out), "Layer forward");
        tap_dump_rows(&eng->taps, "layer", i, ctx.device_id, ctx.stream, act, tokens,
                      eng->dims.hidden_size);
    }

    if (h_logits) {
        const int last_idx = eng->num_devices - 1;
        __nv_bfloat16 *act = move_activation(eng, current, last_idx, tokens);
        current = last_idx;
        DeviceCtx &last = eng->ctx[last_idx];
        check_cuda(cudaSetDevice(last.device_id), "Select logits device");
        compute_logits(eng, last, act, tokens, h_logits);
    }
    check_cuda(cudaSetDevice(eng->devices[current]), "Select final forward device");
    check_cuda(cudaStreamSynchronize(eng->ctx[current].stream), "Finish forward");
}

/* Replicated placement (tensor parallel): every rank holds the whole model with
 * its weight shards, activations are replicas, and a sublayer whose weights were
 * split all-reduces its partial output before the residual add. */
static void forward_replicated(EngineHandle *eng, const int64_t *token_ids, int tokens,
                               float *h_logits) {
    for (int d = 0; d < eng->num_devices; ++d) {
        DeviceCtx &ctx = eng->ctx[d];
        check_cuda(cudaSetDevice(ctx.device_id), "Select embedding device");
        check_cuda(cudaMemcpyAsync(ctx.token_ids, token_ids, tokens * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, ctx.stream), "Upload token IDs");
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish token upload");
        kernel_embedding(ctx.residual, ctx.embed_w, ctx.token_ids, eng->dims.hidden_size,
                         tokens, ctx.stream);
        check_cuda(cudaGetLastError(), "Embedding");
    }

    const size_t elements = (size_t)tokens * eng->dims.hidden_size;
    for (int i = 0; i < eng->num_layers; ++i) {
        const LayerWeights &wa = layer_weights(eng, i, 0);
        const bool reduce_mixer = wa.mixer_sharded;
        const bool reduce_ffn = wa.ffn_sharded;
        for (int phase = 0; phase < 2; ++phase) {
            const bool is_mixer = phase == 0;
            /* Expert parallelism splits the MoE feed-forward into the routed
             * partial (every rank) and the shared experts (replicated): the
             * reduce belongs between them, and it must run after *every* rank
             * has queued its partial, so the caller drives two passes. */
            const bool ep_moe = !is_mixer && eng->expert_parallel &&
                                layer_weights(eng, i, 0).plan.ffn == ENGINE_FFN_MOE;
            for (int pass = 0; pass < (ep_moe ? 2 : 1); ++pass) {
                for (int d = 0; d < eng->num_devices; ++d) {
                    DeviceCtx &ctx = eng->ctx[d];
                    check_cuda(cudaSetDevice(ctx.device_id), "Select layer device");
                    LayerContext lctx = make_layer_context(eng, i, d, tokens, d == 0);
                    lctx.split_phase = pass;
                    const LayerWeights &w = layer_weights(eng, i, d);
                    if (ep_moe) {
                        lctx.reduce = engine_reduce_activation;
                        lctx.reduce_opaque = eng;
                    }
                    int status = is_mixer
                        ? forward_mixer(&lctx, &w, ctx.residual, ctx.layer_out)
                        : forward_ffn(&lctx, &w, ctx.residual, ctx.layer_out);
                    check_forward(status, is_mixer ? "Layer mixer" : "Layer feed-forward");
                }
                if (ep_moe && pass == 0 && eng->num_devices > 1) {
                    int status = allreduce_layer_out(eng, elements);
                    if (status != 0)
                        throw EngineError(ENGINE_ERR_CUDA, "Expert all-reduce failed");
                }
            }
            if ((is_mixer ? reduce_mixer : reduce_ffn) && eng->num_devices > 1) {
                int status = allreduce_layer_out(eng, elements);
                if (status != 0)
                    throw EngineError(ENGINE_ERR_CUDA, "All-reduce failed");
            }
            for (int d = 0; d < eng->num_devices; ++d) residual_add_device(eng, d, elements);
        }
        tap_dump_rows(&eng->taps, "layer", i, eng->ctx[0].device_id, eng->ctx[0].stream,
                      eng->ctx[0].residual, tokens, eng->dims.hidden_size);
    }

    if (h_logits) {
        const int last_idx = eng->num_devices - 1;
        DeviceCtx &last = eng->ctx[last_idx];
        check_cuda(cudaSetDevice(last.device_id), "Select logits device");
        compute_logits(eng, last, last.residual, tokens, h_logits);
    }
    for (int d = 0; d < eng->num_devices; ++d) {
        DeviceCtx &ctx = eng->ctx[d];
        check_cuda(cudaSetDevice(ctx.device_id), "Select final forward device");
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish forward");
    }
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
    if (eng->replicated) {
        forward_replicated(eng, token_ids, tokens, h_logits);
    } else {
        forward_pipelined(eng, token_ids, tokens, h_logits);
    }
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
                if (!eng->replicated && eng->layer_device[i] != d) continue;
                for (const auto &buffer : layer_weights(eng, i, d).reset_zero) {
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
        const int slots = eng->replicated ? eng->num_layers * eng->num_devices : eng->num_layers;
        for (int i = 0; i < slots; ++i) {
            const int dev_idx = eng->replicated ? i % eng->num_devices : eng->layer_device[i];
            cleanup_cuda(cudaSetDevice(eng->devices[dev_idx]));
            for (void *ptr : eng->layers[i].owned)
                if (ptr) cleanup_cuda(cudaFree(ptr));
        }
    }
    if (eng->ctx) {
        for (int d = 0; d < eng->num_devices; ++d) {
            DeviceCtx &ctx = eng->ctx[d];
            if (ctx.device_id < 0) continue;
            cleanup_cuda(cudaSetDevice(ctx.device_id));
            void *buffers[] = {ctx.residual, ctx.layer_out, ctx.workspace, ctx.conv_bias_zero,
                               ctx.positions, ctx.fla_scratch, ctx.moe_scratch, ctx.mla_scratch,
                               ctx.token_ids, ctx.embed_w, ctx.lm_head_w, ctx.final_norm_w,
                               ctx.d_logits, ctx.reduce_staging};
            for (void *ptr : buffers) if (ptr) cleanup_cuda(cudaFree(ptr));
            if (ctx.cublas) {
                cublasStatus_t status = cublasDestroy(ctx.cublas);
                if (status != CUBLAS_STATUS_SUCCESS && !g_error_buf[0])
                    set_error("engine_destroy: cuBLAS status %d", (int)status);
            }
            if (ctx.copy_event) cleanup_cuda(cudaEventDestroy(ctx.copy_event));
            if (ctx.read_done_event) cleanup_cuda(cudaEventDestroy(ctx.read_done_event));
            if (ctx.stream) cleanup_cuda(cudaStreamDestroy(ctx.stream));
        }
    }
    delete[] eng->ctx;
    delete[] eng->layers;
    delete eng;
}

int engine_vocab_size(const EngineHandle *eng) { return eng ? eng->dims.vocab_size : 0; }
int engine_seq_len(const EngineHandle *eng) { return eng ? eng->seq_len : 0; }
