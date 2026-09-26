/** Weight loading and chunked, layer-partitioned multi-GPU inference. */
#include "backward.h"
#include "backward_layers.h"
#include "engine.h"
#include "kernels.h"
#include "profile.h"
#include "layers.h"
#include "manifest.h"
#include "model_desc.h"
#include "moe.h"
#include "sha256.h"
#include "flashinfer_ops.h"
#include "fla_ops.h"
#include "build_info.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
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
    /* Expert parallelism: this rank's routed partial in FP32 and the FP32
     * leader staging the merge goes through. */
    float *moe_partial_f32 = nullptr;
    float *moe_staging_f32 = nullptr;
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
    /* Execution manifest inputs that are facts about *this* load: the canonical
     * descriptor digest (the descriptor itself stays portable) and the immutable
     * parameter identity, both computed once at create. */
    manifest_hex_t desc_sha256{};
    long long desc_bytes = 0;
    struct ManifestWeights weights{};
    /* Debug taps: INFER_TAP_LAYERS / INFER_TAP_DIR, see tap.cu. */
    TapConfig taps;
    /* Stage 3: the checkpoint's tensor index is kept so a training parameter's
     * element count comes from the tensor that was actually loaded rather than from
     * a shape formula, and the training path's device scratch. */
    std::map<std::string, TensorInfo> tensor_index;
    float *train_scratch = nullptr;   /* [max_chunk] fp32: fused log-probabilities */
    int *train_labels = nullptr;      /* [max_chunk] int32: labels for the gather */
    struct TrainStore *train_store = nullptr;
    /* Training state attach allocated (masters, gradients, optimizer slots), freed at
     * destroy. A frozen parameter contributes nothing here. */
    std::vector<void *> train_buffers;
    /* Stage 5: one FP32 pool for the SFT step's scratch (the retained final hidden
     * state, the running gradients, the loss's per-row LM-head buffers, and the layer
     * backward's cast/gradient pools), sized on the first step from the token count and
     * the vocabulary. One allocation keeps the leak accounting in one place. */
    float *train_pool = nullptr;
    size_t train_pool_floats = 0;
    size_t train_pool_tokens = 0;
    struct TrainStepOffsets {
        size_t final_hidden = 0;   /* tokens * hidden */
        size_t d_hidden = 0;
        size_t d_next = 0;
        size_t lm_weight = 0;      /* vocab * hidden (the widened LM head) */
        size_t logits_row = 0;     /* vocab */
        size_t d_logits_row = 0;   /* vocab */
        size_t row_hidden = 0;     /* hidden, the widened row */
        size_t row_normed = 0;     /* hidden */
        size_t row_extra = 0;      /* 2: inv_rms and the loss slope */
        size_t layer_cast = 0;
        size_t layer_grad = 0;
        size_t total = 0;
    } train_off;
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

/* The rows and columns this rank keeps of a role: the whole tensor when the descriptor's
 * rule does not shard it, otherwise the descriptor's shard view. `split` says whether a rule
 * actually applied, which is what the caller reports as "this sublayer all-reduces". */
static void role_extent(const struct ModelDesc &desc, int role, const TensorInfo &ti, int rank,
                        struct ShardView *out, bool *split) {
    *split = false;
    const int slot = model_desc_role_index(&desc, role);
    const int rule = slot >= 0 && slot < desc.role_shard_count ? desc.role_shards[slot]
                                                              : ENGINE_SHARD_NONE;
    if (desc.tp_size <= 1 || rule == ENGINE_SHARD_NONE) {
        if (ti.ndim != 2)
            throw EngineError(ENGINE_ERR_WEIGHTS,
                              std::string("Cannot pack a non-2-D tensor: ") + ti.name);
        out->row_off = 0;
        out->rows = ti.shape[0];
        out->col_off = 0;
        out->cols = ti.shape[1];
        return;
    }
    char err[256] = {0};
    if (model_desc_shard_view(&desc, role, ti.shape[0], ti.shape[1], rank, out, err,
                              sizeof(err)) != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS, std::string("Shard view: ") + err);
    *split = true;
}

/* Upload a role's (possibly sharded) slice into a buffer the caller owns, which must hold
 * view.rows * view.cols elements. The packing below needs this: the destination is one half
 * of a shared allocation rather than a buffer this function chose. */
static void load_role_into(const std::map<std::string, TensorInfo> &index,
                           const struct ModelDesc &desc, int role, int layer, int device,
                           const struct ShardView &view, bool split, __nv_bfloat16 *dst) {
    const TensorInfo &ti = find_role(index, desc, role, layer);
    const int64_t bytes = (int64_t)view.rows * view.cols * (int64_t)sizeof(__nv_bfloat16);
    const int status = split
        ? safetensors_load_tensor_slice(ti, dst, bytes, device, view.row_off, view.rows,
                                        view.col_off, view.cols)
        : safetensors_load_tensor(ti, dst, bytes, device);
    check_cuda(cudaGetLastError(), "Upload weight");
    if (status != 0)
        throw EngineError(ENGINE_ERR_WEIGHTS,
                          std::string("Cannot load tensor: ") + ti.name + ": " +
                              safetensors_last_error());
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

/* ------------------------------------------------------------------ */
/* Execution manifest inputs                                          */
/* ------------------------------------------------------------------ */

/* Last path component, so the shard-set identity does not depend on where the
 * checkpoint happens to be mounted. */
static std::string path_basename(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

/* The immutable parameter identity: for every tensor in sorted name order (the
 * index is a std::map, so iteration is by name) its name, dtype, shape and byte
 * count, plus the set of shard basenames. Cheap enough to compute at load. The
 * raw-content hash is deliberately *not* computed here: hashing a 50 GiB
 * checkpoint is minutes of I/O, and a caller that wants it has to ask. */
static void compute_weights_identity(const std::map<std::string, TensorInfo> &index,
                                     struct ManifestWeights *out) {
    std::memset(out, 0, sizeof(*out));
    struct sha256_ctx ctx;
    sha256_init(&ctx);
    std::vector<std::string> shards;
    char line[1024];
    for (std::map<std::string, TensorInfo>::const_iterator it = index.begin();
         it != index.end(); ++it) {
        const TensorInfo &ti = it->second;
        int used = snprintf(line, sizeof(line), "%s|%d|", ti.name.c_str(), ti.dtype);
        for (int d = 0; d < ti.ndim && used > 0 && used < (int)sizeof(line); ++d)
            used += snprintf(line + used, sizeof(line) - (size_t)used, "%lld,",
                             (long long)ti.shape[d]);
        const int tail = snprintf(line + (used > 0 ? used : 0),
                                  sizeof(line) - (size_t)(used > 0 ? used : 0), "|%lld\n",
                                  (long long)ti.data_bytes());
        if (used <= 0 || tail <= 0 || used + tail >= (int)sizeof(line)) {
            /* A name that cannot be represented canonically must not be silently
             * truncated into someone else's identity. */
            out->parameter_manifest_sha256[0] = '\0';
            return;
        }
        sha256_update(&ctx, line, (size_t)(used + tail));
        const std::string base = path_basename(ti.file_path);
        if (std::find(shards.begin(), shards.end(), base) == shards.end())
            shards.push_back(base);
    }
    unsigned char digest[32];
    sha256_final(&ctx, digest);
    sha256_to_hex(digest, out->parameter_manifest_sha256);
    std::sort(shards.begin(), shards.end());
    struct sha256_ctx shard_ctx;
    sha256_init(&shard_ctx);
    for (size_t i = 0; i < shards.size(); ++i) {
        sha256_update(&shard_ctx, shards[i].c_str(), shards[i].size());
        sha256_update(&shard_ctx, "\n", 1);
    }
    sha256_final(&shard_ctx, digest);
    sha256_to_hex(digest, out->shards_sha256);
    out->tensor_count = (long long)index.size();
    out->content_hash_present = 0;
    out->content_sha256[0] = '\0';
}

/* Does a ';'-separated target list contain @sm@ as SASS (want_virtual 0) or as
 * PTX (want_virtual 1)? "90a" counts as 90 and "90-virtual" as PTX for 90. */
static int arch_list_has(const char *list, int sm, int want_virtual) {
    const char *cursor = list;
    while (cursor != NULL && *cursor != '\0') {
        const char *end = strchr(cursor, ';');
        const size_t len = end != NULL ? (size_t)(end - cursor) : strlen(cursor);
        char entry[32];
        if (len > 0 && len < sizeof(entry)) {
            std::memcpy(entry, cursor, len);
            entry[len] = '\0';
            const int is_virtual = strstr(entry, "virtual") != NULL;
            if (atoi(entry) == sm && is_virtual == (want_virtual != 0)) return 1;
        }
        if (end == NULL) break;
        cursor = end + 1;
    }
    return 0;
}

/* The build facts come from the header the build step generated, not from a
 * caller-supplied label (csrc/gen_build_info.cmake). */
static void fill_build_info(struct ManifestBuildInfo *out) {
    std::memset(out, 0, sizeof(*out));
    snprintf(out->engine_version, sizeof(out->engine_version), "%s", ENGINE_BUILD_ENGINE_VERSION);
    snprintf(out->git_commit, sizeof(out->git_commit), "%s", ENGINE_BUILD_GIT_COMMIT);
    snprintf(out->cuda_toolkit, sizeof(out->cuda_toolkit), "%s", ENGINE_BUILD_CUDA_TOOLKIT);
    snprintf(out->cuda_archs, sizeof(out->cuda_archs), "%s", ENGINE_BUILD_CUDA_ARCHS);
    snprintf(out->triton_archs, sizeof(out->triton_archs), "%s", ENGINE_BUILD_TRITON_ARCHS);
    snprintf(out->triton_version, sizeof(out->triton_version), "%s", ENGINE_BUILD_TRITON_VERSION);
    snprintf(out->fla_version, sizeof(out->fla_version), "%s", ENGINE_BUILD_FLA_VERSION);
    snprintf(out->flashinfer_header_sha256, sizeof(out->flashinfer_header_sha256), "%s",
             ENGINE_BUILD_FLASHINFER_HEADER_SHA256);
    snprintf(out->nvcc_flags_sha256, sizeof(out->nvcc_flags_sha256), "%s",
             ENGINE_BUILD_NVCC_FLAGS_SHA256);
    snprintf(out->generated_kernels_sha256, sizeof(out->generated_kernels_sha256), "%s",
             ENGINE_BUILD_GENERATED_KERNELS_SHA256);
}

/* The runtime device facts. kernel_path and triton_cubin_arch are the *selection
 * rule* applied to the build's target lists, because which binary the driver
 * actually launched is not queryable per kernel; the field names say "selected"
 * for that reason. */
static void fill_runtime_devices(const EngineHandle *eng, struct ManifestDevice *out) {
    for (int d = 0; d < eng->num_devices; ++d) {
        std::memset(&out[d], 0, sizeof(out[d]));
        out[d].cuda_ordinal = eng->devices[d];
        int sm = 0;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, eng->devices[d]) == cudaSuccess) {
            snprintf(out[d].name, sizeof(out[d].name), "%s", prop.name);
            snprintf(out[d].compute_capability, sizeof(out[d].compute_capability), "%d.%d",
                     prop.major, prop.minor);
            snprintf(out[d].uuid, sizeof(out[d].uuid),
                     "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                     prop.uuid.bytes[0], prop.uuid.bytes[1], prop.uuid.bytes[2],
                     prop.uuid.bytes[3], prop.uuid.bytes[4], prop.uuid.bytes[5],
                     prop.uuid.bytes[6], prop.uuid.bytes[7], prop.uuid.bytes[8],
                     prop.uuid.bytes[9], prop.uuid.bytes[10], prop.uuid.bytes[11],
                     prop.uuid.bytes[12], prop.uuid.bytes[13], prop.uuid.bytes[14],
                     prop.uuid.bytes[15]);
            sm = prop.major * 10 + prop.minor;
        } else {
            snprintf(out[d].name, sizeof(out[d].name), "%s", MANIFEST_UNAVAILABLE);
            snprintf(out[d].compute_capability, sizeof(out[d].compute_capability), "%s",
                     MANIFEST_UNAVAILABLE);
            snprintf(out[d].uuid, sizeof(out[d].uuid), "%s", MANIFEST_UNAVAILABLE);
        }
        if (sm <= 0) {
            snprintf(out[d].kernel_path, sizeof(out[d].kernel_path), "%s", MANIFEST_UNAVAILABLE);
            snprintf(out[d].triton_cubin_arch, sizeof(out[d].triton_cubin_arch), "%s",
                     MANIFEST_UNAVAILABLE);
        } else {
            const char *path = arch_list_has(ENGINE_BUILD_CUDA_ARCHS, sm, 0) ? "sass"
                             : arch_list_has(ENGINE_BUILD_CUDA_ARCHS, 90, 1) && sm >= 90
                                   ? "ptx-jit"
                                   : MANIFEST_UNSUPPORTED;
            snprintf(out[d].kernel_path, sizeof(out[d].kernel_path), "%s", path);
            if (arch_list_has(ENGINE_BUILD_TRITON_ARCHS, sm, 0))
                snprintf(out[d].triton_cubin_arch, sizeof(out[d].triton_cubin_arch), "%d", sm);
            else
                snprintf(out[d].triton_cubin_arch, sizeof(out[d].triton_cubin_arch), "%s",
                         MANIFEST_UNSUPPORTED);
        }
    }
}

/* CUDA/cuBLAS report their version as major*1000+minor*10+patch; cuBLAS packs it
 * as major*10000+minor*100+patch. An unavailable query is reported as such
 * rather than as a zero version. */
static void format_version_1000(int version, char *out, size_t out_len) {
    snprintf(out, out_len, "%d.%d.%d", version / 1000, (version % 1000) / 10, version % 10);
}

static void fill_runtime_versions(cublasHandle_t cublas_handle, char *runtime,
                                  size_t runtime_len, char *driver, size_t driver_len,
                                  char *cublas, size_t cublas_len) {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) == cudaSuccess)
        format_version_1000(version, runtime, runtime_len);
    else
        snprintf(runtime, runtime_len, "%s", MANIFEST_UNAVAILABLE);
    if (cudaDriverGetVersion(&version) == cudaSuccess)
        format_version_1000(version, driver, driver_len);
    else
        snprintf(driver, driver_len, "%s", MANIFEST_UNAVAILABLE);
    if (cublas_handle != nullptr && cublasGetVersion(cublas_handle, &version) ==
                                        CUBLAS_STATUS_SUCCESS && version > 0)
        snprintf(cublas, cublas_len, "%d.%d.%d", version / 10000, (version % 10000) / 100,
                 version % 100);
    else
        snprintf(cublas, cublas_len, "%s", MANIFEST_UNAVAILABLE);
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
        if (model_desc_check_runtime_support(&eng->desc, desc_error, sizeof(desc_error)) != 0)
            throw EngineError(ENGINE_ERR_CONFIG,
                              std::string("Unsupported descriptor: ") + desc_error);
        fill_dims(eng->dims, eng->desc);
        eng->num_layers = eng->desc.num_layers;

        /* The canonical descriptor digest is what the manifest references. It is
         * taken from the canonical echo, not from the caller's wire text, so two
         * spellings of the same descriptor share one identity. */
        {
            char *desc_text = (char *)::malloc(ENGINE_MANIFEST_MAX);
            if (desc_text == nullptr)
                throw EngineError(ENGINE_ERR_ALLOC, "Cannot allocate the descriptor buffer");
            const int formatted = model_desc_format(&eng->desc, desc_text, ENGINE_MANIFEST_MAX);
            if (formatted < 0) {
                ::free(desc_text);
                throw EngineError(ENGINE_ERR_CONFIG, "Cannot format the descriptor canonically");
            }
            eng->desc_bytes = formatted;
            sha256_hex(desc_text, (size_t)formatted, eng->desc_sha256);
            ::free(desc_text);
        }

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
        compute_weights_identity(index, &eng->weights);
        /* Kept for the training path: a parameter's element count must come from the
         * tensor that was loaded, not from a shape formula that could disagree. */
        eng->tensor_index = index;

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
            /* The Stage-5 layer backward reuses the workspace for its recompute and
             * needs one hidden-state-sized BF16 tail beyond the forward's own layout
             * (the narrowed residual). Two chunks' worth leaves room for the widest
             * kind at the largest chunk. */
            ctx.ws_size = layer_workspace_size(max_chunk, &eng->local_dims) +
                          2 * (size_t)max_chunk * eng->local_dims.hidden_size *
                              sizeof(__nv_bfloat16);
            check_cuda(cudaMalloc(&ctx.workspace, ctx.ws_size), "Allocate layer workspace");
            check_cuda(cudaMalloc(&ctx.token_ids, max_chunk * sizeof(int64_t)),
                       "Allocate token IDs");
            if (eng->replicated && d == 0) {
                check_cuda(cudaMalloc(&ctx.reduce_staging, activation_bytes),
                           "Allocate reduce staging");
            }
            if (eng->expert_parallel) {
                /* One FP32 copy of the activation per rank for the routed partial,
                 * plus the leader's FP32 staging; the merge rounds once, when the
                 * merged partial becomes the activation. */
                check_cuda(cudaMalloc(&ctx.moe_partial_f32, activation_bytes * 2),
                           "Allocate the FP32 expert partial");
                if (d == 0) {
                    check_cuda(cudaMalloc(&ctx.moe_staging_f32, activation_bytes * 2),
                               "Allocate the FP32 expert staging");
                }
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
                bool mlp_packed = false;
                for (int role = 0; role < ROLE_COUNT; ++role) {
                    if (!role_used_by_layer(role, lw.plan.mixer, lw.plan.ffn)) continue;
                    if (model_desc_role_index(&eng->desc, role) < 0) continue;
                    /* A dense MLP's gate and up weights share one [2I, H] allocation, so the
                     * forward can issue a single N = 2I GEMM over a row-interleaved [T, 2I]
                     * output (plan F1). Loading each role into its own half means there is
                     * never a second copy of these weights; both roles must shard on the same
                     * axis, which is checked rather than assumed. */
                    if (role == ROLE_MLP_UP && mlp_packed) continue;
                    if (role == ROLE_MLP_GATE &&
                        role_used_by_layer(ROLE_MLP_UP, lw.plan.mixer, lw.plan.ffn)) {
                        const TensorInfo &gate_ti = find_role(index, eng->desc, ROLE_MLP_GATE, i);
                        const TensorInfo &up_ti = find_role(index, eng->desc, ROLE_MLP_UP, i);
                        struct ShardView gate_view, up_view;
                        bool gate_split = false, up_split = false;
                        role_extent(eng->desc, ROLE_MLP_GATE, gate_ti, r, &gate_view, &gate_split);
                        role_extent(eng->desc, ROLE_MLP_UP, up_ti, r, &up_view, &up_split);
                        if (gate_view.rows != up_view.rows || gate_view.cols != up_view.cols ||
                            gate_split != up_split)
                            throw EngineError(ENGINE_ERR_WEIGHTS,
                                              "Cannot pack gate/up: their shard extents differ");
                        const size_t half = (size_t)gate_view.rows * (size_t)gate_view.cols;
                        __nv_bfloat16 *packed = (__nv_bfloat16 *)alloc_owned(
                            lw.owned, (size_t)2 * half * sizeof(__nv_bfloat16), ctx.device_id,
                            "Allocate packed gate/up weight");
                        lw.gate_proj_w = packed;
                        lw.up_proj_w = packed + half;
                        load_role_into(index, eng->desc, ROLE_MLP_GATE, i, ctx.device_id,
                                       gate_view, gate_split, lw.gate_proj_w);
                        load_role_into(index, eng->desc, ROLE_MLP_UP, i, ctx.device_id, up_view,
                                       up_split, lw.up_proj_w);
                        if (gate_split) ffn_sharded = true;
                        mlp_packed = true;
                        continue;
                    }
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

int engine_manifest_version(void) { return ENGINE_MANIFEST_VERSION; }

int engine_manifest(const EngineHandle *eng, char *buf, int buf_len) {
    if (!eng || !buf || buf_len <= 0) {
        set_error("engine_manifest: invalid arguments");
        return ENGINE_ERR_CONFIG;
    }
    if (eng->num_devices <= 0 || (size_t)eng->num_devices > (size_t)ENGINE_MANIFEST_MAX_DEVICES) {
        set_error("engine_manifest: the engine holds %d devices, the manifest covers at most %d",
                  eng->num_devices, ENGINE_MANIFEST_MAX_DEVICES);
        return ENGINE_ERR_CONFIG;
    }

    struct ManifestBuildInfo build;
    fill_build_info(&build);
    struct ManifestDevice devices[ENGINE_MANIFEST_MAX_DEVICES];
    fill_runtime_devices(eng, devices);

    char runtime_version[32], driver_version[32], cublas_version[32];
    cublasHandle_t cublas_handle = eng->ctx != nullptr ? eng->ctx[0].cublas : nullptr;
    fill_runtime_versions(cublas_handle, runtime_version, sizeof(runtime_version),
                          driver_version, sizeof(driver_version), cublas_version,
                          sizeof(cublas_version));

    struct ManifestInputs in;
    std::memset(&in, 0, sizeof(in));
    in.desc = &eng->desc;
    in.descriptor_sha256 = eng->desc_sha256;
    in.descriptor_bytes = eng->desc_bytes;
    in.build = &build;
    in.cuda_runtime_version = runtime_version;
    in.cuda_driver_version = driver_version;
    in.cublas_version = cublas_version;
    in.devices = devices;
    in.device_count = eng->num_devices;
    in.weights = &eng->weights;
    in.replicated = eng->replicated ? 1 : 0;
    in.ep_size = eng->ep_size;
    in.declared_tp_rank = eng->desc.tp_rank;
    in.declared_ep_rank = eng->desc.ep_rank;
    in.device_ordinals = eng->devices.data();
    in.layer_device = eng->layer_device.empty() ? nullptr : eng->layer_device.data();
    in.num_layers = eng->num_layers;
    in.regions = nullptr;      /* the committed registry */
    in.region_count = 0;
    in.sampling = manifest_default_sampling();

    const int written = manifest_format(&in, buf, buf_len);
    if (written == -2) {
        set_error("engine_manifest: the manifest needs more than %d bytes", buf_len);
        return ENGINE_ERR_CONFIG;
    }
    if (written < 0) {
        set_error("engine_manifest: cannot format the execution manifest");
        return ENGINE_ERR_CONFIG;
    }
    return written;
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
    lctx.moe_partial_f32 = ctx.moe_partial_f32;
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

/* Merge the FP32 expert partials across ranks. These are the values the routed
 * experts produced, before anything rounds them, so the merge is exact and the
 * activation is rounded exactly once afterwards; merging them in BF16 instead
 * would add a rounding the single-rank path does not pay. */
static int allreduce_moe_partial(EngineHandle *eng, size_t elements) {
    std::vector<void *> buffers(eng->num_devices);
    std::vector<cudaStream_t> streams(eng->num_devices);
    std::vector<cudaEvent_t> events(eng->num_devices);
    std::vector<cudaEvent_t> done_events(eng->num_devices);
    for (int d = 0; d < eng->num_devices; ++d) {
        buffers[d] = eng->ctx[d].moe_partial_f32;
        streams[d] = eng->ctx[d].stream;
        events[d] = eng->ctx[d].copy_event;
        done_events[d] = eng->ctx[d].read_done_event;
    }
    return allreduce_sum(eng->devices.data(), streams.data(), events.data(),
                         done_events.data(), eng->num_devices, buffers.data(),
                         eng->ctx[0].moe_staging_f32, elements, COLLECTIVE_F32);
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
        { PROFILE_SCOPE("lm_head.final_norm", last.stream);
        kernel_rms_norm_plain(last.workspace, final_row, last.final_norm_w,
                              eng->dims.hidden_size, 1, eng->dims.rms_eps, last.stream);
        }
    } else {
        { PROFILE_SCOPE("lm_head.final_norm", last.stream);
        kernel_gemma_rms_norm(last.workspace, final_row, last.final_norm_w,
                              eng->dims.hidden_size, 1, eng->dims.rms_eps, last.stream);
        }
    }
    check_cuda(cudaGetLastError(), "Final norm");
    { PROFILE_SCOPE("lm_head.gemm", last.stream);
    check_forward(gemm_bf16_f32out(last.cublas, last.d_logits, last.workspace,
                  last.lm_head_w, 1, eng->dims.vocab_size, eng->dims.hidden_size), "LM head");
    }
    { PROFILE_SCOPE("lm_head.d2h", last.stream);
    check_cuda(cudaMemcpyAsync(h_logits, last.d_logits, eng->dims.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, last.stream), "Download logits");
    }
}

/* Layer-wise placement: the residual hops device-to-device once per layer. */
static void forward_pipelined(EngineHandle *eng, const int64_t *token_ids, int tokens,
                              float *h_logits, __nv_bfloat16 **out_act = nullptr,
                              DeviceCtx **out_ctx = nullptr) {
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

    if (h_logits || out_act) {
        const int last_idx = eng->num_devices - 1;
        __nv_bfloat16 *act = move_activation(eng, current, last_idx, tokens);
        current = last_idx;
        DeviceCtx &last = eng->ctx[last_idx];
        check_cuda(cudaSetDevice(last.device_id), "Select logits device");
        if (h_logits) compute_logits(eng, last, act, tokens, h_logits);
        /* The training path needs every position's hidden states, not just the last
         * row's logits, so it takes the buffer the logits branch would have used. */
        if (out_act) *out_act = act;
        if (out_ctx) *out_ctx = &last;
    }
    check_cuda(cudaSetDevice(eng->devices[current]), "Select final forward device");
    check_cuda(cudaStreamSynchronize(eng->ctx[current].stream), "Finish forward");
}

/* Replicated placement (tensor parallel): every rank holds the whole model with
 * its weight shards, activations are replicas, and a sublayer whose weights were
 * split all-reduces its partial output before the residual add. */
static void forward_replicated(EngineHandle *eng, const int64_t *token_ids, int tokens,
                               float *h_logits, __nv_bfloat16 **out_act = nullptr,
                               DeviceCtx **out_ctx = nullptr) {
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
                    int status = allreduce_moe_partial(eng, elements);
                    if (status != 0)
                        throw EngineError(ENGINE_ERR_CUDA, "Expert all-reduce failed");
                    /* The merged FP32 partial becomes this rank's activation with
                     * a single rounding, exactly as the single-rank path does
                     * after summing every expert. */
                    for (int d = 0; d < eng->num_devices; ++d) {
                        DeviceCtx &ctx = eng->ctx[d];
                        check_cuda(cudaSetDevice(ctx.device_id), "Select fold device");
                        kernel_cast_f32_bf16(ctx.layer_out, ctx.moe_partial_f32,
                                             (int)elements, ctx.stream);
                        check_cuda(cudaGetLastError(), "Fold the expert partial");
                    }
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

    if (h_logits || out_act) {
        const int last_idx = eng->num_devices - 1;
        DeviceCtx &last = eng->ctx[last_idx];
        check_cuda(cudaSetDevice(last.device_id), "Select logits device");
        if (h_logits) compute_logits(eng, last, last.residual, tokens, h_logits);
        if (out_act) *out_act = last.residual;
        if (out_ctx) *out_ctx = &last;
    }
    for (int d = 0; d < eng->num_devices; ++d) {
        DeviceCtx &ctx = eng->ctx[d];
        check_cuda(cudaSetDevice(ctx.device_id), "Select final forward device");
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish forward");
    }
}

static void forward_tokens(EngineHandle *eng, const int64_t *token_ids, int tokens,
                           float *h_logits, const int64_t *positions_override = nullptr,
                           __nv_bfloat16 **out_act = nullptr, DeviceCtx **out_ctx = nullptr) {
    std::vector<int64_t> positions(eng->dims.max_chunk);
    /* Inference derives the positions from the sequence length; a training call may
     * pass them explicitly, which is what the teacher-forcing contract needs. */
    for (int t = 0; t < tokens; ++t) {
        positions[t] = positions_override != nullptr ? positions_override[t]
                                                     : (int64_t)eng->seq_len + t;
    }
    for (int d = 0; d < eng->num_devices; ++d) {
        DeviceCtx &ctx = eng->ctx[d];
        check_cuda(cudaSetDevice(ctx.device_id), "Select position device");
        check_cuda(cudaMemcpyAsync(ctx.positions, positions.data(), tokens * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, ctx.stream), "Upload positions");
        // Finish staging host arrays before any subsequent operation can throw.
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish position upload");
    }
    if (eng->replicated) {
        forward_replicated(eng, token_ids, tokens, h_logits, out_act, out_ctx);
    } else {
        forward_pipelined(eng, token_ids, tokens, h_logits, out_act, out_ctx);
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
    /* The training store owns its own buffers, but it refuses to be destroyed while
     * a context or a step is live; at engine teardown there are none, and a leftover
     * reader would be a bug worth reporting rather than leaking. */
    for (void *ptr : eng->train_buffers) {
        if (ptr != nullptr) cleanup_cuda(cudaFree(ptr));
    }
    eng->train_buffers.clear();
    /* The training scratch lives on the logits device. */
    {
        const int logits_index = eng->num_devices > 0 ? eng->num_devices - 1 : -1;
        if (logits_index >= 0 && eng->ctx != nullptr && eng->ctx[logits_index].device_id >= 0) {
            cleanup_cuda(cudaSetDevice(eng->ctx[logits_index].device_id));
        }
        if (eng->train_scratch != nullptr) cleanup_cuda(cudaFree(eng->train_scratch));
        if (eng->train_pool != nullptr) cleanup_cuda(cudaFree(eng->train_pool));
        if (eng->train_labels != nullptr) cleanup_cuda(cudaFree(eng->train_labels));
        eng->train_scratch = nullptr;
        eng->train_labels = nullptr;
    }
    if (eng->train_store != nullptr) {
        const TrainStatus status = train_store_destroy(eng->train_store);
        if (status != TRAIN_OK)
            fprintf(stderr, "[engine] train store not released at destroy: %s\n",
                    train_last_error());
        eng->train_store = nullptr;
    }
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
                               ctx.d_logits, ctx.reduce_staging, ctx.moe_partial_f32,
                               ctx.moe_staging_f32};
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

/* ------------------------------------------------------------------ */
/* Stage 3: the training path                                         */
/* ------------------------------------------------------------------ */

namespace {

/* The tensor a (layer, role) was loaded from, or null when the descriptor has no
 * template for the role or the checkpoint has no tensor for it (a family that has
 * no such weight). */
const TensorInfo *role_tensor(const EngineHandle *eng, int role, int layer) {
    const int slot = model_desc_role_index(&eng->desc, role);
    if (slot < 0) return nullptr;
    char name[ENGINE_TEMPLATE_MAX];
    model_desc_expand(eng->desc.role_templates[slot], layer, 0, name, sizeof(name));
    auto it = eng->tensor_index.find(name);
    return it == eng->tensor_index.end() ? nullptr : &it->second;
}

long long tensor_elements(const TensorInfo &ti) {
    long long elements = 1;
    for (int i = 0; i < ti.ndim; ++i) elements *= ti.shape[i];
    return elements;
}

/* Where a role's BF16 weight lives for one layer in one rank. The role targets are
 * the same fields the loader filled, so a training write lands on the buffer the
 * forward reads - that is what makes a publication visible to every reader without
 * a second copy of the weights. */
__nv_bfloat16 *layer_role_buffer(EngineHandle *eng, int layer, int role, int rank) {
    if (layer < 0) {
        /* Global roles: the embedding is loaded per device, the LM head and the final
         * norm only on the logits device. */
        DeviceCtx &ctx = eng->ctx[rank == 0 ? 0 : eng->num_devices - 1];
        if (role == ROLE_EMBED) return ctx.embed_w;
        if (role == ROLE_LM_HEAD) return ctx.lm_head_w;
        if (role == ROLE_FINAL_NORM) return ctx.final_norm_w;
        return nullptr;
    }
    LayerWeights &lw = eng->replicated ? eng->layers[layer * eng->num_devices + rank]
                                      : eng->layers[layer];
    __nv_bfloat16 **target = role_target(lw, role);
    return target == nullptr ? nullptr : *target;
}

/* The FP32 derived copy of a role's BF16 weight, when the engine keeps one. The GDN
 * norm weight is the case the plan names: it is cast once at load and the forward
 * reads the FP32 copy, so an update that refreshed only the BF16 source would leave
 * the model reading the old weight. */
float *layer_role_derived(EngineHandle *eng, int layer, int role, int rank) {
    if (role != ROLE_GDN_NORM || layer < 0) return nullptr;
    LayerWeights &lw = eng->replicated ? eng->layers[layer * eng->num_devices + rank]
                                      : eng->layers[layer];
    return lw.gdn_norm_f32;
}

}  // namespace

TrainStore *engine_train_attach(EngineHandle *eng, const struct TrainAttachOptions *options) {
    if (eng == nullptr) {
        set_error("engine_train_attach: null engine");
        return nullptr;
    }
    if (eng->train_store != nullptr) return eng->train_store;
    const struct TrainAttachOptions defaults = {0, nullptr, 0};
    if (options == nullptr) options = &defaults;

    /* One spec per (layer, role) the checkpoint actually provides. The element count
     * comes from the loaded tensor, so a write cannot overrun a buffer. */
    std::vector<TrainParamSpec> specs;
    std::vector<std::string> templates;
    std::vector<std::string> names;
    for (int i = 0; i < eng->desc.role_count; ++i) {
        const int role = eng->desc.role_ids[i];
        const bool global = role == ROLE_EMBED || role == ROLE_LM_HEAD || role == ROLE_FINAL_NORM;
        const int layers = global ? 1 : eng->num_layers;
        for (int layer = 0; layer < layers; ++layer) {
            const int scope_layer = global ? -1 : layer;
            const TensorInfo *ti = role_tensor(eng, role, global ? 0 : layer);
            if (ti == nullptr) continue;
            if (layer_role_buffer(eng, scope_layer, role, 0) == nullptr) continue;
            int frozen = 0;
            for (int f = 0; f < options->frozen_role_count; ++f) {
                if (options->frozen_roles[f] == role) frozen = 1;
            }
            TrainParamSpec spec;
            spec.layer = scope_layer;
            spec.role = role;
            spec.elements = tensor_elements(*ti);
            spec.frozen = frozen;
            spec.trainable = 1;
            specs.push_back(spec);
            templates.push_back(eng->desc.role_templates[i]);
            names.push_back(model_desc_role_name(role));
            /* A derived FP32 copy gets its own logical identity so the store can
             * track it: it is not in the checkpoint, so it is frozen and trainable=0. */
            if (layer_role_derived(eng, scope_layer, role, 0) != nullptr) {
                TrainParamSpec copy = spec;
                copy.role = TRAIN_DERIVED_ROLE_BASE;
                copy.elements = eng->dims.gdn_head_dim;
                copy.frozen = 1;
                copy.trainable = 0;
                specs.push_back(copy);
                /* A distinct template text: the store ties specs by (layer, template),
                 * and a derived buffer is a *different* parameter from its source -
                 * sharing the text would make it an alias of the source and the
                 * trainability disagreement would be rejected (correctly). */
                templates.push_back(std::string(eng->desc.role_templates[i]) + "#f32");
                names.push_back("gdnNormF32");
            }
        }
    }
    if (specs.empty()) {
        set_error("engine_train_attach: the engine has no loadable parameter");
        return nullptr;
    }
    std::vector<const char *> template_ptrs(templates.size());
    std::vector<const char *> name_ptrs(names.size());
    for (size_t i = 0; i < templates.size(); ++i) {
        template_ptrs[i] = templates[i].c_str();
        name_ptrs[i] = names[i].c_str();
    }
    TrainStore *store = train_store_create(specs.data(), template_ptrs.data(), name_ptrs.data(),
                                           (int)specs.size());
    if (store == nullptr) {
        set_error("engine_train_attach: %s", train_last_error());
        return nullptr;
    }

    /* Wire the buffers. COMPUTE is the engine's own weight, so nothing is duplicated
     * and an inference forward sees a published update by construction. */
    for (int i = 0; i < (int)specs.size(); ++i) {
        const int logical = train_store_logical_of(store, specs[i].layer, specs[i].role);
        if (logical < 0) continue;
        __nv_bfloat16 *compute = layer_role_buffer(eng, specs[i].layer, specs[i].role, 0);
        if (compute != nullptr) train_store_set_slot(store, logical, TRAIN_SLOT_COMPUTE, compute);
        if (specs[i].role == TRAIN_DERIVED_ROLE_BASE) {
            float *derived = layer_role_derived(eng, specs[i].layer, ROLE_GDN_NORM, 0);
            train_store_set_slot(store, logical, TRAIN_SLOT_COMPUTE, derived);
            const int source = train_store_logical_of(store, specs[i].layer, ROLE_GDN_NORM);
            if (source >= 0 && derived != nullptr) {
                train_store_register_derived(store, source, logical, TRAIN_DERIVED_BF16_TO_FP32);
            }
            continue;
        }
        if (!specs[i].trainable || specs[i].frozen) continue;
        if (!options->allocate_training_state) continue;
        /* FP32 master, gradient and two optimizer slots, on the device that owns the
         * parameter. A frozen parameter gets none of this, which is what "frozen
         * parameters omit unused training state" means in memory. */
        const int rank_device_index =
            specs[i].layer < 0 ? (specs[i].role == ROLE_EMBED ? 0 : eng->num_devices - 1)
                               : eng->layer_device[specs[i].layer];
        const int device = eng->devices[rank_device_index];
        const size_t bytes = (size_t)specs[i].elements * sizeof(float);
        check_cuda(cudaSetDevice(device), "Select training device");
        float *master = nullptr;
        float *grad = nullptr;
        float *slot_m = nullptr;
        float *slot_v = nullptr;
        check_cuda(cudaMalloc(&master, bytes), "Allocate master weight");
        /* The master starts as *the loaded weight in FP32*, not as zero. A publication
         * casts every master into its compute weight, so a zeroed master would make
         * the first update silently wipe every parameter the caller did not write -
         * which is exactly what the Stage-3 gate caught. */
        __nv_bfloat16 *initial = layer_role_buffer(eng, specs[i].layer, specs[i].role, 0);
        if (initial != nullptr) {
            kernel_cast_bf16_f32(master, initial, (int)specs[i].elements, eng->ctx[0].stream);
            check_cuda(cudaGetLastError(), "Seed the master weight");
            check_cuda(cudaStreamSynchronize(eng->ctx[0].stream), "Finish master seed");
        }
        check_cuda(cudaMalloc(&grad, bytes), "Allocate gradient");
        check_cuda(cudaMemset(grad, 0, bytes), "Clear gradient");
        check_cuda(cudaMalloc(&slot_m, bytes), "Allocate optimizer slot m");
        check_cuda(cudaMemset(slot_m, 0, bytes), "Clear optimizer slot m");
        check_cuda(cudaMalloc(&slot_v, bytes), "Allocate optimizer slot v");
        check_cuda(cudaMemset(slot_v, 0, bytes), "Clear optimizer slot v");
        /* The allocations are owned by the store, so a failed attach cannot leak
         * them: the caller's cleanup path is train_store_release_buffers. */
        train_store_set_slot(store, logical, TRAIN_SLOT_MASTER, master);
        train_store_set_slot(store, logical, TRAIN_SLOT_GRAD, grad);
        train_store_set_slot(store, logical, TRAIN_SLOT_OPT_M, slot_m);
        train_store_set_slot(store, logical, TRAIN_SLOT_OPT_V, slot_v);
        eng->train_buffers.push_back(master);
        eng->train_buffers.push_back(grad);
        eng->train_buffers.push_back(slot_m);
        eng->train_buffers.push_back(slot_v);
    }
    /* Replicas: a replicated parameter has one buffer per rank, and the plan forbids
     * independent optimizers, so the sync requirement is recorded per parameter. */
    if (eng->replicated) {
        for (int i = 0; i < (int)specs.size(); ++i) {
            const int logical = train_store_logical_of(store, specs[i].layer, specs[i].role);
            if (logical >= 0) {
                std::vector<int> ordinals(eng->devices.begin(), eng->devices.end());
                train_store_set_replica_devices(store, logical, (int)ordinals.size(),
                                                ordinals.data());
            }
        }
    }
    eng->train_store = store;
    return store;
}

TrainStore *engine_train_store(EngineHandle *eng) { return eng ? eng->train_store : nullptr; }

namespace {

/* The CUDA device a logical parameter's buffers live on. */
int train_param_device(EngineHandle *eng, int logical) {
    const int index = train_store_alias_at(eng->train_store, logical, 0);
    if (index < 0) return eng->devices[0];
    const int layer = train_store_spec_layer(eng->train_store, index);
    const int role = train_store_spec_role(eng->train_store, index);
    if (layer < 0) return role == ROLE_EMBED ? eng->devices[0] : eng->devices[eng->num_devices - 1];
    return eng->devices[eng->layer_device[layer]];
}

}  // namespace

int engine_train_begin_update(EngineHandle *eng) {
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_begin_update: no training store attached");
        return ENGINE_ERR_STATE;
    }
    const TrainStatus status = train_store_begin_update(eng->train_store);
    if (status != TRAIN_OK) {
        set_error("engine_train_begin_update: %s", train_last_error());
        return ENGINE_ERR_STATE;
    }
    return ENGINE_OK;
}

int engine_train_write_master(EngineHandle *eng, int logical, const float *host_values) {
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_write_master: no training store attached");
        return ENGINE_ERR_STATE;
    }
    if (!train_store_in_update(eng->train_store)) {
        set_error("engine_train_write_master: no update window is open");
        return ENGINE_ERR_STATE;
    }
    float *master = (float *)train_store_slot(eng->train_store, logical, TRAIN_SLOT_MASTER);
    if (master == nullptr) {
        set_error("engine_train_write_master: parameter %d has no master weight (frozen or "
                  "training state not allocated)", logical);
        return ENGINE_ERR_WEIGHTS;
    }
    const long long elements = train_store_elements(eng->train_store, logical);
    /* The upload runs on the engine's own stream and is synchronised before returning.
     * A blocking cudaMemcpy from pageable host memory only guarantees the bytes reached
     * the driver's staging buffer, not the destination; the final DMA is ordered in the
     * calling thread's stream, and every kernel that reads the master runs on a context
     * stream instead. The publication could therefore cast a half-written master - the
     * perturbation was occasionally published as garbage while a no-op write, whose
     * master already equalled the loaded weight, hid it. `engine_train_import_state`
     * already used this stream-ordered form; this writer is the one that did not. */
    const int device = train_param_device(eng, logical);
    int ci = 0;
    for (int i = 0; i < eng->num_devices; ++i) {
        if (eng->ctx[i].device_id == device) {
            ci = i;
            break;
        }
    }
    check_cuda(cudaSetDevice(device), "Select training device");
    check_cuda(cudaMemcpyAsync(master, host_values, (size_t)elements * sizeof(float),
                               cudaMemcpyHostToDevice, eng->ctx[ci].stream),
               "Upload master weight");
    check_cuda(cudaStreamSynchronize(eng->ctx[ci].stream), "Finish master upload");
    return ENGINE_OK;
}

/* The publication: every written master becomes the BF16 compute weight of every
 * reader, and then every derived copy is recomputed from its source. Doing both here
 * is what the plan means by "publishing follows BF16 casting and all derived-copy
 * refreshes (in particular refresh gdn_norm_f32, not just its BF16 source)": the
 * store marks the copies stale on publish and refuses to end the window while one is
 * still stale, so the refresh cannot be skipped. */
int engine_train_publish(EngineHandle *eng) {
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_publish: no training store attached");
        return ENGINE_ERR_STATE;
    }
    TrainStore *store = eng->train_store;
    if (!train_store_in_update(store)) {
        set_error("engine_train_publish: no update window is open");
        return ENGINE_ERR_STATE;
    }
    /* Which DeviceCtx holds a rank's copy of a parameter: a replicated engine has one
     * per rank, a layer-split engine keeps the embedding on the first device and the
     * LM head and final norm on the logits device. */
    auto ctx_index_for = [&](int layer, int role, int rank) -> int {
        if (eng->replicated) return rank;
        if (layer < 0) return role == ROLE_EMBED ? 0 : eng->num_devices - 1;
        return eng->layer_device[layer];
    };
    try {
        /* 1. Every alias of every parameter that has a master becomes the BF16 weight
         * its readers load. A tied parameter has one entry per reader, and both are
         * written: that is what resolving the tie in the store buys. */
        for (int logical = 0; logical < train_store_logical_count(store); ++logical) {
            float *master = (float *)train_store_slot(store, logical, TRAIN_SLOT_MASTER);
            if (master == nullptr) continue;
            const long long elements = train_store_elements(store, logical);
            const int aliases = train_store_alias_count(store, logical);
            const int ranks = eng->replicated ? eng->num_devices : 1;
            for (int a = 0; a < aliases; ++a) {
                const int index = train_store_alias_at(store, logical, a);
                const int layer = train_store_spec_layer(store, index);
                const int role = train_store_spec_role(store, index);
                if (role == TRAIN_DERIVED_ROLE_BASE) continue;
                for (int r = 0; r < ranks; ++r) {
                    __nv_bfloat16 *compute = layer_role_buffer(eng, layer, role, r);
                    if (compute == nullptr) continue;
                    const int ci = ctx_index_for(layer, role, r);
                    check_cuda(cudaSetDevice(eng->devices[ci]), "Select publish device");
                    kernel_cast_f32_bf16(compute, master, (int)elements, eng->ctx[ci].stream);
                    check_cuda(cudaGetLastError(), "Cast master to compute weight");
                    check_cuda(cudaStreamSynchronize(eng->ctx[ci].stream),
                               "Finish weight publication");
                }
            }
        }
        /* 2. The version moves and every derived copy goes stale. */
        const TrainStatus published = train_store_publish(store);
        if (published != TRAIN_OK) {
            set_error("engine_train_publish: %s", train_last_error());
            return ENGINE_ERR_STATE;
        }
        /* 3. Refresh them from the source that was just written, then close. The
         * store refuses to close while a stale copy remains, so this step cannot be
         * skipped without the failure being visible. */
        const int ranks = eng->replicated ? eng->num_devices : 1;
        for (int i = 0; i < train_store_derived_count(store); ++i) {
            int source = -1, derived = -1;
            TrainDerivedKind kind = TRAIN_DERIVED_BF16_TO_FP32;
            int stale = 0;
            if (train_store_derived_at(store, i, &source, &derived, &kind, &stale) != TRAIN_OK)
                continue;
            const int layer = train_store_spec_layer(store, train_store_alias_at(store, derived, 0));
            const long long elements = train_store_elements(store, derived);
            for (int r = 0; r < ranks; ++r) {
                float *target = layer_role_derived(eng, layer, ROLE_GDN_NORM, r);
                __nv_bfloat16 *from = layer_role_buffer(eng, layer, ROLE_GDN_NORM, r);
                if (target == nullptr || from == nullptr) continue;
                const int ci = ctx_index_for(layer, ROLE_GDN_NORM, r);
                check_cuda(cudaSetDevice(eng->devices[ci]), "Select derived-refresh device");
                kernel_cast_bf16_f32(target, from, (int)elements, eng->ctx[ci].stream);
                check_cuda(cudaGetLastError(), "Refresh derived copy");
                check_cuda(cudaStreamSynchronize(eng->ctx[ci].stream), "Finish derived refresh");
            }
            train_store_derived_refreshed(store, i);
        }
        const TrainStatus closed = train_store_end_update(store);
        if (closed != TRAIN_OK) {
            set_error("engine_train_publish: %s", train_last_error());
            return ENGINE_ERR_STATE;
        }
    } catch (const std::exception &e) {
        set_error("engine_train_publish: %s", e.what());
        return ENGINE_ERR_CUDA;
    }
    return ENGINE_OK;
}

int engine_train_end_update(EngineHandle *eng) {
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_end_update: no training store attached");
        return ENGINE_ERR_STATE;
    }
    const TrainStatus status = train_store_end_update(eng->train_store);
    if (status != TRAIN_OK) {
        set_error("engine_train_end_update: %s", train_last_error());
        return ENGINE_ERR_STATE;
    }
    return ENGINE_OK;
}

/* Teacher forcing over one sequence. The LM head is evaluated one row at a time into
 * the engine's single-row logits buffer, so nothing here ever materialises a
 * [tokens, vocab] tensor on the device: the plan's "chunked/fused LM-head/loss
 * evaluation". `output->all_logits`, when asked for, is the debug/reference path and
 * is the caller's own host buffer. */
int engine_train_forward(EngineHandle *eng, const int *token_ids, int tokens,
                         const int64_t *positions, const int *labels, const uint8_t *mask,
                         int shift, struct TrainForwardOutput *output) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || !eng->state_valid) {
        set_error("engine_train_forward: engine is null or needs engine_reset");
        return ENGINE_ERR_STATE;
    }
    if (token_ids == nullptr || output == nullptr || tokens < 1) {
        set_error("engine_train_forward: tokens and an output structure are required");
        return ENGINE_ERR_CONFIG;
    }
    if (output->all_logits == nullptr && output->logprobs == nullptr) {
        set_error("engine_train_forward: ask for all logits, log-probabilities, or both");
        return ENGINE_ERR_CONFIG;
    }
    if (tokens > eng->dims.max_seq_len || tokens > eng->dims.max_chunk) {
        set_error("engine_train_forward: sequence length %d exceeds the %d-token chunk",
                  tokens, eng->dims.max_chunk);
        return ENGINE_ERR_SEQ_FULL;
    }
    /* One sequence at a time, from the start: a teacher-forced step is a full-sequence
     * forward, and continuing a half-consumed sequence would silently mix two
     * traversals (the plan forbids flattening independent sequences). */
    if (eng->seq_len != 0) {
        set_error("engine_train_forward: the sequence is at position %d; engine_reset first",
                  eng->seq_len);
        return ENGINE_ERR_STATE;
    }
    if (shift < 1) {
        set_error("engine_train_forward: shift must be at least 1");
        return ENGINE_ERR_CONFIG;
    }
    for (int t = 0; t < tokens; ++t) {
        if (token_ids[t] < 0 || token_ids[t] >= eng->dims.vocab_size) {
            set_error("engine_train_forward: token %d at offset %d is out of range", token_ids[t], t);
            return ENGINE_ERR_CONFIG;
        }
    }

    /* The selection is the pure plan from train.c, so the same mapping is testable on
     * the CPU. */
    std::vector<int> ids(token_ids, token_ids + tokens);
    std::vector<TrainForcedPosition> selected(tokens);
    const int selected_count = train_plan_teacher_forcing(tokens, ids.data(), labels, mask,
                                                          positions, shift, selected.data(),
                                                          (int)selected.size());
    if (selected_count < 0) {
        set_error("engine_train_forward: %s", train_last_error());
        return ENGINE_ERR_CONFIG;
    }
    /* A caller that asked only for all-logits (the reference/debug path) gets the
     * selection reported and nothing else; a caller that asked for log-probabilities
     * with nothing to compute them for is told, because that is a silent no-op. */
    if (output->all_logits == nullptr && output->logprobs == nullptr) {
        set_error("engine_train_forward: nothing was asked for");
        return ENGINE_ERR_CONFIG;
    }

    try {
        std::vector<int64_t> ids64(token_ids, token_ids + tokens);
        __nv_bfloat16 *act = nullptr;
        DeviceCtx *last = nullptr;
        forward_tokens(eng, ids64.data(), tokens, nullptr, positions, &act, &last);
        if (last == nullptr || act == nullptr) {
            set_error("engine_train_forward: the forward produced no hidden states");
            return ENGINE_ERR_CUDA;
        }
        const int hidden = eng->dims.hidden_size;
        const int vocab = eng->dims.vocab_size;
        check_cuda(cudaSetDevice(last->device_id), "Select logits device");
        if (eng->train_scratch == nullptr) {
            check_cuda(cudaMalloc(&eng->train_scratch, (size_t)eng->dims.max_chunk * sizeof(float)),
                       "Allocate training log-probability scratch");
            check_cuda(cudaMalloc(&eng->train_labels, (size_t)eng->dims.max_chunk * sizeof(int)),
                       "Allocate training label scratch");
        }

        auto norm_row = [&](int row) {
            const __nv_bfloat16 *x = act + (size_t)row * hidden;
            if (eng->dims.norm_style == 1) {
                kernel_rms_norm_plain(last->workspace, x, last->final_norm_w, hidden, 1,
                                      eng->dims.rms_eps, last->stream);
            } else {
                kernel_gemma_rms_norm(last->workspace, x, last->final_norm_w, hidden, 1,
                                      eng->dims.rms_eps, last->stream);
            }
            check_cuda(cudaGetLastError(), "Final norm for one row");
            check_forward(gemm_bf16_f32out(last->cublas, last->d_logits, last->workspace,
                                          last->lm_head_w, 1, vocab, hidden), "LM head for one row");
        };

        if (output->all_logits != nullptr) {
            for (int row = 0; row < tokens; ++row) {
                norm_row(row);
                check_cuda(cudaMemcpyAsync(output->all_logits + (size_t)row * vocab, last->d_logits,
                                           (size_t)vocab * sizeof(float), cudaMemcpyDeviceToHost,
                                           last->stream), "Download one row of logits");
            }
            check_cuda(cudaStreamSynchronize(last->stream), "Finish logits download");
        }

        if (output->logprobs != nullptr && selected_count > 0) {
            if (output->all_logits != nullptr) {
                /* The rows are already here: take the natural-log log-softmax on the
                 * host rather than reading the device again. */
                for (int j = 0; j < selected_count; ++j) {
                    const float *row = output->all_logits + (size_t)selected[j].query * vocab;
                    float row_max = -INFINITY;
                    for (int v = 0; v < vocab; ++v) row_max = std::max(row_max, row[v]);
                    double sum = 0.0;
                    for (int v = 0; v < vocab; ++v) sum += std::exp((double)row[v] - row_max);
                    const int label = selected[j].label;
                    if (label < 0 || label >= vocab) {
                        set_error("engine_train_forward: label %d is out of range", label);
                        return ENGINE_ERR_CONFIG;
                    }
                    output->logprobs[j] = (float)((double)row[label] - (row_max + std::log(sum)));
                }
            } else {
                /* Fused: the row never leaves the device; only the scalar comes back. */
                for (int j = 0; j < selected_count; ++j) {
                    norm_row(selected[j].query);
                    std::vector<int> one{selected[j].label};
                    check_cuda(cudaMemcpyAsync(eng->train_labels + j, one.data(), sizeof(int),
                                               cudaMemcpyHostToDevice, last->stream),
                               "Upload the label");
                    kernel_logprob_gather(eng->train_scratch + j, last->d_logits,
                                          eng->train_labels + j, 1, vocab, last->stream);
                }
                check_cuda(cudaMemcpyAsync(output->logprobs, eng->train_scratch,
                                           (size_t)selected_count * sizeof(float),
                                           cudaMemcpyDeviceToHost, last->stream),
                           "Download log-probabilities");
                check_cuda(cudaStreamSynchronize(last->stream), "Finish logprob download");
            }
        }
        if (output->selected != nullptr) {
            for (int j = 0; j < selected_count; ++j) output->selected[j] = selected[j].query;
        }
        output->selected_count = selected_count;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
    return ENGINE_OK;
}

/* The training step's retention plan: what a sequence's backward needs, per layer.
 * The activation list is per layer because the residual stream is updated in place -
 * which is exactly why the plan requires a training-step context to retain them
 * rather than read them later. A fixed backward sequence (Stage 4) means the free
 * points can be per-layer; it does not mean the values can be skipped. */
/* FP32 elements one layer's retained chunk-boundary states cost. */
long long train_step_gdn_state_elements_plan(int chunk_count, int value_heads, int head_dim);

namespace {

struct StepValue {
    std::string name;
    int layer;
    long long elements;
    int alias_of;
    int free_after;
    int device_index;
};

std::vector<StepValue> step_plan(const EngineHandle *eng, int tokens, int chunk_count) {
    std::vector<StepValue> values;
    const long long activation = (long long)tokens * eng->dims.hidden_size;
    for (int layer = 0; layer < eng->num_layers; ++layer) {
        const int device_index = eng->replicated ? 0 : eng->layer_device[layer];
        const std::string prefix = "layer" + std::to_string(layer) + ".";
        /* The mixer and the ffn outputs are consumed by that layer's backward; the
         * residual is too, and cannot be re-read because the next layer overwrites it. */
        values.push_back({prefix + "mixerOut", layer, activation, -1, 1, device_index});
        values.push_back({prefix + "ffnOut", layer, activation, -1, 1, device_index});
        values.push_back({prefix + "residual", layer, activation, -1, 1, device_index});
        /* A GDN layer's chunk-boundary states are what let a full-sequence gradient
         * cross the internal chunk boundaries instead of being truncated at them. */
        const int mixer = eng->desc.layer_mixers[layer];
        if (mixer == ENGINE_MIXER_GDN) {
            const long long state = train_step_gdn_state_elements_plan(
                chunk_count, eng->dims.gdn_num_v_heads, eng->dims.gdn_head_dim);
            values.push_back({prefix + "gdnChunkState", layer, state, -1, 2, device_index});
        }
    }
    return values;
}

}  // namespace

long long train_step_gdn_state_elements_plan(int chunk_count, int value_heads, int head_dim) {
    return (long long)chunk_count * value_heads * head_dim * head_dim;
}

int engine_train_step_plan(EngineHandle *eng, int tokens, int chunk_count, int *saved_count,
                          long long *gdn_state_elements) {
    if (eng == nullptr || tokens < 1 || chunk_count < 1) {
        set_error("engine_train_step_plan: tokens and chunk_count must be positive");
        return ENGINE_ERR_CONFIG;
    }
    const std::vector<StepValue> values = step_plan(eng, tokens, chunk_count);
    long long gdn = 0;
    for (const StepValue &v : values) {
        if (v.name.find("gdnChunkState") != std::string::npos) gdn += v.elements;
    }
    if (saved_count) *saved_count = (int)values.size();
    if (gdn_state_elements) *gdn_state_elements = gdn;
    return ENGINE_OK;
}

int engine_train_step_begin(EngineHandle *eng, int tokens, int chunk_count,
                           struct TrainStep **out_step) {
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_step_begin: no training store attached");
        return ENGINE_ERR_STATE;
    }
    if (out_step == nullptr) {
        set_error("engine_train_step_begin: null output");
        return ENGINE_ERR_CONFIG;
    }
    *out_step = nullptr;
    if (tokens < 1 || tokens > eng->dims.max_chunk || chunk_count < 1) {
        set_error("engine_train_step_begin: tokens in [1, %d] and chunk_count >= 1 are required",
                  eng->dims.max_chunk);
        return ENGINE_ERR_CONFIG;
    }
    const std::vector<StepValue> values = step_plan(eng, tokens, chunk_count);
    if (values.empty()) {
        set_error("engine_train_step_begin: the model has no layers to retain");
        return ENGINE_ERR_STATE;
    }
    std::vector<TrainSavedSpec> specs(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        specs[i].name = values[i].name.c_str();
        specs[i].layer = values[i].layer;
        specs[i].elements = values[i].elements;
        specs[i].alias_of = values[i].alias_of;
        specs[i].free_after = values[i].free_after;
    }
    TrainStep *step = train_step_create(eng->train_store, specs.data(), (int)specs.size());
    if (step == nullptr) {
        set_error("engine_train_step_begin: %s", train_last_error());
        return ENGINE_ERR_STATE;
    }
    if (train_step_set_bptt(step, TRAIN_BPTT_FULL_SEQUENCE, chunk_count) != TRAIN_OK) {
        set_error("engine_train_step_begin: %s", train_last_error());
        train_step_destroy(step);
        return ENGINE_ERR_STATE;
    }
    /* Retain the buffers. The values are *allocated and kept*: a step that recorded
     * what it needs but held nothing would not be a retention at all. */
    for (size_t i = 0; i < values.size(); ++i) {
        const int ci = values[i].device_index;
        check_cuda(cudaSetDevice(eng->devices[ci]), "Select retention device");
        void *buffer = nullptr;
        const size_t bytes = (size_t)values[i].elements * sizeof(float);
        if (cudaMalloc(&buffer, bytes) != cudaSuccess || buffer == nullptr) {
            set_error("engine_train_step_begin: cannot retain '%s'", values[i].name.c_str());
            train_step_destroy(step);
            return ENGINE_ERR_ALLOC;
        }
        /* The retained bytes are the engine's, so the step's cleanup frees them
         * through the same list; a step whose creation failed has already released
         * everything it retained. */
        eng->train_buffers.push_back(buffer);
        if (train_step_retain(step, (int)i, buffer) != TRAIN_OK) {
            set_error("engine_train_step_begin: %s", train_last_error());
            train_step_destroy(step);
            return ENGINE_ERR_STATE;
        }
    }
    *out_step = step;
    return ENGINE_OK;
}

int engine_train_step_end(struct TrainStep *step) {
    if (step == nullptr) {
        set_error("engine_train_step_end: null step");
        return ENGINE_ERR_CONFIG;
    }
    /* Release in the order the free points allow: the aliases first, then the value
     * they alias. A caller that skips a release is told by train_step_destroy. */
    const int count = train_step_saved_count(step);
    for (int i = count - 1; i >= 0; --i) {
        const TrainStatus status = train_step_free(step, i);
        if (status != TRAIN_OK && status != TRAIN_ERR_STATE) {
            set_error("engine_train_step_end: %s", train_last_error());
            return ENGINE_ERR_STATE;
        }
    }
    const TrainStatus status = train_step_destroy(step);
    if (status != TRAIN_OK) {
        set_error("engine_train_step_end: %s", train_last_error());
        return ENGINE_ERR_STATE;
    }
    return ENGINE_OK;
}

/* ==================================================================== */
/* Stage 5: the SFT step                                                */
/* ==================================================================== */

namespace {

/* (layer, role) -> the BF16 compute weight the forward reads and the FP32 gradient
 * accumulator the optimizer consumes. A role the layer does not have has both null; a
 * frozen role has a compute weight and no gradient, which is why they are separate
 * answers rather than one "exists" flag. */
struct RoleAccess {
    __nv_bfloat16 *compute;
    float *grad;
};

RoleAccess role_access(EngineHandle *eng, int layer, int role) {
    RoleAccess out{nullptr, nullptr};
    out.compute = layer_role_buffer(eng, layer, role, 0);
    if (eng->train_store != nullptr) {
        const int logical = train_store_logical_of(eng->train_store, layer, role);
        if (logical >= 0 && train_store_is_trainable(eng->train_store, logical) == 1) {
            out.grad = static_cast<float *>(
                train_store_slot(eng->train_store, logical, TRAIN_SLOT_GRAD));
        }
    }
    return out;
}

struct TrainRole train_role(EngineHandle *eng, int layer, int role) {
    RoleAccess access = role_access(eng, layer, role);
    struct TrainRole out;
    out.compute = access.compute;
    out.grad = access.grad;
    return out;
}

/* The step's retained buffer for one name, or null. The plan's names are how the
 * forward and the backward agree on which buffer holds which boundary. */
void *step_value(struct TrainStep *step, const std::string &name) {
    const int count = train_step_saved_count(step);
    for (int i = 0; i < count; ++i) {
        const char *saved_name = nullptr;
        int layer = 0;
        long long elements = 0;
        void *buffer = nullptr;
        if (train_step_saved_at(step, i, &saved_name, &layer, &elements, &buffer) != TRAIN_OK) {
            continue;
        }
        if (saved_name != nullptr && name == saved_name) return buffer;
    }
    return nullptr;
}

/* Size the step's pool for `tokens`. The layer backward reports what it needs, so the
 * pool is the sum of reported sizes rather than a formula duplicated here. */
int ensure_train_pool(EngineHandle *eng, int tokens) {
    if (eng->train_pool != nullptr && eng->train_pool_tokens >= tokens) return ENGINE_OK;
    const int H = eng->dims.hidden_size;
    const int V = eng->dims.vocab_size;
    const size_t TH = (size_t)tokens * H;

    struct LayerBackwardScratch scratch{};
    size_t cast = 0;
    size_t grad = 0;
    for (int layer = 0; layer < eng->num_layers; ++layer) {
        layer_backward_scratch(&eng->dims, tokens, eng->desc.layer_mixers[layer],
                               eng->desc.layer_ffns[layer], &scratch);
        cast = std::max(cast, (size_t)scratch.cast_elements);
        grad = std::max(grad, (size_t)scratch.grad_elements);
    }

    struct EngineHandle::TrainStepOffsets off;
    off.final_hidden = 0;
    off.d_hidden = off.final_hidden + TH;
    off.d_next = off.d_hidden + TH;
    off.lm_weight = off.d_next + TH;
    off.logits_row = off.lm_weight + (size_t)V * H;
    off.d_logits_row = off.logits_row + V;
    off.row_hidden = off.d_logits_row + V;
    off.row_normed = off.row_hidden + H;
    /* row_extra holds four scalars per row: the loss slope, the row's log-probability,
     * the inverse RMS the final norm's backward consumes, and a spare. They must be
     * distinct: an earlier version reused one slot for the log-probability and the
     * inverse RMS and reported a *negative* cross-entropy, which is how the collision
     * was found. */
    off.row_extra = off.row_normed + H;
    off.layer_cast = off.row_extra + 4;
    off.layer_grad = off.layer_cast + cast;
    off.total = off.layer_grad + grad;

    if (eng->train_pool != nullptr) {
        cleanup_cuda(cudaFree(eng->train_pool));
        eng->train_pool = nullptr;
    }
    const int logits_index = eng->num_devices > 0 ? eng->num_devices - 1 : 0;
    const cudaError_t selected = cudaSetDevice(eng->ctx[logits_index].device_id);
    if (selected != cudaSuccess) {
        set_error("ensure_train_pool: cannot select the device");
        return ENGINE_ERR_CUDA;
    }
    if (cudaMalloc(&eng->train_pool, off.total * sizeof(float)) != cudaSuccess ||
        eng->train_pool == nullptr) {
        set_error("ensure_train_pool: cannot allocate %zu bytes", off.total * sizeof(float));
        return ENGINE_ERR_ALLOC;
    }
    /* The Stage-3 forward allocated these lazily on its own path; the SFT step needs
     * them too, and only on the device that holds the LM head. */
    if (eng->train_labels == nullptr) {
        if (cudaMalloc(&eng->train_labels, (size_t)eng->dims.max_chunk * sizeof(int)) !=
            cudaSuccess) {
            set_error("ensure_train_pool: cannot allocate the label scratch");
            return ENGINE_ERR_ALLOC;
        }
    }
    if (eng->train_scratch == nullptr) {
        if (cudaMalloc(&eng->train_scratch, (size_t)eng->dims.max_chunk * sizeof(float)) !=
            cudaSuccess) {
            set_error("ensure_train_pool: cannot allocate the loss scratch");
            return ENGINE_ERR_ALLOC;
        }
    }
    eng->train_off = off;
    eng->train_pool_floats = off.total;
    eng->train_pool_tokens = tokens;
    return ENGINE_OK;
}

/* Copy a retained boundary into the step's FP32 buffer as the widening of the BF16
 * activation the forward produced. */
void retain_boundary(struct TrainStep *step, const std::string &name,
                     const __nv_bfloat16 *src, size_t elements, cudaStream_t stream) {
    void *dst = step_value(step, name);
    if (dst == nullptr) {
        throw EngineError(ENGINE_ERR_STATE, "the step did not retain " + name);
    }
    kernel_cast_bf16_f32(static_cast<float *>(dst), src, (int)elements, stream);
    check_cuda(cudaGetLastError(), "Retain a training boundary");
}

}  // namespace

int engine_train_forward_retain(EngineHandle *eng, struct TrainStep *step, const int *token_ids,
                                const int64_t *positions, int tokens) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || !eng->state_valid) {
        set_error("engine_train_forward_retain: engine is null or needs engine_reset");
        return ENGINE_ERR_STATE;
    }
    if (eng->train_store == nullptr) {
        set_error("engine_train_forward_retain: no training store attached");
        return ENGINE_ERR_STATE;
    }
    if (step == nullptr || token_ids == nullptr || positions == nullptr || tokens < 1) {
        set_error("engine_train_forward_retain: a step, tokens and positions are required");
        return ENGINE_ERR_CONFIG;
    }
    if (tokens > eng->dims.max_chunk) {
        set_error("engine_train_forward_retain: %d tokens exceed the %d-token chunk", tokens,
                  eng->dims.max_chunk);
        return ENGINE_ERR_SEQ_FULL;
    }
    if (eng->seq_len != 0) {
        set_error("engine_train_forward_retain: the sequence is at position %d; reset first",
                  eng->seq_len);
        return ENGINE_ERR_STATE;
    }
    /* A pipeline split would have to move gradients between devices and a replicated
     * (tensor-parallel) placement would have to reduce sharded ones. Both are real work
     * the plan places after this bring-up, and both are refused rather than approximated
     * silently. */
    if (eng->replicated || (eng->num_devices > 1 && !eng->replicated)) {
        set_error("engine_train_forward_retain: training is wired for a single device; "
                  "a pipeline or tensor-parallel placement is not supported yet");
        return ENGINE_ERR_CONFIG;
    }
    for (int t = 0; t < tokens; ++t) {
        if (token_ids[t] < 0 || token_ids[t] >= eng->dims.vocab_size) {
            set_error("engine_train_forward_retain: token %d at offset %d is out of range",
                      token_ids[t], t);
            return ENGINE_ERR_CONFIG;
        }
    }
    const int status = ensure_train_pool(eng, tokens);
    if (status != ENGINE_OK) return status;

    try {
        DeviceCtx &ctx = eng->ctx[0];
        const int H = eng->dims.hidden_size;
        const size_t TH = (size_t)tokens * H;
        check_cuda(cudaSetDevice(ctx.device_id), "Select the training device");
        std::vector<int64_t> ids64(tokens);
        for (int t = 0; t < tokens; ++t) ids64[t] = token_ids[t];
        check_cuda(cudaMemcpyAsync(ctx.token_ids, ids64.data(), tokens * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, ctx.stream),
                   "Upload the training token IDs");
        check_cuda(cudaMemcpyAsync(ctx.positions, positions, tokens * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, ctx.stream),
                   "Upload the training positions");
        check_cuda(cudaStreamSynchronize(ctx.stream), "Finish the training upload");

        kernel_embedding(ctx.residual, ctx.embed_w, ctx.token_ids, H, tokens, ctx.stream);
        check_cuda(cudaGetLastError(), "Training embedding");

        for (int i = 0; i < eng->num_layers; ++i) {
            const std::string prefix = "layer" + std::to_string(i) + ".";
            LayerContext lctx = make_layer_context(eng, i, 0, tokens, /*with_taps=*/true);
            /* r0: the layer's input, before the mixer. */
            retain_boundary(step, prefix + "residual", ctx.residual, TH, ctx.stream);
            if (forward_mixer(&lctx, &eng->layers[i], ctx.residual, ctx.layer_out) != 0) {
                throw EngineError(ENGINE_ERR_CUDA, "training mixer failed");
            }
            retain_boundary(step, prefix + "mixerOut", ctx.layer_out, TH, ctx.stream);
            kernel_residual_add(ctx.residual, ctx.layer_out, (int)TH, ctx.stream);
            check_cuda(cudaGetLastError(), "Training mixer residual");
            if (forward_ffn(&lctx, &eng->layers[i], ctx.residual, ctx.layer_out) != 0) {
                throw EngineError(ENGINE_ERR_CUDA, "training feed-forward failed");
            }
            retain_boundary(step, prefix + "ffnOut", ctx.layer_out, TH, ctx.stream);
            kernel_residual_add(ctx.residual, ctx.layer_out, (int)TH, ctx.stream);
            check_cuda(cudaGetLastError(), "Training feed-forward residual");
        }
        /* The final hidden state: the loss and the final norm's backward read it after
         * the layer walk has reused the per-device activations. */
        kernel_cast_bf16_f32(eng->train_pool + eng->train_off.final_hidden, ctx.residual, (int)TH,
                             ctx.stream);
        check_cuda(cudaGetLastError(), "Retain the final hidden state");
        eng->seq_len += tokens;
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
}

int engine_train_loss(EngineHandle *eng, struct TrainStep *step, const int *token_ids,
                      const int *labels, const uint8_t *mask, int shift, int tokens,
                      struct TrainLossOutput *output) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || !eng->state_valid || eng->train_pool == nullptr) {
        set_error("engine_train_loss: run engine_train_forward_retain first");
        return ENGINE_ERR_STATE;
    }
    if (step == nullptr || token_ids == nullptr || labels == nullptr || output == nullptr ||
        tokens < 1) {
        set_error("engine_train_loss: a step, tokens and labels are required");
        return ENGINE_ERR_CONFIG;
    }
    if (tokens > eng->train_pool_tokens) {
        set_error("engine_train_loss: the pool was sized for %d tokens, not %d",
                  eng->train_pool_tokens, tokens);
        return ENGINE_ERR_STATE;
    }

    /* The same pure selection plan the Stage-3 forward uses, so the mapping is testable
     * on the CPU and identical between the two entry points. */
    std::vector<int64_t> positions(tokens);
    for (int t = 0; t < tokens; ++t) positions[t] = t;
    std::vector<int> ids(token_ids, token_ids + tokens);
    std::vector<TrainForcedPosition> selected(tokens);
    const int count = train_plan_teacher_forcing(tokens, ids.data(), labels, mask, positions.data(),
                                                 shift, selected.data(), (int)selected.size());
    if (count < 0) {
        set_error("engine_train_loss: %s", train_last_error());
        return ENGINE_ERR_CONFIG;
    }

    try {
        DeviceCtx &ctx = eng->ctx[0];
        const int H = eng->dims.hidden_size;
        const int V = eng->dims.vocab_size;
        const size_t TH = (size_t)tokens * H;
        float *pool = eng->train_pool;
        const struct EngineHandle::TrainStepOffsets &off = eng->train_off;
        check_cuda(cudaSetDevice(ctx.device_id), "Select the training device");
        output->sum = 0.0;
        output->count = count;
        if (count == 0) {
            /* Nothing selected: the loss is zero and so is the hidden state's gradient,
             * which is a result rather than an error. */
            check_cuda(cudaMemsetAsync(pool + off.d_hidden, 0, TH * sizeof(float), ctx.stream),
                       "Zero the hidden gradient");
            return ENGINE_OK;
        }
        check_cuda(cudaMemsetAsync(pool + off.d_hidden, 0, TH * sizeof(float), ctx.stream),
                   "Zero the hidden gradient");
        /* The LM head's operand: the FP32 widening of the BF16 weight the forward reads
         * (not the master, which differs by the publication rounding). */
        kernel_cast_bf16_f32(pool + off.lm_weight, ctx.lm_head_w, V * H, ctx.stream);
        check_cuda(cudaGetLastError(), "Widen the LM head");
        /* The log-softmax kernel multiplies the row by `(softmax - onehot)`, which is
         * already the *negative* log-likelihood's gradient with respect to the logits;
         * so the slope the loss contributes is the positive 1/count. Passing -1/count
         * inverted every parameter's gradient through the LM head (measured: the tied
         * embedding's gradient had cosine -0.998 against torch's, and the loss climbed). */
        const float slope = 1.0f / (float)count;
        /* The BF16 row scratch rides in the layer workspace: the loss runs before the
         * backward, so they never overlap in time. */
        __nv_bfloat16 *row_hidden = ctx.workspace;
        __nv_bfloat16 *row_normed = ctx.workspace + H;

        double total = 0.0;
        for (int j = 0; j < count; ++j) {
            const int row = selected[j].query;
            const int label = selected[j].label;
            if (label < 0 || label >= V) {
                set_error("engine_train_loss: label %d is out of range", label);
                return ENGINE_ERR_CONFIG;
            }
            /* The final norm for this row, from the retained hidden state. */
            kernel_cast_f32_bf16(row_hidden, pool + off.final_hidden + (size_t)row * H, H,
                                 ctx.stream);
            if (eng->dims.norm_style == 1) {
                kernel_rms_norm_plain(row_normed, row_hidden, ctx.final_norm_w, H, 1,
                                      eng->dims.rms_eps, ctx.stream);
            } else {
                kernel_gemma_rms_norm(row_normed, row_hidden, ctx.final_norm_w, H, 1,
                                      eng->dims.rms_eps, ctx.stream);
            }
            check_cuda(cudaGetLastError(), "Final norm for the loss");
            if (gemm_bf16_f32out(ctx.cublas, pool + off.logits_row, row_normed, ctx.lm_head_w, 1, V,
                                 H) != 0) {
                throw EngineError(ENGINE_ERR_CUDA, "LM head for the loss failed");
            }
            std::vector<int> one_label{label};
            check_cuda(cudaMemcpyAsync(eng->train_labels, one_label.data(), sizeof(int),
                                       cudaMemcpyHostToDevice, ctx.stream),
                       "Upload the loss label");
            /* The slope lives in its own slot: the log-probability's slot is written by
             * the gather below, and an earlier version shared the two, so the backward
             * read the log-probability (about -7) as the loss's upstream slope and
             * produced an over-scaled, inverted gradient. */
            float slope_host = slope;
            check_cuda(cudaMemcpyAsync(pool + off.row_extra + 3, &slope_host, sizeof(float),
                                       cudaMemcpyHostToDevice, ctx.stream),
                       "Upload the loss slope");
            kernel_logprob_gather(pool + off.row_extra + 1, pool + off.logits_row,
                                  eng->train_labels, 1, V, ctx.stream);
            check_cuda(cudaGetLastError(), "Log-probability for the loss");
            /* The diagonal gradient of the log-softmax, row by row: the fused
             * counterpart of `backward_masked_ce`, so no [tokens, vocab] tensor is
             * materialised. */
            kernel_logprob_gather_backward(pool + off.d_logits_row, pool + off.row_extra + 3,
                                           pool + off.logits_row, eng->train_labels,
                                           /*mask=*/nullptr, 1, V, ctx.stream);
            check_cuda(cudaGetLastError(), "Log-probability backward");
            /* The LM head: dW accumulates into the store (the tied embedding sums its own
             * contribution into the same slot), dX accumulates into this row's hidden
             * state gradient. */
            kernel_cast_bf16_f32(pool + off.row_hidden, row_normed, H, ctx.stream);
            check_cuda(cudaGetLastError(), "Widen the normed row");
            if (gemm_backward_dw(ctx.cublas, role_access(eng, -1, ROLE_LM_HEAD).grad,
                                 pool + off.row_hidden, pool + off.d_logits_row, 1, V, H, 1) != 0) {
                throw EngineError(ENGINE_ERR_CUDA, "LM head dW failed");
            }
            float *d_hidden_row = pool + off.d_hidden + (size_t)row * H;
            if (gemm_backward_dx(ctx.cublas, d_hidden_row, pool + off.d_logits_row,
                                 pool + off.lm_weight, 1, V, H, 1) != 0) {
                throw EngineError(ENGINE_ERR_CUDA, "LM head dX failed");
            }
            /* The final norm's own backward, for the same row. */
            kernel_rms_inv(pool + off.row_extra + 2, pool + off.final_hidden + (size_t)row * H, H,
                           1, eng->dims.rms_eps, ctx.stream);
            check_cuda(cudaGetLastError(), "Final norm inverse RMS");
            kernel_cast_bf16_f32(pool + off.row_normed, ctx.final_norm_w, H, ctx.stream);
            check_cuda(cudaGetLastError(), "Widen the final norm weight");
            RoleAccess final_norm = role_access(eng, -1, ROLE_FINAL_NORM);
            /* A frozen final norm discards its weight gradient into a row scratch,
             * never into d_next: that buffer is the layer walk's running gradient. */
            float *dw = final_norm.grad != nullptr ? final_norm.grad : pool + off.row_hidden;
            kernel_rmsnorm_backward(d_hidden_row, dw, d_hidden_row,
                                    pool + off.final_hidden + (size_t)row * H, pool + off.row_normed,
                                    pool + off.row_extra + 2, H, 1,
                                    eng->dims.norm_style == 0 ? 1 : 0, 1, ctx.stream);
            check_cuda(cudaGetLastError(), "Final norm backward");
            float logprob = 0.0f;
            check_cuda(cudaMemcpyAsync(&logprob, pool + off.row_extra + 1, sizeof(float),
                                       cudaMemcpyDeviceToHost, ctx.stream),
                       "Download the loss's log-probability");
            check_cuda(cudaStreamSynchronize(ctx.stream), "Finish the loss row");
            total += -(double)logprob;
        }
        output->sum = total;
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
}

int engine_train_backward(EngineHandle *eng, struct TrainStep *step, int tokens) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || !eng->state_valid || eng->train_pool == nullptr) {
        set_error("engine_train_backward: run engine_train_forward_retain first");
        return ENGINE_ERR_STATE;
    }
    if (step == nullptr || tokens < 1 || tokens > eng->train_pool_tokens) {
        set_error("engine_train_backward: invalid step or token count");
        return ENGINE_ERR_CONFIG;
    }
    try {
        DeviceCtx &ctx = eng->ctx[0];
        check_cuda(cudaSetDevice(ctx.device_id), "Select the training device");
        const int H = eng->dims.hidden_size;
        const size_t TH = (size_t)tokens * H;
        float *pool = eng->train_pool;
        const struct EngineHandle::TrainStepOffsets &off = eng->train_off;

        /* The layer backward reuses the forward's scratch for its recompute, and its
         * pools come from the step's one allocation. */
        const size_t forward_scratch = layer_workspace_size(tokens, &eng->local_dims);
        struct LayerBackwardScratch scratch{};
        size_t cast_max = 0;
        size_t grad_max = 0;
        for (int layer = 0; layer < eng->num_layers; ++layer) {
            layer_backward_scratch(&eng->dims, tokens, eng->desc.layer_mixers[layer],
                                   eng->desc.layer_ffns[layer], &scratch);
            cast_max = std::max(cast_max, (size_t)scratch.cast_elements);
            grad_max = std::max(grad_max, (size_t)scratch.grad_elements);
        }
        if (off.layer_cast + cast_max > off.layer_grad ||
            off.layer_grad + grad_max > eng->train_pool_floats) {
            set_error("engine_train_backward: the step pool does not hold the layer scratch");
            return ENGINE_ERR_STATE;
        }

        float *d_cur = pool + off.d_hidden;
        float *d_next = pool + off.d_next;
        for (int layer = eng->num_layers - 1; layer >= 0; --layer) {
            const std::string prefix = "layer" + std::to_string(layer) + ".";
            struct LayerBackwardCtx bc;
            bc.cublas = ctx.cublas;
            bc.stream = ctx.stream;
            bc.dims = &eng->dims;
            bc.tokens = tokens;
            bc.layer = layer;
            bc.seq_len = tokens; /* a full sequence: Stage 4 requires seq_len == tokens */
            bc.mixer = eng->desc.layer_mixers[layer];
            bc.ffn = eng->desc.layer_ffns[layer];
            bc.w.input_norm = train_role(eng, layer, ROLE_INPUT_NORM);
            bc.w.post_norm = train_role(eng, layer, ROLE_POST_NORM);
            bc.w.q_proj = train_role(eng, layer, ROLE_ATTN_Q);
            bc.w.k_proj = train_role(eng, layer, ROLE_ATTN_K);
            bc.w.v_proj = train_role(eng, layer, ROLE_ATTN_V);
            bc.w.o_proj = train_role(eng, layer, ROLE_ATTN_O);
            bc.w.q_norm = train_role(eng, layer, ROLE_ATTN_Q_NORM);
            bc.w.k_norm = train_role(eng, layer, ROLE_ATTN_K_NORM);
            bc.w.gate_proj = train_role(eng, layer, ROLE_MLP_GATE);
            bc.w.up_proj = train_role(eng, layer, ROLE_MLP_UP);
            bc.w.down_proj = train_role(eng, layer, ROLE_MLP_DOWN);
            bc.workspace = ctx.workspace;
            bc.ws_backward_offset = forward_scratch;
            bc.cast_f32 = pool + off.layer_cast;
            bc.cast_f32_elements = cast_max;
            bc.grad_f32 = pool + off.layer_grad;
            bc.grad_f32_elements = grad_max;
            bc.positions = ctx.positions;
            bc.kv_cache = eng->layers[layer].kv_cache;
            bc.residual_in = static_cast<const float *>(step_value(step, prefix + "residual"));
            bc.mixer_out = static_cast<const float *>(step_value(step, prefix + "mixerOut"));
            if (bc.residual_in == nullptr || bc.mixer_out == nullptr) {
                set_error("engine_train_backward: %s was not retained", prefix.c_str());
                return ENGINE_ERR_STATE;
            }
            backward_layer(&bc, d_cur, d_next);
            std::swap(d_cur, d_next);
        }
        /* The embedding gather's scatter, which sums a repeated token's contributions
         * and lands in the same gradient slot the tied LM head wrote to. */
        RoleAccess embed = role_access(eng, -1, ROLE_EMBED);
        if (embed.grad != nullptr) {
            kernel_embedding_backward(embed.grad, d_cur, ctx.token_ids, H, tokens, ctx.stream);
            check_cuda(cudaGetLastError(), "Embedding backward");
        }
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
}

int engine_train_zero_grads(EngineHandle *eng) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_zero_grads: no training store attached");
        return ENGINE_ERR_STATE;
    }
    const int logits_index = eng->num_devices > 0 ? eng->num_devices - 1 : 0;
    check_cuda(cudaSetDevice(eng->ctx[logits_index].device_id), "Select the gradient device");
    try {
        const int count = train_store_logical_count(eng->train_store);
        for (int logical = 0; logical < count; ++logical) {
            if (train_store_is_trainable(eng->train_store, logical) != 1) continue;
            float *grad = static_cast<float *>(
                train_store_slot(eng->train_store, logical, TRAIN_SLOT_GRAD));
            if (grad == nullptr) continue;
            const long long elements = train_store_elements(eng->train_store, logical);
            check_cuda(cudaMemsetAsync(grad, 0, (size_t)elements * sizeof(float),
                                       eng->ctx[logits_index].stream),
                       "Zero a parameter gradient");
        }
        check_cuda(cudaStreamSynchronize(eng->ctx[logits_index].stream), "Finish zeroing");
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
}

int engine_train_apply(EngineHandle *eng, const struct TrainOptimizerOptions *options,
                       long long *out_changed) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_apply: no training store attached");
        return ENGINE_ERR_STATE;
    }
    if (options == nullptr || options->step_index < 1) {
        set_error("engine_train_apply: hyper-parameters and a 1-based step index are required");
        return ENGINE_ERR_CONFIG;
    }
    if (train_store_in_update(eng->train_store)) {
        set_error("engine_train_apply: an update window is already open");
        return ENGINE_ERR_STATE;
    }
    check_cuda(cudaSetDevice(eng->ctx[0].device_id), "Select the optimizer device");
    try {
        const struct TrainOptimizerOptions &o = *options;
        cudaStream_t ctx_stream = eng->ctx[0].stream;
        /* The window opens first: writing masters is a write, and the store's rule is
         * that a write needs exclusive ownership (no reader, no live step). */
        if (engine_train_begin_update(eng) != ENGINE_OK) return ENGINE_ERR_STATE;
        long long changed = 0;
        for (int logical = 0; logical < train_store_logical_count(eng->train_store); ++logical) {
            if (train_store_is_trainable(eng->train_store, logical) != 1) continue;
            float *master = static_cast<float *>(
                train_store_slot(eng->train_store, logical, TRAIN_SLOT_MASTER));
            float *grad = static_cast<float *>(
                train_store_slot(eng->train_store, logical, TRAIN_SLOT_GRAD));
            float *m_slot = static_cast<float *>(
                train_store_slot(eng->train_store, logical, TRAIN_SLOT_OPT_M));
            float *v_slot = static_cast<float *>(
                train_store_slot(eng->train_store, logical, TRAIN_SLOT_OPT_V));
            if (master == nullptr || grad == nullptr || m_slot == nullptr || v_slot == nullptr) {
                continue;
            }
            const long long elements = train_store_elements(eng->train_store, logical);
            kernel_adamw(master, grad, m_slot, v_slot, elements, o.lr, o.beta1, o.beta2, o.eps,
                         o.weight_decay, o.step_index, /*bf16_out=*/nullptr, ctx_stream);
            check_cuda(cudaGetLastError(), "AdamW step");
            /* The publication (the store's) is what casts and reports the BF16 refresh;
             * this count is the parameters the optimizer actually stepped, so a step
             * over a store with nothing trainable is visible as 0. */
            ++changed;
        }
        /* The publication is the store's: it casts every master into its readers'
         * compute buffers, refreshes the derived copies and *closes the window* — so
         * there is no end_update call here, and a second one would fail. */
        if (engine_train_publish(eng) != ENGINE_OK) return ENGINE_ERR_STATE;
        if (out_changed != nullptr) *out_changed = changed;
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
}

int engine_train_export_state(EngineHandle *eng, int logical, float *master_out, float *m_out,
                              float *v_out) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_export_state: no training store attached");
        return ENGINE_ERR_STATE;
    }
    if (logical < 0 || logical >= train_store_logical_count(eng->train_store)) {
        set_error("engine_train_export_state: logical %d is out of range", logical);
        return ENGINE_ERR_CONFIG;
    }
    if (master_out == nullptr || m_out == nullptr || v_out == nullptr) {
        set_error("engine_train_export_state: three host buffers are required");
        return ENGINE_ERR_CONFIG;
    }
    if (train_store_is_trainable(eng->train_store, logical) != 1) {
        set_error("engine_train_export_state: logical %d is not trainable", logical);
        return ENGINE_ERR_STATE;
    }
    check_cuda(cudaSetDevice(eng->ctx[0].device_id), "Select the export device");
    const long long elements = train_store_elements(eng->train_store, logical);
    const size_t bytes = (size_t)elements * sizeof(float);
    for (int slot = 0; slot < 3; ++slot) {
        const TrainSlot which = slot == 0 ? TRAIN_SLOT_MASTER
                                          : (slot == 1 ? TRAIN_SLOT_OPT_M : TRAIN_SLOT_OPT_V);
        const void *src = train_store_slot(eng->train_store, logical, which);
        void *dst = slot == 0 ? (void *)master_out : (slot == 1 ? (void *)m_out : (void *)v_out);
        if (src == nullptr) {
            set_error("engine_train_export_state: logical %d has no %s slot", logical,
                      slot == 0 ? "master" : (slot == 1 ? "moment m" : "moment v"));
            return ENGINE_ERR_STATE;
        }
        check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, eng->ctx[0].stream),
                   "Export training state");
    }
    check_cuda(cudaStreamSynchronize(eng->ctx[0].stream), "Finish the export");
    return ENGINE_OK;
}

int engine_train_import_state(EngineHandle *eng, int logical, const float *master, const float *m,
                              const float *v) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || eng->train_store == nullptr) {
        set_error("engine_train_import_state: no training store attached");
        return ENGINE_ERR_STATE;
    }
    if (logical < 0 || logical >= train_store_logical_count(eng->train_store)) {
        set_error("engine_train_import_state: logical %d is out of range", logical);
        return ENGINE_ERR_CONFIG;
    }
    if (master == nullptr || m == nullptr || v == nullptr) {
        set_error("engine_train_import_state: three host buffers are required");
        return ENGINE_ERR_CONFIG;
    }
    if (train_store_is_trainable(eng->train_store, logical) != 1) {
        set_error("engine_train_import_state: logical %d is not trainable", logical);
        return ENGINE_ERR_STATE;
    }
    /* Importing a parameter is a write, so it needs the exclusive window: a reader that
     * held the old version would otherwise see a half-restored set. */
    if (!train_store_in_update(eng->train_store)) {
        set_error("engine_train_import_state: open an update window first");
        return ENGINE_ERR_STATE;
    }
    check_cuda(cudaSetDevice(eng->ctx[0].device_id), "Select the import device");
    const long long elements = train_store_elements(eng->train_store, logical);
    const size_t bytes = (size_t)elements * sizeof(float);
    for (int slot = 0; slot < 3; ++slot) {
        const TrainSlot which = slot == 0 ? TRAIN_SLOT_MASTER
                                          : (slot == 1 ? TRAIN_SLOT_OPT_M : TRAIN_SLOT_OPT_V);
        void *dst = train_store_slot(eng->train_store, logical, which);
        const void *src = slot == 0 ? (const void *)master
                                    : (slot == 1 ? (const void *)m : (const void *)v);
        if (dst == nullptr) {
            set_error("engine_train_import_state: logical %d has no %s slot", logical,
                      slot == 0 ? "master" : (slot == 1 ? "moment m" : "moment v"));
            return ENGINE_ERR_STATE;
        }
        check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, eng->ctx[0].stream),
                   "Import training state");
    }
    check_cuda(cudaStreamSynchronize(eng->ctx[0].stream), "Finish the import");
    return ENGINE_OK;
}

long long engine_train_version(EngineHandle *eng) {
    if (eng == nullptr || eng->train_store == nullptr) return -1;
    return train_store_version(eng->train_store);
}

int engine_rollout_sample(EngineHandle *eng, const int *prompt, int prompt_tokens, int max_tokens,
                          int eos_token, long long policy_id, struct BackwardRng *rng,
                          struct TrainSampleRecord *record) {
    g_error_buf[0] = '\0';
    if (eng == nullptr || !eng->state_valid) {
        set_error("engine_rollout_sample: engine is null or needs engine_reset");
        return ENGINE_ERR_STATE;
    }
    if (eng->train_store == nullptr) {
        set_error("engine_rollout_sample: no training store is attached, so there is no "
                  "version to bind the record to");
        return ENGINE_ERR_STATE;
    }
    if (prompt == nullptr || rng == nullptr || record == nullptr) {
        set_error("engine_rollout_sample: a prompt, an RNG and a record are required");
        return ENGINE_ERR_CONFIG;
    }
    if (prompt_tokens < 1 || max_tokens < 1) {
        set_error("engine_rollout_sample: prompt_tokens and max_tokens must be positive");
        return ENGINE_ERR_CONFIG;
    }
    if (max_tokens > TRAIN_LOOP_MAX_TOKENS) {
        set_error("engine_rollout_sample: %d tokens exceeds the record's %d", max_tokens,
                  TRAIN_LOOP_MAX_TOKENS);
        return ENGINE_ERR_CONFIG;
    }
    if (eng->seq_len != 0) {
        set_error("engine_rollout_sample: the sequence is at position %d; reset first",
                  eng->seq_len);
        return ENGINE_ERR_STATE;
    }
    for (int t = 0; t < prompt_tokens; ++t) {
        if (prompt[t] < 0 || prompt[t] >= eng->dims.vocab_size) {
            set_error("engine_rollout_sample: prompt token %d is out of range", prompt[t]);
            return ENGINE_ERR_CONFIG;
        }
    }
    const int vocab = eng->dims.vocab_size;
    std::vector<float> logits((size_t)vocab);
    std::vector<int64_t> ids(prompt_tokens);
    for (int t = 0; t < prompt_tokens; ++t) ids[t] = prompt[t];
    memset(record, 0, sizeof(*record));
    record->policy_id = policy_id;
    try {
        /* A rollout context borrows the store's committed version for the whole
         * generation: an update cannot begin while this is alive, and the version the
         * record is stamped with is the one this context read rather than one the caller
         * asserted. */
        TrainContext *rollout = train_context_create(eng->train_store, 1);
        if (rollout == nullptr) {
            set_error("engine_rollout_sample: %s", train_last_error());
            return ENGINE_ERR_STATE;
        }
        record->version = train_context_borrowed_version(rollout);
        /* The whole rollout is one sequence: it is reset here and advanced token by token,
         * so the KV cache and the recurrent state belong to this completion alone. */
        engine_reset(eng);
        if (engine_prefill(eng, ids.data(), prompt_tokens, logits.data()) != ENGINE_OK) {
            set_error("engine_rollout_sample: %s", engine_last_error());
            train_context_destroy(rollout);
            return ENGINE_ERR_STATE;
        }
        int generated = 0;
        record->terminal = TRAIN_TERMINAL_LENGTH;
        for (int step = 0; step < max_tokens; ++step) {
            int token = 0;
            float model_logprob = 0.0f;
            float sampled_logprob = 0.0f;
            const TrainStatus sampled = train_loop_sample_fp64(logits.data(), vocab, rng, &token,
                                                               &model_logprob, &sampled_logprob);
            if (sampled != TRAIN_OK) {
                set_error("engine_rollout_sample: %s", train_loop_last_error());
                train_context_destroy(rollout);
                return ENGINE_ERR_STATE;
            }
            record->token_ids[generated] = token;
            record->logprobs[generated] = model_logprob;
            record->sampled_logprobs[generated] = sampled_logprob;
            record->mask[generated] = 1; /* everything generated is a completion token */
            ++generated;
            if (token == eos_token) {
                record->terminal = TRAIN_TERMINAL_EOS;
                break;
            }
            if (engine_decode(eng, (int64_t)token, logits.data()) != ENGINE_OK) {
                set_error("engine_rollout_sample: %s", engine_last_error());
                train_context_destroy(rollout);
                return ENGINE_ERR_STATE;
            }
        }
        record->tokens = generated;
        /* The reward is the verifier's: a deterministic check on the completion, which is
         * what the plan wants before a reward model exists. The caller fills it. */
        record->reward = 0.0f;
        engine_reset(eng);
        train_context_destroy(rollout);
        return ENGINE_OK;
    } catch (const std::exception &e) {
        return forward_error(eng, e);
    }
}
