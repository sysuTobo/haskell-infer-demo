/**
 * engine.cu - Main engine implementation.
 *
 * Phase 1: hello-world GPU round-trip for FFI verification.
 * Later phases fill in engine_create / prefill / decode.
 */

#include "engine.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

/* ------------------------------------------------------------------ */
/*  Thread-local error string                                         */
/* ------------------------------------------------------------------ */

static thread_local char g_error_buf[512] = {0};

static void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error_buf, sizeof(g_error_buf), fmt, ap);
    va_end(ap);
}

const char *engine_last_error(void) {
    return g_error_buf;
}

/* ------------------------------------------------------------------ */
/*  CUDA error checking macro                                         */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            set_error("CUDA error at %s:%d: %s", __FILE__, __LINE__,          \
                      cudaGetErrorString(err_));                               \
            return ENGINE_ERR_CUDA;                                           \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_PTR(call)                                                  \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            set_error("CUDA error at %s:%d: %s", __FILE__, __LINE__,          \
                      cudaGetErrorString(err_));                               \
            return NULL;                                                      \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Hello-world kernel (Phase 1 FFI verification)                     */
/* ------------------------------------------------------------------ */

__global__ void hello_kernel(int *buf, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *buf = value;
    }
}

int engine_hello_gpu(int device, int value) {
    CUDA_CHECK(cudaSetDevice(device));

    int *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buf, 0, sizeof(int)));

    hello_kernel<<<1, 1>>>(d_buf, value);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result = 0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_buf, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buf));

    return h_result;
}

/* ------------------------------------------------------------------ */
/*  Engine handle (internal structure)                                */
/* ------------------------------------------------------------------ */

/**
 * Per-device state: weights, KV cache, GDN state, cuBLAS handle, streams.
 */
struct DeviceState {
    int device_id;
    cublasHandle_t cublas;
    cudaStream_t stream;

    /* Weight buffers for layers assigned to this device.
     * Indexed by local layer number (0-based within this device). */
    void **layer_weights;     /* array of weight tensor pointers */
    int num_local_layers;
    int first_global_layer;   /* global index of the first layer on this device */

    /* Embedding table (only on first device) */
    void *embed_tokens;       /* [vocab_size, hidden_size] BF16 */

    /* LM head + final norm (only on last device) */
    void *lm_head;            /* [vocab_size, hidden_size] BF16 */
    void *final_norm_weight;  /* [hidden_size] BF16 (stored as weight+1 in f32) */

    /* KV cache for attention layers on this device.
     * Layout: [num_attn_layers_local, 2, max_seq_len, num_kv_heads, head_dim] BF16 */
    void *kv_cache;
    int num_attn_layers_local;

    /* GDN state for GDN layers on this device.
     * conv_state: [num_gdn_layers_local, conv_dim, kernel_size-1] BF16
     * ssm_state:  [num_gdn_layers_local, num_v_heads, head_dim, head_dim] F32 */
    void *gdn_conv_state;
    void *gdn_ssm_state;
    int num_gdn_layers_local;

    /* Activation buffers (double-buffered for pipeline) */
    void *act_in;             /* [max_seq_len, hidden_size] BF16 */
    void *act_out;            /* [max_seq_len, hidden_size] BF16 */

    /* Scratch space for intermediate computations */
    void *scratch;
    size_t scratch_size;
};

struct EngineHandle {
    EngineConfig config;
    DeviceState *devices;     /* array of num_devices */
    int vocab_size;
    int hidden_size;
    int seq_len;              /* tokens processed so far */

    /* Model dimensions (from config.json or hardcoded) */
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int num_layers;
    int full_attention_interval;
};

/* ------------------------------------------------------------------ */
/*  Stub implementations (filled in Phase 5-6)                        */
/* ------------------------------------------------------------------ */

EngineHandle *engine_create(const char *model_dir, const EngineConfig *config) {
    (void)model_dir;
    (void)config;
    set_error("engine_create not yet implemented (Phase 5-6)");
    return NULL;
}

void engine_destroy(EngineHandle *engine) {
    (void)engine;
}

int engine_prefill(EngineHandle *engine, const int64_t *token_ids,
                   int num_tokens, float *out_logits) {
    (void)engine; (void)token_ids; (void)num_tokens; (void)out_logits;
    set_error("engine_prefill not yet implemented (Phase 6)");
    return ENGINE_ERR_STATE;
}

int engine_decode(EngineHandle *engine, int64_t token_id, float *out_logits) {
    (void)engine; (void)token_id; (void)out_logits;
    set_error("engine_decode not yet implemented (Phase 6)");
    return ENGINE_ERR_STATE;
}

void engine_reset(EngineHandle *engine) {
    if (engine) engine->seq_len = 0;
}

int engine_vocab_size(const EngineHandle *engine) {
    return engine ? engine->vocab_size : 0;
}

int engine_seq_len(const EngineHandle *engine) {
    return engine ? engine->seq_len : 0;
}
