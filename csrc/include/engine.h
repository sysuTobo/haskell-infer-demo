/**
 * engine.h - Public C API for the Haskell inference engine.
 *
 * This header defines the FFI boundary between Haskell (orchestration) and
 * C/CUDA (tensor computation). Haskell calls these functions via
 * `foreign import ccall`.
 *
 * Threading model: all functions are synchronous and must be called from
 * the same OS thread that created the engine (CUDA context affinity).
 *
 * Memory model: the engine owns all GPU memory. Callers provide host
 * buffers for input/output; the engine handles H2D/D2H transfers.
 */

#ifndef HASKELL_INFER_ENGINE_H
#define HASKELL_INFER_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded model engine. */
typedef struct EngineHandle EngineHandle;

/* ------------------------------------------------------------------ */
/*  Model configuration passed from Haskell                           */
/* ------------------------------------------------------------------ */

/**
 * Layer-to-device assignment. Haskell computes the partition and passes
 * an array where layer_devices[i] = CUDA device ordinal for layer i.
 */
typedef struct {
    int num_layers;          /* total transformer layers (64 for Qwen3.8-27B) */
    int num_devices;         /* number of GPUs to use */
    const int *devices;      /* device ordinals, length num_devices */
    const int *layer_devices;/* per-layer device assignment, length num_layers */
    int max_seq_len;         /* maximum context length (e.g. 4096) */
} EngineConfig;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/**
 * Create an engine: allocate GPU memory, load weights, initialize state.
 *
 * @param model_dir   Path to the model directory containing safetensors
 *                    shards and config.json.
 * @param config      Layer partition and runtime configuration.
 * @return            Opaque handle, or NULL on failure.
 *
 * Weight loading: the engine reads safetensors files from model_dir,
 * maps each tensor to its target device based on config.layer_devices,
 * and uploads via cudaMemcpyAsync. Host-side mmap is released after upload.
 *
 * Derived weights (GemmaRMSNorm weight+1, RoPE cos/sin tables) are
 * computed on-device after loading.
 */
EngineHandle *engine_create(const char *model_dir, const EngineConfig *config);

/**
 * Destroy the engine and free all GPU memory.
 */
void engine_destroy(EngineHandle *engine);

/* ------------------------------------------------------------------ */
/*  Inference                                                         */
/* ------------------------------------------------------------------ */

/**
 * Prefill: process a sequence of tokens and return logits for the last
 * position.
 *
 * First version processes tokens one at a time (recurrent prefill) to
 * reuse the decode path for GDN layers. Each token traverses all layers
 * across all devices.
 *
 * @param engine      Engine handle.
 * @param token_ids   Host array of int64 token IDs, length num_tokens.
 * @param num_tokens  Number of prompt tokens (>= 1).
 * @param out_logits  Host buffer for output logits, size vocab_size floats.
 *                    vocab_size = 248320 for Qwen3.8-27B.
 * @return            0 on success, negative error code on failure.
 */
int engine_prefill(EngineHandle *engine,
                   const int64_t *token_ids,
                   int num_tokens,
                   float *out_logits);

/**
 * Decode: process a single token and return logits.
 *
 * Appends the token to the KV cache / GDN state, runs one forward pass
 * through all layers, and writes logits to out_logits.
 *
 * @param engine      Engine handle.
 * @param token_id    The token to decode (int64).
 * @param out_logits  Host buffer for output logits, size vocab_size floats.
 * @return            0 on success, negative error code on failure.
 */
int engine_decode(EngineHandle *engine,
                  int64_t token_id,
                  float *out_logits);

/**
 * Reset all per-sequence state (KV cache, GDN conv/SSM state, position
 * counter). Call before each new request.
 */
void engine_reset(EngineHandle *engine);

/* ------------------------------------------------------------------ */
/*  Queries                                                           */
/* ------------------------------------------------------------------ */

/**
 * Get the vocabulary size (needed to allocate out_logits buffer).
 */
int engine_vocab_size(const EngineHandle *engine);

/**
 * Get the number of tokens processed so far in the current sequence.
 */
int engine_seq_len(const EngineHandle *engine);

/* ------------------------------------------------------------------ */
/*  Error codes                                                       */
/* ------------------------------------------------------------------ */

#define ENGINE_OK              0
#define ENGINE_ERR_CUDA       -1   /* CUDA runtime/driver error */
#define ENGINE_ERR_ALLOC      -2   /* GPU memory allocation failed */
#define ENGINE_ERR_WEIGHTS    -3   /* Weight file not found or parse error */
#define ENGINE_ERR_CONFIG     -4   /* Invalid configuration */
#define ENGINE_ERR_STATE      -5   /* Invalid engine state */
#define ENGINE_ERR_SEQ_FULL   -6   /* Sequence length exceeds max_seq_len */

/**
 * Get a human-readable error string for the last error on this thread.
 * The returned pointer is valid until the next engine_* call.
 */
const char *engine_last_error(void);

/* ------------------------------------------------------------------ */
/*  Hello-world (Phase 1 FFI verification)                            */
/* ------------------------------------------------------------------ */

/**
 * Minimal test function: launches a CUDA kernel that writes `value` into
 * a device buffer, copies it back, and returns it. Verifies the full
 * Haskell -> C -> CUDA -> C -> Haskell call chain.
 *
 * @param device  CUDA device ordinal to test on.
 * @param value   Integer to round-trip through the GPU.
 * @return        The same value if the GPU round-trip succeeded, or
 *                a negative error code.
 */
int engine_hello_gpu(int device, int value);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_ENGINE_H */
