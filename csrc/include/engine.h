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
 * The engine is created from a *descriptor* (flat JSON, see model_desc.h) that
 * carries the architecture, plus a layer placement:
 *
 *   devices        - CUDA device ordinals to use, length num_devices
 *   layer_devices  - device ordinal owning each layer, length num_layers
 *
 * The descriptor is the single source of truth for model dimensions: the C side
 * holds no per-family constants. A hand-packed struct was deliberately avoided:
 * the descriptor spans variable-length per-layer data (layer kinds, expert
 * counts) and previously had to be mirrored field-for-field in Haskell, C and
 * two Python tests.
 */

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/**
 * Create an engine: parse the descriptor, allocate GPU memory, load weights,
 * initialize state.
 *
 * @param model_dir      Path to the model directory (safetensors shards).
 * @param descriptor_json Flat JSON architecture descriptor.
 * @param num_devices    Number of devices in @devices (>= 1).
 * @param devices        CUDA device ordinals, length num_devices.
 * @param layer_devices  Per-layer device ordinal, length descriptor num_layers.
 * @return               Opaque handle, or NULL on failure.
 *
 * Weight loading: the engine reads safetensors files from model_dir, expands the
 * descriptor's weight-name templates, validates each tensor shape against the
 * role, and uploads to the owning device. Derived weights (GemmaRMSNorm weight+1,
 * RoPE cos/sin tables) are computed on-device after loading.
 */
EngineHandle *engine_create(const char *model_dir, const char *descriptor_json,
                            int num_devices, const int *devices,
                            const int *layer_devices);

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

/**
 * Descriptor wire-format version supported by this library.
 */
int engine_desc_version(void);

/**
 * Write the canonical form of the descriptor the engine parsed into buf.
 * Returns the number of bytes written (excluding the NUL terminator), or a
 * negative error code when buf is too small.
 */
int engine_describe(const EngineHandle *engine, char *buf, int buf_len);

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
