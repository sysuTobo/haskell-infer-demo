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

#include "backward.h"   /* the RNG and the sampler's log-probability transform (Stage 4) */
#include "train.h"      /* the trainable runtime's handles and status codes (Stage 3) */
#include "train_loop.h" /* the phase machine and the selection record (Stage 5) */

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

/**
 * Load weight-only INT4 operands for the dense FFN roles from a converted sidecar (plan Q2).
 *
 * `manifest_dir` holds a `weights.manifest.json` written by scripts/quantize_weights.py and the
 * `artifacts/` it names. The engine **keeps** the BF16 weights it loaded and adds the packed
 * operands beside them: decode (M = 1) reads the INT4 weights, batched M keeps reading BF16,
 * which is the split the Q2 measurement supports (the warp-per-row GEMV is 2.25x faster than
 * BF16 at M = 1, the tiled kernel 8-30x slower at M > 1). Call it after engine_create and
 * before any forward.
 *
 * Every dense layer of the descriptor must have both an `mlpGateUp` pair and an `mlpDown`, or
 * the call fails: a layer with one and not the other would silently run mixed precision. A
 * tensor-parallel or expert-parallel engine is refused, because the converter quantizes whole
 * tensors and a rank's shard is not what the sidecar describes. The manifest's extents are
 * checked against this layer's own weights and its SHA-256 against the bytes on disk before
 * anything is uploaded.
 *
 * @return ENGINE_OK, or a negative error code with the reason in engine_last_error().
 */
int engine_load_quantized_ffn(EngineHandle *engine, const char *manifest_dir);

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

/**
 * Verify: consume @num_tokens@ ids as one bounded batch and return **every** row's logits
 * (plan S1). Row @i@ is the distribution that conditions on @token_ids[0..i]@, so a caller
 * proposing a window can score each candidate against the target's own state instead of paying
 * one decode per proposal.
 *
 * The ordinary entry points deliberately compute only the final row to avoid a
 * @[max_chunk, vocab]@ buffer; this one allocates that buffer lazily on first use. The tokens
 * are appended to the current sequence exactly as @engine_decode@ would append them, and the
 * caller is responsible for truncating with engine_truncate when a proposal is rejected.
 *
 * @param out_logits  Host buffer of at least @num_tokens * vocab_size@ floats, row-major.
 * @param capacity    Its length in floats; a short buffer is refused rather than overrun.
 * @return            ENGINE_OK, or a negative error code with the reason in engine_last_error().
 */
int engine_verify_rows(EngineHandle *engine, const int64_t *token_ids, int num_tokens,
                       float *out_logits, long long capacity);

/**
 * Truncate the sequence back to @retain_len@ tokens, so the next append overwrites what was
 * dropped (plan S1's "append-only cache truncation restricted to the current sequence and an
 * available prefix").
 *
 * Admitted only for a model whose every layer is full attention. A GDN layer's recurrent state
 * cannot be inverted, and MLA needs its own admission, so both are refused rather than left to
 * disagree with the cache: that admission is S2's hybrid checkpoints, not this call.
 *
 * @return ENGINE_OK, or a negative error code with the reason in engine_last_error().
 */
int engine_truncate(EngineHandle *engine, int retain_len);

/**
 * Round checkpoints (plan S2): save, restore and release the engine's per-sequence state.
 *
 * A checkpoint is **engine-owned and opaque**, and an engine holds at most one at a time - the
 * plan's "at most one round checkpoint per engine initially" - so `save` reports success rather
 * than handing back a handle, and a second `save` while one is live is refused. What is saved is
 * every buffer `engine_reset` would clear (the attention and MLA caches, and the GDN convolution
 * and SSM state), plus the sequence length, the reset generation and the parameter identity.
 *
 * The subtlety is *why* the whole set is copied rather than only the recurrent part: the KV and
 * MLA entries up to the retained length are immutable under the append-only cache design, so
 * their logical length would suffice - but the length alone cannot put them back if a caller
 * restores across a longer sequence, and classifying buffers by role would be a second registry
 * to keep in step. Copying the reset set is redundant for those caches and correct for all of
 * them.
 *
 * `restore` refuses rather than guessing when the saved state is no longer the engine's: after
 * `engine_reset` (the reset generation moved), after a failed forward (the engine requires a
 * reset), or after the weights or numerical policy changed (the parameter digest differs).
 * Restoring does **not** clear that invalid state, and it does not replay anything: the plan's
 * protocol is restore the round-start state and then replay exactly the retained inputs, which
 * is the caller's job. `release` frees the copies and is a no-op when none is live, because it is
 * also a cleanup path.
 *
 * @return ENGINE_OK, or a negative error code with the reason in engine_last_error().
 */
int engine_checkpoint_save(EngineHandle *engine);
int engine_checkpoint_restore(EngineHandle *engine);
int engine_checkpoint_release(EngineHandle *engine);

/** Bytes a live checkpoint holds, or 0 when none is saved (for a report, not a gate). */
long long engine_checkpoint_bytes(const EngineHandle *engine);

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

/**
 * Execution-manifest wire version supported by this library.
 */
int engine_manifest_version(void);

/**
 * Write the canonical execution manifest for this engine into buf (see
 * csrc/include/manifest.h and docs/manifest-contract.md).
 *
 * The manifest references the canonical descriptor and reports the three
 * content identities (semantic, numerical policy, deployment) plus the
 * build/runtime provenance and parameter identity that a bitwise capture
 * comparison has to agree on. Every fact is resolved from this build and this
 * runtime; a fact that could not be established is reported as
 * "unknown"/"unavailable"/"unsupported" instead of being defaulted, and a
 * strict comparison must refuse it.
 *
 * @param engine   Engine handle.
 * @param buf      Destination, at least ENGINE_MANIFEST_MAX bytes.
 * @param buf_len  Capacity of buf.
 * @return         Bytes written (excluding the NUL terminator), or a negative
 *                 error code (ENGINE_ERR_CONFIG when the buffer is too small).
 */
int engine_manifest(const EngineHandle *engine, char *buf, int buf_len);

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

/* ------------------------------------------------------------------ */
/*  Stage 3: the training path                                        */
/* ------------------------------------------------------------------ */

/*
 * These calls add a training lifecycle to the same weights the inference API already
 * uses; they do not change `engine_create/prefill/decode/reset/destroy`.
 *
 * The ownership model is the plan's: the parameter store owns the *identity* of a
 * parameter (its logical id, whether it is tied, frozen, trainable, how many readers
 * hold it) while the engine owns the *buffers*. A parameter's compute buffer is the
 * engine's own BF16 weight, so a published update is visible to an inference forward
 * without any second copy, and a publication that forgot to refresh a derived copy
 * (the FP32 GDN norm weight) is refused rather than left silently disagreeing.
 */

/** What a caller wants attached. Training state is optional: a rollout-only caller
 *  needs the parameter identities without paying for FP32 masters. */
struct TrainAttachOptions {
    int allocate_training_state;   /* FP32 master, gradient and two optimizer slots */
    const int *frozen_roles;       /* role ids to freeze (no training state allocated) */
    int frozen_role_count;
};

/**
 * Attach a parameter store to this engine. One parameter per (layer, role) the
 * checkpoint actually provides; tied roles collapse to one logical parameter whose
 * compute slot is written once per reader. Returns the store (owned by the engine, or
 * NULL on failure with engine_last_error set). Calling it twice returns the same
 * store.
 */
struct TrainStore *engine_train_attach(EngineHandle *engine,
                                       const struct TrainAttachOptions *options);

/** The attached store, or NULL. */
struct TrainStore *engine_train_store(EngineHandle *engine);

/** Open the exclusive update window: refused while a context or a step is live. */
int engine_train_begin_update(EngineHandle *engine);

/** Upload FP32 master values for one logical parameter. Requires an open window and
 *  a parameter whose training state was allocated (a frozen one has none). */
int engine_train_write_master(EngineHandle *engine, int logical, const float *host_values);

/**
 * Publish: cast every written master into the BF16 weight of every reader, bump the
 * version, recompute every derived copy (the FP32 GDN norm weight among them) and
 * close the window. Refuses to close while a derived copy is stale, so the refresh
 * cannot be skipped.
 */
int engine_train_publish(EngineHandle *engine);

/** Close the window without publishing (an aborted update). */
int engine_train_end_update(EngineHandle *engine);

/** What a teacher-forced forward produces. `all_logits` is the debug/reference path
 *  and is the caller's host buffer of [tokens x vocab]; `logprobs` is the path a loss
 *  uses and needs only [tokens]. */
struct TrainForwardOutput {
    float *all_logits;
    float *logprobs;
    int *selected;
    int selected_count;
};

/**
 * Begin a training step: retain the values this sequence's backward will consume -
 * one mixer output, one ffn output and one residual per layer (the residual stream is
 * overwritten in place, so each layer needs its own copy), plus the GDN chunk-boundary
 * state per GDN layer under a full-sequence schedule. The step is a reader of the
 * current version, so an update is refused until it ends; its buffers are freed by
 * engine_train_step_end in the order the free points allow.
 *
 * This is the retention half of the plan's training-step context; the backward that
 * consumes the values is Stage 4.
 */
int engine_train_step_begin(EngineHandle *engine, int tokens, int chunk_count,
                           struct TrainStep **out_step);
int engine_train_step_end(struct TrainStep *step);

/** How many values a step of this size retains, and how many bytes of GDN
 *  chunk-boundary state they include - the numbers the plan's truncated-vs-full BPTT
 *  choice is made from. */
int engine_train_step_plan(EngineHandle *engine, int tokens, int chunk_count,
                          int *saved_count, long long *gdn_state_elements);

/**
 * Teacher-forced forward over ONE sequence, from position 0 (call engine_reset
 * between steps). For every selected position - next-token label shift, optional
 * prompt/padding mask, explicit positions - the natural-log log-probability of the
 * label is written to `output->logprobs`, computed on the device row by row so no
 * [tokens, vocab] tensor is ever materialised. Independent sequences are never
 * flattened into one causal sequence.
 */
int engine_train_forward(EngineHandle *engine, const int *token_ids, int tokens,
                         const int64_t *positions, const int *labels, const uint8_t *mask,
                         int shift, struct TrainForwardOutput *output);

/* ------------------------------------------------------------------ */
/* The SFT step (plan Stage 5)                                        */
/* ------------------------------------------------------------------ */

/**
 * Run the training forward and retain what the backward consumes: one mixer output, one
 * feed-forward output and the residual stream per layer, into the buffers
 * `engine_train_step_begin` allocated, plus the final hidden state (the loss and the
 * final norm's backward read it after the walk has reused the per-device activations).
 *
 * Requires a *full sequence* from position 0 and a single-device placement: a pipeline
 * split would have to move gradients between devices and a tensor-parallel placement
 * would have to reduce sharded ones, and neither is this stage's bring-up. Both are
 * refusals rather than silent approximations.
 */
int engine_train_forward_retain(EngineHandle *engine, struct TrainStep *step,
                                const int *token_ids, const int64_t *positions, int tokens);

/** The loss and its gradient's seed. */
struct TrainLossOutput {
    double sum;          /* the summed negative log-likelihood over the selected rows */
    long long count;     /* how many rows the teacher forcing selected */
};

/**
 * For every position the teacher forcing selects, recompute the final norm and the LM
 * head row by row (so no [tokens, vocab] logits tensor is materialised), accumulate the
 * LM head's dW into the store, and leave the hidden state's gradient for
 * `engine_train_backward`. A mask byte of 0 and a label below zero both deselect.
 */
int engine_train_loss(EngineHandle *engine, struct TrainStep *step, const int *token_ids,
                      const int *labels, const uint8_t *mask, int shift, int tokens,
                      struct TrainLossOutput *output);

/**
 * Walk the layers in reverse from the hidden state's gradient the loss left, accumulate
 * every role's FP32 gradient, and scatter the embedding's. `step` must be the one
 * `engine_train_forward_retain` filled; the gradients are *added* to the store's slots,
 * so a caller that wants a fresh step zeroes them first.
 */
int engine_train_backward(EngineHandle *engine, struct TrainStep *step, int tokens);

/** AdamW over every trainable logical parameter, then a publication. */
struct TrainOptimizerOptions {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_index;      /* 1-based: the bias correction continues across resumes */
};

/**
 * One optimizer step over every trainable logical parameter, then a publication: the
 * FP32 masters step, the derived copies are refreshed and each compute weight becomes
 * the single BF16 rounding of its new master. Refused while a step is live (an optimizer
 * write must not overlap a reader), so end the step first. `out_changed` reports how many
 * BF16 elements actually moved, which is what distinguishes a step from a no-op.
 */
int engine_train_apply(EngineHandle *engine, const struct TrainOptimizerOptions *options,
                       long long *out_changed);

/** Zero every trainable parameter's FP32 gradient accumulator. */
int engine_train_zero_grads(EngineHandle *engine);

/** The parameter version the store currently publishes, or -1 when no training store is
 * attached. This is the version a rollout is bound to and the one a caller compares a
 * record's against, so it has to be readable rather than assumed. */
long long engine_train_version(EngineHandle *engine);

/**
 * Generate one completion from the model as it currently stands and fill a Stage-5
 * selection record: the sampled token ids, each token's model log-probability *and* the
 * sampler's, the completion mask, the terminal reason and the version.
 *
 * The sampling is `train_loop_sample_fp64` -- host FP64, temperature 1, no truncation --
 * so the sampler's distribution is the model's softmax, which is what makes the ratio's
 * exactly-one property at unchanged parameters true rather than approximate. The logits
 * come back to the host one row at a time (the engine's logits are for the last position
 * only), so no [tokens, vocab] tensor is retained anywhere.
 *
 * The rollout runs inside a real borrowing `TrainContext` (train_store's rollout kind), and
 * the record's version is the one *that context* borrowed -- not a value the caller hands
 * in. Two things follow, and they are why it is done this way: an optimizer update cannot
 * open while the generation is live (`train_store_begin_update` refuses a borrowed
 * version), and the record cannot be stamped with a version the rollout did not read.
 * `policy_id` stays the caller's, because "behavior policy vs frozen reference" is a
 * question about the run, not about the store.
 */
int engine_rollout_sample(EngineHandle *engine, const int *prompt, int prompt_tokens,
                          int max_tokens, int eos_token, long long policy_id,
                          struct BackwardRng *rng, struct TrainSampleRecord *record);

/**
 * Copy one trainable logical parameter's training state out to (or in from) the host:
 * the FP32 master and the optimizer's two moments. The checkpoint *format* is
 * `backward_checkpoint_write/read` (plan Stage 4, gated on the CPU); these two entry
 * points are what a device run uses to fill or restore those buffers, so a resume can
 * be tested without a second copy of the state in the engine.
 */
int engine_train_export_state(EngineHandle *engine, int logical, float *master_out,
                              float *m_out, float *v_out);
int engine_train_import_state(EngineHandle *engine, int logical, const float *master,
                              const float *m, const float *v);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_ENGINE_H */
