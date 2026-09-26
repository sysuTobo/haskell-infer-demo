/*
 * generation_engine_stub.c - scriptable stand-in for the engine and tokenizer
 * C ABIs, linked only into the CPU generation test suite.
 *
 * The suite exercises the real Infer.Generation / Infer.Runtime / Infer.Tokenizer
 * code on a machine without CUDA, so the engine has to answer deterministically:
 * a logits vector whose argmax is a scripted token, per-call counters, and
 * injectable failures. Including engine.h means a drift in the public engine
 * signature breaks this file at compile time instead of at runtime.
 *
 * Two things this file owes the plan:
 *
 *   1. **Two independent handles** (plan S0 owns two runtimes). State is per
 *      handle; a NULL handle means the default one, which is what the
 *      pre-existing single-engine fixtures pass. A handle created by
 *      engine_create inherits the default's *configuration* (vocabulary, row
 *      values, injectable statuses) so a fixture can configure once before
 *      creating, and keeps its own script and counters.
 *   2. **Consumed histories.** Every token a handle consumes - the prefill's
 *      tokens and every decode - is recorded, because the speculative round
 *      protocol's claim is about *which inputs both engines have consumed*
 *      ("retain exactly P + [x] + y[1:r] in both engines"). The history is a
 *      test-side log: engine_reset does not clear it, so a run that resets and
 *      replays is observable; a fixture clears it explicitly instead.
 *
 * The script has two modes, and the difference matters:
 *
 *   * call-indexed (the default, and what the pre-existing fixtures assume): the
 *     first entry answers the prefill and each later entry answers one decode.
 *     A script of "the greedy output" is written this way.
 *   * prefix-indexed: the script is the engine's whole token sequence, and the
 *     argmax after consuming a prefix p is script[len(p)]. This is the only mode
 *     in which reset-and-replay is faithful - the answer depends on what was
 *     consumed rather than on how many calls have happened - so the speculative
 *     fixtures use it.
 *
 * The control surface (stub_*) is not part of engine.h; GenerationSpec declares
 * it with its own foreign imports.
 */
#include "engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUB_MAX_SCRIPT 1024
#define STUB_MAX_CONSUMED 4096
#define STUB_STREAM_CAP 64

/* --- per-handle engine state ----------------------------------------- */

typedef struct {
    int vocab;
    float winner_value;
    float other_value;
    int64_t script[STUB_MAX_SCRIPT];
    int script_len;
    int script_pos;
    int prefill_status;
    int decode_status;
    int prefix_indexed;
    /* The logical position: tokens consumed *since the last reset*. In prefix-indexed mode the
     * script is the whole token sequence, so the argmax after consuming a prefix p is
     * script[len(p)] - and the position is reset's business, not the consumed history's, because
     * the history also records verification and replay work that a reset discards. */
    int pos;
    int prefill_calls;
    int decode_calls;
    int verify_calls;
    int truncate_calls;
    int save_calls;
    int restore_calls;
    int reset_calls;
    /* Every id this handle has consumed, in order. Not cleared by engine_reset. */
    int64_t consumed[STUB_MAX_CONSUMED];
    int consumed_len;
    /* The one live round checkpoint (plan S2): a save remembers the logical position, a restore
     * puts it back, and the release is what lets the next round save again. */
    int checkpoint_live;
    int checkpoint_pos;
} StubEngine;

/* --- control surface ------------------------------------------------- */

static int g_prefill_calls;
static int g_decode_calls;
static int g_reset_calls;
static int g_destroy_calls;
static int g_create_fails;
static int g_vocab = 64;
static int g_prefill_status;
static int g_decode_status;
static int g_tokenizer_loads;
static int g_tokenizer_frees;
static int g_stream_new_calls;
static int g_stream_free_calls;
/* The values a scripted row is built from. The default row is the one every pre-existing
 * fixture assumes: +1 at the scripted winner and -1 everywhere else. A test that needs a
 * *decided* draw sets a wider gap, which is what makes a sampled selection deterministic
 * enough to assert (the plan's "extend the C stub to supply arbitrary per-step vocabulary
 * rows"). */
static float g_winner_value = 1.0f;
static float g_other_value = -1.0f;
static int64_t g_script[STUB_MAX_SCRIPT];
static char g_last_error[256];

/* The handle a NULL engine means: the pre-existing fixtures pass one placeholder. */
static StubEngine g_default;

static StubEngine *handle_of(EngineHandle *engine) {
    return engine == NULL ? &g_default : (StubEngine *)engine;
}

void stub_reset(void) {
    g_prefill_calls = 0;
    g_decode_calls = 0;
    g_reset_calls = 0;
    g_destroy_calls = 0;
    g_create_fails = 0;
    g_vocab = 64;
    g_winner_value = 1.0f;
    g_other_value = -1.0f;
    g_prefill_status = 0;
    g_decode_status = 0;
    g_tokenizer_loads = 0;
    g_tokenizer_frees = 0;
    g_stream_new_calls = 0;
    g_stream_free_calls = 0;
    g_last_error[0] = '\0';
    memset(&g_default, 0, sizeof g_default);
    g_default.vocab = g_vocab;
    g_default.winner_value = g_winner_value;
    g_default.other_value = g_other_value;
}

void stub_set_vocab(int n) {
    g_vocab = n;
    g_default.vocab = n;
}

void stub_set_row_values(float winner, float other) {
    g_winner_value = winner;
    g_other_value = other;
    g_default.winner_value = winner;
    g_default.other_value = other;
}

void stub_set_script(const int64_t *tokens, int count) {
    if (count < 0) count = 0;
    if (count > STUB_MAX_SCRIPT) count = STUB_MAX_SCRIPT;
    for (int i = 0; i < count; ++i) g_script[i] = tokens[i];
    g_default.script_len = count;
    memcpy(g_default.script, g_script, (size_t)count * sizeof(int64_t));
    g_default.script_pos = 0;
}

void stub_set_prefill_status(int status) {
    g_prefill_status = status;
    g_default.prefill_status = status;
}
void stub_set_decode_status(int status) {
    g_decode_status = status;
    g_default.decode_status = status;
}
void stub_set_create_fails(int flag) { g_create_fails = flag; }

int stub_prefill_calls(void) { return g_default.prefill_calls; }
int stub_decode_calls(void) { return g_default.decode_calls; }
int stub_reset_calls(void) { return g_default.reset_calls; }
int stub_destroy_calls(void) { return g_destroy_calls; }
int stub_tokenizer_loads(void) { return g_tokenizer_loads; }
int stub_tokenizer_frees(void) { return g_tokenizer_frees; }
int stub_stream_create_calls(void) { return g_stream_new_calls; }
int stub_stream_free_calls(void) { return g_stream_free_calls; }

/* --- the same surface, addressed by handle (plan S0's two runtimes) --- */

void stub_set_vocab_h(EngineHandle *engine, int n) { handle_of(engine)->vocab = n; }

void stub_set_row_values_h(EngineHandle *engine, float winner, float other) {
    StubEngine *e = handle_of(engine);
    e->winner_value = winner;
    e->other_value = other;
}

void stub_set_script_h(EngineHandle *engine, const int64_t *tokens, int count) {
    StubEngine *e = handle_of(engine);
    if (count < 0) count = 0;
    if (count > STUB_MAX_SCRIPT) count = STUB_MAX_SCRIPT;
    for (int i = 0; i < count; ++i) e->script[i] = tokens[i];
    e->script_len = count;
    e->script_pos = 0;
}

void stub_set_script_mode_h(EngineHandle *engine, int prefix_indexed) {
    handle_of(engine)->prefix_indexed = prefix_indexed;
}

void stub_clear_consumed_h(EngineHandle *engine) {
    StubEngine *e = handle_of(engine);
    e->consumed_len = 0;
    e->script_pos = 0;
}

int stub_consumed_len_h(EngineHandle *engine) { return handle_of(engine)->consumed_len; }

int stub_consumed_copy_h(EngineHandle *engine, int64_t *out, int cap) {
    StubEngine *e = handle_of(engine);
    if (out == NULL || cap < e->consumed_len) return -1;
    memcpy(out, e->consumed, (size_t)e->consumed_len * sizeof(int64_t));
    return e->consumed_len;
}

int stub_prefill_calls_h(EngineHandle *engine) { return handle_of(engine)->prefill_calls; }
int stub_decode_calls_h(EngineHandle *engine) { return handle_of(engine)->decode_calls; }
int stub_verify_calls_h(EngineHandle *engine) { return handle_of(engine)->verify_calls; }
int stub_save_calls_h(EngineHandle *engine) { return handle_of(engine)->save_calls; }
int stub_restore_calls_h(EngineHandle *engine) { return handle_of(engine)->restore_calls; }
int stub_truncate_calls_h(EngineHandle *engine) { return handle_of(engine)->truncate_calls; }
int stub_reset_calls_h(EngineHandle *engine) { return handle_of(engine)->reset_calls; }

static void set_error(const char *message) {
    snprintf(g_last_error, sizeof g_last_error, "%s", message);
}

/* --- engine ABI ------------------------------------------------------ */

EngineHandle *engine_create(const char *model_dir, const char *descriptor_json,
                            int num_devices, const int *devices,
                            const int *layer_devices) {
    (void)model_dir;
    (void)descriptor_json;
    (void)num_devices;
    (void)devices;
    (void)layer_devices;
    if (g_create_fails) {
        set_error("stub: engine_create was asked to fail");
        return NULL;
    }
    StubEngine *e = (StubEngine *)calloc(1, sizeof(StubEngine));
    if (e == NULL) {
        set_error("stub: out of memory creating an engine handle");
        return NULL;
    }
    /* A new handle starts from the default's configuration, so a fixture can configure the
     * stub once before creating its handles. The script stays per handle. */
    e->vocab = g_default.vocab;
    e->winner_value = g_default.winner_value;
    e->other_value = g_default.other_value;
    e->prefill_status = g_default.prefill_status;
    e->decode_status = g_default.decode_status;
    return (EngineHandle *)e;
}

void engine_destroy(EngineHandle *engine) {
    g_destroy_calls++;
    if (engine != NULL) free(engine);
}

static void fill_logits(const StubEngine *e, float *out_logits, int64_t token) {
    for (int i = 0; i < e->vocab; ++i) out_logits[i] = e->other_value;
    if (token >= 0 && token < e->vocab) out_logits[token] = e->winner_value;
}

/* The token the script names for this call: the call index in call-indexed mode, the logical
 * position in prefix-indexed mode (which is what makes a reset-and-replay faithful). */
static int64_t scripted_token(const StubEngine *e) {
    if (e->script_len <= 0) return 0;
    int index = e->prefix_indexed ? e->pos : e->script_pos;
    if (index < 0) index = 0;
    if (index >= e->script_len) index = e->script_len - 1;
    return e->script[index];
}

static void record_consumed(StubEngine *e, const int64_t *ids, int count) {
    for (int i = 0; i < count; ++i) {
        if (e->consumed_len < STUB_MAX_CONSUMED) e->consumed[e->consumed_len++] = ids[i];
    }
}

int engine_prefill(EngineHandle *engine, const int64_t *token_ids, int num_tokens,
                   float *out_logits) {
    StubEngine *e = handle_of(engine);
    e->prefill_calls++;
    if (e->prefill_status != 0) {
        set_error("stub: prefill failed");
        return e->prefill_status;
    }
    record_consumed(e, token_ids, num_tokens);
    e->pos += num_tokens;
    /* The first scripted token is the argmax of the prefill logits. */
    if (!e->prefix_indexed) e->script_pos = 0;
    fill_logits(e, out_logits, scripted_token(e));
    return 0;
}

int engine_decode(EngineHandle *engine, int64_t token_id, float *out_logits) {
    StubEngine *e = handle_of(engine);
    e->decode_calls++;
    if (e->decode_status != 0) {
        set_error("stub: decode failed");
        return e->decode_status;
    }
    if (!e->prefix_indexed && e->script_pos + 1 < e->script_len) e->script_pos++;
    record_consumed(e, &token_id, 1);
    e->pos += 1;
    fill_logits(e, out_logits, scripted_token(e));
    return 0;
}

void engine_reset(EngineHandle *engine) {
    /* A reset starts a new sequence, so the logical position goes back to zero. The consumed
     * history is a test-side log and is *not* cleared: a run that resets and replays is exactly
     * what the speculative fixtures need to observe. */
    StubEngine *e = handle_of(engine);
    e->reset_calls++;
    e->pos = 0;
}

int engine_vocab_size(const EngineHandle *engine) {
    return ((const StubEngine *)engine == NULL ? g_default.vocab
                                              : ((const StubEngine *)engine)->vocab);
}

int engine_seq_len(const EngineHandle *engine) {
    /* The logical position, which is what the speculative rollback reconciles against: a
     * `NULL` handle means the default one. */
    return ((const StubEngine *)engine == NULL ? g_default.pos
                                               : ((const StubEngine *)engine)->pos);
}

int engine_verify_rows(EngineHandle *engine, const int64_t *token_ids, int num_tokens,
                       float *out_logits, long long capacity) {
    StubEngine *e = handle_of(engine);
    e->verify_calls++;
    if (num_tokens < 1) {
        set_error("stub: verify needs at least one token");
        return ENGINE_ERR_CONFIG;
    }
    const long long needed = (long long)num_tokens * e->vocab;
    if (capacity < needed) {
        set_error("stub: verify capacity is smaller than tokens x vocab");
        return ENGINE_ERR_CONFIG;
    }
    record_consumed(e, token_ids, num_tokens);
    /* Row i conditions on inputs[0..i], so it answers from the position after consuming token i
     * - the same tokens the sequential decodes would have consumed. */
    for (int row = 0; row < num_tokens; ++row) {
        e->pos += 1;
        fill_logits(e, out_logits + (size_t)row * (size_t)e->vocab, scripted_token(e));
    }
    return 0;
}

int engine_truncate(EngineHandle *engine, int retain_len) {
    StubEngine *e = handle_of(engine);
    e->truncate_calls++;
    if (retain_len < 0 || retain_len > e->pos) {
        set_error("stub: truncate is outside the current sequence");
        return ENGINE_ERR_CONFIG;
    }
    e->pos = retain_len;
    return 0;
}

int engine_checkpoint_save(EngineHandle *engine) {
    StubEngine *e = handle_of(engine);
    e->save_calls++;
    if (e->checkpoint_live) {
        set_error("stub: this engine already holds a checkpoint");
        return ENGINE_ERR_STATE;
    }
    e->checkpoint_pos = e->pos;
    e->checkpoint_live = 1;
    return 0;
}

int engine_checkpoint_restore(EngineHandle *engine) {
    StubEngine *e = handle_of(engine);
    e->restore_calls++;
    if (!e->checkpoint_live) {
        set_error("stub: this engine has no checkpoint");
        return ENGINE_ERR_STATE;
    }
    e->pos = e->checkpoint_pos;
    return 0;
}

int engine_checkpoint_release(EngineHandle *engine) {
    handle_of(engine)->checkpoint_live = 0;
    return 0;
}

long long engine_checkpoint_bytes(const EngineHandle *engine) {
    const StubEngine *e = (const StubEngine *)engine;
    return (e == NULL ? g_default.checkpoint_live : e->checkpoint_live) ? 128 : 0;
}

int engine_desc_version(void) { return 1; }

int engine_describe(const EngineHandle *engine, char *buf, int buf_len) {
    (void)engine;
    if (buf_len > 0) buf[0] = '\0';
    return 0;
}

int engine_manifest_version(void) { return 1; }

/* There is no execution to describe: the stub reports the manifest as
 * unavailable rather than inventing identities a caller would then treat as a
 * real capture. The generation loop never asks for one. */
int engine_manifest(const EngineHandle *engine, char *buf, int buf_len) {
    (void)engine;
    (void)buf;
    (void)buf_len;
    snprintf(g_last_error, sizeof(g_last_error),
             "engine_manifest: the generation stub has no execution manifest");
    return ENGINE_ERR_CONFIG;
}

const char *engine_last_error(void) { return g_last_error; }

int engine_hello_gpu(int device, int value) {
    (void)device;
    return value;
}

/* --- tokenizer ABI --------------------------------------------------- */

/* One character per id, so a streamed chunk is easy to predict. */
static char id_to_char(int64_t id) { return (char)('a' + (char)(id % 26)); }

typedef struct {
    char pending[STUB_STREAM_CAP];
    int pending_len;
    int finished;
} StubStream;

void *tokenizer_load(const char *path) {
    (void)path;
    g_tokenizer_loads++;
    return (void *)0x2;
}

void tokenizer_free(void *handle) {
    (void)handle;
    g_tokenizer_frees++;
}

int tokenizer_vocab_size(void *handle) {
    (void)handle;
    return g_vocab;
}

int tokenizer_encode_len(void *handle, const char *text) {
    (void)handle;
    return text == NULL ? -1 : (int)strlen(text);
}

int tokenizer_encode(void *handle, const char *text, int64_t *out_ids, int max_len) {
    (void)handle;
    if (text == NULL) return -1;
    int count = (int)strlen(text);
    if (count == 0) return 0;
    if (out_ids == NULL || max_len < count) return -2;
    for (int i = 0; i < count; ++i) out_ids[i] = (int64_t)(unsigned char)text[i];
    return count;
}

int tokenizer_decode_len(void *handle, const int64_t *ids, int len) {
    (void)handle;
    if (len < 0 || (len > 0 && ids == NULL)) return -1;
    return len + 1;
}

int tokenizer_decode(void *handle, const int64_t *ids, int len, char *out_buf, int buf_size) {
    (void)handle;
    if (len < 0 || (len > 0 && ids == NULL) || buf_size < 0) return -1;
    if (len == 0) {
        if (out_buf != NULL && buf_size >= 1) out_buf[0] = '\0';
        return 0;
    }
    if (out_buf == NULL || buf_size < len + 1) return -2;
    for (int i = 0; i < len; ++i) out_buf[i] = id_to_char(ids[i]);
    out_buf[len] = '\0';
    return len;
}

void *tokenizer_stream_new(void *handle) {
    (void)handle;
    StubStream *stream = (StubStream *)calloc(1, sizeof(StubStream));
    if (stream == NULL) return NULL;
    g_stream_new_calls++;
    return stream;
}

void tokenizer_stream_free(void *handle) {
    if (handle == NULL) return;
    g_stream_free_calls++;
    free(handle);
}

int tokenizer_stream_reset(void *handle) {
    StubStream *stream = (StubStream *)handle;
    if (stream == NULL) return -1;
    stream->pending_len = 0;
    stream->finished = 0;
    return 0;
}

int tokenizer_stream_feed(void *handle, int64_t id) {
    StubStream *stream = (StubStream *)handle;
    if (stream == NULL || stream->finished) return -1;
    if (stream->pending_len + 1 >= STUB_STREAM_CAP) return -1;
    stream->pending[stream->pending_len++] = id_to_char(id);
    return 0;
}

int tokenizer_stream_pending(void *handle) {
    StubStream *stream = (StubStream *)handle;
    if (stream == NULL) return -1;
    return stream->pending_len + 1;
}

int tokenizer_stream_drain(void *handle, char *buf, int buf_size) {
    StubStream *stream = (StubStream *)handle;
    if (stream == NULL || buf_size < 0) return -1;
    if (stream->pending_len == 0) {
        if (buf != NULL && buf_size >= 1) buf[0] = '\0';
        return 0;
    }
    if (buf == NULL || buf_size < stream->pending_len + 1) return -2;
    memcpy(buf, stream->pending, (size_t)stream->pending_len);
    buf[stream->pending_len] = '\0';
    int written = stream->pending_len;
    stream->pending_len = 0;
    return written;
}

int tokenizer_stream_finish(void *handle, char *buf, int buf_size) {
    StubStream *stream = (StubStream *)handle;
    if (stream == NULL || buf_size < 0) return -1;
    stream->finished = 1;
    return tokenizer_stream_drain(handle, buf, buf_size);
}
