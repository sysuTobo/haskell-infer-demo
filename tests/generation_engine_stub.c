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
 * The control surface (stub_*) is not part of engine.h; GenerationSpec declares
 * it with its own foreign imports.
 */
#include "engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUB_MAX_SCRIPT 1024
#define STUB_STREAM_CAP 64

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
static int g_script_len;
static int g_script_pos;
static char g_last_error[256];

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
    g_script_len = 0;
    g_script_pos = 0;
    g_last_error[0] = '\0';
}

void stub_set_vocab(int n) { g_vocab = n; }

void stub_set_row_values(float winner, float other) {
    g_winner_value = winner;
    g_other_value = other;
}

void stub_set_script(const int64_t *tokens, int count) {
    if (count < 0) count = 0;
    if (count > STUB_MAX_SCRIPT) count = STUB_MAX_SCRIPT;
    for (int i = 0; i < count; ++i) g_script[i] = tokens[i];
    g_script_len = count;
}

void stub_set_prefill_status(int status) { g_prefill_status = status; }
void stub_set_decode_status(int status) { g_decode_status = status; }
void stub_set_create_fails(int flag) { g_create_fails = flag; }

int stub_prefill_calls(void) { return g_prefill_calls; }
int stub_decode_calls(void) { return g_decode_calls; }
int stub_reset_calls(void) { return g_reset_calls; }
int stub_destroy_calls(void) { return g_destroy_calls; }
int stub_tokenizer_loads(void) { return g_tokenizer_loads; }
int stub_tokenizer_frees(void) { return g_tokenizer_frees; }
int stub_stream_create_calls(void) { return g_stream_new_calls; }
int stub_stream_free_calls(void) { return g_stream_free_calls; }

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
    return (EngineHandle *)0x1;
}

void engine_destroy(EngineHandle *engine) {
    (void)engine;
    g_destroy_calls++;
}

static void fill_logits(float *out_logits, int64_t token) {
    for (int i = 0; i < g_vocab; ++i) out_logits[i] = g_other_value;
    if (token >= 0 && token < g_vocab) out_logits[token] = g_winner_value;
}

int engine_prefill(EngineHandle *engine, const int64_t *token_ids, int num_tokens,
                   float *out_logits) {
    (void)engine;
    (void)token_ids;
    (void)num_tokens;
    g_prefill_calls++;
    if (g_prefill_status != 0) {
        set_error("stub: prefill failed");
        return g_prefill_status;
    }
    /* The first scripted token is the argmax of the prefill logits. */
    g_script_pos = 0;
    fill_logits(out_logits, g_script_len > 0 ? g_script[0] : 0);
    return 0;
}

int engine_decode(EngineHandle *engine, int64_t token_id, float *out_logits) {
    (void)engine;
    (void)token_id;
    g_decode_calls++;
    if (g_decode_status != 0) {
        set_error("stub: decode failed");
        return g_decode_status;
    }
    if (g_script_pos + 1 < g_script_len) g_script_pos++;
    fill_logits(out_logits, g_script_len > 0 ? g_script[g_script_pos] : 0);
    return 0;
}

void engine_reset(EngineHandle *engine) {
    (void)engine;
    g_reset_calls++;
}

int engine_vocab_size(const EngineHandle *engine) {
    (void)engine;
    return g_vocab;
}

int engine_seq_len(const EngineHandle *engine) {
    (void)engine;
    return 0;
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
