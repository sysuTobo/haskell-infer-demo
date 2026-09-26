/**
 * quant_manifest.c - see quant_manifest.h for why this exists and what it refuses.
 *
 * The scanner is a small recursive-descent JSON reader rather than a general library: the
 * document is machine-written by scripts/quantize_weights.py, keys are sorted and values are
 * plain, so what is needed is a reader that knows exactly which fields the schema has and
 * refuses anything it does not recognise *as a required field* while skipping unknown ones. The
 * one thing it does not do is guess: a missing required field, a truncated document and a
 * duplicate key are all errors, because a loader that silently defaults a shape is how a wrong
 * artifact reaches a GPU.
 */
#include "quant_manifest.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QUANT_MANIFEST_MAX_DEPTH 32

static char g_error[512];

const char *quant_manifest_last_error(void) { return g_error; }

void quant_manifest_clear_error(void) { g_error[0] = '\0'; }

static QuantManifestStatus fail(QuantManifestStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

/* ------------------------------------------------------------------ */
/* The scanner                                                        */
/* ------------------------------------------------------------------ */

struct Cursor {
    const char *begin;
    const char *p;
    const char *end;
};

/* A byte offset from the document start, so an error names a position rather than a pointer. */
static long long offset_of(const struct Cursor *c) { return (long long)(c->p - c->begin); }

static QuantManifestStatus syntax(struct Cursor *c, const char *what) {
    return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: %s at byte %lld", what, offset_of(c));
}

static void skip_ws(struct Cursor *c) {
    while (c->p < c->end) {
        const char ch = *c->p;
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            ++c->p;
            continue;
        }
        break;
    }
}

static QuantManifestStatus parse_string(struct Cursor *c, char *out, size_t out_len) {
    if (c->p >= c->end || *c->p != '"') return syntax(c, "expected a string");
    ++c->p;
    size_t used = 0;
    while (c->p < c->end && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\') {
            if (c->p >= c->end) return syntax(c, "unterminated escape");
            const char esc = *c->p++;
            switch (esc) {
                case '"': ch = '"'; break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                case 'u': {
                    /* The schema's strings are model paths, role names and hex digests: ASCII
                     * only. A code point that would need more than one byte is refused rather
                     * than mangled into something that compares unequal to the original. */
                    unsigned value = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (c->p >= c->end) return syntax(c, "truncated \\u escape");
                        const char h = *c->p++;
                        unsigned digit;
                        if (h >= '0' && h <= '9') digit = (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') digit = (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') digit = (unsigned)(h - 'A' + 10);
                        else return syntax(c, "malformed \\u escape");
                        value = value * 16u + digit;
                    }
                    if (value > 0x7Fu) {
                        return fail(QUANT_MANIFEST_ERR_SYNTAX,
                                    "quant manifest: a string outside ASCII at byte %lld "
                                    "(the schema is ASCII-only)",
                                    offset_of(c));
                    }
                    ch = (char)value;
                    break;
                }
                default: return syntax(c, "unknown escape");
            }
        }
        if (used + 1 >= out_len) {
            return fail(QUANT_MANIFEST_ERR_ARG,
                        "quant manifest: a string at byte %lld does not fit %zu bytes",
                        offset_of(c), out_len);
        }
        out[used++] = ch;
    }
    if (c->p >= c->end) return syntax(c, "unterminated string");
    ++c->p; /* closing quote */
    out[used] = '\0';
    return QUANT_MANIFEST_OK;
}

static QuantManifestStatus parse_number(struct Cursor *c, double *out) {
    const char *start = c->p;
    if (c->p < c->end && (*c->p == '-' || *c->p == '+')) ++c->p;
    int digits = 0;
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
        ++c->p;
        ++digits;
    }
    if (c->p < c->end && *c->p == '.') {
        ++c->p;
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
            ++c->p;
            ++digits;
        }
    }
    if (digits == 0) return syntax(c, "expected a number");
    if (c->p < c->end && (*c->p == 'e' || *c->p == 'E')) {
        ++c->p;
        if (c->p < c->end && (*c->p == '-' || *c->p == '+')) ++c->p;
        int exp_digits = 0;
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
            ++c->p;
            ++exp_digits;
        }
        if (exp_digits == 0) return syntax(c, "malformed exponent");
    }
    char buf[64];
    const size_t len = (size_t)(c->p - start);
    if (len >= sizeof(buf)) return syntax(c, "a number is longer than the schema allows");
    memcpy(buf, start, len);
    buf[len] = '\0';
    char *stop = NULL;
    errno = 0;
    *out = strtod(buf, &stop);
    if (stop == buf || *stop != '\0' || errno == ERANGE) return syntax(c, "malformed number");
    return QUANT_MANIFEST_OK;
}

static QuantManifestStatus parse_integer(struct Cursor *c, long long *out) {
    double value = 0.0;
    const QuantManifestStatus status = parse_number(c, &value);
    if (status != QUANT_MANIFEST_OK) return status;
    const double rounded = (double)(long long)value;
    if (rounded != value) return syntax(c, "expected an integer");
    *out = (long long)value;
    return QUANT_MANIFEST_OK;
}

static QuantManifestStatus expect(struct Cursor *c, char ch) {
    if (c->p >= c->end || *c->p != ch) {
        char what[48];
        snprintf(what, sizeof(what), "expected '%c'", ch);
        return syntax(c, what);
    }
    ++c->p;
    return QUANT_MANIFEST_OK;
}

static int is_lower_hex(const char *text, size_t len) {
    if (strlen(text) != len) return 0;
    for (size_t i = 0; i < len; ++i) {
        const char ch = text[i];
        const int digit = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        if (!digit) return 0;
    }
    return 1;
}

/* Skip any value, so a field the schema does not use cannot break the parse but also cannot be
 * mistaken for one it does. */
static QuantManifestStatus skip_value(struct Cursor *c, int depth) {
    if (depth > QUANT_MANIFEST_MAX_DEPTH) return syntax(c, "nesting is too deep");
    skip_ws(c);
    if (c->p >= c->end) return syntax(c, "expected a value");
    const char ch = *c->p;
    if (ch == '"') {
        char scratch[QUANT_MANIFEST_PATH_MAX];
        return parse_string(c, scratch, sizeof(scratch));
    }
    if (ch == '{' || ch == '[') {
        const char close = (ch == '{') ? '}' : ']';
        ++c->p;
        skip_ws(c);
        if (c->p < c->end && *c->p == close) {
            ++c->p;
            return QUANT_MANIFEST_OK;
        }
        for (;;) {
            skip_ws(c);
            if (ch == '{') {
                char key[QUANT_MANIFEST_PATH_MAX];
                const QuantManifestStatus status = parse_string(c, key, sizeof(key));
                if (status != QUANT_MANIFEST_OK) return status;
                skip_ws(c);
                const QuantManifestStatus colon = expect(c, ':');
                if (colon != QUANT_MANIFEST_OK) return colon;
                const QuantManifestStatus inner = skip_value(c, depth + 1);
                if (inner != QUANT_MANIFEST_OK) return inner;
            } else {
                const QuantManifestStatus inner = skip_value(c, depth + 1);
                if (inner != QUANT_MANIFEST_OK) return inner;
            }
            skip_ws(c);
            if (c->p < c->end && *c->p == ',') {
                ++c->p;
                continue;
            }
            break;
        }
        return expect(c, close);
    }
    if (ch == 't' && c->end - c->p >= 4 && memcmp(c->p, "true", 4) == 0) { c->p += 4; return QUANT_MANIFEST_OK; }
    if (ch == 'f' && c->end - c->p >= 5 && memcmp(c->p, "false", 5) == 0) { c->p += 5; return QUANT_MANIFEST_OK; }
    if (ch == 'n' && c->end - c->p >= 4 && memcmp(c->p, "null", 4) == 0) { c->p += 4; return QUANT_MANIFEST_OK; }
    double scratch = 0.0;
    return parse_number(c, &scratch);
}

/* A key in an object: the string, then the colon. */
static QuantManifestStatus parse_key(struct Cursor *c, char *key, size_t key_len) {
    const QuantManifestStatus status = parse_string(c, key, key_len);
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    return expect(c, ':');
}

/* `[n, k]`, exactly two integers. */
static QuantManifestStatus parse_shape2(struct Cursor *c, long long *n, long long *k) {
    skip_ws(c);
    QuantManifestStatus status = expect(c, '[');
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    status = parse_integer(c, n);
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    status = expect(c, ',');
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    status = parse_integer(c, k);
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    return expect(c, ']');
}

/* { "bytes": b, "count": c, "dtype": "…", "file": "…", "sha256": "…" }
 * `has_count` is 0 for the packed payload, whose count is an element count the schema does not
 * give - the byte count is the whole story there, and reading a count as if it existed would be
 * the kind of defaulting this module refuses. */
static QuantManifestStatus parse_artifact(struct Cursor *c, struct QuantArtifact *artifact,
                                          int has_count) {
    artifact->count = -1;
    skip_ws(c);
    QuantManifestStatus status = expect(c, '{');
    if (status != QUANT_MANIFEST_OK) return status;
    int seen_bytes = 0, seen_dtype = 0, seen_file = 0, seen_sha = 0, seen_count = 0;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') {
        ++c->p;
        return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: an artifact object is empty");
    }
    for (;;) {
        char key[QUANT_MANIFEST_NAME_MAX];
        status = parse_key(c, key, sizeof(key));
        if (status != QUANT_MANIFEST_OK) return status;
        skip_ws(c);
        if (strcmp(key, "bytes") == 0) {
            if (seen_bytes++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"bytes\"");
            status = parse_integer(c, &artifact->bytes);
        } else if (strcmp(key, "count") == 0) {
            if (seen_count++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"count\"");
            status = parse_integer(c, &artifact->count);
        } else if (strcmp(key, "dtype") == 0) {
            if (seen_dtype++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"dtype\"");
            status = parse_string(c, artifact->dtype, sizeof(artifact->dtype));
        } else if (strcmp(key, "file") == 0) {
            if (seen_file++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"file\"");
            status = parse_string(c, artifact->file, sizeof(artifact->file));
        } else if (strcmp(key, "sha256") == 0) {
            if (seen_sha++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"sha256\"");
            status = parse_string(c, artifact->sha256, sizeof(artifact->sha256));
        } else {
            status = skip_value(c, 0);
        }
        if (status != QUANT_MANIFEST_OK) return status;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') {
            ++c->p;
            skip_ws(c);
            continue;
        }
        break;
    }
    status = expect(c, '}');
    if (status != QUANT_MANIFEST_OK) return status;
    if (!seen_bytes || !seen_dtype || !seen_file || !seen_sha) {
        return fail(QUANT_MANIFEST_ERR_SYNTAX,
                    "quant manifest: an artifact is missing one of bytes/dtype/file/sha256");
    }
    if (has_count && !seen_count) {
        return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: a scale artifact has no count");
    }
    if (!has_count && seen_count) {
        return fail(QUANT_MANIFEST_ERR_SYNTAX,
                    "quant manifest: a packed payload carries a count it has no element type for");
    }
    return QUANT_MANIFEST_OK;
}

/* ------------------------------------------------------------------ */
/* The schema                                                         */
/* ------------------------------------------------------------------ */

static QuantManifestStatus require_artifact(struct QuantArtifact *artifact, const char *want_dtype,
                                            const char *where, int scale) {
    if (!is_lower_hex(artifact->sha256, 64)) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: %s has a malformed sha256", where);
    }
    if (artifact->bytes <= 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: %s has a nonpositive byte count",
                    where);
    }
    if (artifact->file[0] == '\0') {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: %s has no file", where);
    }
    if (strcmp(artifact->dtype, want_dtype) != 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT,
                    "quant manifest: %s is %s, the frozen format is %s", where, artifact->dtype,
                    want_dtype);
    }
    if (scale != (artifact->count >= 0)) {
        return fail(QUANT_MANIFEST_ERR_FORMAT,
                    "quant manifest: %s %s a scale count", where, scale ? "is missing" : "carries");
    }
    return QUANT_MANIFEST_OK;
}

/* The layout the format implies, so extents and byte counts cannot be invented here. */
QuantManifestStatus quant_manifest_layout(const struct QuantEntry *entry,
                                          struct LinearWeightLayout *out,
                                          char *err, size_t err_len) {
    if (entry == NULL || out == NULL) {
        if (err != NULL && err_len > 0) err[0] = '\0';
        return QUANT_MANIFEST_ERR_ARG;
    }
    const LinearStatus status = linear_layout_init(out, entry->n, entry->k, entry->group);
    if (status != LINEAR_OK) {
        if (err != NULL && err_len > 0) {
            snprintf(err, err_len, "quant manifest: %s layer %d is not a shape the format holds: %s",
                     entry->role, entry->layer, linear_last_error());
        }
        return QUANT_MANIFEST_ERR_SHAPE;
    }
    return QUANT_MANIFEST_OK;
}

/* Extents and dtypes of one entry, against the format and against its own artifact sizes. */
static QuantManifestStatus check_entry(struct QuantEntry *entry, char *err, size_t err_len) {
    if (entry->role[0] == '\0') {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: an entry has no role");
    }
    if (entry->layer < 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: %s has a negative layer",
                    entry->role);
    }
    struct LinearWeightLayout layout;
    const QuantManifestStatus laid = quant_manifest_layout(entry, &layout, err, err_len);
    if (laid != QUANT_MANIFEST_OK) return laid;

    char where[QUANT_MANIFEST_NAME_MAX + 32];
    snprintf(where, sizeof(where), "%s layer %d", entry->role, entry->layer);
    QuantManifestStatus status = require_artifact(&entry->packed, "u8", where, 0);
    if (status != QUANT_MANIFEST_OK) return status;
    status = require_artifact(&entry->scales, "bf16", where, 1);
    if (status != QUANT_MANIFEST_OK) return status;
    if (entry->packed.bytes != layout.packed_bytes) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %s records %lld packed bytes, [%lld, %lld] at group %lld is "
                    "%lld",
                    where, entry->packed.bytes, entry->n, entry->k, entry->group,
                    layout.packed_bytes);
    }
    if (entry->scales.count != layout.scale_count) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %s records %lld scales, [%lld, %lld] at group %lld is %lld",
                    where, entry->scales.count, entry->n, entry->k, entry->group,
                    layout.scale_count);
    }
    if (entry->scales.bytes != entry->scales.count * 2) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %s records %lld scale bytes for %lld bf16 scales", where,
                    entry->scales.bytes, entry->scales.count);
    }
    return QUANT_MANIFEST_OK;
}

static QuantManifestStatus parse_entries(struct Cursor *c, struct QuantManifest *manifest) {
    QuantManifestStatus status = expect(c, '[');
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') {
        ++c->p;
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: entries is empty");
    }
    for (;;) {
        if (manifest->entry_count >= QUANT_MANIFEST_MAX_ENTRIES) {
            return fail(QUANT_MANIFEST_ERR_FORMAT,
                        "quant manifest: more than %d entries", QUANT_MANIFEST_MAX_ENTRIES);
        }
        struct QuantEntry *entry = &manifest->entries[manifest->entry_count];
        memset(entry, 0, sizeof(*entry));
        entry->role_index = -1;
        status = expect(c, '{');
        if (status != QUANT_MANIFEST_OK) return status;
        int seen[8] = {0};
        const char *names[8] = {"role", "role_index", "layer", "logical_shape", "packed",
                                "scales", "group", "group_axis"};
        skip_ws(c);
        for (;;) {
            char key[QUANT_MANIFEST_NAME_MAX];
            status = parse_key(c, key, sizeof(key));
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(c);
            int which = -1;
            for (int i = 0; i < 8; ++i) {
                if (strcmp(key, names[i]) == 0) which = i;
            }
            if (which < 0) {
                status = skip_value(c, 0);
            } else if (seen[which]++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX,
                            "quant manifest: duplicate \"%s\" in an entry", key);
            } else if (which == 0) {
                status = parse_string(c, entry->role, sizeof(entry->role));
            } else if (which == 1) {
                long long value = 0;
                status = parse_integer(c, &value);
                entry->role_index = (int)value;
            } else if (which == 2) {
                long long value = 0;
                status = parse_integer(c, &value);
                entry->layer = (int)value;
            } else if (which == 3) {
                status = parse_shape2(c, &entry->n, &entry->k);
            } else if (which == 4) {
                status = parse_artifact(c, &entry->packed, 0);
            } else if (which == 5) {
                status = parse_artifact(c, &entry->scales, 1);
            } else if (which == 6) {
                status = parse_integer(c, &entry->group);
            } else {
                char axis[QUANT_MANIFEST_NAME_MAX];
                status = parse_string(c, axis, sizeof(axis));
                if (status == QUANT_MANIFEST_OK && strcmp(axis, "k") != 0) {
                    status = fail(QUANT_MANIFEST_ERR_FORMAT,
                                  "quant manifest: the group axis is %s, the format groups K",
                                  axis);
                }
            }
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(c);
            if (c->p < c->end && *c->p == ',') {
                ++c->p;
                skip_ws(c);
                continue;
            }
            break;
        }
        status = expect(c, '}');
        if (status != QUANT_MANIFEST_OK) return status;
        if (!seen[0] || !seen[2] || !seen[3] || !seen[4] || !seen[5] || !seen[6]) {
            return fail(QUANT_MANIFEST_ERR_SYNTAX,
                        "quant manifest: an entry is missing one of role/layer/logical_shape/"
                        "packed/scales/group");
        }
        status = check_entry(entry, NULL, 0);
        if (status != QUANT_MANIFEST_OK) return status;
        ++manifest->entry_count;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') {
            ++c->p;
            skip_ws(c);
            continue;
        }
        break;
    }
    return expect(c, ']');
}

/* An F1 pair: its shape and its bytes must be its members', in the recorded order. This is the
 * one check that ties the convenient operand to the entries it was built from, so a pair that
 * agreed only with itself could not pass. */
static QuantManifestStatus check_pair(struct QuantManifest *manifest, struct QuantPair *pair) {
    char where[QUANT_MANIFEST_NAME_MAX + 32];
    snprintf(where, sizeof(where), "pair %s layer %d", pair->name, pair->layer);
    QuantManifestStatus status = require_artifact(&pair->packed, "u8", where, 0);
    if (status != QUANT_MANIFEST_OK) return status;
    status = require_artifact(&pair->scales, "bf16", where, 1);
    if (status != QUANT_MANIFEST_OK) return status;
    if (pair->n <= 0 || pair->k <= 0) {
        return fail(QUANT_MANIFEST_ERR_SHAPE, "quant manifest: %s has a nonpositive shape", where);
    }
    long long rows = 0, packed_bytes = 0, scales = 0;
    long long k = -1;
    for (int member = 0; member < 2; ++member) {
        const int index = quant_manifest_find_entry(manifest, pair->layer, pair->member_role[member]);
        if (index < 0) {
            return fail(QUANT_MANIFEST_ERR_SHAPE,
                        "quant manifest: %s names member %s, which is not an entry of layer %d",
                        where, pair->member_role[member], pair->layer);
        }
        const struct QuantEntry *entry = &manifest->entries[index];
        if (pair->member_role_index[member] >= 0 &&
            pair->member_role_index[member] != entry->role_index) {
            return fail(QUANT_MANIFEST_ERR_SHAPE,
                        "quant manifest: %s records member %s as role_index %d, the entry is %d",
                        where, entry->role, pair->member_role_index[member], entry->role_index);
        }
        if (k < 0) k = entry->k;
        else if (entry->k != k) {
            return fail(QUANT_MANIFEST_ERR_SHAPE,
                        "quant manifest: %s concatenates members of different K (%lld and %lld)",
                        where, k, entry->k);
        }
        rows += entry->n;
        packed_bytes += entry->packed.bytes;
        scales += entry->scales.count;
    }
    if (pair->k != k || pair->n != rows) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %s is [%lld, %lld], its members are [%lld, %lld]", where,
                    pair->n, pair->k, rows, k);
    }
    if (pair->packed.bytes != packed_bytes) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %s records %lld packed bytes, its members hold %lld", where,
                    pair->packed.bytes, packed_bytes);
    }
    if (pair->scales.count != scales || pair->scales.bytes != scales * 2) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %s records %lld scales, its members hold %lld", where,
                    pair->scales.count, scales);
    }
    if (pair->group != LINEAR_WEIGHT_GROUP_SIZE) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: %s groups %lld, the format groups "
                    "%d", where, pair->group, LINEAR_WEIGHT_GROUP_SIZE);
    }
    return QUANT_MANIFEST_OK;
}

static QuantManifestStatus parse_pairs(struct Cursor *c, struct QuantManifest *manifest) {
    QuantManifestStatus status = expect(c, '[');
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') {
        ++c->p;
        return QUANT_MANIFEST_OK; /* a manifest whose roles are not both quantized has none */
    }
    for (;;) {
        if (manifest->pair_count >= QUANT_MANIFEST_MAX_PAIRS) {
            return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: more than %d pairs",
                        QUANT_MANIFEST_MAX_PAIRS);
        }
        struct QuantPair *pair = &manifest->pairs[manifest->pair_count];
        memset(pair, 0, sizeof(*pair));
        pair->member_role_index[0] = -1;
        pair->member_role_index[1] = -1;
        status = expect(c, '{');
        if (status != QUANT_MANIFEST_OK) return status;
        int seen_name = 0, seen_layer = 0, seen_members = 0, seen_shape = 0;
        int seen_packed = 0, seen_scales = 0, seen_group = 0;
        skip_ws(c);
        for (;;) {
            char key[QUANT_MANIFEST_NAME_MAX];
            status = parse_key(c, key, sizeof(key));
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(c);
            if (strcmp(key, "name") == 0) {
                if (seen_name++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"name\"");
                status = parse_string(c, pair->name, sizeof(pair->name));
            } else if (strcmp(key, "layer") == 0) {
                long long value = 0;
                if (seen_layer++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"layer\"");
                status = parse_integer(c, &value);
                pair->layer = (int)value;
            } else if (strcmp(key, "logical_shape") == 0) {
                if (seen_shape++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate shape");
                status = parse_shape2(c, &pair->n, &pair->k);
            } else if (strcmp(key, "packed") == 0) {
                if (seen_packed++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate packed");
                status = parse_artifact(c, &pair->packed, 0);
            } else if (strcmp(key, "scales") == 0) {
                if (seen_scales++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate scales");
                status = parse_artifact(c, &pair->scales, 1);
            } else if (strcmp(key, "group") == 0) {
                if (seen_group++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate group");
                status = parse_integer(c, &pair->group);
            } else if (strcmp(key, "members") == 0) {
                if (seen_members++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate members");
                status = expect(c, '[');
                if (status != QUANT_MANIFEST_OK) return status;
                for (int member = 0; member < 2; ++member) {
                    skip_ws(c);
                    status = expect(c, '{');
                    if (status != QUANT_MANIFEST_OK) return status;
                    int seen_role = 0, seen_index = 0;
                    skip_ws(c);
                    for (;;) {
                        char mkey[QUANT_MANIFEST_NAME_MAX];
                        status = parse_key(c, mkey, sizeof(mkey));
                        if (status != QUANT_MANIFEST_OK) return status;
                        skip_ws(c);
                        if (strcmp(mkey, "role") == 0) {
                            if (seen_role++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate member role");
                            status = parse_string(c, pair->member_role[member],
                                                  sizeof(pair->member_role[member]));
                        } else if (strcmp(mkey, "role_index") == 0) {
                            long long value = 0;
                            if (seen_index++) return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate member role_index");
                            status = parse_integer(c, &value);
                            pair->member_role_index[member] = (int)value;
                        } else {
                            status = skip_value(c, 0);
                        }
                        if (status != QUANT_MANIFEST_OK) return status;
                        skip_ws(c);
                        if (c->p < c->end && *c->p == ',') {
                            ++c->p;
                            skip_ws(c);
                            continue;
                        }
                        break;
                    }
                    status = expect(c, '}');
                    if (status != QUANT_MANIFEST_OK) return status;
                    if (!seen_role) {
                        return fail(QUANT_MANIFEST_ERR_SYNTAX,
                                    "quant manifest: a pair member has no role");
                    }
                    skip_ws(c);
                    if (member == 0) {
                        status = expect(c, ',');
                        if (status != QUANT_MANIFEST_OK) return status;
                    }
                }
                skip_ws(c);
                status = expect(c, ']');
            } else {
                status = skip_value(c, 0);
            }
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(c);
            if (c->p < c->end && *c->p == ',') {
                ++c->p;
                skip_ws(c);
                continue;
            }
            break;
        }
        status = expect(c, '}');
        if (status != QUANT_MANIFEST_OK) return status;
        if (!seen_name || !seen_layer || !seen_members || !seen_shape || !seen_packed ||
            !seen_scales || !seen_group) {
            return fail(QUANT_MANIFEST_ERR_SYNTAX,
                        "quant manifest: a pair is missing one of name/layer/members/"
                        "logical_shape/packed/scales/group");
        }
        if (pair->member_role[0][0] == '\0' || pair->member_role[1][0] == '\0') {
            return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: a pair has fewer than 2 members");
        }
        status = check_pair(manifest, pair);
        if (status != QUANT_MANIFEST_OK) return status;
        ++manifest->pair_count;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') {
            ++c->p;
            skip_ws(c);
            continue;
        }
        break;
    }
    return expect(c, ']');
}

/* The precision map must describe the same cells as the entries: every entry is a quantized
 * cell, and no cell claims int4 without an entry. A duplicate quantized cell is refused rather
 * than counted twice. */
static QuantManifestStatus parse_precision_map(struct Cursor *c, struct QuantManifest *manifest) {
    QuantManifestStatus status = expect(c, '[');
    if (status != QUANT_MANIFEST_OK) return status;
    unsigned char seen[QUANT_MANIFEST_MAX_ENTRIES];
    memset(seen, 0, sizeof(seen));
    skip_ws(c);
    if (c->p < c->end && *c->p == ']') {
        ++c->p;
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: the precision map is empty");
    }
    for (;;) {
        status = expect(c, '{');
        if (status != QUANT_MANIFEST_OK) return status;
        char role[QUANT_MANIFEST_NAME_MAX] = {0};
        char precision[QUANT_MANIFEST_NAME_MAX] = {0};
        int layer = -1, seen_role = 0, seen_layer = 0, seen_precision = 0;
        skip_ws(c);
        for (;;) {
            char key[QUANT_MANIFEST_NAME_MAX];
            status = parse_key(c, key, sizeof(key));
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(c);
            if (strcmp(key, "role") == 0) {
                seen_role = 1;
                status = parse_string(c, role, sizeof(role));
            } else if (strcmp(key, "layer") == 0) {
                long long value = 0;
                seen_layer = 1;
                status = parse_integer(c, &value);
                layer = (int)value;
            } else if (strcmp(key, "precision") == 0) {
                seen_precision = 1;
                status = parse_string(c, precision, sizeof(precision));
            } else {
                status = skip_value(c, 0);
            }
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(c);
            if (c->p < c->end && *c->p == ',') {
                ++c->p;
                skip_ws(c);
                continue;
            }
            break;
        }
        status = expect(c, '}');
        if (status != QUANT_MANIFEST_OK) return status;
        if (!seen_role || !seen_layer || !seen_precision) {
            return fail(QUANT_MANIFEST_ERR_SYNTAX,
                        "quant manifest: a precision cell is missing role/layer/precision");
        }
        if (strcmp(precision, "int4") == 0) {
            const int index = quant_manifest_find_entry(manifest, layer, role);
            if (index < 0) {
                return fail(QUANT_MANIFEST_ERR_SHAPE,
                            "quant manifest: the precision map calls %s layer %d int4, which has "
                            "no entry",
                            role, layer);
            }
            if (seen[index]) {
                return fail(QUANT_MANIFEST_ERR_SHAPE,
                            "quant manifest: %s layer %d appears twice as int4", role, layer);
            }
            seen[index] = 1;
            ++manifest->precision_int4_cells;
        } else if (strcmp(precision, "bf16") == 0) {
            ++manifest->precision_bf16_cells;
        } else {
            return fail(QUANT_MANIFEST_ERR_FORMAT,
                        "quant manifest: a precision cell is %s, expected int4 or bf16",
                        precision);
        }
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') {
            ++c->p;
            skip_ws(c);
            continue;
        }
        break;
    }
    status = expect(c, ']');
    if (status != QUANT_MANIFEST_OK) return status;
    for (int i = 0; i < manifest->entry_count; ++i) {
        if (!seen[i]) {
            return fail(QUANT_MANIFEST_ERR_SHAPE,
                        "quant manifest: %s layer %d has an entry but no int4 precision cell",
                        manifest->entries[i].role, manifest->entries[i].layer);
        }
    }
    if (manifest->precision_int4_cells != manifest->entry_count) {
        return fail(QUANT_MANIFEST_ERR_SHAPE,
                    "quant manifest: %d int4 cells for %d entries",
                    manifest->precision_int4_cells, manifest->entry_count);
    }
    return QUANT_MANIFEST_OK;
}

static QuantManifestStatus parse_format(struct Cursor *c, struct QuantManifest *manifest) {
    QuantManifestStatus status = expect(c, '{');
    if (status != QUANT_MANIFEST_OK) return status;
    int seen_name = 0, seen_abi = 0, seen_group = 0, seen_qmin = 0, seen_qmax = 0;
    int seen_zero = 0, seen_invalid = 0, seen_packed = 0, seen_scale = 0;
    long long abi = 0, group = 0, qmin = 0, qmax = 0, zero = 0, invalid = 0;
    skip_ws(c);
    for (;;) {
        char key[QUANT_MANIFEST_NAME_MAX];
        status = parse_key(c, key, sizeof(key));
        if (status != QUANT_MANIFEST_OK) return status;
        skip_ws(c);
        if (strcmp(key, "name") == 0) {
            seen_name = 1;
            char name[QUANT_MANIFEST_NAME_MAX];
            status = parse_string(c, name, sizeof(name));
        } else if (strcmp(key, "abi_version") == 0) {
            seen_abi = 1;
            status = parse_integer(c, &abi);
        } else if (strcmp(key, "group") == 0) {
            seen_group = 1;
            status = parse_integer(c, &group);
        } else if (strcmp(key, "qmin") == 0) {
            seen_qmin = 1;
            status = parse_integer(c, &qmin);
        } else if (strcmp(key, "qmax") == 0) {
            seen_qmax = 1;
            status = parse_integer(c, &qmax);
        } else if (strcmp(key, "zero_point") == 0) {
            seen_zero = 1;
            status = parse_integer(c, &zero);
        } else if (strcmp(key, "invalid_code") == 0) {
            seen_invalid = 1;
            status = parse_integer(c, &invalid);
        } else if (strcmp(key, "packed_dtype") == 0) {
            seen_packed = 1;
            status = parse_string(c, manifest->packed_dtype, sizeof(manifest->packed_dtype));
        } else if (strcmp(key, "scale_dtype") == 0) {
            seen_scale = 1;
            status = parse_string(c, manifest->scale_dtype, sizeof(manifest->scale_dtype));
        } else {
            status = skip_value(c, 0);
        }
        if (status != QUANT_MANIFEST_OK) return status;
        skip_ws(c);
        if (c->p < c->end && *c->p == ',') {
            ++c->p;
            skip_ws(c);
            continue;
        }
        break;
    }
    status = expect(c, '}');
    if (status != QUANT_MANIFEST_OK) return status;
    if (!seen_name || !seen_abi || !seen_group || !seen_qmin || !seen_qmax || !seen_zero ||
        !seen_invalid || !seen_packed || !seen_scale) {
        return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: the format block is incomplete");
    }
    if (abi != LINEAR_WEIGHT_ABI_VERSION) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: format abi_version %lld, this "
                    "reader speaks %d", abi, LINEAR_WEIGHT_ABI_VERSION);
    }
    if (group != LINEAR_WEIGHT_GROUP_SIZE || qmin != LINEAR_WEIGHT_QMIN ||
        qmax != LINEAR_WEIGHT_QMAX || zero != 0 || invalid != LINEAR_WEIGHT_INVALID_CODE) {
        return fail(QUANT_MANIFEST_ERR_FORMAT,
                    "quant manifest: the format block is not the frozen format (group %lld, q "
                    "[%lld, %lld], zero_point %lld, invalid_code %lld)",
                    group, qmin, qmax, zero, invalid);
    }
    if (strcmp(manifest->packed_dtype, "u8") != 0 || strcmp(manifest->scale_dtype, "bf16") != 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT,
                    "quant manifest: packed/scales dtype %s/%s, the frozen format is u8/bf16",
                    manifest->packed_dtype, manifest->scale_dtype);
    }
    manifest->format_abi_version = (int)abi;
    manifest->group = group;
    manifest->qmin = (int)qmin;
    manifest->qmax = (int)qmax;
    manifest->zero_point = (int)zero;
    manifest->invalid_code = (int)invalid;
    return QUANT_MANIFEST_OK;
}

QuantManifestStatus quant_manifest_parse(const char *json, struct QuantManifest *out,
                                         char *err, size_t err_len) {
    quant_manifest_clear_error();
    if (json == NULL || out == NULL) {
        if (err != NULL && err_len > 0) snprintf(err, err_len, "quant manifest: null argument");
        return QUANT_MANIFEST_ERR_ARG;
    }
    memset(out, 0, sizeof(*out));

    struct Cursor c;
    c.begin = json;
    c.p = json;
    c.end = json + strlen(json);

    skip_ws(&c);
    QuantManifestStatus status = expect(&c, '{');
    if (status != QUANT_MANIFEST_OK) return status;
    int seen_version = 0, seen_format = 0, seen_converter = 0, seen_source = 0;
    int seen_entries = 0, seen_precision = 0, seen_pairs = 0;
    skip_ws(&c);
    if (c.p < c.end && *c.p == '}') {
        return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: the document is empty");
    }
    for (;;) {
        char key[QUANT_MANIFEST_NAME_MAX];
        status = parse_key(&c, key, sizeof(key));
        if (status != QUANT_MANIFEST_OK) return status;
        skip_ws(&c);
        if (strcmp(key, "manifest_version") == 0) {
            long long value = 0;
            if (seen_version++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            status = parse_integer(&c, &value);
            out->manifest_version = (int)value;
        } else if (strcmp(key, "format") == 0) {
            if (seen_format++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            status = parse_format(&c, out);
        } else if (strcmp(key, "converter") == 0) {
            if (seen_converter++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            status = expect(&c, '{');
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(&c);
            for (;;) {
                char ckey[QUANT_MANIFEST_NAME_MAX];
                status = parse_key(&c, ckey, sizeof(ckey));
                if (status != QUANT_MANIFEST_OK) return status;
                skip_ws(&c);
                if (strcmp(ckey, "name") == 0) {
                    status = parse_string(&c, out->converter_name, sizeof(out->converter_name));
                } else if (strcmp(ckey, "version") == 0) {
                    long long value = 0;
                    status = parse_integer(&c, &value);
                    out->converter_version = (int)value;
                } else {
                    status = skip_value(&c, 0);
                }
                if (status != QUANT_MANIFEST_OK) return status;
                skip_ws(&c);
                if (c.p < c.end && *c.p == ',') {
                    ++c.p;
                    skip_ws(&c);
                    continue;
                }
                break;
            }
            status = expect(&c, '}');
        } else if (strcmp(key, "source") == 0) {
            if (seen_source++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            status = expect(&c, '{');
            if (status != QUANT_MANIFEST_OK) return status;
            skip_ws(&c);
            for (;;) {
                char skey[QUANT_MANIFEST_NAME_MAX];
                status = parse_key(&c, skey, sizeof(skey));
                if (status != QUANT_MANIFEST_OK) return status;
                skip_ws(&c);
                if (strcmp(skey, "model_dir") == 0) {
                    status = parse_string(&c, out->source_model_dir,
                                          sizeof(out->source_model_dir));
                } else {
                    status = skip_value(&c, 0);
                }
                if (status != QUANT_MANIFEST_OK) return status;
                skip_ws(&c);
                if (c.p < c.end && *c.p == ',') {
                    ++c.p;
                    skip_ws(&c);
                    continue;
                }
                break;
            }
            status = expect(&c, '}');
        } else if (strcmp(key, "entries") == 0) {
            if (seen_entries++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            status = parse_entries(&c, out);
        } else if (strcmp(key, "precision_map") == 0) {
            /* The map is validated against the entries, so it must follow them. The converter's
             * encoder sorts keys, and `precision_map` sorts before `pairs` but after `entries`,
             * so the order is a property of the document rather than an accident. */
            if (seen_precision++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            if (!seen_entries) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX,
                            "quant manifest: precision_map precedes entries, so it cannot be "
                            "checked against them");
            }
            status = parse_precision_map(&c, out);
        } else if (strcmp(key, "pairs") == 0) {
            if (seen_pairs++) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX, "quant manifest: duplicate \"%s\"", key);
            }
            /* A pair is its *members'* rows, so it cannot be checked before them. The
             * converter's encoder sorts keys (`entries` < `pairs`), so this is a property of the
             * document rather than a preference. */
            if (!seen_entries) {
                return fail(QUANT_MANIFEST_ERR_SYNTAX,
                            "quant manifest: pairs precede entries, so a pair cannot be checked "
                            "against its members");
            }
            status = parse_pairs(&c, out);
        } else {
            status = skip_value(&c, 0);
        }
        if (status != QUANT_MANIFEST_OK) return status;
        skip_ws(&c);
        if (c.p < c.end && *c.p == ',') {
            ++c.p;
            skip_ws(&c);
            continue;
        }
        break;
    }
    status = expect(&c, '}');
    if (status != QUANT_MANIFEST_OK) return status;
    skip_ws(&c);
    if (c.p != c.end) return syntax(&c, "trailing bytes after the document");

    if (!seen_version || !seen_format || !seen_converter || !seen_source || !seen_entries ||
        !seen_precision) {
        return fail(QUANT_MANIFEST_ERR_SYNTAX,
                    "quant manifest: missing one of manifest_version/format/converter/source/"
                    "entries/precision_map");
    }
    if (out->manifest_version != QUANT_MANIFEST_VERSION) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: version %d, this reader speaks %d",
                    out->manifest_version, QUANT_MANIFEST_VERSION);
    }
    if (out->converter_name[0] == '\0' || out->converter_version <= 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: the converter is unnamed or "
                    "unversioned");
    }
    if (out->source_model_dir[0] == '\0') {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: no source model directory");
    }
    if (out->entry_count == 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: no entries");
    }
    return QUANT_MANIFEST_OK;
}

int quant_manifest_find_entry(const struct QuantManifest *manifest, int layer, const char *role) {
    if (manifest == NULL || role == NULL) return -1;
    for (int i = 0; i < manifest->entry_count; ++i) {
        if (manifest->entries[i].layer == layer && strcmp(manifest->entries[i].role, role) == 0) {
            return i;
        }
    }
    return -1;
}

int quant_manifest_find_pair(const struct QuantManifest *manifest, const char *name, int layer) {
    if (manifest == NULL || name == NULL) return -1;
    for (int i = 0; i < manifest->pair_count; ++i) {
        if (manifest->pairs[i].layer == layer && strcmp(manifest->pairs[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

QuantManifestStatus quant_artifact_read(const char *root, const struct QuantArtifact *artifact,
                                        const char *want_dtype, uint8_t **out, size_t *out_len,
                                        char *err, size_t err_len) {
    quant_manifest_clear_error();
    if (root == NULL || artifact == NULL || out == NULL || out_len == NULL) {
        if (err != NULL && err_len > 0) snprintf(err, err_len, "quant manifest: null argument");
        return QUANT_MANIFEST_ERR_ARG;
    }
    *out = NULL;
    *out_len = 0;
    if (want_dtype != NULL && strcmp(artifact->dtype, want_dtype) != 0) {
        return fail(QUANT_MANIFEST_ERR_FORMAT, "quant manifest: %s is %s, expected %s",
                    artifact->file, artifact->dtype, want_dtype);
    }
    char path[QUANT_MANIFEST_PATH_MAX * 2];
    const int written = snprintf(path, sizeof(path), "%s/%s", root, artifact->file);
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return fail(QUANT_MANIFEST_ERR_ARG, "quant manifest: the artifact path is too long");
    }
    struct stat info;
    if (stat(path, &info) != 0) {
        return fail(QUANT_MANIFEST_ERR_IO, "quant manifest: %s: %s", path, strerror(errno));
    }
    if ((long long)info.st_size != artifact->bytes) {
        return fail(QUANT_MANIFEST_ERR_IO, "quant manifest: %s is %lld bytes, the manifest "
                    "records %lld", path, (long long)info.st_size, artifact->bytes);
    }
    FILE *handle = fopen(path, "rb");
    if (handle == NULL) {
        return fail(QUANT_MANIFEST_ERR_IO, "quant manifest: %s: %s", path, strerror(errno));
    }
    uint8_t *buffer = (uint8_t *)malloc((size_t)artifact->bytes);
    if (buffer == NULL) {
        fclose(handle);
        return fail(QUANT_MANIFEST_ERR_IO, "quant manifest: cannot allocate %lld bytes for %s",
                    artifact->bytes, path);
    }
    const size_t got = fread(buffer, 1, (size_t)artifact->bytes, handle);
    fclose(handle);
    if (got != (size_t)artifact->bytes) {
        free(buffer);
        return fail(QUANT_MANIFEST_ERR_IO, "quant manifest: %s read %zu of %lld bytes", path, got,
                    artifact->bytes);
    }
    char digest[SHA256_HEX_LEN];
    sha256_hex(buffer, got, digest);
    if (strcmp(digest, artifact->sha256) != 0) {
        free(buffer);
        return fail(QUANT_MANIFEST_ERR_HASH, "quant manifest: %s hashes to %s, the manifest "
                    "records %s", path, digest, artifact->sha256);
    }
    *out = buffer;
    *out_len = got;
    return QUANT_MANIFEST_OK;
}

#ifdef __cplusplus
}
#endif
