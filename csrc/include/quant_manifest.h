/**
 * quant_manifest.h - The reader and validator for the `weights.manifest.json` sidecar.
 *
 * Q1's converter writes quantized artifacts into a directory of its own plus a versioned
 * sidecar; Q2 is where a kernel consumes them. Between those two there has to be something that
 * decides whether the artifact may reach a GPU at all, and that is this file: it parses the
 * sidecar, refuses one whose format block is not the frozen format, checks every entry's extents
 * against the layout the format implies (so the byte counts and the scale counts cannot disagree
 * with the shape), checks that the precision map and the entries describe the same set of
 * (layer, role) cells, checks that each F1 pair is its members' rows, and can read an artifact
 * file while verifying the SHA-256 the manifest recorded.
 *
 * Two boundaries are deliberate.
 *
 *   1. **The format's refusals are not re-implemented.** An entry's shape is put through
 *      `linear_layout_init`, the same function the quantizer and the Q2 kernel gate use, so a
 *      K that is not a whole number of groups or an N that is not a whole number of the tile is
 *      the format's error rather than this reader's opinion.
 *   2. **Parsing is not trusting.** Everything the sidecar claims that this module can re-derive
 *      is re-derived (extents from the shape, pair extents from the members), and the parts it
 *      cannot re-derive without the source checkpoint - the payload bytes - are tied to a digest
 *      that `quant_artifact_read` verifies before handing the bytes on.
 *
 * CUDA-free on purpose, like the other CPU gates: `tests/quant_manifest_test.c` builds a
 * manifest fixture by hand and requires the refusals to fire on a machine with no GPU.
 */
#ifndef HASKELL_INFER_QUANT_MANIFEST_H
#define HASKELL_INFER_QUANT_MANIFEST_H

#include <stddef.h>
#include <stdint.h>

#include "linear_weight.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wire version of the sidecar document (the converter's MANIFEST_VERSION). */
#define QUANT_MANIFEST_VERSION 1

/* A manifest that names more entries, pairs or bytes than these is refused rather than
 * silently truncated. The bounds are deliberate: the deployment model's dense FFN is 64 layers
 * x 3 roles = 192 entries and 64 pairs, so the ceilings leave room without letting a malformed
 * document size a caller's stack. Because the parsed manifest is ~1 MiB at these limits, a
 * caller must not keep it on the stack - the engine's loader allocates it on the heap. */
#define QUANT_MANIFEST_MAX_ENTRIES 256
#define QUANT_MANIFEST_MAX_PAIRS 128
#define QUANT_MANIFEST_NAME_MAX 64
#define QUANT_MANIFEST_PATH_MAX 512
#define QUANT_MANIFEST_SHA256_HEX 65

typedef enum {
    QUANT_MANIFEST_OK = 0,
    QUANT_MANIFEST_ERR_ARG = 1,     /* null argument or a buffer too small */
    QUANT_MANIFEST_ERR_SYNTAX = 2,  /* the document is not the JSON the schema requires */
    QUANT_MANIFEST_ERR_FORMAT = 3,  /* a field disagrees with the frozen format */
    QUANT_MANIFEST_ERR_SHAPE = 4,   /* an extent disagrees with the shape or the members */
    QUANT_MANIFEST_ERR_IO = 5,      /* an artifact could not be read at the size recorded */
    QUANT_MANIFEST_ERR_HASH = 6,    /* an artifact's SHA-256 is not the one recorded */
} QuantManifestStatus;

const char *quant_manifest_last_error(void);
void quant_manifest_clear_error(void);

/* One artifact file. `count` is the number of scale elements and -1 for a packed payload,
 * which is a byte buffer rather than an element array. */
struct QuantArtifact {
    char file[QUANT_MANIFEST_PATH_MAX];
    char dtype[QUANT_MANIFEST_NAME_MAX];
    long long bytes;
    long long count;
    char sha256[QUANT_MANIFEST_SHA256_HEX];
};

struct QuantEntry {
    char role[QUANT_MANIFEST_NAME_MAX];
    int role_index;
    int layer;
    long long n;
    long long k;
    long long group;
    struct QuantArtifact packed;
    struct QuantArtifact scales;
};

/* The F1-compatible concatenation: `n` rows are the members' rows in the recorded order. */
struct QuantPair {
    char name[QUANT_MANIFEST_NAME_MAX];
    int layer;
    char member_role[2][QUANT_MANIFEST_NAME_MAX];
    int member_role_index[2];
    long long n;
    long long k;
    long long group;
    struct QuantArtifact packed;
    struct QuantArtifact scales;
};

struct QuantManifest {
    int manifest_version;
    int format_abi_version;
    long long group;
    int qmin;
    int qmax;
    int zero_point;
    int invalid_code;
    char packed_dtype[QUANT_MANIFEST_NAME_MAX];
    char scale_dtype[QUANT_MANIFEST_NAME_MAX];
    char converter_name[QUANT_MANIFEST_PATH_MAX];
    int converter_version;
    char source_model_dir[QUANT_MANIFEST_PATH_MAX];
    int entry_count;
    struct QuantEntry entries[QUANT_MANIFEST_MAX_ENTRIES];
    int pair_count;
    struct QuantPair pairs[QUANT_MANIFEST_MAX_PAIRS];
    int precision_int4_cells;
    int precision_bf16_cells;
};

/* Parse and validate a sidecar. On failure the status is returned and `err` carries the
 * reason. Nothing is written to `out` that a later stage may use: a rejected manifest is
 * left unusable rather than partially trusted. */
QuantManifestStatus quant_manifest_parse(const char *json, struct QuantManifest *out,
                                         char *err, size_t err_len);

/* The layout an entry implies, through the format's own extent rules. */
QuantManifestStatus quant_manifest_layout(const struct QuantEntry *entry,
                                          struct LinearWeightLayout *out,
                                          char *err, size_t err_len);

/* Lookup helpers: the index into entries/pairs, or -1. */
int quant_manifest_find_entry(const struct QuantManifest *manifest, int layer, const char *role);
int quant_manifest_find_pair(const struct QuantManifest *manifest, const char *name, int layer);

/* Read one artifact from `root` (the directory that holds the manifest), requiring the recorded
 * byte count, the recorded dtype and the recorded SHA-256. `*out` is malloc'd and the caller
 * frees it. `out_len` is the byte count read, which is `artifact->bytes` on success. */
QuantManifestStatus quant_artifact_read(const char *root, const struct QuantArtifact *artifact,
                                        const char *want_dtype, uint8_t **out, size_t *out_len,
                                        char *err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_QUANT_MANIFEST_H */
