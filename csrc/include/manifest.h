/**
 * manifest.h - The versioned execution manifest: which numerical execution a
 * capture actually observed.
 *
 * The architecture descriptor stays portable (dimensions and semantics, no
 * runtime facts). This module derives three *separate* content identities from
 * it plus the actual execution, and reports the build/runtime provenance that a
 * bitwise comparison has to agree on:
 *
 *   semantic_id           architecture dimensions, layer/role semantics, tied
 *                         parameters and the mathematical conventions. Weights
 *                         are a separate identity (see weights.parameter_*).
 *   numerical_policy_id   region/case -> implementation bindings, the effective
 *                         constants and dtype/rounding/reduction choices.
 *   deployment_id         placement, allocation and transfer choices.
 *
 * Field ownership is explicit and documented in docs/manifest-contract.md: a
 * constant that describes both the mathematical function and its numerical
 * realization is projected into *both* identities (rms_eps_declared vs
 * effective_rms_eps) rather than silently omitted from one of them.
 *
 * Canonical encoding: every object is emitted with its keys sorted by byte
 * value, no whitespace, integers bare, non-integer constants as decimal strings
 * and booleans/null as JSON literals; strings are printable ASCII only, so no
 * escaping convention can differ between this encoder and another language's
 * JSON writer. Each identity block carries its own SHA-256, computed over the
 * canonical text of that block's fields, which lets any consumer re-derive the
 * digest from the parsed JSON instead of trusting the emitter.
 *
 * CUDA-free on purpose: tests/manifest_test.c builds a manifest from a parsed
 * descriptor plus caller-supplied runtime facts on a machine with no GPU, so the
 * identity matrix and the canonical round-trip are CPU gates. csrc/engine.cu
 * supplies the real build/runtime facts for the engine query.
 */
#ifndef HASKELL_INFER_MANIFEST_H
#define HASKELL_INFER_MANIFEST_H

#include <stddef.h>

#include "model_desc.h"
#include "sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wire version of the manifest document (distinct from ENGINE_DESC_VERSION). */
#define ENGINE_MANIFEST_VERSION 1
/* Buffer an identity-bearing manifest needs. engine_manifest() returns a
 * negative code rather than truncating. */
#define ENGINE_MANIFEST_MAX 262144

/* Device facts the manifest records. ENGINE_MANIFEST_MAX_DEVICES bounds the
 * array; an engine on more devices than this fails the query instead of
 * silently dropping provenance. */
#define ENGINE_MANIFEST_MAX_DEVICES 16

/* A digest rendered as lowercase hex, NUL included. */
typedef char manifest_hex_t[SHA256_HEX_LEN];

/* Text that is unknown to the *build* still has to be reported, so a strict
 * comparison can refuse it. These are the spellings a consumer must treat as
 * "not established" (see docs/manifest-contract.md). */
#define MANIFEST_UNKNOWN "unknown"
#define MANIFEST_UNAVAILABLE "unavailable"
#define MANIFEST_UNSUPPORTED "unsupported"

/* Build facts. Filled from a generated header (csrc/gen_build_info.cmake) that
 * is produced by the same build step that compiles the kernels, so the values
 * come from the actual build instead of from a caller-supplied label. Empty
 * strings are reported as MANIFEST_UNKNOWN. */
struct ManifestBuildInfo {
    char engine_version[32];
    char git_commit[64];
    char cuda_toolkit[32];
    char cuda_archs[64];            /* SASS/PTX targets, e.g. "86;89;90a;90-virtual" */
    char triton_archs[32];          /* AOT cubin targets, e.g. "86;89;90" */
    char triton_version[32];
    char fla_version[32];
    char flashinfer_header_sha256[SHA256_HEX_LEN];  /* header the engine compiled against */
    char nvcc_flags_sha256[SHA256_HEX_LEN];
    char generated_kernels_sha256[SHA256_HEX_LEN];  /* AOT/generated sources */
};

/* Runtime facts for one device the engine holds. kernel_path/triton_cubin_arch
 * are derived from the device's compute capability and the *build* target lists
 * (the selector's rule, not an observation of the launched binary); the field
 * names say "selected" for that reason. */
struct ManifestDevice {
    int cuda_ordinal;
    char name[96];
    char compute_capability[16];
    char uuid[48];              /* "GPU-" + a 36 character UUID */
    char kernel_path[32];
    char triton_cubin_arch[16];
};

/* The checkpoint's immutable parameter identity. parameter_manifest_sha256
 * covers the canonical tensor index (name, dtype, shape, byte count in sorted
 * name order) and is cheap to compute at load; content_sha256 covers the raw
 * tensor bytes and is optional, because hashing a 50 GiB checkpoint is minutes
 * of I/O and the caller has to ask for it. */
struct ManifestWeights {
    manifest_hex_t parameter_manifest_sha256;
    long long tensor_count;
    manifest_hex_t shards_sha256;      /* sorted shard basenames */
    int content_hash_present;
    manifest_hex_t content_sha256;
};

/* One forward or backward region and the case pair it was executed for.
 * determinism is about *repeatability* of that implementation for a fixed
 * build/device/launch configuration: "deterministic" only where the
 * implementation has no cross-thread reduction to reorder, "unverified" where
 * the reduction or library tiling order has not been established, and
 * "not_implemented" for regions that do not exist yet. rng_dependency is a
 * separate axis: a seed does not order atomics.
 *
 * `cases` here is the coarse availability string a capture records. The Stage-1
 * inventory (csrc/include/regions.h) refines it into a per-case-pair verdict with
 * evidence; test_region_inventory requires the two to name the same regions, and
 * an `exact` verdict there is accepted only where this registry says
 * "deterministic". */
struct ManifestRegion {
    const char *region;
    const char *cases;
    const char *implementation;
    const char *determinism;
    const char *mechanism;
    int rng_dependency;
};

/* The generation-only selection policy. The transform and its arithmetic belong
 * to the numerical policy; the concrete temperature and seed of one request are
 * per-request replay data that lives in the capture record, so changing only
 * the temperature never changes a model, weight or policy identity. */
struct ManifestSampling {
    const char *mode;
    const char *transform;
    int transform_version;
    const char *arithmetic;
    const char *rng;                /* "none" for greedy */
};

struct ManifestInputs {
    /* Architecture + semantics. */
    const struct ModelDesc *desc;
    /* SHA-256 (and byte length) of the canonical descriptor text, so the
     * manifest references the descriptor without embedding runtime facts in it. */
    const char *descriptor_sha256;
    long long descriptor_bytes;

    /* Provenance. */
    const struct ManifestBuildInfo *build;
    const char *cuda_runtime_version;
    const char *cuda_driver_version;
    const char *cublas_version;
    const struct ManifestDevice *devices;
    int device_count;

    /* Weights. */
    const struct ManifestWeights *weights;

    /* Weight-only quantization (plan Q2). Nonzero when the dense FFN's operands were replaced
     * by packed INT4 weights, which is a *numerical policy* change - the products are formed by
     * a different kernel against a different operand - so it has to move
     * numerical_policy_id even though the architecture and the deployment are unchanged. */
    int weight_only_int4;

    /* Plan F2: the mixer's residual update and the FFN's post-norm are one pass, which changes
     * the region table's entries and the arithmetic the norm sees - a numerical policy. */
    int fuse_residual_norm;

    /* Deployment: placement plus allocation/transfer choices. layer_device holds
     * internal device indices (the engine's own numbering); device_ordinals maps
     * them to CUDA ordinals. */
    int replicated;
    int ep_size;
    int declared_tp_rank;
    int declared_ep_rank;
    const int *device_ordinals;
    const int *layer_device;
    int num_layers;

    /* Region registry and sampling policy (defaults exist for both). */
    const struct ManifestRegion *regions;
    int region_count;
    const struct ManifestSampling *sampling;
};

/* Write the canonical manifest into buf. Returns the number of bytes written
 * (excluding the terminator) or a negative error code: -2 when buf is too small,
 * -1 for invalid or non-ASCII-printable input. */
int manifest_format(const struct ManifestInputs *in, char *buf, int buf_len);

/* The committed region registry for the current forward path, in a stable order
 * so its digest is reproducible. */
const struct ManifestRegion *manifest_default_regions(int *count);

/* The current generation policy: greedy, no RNG consumption. */
const struct ManifestSampling *manifest_default_sampling(void);

/* Report the empty/unknown spellings as the reported value they stand for. */
const char *manifest_or_unknown(const char *value);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_MANIFEST_H */
