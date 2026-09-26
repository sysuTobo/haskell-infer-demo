/**
 * linear_weight.h - The weight-only INT4 format and its reference arithmetic
 * (plan "Q - Weight-only quantization", Q0).
 *
 * docs/plan-numeric-contract.md's Q starts with the format, and it is specific about the
 * rules because a quantization format that is "roughly INT4 with scales" is not one:
 *
 *   symmetric signed INT4, K-axis groups of 128 and BF16 scales; source values are FP32, the
 *   scale is `s = BF16(max(abs(group))/7)`, the code is round-to-nearest-even `q = round(w/s)`
 *   clipped to [-7, 7], the zero-point is 0, and an all-zero group takes s = 1; nonfinite
 *   inputs and nonpositive or nonfinite rounded scales are rejected. Two two's-complement
 *   nibbles are packed per U8 byte with the *lower* K index in the low nibble, and -8 is
 *   reserved invalid in this format.
 *
 * Three things follow from writing it as code rather than as a paragraph, and they are the
 * reason this file exists:
 *
 *   1. **The reference is independent of any kernel.** `linear_dequantize_int4` is the
 *      definition a kernel has to match, in the same CUDA-free arithmetic, so "the kernel
 *      agrees with the reference" is a comparison two implementations can actually lose.
 *      Q2 is where a kernel is admitted against it; the format is *fixed* here (Q0) and the
 *      sm_86 kernel feasibility check is the gate that runs before a kernel uses it.
 *   2. **The packing order is testable.** Nibble order inside a byte is the kind of detail
 *      that a round trip cannot catch - unpacking with the same wrong convention reads back
 *      correctly - so the gate checks a hand-computed byte pattern, not only a round trip.
 *   3. **Unsupported shapes are refused, not padded.** `linear_layout_init` refuses a K that
 *      is not a whole number of groups and an N that is not a whole number of the 8-row tile
 *      the plan's alignment requirement names, rather than inventing hidden padding that
 *      would make the artifact's byte count disagree with its manifest.
 *
 * Scope: the plan's initial quantization scope is the dense FFN gate/up/down roles with BF16
 * activations, so nothing here knows about MoE, attention or GDN. `-8` staying invalid is
 * what keeps this format's code range symmetric: the quantizer never produces it, and a
 * reader that sees it rejects the artifact instead of reading it as a real value.
 */
#ifndef HASKELL_INFER_LINEAR_WEIGHT_H
#define HASKELL_INFER_LINEAR_WEIGHT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LINEAR_WEIGHT_ABI_VERSION 1

/* The frozen format's constants. `group` is the K-axis group size, `tile` the row multiple the
 * backend tile needs; both are part of the format, not parameters a caller may vary. */
#define LINEAR_WEIGHT_GROUP_SIZE 128
#define LINEAR_WEIGHT_TILE_ROWS 8
#define LINEAR_WEIGHT_QMIN (-7)
#define LINEAR_WEIGHT_QMAX 7
#define LINEAR_WEIGHT_INVALID_CODE (-8)

typedef enum {
    LINEAR_OK = 0,
    LINEAR_ERR_ARG = 1,     /* null or inconsistent argument (see linear_last_error) */
    LINEAR_ERR_RANGE = 2,   /* a dimension, group or alignment out of range */
    LINEAR_ERR_VALUE = 3,   /* a nonfinite input, or a scale that is not positive and finite */
    LINEAR_ERR_FORMAT = 4,  /* the artifact is malformed: reserved code, missing scales, ... */
} LinearStatus;

const char *linear_last_error(void);
void linear_clear_error(void);

/* The logical shape and the derived packed extents. `packed_bytes` counts the nibble payload
 * only (N*K/2); `scale_count` is the number of BF16 scales (N * K/group). */
struct LinearWeightLayout {
    long long n;
    long long k;
    long long group;
    long long packed_bytes;
    long long scale_count;
};

/* Fill a layout for a logical [N, K] weight. Refuses a negative or zero extent, a group that
 * is not the frozen one, a K that is not a whole number of groups, and an N that is not a
 * whole number of the tile rows - an unsupported shape is an error, not something to pad. */
LinearStatus linear_layout_init(struct LinearWeightLayout *out, long long n, long long k,
                                long long group);

/* BF16 conversion, round-to-nearest-even. The format's scales are BF16, so this is part of the
 * contract rather than a helper: `s = BF16(max|g|/7)` and the code is `round(w / s)` against
 * the *rounded* scale, which is what makes the reference and a kernel comparable at all. */
uint16_t linear_f32_to_bf16(float value);
float linear_bf16_to_f32(uint16_t bits);

/* Quantize an FP32 [N, K] row-major weight. `packed` receives `layout->packed_bytes` bytes and
 * `scales` receives `layout->scale_count` BF16 values in row-major (row, group) order. Both
 * are left untouched on any failure. */
LinearStatus linear_quantize_int4(const struct LinearWeightLayout *layout, const float *weights,
                                  uint8_t *packed, uint16_t *scales);

/* Dequantize back to FP32. This is the *reference* a kernel is admitted against (plan Q2):
 * `w = code * bf16_to_f32(scale)`, with each code read from its nibble. A reserved code (-8)
 * is a format error, not a value. */
LinearStatus linear_dequantize_int4(const struct LinearWeightLayout *layout, const uint8_t *packed,
                                    const uint16_t *scales, float *out);

/* Read one code out of the packed payload: the lower K index lives in the low nibble. Exposed
 * because the gate checks it directly - a round trip cannot catch a self-consistent swap. */
int linear_packed_code(const uint8_t *packed, long long row, long long k, long long k_extent);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_LINEAR_WEIGHT_H */
