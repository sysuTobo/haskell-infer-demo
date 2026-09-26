/**
 * linear_weight.cpp - The weight-only INT4 format's reference arithmetic (plan Q0).
 * See csrc/include/linear_weight.h for the contract.
 *
 * CUDA-free on purpose: the converter, the format checks and the reference dequantizer run on
 * the CPU, so the artifact a GPU kernel is admitted against is produced and checked without a
 * device, and the kernel's gate (Q2) has something independent to compare to.
 */
#include "linear_weight.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static thread_local char g_error[256];

const char *linear_last_error(void) { return g_error; }

void linear_clear_error(void) { g_error[0] = '\0'; }

static LinearStatus fail(LinearStatus status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    return status;
}

/* ------------------------------------------------------------------ */
/* BF16                                                                */
/* ------------------------------------------------------------------ */

uint16_t linear_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* Round to nearest even on the 16 discarded bits, with the IEEE-754 overflow rule: a
     * mantissa that carries all the way up increments the exponent, which is what makes
     * 0x7F7FFFFF round to infinity rather than to a NaN pattern. */
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

float linear_bf16_to_f32(uint16_t bits) {
    const uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

/* ------------------------------------------------------------------ */
/* Layout                                                              */
/* ------------------------------------------------------------------ */

LinearStatus linear_layout_init(struct LinearWeightLayout *out, long long n, long long k,
                                long long group) {
    if (out == NULL) return fail(LINEAR_ERR_ARG, "linear_layout_init: null layout");
    if (n <= 0 || k <= 0) {
        return fail(LINEAR_ERR_RANGE, "linear_layout_init: extents must be positive (n=%lld k=%lld)",
                    n, k);
    }
    if (group != LINEAR_WEIGHT_GROUP_SIZE) {
        return fail(LINEAR_ERR_RANGE,
                    "linear_layout_init: this format's group is %d, not %lld (a different group "
                    "is a different format, not a parameter)",
                    LINEAR_WEIGHT_GROUP_SIZE, group);
    }
    if (k % group != 0) {
        return fail(LINEAR_ERR_RANGE,
                    "linear_layout_init: K=%lld is not a whole number of %lld-wide groups; an "
                    "unsupported shape is refused rather than padded", k, group);
    }
    if (n % LINEAR_WEIGHT_TILE_ROWS != 0) {
        return fail(LINEAR_ERR_RANGE,
                    "linear_layout_init: N=%lld is not a whole number of the %d-row backend tile",
                    n, LINEAR_WEIGHT_TILE_ROWS);
    }
    out->n = n;
    out->k = k;
    out->group = group;
    out->packed_bytes = (n * k) / 2; /* two 4-bit codes per byte */
    out->scale_count = n * (k / group);
    return LINEAR_OK;
}

/* ------------------------------------------------------------------ */
/* Packing                                                             */
/* ------------------------------------------------------------------ */

/* A code in [-7, 7] as a two's-complement nibble. -8 is the format's reserved invalid value,
 * so it is never produced here. */
static uint8_t code_to_nibble(int code) { return (uint8_t)(code & 0x0F); }

int linear_packed_code(const uint8_t *packed, long long row, long long k, long long k_extent) {
    const long long byte = (row * k_extent + k) / 2;
    const uint8_t value = packed[byte];
    const int nibble = (k % 2 == 0) ? (int)(value & 0x0F) : (int)(value >> 4);
    /* Sign-extend the 4-bit two's-complement field. */
    return (nibble >= 8) ? nibble - 16 : nibble;
}

/* ------------------------------------------------------------------ */
/* Quantize / dequantize                                               */
/* ------------------------------------------------------------------ */

LinearStatus linear_quantize_int4(const struct LinearWeightLayout *layout, const float *weights,
                                  uint8_t *packed, uint16_t *scales) {
    if (layout == NULL || weights == NULL || packed == NULL || scales == NULL) {
        return fail(LINEAR_ERR_ARG, "linear_quantize_int4: null argument");
    }
    const long long n = layout->n;
    const long long k = layout->k;
    const long long group = layout->group;
    const long long groups_per_row = k / group;

    /* Validate every input before writing anything, so a failure leaves the caller's buffers
     * untouched (a half-written artifact is worse than none). */
    for (long long i = 0; i < n * k; ++i) {
        if (!isfinite(weights[i])) {
            return fail(LINEAR_ERR_VALUE,
                        "linear_quantize_int4: element %lld is not finite; a quantizer cannot "
                        "invent a scale for it", i);
        }
    }

    /* Both nibbles of every byte are written below, but zeroing first makes the artifact
     * independent of what the caller's buffer held. */
    memset(packed, 0, (size_t)layout->packed_bytes);

    for (long long row = 0; row < n; ++row) {
        for (long long g = 0; g < groups_per_row; ++g) {
            const long long base = row * k + g * group;
            float max_abs = 0.0f;
            for (long long i = 0; i < group; ++i) {
                const float magnitude = fabsf(weights[base + i]);
                if (magnitude > max_abs) max_abs = magnitude;
            }
            /* s = BF16(max|g| / 7), with the all-zero group's s = 1 as the plan specifies. */
            uint16_t scale_bits;
            if (max_abs == 0.0f) {
                scale_bits = linear_f32_to_bf16(1.0f);
            } else {
                scale_bits = linear_f32_to_bf16(max_abs / (float)LINEAR_WEIGHT_QMAX);
            }
            const float scale = linear_bf16_to_f32(scale_bits);
            if (!(scale > 0.0f) || !isfinite(scale)) {
                return fail(LINEAR_ERR_VALUE,
                            "linear_quantize_int4: row %lld group %lld rounds to a scale that is "
                            "not positive and finite (%g); the format cannot represent it",
                            row, g, (double)scale);
            }
            scales[row * groups_per_row + g] = scale_bits;

            for (long long i = 0; i < group; ++i) {
                const float w = weights[base + i];
                /* Round to nearest even against the *rounded* scale, which is the value a
                 * kernel dequantizes with. */
                float q = nearbyintf(w / scale);
                if (q > (float)LINEAR_WEIGHT_QMAX) q = (float)LINEAR_WEIGHT_QMAX;
                if (q < (float)LINEAR_WEIGHT_QMIN) q = (float)LINEAR_WEIGHT_QMIN;
                const int code = (int)q;
                const long long k_index = g * group + i;
                const long long byte = (row * k + k_index) / 2;
                if (k_index % 2 == 0) {
                    packed[byte] = (uint8_t)((packed[byte] & 0xF0) | code_to_nibble(code));
                } else {
                    packed[byte] = (uint8_t)((packed[byte] & 0x0F) |
                                             (uint8_t)(code_to_nibble(code) << 4));
                }
            }
        }
    }
    return LINEAR_OK;
}

LinearStatus linear_dequantize_int4(const struct LinearWeightLayout *layout, const uint8_t *packed,
                                    const uint16_t *scales, float *out) {
    if (layout == NULL || packed == NULL || scales == NULL || out == NULL) {
        return fail(LINEAR_ERR_ARG, "linear_dequantize_int4: null argument");
    }
    const long long n = layout->n;
    const long long k = layout->k;
    const long long group = layout->group;
    const long long groups_per_row = k / group;

    for (long long row = 0; row < n; ++row) {
        for (long long g = 0; g < groups_per_row; ++g) {
            const uint16_t scale_bits = scales[row * groups_per_row + g];
            const float scale = linear_bf16_to_f32(scale_bits);
            if (!(scale > 0.0f) || !isfinite(scale)) {
                return fail(LINEAR_ERR_FORMAT,
                            "linear_dequantize_int4: row %lld group %lld has a scale that is not "
                            "positive and finite (%g)", row, g, (double)scale);
            }
            for (long long i = 0; i < group; ++i) {
                /* The reader's own check of the reserved code: a nibble of 0x8 is not a value
                 * in this format, so an artifact containing one is malformed rather than
                 * slightly wrong. */
                const long long k_index = g * group + i;
                const long long byte = (row * k + k_index) / 2;
                const int nibble = (k_index % 2 == 0) ? (int)(packed[byte] & 0x0F)
                                                      : (int)(packed[byte] >> 4);
                if (nibble == (LINEAR_WEIGHT_INVALID_CODE & 0x0F)) {
                    return fail(LINEAR_ERR_FORMAT,
                                "linear_dequantize_int4: row %lld column %lld carries the reserved "
                                "code -8", row, k_index);
                }
                const int code = (nibble >= 8) ? nibble - 16 : nibble;
                out[row * k + k_index] = (float)code * scale;
            }
        }
    }
    return LINEAR_OK;
}
