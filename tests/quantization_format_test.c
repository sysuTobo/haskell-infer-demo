/*
 * quantization_format_test.c - CPU gate for the weight-only INT4 format (plan Q0).
 *
 * No GPU and no weights. Q0's deliverable is a *frozen* format plus a reference an eventual
 * kernel can be admitted against, so this gate checks the rules the plan states one by one:
 * the layout's refusals, BF16 rounding, the two's-complement nibble packing (including the
 * byte pattern by hand, because a round trip cannot catch a self-consistent swap of the
 * nibbles), the reserved code, the all-zero group's scale, signed extrema and clipping,
 * nonfinite and denormal-scale refusals, and quantization as a fixed point of dequantization.
 *
 * `--emit PATH` writes a canonical JSON fixture - the input as hex floats, the payload as hex
 * bytes and the scales as hex floats - which tests/test_quantization_format.py re-derives
 * *from the format's definition* with numpy, so the C reference cannot certify itself.
 *
 * Run: ctest --test-dir csrc/build-libs -R test_quantization_format.
 */
#include "linear_weight.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

static void check(int condition, const char *fmt, ...) {
    ++g_checks;
    if (condition) return;
    ++g_failures;
    va_list ap;
    va_start(ap, fmt);
    fputs("quantization_format_test: FAIL: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void check_close(double got, double want, double tol, const char *what) {
    check(fabs(got - want) <= tol, "%s: got %.9g want %.9g (tol %.3g)", what, got, want, tol);
}

/* ------------------------------------------------------------------ */
/* The layout                                                         */
/* ------------------------------------------------------------------ */

static void test_layout(void) {
    struct LinearWeightLayout layout;
    check(linear_layout_init(&layout, 8, 128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_OK,
          "a [8, 128] weight was refused: %s", linear_last_error());
    check(layout.packed_bytes == 8 * 128 / 2 && layout.scale_count == 8,
          "a [8, 128] weight packs to %lld bytes and %lld scales",
          layout.packed_bytes, layout.scale_count);

    check(linear_layout_init(&layout, 8, 256, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_OK,
          "a [8, 256] weight was refused: %s", linear_last_error());
    check(layout.scale_count == 16, "a [8, 256] weight has %lld scales, expected 16",
          layout.scale_count);

    /* An unsupported shape is an error rather than hidden padding. */
    check(linear_layout_init(&layout, 8, 192, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_ERR_RANGE,
          "K=192 (1.5 groups) was accepted");
    check(linear_layout_init(&layout, 12, 128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_ERR_RANGE,
          "N=12 (1.5 tiles) was accepted");
    check(linear_layout_init(&layout, 0, 128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_ERR_RANGE,
          "N=0 was accepted");
    check(linear_layout_init(&layout, 8, -128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_ERR_RANGE,
          "K<0 was accepted");
    /* The group is part of the format, not a parameter. */
    check(linear_layout_init(&layout, 8, 128, 64) == LINEAR_ERR_RANGE,
          "a 64-wide group was accepted by a format whose group is 128");
    check(linear_layout_init(NULL, 8, 128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_ERR_ARG,
          "a null layout was accepted");
}

/* ------------------------------------------------------------------ */
/* BF16 rounding                                                      */
/* ------------------------------------------------------------------ */

static void test_bf16(void) {
    check(linear_f32_to_bf16(1.0f) == 0x3F80, "1.0 did not round to 0x3F80");
    check(linear_f32_to_bf16(-2.0f) == 0xC000, "-2.0 did not round to 0xC000");
    check(linear_f32_to_bf16(0.0f) == 0x0000, "0.0 did not round to 0");
    check_close(linear_bf16_to_f32(0x3F80), 1.0, 0.0, "0x3F80 did not read back as 1.0");
    check_close(linear_bf16_to_f32(0xC000), -2.0, 0.0, "0xC000 did not read back as -2.0");

    /* Round to nearest even: the value exactly between two BF16 neighbours takes the even
     * one. 1 + 2^-8 is halfway between 1.0 (mantissa 0) and the next BF16 (mantissa 1), so
     * it rounds back down to 1.0; the tie one ULP higher rounds up. */
    const float halfway = 1.0f + ldexpf(1.0f, -8);
    check(linear_f32_to_bf16(halfway) == 0x3F80,
          "1 + 2^-8 rounded to 0x%04X, expected the even neighbour 0x3F80",
          linear_f32_to_bf16(halfway));
    const float halfway_odd = 1.0f + 3.0f * ldexpf(1.0f, -8);
    check(linear_f32_to_bf16(halfway_odd) == 0x3F82,
          "1 + 3*2^-8 rounded to 0x%04X, expected the even neighbour 0x3F82",
          linear_f32_to_bf16(halfway_odd));

    /* An FP32 value above the BF16 range rounds to infinity rather than to a NaN pattern. */
    const uint16_t big = linear_f32_to_bf16(3.4e38f);
    check(big == 0x7F80 || big == 0x7F7F, "an overflowing value rounded to 0x%04X", big);
}

/* ------------------------------------------------------------------ */
/* The packing                                                        */
/* ------------------------------------------------------------------ */

static void test_packing(void) {
    struct LinearWeightLayout layout;
    check(linear_layout_init(&layout, 8, 128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_OK,
          "layout failed");
    float *weights = (float *)calloc((size_t)(8 * 128), sizeof(float));
    uint8_t *packed = (uint8_t *)calloc((size_t)layout.packed_bytes, 1);
    uint16_t *scales = (uint16_t *)calloc((size_t)layout.scale_count, sizeof(uint16_t));
    float *back = (float *)calloc((size_t)(8 * 128), sizeof(float));

    /* Row 0 holds hand-picked values whose codes are exactly themselves once the group's
     * maximum is 7, which makes s = BF16(1.0) = 1 and takes the arithmetic out of the way. */
    float hand[128];
    memset(hand, 0, sizeof(hand));
    hand[0] = 0.0f;
    hand[1] = 1.0f;
    hand[2] = -1.0f;
    hand[3] = -7.0f;
    hand[4] = 7.0f;
    hand[5] = 3.0f;
    hand[6] = -3.0f;
    hand[7] = 2.0f;
    memcpy(weights, hand, sizeof(hand));

    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_OK,
          "quantization failed: %s", linear_last_error());
    check_close(linear_bf16_to_f32(scales[0]), 1.0, 0.0,
                "a group whose maximum is 7 must scale by 1.0");

    /* The byte pattern by hand: byte j packs K indices 2j (low nibble) and 2j+1 (high). */
    check(packed[0] == (uint8_t)((0x0 & 0x0F) | (0x1 << 4)),
          "byte 0 is 0x%02X, expected 0x10 (k=0 low, k=1 high)", packed[0]);
    check(packed[1] == (uint8_t)((0xF & 0x0F) | (0x9 << 4)),
          "byte 1 is 0x%02X, expected 0x9F (k=2 is -1, k=3 is -7)", packed[1]);
    check(packed[2] == (uint8_t)((0x7 & 0x0F) | (0x3 << 4)),
          "byte 2 is 0x%02X, expected 0x37 (k=4 is 7, k=5 is 3)", packed[2]);
    check(packed[3] == (uint8_t)((0xD & 0x0F) | (0x2 << 4)),
          "byte 3 is 0x%02X, expected 0x2D (k=6 is -3, k=7 is 2)", packed[3]);

    /* And the reader agrees with the writer, index by index. */
    const int expected_codes[8] = {0, 1, -1, -7, 7, 3, -3, 2};
    for (long long k = 0; k < 8; ++k) {
        check(linear_packed_code(packed, 0, k, 128) == expected_codes[k],
              "code at k=%lld is %d, expected %d", k, linear_packed_code(packed, 0, k, 128),
              expected_codes[k]);
    }

    /* Round trip: every element reads back as its own code times the BF16 scale. */
    check(linear_dequantize_int4(&layout, packed, scales, back) == LINEAR_OK,
          "dequantization failed: %s", linear_last_error());
    for (long long k = 0; k < 128; ++k) {
        const float want = (float)expected_codes[k < 8 ? k : 0] * linear_bf16_to_f32(scales[0]);
        const float got = back[k];
        if (k < 8) {
            check_close(got, want, 0.0, "the first row's round trip");
        } else {
            check(got == 0.0f, "a zero element came back as %g", (double)got);
        }
    }

    free(weights);
    free(packed);
    free(scales);
    free(back);
}

/* The zero group, the denormal group, the reserved code and the fixed-point property. */
static void test_edge_groups(void) {
    struct LinearWeightLayout layout;
    check(linear_layout_init(&layout, 8, 128, LINEAR_WEIGHT_GROUP_SIZE) == LINEAR_OK, "layout");
    float *weights = (float *)calloc(8 * 128, sizeof(float));
    uint8_t *packed = (uint8_t *)calloc((size_t)layout.packed_bytes, 1);
    uint16_t *scales = (uint16_t *)calloc((size_t)layout.scale_count, sizeof(uint16_t));
    float *back = (float *)calloc(8 * 128, sizeof(float));

    /* An all-zero group takes s = 1, as the plan specifies, rather than s = 0/0. */
    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_OK,
          "an all-zero weight was refused: %s", linear_last_error());
    check(linear_bf16_to_f32(scales[0]) == 1.0f, "an all-zero group scaled by %g",
          (double)linear_bf16_to_f32(scales[0]));
    for (long long i = 0; i < layout.packed_bytes; ++i) {
        check(packed[i] == 0, "an all-zero group produced a nonzero byte at %lld", i);
    }

    /* A maximum that is not representable: the scale rounds, and the code clips to 7 rather
     * than reaching the reserved -8 or an out-of-range 8. */
    weights[0] = 0.7000001f;
    weights[1] = -0.7000001f;
    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_OK,
          "the clipping fixture failed: %s", linear_last_error());
    check(linear_packed_code(packed, 0, 0, 128) == 7, "the positive extreme is %d, expected 7",
          linear_packed_code(packed, 0, 0, 128));
    check(linear_packed_code(packed, 0, 1, 128) == -7, "the negative extreme is %d, expected -7",
          linear_packed_code(packed, 0, 1, 128));
    weights[0] = 0.0f;
    weights[1] = 0.0f;

    /* A denormal group: max|w|/7 rounds to zero in BF16, and the format cannot represent it,
     * so it is refused rather than quantized against s = 0. */
    weights[0] = 1e-40f;
    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_ERR_VALUE,
          "a group whose scale rounds to zero was accepted");
    weights[0] = 0.0f;

    /* Nonfinite input is refused. */
    weights[5] = NAN;
    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_ERR_VALUE,
          "a NaN weight was accepted");
    weights[5] = INFINITY;
    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_ERR_VALUE,
          "an infinite weight was accepted");
    weights[5] = 0.0f;

    /* A malformed artifact: the reserved code -8 in the payload is a format error. */
    check(linear_quantize_int4(&layout, weights, packed, scales) == LINEAR_OK, "quantize");
    packed[0] = (uint8_t)((packed[0] & 0xF0) | 0x08);
    check(linear_dequantize_int4(&layout, packed, scales, back) == LINEAR_ERR_FORMAT,
          "the reserved code -8 was read as a value");
    packed[0] = (uint8_t)(packed[0] & 0xF0);

    /* Quantization is a fixed point of dequantization: quantizing what was quantized gives
     * the same artifact, which is what "the kernel matches the reference" reduces to. */
    check(linear_dequantize_int4(&layout, packed, scales, back) == LINEAR_OK, "dequantize");
    uint8_t *again = (uint8_t *)calloc((size_t)layout.packed_bytes, 1);
    uint16_t *scales_again = (uint16_t *)calloc((size_t)layout.scale_count, sizeof(uint16_t));
    check(linear_quantize_int4(&layout, back, again, scales_again) == LINEAR_OK, "requantize");
    check(memcmp(packed, again, (size_t)layout.packed_bytes) == 0,
          "requantizing the dequantized weight changed the payload");
    check(memcmp(scales, scales_again, (size_t)layout.scale_count * sizeof(uint16_t)) == 0,
          "requantizing the dequantized weight changed the scales");

    free(weights);
    free(packed);
    free(scales);
    free(back);
    free(again);
    free(scales_again);
}

/* ------------------------------------------------------------------ */
/* The fixture the Python re-derivation reads                         */
/* ------------------------------------------------------------------ */

static int emit_fixture(const char *path) {
    /* Eight rows (one tile), 256 columns (two groups), so the fixture covers a group
     * boundary and the tile rule at once. The values are deterministic. */
    const long long n = 8, k = 256;
    struct LinearWeightLayout layout;
    if (linear_layout_init(&layout, n, k, LINEAR_WEIGHT_GROUP_SIZE) != LINEAR_OK) return 1;
    float *weights = (float *)calloc((size_t)(n * k), sizeof(float));
    uint8_t *packed = (uint8_t *)calloc((size_t)layout.packed_bytes, 1);
    uint16_t *scales = (uint16_t *)calloc((size_t)layout.scale_count, sizeof(uint16_t));
    for (long long i = 0; i < n * k; ++i) {
        const int pattern = (int)(i % 17) - 8;
        weights[i] = (float)pattern * 0.03125f + (float)(i % 3) * 0.0078125f;
    }
    if (linear_quantize_int4(&layout, weights, packed, scales) != LINEAR_OK) {
        fprintf(stderr, "emit: %s\n", linear_last_error());
        return 1;
    }
    FILE *out = fopen(path, "w");
    if (out == NULL) return 1;
    fprintf(out, "{\n  \"n\": %lld,\n  \"k\": %lld,\n  \"group\": %lld,\n", n, k, layout.group);
    fprintf(out, "  \"packed_bytes\": %lld,\n  \"scale_count\": %lld,\n", layout.packed_bytes,
            layout.scale_count);
    /* The FP32 values go out as their bit patterns: a hex-float literal is not valid JSON,
     * and a decimal round trip would blur exactly what this fixture is for. */
    fprintf(out, "  \"weights_f32_bits\": [");
    for (long long i = 0; i < n * k; ++i) {
        uint32_t bits;
        memcpy(&bits, &weights[i], sizeof(bits));
        fprintf(out, "%s%u", i ? ", " : "", (unsigned)bits);
    }
    fprintf(out, "],\n  \"packed\": [");
    for (long long i = 0; i < layout.packed_bytes; ++i) {
        fprintf(out, "%s%d", i ? ", " : "", (int)packed[i]);
    }
    fprintf(out, "],\n  \"scales_bf16\": [");
    for (long long i = 0; i < layout.scale_count; ++i) {
        fprintf(out, "%s%d", i ? ", " : "", (int)scales[i]);
    }
    /* The reference dequantization, so the Python side can compare it as well as the payload. */
    float *back = (float *)calloc((size_t)(n * k), sizeof(float));
    if (back == NULL || linear_dequantize_int4(&layout, packed, scales, back) != LINEAR_OK) {
        fprintf(stderr, "emit: %s\n", linear_last_error());
        return 1;
    }
    fprintf(out, "],\n  \"dequantized_f32_bits\": [");
    for (long long i = 0; i < n * k; ++i) {
        uint32_t bits;
        memcpy(&bits, &back[i], sizeof(bits));
        fprintf(out, "%s%u", i ? ", " : "", (unsigned)bits);
    }
    free(back);
    fprintf(out, "]\n}\n");
    fclose(out);
    free(weights);
    free(packed);
    free(scales);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) return emit_fixture(argv[2]);
    test_layout();
    test_bf16();
    test_packing();
    test_edge_groups();

    if (g_failures != 0) {
        fprintf(stderr, "quantization_format_test: %d of %d check(s) failed\n", g_failures,
                g_checks);
        return EXIT_FAILURE;
    }
    printf("quantization_format_test: the INT4 layout, its packing, its refusals and the "
           "reference dequantizer behave as the format states\n");
    return EXIT_SUCCESS;
}
