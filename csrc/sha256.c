/**
 * sha256.c - FIPS 180-4 SHA-256.
 *
 * Straight line implementation: no tables beyond the round constants, no
 * allocation, no dependency. tests/manifest_test.c pins it against the published
 * FIPS vectors, including the two-block case that exercises the padding
 * boundary.
 */
#include "sha256.h"

#include <string.h>

static const uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static uint32_t rotr(uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

static void sha256_compress(struct sha256_ctx *ctx, const unsigned char block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

void sha256_init(struct sha256_ctx *ctx) {
    ctx->state[0] = 0x6a09e667u;
    ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u;
    ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu;
    ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu;
    ctx->state[7] = 0x5be0cd19u;
    ctx->bit_length = 0;
    ctx->block_len = 0;
}

void sha256_update(struct sha256_ctx *ctx, const void *data, size_t len) {
    const unsigned char *bytes = (const unsigned char *)data;
    ctx->bit_length += (uint64_t)len * 8u;
    while (len > 0) {
        size_t room = 64 - ctx->block_len;
        size_t take = len < room ? len : room;
        memcpy(ctx->block + ctx->block_len, bytes, take);
        ctx->block_len += take;
        bytes += take;
        len -= take;
        if (ctx->block_len == 64) {
            sha256_compress(ctx, ctx->block);
            ctx->block_len = 0;
        }
    }
}

void sha256_final(struct sha256_ctx *ctx, unsigned char out[32]) {
    uint64_t bit_length = ctx->bit_length;
    unsigned char pad = 0x80;
    sha256_update(ctx, &pad, 1);
    unsigned char zero = 0;
    while (ctx->block_len != 56) sha256_update(ctx, &zero, 1);
    unsigned char length_bytes[8];
    for (int i = 0; i < 8; ++i)
        length_bytes[i] = (unsigned char)(bit_length >> (56 - i * 8));
    sha256_update(ctx, length_bytes, 8);
    for (int i = 0; i < 8; ++i) {
        out[i * 4] = (unsigned char)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(ctx->state[i] >> 8);
        out[i * 4 + 3] = (unsigned char)ctx->state[i];
    }
}

void sha256_to_hex(const unsigned char digest[32], char out[SHA256_HEX_LEN]) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[i * 2] = digits[digest[i] >> 4];
        out[i * 2 + 1] = digits[digest[i] & 0x0f];
    }
    out[64] = '\0';
}

void sha256_hex_sum(const unsigned char digest[32], char out[SHA256_HEX_LEN]) {
    sha256_to_hex(digest, out);
}

void sha256_hex(const void *data, size_t len, char out[SHA256_HEX_LEN]) {
    struct sha256_ctx ctx;
    unsigned char digest[32];
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
    sha256_to_hex(digest, out);
}
