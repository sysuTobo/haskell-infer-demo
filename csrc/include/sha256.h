/**
 * sha256.h - Streaming SHA-256, dependency-free.
 *
 * The execution manifest derives every identity from the *content* of a
 * canonical encoding, so the engine needs a hash it can compute itself. A
 * streaming context is exposed because the parameter identity is built by
 * walking a 1000+ entry tensor index in canonical order and must not be
 * assembled in one buffer first.
 */
#ifndef HASKELL_INFER_SHA256_H
#define HASKELL_INFER_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hex digits a digest renders to, including the NUL terminator. */
#define SHA256_HEX_LEN 65

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[64];
    size_t block_len;
};

void sha256_init(struct sha256_ctx *ctx);
void sha256_update(struct sha256_ctx *ctx, const void *data, size_t len);

/* Writes the 32-byte digest. The context must not be updated afterwards. */
void sha256_final(struct sha256_ctx *ctx, unsigned char out[32]);

/* Lowercase hex of a raw digest, NUL-terminated (needs SHA256_HEX_LEN bytes). */
void sha256_to_hex(const unsigned char digest[32], char out[SHA256_HEX_LEN]);

/* One-shot convenience wrappers. */
void sha256_hex(const void *data, size_t len, char out[SHA256_HEX_LEN]);
void sha256_hex_sum(const unsigned char digest[32], char out[SHA256_HEX_LEN]);

#ifdef __cplusplus
}
#endif

#endif /* HASKELL_INFER_SHA256_H */
