/* RoPE: split-half rotation of the first `rotary_dim` elements of each head,
 * against a CPU reference. Covers full rotation (rotary_dim == head_dim, the
 * Qwen3/Mixtral case) and partial rotation (Qwen3.5 rotates 64 of 256). */
#include "flashinfer_ops.h"
#include "test_utils.h"

#include <cmath>

using test::Bf16;
using test::DeviceBuffer;

namespace {

std::vector<Bf16> reference_rope(const std::vector<Bf16> &in, const std::vector<int64_t> &positions,
                                 int tokens, int heads, int head_dim, int rotary_dim, float theta) {
    std::vector<Bf16> out = in;
    const int half = rotary_dim / 2;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            Bf16 *row = out.data() + ((size_t)t * heads + h) * head_dim;
            const Bf16 *src = in.data() + ((size_t)t * heads + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                const double angle = (double)positions[t] * std::pow((double)theta, -2.0 * i / rotary_dim);
                const double cos_a = std::cos(angle), sin_a = std::sin(angle);
                const double x1 = (double)test::value(src[i]);
                const double x2 = (double)test::value(src[i + half]);
                row[i] = test::bf16((float)(x1 * cos_a - x2 * sin_a));
                row[i + half] = test::bf16((float)(x1 * sin_a + x2 * cos_a));
            }
        }
    }
    return out;
}

bool run_case(const char *name, int tokens, int q_heads, int kv_heads, int head_dim,
              int rotary_dim, float theta, test::Stream &stream) {
    std::vector<Bf16> q((size_t)tokens * q_heads * head_dim);
    std::vector<Bf16> k((size_t)tokens * kv_heads * head_dim);
    for (size_t i = 0; i < q.size(); ++i) q[i] = test::bf16(test::sample(i, 31));
    for (size_t i = 0; i < k.size(); ++i) k[i] = test::bf16(test::sample(i, 37));
    std::vector<int64_t> positions(tokens);
    for (int t = 0; t < tokens; ++t) positions[t] = t;

    DeviceBuffer<Bf16> d_q(q.size()), d_k(k.size());
    DeviceBuffer<int64_t> d_pos(positions.size());
    d_q.upload(q, stream.get());
    d_k.upload(k, stream.get());
    d_pos.upload(positions, stream.get());
    kernel_flashinfer_rope(d_q.get(), d_k.get(), d_pos.get(), tokens, q_heads, kv_heads, head_dim,
                           rotary_dim, theta, stream.get());
    CUDA_CHECK(cudaGetLastError());

    bool ok = test::compare(name, d_q.download(stream.get()),
                            reference_rope(q, positions, tokens, q_heads, head_dim, rotary_dim, theta),
                            2e-2, 2e-2);
    ok &= test::compare(name, d_k.download(stream.get()),
                        reference_rope(k, positions, tokens, kv_heads, head_dim, rotary_dim, theta),
                        2e-2, 2e-2);
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    test::Stream stream;
    // Qwen3.8: head_dim 256, partial rotation of 64 dims, GQA 24:4.
    ok &= run_case("rope 64/256 gqa24:4", 3, 24, 4, 256, 64, 1e7f, stream);
    // Qwen3/Mixtral: full rotation of 128 dims, GQA 32:8.
    ok &= run_case("rope 128/128 gqa32:8", 3, 32, 8, 128, 128, 1e6f, stream);
    // Single token, full rotation (the decode/prefill-first-token shape).
    ok &= run_case("rope 128/128 single token", 1, 32, 8, 128, 128, 1e6f, stream);
    return test::finish("test_rope", ok);
}
