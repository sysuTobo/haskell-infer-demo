#include "fla_ops.h"
#include "aot_kernels.h"

#include <cmath>
#include <stdexcept>

namespace {
size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

struct Workspace {
    char *base;
    size_t offset = 0;

    template <typename T> T *take(size_t count) {
        T *result = reinterpret_cast<T *>(base + offset);
        offset += aligned(count * sizeof(T));
        return result;
    }
};
}

size_t kernel_fla_workspace_size(int tokens, int value_heads) {
    if (tokens < 1 || tokens > 128 || value_heads != 48)
        throw std::invalid_argument("FLA supports 1..128 tokens and 48 value heads");
    size_t rows = size_t(tokens) * value_heads;
    return 6 * aligned(rows * 128 * sizeof(__nv_bfloat16)) +
           3 * aligned(rows * sizeof(float)) +
           aligned(rows * 64 * sizeof(float)) +
           aligned(rows * 64 * sizeof(__nv_bfloat16)) +
           aligned(size_t((tokens + 63) / 64) * value_heads * 128 * 128 * sizeof(float));
}

void kernel_fla_recurrent(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                          const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                          const float *log_decay, const float *beta,
                          float *state, cudaStream_t stream) {
    aot_recurrent16(stream, q, k, v, log_decay, nullptr, nullptr, beta,
                    nullptr, nullptr, out, state, state, nullptr,
                    1.0f / std::sqrt(128.0f), 1);
}

void kernel_fla_gdn(__nv_bfloat16 *out, const __nv_bfloat16 *qkv,
                    const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                    const __nv_bfloat16 *A_log, const __nv_bfloat16 *dt_bias,
                    float *state, void *workspace, int tokens,
                    int key_heads, int value_heads, cudaStream_t stream) {
    kernel_fla_workspace_size(tokens, value_heads);
    if (key_heads != 16)
        throw std::invalid_argument("FLA requires 16 key heads");
    size_t rows = size_t(tokens) * value_heads;
    Workspace ws{static_cast<char *>(workspace)};
    auto q = ws.take<__nv_bfloat16>(rows * 128);
    auto k = ws.take<__nv_bfloat16>(rows * 128);
    auto v = ws.take<__nv_bfloat16>(rows * 128);
    auto g = ws.take<float>(rows);
    auto beta = ws.take<float>(rows);
    auto cumulative_g = ws.take<float>(rows);
    auto matrix = ws.take<float>(rows * 64);
    auto inverse = ws.take<__nv_bfloat16>(rows * 64);
    auto w = ws.take<__nv_bfloat16>(rows * 128);
    auto u = ws.take<__nv_bfloat16>(rows * 128);
    auto v_new = ws.take<__nv_bfloat16>(rows * 128);
    auto chunk_state = ws.take<float>(size_t((tokens + 63) / 64) * value_heads * 128 * 128);
    float scale = 1.0f / std::sqrt(128.0f);

    aot_prepare(stream, qkv, b, a, dt_bias, A_log, q, k, v, g, beta,
                key_heads, value_heads, (2 * key_heads + value_heads) * 128, tokens);
    if (tokens == 1) {
        aot_recurrent48(stream, q, k, v, g, nullptr, nullptr, beta, nullptr,
                      nullptr, out, state, state, nullptr, scale, tokens);
        return;
    }
    aot_cumsum(stream, g, cumulative_g, tokens, value_heads);
    aot_kkt(stream, k, cumulative_g, beta, matrix, tokens, value_heads);
    // The triangular solver writes only diagonal and lower blocks.
    cudaError_t status = cudaMemsetAsync(inverse, 0, rows * 64 * sizeof(__nv_bfloat16), stream);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    aot_solve(stream, matrix, inverse, tokens, value_heads);
    aot_recompute(stream, k, v, beta, w, u, inverse, cumulative_g, tokens, value_heads);
    aot_state(stream, k, w, u, cumulative_g, state, chunk_state, v_new,
              state, tokens, value_heads);
    aot_output(stream, q, k, v_new, chunk_state, cumulative_g, out,
               tokens, value_heads, scale);
}
