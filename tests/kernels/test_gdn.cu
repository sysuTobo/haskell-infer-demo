// CPU-reference regressions for the existing decode, convolution and norm APIs.
#include "kernels.h"
#include "fla_ops.h"
#include "test_utils.h"

namespace {
using namespace test;

constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kKeyDim = 128;
constexpr int kValueDim = 128;

void normalized_qk(int step, std::vector<Bf16> &q, std::vector<Bf16> &k) {
    // Normalize on the CPU BEFORE calling the API, then round to BF16. The
    // reference below consumes those exact rounded values without renormalizing.
    for (int h = 0; h < kKeyHeads; ++h) {
        std::vector<double> qr(kKeyDim), kr(kKeyDim);
        double q2 = 0.0, k2 = 0.0;
        for (int j = 0; j < kKeyDim; ++j) {
            const int i = h * kKeyDim + j;
            kr[j] = value(bf16(0.3f + 0.8f * sample(i, 70 + step)));
            // Correlated (not identical) Q/K makes the missing 1/sqrt(K)
            // observable even with zero initial state, for every value head.
            qr[j] = value(bf16(static_cast<float>(kr[j]) + 0.3f * sample(i, 80 + step)));
            q2 += qr[j] * qr[j];
            k2 += kr[j] * kr[j];
        }
        for (int j = 0; j < kKeyDim; ++j) {
            q[h * kKeyDim + j] = bf16(static_cast<float>(qr[j] / std::sqrt(q2)));
            k[h * kKeyDim + j] = bf16(static_cast<float>(kr[j] / std::sqrt(k2)));
        }
    }
}

std::vector<float> delta_reference(const std::vector<Bf16> &q,
                                    const std::vector<Bf16> &k,
                                    const std::vector<Bf16> &v,
                                    const std::vector<float> &alpha,
                                    const std::vector<float> &beta,
                                    std::vector<float> &state) {
    // Mathematical contract (the older formula in kernels.h is not the oracle):
    //   D = alpha * S; delta = v - k @ D;
    //   S_new = D + outer(k, beta * delta); out = (q / sqrt(K)) @ S_new.
    // Double dot products, with FP32 persistent state and BF16 output boundaries.
    std::vector<float> out(kValueHeads * kValueDim);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kKeyDim));
    for (int h = 0; h < kValueHeads; ++h) {
        const int kh = h / (kValueHeads / kKeyHeads);
        const size_t base = static_cast<size_t>(h) * kKeyDim * kValueDim;
        std::vector<double> decayed(kKeyDim * kValueDim);
        for (size_t i = 0; i < decayed.size(); ++i)
            decayed[i] = static_cast<double>(alpha[h]) * state[base + i];
        for (int c = 0; c < kValueDim; ++c) {
            double prediction = 0.0;
            for (int j = 0; j < kKeyDim; ++j)
                prediction += value(k[kh * kKeyDim + j]) * decayed[j * kValueDim + c];
            const double delta = value(v[h * kValueDim + c]) - prediction;
            for (int j = 0; j < kKeyDim; ++j) {
                const size_t i = j * kValueDim + c;
                state[base + i] = static_cast<float>(decayed[i] +
                    value(k[kh * kKeyDim + j]) * static_cast<double>(beta[h]) * delta);
            }
            double sum = 0.0;
            for (int j = 0; j < kKeyDim; ++j)
                sum += (value(q[kh * kKeyDim + j]) * scale) * state[base + j * kValueDim + c];
            out[h * kValueDim + c] = round_bf16(static_cast<float>(sum));
        }
    }
    return out;
}

bool test_delta_decode(bool nonzero_initial_state, cudaStream_t stream) {
    std::vector<float> state(kValueHeads * kKeyDim * kValueDim, 0.0f);
    if (nonzero_initial_state) {
        for (size_t i = 0; i < state.size(); ++i)
            state[i] = 0.04f + 0.15f * sample(i, 90);
    }
    DeviceBuffer<float> ds(state.size()), da(kValueHeads), db(kValueHeads);
    DeviceBuffer<Bf16> dq(kKeyHeads * kKeyDim), dk(kKeyHeads * kKeyDim);
    DeviceBuffer<Bf16> dv(kValueHeads * kValueDim), out(kValueHeads * kValueDim);
    ds.upload(state, stream);
    bool ok = true;
    for (int step = 0; step < 5; ++step) {
        std::vector<Bf16> q(kKeyHeads * kKeyDim), k(q.size());
        std::vector<Bf16> v(kValueHeads * kValueDim);
        std::vector<float> alpha(kValueHeads), beta(kValueHeads);
        normalized_qk(step, q, k);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = bf16(0.2f + sample(i, 100 + step));
        // Decay probabilities, NOT log-decays. Exercise ordinary gates plus
        // alpha=0/1 and beta=0/1 to isolate reset, retention and update behavior.
        const float decays[] = {0.0f, 0.125f, 0.5f, 0.875f, 1.0f};
        const float updates[] = {0.0f, 0.25f, 0.5f, 0.875f, 1.0f};
        for (int h = 0; h < kValueHeads; ++h) {
            alpha[h] = decays[(h + step) % 5];
            beta[h] = updates[(h * 3 + step + 1) % 5];
        }
        const auto expected = delta_reference(q, k, v, alpha, beta, state);
        dq.upload(q, stream);
        dk.upload(k, stream);
        dv.upload(v, stream);
        std::vector<float> log_decay(alpha.size());
        for (size_t h = 0; h < alpha.size(); ++h) log_decay[h] = std::log(alpha[h]);
        da.upload(log_decay, stream);
        db.upload(beta, stream);
        out.upload(poison(v.size()), stream);
        kernel_fla_recurrent(out.get(), dq.get(), dk.get(), dv.get(),
                              da.get(), db.get(), ds.get(), stream);
        CUDA_CHECK(cudaGetLastError());
        const std::string name = std::string("delta/") +
            (nonzero_initial_state ? "nonzero" : "zero") + "/step=" + std::to_string(step);
        // State uses FP32 tolerances, not BF16 output tolerances. Never reseed
        // device state from the oracle: every step tests the recurrent trajectory.
        ok &= compare(name + "/state[h,k,v]", ds.download(stream), state, 2e-6, 3e-5);
        ok &= compare(name + "/out[h,v]", out.download(stream), expected, 2e-5, 8e-3);
    }
    return ok;
}

std::vector<float> conv_reference(const std::vector<Bf16> &x,
                                  const std::vector<Bf16> &weight,
                                  const std::vector<Bf16> &bias,
                                  std::vector<Bf16> &state, int channels, int tokens) {
    constexpr int width = 4;
    std::vector<float> out(tokens * channels);
    // Build chronological histories, independent of the device's shift-register
    // implementation. No SiLU here: activation is a separate layer operation.
    for (int c = 0; c < channels; ++c) {
        std::vector<double> history(width - 1 + tokens);
        for (int j = 0; j < width - 1; ++j) history[j] = value(state[c * (width - 1) + j]);
        for (int t = 0; t < tokens; ++t) history[width - 1 + t] = value(x[t * channels + c]);
        for (int t = 0; t < tokens; ++t) {
            double sum = value(bias[c]);
            for (int j = 0; j < width; ++j)
                sum += value(weight[c * width + j]) * history[t + j];
            out[t * channels + c] = round_bf16(static_cast<float>(sum));
        }
        for (int j = 0; j < width - 1; ++j)
            state[c * (width - 1) + j] = bf16(static_cast<float>(history[tokens + j]));
    }
    return out;
}

bool test_conv(int channels, bool nonzero_initial_state, cudaStream_t stream) {
    std::vector<Bf16> weights(channels * 4), bias(channels, bf16(0.0f));
    std::vector<Bf16> state(channels * 3, bf16(0.0f));
    // Dyadic inputs/weights make these four-term sums exact in FP32 as well
    // as double, so exact BF16 output comparisons do not depend on FMA order.
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = bf16(static_cast<int>(sample(i, 110) * 16) / 32.0f);
    if (nonzero_initial_state) {
        for (size_t i = 0; i < state.size(); ++i)
            state[i] = bf16(static_cast<int>(sample(i, 111) * 32) / 16.0f);
        for (int i = 0; i < channels; ++i)
            bias[i] = bf16(static_cast<int>(sample(i, 112) * 16) / 32.0f);
    }
    DeviceBuffer<Bf16> dw(weights.size()), db(bias.size()), ds(state.size());
    dw.upload(weights, stream);
    db.upload(bias, stream);  // Explicit zero bias also covers the no-bias math.
    ds.upload(state, stream);
    bool ok = true;
    const int chunks[] = {1, 3, 2, 1, 5};
    for (int step = 0; step < 5; ++step) {
        const int tokens = chunks[step];
        std::vector<Bf16> x(tokens * channels);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = bf16(static_cast<int>(sample(i, 120 + step) * 32) / 16.0f);
        const auto expected = conv_reference(x, weights, bias, state, channels, tokens);
        DeviceBuffer<Bf16> dx(x.size()), out(x.size());
        dx.upload(x, stream);
        out.upload(poison(x.size()), stream);
        kernel_causal_conv1d(out.get(), dx.get(), dw.get(), db.get(), ds.get(),
                             channels, tokens, 4, stream);
        CUDA_CHECK(cudaGetLastError());
        const std::string name = "conv/channels=" + std::to_string(channels) +
            (nonzero_initial_state ? "/nonzero" : "/zero") + "/step=" + std::to_string(step);
        ok &= compare(name + "/out_without_activation", out.download(stream), expected, 0.0, 0.0);
        ok &= compare(name + "/state[channel,history]", ds.download(stream), state, 0.0, 0.0);
    }
    return ok;
}

std::vector<float> gated_norm_reference(const std::vector<Bf16> &x,
                                        const std::vector<Bf16> &z,
                                        const std::vector<float> &weight,
                                        int rows, int dim, float eps) {
    std::vector<float> out(x.size());
    for (int row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const float f = static_cast<float>(value(x[row * dim + d]));
            sum_sq += f * f;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / dim + eps);
        for (int d = 0; d < dim; ++d) {
            const int i = row * dim + d;
            const float f = static_cast<float>(value(x[i]));
            const float gate = static_cast<float>(value(z[i]));
            // Transformers dtype boundaries: norm FP32 -> BF16, BF16 weight
            // product -> BF16, then multiply FP32 SiLU(z) -> BF16. The API's
            // float weight contains the effective BF16 checkpoint weight;
            // do NOT add 1 or fuse away either intermediate rounding.
            const float normalized = round_bf16(f * inv_rms);
            const float weighted = round_bf16(normalized * round_bf16(weight[d]));
            const float silu = gate / (1.0f + std::exp(-gate));
            out[i] = round_bf16(weighted * silu);
        }
    }
    return out;
}

bool test_gated_norm(cudaStream_t stream) {
    constexpr int dim = 128;
    constexpr int rows = 2 * kValueHeads;  // Per-head normalization, two tokens.
    constexpr float eps = 1e-6f;
    std::vector<Bf16> x(rows * dim), z(x.size());
    std::vector<float> weight(dim);
    for (int d = 0; d < dim; ++d) weight[d] = round_bf16(0.625f + (d % 31) / 32.0f);
    const float gates[] = {-4.0f, -2.0f, -1.0f, -0.25f, 0.0f, 0.375f, 1.0f, 2.0f, 4.0f};
    for (int row = 0; row < rows; ++row) {
        for (int d = 0; d < dim; ++d) {
            const int i = row * dim + d;
            // Exact FP32 squared sums remove reduction-order ambiguity. Include
            // an all-zero head and a small-magnitude head where epsilon matters.
            float f = static_cast<int>(sample(i, 130) * 96) / 64.0f;
            if (row == 0) f = 0.0f;
            if (row == 1) f /= 512.0f;
            x[i] = bf16(f);
            z[i] = bf16(gates[(d + 2 * row) % 9]);
        }
    }
    const auto expected = gated_norm_reference(x, z, weight, rows, dim, eps);
    DeviceBuffer<Bf16> dx(x.size()), dz(z.size()), out(x.size());
    DeviceBuffer<float> dw(weight.size());
    dx.upload(x, stream);
    dz.upload(z, stream);
    dw.upload(weight, stream);
    out.upload(poison(x.size()), stream);
    kernel_gdn_gated_norm(out.get(), dx.get(), dz.get(), dw.get(), dim, rows, eps, stream);
    CUDA_CHECK(cudaGetLastError());
    // Much tighter than one BF16 ULP: a usual 1% tolerance would hide precisely
    // the missing intermediate BF16 casts this regression is meant to catch.
    return compare("gated_norm/transformers_bf16_boundaries", out.download(stream),
                   expected, 1e-6, 1e-5);
}

}  // namespace

int main() {
    Stream stream;
    bool ok = test_delta_decode(false, stream.get());
    ok &= test_delta_decode(true, stream.get());
    ok &= test_conv(259, false, stream.get());  // Partial last block.
    ok &= test_conv(10240, true, stream.get());
    ok &= test_gated_norm(stream.get());
    return finish("test_gdn", ok);
}
