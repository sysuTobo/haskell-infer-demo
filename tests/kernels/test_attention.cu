// Numerical regressions for the public attention APIs and their layer wiring.
// No weights/model download or device-side reference implementation is required.
#include "kernels.h"
#include "layers.h"
#include "test_utils.h"

namespace {
using namespace test;

constexpr int kHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kQueryDim = kHeads * kHeadDim;

bool test_kv_write(cudaStream_t stream) {
    const int max_seq = 19;
    std::vector<Bf16> expected(2 * max_seq * kKvDim);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = bf16(-3.0f + sample(i, 10));
    DeviceBuffer<Bf16> cache(expected.size());
    cache.upload(expected, stream);

    bool ok = true;
    // Nonzero offsets, multiple tokens, then a decode append. Neither logical
    // sequence length equals the physical K/V plane stride, max_seq.
    const int starts[] = {3, 7};
    const int counts[] = {4, 1};
    for (int step = 0; step < 2; ++step) {
        const int start = starts[step], tokens = counts[step];
        std::vector<Bf16> k(tokens * kKvDim), v(k.size());
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = bf16(sample(i, 20 + step));
            v[i] = bf16(2.0f + sample(i, 30 + step));
        }
        DeviceBuffer<Bf16> dk(k.size()), dv(v.size());
        dk.upload(k, stream);
        dv.upload(v, stream);
        kernel_kv_cache_write(cache.get(), dk.get(), dv.get(), start, tokens,
                              kKvHeads, kHeadDim, max_seq, stream);
        CUDA_CHECK(cudaGetLastError());
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < kKvDim; ++d) {
                expected[(start + t) * kKvDim + d] = k[t * kKvDim + d];
                expected[(max_seq + start + t) * kKvDim + d] = v[t * kKvDim + d];
            }
        }
        // Exact copies: check both planes AND every untouched prefix/suffix.
        ok &= compare("kv_write/step=" + std::to_string(step),
                      cache.download(stream), expected, 0.0, 0.0);
    }
    return ok;
}

std::vector<double> attention_reference(const std::vector<Bf16> &q,
                                        const std::vector<Bf16> &cache,
                                        int start, int tokens, int max_seq) {
    std::vector<double> out(tokens * kQueryDim);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (int t = 0; t < tokens; ++t) {
        const int visible = start + t + 1;  // Bottom-right causal mask.
        for (int h = 0; h < kHeads; ++h) {
            const int kh = h / (kHeads / kKvHeads);
            std::vector<double> scores(visible);
            for (int p = 0; p < visible; ++p) {
                double dot = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += value(q[t * kQueryDim + h * kHeadDim + d]) *
                           value(cache[p * kKvDim + kh * kHeadDim + d]);
                }
                scores[p] = dot * scale;
            }
            const double largest = *std::max_element(scores.begin(), scores.end());
            double denominator = 0.0;
            for (double &score : scores) {
                score = std::exp(score - largest);
                denominator += score;
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double sum = 0.0;
                for (int p = 0; p < visible; ++p) {
                    sum += (scores[p] / denominator) *
                           value(cache[(max_seq + p) * kKvDim + kh * kHeadDim + d]);
                }
                out[t * kQueryDim + h * kHeadDim + d] = sum;
            }
        }
    }
    return out;
}

bool test_attention(int start, int tokens, int max_seq, cudaStream_t stream) {
    const int seq_len = start + tokens;
    std::vector<Bf16> q(tokens * kQueryDim);
    // A finite, distinct sentinel in unused cache positions catches a wrong V
    // plane stride without relying on undefined/uninitialized device memory.
    std::vector<Bf16> cache(2 * max_seq * kKvDim, bf16(31.0f));
    for (size_t i = 0; i < q.size(); ++i) q[i] = bf16(sample(i, 41));
    for (int p = 0; p < seq_len; ++p) {
        for (int h = 0; h < kKvHeads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                const int i = p * kKvDim + h * kHeadDim + d;
                cache[i] = bf16(sample(i, 42));
                cache[max_seq * kKvDim + i] =
                    bf16(2.0f * h + 0.5f * (p % 5) + sample(i, 43));
            }
        }
    }
    // Construct the cache independently, not via the GPU writer under test.
    const auto expected = attention_reference(q, cache, start, tokens, max_seq);
    DeviceBuffer<Bf16> dq(q.size()), dcache(cache.size()), out(q.size());
    dq.upload(q, stream);
    dcache.upload(cache, stream);
    out.upload(poison(q.size()), stream);
    kernel_attention(out.get(), dq.get(), dcache.get(), start, tokens, seq_len,
                     kHeads, kKvHeads, kHeadDim, 1.0f / 16.0f, max_seq, stream);
    CUDA_CHECK(cudaGetLastError());
    const std::string name = "attention/start=" + std::to_string(start) +
                             "/tokens=" + std::to_string(tokens) +
                             "/capacity=" + std::to_string(max_seq);
    // Tensor Core attention also rounds probabilities to BF16 before the PV product.
    bool ok = compare(name + "/out", out.download(stream), expected, 3e-3, 8e-3);
    ok &= compare(name + "/cache_unchanged", dcache.download(stream), cache, 0.0, 0.0);
    return ok;
}

void check_cublas(cublasStatus_t status, const char *call) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLAS error %d\n", call, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

struct BlasHandle {
    cublasHandle_t handle = nullptr;
    BlasHandle() { check_cublas(cublasCreate(&handle), "cublasCreate"); }
    ~BlasHandle() { check_cublas(cublasDestroy(handle), "cublasDestroy"); }
    BlasHandle(const BlasHandle &) = delete;
    BlasHandle &operator=(const BlasHandle &) = delete;
};

bool test_attention_layer_all_heads(cudaStream_t stream) {
    // Regression: a <<<1,256>>> launch without a grid-stride loop writes only
    // head 0 of 6144 gated elements. Poison all scratch so stale data cannot pass.
    constexpr int hidden = 64;
    constexpr int max_seq = 7;
    constexpr float eps = 1e-6f;
    ModelDims dims{};
    dims.hidden_size = hidden;
    dims.num_heads = kHeads;
    dims.num_kv_heads = kKvHeads;
    dims.head_dim = kHeadDim;
    dims.rotary_dim = 64;
    dims.norm_style = 0;          /* Gemma: the Qwen3.8 family norm */
    dims.attn_output_gate = 1;    /* q_proj carries the fused output gate */
    dims.q_gate_interleave = 1;
    dims.rope_theta = 1e7f;
    dims.rms_eps = eps;
    dims.max_seq_len = max_seq;

    std::vector<Bf16> x(hidden);
    for (int i = 0; i < hidden; ++i) x[i] = bf16((i % 13 - 6) / 8.0f);
    x[0] = bf16(0.75f);
    std::vector<Bf16> norm_weight(hidden, bf16(0.0f)), head_weight(kHeadDim, bf16(0.0f));
    std::vector<Bf16> qw(2 * kQueryDim * hidden, bf16(0.0f));
    std::vector<Bf16> kw(kKvDim * hidden, bf16(0.0f)), vw(kw.size(), bf16(0.0f));
    std::vector<Bf16> ow(hidden * kQueryDim, bf16(0.0f));
    for (int row = 0; row < kKvDim; ++row) vw[row * hidden] = bf16(1.0f);
    for (int row = 0; row < hidden; ++row) {
        // Cover all 24 heads, not just the first block's 256 channels.
        const int column = (row % kHeads) * kHeadDim + (row * 37 + 13) % kHeadDim;
        ow[row * kQueryDim + column] = bf16(0.5f + 0.25f * (row % 4));
    }

    // Q=K=gate=0, one visible token: attention=V=normalized_x[0], gate=1/2.
    // Model the BF16 boundaries between norm, GEMMs, attention and gating.
    float sum_sq = 0.0f;
    for (Bf16 v : x) {
        const float f = static_cast<float>(value(v));
        sum_sq += f * f;
    }
    const float normalized0 = round_bf16(static_cast<float>(value(x[0])) /
                                        std::sqrt(sum_sq / hidden + eps));
    const float gated = round_bf16(normalized0 * 0.5f);
    std::vector<float> expected(hidden);
    for (int row = 0; row < hidden; ++row)
        expected[row] = round_bf16(gated * (0.5f + 0.25f * (row % 4)));
    std::vector<Bf16> cache(2 * max_seq * kKvDim, bf16(-7.0f));
    auto expected_cache = cache;
    for (int i = 0; i < kKvDim; ++i) {
        expected_cache[i] = bf16(0.0f);
        expected_cache[max_seq * kKvDim + i] = bf16(normalized0);
    }

    DeviceBuffer<Bf16> dx(x.size()), dqw(qw.size()), dkw(kw.size()), dvw(vw.size());
    DeviceBuffer<Bf16> dow(ow.size()), dcache(cache.size()), out(hidden);
    DeviceBuffer<Bf16> dnorm(norm_weight.size()), dhead(head_weight.size());
    DeviceBuffer<int64_t> positions(1);
    positions.upload({0}, stream);
    // Current caller workspace: normed + raw Q/gate + K/V + Q + gate + attn + gated.
    DeviceBuffer<Bf16> ws(hidden + 6 * kQueryDim + 2 * kKvDim);
    dx.upload(x, stream);
    dqw.upload(qw, stream);
    dkw.upload(kw, stream);
    dvw.upload(vw, stream);
    dow.upload(ow, stream);
    dnorm.upload(norm_weight, stream);
    dhead.upload(head_weight, stream);
    dcache.upload(cache, stream);
    ws.upload(poison(hidden + 6 * kQueryDim + 2 * kKvDim), stream);
    out.upload(poison(hidden), stream);

    AttentionWeights weights{};
    weights.q_proj_w = dqw.get();
    weights.k_proj_w = dkw.get();
    weights.v_proj_w = dvw.get();
    weights.o_proj_w = dow.get();
    weights.input_norm_w = dnorm.get();
    weights.q_norm_w = dhead.get();
    weights.k_norm_w = dhead.get();
    BlasHandle blas;
    check_cublas(cublasSetStream(blas.handle, stream), "cublasSetStream");
    const int status = forward_attention_layer(
        blas.handle, stream, dx.get(), ws.get(), out.get(), &weights, dcache.get(),
        positions.get(), 1, 1, &dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bool ok = status == 0;
    if (!ok) std::fprintf(stderr, "forward_attention_layer returned %d\n", status);
    ok &= compare("attention_layer/all_64_outputs", out.download(stream), expected, 1e-5, 4e-3);
    ok &= compare("attention_layer/kv_state", dcache.download(stream), expected_cache, 0.0, 0.0);
    ok &= compare("attention_layer/residual_unchanged", dx.download(stream), x, 0.0, 0.0);
    return ok;
}

}  // namespace

int main() {
    // Use the selected visible GPU; never hard-code a physical device index.
    Stream stream;
    bool ok = test_kv_write(stream.get());
    ok &= test_attention(0, 1, 9, nullptr);  // Default stream baseline.
    ok &= test_attention(0, 4, 13, stream.get());
    ok &= test_attention(3, 5, 19, stream.get());
    ok &= test_attention(31, 3, 47, stream.get());
    ok &= test_attention(255, 3, 269, stream.get());  // Score loop crosses block size.
    ok &= test_attention_layer_all_heads(stream.get());
    return finish("test_attention", ok);
}
