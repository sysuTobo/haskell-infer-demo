/* MoE feed-forward: router and expert execution against an independent CPU
 * reference that mimics the device's arithmetic order (BF16 GEMM outputs, FP32
 * scoring and combine). */
#include "kernels.h"
#include "layers.h"
#include "moe.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>

using test::Bf16;
using test::DeviceBuffer;

namespace {

constexpr int kTokens = 4;
constexpr int kExperts = 6;
constexpr int kTopK = 2;
constexpr int kHidden = 8;
constexpr int kInner = 8;  // multiple of 8: the activation kernel vectorizes

/* Routing reference: softmax in double, deterministic lowest-index tie break,
 * optional renormalization and scaling -- written from the definition. */
void cpu_router(const std::vector<Bf16> &logits, bool norm_topk, float scaling,
                std::vector<int> &ids, std::vector<double> &weights) {
    for (int t = 0; t < kTokens; ++t) {
        std::vector<double> scores(kExperts);
        double max_score = -1e30, sum = 0.0;
        for (int e = 0; e < kExperts; ++e)
            max_score = std::max(max_score, (double)test::value(logits[t * kExperts + e]));
        for (int e = 0; e < kExperts; ++e) {
            scores[e] = std::exp((double)test::value(logits[t * kExperts + e]) - max_score);
            sum += scores[e];
        }
        for (int e = 0; e < kExperts; ++e) scores[e] /= sum;
        std::vector<int> chosen(kExperts, 0);
        for (int k = 0; k < kTopK; ++k) {
            int best = -1;
            double best_value = -1e30;
            for (int e = 0; e < kExperts; ++e) {
                if (chosen[e]) continue;
                if (scores[e] > best_value) {  // strict: ties keep the lower index
                    best_value = scores[e];
                    best = e;
                }
            }
            chosen[best] = 1;
            ids[t * kTopK + k] = best;
            weights[t * kTopK + k] = scores[best];
        }
        if (norm_topk) {
            double total = 0.0;
            for (int k = 0; k < kTopK; ++k) total += weights[t * kTopK + k];
            for (int k = 0; k < kTopK; ++k) weights[t * kTopK + k] /= total;
        }
        for (int k = 0; k < kTopK; ++k) weights[t * kTopK + k] *= scaling;
    }
}

/* Router logits the way the device produces them: BF16 weights and inputs, FP32
 * accumulation, BF16 output. */
std::vector<Bf16> cpu_router_logits(const std::vector<Bf16> &input,
                                    const std::vector<Bf16> &router) {
    std::vector<Bf16> logits((size_t)kTokens * kExperts);
    for (int t = 0; t < kTokens; ++t) {
        for (int e = 0; e < kExperts; ++e) {
            double acc = 0.0;
            for (int j = 0; j < kHidden; ++j)
                acc += (double)test::value(input[t * kHidden + j]) *
                       (double)test::value(router[e * kHidden + j]);
            logits[t * kExperts + e] = test::bf16((float)acc);
        }
    }
    return logits;
}

/* Expert MLP on each token's selected experts, then the weighted sum. */
std::vector<Bf16> cpu_moe(const std::vector<Bf16> &input, const std::vector<int> &ids,
                          const std::vector<double> &weights,
                          const std::vector<Bf16> &gate, const std::vector<Bf16> &up,
                          const std::vector<Bf16> &down) {
    std::vector<Bf16> out((size_t)kTokens * kHidden);
    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kHidden; ++h) {
            double acc = 0.0;
            for (int k = 0; k < kTopK; ++k) {
                const int e = ids[t * kTopK + k];
                double hidden[kInner];
                for (int i = 0; i < kInner; ++i) {
                    double g = 0.0, u = 0.0;
                    for (int j = 0; j < kHidden; ++j) {
                        const double x = (double)test::value(input[t * kHidden + j]);
                        g += x * (double)test::value(gate[((size_t)e * kInner + i) * kHidden + j]);
                        u += x * (double)test::value(up[((size_t)e * kInner + i) * kHidden + j]);
                    }
                    const double gb = (double)test::value(test::bf16((float)g));
                    const double ub = (double)test::value(test::bf16((float)u));
                    const double silu = gb / (1.0 + std::exp(-gb));
                    hidden[i] = (double)test::value(test::bf16((float)(silu * ub)));
                }
                double projected[kHidden];
                for (int j = 0; j < kHidden; ++j) {
                    double value = 0.0;
                    for (int i = 0; i < kInner; ++i)
                        value += hidden[i] *
                                 (double)test::value(down[((size_t)e * kHidden + j) * kInner + i]);
                    projected[j] = (double)test::value(test::bf16((float)value));
                }
                acc += weights[t * kTopK + k] * projected[h];
            }
            out[t * kHidden + h] = test::bf16((float)acc);
        }
    }
    return out;
}

}  // namespace

int main() {
    bool ok = true;
    test::Stream stream;

    /* --- 1) Router: scoring, top-k, renorm, deterministic tie break ----- */
    std::vector<Bf16> logits(kTokens * kExperts);
    for (int i = 0; i < kTokens * kExperts; ++i)
        logits[i] = test::bf16(test::sample(i, 11) * 2.0f);
    // Force a tie to pin the lowest-index rule.
    logits[1 * kExperts + 2] = test::bf16(1.5f);
    logits[1 * kExperts + 4] = test::bf16(1.5f);

    for (int norm_topk = 0; norm_topk <= 1; ++norm_topk) {
        DeviceBuffer<Bf16> d_logits(logits.size());
        DeviceBuffer<int> d_ids(kTokens * kTopK);
        DeviceBuffer<float> d_weights(kTokens * kTopK);
        d_logits.upload(logits, stream.get());
        kernel_moe_router_topk(d_logits.get(), kTokens, kExperts, kTopK, norm_topk,
                               0, 1.0f, d_ids.get(), d_weights.get(), stream.get());
        CUDA_CHECK(cudaGetLastError());
        const auto ids = d_ids.download(stream.get());
        const auto weights = d_weights.download(stream.get());

        std::vector<int> want_ids(kTokens * kTopK);
        std::vector<double> want_weights(kTokens * kTopK);
        cpu_router(logits, norm_topk != 0, 1.0f, want_ids, want_weights);
        ok &= test::compare(norm_topk ? "router ids (renorm)" : "router ids", ids, want_ids,
                            0.0, 0.0);
        ok &= test::compare(norm_topk ? "router weights (renorm)" : "router weights",
                            weights, want_weights, 1e-6, 1e-6);
    }

    /* --- 2) Full MoE forward vs the CPU reference ----------------------- */
    std::vector<Bf16> router(kExperts * kHidden), gate(kExperts * kInner * kHidden);
    std::vector<Bf16> up(kExperts * kInner * kHidden), down(kExperts * kHidden * kInner);
    for (size_t i = 0; i < router.size(); ++i) router[i] = test::bf16(test::sample(i, 3));
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = test::bf16(test::sample(i, 5) * 0.5f);
    for (size_t i = 0; i < up.size(); ++i) up[i] = test::bf16(test::sample(i, 7) * 0.5f);
    for (size_t i = 0; i < down.size(); ++i) down[i] = test::bf16(test::sample(i, 9) * 0.5f);
    std::vector<Bf16> input(kTokens * kHidden);
    for (size_t i = 0; i < input.size(); ++i) input[i] = test::bf16(test::sample(i, 13));

    ModelDims dims{};
    dims.hidden_size = kHidden;
    dims.max_chunk = kTokens;
    MoeConfig moe{kExperts, kTopK, kInner, /*norm_topk_prob=*/1, /*sigmoid=*/0, 1.0f};

    DeviceBuffer<Bf16> d_input(input.size()), d_out(kTokens * kHidden);
    DeviceBuffer<Bf16> d_router(router.size()), d_gate(gate.size());
    DeviceBuffer<Bf16> d_up(up.size()), d_down(down.size());
    d_input.upload(input, stream.get());
    d_router.upload(router, stream.get());
    d_gate.upload(gate, stream.get());
    d_up.upload(up, stream.get());
    d_down.upload(down, stream.get());

    const size_t scratch_bytes = moe_workspace_size(kTokens, &dims, &moe);
    void *scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    MoeWeights weights{d_router.get(), d_gate.get(), d_up.get(), d_down.get()};
    MoeScratch workspace{scratch, scratch_bytes};
    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(cublas, stream.get()) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS handle setup failed\n");
        return EXIT_FAILURE;
    }

    const int status = forward_moe_ffn(cublas, stream.get(), d_input.get(), d_out.get(),
                                       &weights, &moe, workspace, kTokens, &dims);
    if (status != 0) {
        std::fprintf(stderr, "forward_moe_ffn failed: %d\n", status);
        return EXIT_FAILURE;
    }

    const auto ref_logits = cpu_router_logits(input, router);
    std::vector<int> ids(kTokens * kTopK);
    std::vector<double> route_weights(kTokens * kTopK);
    cpu_router(ref_logits, /*norm_topk=*/true, 1.0f, ids, route_weights);
    const auto want = cpu_moe(input, ids, route_weights, gate, up, down);
    ok &= test::compare("moe forward", d_out.download(stream.get()), want, 5e-2, 5e-2);

    /* --- 3) Concentrated routing: one expert takes every token, the rest get
     *        none, so the expert loop must skip empty slices correctly. */
    std::vector<Bf16> skewed = router;
    for (int e = 0; e < kExperts; ++e)
        for (int j = 0; j < kHidden; ++j)
            skewed[e * kHidden + j] = test::bf16(e == 2 ? 1.0f : 0.0f);
    d_router.upload(skewed, stream.get());
    const int skewed_status = forward_moe_ffn(cublas, stream.get(), d_input.get(), d_out.get(),
                                              &weights, &moe, workspace, kTokens, &dims);
    if (skewed_status != 0) {
        std::fprintf(stderr, "skewed forward_moe_ffn failed: %d\n", skewed_status);
        return EXIT_FAILURE;
    }
    const auto skewed_logits = cpu_router_logits(input, skewed);
    cpu_router(skewed_logits, /*norm_topk=*/true, 1.0f, ids, route_weights);
    /* Expert 2 carries the only non-zero router row and the other five rows are
     * zero, so tokens collapse onto a handful of experts; the point of this case
     * is that some expert slices are empty and must be skipped. */
    std::vector<int> distinct(ids);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    std::printf("skewed routing uses %zu of %d experts\n", distinct.size(), kExperts);
    ok &= distinct.size() < (size_t)kExperts;  // at least one expert is skipped
    const auto skewed_want = cpu_moe(input, ids, route_weights, gate, up, down);
    ok &= test::compare("moe forward (single expert)", d_out.download(stream.get()),
                        skewed_want, 5e-2, 5e-2);

    cublasDestroy(cublas);
    CUDA_CHECK(cudaFree(scratch));
    return test::finish("test_moe", ok);
}
