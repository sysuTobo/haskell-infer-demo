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
constexpr int kInner = 8;         // multiple of 8: the activation kernel vectorizes
constexpr int kSharedInner = 8;   // shared expert width, also a multiple of 8

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
        // The kernel rounds the weights to bf16 before the combine, as the
        // reference implementation does.
        for (int k = 0; k < kTopK; ++k)
            weights[t * kTopK + k] = (double)test::value(test::bf16((float)weights[t * kTopK + k]));
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

/* Shared expert reference: a dense MLP on the normed input, scaled per token by
 * sigmoid(x @ w) when a gate is present, added to the routed result. */
std::vector<Bf16> cpu_shared_expert(const std::vector<Bf16> &normed,
                                    const std::vector<Bf16> &gate_w,
                                    const std::vector<Bf16> &up_w,
                                    const std::vector<Bf16> &down_w,
                                    const std::vector<Bf16> &scalar_w, bool use_gate) {
    std::vector<Bf16> out((size_t)kTokens * kHidden);
    for (int t = 0; t < kTokens; ++t) {
        double hidden[kSharedInner];
        for (int i = 0; i < kSharedInner; ++i) {
            double g = 0.0, u = 0.0;
            for (int j = 0; j < kHidden; ++j) {
                const double x = (double)test::value(normed[t * kHidden + j]);
                g += x * (double)test::value(gate_w[i * kHidden + j]);
                u += x * (double)test::value(up_w[i * kHidden + j]);
            }
            const double gb = (double)test::value(test::bf16((float)g));
            const double ub = (double)test::value(test::bf16((float)u));
            const double silu = gb / (1.0 + std::exp(-gb));
            hidden[i] = (double)test::value(test::bf16((float)(silu * ub)));
        }
        double scale = 1.0;
        if (use_gate) {
            double logit = 0.0;
            for (int j = 0; j < kHidden; ++j)
                logit += (double)test::value(normed[t * kHidden + j]) *
                         (double)test::value(scalar_w[j]);
            scale = 1.0 / (1.0 + std::exp(-logit));
        }
        for (int j = 0; j < kHidden; ++j) {
            double value = 0.0;
            for (int i = 0; i < kSharedInner; ++i)
                value += hidden[i] * (double)test::value(down_w[j * kSharedInner + i]);
            const double rounded = (double)test::value(test::bf16((float)value));
            out[(size_t)t * kHidden + j] = test::bf16((float)(scale * rounded));
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
    std::vector<Bf16> post_norm(kHidden);
    for (size_t i = 0; i < post_norm.size(); ++i) post_norm[i] = test::bf16(1.0f + test::sample(i, 17) * 0.25f);

    ModelDims dims{};
    dims.hidden_size = kHidden;
    dims.max_chunk = kTokens;
    dims.norm_style = 1;   /* plain RMSNorm, as the MoE families use */
    dims.rms_eps = 1e-6f;
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
    DeviceBuffer<Bf16> d_post_norm(post_norm.size());
    d_post_norm.upload(post_norm, stream.get());
    MoeWeights weights{d_post_norm.get(), d_router.get(), d_gate.get(), d_up.get(),
                       d_down.get()};
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

    /* The FFN applies the post-attention norm itself; mirror it here (BF16 out). */
    std::vector<Bf16> normed_input((size_t)kTokens * kHidden);
    for (int t = 0; t < kTokens; ++t) {
        double mean_square = 0.0;
        for (int j = 0; j < kHidden; ++j) {
            const double x = (double)test::value(input[t * kHidden + j]);
            mean_square += x * x;
        }
        mean_square /= kHidden;
        for (int j = 0; j < kHidden; ++j) {
            const double x = (double)test::value(input[t * kHidden + j]);
            const double scaled = x / std::sqrt(mean_square + 1e-6) *
                                  (double)test::value(post_norm[j]);
            normed_input[t * kHidden + j] = test::bf16((float)scaled);
        }
    }
    const auto ref_logits = cpu_router_logits(normed_input, router);
    std::vector<int> ids(kTokens * kTopK);
    std::vector<double> route_weights(kTokens * kTopK);
    cpu_router(ref_logits, /*norm_topk=*/true, 1.0f, ids, route_weights);
    const auto want = cpu_moe(normed_input, ids, route_weights, gate, up, down);
    ok &= test::compare("moe forward", d_out.download(stream.get()), want, 5e-2, 5e-2);

    /* --- 2b) Expert parallelism: two ranks holding half the experts each must
     *         sum to the full routed output (the engine all-reduces the routed
     *         partials across ranks; the reduction is done here on the host). */
    {
        const int half = kExperts / 2;
        std::vector<float> partial_sum((size_t)kTokens * kHidden, 0.0f);
        for (int rank = 0; rank < 2; ++rank) {
            const int offset = rank * half;
            std::vector<Bf16> local_gate((size_t)half * kInner * kHidden);
            std::vector<Bf16> local_up(local_gate.size());
            std::vector<Bf16> local_down((size_t)half * kHidden * kInner);
            for (int e = 0; e < half; ++e) {
                std::memcpy(&local_gate[(size_t)e * kInner * kHidden],
                            &gate[(size_t)(offset + e) * kInner * kHidden],
                            (size_t)kInner * kHidden * sizeof(Bf16));
                std::memcpy(&local_up[(size_t)e * kInner * kHidden],
                            &up[(size_t)(offset + e) * kInner * kHidden],
                            (size_t)kInner * kHidden * sizeof(Bf16));
                std::memcpy(&local_down[(size_t)e * kHidden * kInner],
                            &down[(size_t)(offset + e) * kHidden * kInner],
                            (size_t)kHidden * kInner * sizeof(Bf16));
            }
            DeviceBuffer<Bf16> d_local_gate(local_gate.size());
            DeviceBuffer<Bf16> d_local_up(local_up.size());
            DeviceBuffer<Bf16> d_local_down(local_down.size());
            d_local_gate.upload(local_gate, stream.get());
            d_local_up.upload(local_up, stream.get());
            d_local_down.upload(local_down, stream.get());

            MoeConfig local_moe = moe;
            local_moe.expert_offset = offset;
            local_moe.num_local_experts = half;
            MoeWeights local_weights{d_post_norm.get(), d_router.get(), d_local_gate.get(),
                                     d_local_up.get(), d_local_down.get()};
            const size_t local_bytes = moe_workspace_size(kTokens, &dims, &local_moe);
            void *local_scratch = nullptr;
            CUDA_CHECK(cudaMalloc(&local_scratch, local_bytes));
            MoeScratch local_workspace{local_scratch, local_bytes};
            DeviceBuffer<Bf16> d_partial(kTokens * kHidden);
            const int local_status = forward_moe_routed(cublas, stream.get(), d_input.get(),
                                                        d_partial.get(), &local_weights,
                                                        &local_moe, local_workspace, kTokens,
                                                        &dims);
            if (local_status != 0) {
                std::fprintf(stderr, "forward_moe_routed (rank %d) failed: %d\n",
                             rank, local_status);
                return EXIT_FAILURE;
            }
            const auto partial = d_partial.download(stream.get());
            for (size_t i = 0; i < partial_sum.size(); ++i)
                partial_sum[i] += (float)test::value(partial[i]);
            CUDA_CHECK(cudaFree(local_scratch));
        }
        ok &= test::compare("moe ep2 partial sum vs full forward", partial_sum,
                            d_out.download(stream.get()), 5e-2, 5e-2);
        ok &= test::compare("moe ep2 partial sum vs cpu reference", partial_sum, want,
                            5e-2, 5e-2);
    }

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
    const auto skewed_logits = cpu_router_logits(normed_input, skewed);
    cpu_router(skewed_logits, /*norm_topk=*/true, 1.0f, ids, route_weights);
    /* Expert 2 carries the only non-zero router row and the other five rows are
     * zero, so tokens collapse onto a handful of experts; the point of this case
     * is that some expert slices are empty and must be skipped. */
    std::vector<int> distinct(ids);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    std::printf("skewed routing uses %zu of %d experts\n", distinct.size(), kExperts);
    ok &= distinct.size() < (size_t)kExperts;  // at least one expert is skipped
    const auto skewed_want = cpu_moe(normed_input, ids, route_weights, gate, up, down);
    ok &= test::compare("moe forward (single expert)", d_out.download(stream.get()),
                        skewed_want, 5e-2, 5e-2);

    /* --- 4) Shared expert with its sigmoid gate, on the same normed input --- */
    std::vector<Bf16> shared_gate((size_t)kSharedInner * kHidden);
    std::vector<Bf16> shared_up((size_t)kSharedInner * kHidden);
    std::vector<Bf16> shared_down((size_t)kHidden * kSharedInner);
    std::vector<Bf16> gate_scalar(kHidden);
    for (size_t i = 0; i < shared_gate.size(); ++i)
        shared_gate[i] = test::bf16(test::sample(i, 41) * 0.5f);
    for (size_t i = 0; i < shared_up.size(); ++i)
        shared_up[i] = test::bf16(test::sample(i, 43) * 0.5f);
    for (size_t i = 0; i < shared_down.size(); ++i)
        shared_down[i] = test::bf16(test::sample(i, 47) * 0.5f);
    for (size_t i = 0; i < gate_scalar.size(); ++i)
        gate_scalar[i] = test::bf16(test::sample(i, 53));

    DeviceBuffer<Bf16> d_shared_gate(shared_gate.size()), d_shared_up(shared_up.size());
    DeviceBuffer<Bf16> d_shared_down(shared_down.size()), d_gate_scalar(gate_scalar.size());
    d_shared_gate.upload(shared_gate, stream.get());
    d_shared_up.upload(shared_up, stream.get());
    d_shared_down.upload(shared_down, stream.get());
    d_gate_scalar.upload(gate_scalar, stream.get());

    MoeConfig moe_shared = moe;
    moe_shared.num_shared_experts = 1;
    moe_shared.shared_intermediate_size = kSharedInner;
    moe_shared.shared_gate_scalar = 1;
    MoeWeights weights_shared = weights;
    weights_shared.shared_gate = d_shared_gate.get();
    weights_shared.shared_up = d_shared_up.get();
    weights_shared.shared_down = d_shared_down.get();
    weights_shared.shared_gate_scalar_w = d_gate_scalar.get();

    d_router.upload(router, stream.get());
    const size_t shared_scratch_bytes = moe_workspace_size(kTokens, &dims, &moe_shared);
    void *shared_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&shared_scratch, shared_scratch_bytes));
    MoeScratch shared_workspace{shared_scratch, shared_scratch_bytes};
    const int shared_status = forward_moe_ffn(cublas, stream.get(), d_input.get(), d_out.get(),
                                             &weights_shared, &moe_shared, shared_workspace,
                                             kTokens, &dims);
    if (shared_status != 0) {
        std::fprintf(stderr, "shared-expert forward failed: %d\n", shared_status);
        return EXIT_FAILURE;
    }
    const auto shared_logits = cpu_router_logits(normed_input, router);
    cpu_router(shared_logits, /*norm_topk=*/true, 1.0f, ids, route_weights);
    auto want_shared = cpu_moe(normed_input, ids, route_weights, gate, up, down);
    const auto extra = cpu_shared_expert(normed_input, shared_gate, shared_up, shared_down,
                                         gate_scalar, /*use_gate=*/true);
    for (int t = 0; t < kTokens; ++t) {
        for (int j = 0; j < kHidden; ++j) {
            const double routed = (double)test::value(want_shared[t * kHidden + j]);
            const double shared_value = (double)test::value(extra[t * kHidden + j]);
            want_shared[t * kHidden + j] = test::bf16((float)(routed + shared_value));
        }
    }
    ok &= test::compare("moe forward (shared expert + gate)",
                        d_out.download(stream.get()), want_shared, 5e-2, 5e-2);
    CUDA_CHECK(cudaFree(shared_scratch));

    cublasDestroy(cublas);
    CUDA_CHECK(cudaFree(scratch));
    return test::finish("test_moe", ok);
}
