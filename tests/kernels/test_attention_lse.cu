/*
 * test_attention_lse.cu - Stage 2 claim E, forward half.
 *
 * "Determine a usable paired implementation or an explicitly compatible
 * saved-state/recomputation adapter, including LSE, masks, GQA and determinism.
 * Current forward passes lse=nullptr; a backward from another library cannot just
 * be plugged in without a compatibility/gradient check."
 *
 * This file establishes the forward half of that pair, without touching the
 * engine's arithmetic: it calls the same FlashInfer dispatcher the engine calls
 * (csrc/kernels/attention.cu), once with a null LSE like the engine and once with
 * a real LSE buffer, and shows
 *
 *   1. the attention output is bitwise unchanged when the LSE is requested, so
 *      producing it costs nothing numerically;
 *   2. what the kernel actually writes: layout and convention. The source writes
 *      lse[qo_idx * num_qo_heads + qo_head_idx] = ptx_log2(d) + m over scores that
 *      already carry a log2(e) factor, so the returned value is log2 of the
 *      *natural* log-sum-exp -- log2(sum_j e^{s_j}), not ln(sum_j e^{s_j}) and not
 *      log2(sum_j 2^{s_j}). The three readings differ by a scale, so this is
 *      checked against all three double-precision references rather than taken
 *      from the source; a backward that read the wrong one would be silently off;
 *   3. how much extra memory it costs, which is the input to the resource
 *      estimate in docs/worklog.md.
 */
#include "invariance_utils.h"
#include "kernels.h"

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace invariance;
using test::Bf16;
using test::bf16;
using test::DeviceBuffer;
using test::poison;
using test::sample;
using test::value;

constexpr int kHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kTokens = 96;  // long enough for several query tiles
constexpr int kRows = kHeads * kHeadDim;
constexpr int kKvRows = kKvHeads * kHeadDim;

using Params = flashinfer::SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;

/* Exactly the dispatcher instantiation kernel_attention uses, with the LSE
 * argument under this test's control. */
template <int DIM>
cudaError_t launch(const Params &params, cudaStream_t stream) {
    // The same instantiation csrc/kernels/attention.cu uses: partial-RoPE off,
    // fp16 QK reduction off, the causal mask, and the default attention variant.
    using Attention = flashinfer::DefaultAttention<false, false, false, false>;
    return flashinfer::SinglePrefillWithKVCacheDispatched<
        DIM, DIM, flashinfer::PosEncodingMode::kNone, false, flashinfer::MaskMode::kCausal,
        Attention>(params, /*tmp=*/nullptr, stream);
}

Params make_params(const std::vector<Bf16> &q, const std::vector<Bf16> &cache, Bf16 *out,
                   float *lse) {
    const int kv_stride = kKvRows;
    return Params(const_cast<__nv_bfloat16 *>(q.data()), const_cast<__nv_bfloat16 *>(cache.data()),
                  const_cast<__nv_bfloat16 *>(cache.data() + (size_t)kTokens * kv_stride),
                  /*maybe_custom_mask=*/nullptr, out, lse, /*maybe_alibi_slopes=*/nullptr, kHeads,
                  kKvHeads, kTokens, kTokens, kRows, kHeadDim, kv_stride, kHeadDim, kHeadDim,
                  /*window_left=*/-1, /*logits_soft_cap=*/0.0f,
                  1.0f / std::sqrt((float)kHeadDim), /*rope_scale=*/1.0f, /*rope_theta=*/1.0f);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::printf("invariance: E: stream creation failed\n");
        return EXIT_FAILURE;
    }

    // Fixed Q/K/V; the cache is one [2, kTokens, kv_heads, head_dim] plane.
    std::vector<Bf16> q((size_t)kTokens * kRows);
    std::vector<Bf16> cache((size_t)2 * kTokens * kKvRows);
    for (size_t i = 0; i < q.size(); ++i) q[i] = bf16(sample(i, 750));
    for (size_t i = 0; i < cache.size(); ++i) cache[i] = bf16(sample(i, 751) * 0.5f);

    DeviceBuffer<Bf16> dq(q.size()), dcache(cache.size());
    DeviceBuffer<Bf16> out_no_lse((size_t)kTokens * kRows), out_lse((size_t)kTokens * kRows);
    DeviceBuffer<float> lse((size_t)kTokens * kHeads);
    dq.upload(q, stream);
    dcache.upload(cache, stream);
    out_no_lse.upload(poison((size_t)kTokens * kRows), stream);
    out_lse.upload(poison((size_t)kTokens * kRows), stream);
    std::vector<float> lse_sentinel((size_t)kTokens * kHeads, -12345.0f);
    lse.upload(lse_sentinel, stream);

    // 1. The engine's call: no LSE.
    cudaError_t status = launch<kHeadDim>(make_params(q, cache, out_no_lse.get(), nullptr), stream);
    if (status != cudaSuccess) {
        std::printf("invariance: E: lse=null launch failed: %s\n", cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();

    // 2. The paired call: same everything, plus an LSE buffer.
    status = launch<kHeadDim>(make_params(q, cache, out_lse.get(), lse.get()), stream);
    if (status != cudaSuccess) {
        std::printf("invariance: E: lse launch failed: %s\n", cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();

    Recorder rec("E");
    const Delta out = delta(out_no_lse.download(stream), out_lse.download(stream));
    rec.record("output lse=null vs lse=buffer", out, "bitwise");
    std::printf("invariance: E requesting the LSE %s the attention output\n",
                out.bitwise() ? "leaves bitwise unchanged" : "CHANGES");
    if (!out.bitwise()) {
        std::printf("invariance: E FAIL: the LSE request perturbs the forward\n");
        return EXIT_FAILURE;
    }

    // 3. What the LSE actually is. The reference is a double-precision base-2 and
    // base-e log-sum-exp over the causal scores of one (query, head) pair, using
    // the rounded BF16 inputs the kernel reads.
    const std::vector<float> got = lse.download(stream);
    size_t untouched = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == lse_sentinel[i]) ++untouched;
    }
    std::printf("invariance: E lse untouched elements: %zu/%zu (0 means the kernel wrote all of "
                "it)\n",
                untouched, got.size());

    // Both bases are computed, so "which one is it" is measured rather than
    // assumed: a backward that used a natural-log LSE would be wrong by ln 2.
    double worst_base2 = 0.0, worst_base_e = 0.0, worst_softmax2 = 0.0;
    double diff2_min = std::numeric_limits<double>::infinity();
    double diff2_max = -std::numeric_limits<double>::infinity();
    double diff2_sum = 0.0;
    size_t diff2_count = 0;
    int samples = 0;
    const double scale = 1.0 / std::sqrt((double)kHeadDim);
    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            const int kvh = h / (kHeads / kKvHeads);
            double max_score = -std::numeric_limits<double>::infinity();
            std::vector<double> scores(t + 1);
            for (int p = 0; p <= t; ++p) {
                double dot = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += value(q[(size_t)t * kRows + h * kHeadDim + d]) *
                           value(cache[(size_t)p * kKvRows + kvh * kHeadDim + d]);
                }
                scores[p] = dot * scale;
                max_score = std::max(max_score, scores[p]);
            }
            double sum_e = 0.0, sum_2 = 0.0;
            for (int p = 0; p <= t; ++p) {
                sum_e += std::exp(scores[p] - max_score);
                sum_2 += std::exp2(scores[p] - max_score);
            }
            // Three candidate readings of the returned 32-bit value:
            //   natural     ln(sum e^s)                  -- the natural log-sum-exp
            //   natural_in_l2  log2(sum e^s)             -- the natural LSE expressed in
            //                                            base 2 (what exp2/LOG2E kernels
            //                                            actually compute)
            //   base2_softmax  log2(sum 2^s)              -- the LSE of a softmax whose
            //                                            scores were already in base 2
            // The max is subtracted in the natural domain, so adding it back after
            // a log2 has to convert it: log2(sum e^s) = max*log2(e) + log2(sum e^{s-max}).
            // Getting this wrong (adding the natural max to a base-2 log) is a
            // mixed-domain error of max*(log2(e)-1), which is what made this probe
            // disagree with the kernel before the reference was fixed.
            const double ref_natural = std::log(sum_e) + max_score;
            const double ref_natural_in_l2 = std::log2(sum_e) + max_score * std::log2(M_E);
            const double ref_base2_softmax = std::log2(sum_2) + max_score;
            const double actual = got[(size_t)t * kHeads + h];
            worst_base2 = std::max(worst_base2, std::abs(actual - ref_natural_in_l2));
            worst_base_e = std::max(worst_base_e, std::abs(actual - ref_natural));
            worst_softmax2 = std::max(worst_softmax2, std::abs(actual - ref_base2_softmax));
            diff2_min = std::min(diff2_min, actual - ref_natural_in_l2);
            diff2_max = std::max(diff2_max, actual - ref_natural_in_l2);
            diff2_sum += actual - ref_natural_in_l2;
            ++diff2_count;
            if (samples < 6 && (t % 31 == 0) && h < 2) {
                std::printf("invariance: E sample t=%2d h=%2d actual=%.6f nat=%.6f nat_in_log2=%.6f "
                            "softmax2=%.6f\n",
                            t, h, actual, ref_natural, ref_natural_in_l2, ref_base2_softmax);
                ++samples;
            }
        }
    }
    std::printf("invariance: E lse vs double reference:\n");
    std::printf("  log2(sum e^s)  (natural LSE in base 2) worst |diff| = %.3e  <- the kernel's\n",
                worst_base2);
    std::printf("  ln(sum e^s)    (natural LSE)           worst |diff| = %.3e\n", worst_base_e);
    std::printf("  log2(sum 2^s)  (base-2 softmax LSE)    worst |diff| = %.3e\n", worst_softmax2);
    // The kernel folds LOG2E into the scores so it can use exp2: the value it
    // returns is log2 of the *natural* log-sum-exp. Reading that as a base-2
    // softmax LSE (or as a natural LSE) is off by a scale, which is exactly the
    // compatibility trap a backward has to avoid.
    // The gate is identification, not exactness: the kernel's value matches
    // log2(sum e^s) to ~2e-3 absolute while the two alternative readings are off by
    // 0.15 and 2.0, so requiring an order of magnitude of separation identifies the
    // convention unambiguously. The residual is the kernel's reduced-precision
    // softmax denominator (the same bf16 probability rounding the PV product uses),
    // not a different base; it is reported rather than gated tighter.
    const bool is_natural_in_log2 = worst_base2 < 1e-2 &&
                                    worst_base2 * 10.0 < worst_base_e &&
                                    worst_base2 * 10.0 < worst_softmax2;
    const bool layout_confirmed = untouched == 0;
    std::printf("invariance: E lse convention: log2(SUM e^s) %s (%.3e, i.e. %.1fx closer than "
                "the next candidate); layout [qo_idx * num_heads + head] fully written: %s\n",
                is_natural_in_log2 ? "identified" : "NOT identified", worst_base2,
                std::min(worst_base_e, worst_softmax2) / std::max(worst_base2, 1e-30),
                layout_confirmed ? "yes" : "NO");
    Recorder base_rec("E");
    base_rec.record("lse == log2(sum e^s): identified vs the alternatives (nat=%.3e, softmax2=%.3e)",
                    Delta{worst_base2, worst_base2, is_natural_in_log2 ? 0u : 1u,
                          (size_t)kTokens * kHeads, true},
                    "base");
    if (!is_natural_in_log2 || !layout_confirmed || !out.bitwise()) {
        std::printf("invariance: E FAIL\n");
        return EXIT_FAILURE;
    }

    // The resource cost of the pair's saved state, per token per layer.
    const size_t lse_bytes = (size_t)kTokens * kHeads * sizeof(float);
    std::printf("invariance: E saved state per layer: lse [%d, %d] f32 = %zu bytes for %d tokens "
                "(%.1f bytes/token), against q/o [%d, %d, %d] bf16 = %zu bytes each\n",
                kTokens, kHeads, lse_bytes, kTokens, (double)lse_bytes / kTokens, kTokens, kHeads,
                kHeadDim, (size_t)kTokens * kRows * sizeof(Bf16));
    rec.summary();
    cudaStreamDestroy(stream);
    std::printf("test_attention_lse: PASS (the forward can produce the LSE the backward needs, "
                "and the engine's output is unchanged)\n");
    return EXIT_SUCCESS;
}
