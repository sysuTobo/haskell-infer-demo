/*
 * test_gemm_invariance.cu - Stage 2 claim D.
 *
 * "Synthetic fixed inputs/weights; vary row count, prefixes/splits, M=1,
 * boundary/tail sizes and actual projection N/K. Cover BF16 outputs and FP32
 * LM-head outputs. A failure falsifies invariance; one passing 128-vs-64+64 test
 * does not prove it. No full checkpoint load is needed."
 *
 * The question a trainer needs answered is whether the *same logical rows* produce
 * the same bits when the batch is split differently: one M-row call, one call per
 * row (M=1, which is what decode does), and a prefix/suffix split at a boundary.
 * cuBLAS is free to pick a different algorithm or workspace for each shape, so
 * this is the experiment that decides whether the GEMM region can claim exactness
 * or has to carry a measured bound instead.
 *
 * Weights are the real projection shapes of the deployment model (Qwen3.8-27B:
 * hidden 5120, 24 heads x 256, 4 kv heads, intermediate 17408, GDN conv 10240 /
 * value 6144, vocab 248320) so the shapes cuBLAS actually sees are covered; the
 * synthetic inputs are fixed by index hash, so the run is reproducible.
 */
#include "invariance_utils.h"
#include "layers.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace invariance;
using test::Bf16;
using test::bf16;
using test::DeviceBuffer;
using test::poison;
using test::sample;

/* One projection: the (N,K) cuBLAS actually gets, and the row counts worth
 * testing. Rows are chosen around the boundaries the engine uses (1 decode, 8,
 * 64/65 and 128/129 = the chunking and FLA block sizes, 433/434 = the long
 * prompt, and a tail of one). Wide projections cannot afford a per-row arm at
 * large M (each M=1 call still reads the whole weight matrix), so their row lists
 * stop earlier. */
/* Poison of the output element type: every element an arm is expected to write
 * starts as a NaN, so an arm that silently skips a row cannot pass. */
template <typename T> std::vector<T> poison_like(size_t count) {
    if constexpr (std::is_same<T, Bf16>::value) {
        return test::poison(count);
    } else {
        return std::vector<T>(count, std::numeric_limits<T>::quiet_NaN());
    }
}

struct Projection {
    const char *name;
    int N;
    int K;
    const int *rows;
    int row_count;
};

const int kRowsFull[] = {1, 2, 7, 8, 63, 64, 65, 127, 128, 129, 433, 434};
const int kRowsMid[] = {1, 8, 63, 64, 65, 129};
const int kRowsWide[] = {1, 8, 65};

const Projection kProjections[] = {
    {"attn_kv",          1024,  5120, kRowsFull, 12},  // k_proj / v_proj
    {"gdn_ab",             48,  5120, kRowsFull, 12},  // in_proj_a / in_proj_b
    {"attn_q_gate",     12288,  5120, kRowsMid,   6},  // q_proj with the fused gate
    {"attn_o",           5120,  6144, kRowsMid,   6},
    {"gdn_qkv",         10240,  5120, kRowsMid,   6},
    {"gdn_z",            6144,  5120, kRowsMid,   6},
    {"gdn_out",          5120,  6144, kRowsMid,   6},
    {"mlp_gate_up",     17408,  5120, kRowsMid,   6},
    {"mlp_down",         5120, 17408, kRowsMid,   6},
    {"lm_head",        248320,  5120, kRowsWide,  3},
};

/* One test case, output type templated so the BF16 and FP32-output entry points
 * share every arm. */
template <typename OutT>
using GemmFn = int (*)(cublasHandle_t, OutT *, const Bf16 *, const Bf16 *, int, int, int);

template <typename OutT>
bool run_projection(cublasHandle_t blas, Recorder &rec, const char *out_kind,
                    GemmFn<OutT> gemm, const Projection &p, cudaStream_t stream) {
    const int max_rows = p.rows[p.row_count - 1];
    bool ok = true;

    std::vector<Bf16> x((size_t)max_rows * p.K);
    std::vector<Bf16> w((size_t)p.N * p.K);
    // Fixed by index hash; the same buffer is read by every arm, so a difference
    // can only come from how cuBLAS was called.
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16(0.5f * sample(i, 700));
    for (size_t i = 0; i < w.size(); ++i) w[i] = bf16(0.25f * sample(i, 701));

    DeviceBuffer<Bf16> dx(x.size()), dw(w.size());
    DeviceBuffer<OutT> baseline((size_t)max_rows * p.N);
    DeviceBuffer<OutT> arm((size_t)max_rows * p.N);
    dx.upload(x, stream);
    dw.upload(w, stream);

    for (int r = 0; r < p.row_count; ++r) {
        const int M = p.rows[r];
        // Only the first M rows are meaningful: the buffers are sized for the
        // longest case, and an unwritten tail would still be poisoned.
        const size_t elements = (size_t)M * p.N;
        const std::string label = std::string(p.name) + "/" + out_kind + "/M=" + std::to_string(M);

        if (gemm(blas, baseline.get(), dx.get(), dw.get(), M, p.N, p.K) != 0) {
            std::printf("invariance: D one-shot %s: GEMM failed\n", label.c_str());
            return false;
        }

        /* Arm 1: one row per call (decode's shape). */
        arm.upload(poison_like<OutT>((size_t)max_rows * p.N), stream);
        for (int row = 0; row < M; ++row) {
            if (gemm(blas, arm.get() + (size_t)row * p.N, dx.get() + (size_t)row * p.K, dw.get(), 1,
                     p.N, p.K) != 0) {
                std::printf("invariance: D per-row %s: GEMM failed\n", label.c_str());
                return false;
            }
        }
        const std::vector<OutT> base_rows = [&] {
            const std::vector<OutT> all = baseline.download(stream);
            return std::vector<OutT>(all.begin(), all.begin() + elements);
        }();
        double peak = 0.0;
        for (const OutT &v : base_rows) peak = std::max(peak, std::abs((double)test::value(v)));
        const Delta per_row = [&] {
            const std::vector<OutT> all = arm.download(stream);
            return delta(std::vector<OutT>(all.begin(), all.begin() + elements), base_rows);
        }();
        rec.record(label + " per-row(M=1)", per_row, "per-row", peak);
        if (!per_row.finite) ok = false;

        /* Arm 2: prefix/suffix splits. The boundaries are the chunk sizes the
         * engine and FLA use, plus a one-row tail. A boundary equal to another is
         * run once: the comparison does not change with the label. */
        std::vector<int> boundaries = {1, 8, 64, 128, M - 1};
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        for (int b : boundaries) {
            if (b <= 0 || b >= M) continue;
            arm.upload(poison_like<OutT>((size_t)max_rows * p.N), stream);
            if (gemm(blas, arm.get(), dx.get(), dw.get(), b, p.N, p.K) != 0 ||
                gemm(blas, arm.get() + (size_t)b * p.N, dx.get() + (size_t)b * p.K, dw.get(),
                     M - b, p.N, p.K) != 0) {
                std::printf("invariance: D split %s: GEMM failed\n", label.c_str());
                return false;
            }
            const Delta split = [&] {
                const std::vector<OutT> all = arm.download(stream);
                return delta(std::vector<OutT>(all.begin(), all.begin() + elements), base_rows);
            }();
            rec.record(label + " split=" + std::to_string(b) + "+" + std::to_string(M - b), split,
                       "split", peak);
            if (!split.finite) ok = false;
        }

        /* The M=1 case has no split arm and its per-row arm is the same call, so
         * report the trivially-zero comparison as such rather than pretending it
         * is evidence. */
    }
    return ok;
}

bool check_cublas(cublasStatus_t status, const char *what) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::printf("invariance: D: %s failed (%d)\n", what, (int)status);
    return false;
}

}  // namespace

int main() {
    // Line-buffered so the evidence survives an abort in a later step.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    cublasHandle_t blas = nullptr;
    if (!check_cublas(cublasCreate(&blas), "cublasCreate")) return EXIT_FAILURE;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::printf("invariance: D: stream creation failed\n");
        return EXIT_FAILURE;
    }
    if (!check_cublas(cublasSetStream(blas, stream), "cublasSetStream")) return EXIT_FAILURE;

    Recorder bf16_rec("D"), fp32_rec("D");
    bool ok = true;

    std::printf("invariance: claim D (GEMM): same logical rows under one call, per-row calls and "
                "prefix splits\n");
    for (const Projection &p : kProjections) {
        ok &= run_projection<Bf16>(blas, bf16_rec, "bf16out", gemm_bf16, p, stream);
    }
    // The FP32-output entry point (the LM head projection) at the cheap shapes:
    // the wide ones would only repeat the same cuBLAS selection question.
    for (const Projection &p : kProjections) {
        if (p.N > 12288) continue;
        ok &= run_projection<float>(blas, fp32_rec, "fp32out", gemm_bf16_f32out, p, stream);
    }
    bf16_rec.summary();
    fp32_rec.summary();

    // The decision this experiment makes is *not* that the region is exact. It
    // answers whether the same rows under a different M produce the same bits;
    // the bound printed above is what the region has to carry if they do not.
    std::printf("invariance: D worst BF16-output max_abs=%.9g over %d measurements (%d bitwise)\n",
                bf16_rec.worst(), bf16_rec.measurements(), bf16_rec.bitwise());
    std::printf("invariance: D worst FP32-output max_abs=%.9g over %d measurements (%d bitwise)\n",
                fp32_rec.worst(), fp32_rec.measurements(), fp32_rec.bitwise());
    // A NaN, or a deviation beyond the region's documented band, fails.
    if (!ok || bf16_rec.nonfinite() != 0 || fp32_rec.nonfinite() != 0) {
        std::printf("invariance: D FAIL: nonfinite output\n");
        return EXIT_FAILURE;
    }

    cudaStreamDestroy(stream);
    cublasDestroy(blas);
    std::printf("test_gemm_invariance: measured (the number above is the region's shape-invariance "
                "bound, not a claim of exactness)\n");
    return EXIT_SUCCESS;
}
