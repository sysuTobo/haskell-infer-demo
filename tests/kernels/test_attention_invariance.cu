/*
 * test_attention_invariance.cu - Stage 2 claim C.
 *
 * "Record disabled split-KV, then compare the same causal positions from full,
 * split and single-query execution with identical Q/K/V; vary KV length, GQA and
 * supported head dimensions."
 *
 * The cache is built on the host and handed to every arm unchanged, so all three
 * arms see literally the same K/V. Only the query grouping differs:
 *
 *   full     one call with all L queries (seq_len = L)
 *   split    two calls at a boundary b (which is what the engine's chunked prefill
 *            does when a prompt exceeds max_chunk)
 *   single   one call per query at its own seq_len (which is what decode does)
 *
 * Split-KV, recorded: kernel_attention passes a null split-KV workspace to
 * FlashInfer's SinglePrefillWithKVCacheDispatched (csrc/kernels/attention.cu, the
 * SinglePrefillParams constructor's tmp argument), and that dispatcher clears
 * partition_kv when the workspace is null -- flashinfer/attention/prefill.cuh:
 * "if (num_chunks <= 1 || tmp == nullptr) { params.partition_kv = false; }" -- so
 * this forward runs the no-split-KV mode. The manifest already records that as
 * attention_split_kv = disabled_null_workspace.
 */
#include "invariance_utils.h"
#include "kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
using namespace invariance;
using test::Bf16;
using test::bf16;
using test::DeviceBuffer;
using test::poison;
using test::sample;

struct Gqa {
    int heads;
    int kv_heads;
};

const int kHeadDims[] = {128, 256};
const Gqa kGqas[] = {{24, 4}, {24, 8}, {8, 4}};
const int kLengths[] = {1, 2, 7, 8, 63, 64, 65, 128, 129, 255, 256, 269};

/* One arm's queries, executed and joined back into a [length, heads * head_dim]
 * output so every arm is comparable row for row. */
std::vector<Bf16> run_full(const std::vector<Bf16> &q, const std::vector<Bf16> &cache, int length,
                           int heads, int kv_heads, int head_dim, int max_seq,
                           cudaStream_t stream) {
    DeviceBuffer<Bf16> dq(q.size()), dcache(cache.size());
    DeviceBuffer<Bf16> out((size_t)length * heads * head_dim);
    dq.upload(q, stream);
    dcache.upload(cache, stream);
    out.upload(poison((size_t)length * heads * head_dim), stream);
    kernel_attention(out.get(), dq.get(), dcache.get(), 0, length, length, heads, kv_heads, head_dim,
                     1.0f / std::sqrt((float)head_dim), max_seq, stream);
    return out.download(stream);
}

std::vector<Bf16> run_split(const std::vector<Bf16> &q, const std::vector<Bf16> &cache, int length,
                            int boundary, int heads, int kv_heads, int head_dim, int max_seq,
                            cudaStream_t stream) {
    const int rows = heads * head_dim;
    DeviceBuffer<Bf16> dq(q.size()), dcache(cache.size());
    DeviceBuffer<Bf16> out((size_t)length * rows);
    dq.upload(q, stream);
    dcache.upload(cache, stream);
    out.upload(poison((size_t)length * rows), stream);
    // The prefix is its own call (seq_len = boundary), the suffix another
    // (seq_start = boundary, seq_len = length): the same causal positions, but
    // FlashInfer sees two different query blocks.
    kernel_attention(out.get(), dq.get(), dcache.get(), 0, boundary, boundary, heads, kv_heads,
                     head_dim, 1.0f / std::sqrt((float)head_dim), max_seq, stream);
    kernel_attention(out.get() + (size_t)boundary * rows, dq.get() + (size_t)boundary * rows,
                     dcache.get(), boundary, length - boundary, length, heads, kv_heads, head_dim,
                     1.0f / std::sqrt((float)head_dim), max_seq, stream);
    return out.download(stream);
}

std::vector<Bf16> run_single(const std::vector<Bf16> &q, const std::vector<Bf16> &cache, int length,
                             int heads, int kv_heads, int head_dim, int max_seq,
                             cudaStream_t stream) {
    const int rows = heads * head_dim;
    DeviceBuffer<Bf16> dq(q.size()), dcache(cache.size());
    DeviceBuffer<Bf16> out((size_t)length * rows);
    dq.upload(q, stream);
    dcache.upload(cache, stream);
    out.upload(poison((size_t)length * rows), stream);
    for (int t = 0; t < length; ++t) {
        kernel_attention(out.get() + (size_t)t * rows, dq.get() + (size_t)t * rows, dcache.get(), t,
                         1, t + 1, heads, kv_heads, head_dim,
                         1.0f / std::sqrt((float)head_dim), max_seq, stream);
    }
    return out.download(stream);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::printf("invariance: C: stream creation failed\n");
        return EXIT_FAILURE;
    }

    Recorder rec("C");
    bool ok = true;
    std::printf("invariance: claim C (attention): full vs split-prefill vs single-query, "
                "split-KV disabled by the null workspace\n");

    for (int head_dim : kHeadDims) {
        for (const Gqa &gqa : kGqas) {
            const int rows = gqa.heads * head_dim;
            const int kv_rows = gqa.kv_heads * head_dim;
            for (int length : kLengths) {
                const int max_seq = length;
                std::vector<Bf16> q((size_t)length * rows);
                std::vector<Bf16> cache((size_t)2 * max_seq * kv_rows);
                for (size_t i = 0; i < q.size(); ++i) q[i] = bf16(sample(i, 740));
                for (size_t i = 0; i < cache.size(); ++i) cache[i] = bf16(sample(i, 741) * 0.5f);
                const std::string shape = "hd" + std::to_string(head_dim) + "/gqa" +
                                          std::to_string(gqa.heads) + "x" +
                                          std::to_string(gqa.kv_heads) + "/L=" +
                                          std::to_string(length);

                const std::vector<Bf16> full =
                    run_full(q, cache, length, gqa.heads, gqa.kv_heads, head_dim, max_seq, stream);
                double peak = 0.0;
                for (const Bf16 &v : full) peak = std::max(peak, std::abs((double)test::value(v)));
                const Delta single = delta(
                    run_single(q, cache, length, gqa.heads, gqa.kv_heads, head_dim, max_seq, stream),
                    full);
                rec.record(shape + " single-query", single, "single", peak);
                if (!single.finite) ok = false;

                // Split boundaries: the engine's chunking at 64/128, plus the
                // first token and the midpoint as the smallest and largest a
                // split can be.
                const int boundaries[] = {1, length / 2, 64, 128};
                for (int b : boundaries) {
                    if (b <= 0 || b >= length) continue;
                    const Delta split = delta(run_split(q, cache, length, b, gqa.heads, gqa.kv_heads,
                                                        head_dim, max_seq, stream),
                                              full);
                    rec.record(shape + " split=" + std::to_string(b) + "+" +
                                   std::to_string(length - b),
                               split, "split", peak);
                    if (!split.finite) ok = false;
                }
            }
        }
        std::printf("invariance: C: head_dim=%d done\n", head_dim);
    }

    rec.summary();
    if (!ok || rec.nonfinite() != 0) {
        std::printf("invariance: C FAIL: nonfinite output\n");
        return EXIT_FAILURE;
    }
    std::printf("invariance: C worst max_abs=%.9g over %d measurements (%d bitwise): the "
                "attention region is %s across these tilings\n",
                rec.worst(), rec.measurements(), rec.bitwise(),
                rec.bitwise() == rec.measurements() ? "bitwise invariant"
                                                    : "NOT bitwise invariant; the number above is "
                                                      "the bound a trainer has to carry");
    cudaStreamDestroy(stream);
    std::printf("test_attention_invariance: measured\n");
    return EXIT_SUCCESS;
}
