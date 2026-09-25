/*
 * test_region_cases.cu - The Stage-1 cross-case harness.
 *
 * docs/plan-numeric-contract.md, Stage 1: "fixtures compare identical inputs and
 * persistent state across applicable cases; unsupported shapes/cases fail
 * explicitly". This file runs those fixtures for every region whose case pair is
 * drivable from the device side, and adjudicates each comparison against the
 * verdict the committed inventory registers (csrc/regions.c):
 *
 *   exact           must come out bitwise identical for output *and* persistent
 *                   state, or the run fails;
 *   unverified      the difference is measured and printed -- a library tiling
 *                   order or a decomposition schedule is not established, so
 *                   invariance is not claimed. Where the region owns persistent
 *                   state, a state-LOSS guard (rms < 5) still applies: that is
 *                   the coarse guard the plan keeps, not an invariance claim;
 *   exception       must stay inside the registered bound (none registered yet);
 *   not_applicable  must not be executed at all, and asking for it is a failure.
 *
 * The harness also re-runs the unsupported-shape/case rejections (an over-long
 * chunk, an over-long sequence, a non-contiguous SiLU pair, a wrong key-head
 * count) and measures the region entry-point host cost against its device cost,
 * so "the boundary is free" is a measurement rather than an assumption.
 *
 * The engine's Haskell FFI is model-level today (engine_create/prefill/decode);
 * the cost measured at the bottom is the C region entry point -- argument
 * validation, workspace arithmetic and the enqueue -- not a Haskell ccall. Stage
 * 3 introduces region handles; that is when the two can be compared.
 *
 * Runs on 2xA40 sm_86. No weights and no checkpoint are needed.
 */
#include "fla_ops.h"
#include "flashinfer_ops.h"
#include "kernels.h"
#include "layers.h"
#include "regions.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace test;

/* ---------------------------------------------------------------- */
/* Shapes                                                           */
/* ---------------------------------------------------------------- */

/* Attention. kernel_attention requires head_dim == 256, so this is the Qwen3.8
 * shape rather than a small synthetic one. */
constexpr int kHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 64;
constexpr int kQDim = kHeads * kHeadDim;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kMaxSeq = 16;

/* Dense path. */
constexpr int kHidden = 256;
constexpr int kIntermediate = 512;
constexpr int kVocab = 64;

/* GDN. kernel_fla_gdn requires exactly 16 key heads and head_dim == 128, and the
 * AOT recurrent kernel exists for 48 value heads, so this is the Qwen3.8 GDN
 * layout too. conv_dim = (2 * key_heads + value_heads) * head_dim. */
constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kGdnHeadDim = 128;
constexpr int kConvDim = (2 * kKeyHeads + kValueHeads) * kGdnHeadDim;
constexpr int kVdim = kValueHeads * kGdnHeadDim;
constexpr int kConvWidth = 4;
constexpr int kStateElems = kValueHeads * kGdnHeadDim * kGdnHeadDim;

/* The chunk length every fixture compares against. */
constexpr int kTokens = 8;

/* ---------------------------------------------------------------- */
/* Comparison and adjudication                                      */
/* ---------------------------------------------------------------- */

struct Delta {
    double max_abs = 0.0;
    double rms = 0.0;
    bool finite = true;
};

template <typename T>
Delta difference(const std::vector<T> &a, const std::vector<T> &b) {
    Delta d;
    if (a.size() != b.size()) {
        d.finite = false;
        d.max_abs = std::numeric_limits<double>::infinity();
        return d;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = value(a[i]), y = value(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            d.finite = false;
            continue;
        }
        const double error = std::abs(x - y);
        d.max_abs = std::max(d.max_abs, error);
        sum += error * error;
    }
    if (!d.finite) d.max_abs = std::numeric_limits<double>::infinity();
    d.rms = a.empty() ? 0.0 : std::sqrt(sum / a.size());
    return d;
}

std::vector<std::string> g_executed;

std::string key(const char *region, const char *left, const char *right) {
    /* Order-free, like region_find_pair. */
    std::string a = left, b = right;
    if (b < a) std::swap(a, b);
    return std::string(region) + "|" + a + "|" + b;
}

bool was_executed(const char *region, const char *left, const char *right) {
    const std::string want = key(region, left, right);
    for (const std::string &k : g_executed) {
        if (k == want) return true;
    }
    return false;
}

/* Print one line per comparison. It is the recorded evidence: which region, which
 * two cases, the registered verdict, and what the two came out to. */
bool adjudicate(const char *region, const char *left, const char *right,
                const Delta &out, const Delta *state) {
    const struct RegionCasePair *pair = region_find_pair(region, left, right);
    if (pair == nullptr) {
        printf("region_cases: %-22s %s vs %s: FAIL: no registered pair\n", region, left, right);
        return false;
    }
    static const Delta none;
    const Delta &st = state != nullptr ? *state : none;
    g_executed.push_back(key(region, left, right));
    printf("region_cases: %-22s %-18s vs %-18s verdict=%-14s out_max_abs=%.9g out_rms=%.9g "
           "state_max_abs=%.9g state_rms=%.9g\n",
           region, left, right, pair->verdict, out.max_abs, out.rms, st.max_abs, st.rms);

    if (strcmp(pair->verdict, REGION_VERDICT_NOT_APPLICABLE) == 0) {
        printf("region_cases: %s: FAIL: a not_applicable pair was executed\n", region);
        return false;
    }
    if (!out.finite || !st.finite) {
        printf("region_cases: %s/%s/%s: FAIL: a nonfinite value was compared\n", region, left,
               right);
        return false;
    }
    if (strcmp(pair->verdict, REGION_VERDICT_EXACT) == 0) {
        if (out.max_abs != 0.0 || st.max_abs != 0.0) {
            printf("region_cases: %s/%s/%s: FAIL: registered exact but not bitwise\n", region,
                   left, right);
            return false;
        }
        return true;
    }
    if (strcmp(pair->verdict, REGION_VERDICT_EXCEPTION) == 0) {
        if (out.max_abs > pair->max_abs || out.rms > pair->rms) {
            printf("region_cases: %s/%s/%s: FAIL: outside the registered exception bound\n",
                   region, left, right);
            return false;
        }
        return true;
    }
    /* unverified: measured and reported, not established. Persistent state still
     * gets the plan's coarse state-loss guard so a decomposition that loses state
     * fails loudly instead of being reported as an ordinary difference. */
    if (strcmp(pair->state, "-") != 0 && st.rms >= 5.0) {
        printf("region_cases: %s/%s/%s: FAIL: state-loss guard (rms %.9g >= 5) -- this is the "
               "coarse guard, not an invariance claim\n",
               region, left, right, st.rms);
        return false;
    }
    return true;
}

/* ---------------------------------------------------------------- */
/* Fixtures                                                         */
/* ---------------------------------------------------------------- */

/* Embedding: one gather per token. */
bool case_embedding(cudaStream_t stream) {
    std::vector<Bf16> table(kVocab * kHidden), ids_host(kTokens);
    for (size_t i = 0; i < table.size(); ++i) table[i] = bf16(sample(i, 200));
    std::vector<int64_t> ids(kTokens);
    for (int t = 0; t < kTokens; ++t) ids[t] = (3 * t + 1) % kVocab;
    DeviceBuffer<Bf16> dtable(table.size()), out_a(kTokens * kHidden), out_b(kTokens * kHidden);
    DeviceBuffer<int64_t> dids(kTokens);
    dtable.upload(table, stream);
    dids.upload(ids, stream);

    kernel_embedding(out_a.get(), dtable.get(), dids.get(), kHidden, kTokens, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_embedding(out_b.get() + (size_t)t * kHidden, dtable.get(), dids.get() + t, kHidden,
                         1, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("embedding", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d, nullptr);
}

/* RMSNorm (Gemma weight+1): the row width is fixed, the row count is not. */
bool case_rmsnorm(cudaStream_t stream) {
    const int rows = kTokens;
    std::vector<Bf16> x(rows * kHidden), weight(kHidden, bf16(0.5f));
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16(sample(i, 201));
    DeviceBuffer<Bf16> dx(x.size()), dw(weight.size()), out_a(x.size()), out_b(x.size());
    dx.upload(x, stream);
    dw.upload(weight, stream);
    kernel_gemma_rms_norm(out_a.get(), dx.get(), dw.get(), kHidden, rows, 1e-6f, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < rows; ++t) {
        kernel_gemma_rms_norm(out_b.get() + (size_t)t * kHidden, dx.get() + (size_t)t * kHidden,
                              dw.get(), kHidden, 1, 1e-6f, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("rmsnorm", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d, nullptr);
}

/* Per-head Q/K norm: [tokens * heads, head_dim] rows normalized independently. */
bool case_per_head_norm(cudaStream_t stream) {
    const int rows = kTokens * kHeads;
    std::vector<Bf16> x(rows * kHeadDim), weight(kHeadDim, bf16(0.25f));
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16(sample(i, 202));
    DeviceBuffer<Bf16> dx(x.size()), dw(weight.size()), out_a(x.size()), out_b(x.size());
    dx.upload(x, stream);
    dw.upload(weight, stream);
    kernel_gemma_rms_norm(out_a.get(), dx.get(), dw.get(), kHeadDim, rows, 1e-6f, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        const size_t offset = (size_t)t * kHeads * kHeadDim;
        kernel_gemma_rms_norm(out_b.get() + offset, dx.get() + offset, dw.get(), kHeadDim, kHeads,
                              1e-6f, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("per_head_norm", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d,
                      nullptr);
}

/* RoPE: the rotated values depend only on the row and its position. */
bool case_rope(cudaStream_t stream) {
    std::vector<Bf16> q(kTokens * kQDim), k(kTokens * kKvDim);
    for (size_t i = 0; i < q.size(); ++i) q[i] = bf16(sample(i, 203));
    for (size_t i = 0; i < k.size(); ++i) k[i] = bf16(sample(i, 204));
    std::vector<int64_t> positions(kTokens);
    for (int t = 0; t < kTokens; ++t) positions[t] = t + 3;
    DeviceBuffer<Bf16> dq_a(q.size()), dk_a(k.size()), dq_b(q.size()), dk_b(k.size());
    DeviceBuffer<int64_t> dpos(positions.size());
    dq_a.upload(q, stream);
    dk_a.upload(k, stream);
    dq_b.upload(q, stream);
    dk_b.upload(k, stream);
    dpos.upload(positions, stream);

    kernel_flashinfer_rope(dq_a.get(), dk_a.get(), dpos.get(), kTokens, kHeads, kKvHeads, kHeadDim,
                           kRotaryDim, 1e7f, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_flashinfer_rope(dq_b.get() + (size_t)t * kQDim, dk_b.get() + (size_t)t * kKvDim,
                               dpos.get() + t, 1, kHeads, kKvHeads, kHeadDim, kRotaryDim, 1e7f,
                               stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta dq = difference(dq_a.download(stream), dq_b.download(stream));
    const Delta dk = difference(dk_a.download(stream), dk_b.download(stream));
    const Delta d{std::max(dq.max_abs, dk.max_abs), std::max(dq.rms, dk.rms),
                  dq.finite && dk.finite};
    return adjudicate("rope", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, d,
                      nullptr);
}

/* Q/gate de-interleave: a permutation of the fused projection. `total` is the
 * number of Q elements (rows times head_dim), which is how the real caller
 * computes it: `tokens * heads * head_dim`. The head index the kernel derives is
 * therefore the flattened (token, head) index, so a per-token call must pass the
 * token slice of `raw` and the token-local element count. */
bool case_q_gate_split(cudaStream_t stream) {
    const int rows = kTokens * kHeads;
    const int total = rows * kHeadDim;
    std::vector<Bf16> raw((size_t)rows * 2 * kHeadDim);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = bf16(sample(i, 205));
    DeviceBuffer<Bf16> draw(raw.size());
    DeviceBuffer<Bf16> dq_a(total), dg_a(total), dq_b(total), dg_b(total);
    draw.upload(raw, stream);

    kernel_q_gate_split(dq_a.get(), dg_a.get(), draw.get(), total, kHeadDim, stream);
    CUDA_CHECK(cudaGetLastError());
    const int chunk_rows = kHeads;  // one token's (token, head) rows
    for (int t = 0; t < kTokens; ++t) {
        kernel_q_gate_split(dq_b.get() + (size_t)t * chunk_rows * kHeadDim,
                            dg_b.get() + (size_t)t * chunk_rows * kHeadDim,
                            draw.get() + (size_t)t * chunk_rows * 2 * kHeadDim,
                            chunk_rows * kHeadDim, kHeadDim, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta dq = difference(dq_a.download(stream), dq_b.download(stream));
    const Delta dg = difference(dg_a.download(stream), dg_b.download(stream));
    const Delta d{std::max(dq.max_abs, dg.max_abs), std::max(dq.rms, dg.rms),
                  dq.finite && dg.finite};
    return adjudicate("q_gate_split", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d, nullptr);
}

/* KV write: same k/v, written in one chunk or one token at a time. The whole
 * cache is the persistent state, and it is compared byte for byte. */
bool case_kv_write(cudaStream_t stream) {
    std::vector<Bf16> k(kTokens * kKvDim), v(kTokens * kKvDim);
    for (size_t i = 0; i < k.size(); ++i) k[i] = bf16(sample(i, 206));
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16(2.0f + sample(i, 207));
    const size_t cache_elems = 2 * (size_t)kMaxSeq * kKvDim;
    std::vector<Bf16> sentinel(cache_elems, bf16(-7.0f));
    DeviceBuffer<Bf16> dk(k.size()), dv(v.size());
    DeviceBuffer<Bf16> cache_a(cache_elems), cache_b(cache_elems);
    dk.upload(k, stream);
    dv.upload(v, stream);
    cache_a.upload(sentinel, stream);
    cache_b.upload(sentinel, stream);

    kernel_kv_cache_write(cache_a.get(), dk.get(), dv.get(), 0, kTokens, kKvHeads, kHeadDim,
                          kMaxSeq, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_kv_cache_write(cache_b.get(), dk.get() + (size_t)t * kKvDim,
                              dv.get() + (size_t)t * kKvDim, t, 1, kKvHeads, kHeadDim, kMaxSeq,
                              stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(cache_a.download(stream), cache_b.download(stream));
    return adjudicate("kv_write", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d, &d);
}

/* Attention core. Both pairs share one cache trajectory, so the only difference
 * is how the queries are handed to FlashInfer. */
bool case_attention(cudaStream_t stream) {
    std::vector<Bf16> q(kTokens * kQDim), k(kTokens * kKvDim), v(kTokens * kKvDim);
    for (size_t i = 0; i < q.size(); ++i) q[i] = bf16(sample(i, 208));
    for (size_t i = 0; i < k.size(); ++i) k[i] = bf16(sample(i, 209));
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16(sample(i, 210) * 0.5f);
    const size_t cache_elems = 2 * (size_t)kMaxSeq * kKvDim;
    const std::vector<Bf16> sentinel(cache_elems, bf16(31.0f));
    const float scale = 1.0f / std::sqrt((float)kHeadDim);

    DeviceBuffer<Bf16> dq(q.size()), dk(k.size()), dv(v.size());
    DeviceBuffer<Bf16> cache_a(cache_elems), cache_b(cache_elems), cache_c(cache_elems);
    DeviceBuffer<Bf16> out_a(kTokens * kQDim), out_b(kTokens * kQDim), out_c(kTokens * kQDim);
    dq.upload(q, stream);
    dk.upload(k, stream);
    dv.upload(v, stream);
    cache_a.upload(sentinel, stream);
    cache_b.upload(sentinel, stream);
    cache_c.upload(sentinel, stream);
    out_a.upload(poison(kTokens * kQDim), stream);
    out_b.upload(poison(kTokens * kQDim), stream);
    out_c.upload(poison(kTokens * kQDim), stream);

    /* Arm A: one 8-token prefill. */
    kernel_kv_cache_write(cache_a.get(), dk.get(), dv.get(), 0, kTokens, kKvHeads, kHeadDim,
                          kMaxSeq, stream);
    kernel_attention(out_a.get(), dq.get(), cache_a.get(), 0, kTokens, kTokens, kHeads, kKvHeads,
                     kHeadDim, scale, kMaxSeq, stream);
    CUDA_CHECK(cudaGetLastError());

    /* Arm B (decode): one query at a time, at its own sequence length. */
    for (int t = 0; t < kTokens; ++t) {
        kernel_kv_cache_write(cache_b.get(), dk.get() + (size_t)t * kKvDim,
                              dv.get() + (size_t)t * kKvDim, t, 1, kKvHeads, kHeadDim, kMaxSeq,
                              stream);
        kernel_attention(out_b.get() + (size_t)t * kQDim, dq.get() + (size_t)t * kQDim,
                         cache_b.get(), t, 1, t + 1, kHeads, kKvHeads, kHeadDim, scale, kMaxSeq,
                         stream);
    }
    CUDA_CHECK(cudaGetLastError());

    /* Arm C (tail1): a 7-token prefill then a one-token tail. */
    kernel_kv_cache_write(cache_c.get(), dk.get(), dv.get(), 0, kTokens - 1, kKvHeads, kHeadDim,
                          kMaxSeq, stream);
    kernel_attention(out_c.get(), dq.get(), cache_c.get(), 0, kTokens - 1, kTokens - 1, kHeads,
                     kKvHeads, kHeadDim, scale, kMaxSeq, stream);
    kernel_kv_cache_write(cache_c.get(), dk.get() + (size_t)(kTokens - 1) * kKvDim,
                          dv.get() + (size_t)(kTokens - 1) * kKvDim, kTokens - 1, 1, kKvHeads,
                          kHeadDim, kMaxSeq, stream);
    kernel_attention(out_c.get() + (size_t)(kTokens - 1) * kQDim,
                     dq.get() + (size_t)(kTokens - 1) * kQDim, cache_c.get(), kTokens - 1, 1,
                     kTokens, kHeads, kKvHeads, kHeadDim, scale, kMaxSeq, stream);
    CUDA_CHECK(cudaGetLastError());

    const std::vector<Bf16> a = out_a.download(stream);
    const std::vector<Bf16> cache_a_host = cache_a.download(stream);
    const Delta decode = difference(a, out_b.download(stream));
    const Delta tail = difference(a, out_c.download(stream));
    const Delta cache = difference(cache_a_host, cache_b.download(stream));
    bool ok = adjudicate("attention_core", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE,
                         decode, &cache);
    ok &= adjudicate("attention_core", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_TAIL1, tail,
                     nullptr);
    return ok;
}

/* Output gate: an elementwise multiply with a per-token gate stride. */
bool case_attention_output_gate(cudaStream_t stream) {
    const int total = kTokens * kQDim;
    std::vector<Bf16> attn(total), gate(total);
    for (size_t i = 0; i < attn.size(); ++i) attn[i] = bf16(sample(i, 211));
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = bf16(4.0f * sample(i, 212));
    DeviceBuffer<Bf16> dattn(attn.size()), dgate(gate.size());
    DeviceBuffer<Bf16> out_a(attn.size()), out_b(attn.size());
    dattn.upload(attn, stream);
    dgate.upload(gate, stream);

    kernel_sigmoid_mul(out_a.get(), dattn.get(), dgate.get(), kQDim, kTokens, kQDim, 0, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_sigmoid_mul(out_b.get() + (size_t)t * kQDim, dattn.get() + (size_t)t * kQDim,
                           dgate.get() + (size_t)t * kQDim, kQDim, 1, kQDim, 0, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("attention_output_gate", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d,
                      nullptr);
}

/* GEMM, BF16 output: the M dimension is what the case changes. */
bool case_gemm_bf16(cublasHandle_t blas, cudaStream_t stream) {
    std::vector<Bf16> x(kTokens * kHidden), w(kIntermediate * kHidden);
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16(sample(i, 213) * 0.5f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = bf16(sample(i, 214) * 0.25f);
    DeviceBuffer<Bf16> dx(x.size()), dw(w.size());
    DeviceBuffer<Bf16> out_a((size_t)kTokens * kIntermediate);
    DeviceBuffer<Bf16> out_b((size_t)kTokens * kIntermediate);
    dx.upload(x, stream);
    dw.upload(w, stream);

    if (gemm_bf16(blas, out_a.get(), dx.get(), dw.get(), kTokens, kIntermediate, kHidden) != 0) {
        printf("region_cases: gemm_bf16: FAIL: prefill GEMM returned an error\n");
        return false;
    }
    for (int t = 0; t < kTokens; ++t) {
        if (gemm_bf16(blas, out_b.get() + (size_t)t * kIntermediate, dx.get() + (size_t)t * kHidden,
                      dw.get(), 1, kIntermediate, kHidden) != 0) {
            printf("region_cases: gemm_bf16: FAIL: decode GEMM returned an error\n");
            return false;
        }
    }
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("gemm_bf16", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d, nullptr);
}

/* GEMM, FP32 output (the LM head projection). The released engine enters this
 * with M=1 in both cases, so the fixture exercises the region's shape question
 * (claim D) rather than a case difference the engine shows today. */
bool case_gemm_fp32_lmhead(cublasHandle_t blas, cudaStream_t stream) {
    std::vector<Bf16> x(kTokens * kHidden), w(kVocab * kHidden);
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16(sample(i, 215) * 0.5f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = bf16(sample(i, 216) * 0.25f);
    DeviceBuffer<Bf16> dx(x.size()), dw(w.size());
    DeviceBuffer<float> out_a((size_t)kTokens * kVocab);
    DeviceBuffer<float> out_b((size_t)kTokens * kVocab);
    dx.upload(x, stream);
    dw.upload(w, stream);

    if (gemm_bf16_f32out(blas, out_a.get(), dx.get(), dw.get(), kTokens, kVocab, kHidden) != 0) {
        printf("region_cases: gemm_fp32_lmhead: FAIL: M=tokens GEMM returned an error\n");
        return false;
    }
    for (int t = 0; t < kTokens; ++t) {
        if (gemm_bf16_f32out(blas, out_b.get() + (size_t)t * kVocab,
                             dx.get() + (size_t)t * kHidden, dw.get(), 1, kVocab, kHidden) != 0) {
            printf("region_cases: gemm_fp32_lmhead: FAIL: M=1 GEMM returned an error\n");
            return false;
        }
    }
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("gemm_fp32_lmhead", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d,
                      nullptr);
}

/* Residual add: dst += src, elementwise. The residual stream is the persistent
 * state, so the two arms must also leave it identical. */
bool case_residual_add(cudaStream_t stream) {
    const int n = kTokens * kHidden;
    std::vector<Bf16> dst(n), src(n);
    for (int i = 0; i < n; ++i) {
        dst[i] = bf16(sample(i, 217));
        src[i] = bf16(sample(i, 218) * 0.5f);
    }
    DeviceBuffer<Bf16> da(dst.size()), db(dst.size());
    DeviceBuffer<Bf16> sa(src.size()), sb(src.size());
    da.upload(dst, stream);
    db.upload(dst, stream);
    sa.upload(src, stream);
    sb.upload(src, stream);

    kernel_residual_add(da.get(), sa.get(), n, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_residual_add(db.get() + (size_t)t * kHidden, sb.get() + (size_t)t * kHidden, kHidden,
                            stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(da.download(stream), db.download(stream));
    return adjudicate("residual_add", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d,
                      nullptr);
}

/* Dense MLP SiLU-multiply. The region's contract *requires* one buffer holding
 * [gate[n], up[n]], so a per-token call needs a per-token contiguous pair: the
 * whole-batch buffer interleaves gate and up per token, which the region
 * deliberately refuses. Each token's two slices are gathered into a scratch pair
 * before the call. */
bool case_silu_mul(cudaStream_t stream) {
    const int n = kTokens * kIntermediate;
    const size_t chunk = kIntermediate;
    std::vector<Bf16> buf(2 * (size_t)n);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = bf16(3.0f * sample(i, 219));
    DeviceBuffer<Bf16> d2(2 * (size_t)n);
    DeviceBuffer<Bf16> out_a(n), out_b(n);
    DeviceBuffer<Bf16> pair(2 * chunk);
    d2.upload(buf, stream);
    kernel_silu_mul(out_a.get(), d2.get(), d2.get() + n, n, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        const size_t offset = (size_t)t * chunk;
        CUDA_CHECK(cudaMemcpyAsync(pair.get(), d2.get() + offset, chunk * sizeof(Bf16),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(pair.get() + chunk, d2.get() + n + offset,
                                   chunk * sizeof(Bf16), cudaMemcpyDeviceToDevice, stream));
        kernel_silu_mul(out_b.get() + offset, pair.get(), pair.get() + chunk, (int)chunk, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("silu_mul", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_DECODE, d, nullptr);
}

/* GDN conv activation: elementwise, in place. */
bool case_conv_silu(cudaStream_t stream) {
    const int n = kTokens * kConvDim;
    std::vector<Bf16> x(n);
    for (int i = 0; i < n; ++i) x[i] = bf16(3.0f * sample(i, 220));
    DeviceBuffer<Bf16> da(x.size()), db(x.size());
    da.upload(x, stream);
    db.upload(x, stream);
    kernel_silu_inplace(da.get(), n, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_silu_inplace(db.get() + (size_t)t * kConvDim, kConvDim, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(da.download(stream), db.download(stream));
    return adjudicate("conv_silu", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, d,
                      nullptr);
}

/* GDN causal conv1d: the chunked call versus one token at a time, both carrying
 * the same shift register. The final register is part of the comparison. */
bool case_gdn_conv1d(cudaStream_t stream) {
    std::vector<Bf16> x((size_t)kTokens * kConvDim), w((size_t)kConvDim * kConvWidth), bias(kConvDim);
    for (size_t i = 0; i < x.size(); ++i) {
        // Dyadic values keep the four-term sum exact, so a bitwise difference
        // cannot come from FMA order.
        x[i] = bf16(static_cast<int>(sample(i, 221) * 16) / 32.0f);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = bf16(static_cast<int>(sample(i, 222) * 16) / 32.0f);
    }
    for (int i = 0; i < kConvDim; ++i) {
        bias[i] = bf16(static_cast<int>(sample(i, 223) * 8) / 32.0f);
    }
    std::vector<Bf16> state((size_t)kConvDim * (kConvWidth - 1));
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = bf16(static_cast<int>(sample(i, 224) * 16) / 16.0f);
    }
    DeviceBuffer<Bf16> dx(x.size()), dw(w.size()), dbias(bias.size());
    DeviceBuffer<Bf16> state_a(state.size()), state_b(state.size());
    DeviceBuffer<Bf16> out_a(x.size()), out_b(x.size());
    dx.upload(x, stream);
    dw.upload(w, stream);
    dbias.upload(bias, stream);
    state_a.upload(state, stream);
    state_b.upload(state, stream);

    kernel_causal_conv1d(out_a.get(), dx.get(), dw.get(), dbias.get(), state_a.get(), kConvDim,
                         kTokens, kConvWidth, stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        kernel_causal_conv1d(out_b.get() + (size_t)t * kConvDim, dx.get() + (size_t)t * kConvDim,
                             dw.get(), dbias.get(), state_b.get(), kConvDim, 1, kConvWidth,
                             stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    const Delta s = difference(state_a.download(stream), state_b.download(stream));
    return adjudicate("gdn_conv1d", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL, d,
                      &s);
}

/* GDN core. A scratch allocator for the FLA workspace: the harness needs a fresh
 * poison pattern per call, and the region is the only thing that reads it. */
class FlaScratch {
public:
    explicit FlaScratch(int tokens)
        : size_(kernel_fla_workspace_size(tokens, kValueHeads)),
          buffer_(static_cast<unsigned char *>(nullptr)) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_), size_));
    }
    ~FlaScratch() { CUDA_CHECK(cudaFree(buffer_)); }
    FlaScratch(const FlaScratch &) = delete;
    FlaScratch &operator=(const FlaScratch &) = delete;
    void *fresh(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(buffer_, 0xff, size_, stream));
        return buffer_;
    }

private:
    size_t size_;
    unsigned char *buffer_;
};

bool case_gdn_core(cudaStream_t stream) {
    std::vector<Bf16> conv_out((size_t)kTokens * kConvDim), a((size_t)kTokens * kValueHeads),
        b((size_t)kTokens * kValueHeads), a_log(kValueHeads), dt_bias(kValueHeads);
    for (size_t i = 0; i < conv_out.size(); ++i) conv_out[i] = bf16(sample(i, 225) * 0.5f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = bf16(0.25f + 0.5f * std::abs(sample(i, 226)));
    for (size_t i = 0; i < b.size(); ++i) b[i] = bf16(0.5f + 0.5f * sample(i, 227));
    for (int h = 0; h < kValueHeads; ++h) {
        a_log[h] = bf16(-1.0f - 0.01f * h);
        dt_bias[h] = bf16(0.1f * h);
    }
    DeviceBuffer<Bf16> dconv(conv_out.size()), da(a.size()), db(b.size());
    DeviceBuffer<Bf16> dalog(a_log.size()), ddt(dt_bias.size());
    DeviceBuffer<float> state_a(kStateElems), state_b(kStateElems), state_c(kStateElems);
    DeviceBuffer<Bf16> out_a((size_t)kTokens * kVdim), out_b((size_t)kTokens * kVdim);
    DeviceBuffer<Bf16> out_c((size_t)kTokens * kVdim);
    dconv.upload(conv_out, stream);
    da.upload(a, stream);
    db.upload(b, stream);
    dalog.upload(a_log, stream);
    ddt.upload(dt_bias, stream);
    state_a.upload(std::vector<float>(kStateElems, 0.0f), stream);
    state_b.upload(std::vector<float>(kStateElems, 0.0f), stream);
    state_c.upload(std::vector<float>(kStateElems, 0.0f), stream);
    out_a.upload(poison((size_t)kTokens * kVdim), stream);
    out_b.upload(poison((size_t)kTokens * kVdim), stream);
    out_c.upload(poison((size_t)kTokens * kVdim), stream);

    FlaScratch scratch_a(kTokens), scratch_b(1), scratch_c(kTokens), scratch_d(1);

    /* Arm A: the whole chunk in one call. */
    kernel_fla_gdn(out_a.get(), dconv.get(), da.get(), db.get(), dalog.get(), ddt.get(),
                   state_a.get(), scratch_a.fresh(stream), kTokens, kKeyHeads, kValueHeads,
                   stream, nullptr);
    CUDA_CHECK(cudaGetLastError());

    /* Arm B: one token at a time (the recurrent path), state carried. */
    for (int t = 0; t < kTokens; ++t) {
        kernel_fla_gdn(out_b.get() + (size_t)t * kVdim, dconv.get() + (size_t)t * kConvDim,
                       da.get() + (size_t)t * kValueHeads, db.get() + (size_t)t * kValueHeads,
                       dalog.get(), ddt.get(), state_b.get(), scratch_b.fresh(stream), 1,
                       kKeyHeads, kValueHeads, stream, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());

    /* Arm C: a 7-token chunk then a one-token tail. */
    kernel_fla_gdn(out_c.get(), dconv.get(), da.get(), db.get(), dalog.get(), ddt.get(),
                   state_c.get(), scratch_c.fresh(stream), kTokens - 1, kKeyHeads, kValueHeads,
                   stream, nullptr);
    kernel_fla_gdn(out_c.get() + (size_t)(kTokens - 1) * kVdim,
                   dconv.get() + (size_t)(kTokens - 1) * kConvDim,
                   da.get() + (size_t)(kTokens - 1) * kValueHeads,
                   db.get() + (size_t)(kTokens - 1) * kValueHeads, dalog.get(), ddt.get(),
                   state_c.get(), scratch_d.fresh(stream), 1, kKeyHeads, kValueHeads, stream,
                   nullptr);
    CUDA_CHECK(cudaGetLastError());

    const std::vector<Bf16> a_out = out_a.download(stream);
    const std::vector<float> state_a_host = state_a.download(stream);
    const Delta rec = difference(a_out, out_b.download(stream));
    const Delta rec_state = difference(state_a_host, state_b.download(stream));
    const Delta tail = difference(a_out, out_c.download(stream));
    const Delta tail_state = difference(state_a_host, state_c.download(stream));
    bool ok = adjudicate("gdn_core", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_RECURRENT_PREFILL,
                         rec, &rec_state);
    ok &= adjudicate("gdn_core", REGION_CASE_CHUNKED_PREFILL, REGION_CASE_TAIL1, tail,
                     &tail_state);
    return ok;
}

/* GDN gated norm: per-row reduction over head_dim, row count varies. */
bool case_gdn_gated_norm(cudaStream_t stream) {
    const int rows = kTokens * kValueHeads;
    std::vector<Bf16> x((size_t)rows * kGdnHeadDim), z(x.size());
    std::vector<float> weight(kGdnHeadDim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf16(sample(i, 228));
    for (size_t i = 0; i < z.size(); ++i) z[i] = bf16(4.0f * sample(i, 229));
    for (int d = 0; d < kGdnHeadDim; ++d) weight[d] = round_bf16(0.5f + (d % 31) / 32.0f);
    DeviceBuffer<Bf16> dx(x.size()), dz(z.size());
    DeviceBuffer<float> dw(weight.size());
    DeviceBuffer<Bf16> out_a(x.size()), out_b(x.size());
    dx.upload(x, stream);
    dz.upload(z, stream);
    dw.upload(weight, stream);

    kernel_gdn_gated_norm(out_a.get(), dx.get(), dz.get(), dw.get(), kGdnHeadDim, rows, 1e-6f,
                          stream);
    CUDA_CHECK(cudaGetLastError());
    for (int t = 0; t < kTokens; ++t) {
        const size_t offset = (size_t)t * kValueHeads * kGdnHeadDim;
        kernel_gdn_gated_norm(out_b.get() + offset, dx.get() + offset, dz.get() + offset, dw.get(),
                              kGdnHeadDim, kValueHeads, 1e-6f, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const Delta d = difference(out_a.download(stream), out_b.download(stream));
    return adjudicate("gdn_gated_norm", REGION_CASE_CHUNKED_PREFILL,
                      REGION_CASE_RECURRENT_PREFILL, d, nullptr);
}

/* ---------------------------------------------------------------- */
/* Unsupported shapes and cases                                     */
/* ---------------------------------------------------------------- */

template <typename Fn>
bool expect_rejected(const char *label, Fn fn) {
    try {
        fn();
    } catch (const std::exception &error) {
        printf("region_cases: rejected/%s: %s\n", label, error.what());
        return true;
    }
    printf("region_cases: rejected/%s: FAIL: an unsupported shape or case was accepted\n", label);
    return false;
}

bool unsupported(cudaStream_t stream) {
    bool ok = true;
    ModelDims dims{};
    dims.hidden_size = kHidden;
    dims.num_heads = kHeads;
    dims.num_kv_heads = kKvHeads;
    dims.head_dim = kHeadDim;
    dims.rotary_dim = kRotaryDim;
    dims.intermediate_size = kIntermediate;
    dims.vocab_size = kVocab;
    dims.max_seq_len = kMaxSeq;
    dims.rms_eps = 1e-6f;
    dims.rope_theta = 1e7f;
    dims.max_chunk = 4;
    dims.gdn_conv_dim = kConvDim;
    dims.gdn_value_dim = kVdim;
    dims.gdn_num_v_heads = kValueHeads;
    dims.gdn_num_k_heads = kKeyHeads;
    dims.gdn_head_dim = kGdnHeadDim;
    dims.gdn_conv_kernel = kConvWidth;

    /* A chunk longer than the descriptor's max_chunk. */
    ok &= expect_rejected("workspace/tokens_above_max_chunk",
                          [&] { (void)layer_workspace_size(kTokens, &dims); });

    /* A sequence longer than max_seq_len. This needs its own dims: the shared
     * `dims` above has a deliberately small max_chunk, and the token-count check
     * runs first, so the same fixture would test the wrong bound. */
    ModelDims attn_dims = dims;
    attn_dims.max_chunk = 128;
    std::vector<Bf16> residual(kTokens * kHidden), ws(1);
    std::vector<int64_t> positions(kTokens, 0);
    DeviceBuffer<Bf16> dres(residual.size()), dout(residual.size());
    DeviceBuffer<int64_t> dpos(positions.size());
    dres.upload(residual, stream);
    dpos.upload(positions, stream);
    AttentionWeights weights{};
    ok &= expect_rejected("attention/seq_len_above_max_seq_len", [&] {
        (void)forward_attention_layer(nullptr, stream, dres.get(), ws.data(), dout.get(),
                                      &weights, nullptr, dpos.get(), kTokens, kTokens + kMaxSeq,
                                      &attn_dims);
    });

    /* SiLU-multiply requires the up matrix to follow the gate matrix exactly. */
    DeviceBuffer<Bf16> dgate(16), dup(16), dsilu(16);
    ok &= expect_rejected("silu_mul/non_contiguous_pair",
                          [&] { kernel_silu_mul(dsilu.get(), dgate.get(), dup.get(), 16, stream); });

    /* The FLA entry point supports exactly 16 key heads. */
    DeviceBuffer<Bf16> dfla_out(1), dfla_in(1), dfla_ab(1);
    DeviceBuffer<float> dstate(1);
    ok &= expect_rejected("gdn_core/wrong_key_head_count", [&] {
        kernel_fla_gdn(dfla_out.get(), dfla_in.get(), dfla_ab.get(), dfla_ab.get(), dfla_ab.get(),
                       dfla_ab.get(), dstate.get(), dfla_in.get(), 1, 8, kValueHeads, stream,
                       nullptr);
    });
    ok &= expect_rejected("gdn_core/tokens_above_128",
                          [&] { (void)kernel_fla_workspace_size(200, kValueHeads); });
    ok &= expect_rejected("conv_silu/negative_count",
                          [&] { kernel_silu_inplace(nullptr, -1, stream); });
    ok &= expect_rejected("q_gate_split/zero_head_dim", [&] {
        kernel_q_gate_split(nullptr, nullptr, nullptr, 4, 0, stream);
    });

    /* The trainer traversal does not exist, so no region may offer it. */
    int count = 0;
    const struct RegionInventoryEntry *inventory = region_inventory(&count);
    const char *trainer_cases[] = {REGION_CASE_TRAIN_FORWARD, REGION_CASE_EVAL_NO_AUTOGRAD,
                                   REGION_CASE_RECOMPUTE, REGION_CASE_BACKWARD};
    for (int i = 0; i < count; ++i) {
        for (const char *c : trainer_cases) {
            if (region_case_available(inventory[i].region, c) != 0) {
                printf("region_cases: FAIL: %s advertises the unavailable case %s\n",
                       inventory[i].region, c);
                ok = false;
            }
        }
    }
    printf("region_cases: unsupported cases are unavailable from all %d inventoried regions\n",
           count);

    /* A region that is registered as not_applicable must not be runnable. */
    const struct RegionCasePair *na =
        region_find_pair("backward", REGION_CASE_TRAIN_FORWARD, REGION_CASE_BACKWARD);
    if (na == nullptr || strcmp(na->verdict, REGION_VERDICT_NOT_APPLICABLE) != 0) {
        printf("region_cases: FAIL: the backward region is not registered not_applicable\n");
        ok = false;
    }
    return ok;
}

/* ---------------------------------------------------------------- */
/* Region entry-point cost                                          */
/* ---------------------------------------------------------------- */

template <typename Fn>
void measure_region(const char *label, Fn fn, cudaStream_t stream, int repeats) {
    for (int i = 0; i < 4; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float device_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&device_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const auto host_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    const auto host_end = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto drained = std::chrono::steady_clock::now();

    const double enqueue_ns =
        std::chrono::duration<double, std::nano>(host_end - host_begin).count() / repeats;
    const double total_ns = std::chrono::duration<double, std::nano>(drained - host_begin).count() /
                            repeats;
    printf("region_ffi: %-22s host_enqueue_ns=%.1f per_call_total_us=%.3f device_us=%.3f "
           "host_share=%.1f%%\n",
           label, enqueue_ns, total_ns / 1000.0, device_ms * 1000.0,
           100.0 * enqueue_ns / total_ns);
}

bool measure_entry_points(cublasHandle_t blas, cudaStream_t stream) {
    constexpr int kRepeats = 200;
    std::vector<Bf16> x((size_t)kTokens * kHidden, bf16(0.5f));
    std::vector<Bf16> w((size_t)kHidden * kIntermediate, bf16(0.25f));
    std::vector<int64_t> token_ids(kTokens, 1);
    DeviceBuffer<Bf16> dx(x.size()), dw(w.size());
    DeviceBuffer<Bf16> dout(x.size()), daux((size_t)kTokens * kIntermediate);
    // The dense MLP SiLU-multiply region takes one buffer holding [gate[n], up[n]].
    const int silu_n = kTokens * kIntermediate;
    DeviceBuffer<Bf16> dpair(2 * (size_t)silu_n);
    DeviceBuffer<Bf16> dtable((size_t)kVocab * kHidden);
    DeviceBuffer<int64_t> dids(kTokens);
    dx.upload(x, stream);
    dw.upload(w, stream);
    dids.upload(token_ids, stream);
    dtable.upload(std::vector<Bf16>((size_t)kVocab * kHidden, bf16(0.1f)), stream);
    dpair.upload(std::vector<Bf16>(2 * (size_t)silu_n, bf16(0.25f)), stream);

    measure_region("embedding", [&] { kernel_embedding(dout.get(), dtable.get(), dids.get(),
                                                       kHidden, kTokens, stream); },
                   stream, kRepeats);
    measure_region("rmsnorm",
                   [&] { kernel_gemma_rms_norm(dout.get(), dx.get(), dx.get(), kHidden, kTokens,
                                               1e-6f, stream); },
                   stream, kRepeats);
    measure_region("residual_add",
                   [&] { kernel_residual_add(dout.get(), dx.get(), kTokens * kHidden, stream); },
                   stream, kRepeats);
    measure_region("silu_mul",
                   [&] { kernel_silu_mul(dout.get(), dpair.get(), dpair.get() + silu_n, silu_n,
                                        stream); },
                   stream, kRepeats);
    measure_region("gemm_bf16", [&] { (void)gemm_bf16(blas, daux.get(), dx.get(), dw.get(),
                                                     kTokens, kIntermediate, kHidden); },
                   stream, kRepeats);
    measure_region("conv_silu",
                   [&] { kernel_silu_inplace(dout.get(), kTokens * kHidden, stream); }, stream,
                   kRepeats);

    /* Two host threads, one handle and one stream each: this is what a region
     * boundary costs per thread when two callers drive different regions. The
     * engine's regions share a workspace today, so this measures the host-side
     * call, not concurrent execution of the same region. */
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const int per_thread = kRepeats;
    double thread_ns[2] = {0.0, 0.0};
    auto worker = [&](int slot) {
        cudaSetDevice(device);
        cublasHandle_t handle = nullptr;
        cudaStream_t own = nullptr;
        cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, own);
        Bf16 *x_ptr = dx.get(), *w_ptr = dw.get(), *out_ptr = daux.get();
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < per_thread; ++i) {
            (void)gemm_bf16(handle, out_ptr, x_ptr, w_ptr, kTokens, kIntermediate, kHidden);
        }
        const auto end = std::chrono::steady_clock::now();
        cudaStreamSynchronize(own);
        thread_ns[slot] = std::chrono::duration<double, std::nano>(end - begin).count() / per_thread;
        cublasDestroy(handle);
        cudaStreamDestroy(own);
    };
    std::thread t0(worker, 0), t1(worker, 1);
    t0.join();
    t1.join();
    CUDA_CHECK(cudaGetLastError());
    printf("region_ffi: gemm_bf16 two-thread host_enqueue_ns=%.1f / %.1f (one handle and one "
           "non-blocking stream per thread; the engine's regions are single-threaded today)\n",
           thread_ns[0], thread_ns[1]);
    printf("region_ffi: the engine's Haskell FFI is model-level (engine_create/prefill/decode), so "
           "these are C region entry-point costs (validation + enqueue), not a Haskell ccall; "
           "Stage 3 introduces region handles\n");
    return true;
}

/* ---------------------------------------------------------------- */
/* Coverage check                                                   */
/* ---------------------------------------------------------------- */

/* Every registered pair the harness is supposed to run must have run: nothing may
 * be skipped silently. */
bool coverage() {
    bool ok = true;
    int count = 0;
    const struct RegionInventoryEntry *inventory = region_inventory(&count);
    int expected = 0;
    for (int i = 0; i < count; ++i) {
        const struct RegionInventoryEntry *entry = &inventory[i];
        for (int j = 0; j < entry->pair_count; ++j) {
            const struct RegionCasePair *pair = &entry->pairs[j];
            const bool gated = strcmp(pair->verdict, REGION_VERDICT_EXACT) == 0 ||
                               strcmp(pair->verdict, REGION_VERDICT_UNVERIFIED) == 0 ||
                               strcmp(pair->verdict, REGION_VERDICT_EXCEPTION) == 0;
            if (!gated) continue;
            ++expected;
            if (!was_executed(entry->region, pair->left, pair->right)) {
                printf("region_cases: FAIL: %s/%s/%s is registered but no fixture ran it\n",
                       entry->region, pair->left, pair->right);
                ok = false;
            }
        }
    }
    printf("region_cases: all %d registered exact/unverified/exception pairs ran "
           "(%zu comparisons recorded)\n", expected, g_executed.size());
    return ok;
}

}  // namespace

int main() {
    // Line-buffered: the evidence lines have to survive an abort in a later step.
    // A block-buffered stdout is discarded by abort(), which turns a diagnosable
    // failure into "Subprocess aborted" with no output at all.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Stream stream;
    cudaStream_t s = stream.get();
    struct BlasHandle {
        cublasHandle_t handle = nullptr;
        BlasHandle() {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "region_cases: cublasCreate failed\n");
                exit(EXIT_FAILURE);
            }
        }
        ~BlasHandle() { cublasDestroy(handle); }
        BlasHandle(const BlasHandle &) = delete;
        BlasHandle &operator=(const BlasHandle &) = delete;
    } blas;
    if (cublasSetStream(blas.handle, s) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "region_cases: cublasSetStream failed\n");
        return EXIT_FAILURE;
    }

    bool ok = true;
    ok &= case_embedding(s);
    ok &= case_rmsnorm(s);
    ok &= case_per_head_norm(s);
    ok &= case_rope(s);
    ok &= case_q_gate_split(s);
    ok &= case_kv_write(s);
    ok &= case_attention(s);
    ok &= case_attention_output_gate(s);
    ok &= case_gemm_bf16(blas.handle, s);
    ok &= case_gemm_fp32_lmhead(blas.handle, s);
    ok &= case_residual_add(s);
    ok &= case_silu_mul(s);
    ok &= case_conv_silu(s);
    ok &= case_gdn_conv1d(s);
    ok &= case_gdn_core(s);
    ok &= case_gdn_gated_norm(s);
    ok &= unsupported(s);
    ok &= measure_entry_points(blas.handle, s);
    ok &= coverage();
    return finish("test_region_cases", ok);
}
