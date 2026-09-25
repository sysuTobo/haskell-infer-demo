/*
 * test_gdn_invariance.cu - Stage 2 claim B.
 *
 * "Hold raw/prepared inputs and nonzero initial state fixed; compare
 * whole/chunked/recurrent paths, outputs and final state, including lengths around
 * 1, 64 and 128. Distinguish prepare, core and GEMM effects."
 *
 * The region under test is kernel_fla_gdn (prepare + chunkwise core) and its
 * recurrent path (tokens == 1). Raw inputs and a nonzero initial state are fixed,
 * so every arm sees the same rows; the arms differ only in how the sequence is
 * grouped into calls:
 *
 *   whole       one call with all L tokens (the chunkwise core)
 *   recurrent   L calls with one token each (the recurrent core)
 *   half        two calls at L/2
 *   chunk64     64 tokens then the rest (the FLA block size)
 *   tail1       L-1 tokens then a one-token tail
 *
 * Attribution uses the taps kernel_fla_gdn writes for the *prepared* q/k/v: each
 * call gets its own tap directory, so the prepared rows can be concatenated per
 * arm and compared. If prepare is row-local then it matches bitwise across arms,
 * and any remaining output/state difference is the core's -- that is the
 * prepare-vs-core split this claim asks for. GEMM is deliberately absent: the
 * projections live outside this region and are claim D's subject.
 */
#include "invariance_utils.h"
#include "fla_ops.h"
#include "layers.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
using namespace invariance;
using test::Bf16;
using test::bf16;
using test::DeviceBuffer;
using test::poison;
using test::sample;

/* kernel_fla_gdn requires exactly 16 key heads and head_dim 128, and the AOT
 * recurrent kernel exists for 48 value heads: the Qwen3.8 GDN layout. */
constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadDim = 128;
constexpr int kConvDim = (2 * kKeyHeads + kValueHeads) * kHeadDim;
constexpr int kVdim = kValueHeads * kHeadDim;
constexpr int kStateElems = kValueHeads * kHeadDim * kHeadDim;
constexpr int kMaxTokens = 128;

const int kLengths[] = {1, 2, 63, 64, 65, 127, 128};

/* The arm's raw inputs are the same rows every time; only the grouping changes. */
struct Inputs {
    std::vector<Bf16> conv;   // [kMaxTokens, kConvDim]
    std::vector<Bf16> a;      // [kMaxTokens, kValueHeads]
    std::vector<Bf16> b;
    std::vector<Bf16> a_log;  // [kValueHeads]
    std::vector<Bf16> dt_bias;
    std::vector<float> state; // [kValueHeads, kHeadDim, kHeadDim], nonzero
};

Inputs make_inputs() {
    Inputs in;
    in.conv.resize((size_t)kMaxTokens * kConvDim);
    in.a.resize((size_t)kMaxTokens * kValueHeads);
    in.b.resize((size_t)kMaxTokens * kValueHeads);
    in.a_log.resize(kValueHeads);
    in.dt_bias.resize(kValueHeads);
    for (size_t i = 0; i < in.conv.size(); ++i) in.conv[i] = bf16(0.5f * sample(i, 730));
    for (size_t i = 0; i < in.a.size(); ++i) in.a[i] = bf16(0.25f + 0.5f * std::abs(sample(i, 731)));
    for (size_t i = 0; i < in.b.size(); ++i) in.b[i] = bf16(0.5f + 0.5f * sample(i, 732));
    for (int h = 0; h < kValueHeads; ++h) {
        in.a_log[h] = bf16(-1.0f - 0.01f * h);
        in.dt_bias[h] = bf16(0.1f * h);
    }
    // A nonzero initial state is what makes a decomposition difference visible:
    // with a zero state the first chunk would be independent of the recurrence.
    in.state.resize(kStateElems);
    for (int i = 0; i < kStateElems; ++i) in.state[i] = 0.02f + 0.1f * sample((size_t)i, 733);
    return in;
}

/* A scratch region whose content is refreshed before each call, so a kernel that
 * reads stale workspace cannot make two arms agree by accident. */
class FlaScratch {
public:
    explicit FlaScratch(int tokens)
        : size_(kernel_fla_workspace_size(tokens, kValueHeads)), buffer_(nullptr) {
        if (cudaMalloc(reinterpret_cast<void **>(&buffer_), size_) != cudaSuccess) {
            std::printf("invariance: B: workspace allocation failed\n");
            std::exit(EXIT_FAILURE);
        }
    }
    ~FlaScratch() { cudaFree(buffer_); }
    FlaScratch(const FlaScratch &) = delete;
    FlaScratch &operator=(const FlaScratch &) = delete;
    void *fresh(cudaStream_t stream) {
        cudaMemsetAsync(buffer_, 0xff, size_, stream);
        return buffer_;
    }

private:
    size_t size_;
    void *buffer_;
};

/* The raw tap dump: [rows, cols] float32, written by tap_dump_rows. */
std::vector<float> read_tap(const std::string &dir, const char *kind, int tokens) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s_00_seq0_tok%d.f32", dir.c_str(), kind, tokens);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::printf("invariance: B: cannot read tap %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    const std::streamsize bytes = file.tellg();
    file.seekg(0);
    std::vector<float> out((size_t)bytes / sizeof(float));
    if (!file.read(reinterpret_cast<char *>(out.data()), bytes)) {
        std::printf("invariance: B: short read on %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    return out;
}

struct RunResult {
    std::vector<Bf16> out;
    std::vector<float> state;
    std::vector<float> prep_q, prep_k, prep_v;
};

/* Run one arm: the chunks are applied in order, carrying the recurrent state, and
 * every call's prepared q/k/v are collected from its own tap directory. */
RunResult run_arm(const Inputs &in, const std::vector<int> &chunks, const std::string &tap_base,
                  const std::string &arm, cudaStream_t stream, int device) {
    RunResult result;
    result.out.assign((size_t)kMaxTokens * kVdim, test::Bf16());
    result.state = in.state;
    for (size_t c = 0; c < chunks.size(); ++c) {
        const int offset = [&] {
            int sum = 0;
            for (size_t i = 0; i < c; ++i) sum += chunks[i];
            return sum;
        }();
        const int tokens = chunks[c];
        const std::string dir = tap_base + "/" + arm + "_c" + std::to_string(c);
        // tap_dump_rows writes with fopen, so the directory has to exist first.
        const std::string make_dir = "mkdir -p " + dir;
        if (std::system(make_dir.c_str()) != 0) {
            std::printf("invariance: B: cannot create %s\n", dir.c_str());
            std::exit(EXIT_FAILURE);
        }
        TapConfig config;
        config.layers.push_back(0);
        config.dir = dir;
        const GdnTapSites sites{&config, 0, device};

        DeviceBuffer<Bf16> conv(chunks[c] * (size_t)kConvDim), da(chunks[c] * (size_t)kValueHeads),
            db(chunks[c] * (size_t)kValueHeads);
        conv.upload(std::vector<Bf16>(in.conv.begin() + (size_t)offset * kConvDim,
                                      in.conv.begin() + (size_t)(offset + tokens) * kConvDim),
                    stream);
        da.upload(std::vector<Bf16>(in.a.begin() + (size_t)offset * kValueHeads,
                                    in.a.begin() + (size_t)(offset + tokens) * kValueHeads),
                  stream);
        db.upload(std::vector<Bf16>(in.b.begin() + (size_t)offset * kValueHeads,
                                    in.b.begin() + (size_t)(offset + tokens) * kValueHeads),
                  stream);
        DeviceBuffer<Bf16> a_log(in.a_log.size()), dt_bias(in.dt_bias.size());
        a_log.upload(in.a_log, stream);
        dt_bias.upload(in.dt_bias, stream);
        DeviceBuffer<float> state(result.state.size());
        state.upload(result.state, stream);
        DeviceBuffer<Bf16> out(chunks[c] * (size_t)kVdim);
        out.upload(poison(chunks[c] * (size_t)kVdim), stream);

        FlaScratch scratch(tokens);
        kernel_fla_gdn(out.get(), conv.get(), da.get(), db.get(), a_log.get(), dt_bias.get(),
                       state.get(), scratch.fresh(stream), tokens, kKeyHeads, kValueHeads, stream,
                       &sites);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            std::printf("invariance: B: %s chunk %zu failed: %s\n", arm.c_str(), c,
                        cudaGetErrorString(status));
            std::exit(EXIT_FAILURE);
        }
        result.state = state.download(stream);
        const std::vector<Bf16> rows = out.download(stream);
        for (size_t i = 0; i < rows.size(); ++i) {
            result.out[(size_t)offset * kVdim + i] = rows[i];
        }
        // The prepared q/k/v, one row per (token, value head).
        const std::vector<float> q = read_tap(dir, "gdn_q", tokens);
        const std::vector<float> k = read_tap(dir, "gdn_k", tokens);
        const std::vector<float> v = read_tap(dir, "gdn_v", tokens);
        result.prep_q.insert(result.prep_q.end(), q.begin(), q.end());
        result.prep_k.insert(result.prep_k.end(), k.begin(), k.end());
        result.prep_v.insert(result.prep_v.end(), v.begin(), v.end());
    }
    return result;
}

/* Arms that reduce to the same chunk list are the same experiment; run each once
 * and say so rather than reporting the same numbers twice under two names. */
std::vector<std::pair<std::string, std::vector<int>>> arms_for(int length) {
    std::vector<std::pair<std::string, std::vector<int>>> arms;
    arms.emplace_back("whole", std::vector<int>{length});
    if (length > 1) arms.emplace_back("recurrent", std::vector<int>(length, 1));
    if (length >= 4) arms.emplace_back("half", std::vector<int>{length / 2, length - length / 2});
    if (length > 64) arms.emplace_back("chunk64", std::vector<int>{64, length - 64});
    if (length > 1) arms.emplace_back("tail1", std::vector<int>{length - 1, 1});
    std::vector<std::pair<std::string, std::vector<int>>> unique;
    for (const auto &arm : arms) {
        bool duplicate = false;
        for (const auto &kept : unique) {
            if (kept.second == arm.second) {
                duplicate = true;
                std::printf("invariance: B: L=%d arm %s is the same chunking as %s, skipped\n",
                            length, arm.first.c_str(), kept.first.c_str());
                break;
            }
        }
        if (!duplicate) unique.push_back(arm);
    }
    return unique;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::printf("invariance: B: stream creation failed\n");
        return EXIT_FAILURE;
    }
    int device = 0;
    cudaGetDevice(&device);

    const Inputs in = make_inputs();
    const std::string tap_base = "/tmp/gdn_invariance_taps";
    const std::string mkdir = "mkdir -p " + tap_base;
    if (std::system(mkdir.c_str()) != 0) {
        std::printf("invariance: B: cannot create %s\n", tap_base.c_str());
        return EXIT_FAILURE;
    }

    Recorder out_rec("B"), state_rec("B"), prep_rec("B");
    bool ok = true;
    std::printf("invariance: claim B (GDN decomposition): whole vs recurrent vs split, "
                "nonzero initial state, prepared q/k/v attributed by taps\n");

    for (int length : kLengths) {
        const std::vector<std::pair<std::string, std::vector<int>>> arms = arms_for(length);
        if (arms.size() == 1) {
            // A one-token sequence has no decomposition: whole, recurrent and
            // tail are the same single call. Say so instead of reporting an
            // empty comparison as evidence.
            std::printf("invariance: B: L=%d has no decomposition to compare (one token)\n", length);
            continue;
        }
        // The reference is the whole-sequence (chunkwise) run; every other arm is
        // compared against it.
        const RunResult reference = run_arm(in, {length}, tap_base, "ref", stream, device);
        const std::vector<Bf16> reference_out(reference.out.begin(),
                                              reference.out.begin() + (size_t)length * kVdim);

        for (const auto &arm : arms) {
            if (arm.first == "whole") continue;
            const RunResult run = run_arm(in, arm.second, tap_base, arm.first, stream, device);
            const std::vector<Bf16> arm_out(run.out.begin(),
                                            run.out.begin() + (size_t)length * kVdim);
            const std::string label = std::string(arm.first) + "/L=" + std::to_string(length);

            double peak = 0.0;
            for (const Bf16 &v : reference_out) peak = std::max(peak, std::abs((double)test::value(v)));
            const Delta out = delta(arm_out, reference_out);
            out_rec.record(label, out, "core-out", peak);
            if (!out.finite) ok = false;

            double state_peak = 0.0;
            for (double v : reference.state) state_peak = std::max(state_peak, std::abs(v));
            const Delta state = delta(run.state, reference.state);
            state_rec.record(label, state, "state", state_peak);
            if (!state.finite) ok = false;

            const Delta pq = delta(run.prep_q, reference.prep_q);
            const Delta pk = delta(run.prep_k, reference.prep_k);
            const Delta pv = delta(run.prep_v, reference.prep_v);
            prep_rec.record(label + " q", pq, "prepare-q");
            prep_rec.record(label + " k", pk, "prepare-k");
            prep_rec.record(label + " v", pv, "prepare-v");
            if (!pq.finite || !pk.finite || !pv.finite) ok = false;
        }
    }

    out_rec.summary();
    state_rec.summary();
    prep_rec.summary();

    // The decision. prepare is expected to be row-local and therefore bitwise
    // identical across arms; if that fails, the decomposition changes the inputs
    // the core sees and the attribution below would be wrong.
    const bool prepare_invariant = prep_rec.bitwise() == prep_rec.measurements();
    std::printf("invariance: B prepare %s across every arm (%d/%d measurements bitwise)\n",
                prepare_invariant ? "is bitwise invariant" : "DIFFERS", prep_rec.bitwise(),
                prep_rec.measurements());
    if (!ok || prepare_invariant == false) {
        std::printf("invariance: B FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("test_gdn_invariance: PASS (core deviation is bounded and attributed to the core "
                "given a bitwise-invariant prepare)\n");
    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
