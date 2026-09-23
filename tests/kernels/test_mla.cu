/* MLA (DeepSeek-V2 style) chunking self-consistency: the same tokens through
 * one prefill call and through several smaller calls must produce the same
 * layer output. This is what catches kernel defects that only depend on the
 * query-block shape -- the in-block tree reduction used to drop whole lane
 * groups for blockDim 96/160, so any chunk landing in 65..96 or 129..160 tokens
 * silently lost a third of the softmax mass, and every engine-vs-reference run
 * stayed inside the family's BF16 tolerance band because the reference prompt
 * was 5 tokens long. */
#include "layers.h"
#include "test_utils.h"

#include <cublas_v2.h>

#include <cmath>

using test::Bf16;
using test::DeviceBuffer;

namespace {

/* Two chunkings of the same token sequence. `aligned` is what the engine does
 * with max_chunk=128; `split` re-cuts the same work, and each cut must land the
 * query blocks in a different thread-size band than the aligned cut does. */
struct Case {
    int tokens;
    std::vector<int> aligned;
    std::vector<int> split;
};

bool run_case(const Case &c, const ModelDims &dims, const MlaWeights &w,
              cublasHandle_t cublas, test::Stream &stream) {
    const int hidden = dims.hidden_size;
    const int latent_width = dims.mla_kv_lora_rank + dims.mla_qk_rope_head_dim;

    std::vector<Bf16> residual((size_t)c.tokens * hidden);
    for (size_t i = 0; i < residual.size(); ++i) residual[i] = test::bf16(test::sample(i, 23));
    DeviceBuffer<Bf16> d_residual(residual.size());
    d_residual.upload(residual, stream.get());

    const size_t ws_bytes = layer_workspace_size(ENGINE_MAX_CHUNK, &dims);
    DeviceBuffer<Bf16> d_ws(ws_bytes / sizeof(Bf16));
    DeviceBuffer<Bf16> d_out((size_t)ENGINE_MAX_CHUNK * hidden);
    DeviceBuffer<Bf16> d_cache((size_t)dims.max_seq_len * latent_width);
    DeviceBuffer<int64_t> d_pos(ENGINE_MAX_CHUNK);
    const size_t scratch_bytes = kernel_mla_scratch_size(dims.max_seq_len, &dims);
    void *scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));

    auto run = [&](const std::vector<int> &chunks, std::vector<Bf16> &gathered) {
        std::vector<Bf16> zeros((size_t)dims.max_seq_len * latent_width);
        d_cache.upload(zeros, stream.get());
        int start = 0;
        std::vector<int64_t> positions(ENGINE_MAX_CHUNK, 0);
        for (int count : chunks) {
            for (int t = 0; t < count; ++t) positions[t] = start + t;
            d_pos.upload(positions, stream.get());
            const int status = forward_mla_layer(
                cublas, stream.get(), d_residual.get() + (size_t)start * hidden, d_ws.get(),
                d_out.get(), &w, d_cache.get(), scratch, d_pos.get(), count, start + count, &dims);
            if (status != 0) {
                std::fprintf(stderr, "forward_mla_layer failed: %d\n", status);
                return false;
            }
            const auto rows = d_out.download(stream.get());
            std::copy(rows.begin(), rows.begin() + (size_t)count * hidden,
                      gathered.begin() + (size_t)start * hidden);
            start += count;
        }
        return true;
    };

    std::vector<Bf16> aligned((size_t)c.tokens * hidden, test::bf16(0.0f));
    std::vector<Bf16> split((size_t)c.tokens * hidden, test::bf16(0.0f));
    if (!run(c.aligned, aligned) || !run(c.split, split)) return false;
    CUDA_CHECK(cudaFree(scratch));

    double scale = 0.0;
    for (Bf16 x : aligned) scale = std::max(scale, std::abs(test::value(x)));
    char name[128];
    std::snprintf(name, sizeof(name), "mla %d tokens (%zu chunks vs %zu chunks, scale %.3g)",
                  c.tokens, c.aligned.size(), c.split.size(), scale);
    return test::compare(name, split, aligned, 2e-3, 5e-2);
}

}  // namespace

int main() {
    test::Stream stream;
    ModelDims dims{};
    dims.hidden_size = 256;
    dims.num_heads = 8;
    dims.num_layers = 1;
    dims.norm_style = 1; /* plain RMSNorm, as the MLA families use */
    dims.rms_eps = 1e-6f;
    dims.rope_theta = 10000.0f;
    dims.max_seq_len = 160;
    dims.max_chunk = ENGINE_MAX_CHUNK;
    dims.mla_kv_lora_rank = 128;
    dims.mla_qk_nope_head_dim = 128;
    dims.mla_qk_rope_head_dim = 64;
    dims.mla_v_head_dim = 128;

    const int hidden = dims.hidden_size;
    const int heads = dims.num_heads;
    const int q_width = heads * (dims.mla_qk_nope_head_dim + dims.mla_qk_rope_head_dim);
    const int latent_width = dims.mla_kv_lora_rank + dims.mla_qk_rope_head_dim;
    const int kv_row = heads * (dims.mla_qk_nope_head_dim + dims.mla_v_head_dim);
    const float w_scale = 1.0f / std::sqrt((float)hidden);

    std::vector<Bf16> q_proj((size_t)q_width * hidden), kv_a((size_t)latent_width * hidden);
    std::vector<Bf16> kv_a_norm(dims.mla_kv_lora_rank), kv_b((size_t)kv_row * dims.mla_kv_lora_rank);
    std::vector<Bf16> o_proj((size_t)hidden * heads * dims.mla_v_head_dim), input_norm(hidden);
    for (size_t i = 0; i < q_proj.size(); ++i) q_proj[i] = test::bf16(test::sample(i, 3) * w_scale);
    for (size_t i = 0; i < kv_a.size(); ++i) kv_a[i] = test::bf16(test::sample(i, 5) * w_scale);
    for (size_t i = 0; i < kv_a_norm.size(); ++i) kv_a_norm[i] = test::bf16(1.0f + test::sample(i, 7) * 0.1f);
    for (size_t i = 0; i < kv_b.size(); ++i) kv_b[i] = test::bf16(test::sample(i, 11) * w_scale);
    for (size_t i = 0; i < o_proj.size(); ++i) o_proj[i] = test::bf16(test::sample(i, 13) * w_scale);
    for (size_t i = 0; i < input_norm.size(); ++i) input_norm[i] = test::bf16(1.0f + test::sample(i, 17) * 0.1f);

    DeviceBuffer<Bf16> d_q_proj(q_proj.size()), d_kv_a(kv_a.size()), d_kv_a_norm(kv_a_norm.size());
    DeviceBuffer<Bf16> d_kv_b(kv_b.size()), d_o_proj(o_proj.size()), d_input_norm(input_norm.size());
    d_q_proj.upload(q_proj, stream.get());
    d_kv_a.upload(kv_a, stream.get());
    d_kv_a_norm.upload(kv_a_norm, stream.get());
    d_kv_b.upload(kv_b, stream.get());
    d_o_proj.upload(o_proj, stream.get());
    d_input_norm.upload(input_norm, stream.get());
    MlaWeights weights{d_q_proj.get(), d_kv_a.get(), d_kv_a_norm.get(), d_kv_b.get(),
                       d_o_proj.get(), d_input_norm.get()};

    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(cublas, stream.get()) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS handle setup failed\n");
        return EXIT_FAILURE;
    }

    bool ok = true;
    /* 65 and 72 land query blocks in the 65..96 thread band (96 was the broken
     * size), 140 in the 129..160 band (160 was broken); the aligned cuts keep
     * every launch at 64/128/256 threads, which the tree reduction handles. */
    for (const Case &c : {Case{65, {65}, {64, 1}}, Case{72, {72}, {64, 8}},
                          Case{140, {128, 12}, {64, 76}}}) {
        ok &= run_case(c, dims, weights, cublas, stream);
    }
    cublasDestroy(cublas);
    return test::finish("test_mla", ok);
}
