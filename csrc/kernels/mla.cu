/**
 * mla.cu - Multi-head latent attention (DeepSeek-V2 style) forward.
 *
 * The KV cache holds the *latent* per token -- `kv_lora_rank` compressed
 * channels plus the shared RoPE key -- instead of per-head K/V, which is what
 * makes MLA's cache per token small (1152 B for DeepSeek-V2-Lite against 8 KB
 * for the equivalent GQA cache). This first implementation decompresses the
 * latent back to per-head k_nope/v with `kv_b_proj` on every step (the naive
 * path the plan calls out; the weight-absorption path is future work), so the
 * attention sees plain (qk_nope + rope) scores and v-sized values.
 *
 * Layouts (row-major, BF16):
 *   latent cache   [max_seq, kv_lora_rank + qk_rope_head_dim]
 *   decompressed   [seq_len, num_heads * (qk_nope + v_head)]
 *   repacked latent[max_seq, kv_lora_rank]           (contiguous GEMM input)
 */
#include "kernels.h"
#include "layers.h"
#include "flashinfer_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

__global__ void mla_cache_write_kernel(__nv_bfloat16 *__restrict__ cache,
                                       const __nv_bfloat16 *__restrict__ rows,
                                       int seq_start, int tokens, int width) {
    const int t = blockIdx.x;
    if (t >= tokens) return;
    const __nv_bfloat16 *src = rows + (size_t)t * width;
    __nv_bfloat16 *dst = cache + (size_t)(seq_start + t) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

/* Interleaved RoPE, the pairing DeepSeek's attention uses: the rope slice is
 * read as complex numbers, i.e. (x[2i], x[2i+1]) rotates by pos * theta^(-2i/rope)
 * (the oracle builds its frequencies with dim = qk_rope_head_dim). For each
 * (token, head) the q slice [nope, nope+rope) rotates inside itself, and the
 * shared k_pe rotates over its own rope dims. */
__global__ void mla_rope_kernel(__nv_bfloat16 *__restrict__ q, __nv_bfloat16 *__restrict__ k_pe,
                                int k_pe_stride, const int64_t *__restrict__ positions,
                                int tokens, int heads, int nope, int rope, float theta) {
    const int half = rope / 2;
    const int t = blockIdx.x;
    if (t >= tokens) return;
    const float pos = (float)positions[t];
    for (int job = blockIdx.y * blockDim.x + threadIdx.x; job < heads * half;
         job += gridDim.y * blockDim.x) {
        const int head = job / half;
        const int d = job % half;
        const float angle = pos * powf(theta, -2.0f * (float)d / (float)rope);
        const float c = cosf(angle), s = sinf(angle);
        __nv_bfloat16 *base = q + ((size_t)t * heads + head) * (nope + rope) + nope;
        const float x0 = __bfloat162float(base[2 * d]);
        const float x1 = __bfloat162float(base[2 * d + 1]);
        base[2 * d] = __float2bfloat16_rn(x0 * c - x1 * s);
        base[2 * d + 1] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
    for (int d = blockIdx.y * blockDim.x + threadIdx.x; d < half; d += gridDim.y * blockDim.x) {
        const float angle = pos * powf(theta, -2.0f * (float)d / (float)rope);
        const float c = cosf(angle), s = sinf(angle);
        __nv_bfloat16 *base = k_pe + (size_t)t * k_pe_stride;
        const float x0 = __bfloat162float(base[2 * d]);
        const float x1 = __bfloat162float(base[2 * d + 1]);
        base[2 * d] = __float2bfloat16_rn(x0 * c - x1 * s);
        base[2 * d + 1] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
}

/* One block per (query token, head). Scores live in dynamic shared memory
 * (kv_len floats); the causal window ends at the query's own position. */
__global__ void mla_attention_kernel(const __nv_bfloat16 *__restrict__ q,
                                     const __nv_bfloat16 *__restrict__ k_nope,
                                     const __nv_bfloat16 *__restrict__ v,
                                     const __nv_bfloat16 *__restrict__ k_pe,
                                     int k_pe_stride, int query_offset, int heads,
                                     int nope, int rope, int vdim, float scale,
                                     __nv_bfloat16 *__restrict__ out) {
    extern __shared__ float scores[];
    __shared__ float reduce[256];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int kv_len = query_offset + t + 1;
    const size_t dec_row = (size_t)heads * (nope + vdim);
    const __nv_bfloat16 *q_head = q + ((size_t)t * heads + h) * (nope + rope);
    const __nv_bfloat16 *q_r = q_head + nope;

    float local_max = -INFINITY;
    for (int s = threadIdx.x; s < kv_len; s += blockDim.x) {
        float dot = 0.0f;
        const __nv_bfloat16 *k_n = k_nope + (size_t)s * dec_row + (size_t)h * (nope + vdim);
        for (int d = 0; d < nope; ++d) dot += __bfloat162float(q_head[d]) * __bfloat162float(k_n[d]);
        const __nv_bfloat16 *k_r = k_pe + (size_t)s * k_pe_stride;
        for (int d = 0; d < rope; ++d) dot += __bfloat162float(q_r[d]) * __bfloat162float(k_r[d]);
        scores[s] = dot * scale;
        local_max = fmaxf(local_max, scores[s]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if ((int)threadIdx.x < step) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + step]);
        __syncthreads();
    }
    const float block_max = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int s = threadIdx.x; s < kv_len; s += blockDim.x) {
        scores[s] = expf(scores[s] - block_max);
        local_sum += scores[s];
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if ((int)threadIdx.x < step) reduce[threadIdx.x] += reduce[threadIdx.x + step];
        __syncthreads();
    }
    const float inv = 1.0f / reduce[0];
    __syncthreads();

    for (int d = threadIdx.x; d < vdim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < kv_len; ++s) {
            const __nv_bfloat16 *v_s = v + (size_t)s * dec_row + (size_t)h * (nope + vdim) + nope;
            acc += scores[s] * inv * __bfloat162float(v_s[d]);
        }
        out[((size_t)t * heads + h) * vdim + d] = __float2bfloat16_rn(acc);
    }
}

}  // namespace

size_t kernel_mla_scratch_size(int max_seq, const ModelDims *dims) {
    if (max_seq < 1) throw std::invalid_argument("MLA scratch needs a positive max_seq");
    const size_t decompressed =
        (size_t)max_seq * dims->num_heads * (dims->mla_qk_nope_head_dim + dims->mla_v_head_dim) * 2;
    const size_t repack = (size_t)max_seq * dims->mla_kv_lora_rank * 2;
    return decompressed + repack;
}

int forward_mla_layer(cublasHandle_t cublas, cudaStream_t stream,
                      const __nv_bfloat16 *residual, __nv_bfloat16 *ws,
                      __nv_bfloat16 *layer_out, const MlaWeights *w,
                      __nv_bfloat16 *latent_cache, void *scratch,
                      const int64_t *positions, int tokens, int seq_len,
                      const ModelDims *dims, const GdnTapSites *taps) {
    try {
        if (tokens < 1 || tokens > dims->max_chunk)
            throw std::invalid_argument("MLA token count exceeds max_chunk");
        if (seq_len < tokens || seq_len > dims->max_seq_len)
            throw std::invalid_argument("MLA sequence length is outside the cache");
        if (!positions || !latent_cache || !scratch)
            throw std::invalid_argument("MLA positions/cache/scratch are required");
        const int hidden = dims->hidden_size;
        const int heads = dims->num_heads;
        const int nope = dims->mla_qk_nope_head_dim;
        const int rope = dims->mla_qk_rope_head_dim;
        const int vdim = dims->mla_v_head_dim;
        const int lora = dims->mla_kv_lora_rank;
        const int q_width = heads * (nope + rope);
        const int kv_row_dec = heads * (nope + vdim);
        const int latent_width = lora + rope;
        const float scale = 1.0f / sqrtf((float)(nope + rope));

        char *base = (char *)scratch;
        __nv_bfloat16 *decompressed = (__nv_bfloat16 *)base;
        __nv_bfloat16 *repack = (__nv_bfloat16 *)(base +
            (size_t)dims->max_seq_len * kv_row_dec * 2);

        /* Carve the per-token workspace: normed, q, kv_a and the attention out. */
        __nv_bfloat16 *normed = ws;
        __nv_bfloat16 *q = normed + (size_t)tokens * hidden;
        __nv_bfloat16 *kv_a = q + (size_t)tokens * q_width;
        __nv_bfloat16 *attn = kv_a + (size_t)tokens * latent_width;

        /* Stage taps line up with the oracle's submodule hooks: mla_in is the
         * input norm, mla_q_pre/mla_kv_pre the raw projections, mla_latent the
         * normalized latent, mla_q/mla_kv the post-RoPE values, mla_knope the
         * decompressed per-head K/V and mla_attn the attention output. */
        auto tap = [&](const char *kind, const __nv_bfloat16 *data, int cols) {
            if (taps != nullptr)
                tap_dump_rows(taps->config, kind, taps->layer, taps->device, stream, data,
                              tokens, cols);
        };
        if (w->input_norm_w != nullptr) {
            if (dims->norm_style == 1) {
                kernel_rms_norm_plain(normed, residual, w->input_norm_w, hidden, tokens,
                                      dims->rms_eps, stream);
            } else {
                kernel_gemma_rms_norm(normed, residual, w->input_norm_w, hidden, tokens,
                                      dims->rms_eps, stream);
            }
        } else {
            check_cuda(cudaMemcpyAsync(normed, residual, (size_t)tokens * hidden * 2,
                                       cudaMemcpyDeviceToDevice, stream), "MLA input copy");
        }

        tap("mla_in", normed, hidden);
        int status = gemm_bf16(cublas, q, normed, w->q_proj_w, tokens, q_width, hidden);
        if (status != 0) return status;
        status = gemm_bf16(cublas, kv_a, normed, w->kv_a_proj_w, tokens, latent_width, hidden);
        if (status != 0) return status;
        tap("mla_q_pre", q, q_width);
        tap("mla_kv_pre", kv_a, latent_width);

        /* The latent is normalized with a plain RMSNorm (the reference's
         * kv_a_layernorm is an ordinary RMSNorm even in otherwise shifted-norm
         * families). The library norm kernel assumes both operands are
         * contiguous [rows, cols] blocks, while the latent lives inside rows that
         * are (lora + rope) wide: stage it through the repack block (exactly
         * [tokens, lora]), normalize, and copy back. Reading the strided rows
         * directly corrupts every row after the first (measured: 15x the row's
         * scale). */
        check_cuda(cudaMemcpy2DAsync(repack, (size_t)lora * sizeof(__nv_bfloat16),
                                     kv_a, (size_t)latent_width * sizeof(__nv_bfloat16),
                                     (size_t)lora * sizeof(__nv_bfloat16), (size_t)tokens,
                                     cudaMemcpyDeviceToDevice, stream),
                   "MLA latent stage");
        kernel_rms_norm_plain(repack, repack, w->kv_a_norm_w, lora, tokens, dims->rms_eps, stream);
        check_cuda(cudaGetLastError(), "MLA latent norm");
        tap("mla_latent", repack, lora);
        check_cuda(cudaMemcpy2DAsync(kv_a, (size_t)latent_width * sizeof(__nv_bfloat16),
                                     repack, (size_t)lora * sizeof(__nv_bfloat16),
                                     (size_t)lora * sizeof(__nv_bfloat16), (size_t)tokens,
                                     cudaMemcpyDeviceToDevice, stream),
                   "MLA latent copy back");

        {
            const int blocks_x = tokens;
            const int threads = 128;
            const int head_jobs = std::max(heads * (rope / 2), rope / 2);
            const int blocks_y = std::max(1, (head_jobs + threads - 1) / threads);
            mla_rope_kernel<<<dim3(blocks_x, blocks_y), threads, 0, stream>>>(
                q, kv_a + lora, latent_width, positions, tokens, heads, nope, rope,
                dims->rope_theta);
            check_cuda(cudaGetLastError(), "MLA rope");
        }
        tap("mla_q", q, q_width);
        tap("mla_kv", kv_a, latent_width);

        /* Append the new tokens to the latent cache, then decompress the whole
         * cached range (the naive path; weight absorption would avoid it). */
        mla_cache_write_kernel<<<tokens, 128, 0, stream>>>(
            latent_cache, kv_a, seq_len - tokens, tokens, latent_width);
        check_cuda(cudaGetLastError(), "MLA cache write");
        check_cuda(cudaMemcpy2DAsync(repack, (size_t)lora * 2,
                                     latent_cache, (size_t)latent_width * 2,
                                     (size_t)lora * 2, (size_t)seq_len,
                                     cudaMemcpyDeviceToDevice, stream),
                   "MLA latent repack");
        status = gemm_bf16(cublas, decompressed, repack, w->kv_b_proj_w, seq_len,
                           kv_row_dec, lora);
        if (status != 0) return status;
        tap("mla_knope_v", decompressed + (size_t)(seq_len - tokens) * kv_row_dec, kv_row_dec);

        const int kv_len = seq_len;
        /* The halving tree reduction inside the kernel is only correct for
         * power-of-two block sizes (rounding to multiples of 32 drops whole
         * lane groups for 96/160/192); idle lanes carry identity values. */
        int threads = 32;
        while (threads < kv_len && threads < 256) threads <<= 1;
        mla_attention_kernel<<<dim3(tokens, heads), threads, kv_len * sizeof(float), stream>>>(
            q, decompressed, decompressed, latent_cache + lora, latent_width,
            seq_len - tokens, heads, nope, rope, vdim, scale, attn);
        check_cuda(cudaGetLastError(), "MLA attention");
        tap("mla_attn", attn, heads * vdim);

        status = gemm_bf16(cublas, layer_out, attn, w->o_proj_w, tokens, hidden, heads * vdim);
        if (status != 0) return status;
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "MLA forward failed: %s\n", error.what());
        return -1;
    }
}
