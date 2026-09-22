#include "kernels.h"

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

// KV cache: [2, max_seq_len, num_kv_heads, head_dim].
__global__ void kv_cache_write_kernel(__nv_bfloat16 *__restrict__ kv_cache,
                                      const __nv_bfloat16 *__restrict__ k_new,
                                      const __nv_bfloat16 *__restrict__ v_new,
                                      int seq_start, int tokens,
                                      int num_kv_heads, int head_dim,
                                      int max_seq_len) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    int kv_dim = num_kv_heads * head_dim;
    int pos = seq_start + t;
    long long dst = static_cast<long long>(pos) * kv_dim;
    long long src = static_cast<long long>(t) * kv_dim;
    long long v_base = static_cast<long long>(max_seq_len) * kv_dim;
    for (int i = threadIdx.x * 4; i < kv_dim; i += blockDim.x * 4) {
        for (int j = 0; j < 4 && i + j < kv_dim; ++j) {
            kv_cache[dst + i + j] = k_new[src + i + j];
            kv_cache[v_base + dst + i + j] = v_new[src + i + j];
        }
    }
}

__global__ void sigmoid_mul_kernel(__nv_bfloat16 *out,
                                   const __nv_bfloat16 *attn,
                                   const __nv_bfloat16 *gate,
                                   int dim, int tokens,
                                   int gate_stride, int gate_offset) {
    int t = blockIdx.x;
    if (t >= tokens) return;

    const __nv_bfloat16 *attn_row = attn + static_cast<long long>(t) * dim;
    const __nv_bfloat16 *gate_row = gate + static_cast<long long>(t) * gate_stride + gate_offset;
    __nv_bfloat16 *out_row = out + static_cast<long long>(t) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = __bfloat162float(attn_row[i]);
        float g = __bfloat162float(gate_row[i]);
        // Match torch.sigmoid(BF16) followed by BF16 multiplication.
        __nv_bfloat16 sig = __float2bfloat16_rn(1.0f / (1.0f + expf(-g)));
        out_row[i] = __float2bfloat16_rn(a * __bfloat162float(sig));
    }
}

}  // namespace

void kernel_kv_cache_write(__nv_bfloat16 *kv_cache,
                           const __nv_bfloat16 *k_new,
                           const __nv_bfloat16 *v_new,
                           int seq_start, int tokens,
                           int num_kv_heads, int head_dim,
                           int max_seq_len, cudaStream_t stream) {
    if (tokens < 0 || seq_start < 0 || max_seq_len < seq_start ||
        tokens > max_seq_len - seq_start || num_kv_heads <= 0 || head_dim <= 0 ||
        num_kv_heads > std::numeric_limits<int>::max() / head_dim) {
        throw std::runtime_error("kernel_kv_cache_write: invalid shape or cache range");
    }
    if (tokens == 0) return;
    if (!kv_cache || !k_new || !v_new) {
        throw std::runtime_error("kernel_kv_cache_write: null buffer");
    }
    int kv_dim = num_kv_heads * head_dim;
    int block = static_cast<int>(std::min(256LL, (static_cast<long long>(kv_dim) + 3) / 4));
    kv_cache_write_kernel<<<tokens, block, 0, stream>>>(
        kv_cache, k_new, v_new, seq_start, tokens,
        num_kv_heads, head_dim, max_seq_len);
    check_cuda(cudaGetLastError(), "kernel_kv_cache_write");
}

void kernel_attention(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                      const __nv_bfloat16 *kv_cache,
                      int seq_start, int tokens, int seq_len,
                      int num_heads, int num_kv_heads, int head_dim,
                      float scale, int max_seq_len, cudaStream_t stream) {
    if (tokens < 0 || seq_start < 0 || max_seq_len < seq_start ||
        tokens > max_seq_len - seq_start || seq_len != seq_start + tokens ||
        head_dim != 256 || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 ||
        num_heads > std::numeric_limits<int>::max() / head_dim || !std::isfinite(scale)) {
        throw std::runtime_error("kernel_attention: expected head_dim=256, valid GQA and seq_len=seq_start+tokens within cache");
    }
    if (tokens == 0) return;
    if (!out || !q || !kv_cache) {
        throw std::runtime_error("kernel_attention: null buffer");
    }

    using Params = flashinfer::SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
    using Attention = flashinfer::DefaultAttention<false, false, false, false>;
    const int kv_stride = num_kv_heads * head_dim;
    Params params(
        const_cast<__nv_bfloat16 *>(q),
        const_cast<__nv_bfloat16 *>(kv_cache),
        const_cast<__nv_bfloat16 *>(kv_cache + static_cast<long long>(max_seq_len) * kv_stride),
        /*maybe_custom_mask=*/nullptr, out, /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr,
        num_heads, num_kv_heads, tokens, seq_len,
        num_heads * head_dim, head_dim, kv_stride, head_dim, head_dim,
        /*window_left=*/-1, /*logits_soft_cap=*/0.0f, scale,
        /*rope_scale=*/1.0f, /*rope_theta=*/1.0f);

    cudaError_t status;
    try {
        // Prefill also handles single-token GQA6; nullptr disables split-KV workspace.
        status = flashinfer::SinglePrefillWithKVCacheDispatched<
            256, 256, flashinfer::PosEncodingMode::kNone, false,
            flashinfer::MaskMode::kCausal, Attention>(params, /*tmp=*/nullptr, stream);
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string("kernel_attention: ") + error.what());
    }
    check_cuda(status, "kernel_attention");
    check_cuda(cudaGetLastError(), "kernel_attention launch");
}

void kernel_sigmoid_mul(__nv_bfloat16 *out, const __nv_bfloat16 *attn,
                        const __nv_bfloat16 *gate, int dim, int tokens,
                        int gate_stride, int gate_offset, cudaStream_t stream) {
    if (tokens < 0 || dim <= 0 || gate_offset < 0 || gate_stride < dim ||
        gate_offset > gate_stride - dim) {
        throw std::runtime_error("kernel_sigmoid_mul: invalid shape or gate stride");
    }
    if (tokens == 0) return;
    if (!out || !attn || !gate) {
        throw std::runtime_error("kernel_sigmoid_mul: null buffer");
    }
    sigmoid_mul_kernel<<<tokens, 256, 0, stream>>>(
        out, attn, gate, dim, tokens, gate_stride, gate_offset);
    check_cuda(cudaGetLastError(), "kernel_sigmoid_mul");
}
