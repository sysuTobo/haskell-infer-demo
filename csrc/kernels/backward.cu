/**
 * backward.cu - The Stage-4 backward kernels for the elementwise, norm, embedding,
 * RoPE, split, GEMM, loss, optimizer and GDN-conv/prepare regions.
 *
 * See csrc/include/backward.h for the contract and csrc/backward.c for its CUDA-free
 * half. The two paired regions (attention core, GDN core) live in backward_paired.cu
 * because they carry the long derivations.
 *
 * Four rules run through every kernel here:
 *
 *   - gradients are FP32 and are *accumulated* when the caller says so, because a
 *     residual branch, a tied parameter and a row-summed weight all have more than one
 *     consumer;
 *   - a derivative consumes the value the forward actually used. Where the forward
 *     rounded a value before multiplying by it (the BF16 sigmoid of the output gate,
 *     the BF16 normalised value inside the GDN gated norm), the backward reads the
 *     rounded value for the *operand* derivative;
 *   - a cast is identity for gradient propagation (the convention backward.h
 *     reports), which means the pre-cast function's local derivative is evaluated with
 *     the *pre-cast* value. That is a rule, not an accident, and the gate pins it: the
 *     output gate's gate-gradient uses the exact sigmoid while its attention-gradient
 *     uses the rounded one, exactly as autograd of the reference composition does;
 *   - a saved *statistic* (the inverse RMS, the L2 reciprocal norm, the base-2 LSE) is
 *     an input, not a recomputation. The chain rule through it is analytic, but its
 *     value has to be the rounded one, or the pairing is a different function.
 */
#include "kernels.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void check_launch(const char *op) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* silu(x) = x * sigmoid(x); silu'(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x))). */
__device__ __forceinline__ float silu_d(float x) {
    const float s = sigmoidf_(x);
    return s * (1.0f + x * (1.0f - s));
}

/* The BF16 value a forward would have stored, read back as FP32: the value a paired
 * backward has to consume as an operand. */
__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

/* Block-wide sum over 256 threads. The block size is a power of two on purpose: the
 * step-halving tree drops lanes for any other size (the bug the MLA kernel had). */
__device__ __forceinline__ float block_sum_256(float value) {
    __shared__ float shared[256];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) shared[threadIdx.x] += shared[threadIdx.x + step];
        __syncthreads();
    }
    return shared[0];
}

int grid_for(long long n) {
    const int block = 256;
    return (int)std::min((long long)65535, (n + block - 1) / block);
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Elementwise                                                        */
/* ------------------------------------------------------------------ */

namespace {

__global__ void sigmoid_mul_backward_kernel(float *__restrict__ d_attn,
                                            float *__restrict__ d_gate,
                                            const float *__restrict__ d_out,
                                            const __nv_bfloat16 *__restrict__ attn,
                                            const __nv_bfloat16 *__restrict__ gate,
                                            int dim, int gate_stride, int gate_offset,
                                            int accumulate) {
    const int t = blockIdx.x;
    const __nv_bfloat16 *attn_row = attn + (long long)t * dim;
    const __nv_bfloat16 *gate_row = gate + (long long)t * gate_stride + gate_offset;
    const float *dout_row = d_out + (long long)t * dim;
    float *da_row = d_attn + (long long)t * dim;
    float *dg_row = d_gate + (long long)t * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float a = __bfloat162float(attn_row[i]);
        const float g = __bfloat162float(gate_row[i]);
        const float s_exact = sigmoidf_(g);
        const float s_stored = round_bf16(s_exact);
        const float d = dout_row[i];
        /* d_attn uses the sigmoid the forward multiplied by; d_gate uses the exact
         * sigmoid (the pre-cast function's own derivative) because the cast is
         * identity for gradient propagation. */
        const float da = d * s_stored;
        const float dg = d * a * s_exact * (1.0f - s_exact);
        if (accumulate) {
            atomicAdd(&da_row[i], da);
            atomicAdd(&dg_row[i], dg);
        } else {
            da_row[i] = da;
            dg_row[i] = dg;
        }
    }
}

/* The MLP's fused gate*up: the forward reads a contiguous [gate[n], up[n]] pair, so
 * the backward reads both operands and writes two separate gradient halves. */
__global__ void silu_mul_backward_kernel(float *__restrict__ d_gate, float *__restrict__ d_up,
                                         const float *__restrict__ d_out,
                                         const __nv_bfloat16 *__restrict__ gate,
                                         const __nv_bfloat16 *__restrict__ up, int n,
                                         int accumulate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        const float d = d_out[i];
        const float dg = d * u * silu_d(g);
        const float du = d * (g * sigmoidf_(g)); /* d/d(up) of silu(gate)*up */
        if (accumulate) {
            atomicAdd(&d_gate[i], dg);
            atomicAdd(&d_up[i], du);
        } else {
            d_gate[i] = dg;
            d_up[i] = du;
        }
    }
}

__global__ void silu_inplace_backward_kernel(float *__restrict__ d_x,
                                             const float *__restrict__ d_out,
                                             const float *__restrict__ pre_activation, int n,
                                             int accumulate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float d = d_out[i] * silu_d(pre_activation[i]);
        if (accumulate) {
            atomicAdd(&d_x[i], d);
        } else {
            d_x[i] = d;
        }
    }
}

__global__ void branch_backward_kernel(float *__restrict__ d_a, float *__restrict__ d_b,
                                       const float *__restrict__ d_out, long long n,
                                       int accumulate) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float d = d_out[i];
        if (accumulate) {
            atomicAdd(&d_a[i], d);
            atomicAdd(&d_b[i], d);
        } else {
            d_a[i] = d;
            d_b[i] = d;
        }
    }
}

}  // namespace

void kernel_sigmoid_mul_backward(float *d_attn, float *d_gate, const float *d_out,
                                 const __nv_bfloat16 *attn, const __nv_bfloat16 *gate, int dim,
                                 int tokens, int gate_stride, int gate_offset, int accumulate,
                                 cudaStream_t stream) {
    if (tokens < 0 || dim <= 0 || gate_offset < 0 || gate_stride < dim ||
        gate_offset > gate_stride - dim) {
        throw std::runtime_error("kernel_sigmoid_mul_backward: invalid shape or gate stride");
    }
    if (tokens == 0) return;
    if (!d_attn || !d_gate || !d_out || !attn || !gate) {
        throw std::runtime_error("kernel_sigmoid_mul_backward: null buffer");
    }
    sigmoid_mul_backward_kernel<<<tokens, 256, 0, stream>>>(d_attn, d_gate, d_out, attn, gate, dim,
                                                            gate_stride, gate_offset, accumulate);
    check_launch("kernel_sigmoid_mul_backward");
}

void kernel_silu_mul_backward(float *d_gate, float *d_up, const float *d_out,
                              const __nv_bfloat16 *gate, const __nv_bfloat16 *up, int n,
                              int accumulate, cudaStream_t stream) {
    if (n < 0) throw std::runtime_error("kernel_silu_mul_backward: negative element count");
    if (n == 0) return;
    if (!d_gate || !d_up || !d_out || !gate || !up) {
        throw std::runtime_error("kernel_silu_mul_backward: null buffer");
    }
    const int grid = grid_for(n);
    silu_mul_backward_kernel<<<grid, 256, 0, stream>>>(d_gate, d_up, d_out, gate, up, n,
                                                       accumulate);
    check_launch("kernel_silu_mul_backward");
}

void kernel_silu_inplace_backward(float *d_x, const float *d_out, const float *pre_activation,
                                  int n, int accumulate, cudaStream_t stream) {
    if (n < 0) throw std::runtime_error("kernel_silu_inplace_backward: negative element count");
    if (n == 0) return;
    if (!d_x || !d_out || !pre_activation) {
        throw std::runtime_error("kernel_silu_inplace_backward: null buffer");
    }
    const int grid = grid_for(n);
    silu_inplace_backward_kernel<<<grid, 256, 0, stream>>>(d_x, d_out, pre_activation, n,
                                                           accumulate);
    check_launch("kernel_silu_inplace_backward");
}

namespace {

__global__ void f32_add_kernel(float *__restrict__ out, const float *__restrict__ a,
                              const float *__restrict__ b, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

__global__ void f32_accumulate_kernel(float *__restrict__ dst, const float *__restrict__ src,
                                      long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        dst[i] += src[i];
    }
}

}  // namespace

void kernel_f32_add(float *out, const float *a, const float *b, long long n,
                    cudaStream_t stream) {
    if (n < 0 || (n > 0 && (out == nullptr || a == nullptr || b == nullptr))) {
        throw std::runtime_error("kernel_f32_add: invalid buffer");
    }
    if (n == 0) return;
    f32_add_kernel<<<grid_for(n), 256, 0, stream>>>(out, a, b, n);
    check_launch("kernel_f32_add");
}

void kernel_f32_accumulate(float *dst, const float *src, long long n, cudaStream_t stream) {
    if (n < 0 || (n > 0 && (dst == nullptr || src == nullptr))) {
        throw std::runtime_error("kernel_f32_accumulate: invalid buffer");
    }
    if (n == 0) return;
    f32_accumulate_kernel<<<grid_for(n), 256, 0, stream>>>(dst, src, n);
    check_launch("kernel_f32_accumulate");
}

void kernel_branch_backward(float *d_a, float *d_b, const float *d_out, long long n, int accumulate,
                            cudaStream_t stream) {
    if (n < 0) throw std::runtime_error("kernel_branch_backward: negative element count");
    if (n == 0) return;
    if (!d_a || !d_b || !d_out) throw std::runtime_error("kernel_branch_backward: null buffer");
    branch_backward_kernel<<<grid_for(n), 256, 0, stream>>>(d_a, d_b, d_out, n, accumulate);
    check_launch("kernel_branch_backward");
}

/* ------------------------------------------------------------------ */
/* Norms                                                              */
/* ------------------------------------------------------------------ */

namespace {

/* One block per row. y_j = x_j * s * w'_j with s the saved inverse RMS, so the row's
 * gradient needs dot = sum_k w'_k x_k d_out_k and then
 *   d_x_j = s*w'_j*d_out_j - (s^3/cols) * x_j * dot ;  d_w_j = d_out_j * s * x_j.
 * The weight gradient is summed over rows, so it always accumulates. */
__global__ void rmsnorm_backward_kernel(float *__restrict__ d_x, float *__restrict__ d_weight,
                                        const float *__restrict__ d_out,
                                        const float *__restrict__ x,
                                        const float *__restrict__ raw_weight,
                                        const float *__restrict__ inv_rms, int cols, int gemma,
                                        int accumulate) {
    const int row = blockIdx.x;
    const float *x_row = x + (long long)row * cols;
    const float *dout_row = d_out + (long long)row * cols;
    const float s = inv_rms[row];
    float *dx_row = d_x + (long long)row * cols;

    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float w = raw_weight[c] + (gemma ? 1.0f : 0.0f);
        local += w * x_row[c] * dout_row[c];
    }
    const float dot = block_sum_256(local);

    const float correction = s * s * s / (float)cols * dot;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float w = raw_weight[c] + (gemma ? 1.0f : 0.0f);
        const float d = dout_row[c];
        const float dx = s * w * d - correction * x_row[c];
        const float dw = d * s * x_row[c];
        if (accumulate) {
            atomicAdd(&dx_row[c], dx);
        } else {
            dx_row[c] = dx;
        }
        atomicAdd(&d_weight[c], dw);
    }
}

/* y = x * r : d_x_j = r*d_out_j - r^3 * x_j * sum_k(d_out_k x_k). No mean, and the eps
 * lives inside the forward's rsqrt, so `inv_norm` is rsqrt(sum x^2 + eps). */
__global__ void l2norm_backward_kernel(float *__restrict__ d_x, const float *__restrict__ d_out,
                                       const float *__restrict__ x,
                                       const float *__restrict__ inv_norm, int cols, int rows,
                                       int accumulate) {
    const int row = blockIdx.x;
    const float *x_row = x + (long long)row * cols;
    const float *dout_row = d_out + (long long)row * cols;
    float *dx_row = d_x + (long long)row * cols;
    const float r = inv_norm[row];

    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local += dout_row[c] * x_row[c];
    const float dot = block_sum_256(local);

    const float correction = r * r * r * dot;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float dx = r * dout_row[c] - correction * x_row[c];
        if (accumulate) {
            atomicAdd(&dx_row[c], dx);
        } else {
            dx_row[c] = dx;
        }
    }
}

/* The GDN gated norm, in the forward's own boundary order:
 *   s = inv_rms ; n = bf16(x*s) ; weighted = bf16(n*w) ; out = bf16(weighted*swish(z)).
 * Identity casts, so the operand derivatives use the rounded n and weighted, and the
 * normalisation's own correction term uses the rounded n as well. */
__global__ void gdn_gated_norm_backward_kernel(float *__restrict__ d_x,
                                               float *__restrict__ d_weight,
                                               float *__restrict__ d_z,
                                               const float *__restrict__ d_out,
                                               const float *__restrict__ x,
                                               const float *__restrict__ z,
                                               const float *__restrict__ weight,
                                               const float *__restrict__ inv_rms, int dim,
                                               int accumulate) {
    const int row = blockIdx.x;
    const float *x_row = x + (long long)row * dim;
    const float *z_row = z + (long long)row * dim;
    const float *dout_row = d_out + (long long)row * dim;
    float *dx_row = d_x + (long long)row * dim;
    float *dz_row = d_z + (long long)row * dim;
    const float s = inv_rms[row];

    float local = 0.0f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float n = round_bf16(x_row[c] * s);
        const float sw = z_row[c] * sigmoidf_(z_row[c]);
        local += weight[c] * n * (dout_row[c] * sw);
    }
    const float dot = block_sum_256(local);

    const float correction = s * s * s / (float)dim * dot;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float xv = x_row[c];
        const float zv = z_row[c];
        const float n = round_bf16(xv * s);
        const float sw = zv * sigmoidf_(zv);
        const float weighted = round_bf16(n * weight[c]);
        const float d = dout_row[c];
        /* d(n)/d(x*s) is the identity cast, so the x gradient is s * (d * sw * w) minus
         * the normalisation's correction. */
        const float dx = s * (d * sw * weight[c]) - correction * xv;
        const float dw = d * sw * n;
        const float dz = d * weighted * silu_d(zv);
        if (accumulate) {
            atomicAdd(&dx_row[c], dx);
            atomicAdd(&dz_row[c], dz);
        } else {
            dx_row[c] = dx;
            dz_row[c] = dz;
        }
        atomicAdd(&d_weight[c], dw);
    }
}

}  // namespace

void kernel_rmsnorm_backward(float *d_x, float *d_weight, const float *d_out, const float *x,
                             const float *raw_weight, const float *inv_rms, int cols, int rows,
                             int gemma, int accumulate, cudaStream_t stream) {
    if (rows < 0 || cols <= 0 || cols > 65535 * 256) {
        throw std::runtime_error("kernel_rmsnorm_backward: invalid shape");
    }
    if (rows == 0) return;
    if (!d_x || !d_weight || !d_out || !x || !raw_weight || !inv_rms) {
        throw std::runtime_error("kernel_rmsnorm_backward: null buffer");
    }
    rmsnorm_backward_kernel<<<rows, 256, 0, stream>>>(d_x, d_weight, d_out, x, raw_weight, inv_rms,
                                                      cols, gemma, accumulate);
    check_launch("kernel_rmsnorm_backward");
}

void kernel_l2norm_backward(float *d_x, const float *d_out, const float *x, const float *inv_norm,
                            int cols, int rows, int accumulate, cudaStream_t stream) {
    if (rows < 0 || cols <= 0) throw std::runtime_error("kernel_l2norm_backward: invalid shape");
    if (rows == 0) return;
    if (!d_x || !d_out || !x || !inv_norm) {
        throw std::runtime_error("kernel_l2norm_backward: null buffer");
    }
    l2norm_backward_kernel<<<rows, 256, 0, stream>>>(d_x, d_out, x, inv_norm, cols, rows,
                                                     accumulate);
    check_launch("kernel_l2norm_backward");
}

void kernel_gdn_gated_norm_backward(float *d_x, float *d_weight, float *d_z, const float *d_out,
                                    const float *x, const float *z, const float *weight,
                                    const float *inv_rms, int dim, int rows, int accumulate,
                                    cudaStream_t stream) {
    if (rows < 0 || dim <= 0) {
        throw std::runtime_error("kernel_gdn_gated_norm_backward: invalid shape");
    }
    if (rows == 0) return;
    if (!d_x || !d_weight || !d_z || !d_out || !x || !z || !weight || !inv_rms) {
        throw std::runtime_error("kernel_gdn_gated_norm_backward: null buffer");
    }
    gdn_gated_norm_backward_kernel<<<rows, 256, 0, stream>>>(d_x, d_weight, d_z, d_out, x, z,
                                                             weight, inv_rms, dim, accumulate);
    check_launch("kernel_gdn_gated_norm_backward");
}

/* ------------------------------------------------------------------ */
/* Embedding                                                          */
/* ------------------------------------------------------------------ */

namespace {

__global__ void embedding_backward_kernel(float *__restrict__ d_table,
                                          const float *__restrict__ d_out,
                                          const int64_t *__restrict__ token_ids, int hidden,
                                          int tokens) {
    const int t = blockIdx.x;
    if (t >= tokens) return;
    const int64_t id = token_ids[t];
    if (id < 0) return; /* a negative id selects nothing, the same convention as the mask */
    float *dst = d_table + id * (long long)hidden;
    const float *src = d_out + (long long)t * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        /* Atomically, because the same id can appear at several positions and the plan
         * requires those contributions to sum rather than overwrite. */
        atomicAdd(&dst[i], src[i]);
    }
}

}  // namespace

void kernel_embedding_backward(float *d_table, const float *d_out, const int64_t *token_ids,
                               int hidden, int tokens, cudaStream_t stream) {
    if (hidden <= 0 || tokens < 0) {
        throw std::runtime_error("kernel_embedding_backward: invalid shape");
    }
    if (tokens == 0) return;
    if (!d_table || !d_out || !token_ids) {
        throw std::runtime_error("kernel_embedding_backward: null buffer");
    }
    embedding_backward_kernel<<<tokens, 256, 0, stream>>>(d_table, d_out, token_ids, hidden,
                                                          tokens);
    check_launch("kernel_embedding_backward");
}

/* ------------------------------------------------------------------ */
/* RoPE and the Q/gate split                                          */
/* ------------------------------------------------------------------ */

namespace {

/* The forward's split-half rotation and the same table it builds:
 * for i in [0, rotary_dim/2), angle = pos * theta^(-2i/rotary_dim),
 *   y_i          = x_i cos - x_{i + d/2} sin
 *   y_{i + d/2}  = x_i sin + x_{i + d/2} cos
 * A rotation's transpose is the rotation by -angle, so the backward exchanges the two
 * components and negates the sine. */
__device__ __forceinline__ void rope_angle(int i, int rotary_dim, float theta, long long pos,
                                           float *out_cos, float *out_sin) {
    const float freq = powf(theta, -2.0f * (float)i / (float)rotary_dim);
    const float angle = (float)pos * freq;
    *out_cos = cosf(angle);
    *out_sin = sinf(angle);
}

/* One block per (token, head) over q then k, so a head's rows never race. */
__global__ void rope_backward_kernel(float *__restrict__ d_q, float *__restrict__ d_k,
                                     const float *__restrict__ d_out_q,
                                     const float *__restrict__ d_out_k,
                                     const int64_t *__restrict__ positions, int q_heads,
                                     int kv_heads, int head_dim, int rotary_dim, float theta,
                                     int accumulate) {
    const int row = blockIdx.x;
    const int t = row / (q_heads + kv_heads);
    const int h = row % (q_heads + kv_heads);
    const int is_q = h < q_heads;
    const int head = is_q ? h : h - q_heads;
    const int heads = is_q ? q_heads : kv_heads;
    const float *dout = (is_q ? d_out_q : d_out_k) + ((long long)t * heads + head) * head_dim;
    float *dx = (is_q ? d_q : d_k) + ((long long)t * heads + head) * head_dim;

    const int half = rotary_dim / 2;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float value;
        if (i < half) {
            float c, s;
            rope_angle(i, rotary_dim, theta, (long long)positions[t], &c, &s);
            value = dout[i] * c + dout[i + half] * s; /* the transposed rotation */
        } else if (i < rotary_dim) {
            const int j = i - half;
            float c, s;
            rope_angle(j, rotary_dim, theta, (long long)positions[t], &c, &s);
            value = -dout[j] * s + dout[i] * c;
        } else {
            value = dout[i]; /* beyond rotary_dim the forward copied the value through */
        }
        if (accumulate) {
            atomicAdd(&dx[i], value);
        } else {
            dx[i] = value;
        }
    }
}

/* The inverse of the deinterleave: raw holds Q and the gate interleaved per row, and
 * each raw element has exactly one consumer, so this assigns rather than accumulates. */
__global__ void qgate_merge_backward_kernel(float *__restrict__ d_raw,
                                            const float *__restrict__ d_q,
                                            const float *__restrict__ d_gate, long long total,
                                            int head_dim) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long row = i / head_dim;
        const int d = (int)(i % head_dim);
        d_raw[row * 2 * head_dim + d] = d_q[i];
        d_raw[row * 2 * head_dim + head_dim + d] = d_gate[i];
    }
}

}  // namespace

void kernel_rope_backward(float *d_q, float *d_k, const float *d_out_q, const float *d_out_k,
                          const int64_t *positions, int tokens, int q_heads, int kv_heads,
                          int head_dim, int rotary_dim, float theta, int accumulate,
                          cudaStream_t stream) {
    if (tokens < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
        rotary_dim > head_dim || rotary_dim % 2 != 0 || !std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("kernel_rope_backward: invalid shape, alignment or theta");
    }
    if (tokens == 0) return;
    if (!d_q || !d_k || !d_out_q || !d_out_k || !positions) {
        throw std::runtime_error("kernel_rope_backward: null buffer");
    }
    const int blocks = tokens * (q_heads + kv_heads);
    rope_backward_kernel<<<blocks, 256, 0, stream>>>(d_q, d_k, d_out_q, d_out_k, positions,
                                                     q_heads, kv_heads, head_dim, rotary_dim, theta,
                                                     accumulate);
    check_launch("kernel_rope_backward");
}

void kernel_qgate_merge_backward(float *d_raw, const float *d_q, const float *d_gate, int total,
                                 int head_dim, cudaStream_t stream) {
    if (total < 0 || head_dim <= 0) {
        throw std::runtime_error("kernel_qgate_merge_backward: invalid shape");
    }
    if (total == 0) return;
    if (!d_raw || !d_q || !d_gate) {
        throw std::runtime_error("kernel_qgate_merge_backward: null buffer");
    }
    qgate_merge_backward_kernel<<<grid_for(total), 256, 0, stream>>>(d_raw, d_q, d_gate, total,
                                                                    head_dim);
    check_launch("kernel_qgate_merge_backward");
}

/* ------------------------------------------------------------------ */
/* GEMM                                                               */
/* ------------------------------------------------------------------ */

/* cublasSgemm over the FP32 widenings of the forward's operands (see the header for why
 * the product is FP32 rather than BF16-by-BF16). The transposes come from the
 * row-major-to-column-major reading csrc/kernels/gemm.cu documents; the gradient test
 * compares both against a matmul in double, which is what pins them. */
int gemm_backward_dx(cublasHandle_t handle, float *d_x, const float *d_out, const float *W_fp32,
                     int M, int N, int K, int accumulate) {
    if (handle == nullptr || d_x == nullptr || d_out == nullptr || W_fp32 == nullptr || M <= 0 ||
        N <= 0 || K <= 0) {
        return -1;
    }
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    /* dX(M,K) = dOut(M,N) @ W(N,K): no transpose on the mathematical operands, because
     * the gradient of out = x W^T with respect to x contracts dOut with W itself. Read
     * column-major, the result is dX^T (K,M) = W^T(K,N) . dOut^T(N,M), and each operand
     * is exactly its own row-major memory viewed as a column-major matrix: W with ld=K
     * is W^T, dOut with ld=N is dOut^T. */
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha,
                                              W_fp32, K, d_out, N, &beta, d_x, K);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int gemm_backward_dw(cublasHandle_t handle, float *d_w, const float *x_fp32, const float *d_out,
                     int M, int N, int K, int accumulate) {
    if (handle == nullptr || d_w == nullptr || x_fp32 == nullptr || d_out == nullptr || M <= 0 ||
        N <= 0 || K <= 0) {
        return -1;
    }
    /* dW always accumulates into the role's gradient buffer: a weight has exactly one
     * gradient slot but may be touched by more than one call (a tied reader, or a
     * micro-batch), so assigning would drop contributions. `accumulate` is accepted for
     * one contract across both functions and must be 1 for a training step. */
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    /* dW(N,K) = dOut(M,N)^T @ x(M,K): the contraction runs over M, the first operand is
     * read as-is (col-major (K,M) with ld=K, which is x's own memory) and the second is
     * transposed (dOut's memory as col-major (N,M) with ld=N). */
    const cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &alpha,
                                              x_fp32, K, d_out, N, &beta, d_w, K);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Log-softmax gather (the fused masked-CE gradient)                  */
/* ------------------------------------------------------------------ */

namespace {

__global__ void logprob_gather_backward_kernel(float *__restrict__ d_logits,
                                               const float *__restrict__ d_out,
                                               const float *__restrict__ logits,
                                               const int *__restrict__ labels,
                                               const uint8_t *__restrict__ mask, int rows,
                                               int vocab) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float *drow = d_logits + (size_t)row * vocab;
    const float *row_logits = logits + (size_t)row * vocab;
    const int label = labels[row];
    const int selected = (mask == nullptr || mask[row] != 0) && label >= 0 && label < vocab;
    if (!selected) {
        for (int j = threadIdx.x; j < vocab; j += blockDim.x) drow[j] = 0.0f;
        return;
    }
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[j]);
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + step]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        local_sum += expf(row_logits[j] - row_max);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (threadIdx.x < step) shared[threadIdx.x] += shared[threadIdx.x + step];
        __syncthreads();
    }
    const float inv_z = 1.0f / shared[0];
    const float d = d_out[row];
    __syncthreads();
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        /* dlogits = d_out * (softmax - onehot): the gradient of -(log p_label). */
        float g = d * expf(row_logits[j] - row_max) * inv_z;
        if (j == label) g -= d;
        drow[j] = g;
    }
}

}  // namespace

void kernel_logprob_gather_backward(float *d_logits, const float *d_out, const float *logits,
                                    const int *labels, const uint8_t *mask, int rows, int vocab,
                                    cudaStream_t stream) {
    if (rows < 0 || vocab <= 0) {
        throw std::invalid_argument("kernel_logprob_gather_backward: invalid shape");
    }
    if (rows == 0) return;
    if (!d_logits || !d_out || !logits || !labels) {
        throw std::invalid_argument("kernel_logprob_gather_backward: null buffer");
    }
    logprob_gather_backward_kernel<<<rows, 256, 0, stream>>>(d_logits, d_out, logits, labels, mask,
                                                             rows, vocab);
    check_launch("kernel_logprob_gather_backward");
}

/* ------------------------------------------------------------------ */
/* AdamW                                                              */
/* ------------------------------------------------------------------ */

namespace {

__global__ void adamw_kernel(float *__restrict__ master, const float *__restrict__ grad,
                             float *__restrict__ m_slot, float *__restrict__ v_slot, long long n,
                             float beta1, float beta2, float step_size, float sqrt_bc2,
                             float eps, float decay_factor, __nv_bfloat16 *__restrict__ bf16_out) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float g = grad[i];
        const float m = beta1 * m_slot[i] + (1.0f - beta1) * g;
        const float v = beta2 * v_slot[i] + (1.0f - beta2) * g * g;
        m_slot[i] = m;
        v_slot[i] = v;
        /* The same order backward.c documents: the decoupled decay first, then eps
         * after the bias-corrected second moment. */
        float theta = master[i] * decay_factor;
        theta -= step_size * m / (sqrtf(v) / sqrt_bc2 + eps);
        master[i] = theta;
        if (bf16_out != nullptr) bf16_out[i] = __float2bfloat16_rn(theta);
    }
}

}  // namespace

void kernel_adamw(float *master, const float *grad, float *m_slot, float *v_slot, long long n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, int step,
                  __nv_bfloat16 *bf16_out, cudaStream_t stream) {
    if (n < 0 || step < 1) throw std::runtime_error("kernel_adamw: invalid shape or step");
    if (n == 0) return;
    if (!master || !grad || !m_slot || !v_slot) {
        throw std::runtime_error("kernel_adamw: null buffer");
    }
    const float bc1 = 1.0f - powf(beta1, (float)step);
    const float bc2 = 1.0f - powf(beta2, (float)step);
    const float step_size = lr / bc1;
    const float sqrt_bc2 = sqrtf(bc2);
    const float decay_factor = 1.0f - lr * weight_decay;
    adamw_kernel<<<grid_for(n), 256, 0, stream>>>(master, grad, m_slot, v_slot, n, beta1, beta2,
                                                  step_size, sqrt_bc2, eps, decay_factor, bf16_out);
    check_launch("kernel_adamw");
}

/* ------------------------------------------------------------------ */
/* GDN causal conv1d                                                  */
/* ------------------------------------------------------------------ */

namespace {

/* The forward, per channel c and token t:
 *   out[t,c] = bias[c] + sum_{j=0..k-1} w[c,j] * xv_j,   xv_j = x[t-(k-1)+j]
 * with a negative source index read from the state (oldest-first at index 0).
 *
 * One block per channel, so the two per-channel scalars (d_bias, d_weight) reduce in
 * shared memory and every (channel, position) has exactly one owner. The position
 * sweep covers the state entries and the input together, which is why the same loop
 * produces d_x and the incoming-state gradient.
 */
__global__ void causal_conv1d_backward_kernel(float *__restrict__ d_x,
                                              float *__restrict__ d_weight,
                                              float *__restrict__ d_bias,
                                              float *__restrict__ d_state_in,
                                              const float *__restrict__ d_out,
                                              const __nv_bfloat16 *__restrict__ x_in,
                                              const __nv_bfloat16 *__restrict__ weight,
                                              const __nv_bfloat16 *__restrict__ conv_state_in,
                                              int conv_dim, int tokens, int kernel_size,
                                              int accumulate) {
    const int channel = blockIdx.x;
    const int state_len = kernel_size - 1;
    float w[8];
    for (int j = 0; j < kernel_size && j < 8; ++j) {
        w[j] = __bfloat162float(weight[channel * kernel_size + j]);
    }
    const __nv_bfloat16 *state = conv_state_in + (long long)channel * state_len;

    /* The source value at filter index j for output t: x[t-(k-1)+j], or the state. */
    auto source = [&](int t, int j) -> float {
        const int idx = t - state_len + j;
        if (idx >= 0) return __bfloat162float(x_in[(long long)idx * conv_dim + channel]);
        return __bfloat162float(state[idx + state_len]);
    };

    /* d_bias and d_weight: one shared reduction each, then a single writer. The write is
     * a plain add or assign in the owning block, never a host memset over device memory. */
    __shared__ float shared_weight[8];
    for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < tokens; ++t) {
            sum += d_out[(long long)t * conv_dim + channel] * source(t, j);
        }
        shared_weight[j] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double bias_sum = 0.0;
        for (int t = 0; t < tokens; ++t) bias_sum += (double)d_out[(long long)t * conv_dim + channel];
        for (int j = 0; j < kernel_size; ++j) {
            if (accumulate) {
                d_weight[channel * kernel_size + j] += shared_weight[j];
            } else {
                d_weight[channel * kernel_size + j] = shared_weight[j];
            }
        }
        if (accumulate) {
            d_bias[channel] += (float)bias_sum;
        } else {
            d_bias[channel] = (float)bias_sum;
        }
    }

    /* The position sweep: d[position] = sum_j d_out[t = s+(k-1)-j, c] * w[c,j]. */
    const int positions = state_len + tokens;
    for (int p = threadIdx.x; p < positions; p += blockDim.x) {
        const int s = p - state_len;
        float value = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            const int t = s + (kernel_size - 1) - j;
            if (t < 0 || t >= tokens) continue;
            value += d_out[(long long)t * conv_dim + channel] * w[j];
        }
        if (s < 0) {
            if (d_state_in != nullptr) d_state_in[channel * state_len + p] = value;
        } else if (accumulate) {
            d_x[(long long)s * conv_dim + channel] += value;
        } else {
            /* Each (channel, position) has exactly one owner, so an assign is exact. */
            d_x[(long long)s * conv_dim + channel] = value;
        }
    }
}

}  // namespace

void kernel_causal_conv1d_backward(float *d_x, float *d_weight, float *d_bias, float *d_state_in,
                                   const float *d_out, const __nv_bfloat16 *x_in,
                                   const __nv_bfloat16 *weight,
                                   const __nv_bfloat16 *conv_state_in, int conv_dim, int tokens,
                                   int kernel_size, int accumulate, cudaStream_t stream) {
    if (conv_dim <= 0 || tokens < 0 || kernel_size <= 0 || kernel_size > 8) {
        throw std::runtime_error("kernel_causal_conv1d_backward: invalid shape");
    }
    if (tokens == 0) return;
    if (!d_x || !d_weight || !d_bias || !d_out || !x_in || !weight || !conv_state_in) {
        throw std::runtime_error("kernel_causal_conv1d_backward: null buffer");
    }
    causal_conv1d_backward_kernel<<<conv_dim, 256, 0, stream>>>(
        d_x, d_weight, d_bias, d_state_in, d_out, x_in, weight, conv_state_in, conv_dim, tokens,
        kernel_size, accumulate);
    check_launch("kernel_causal_conv1d_backward");
}

/* ------------------------------------------------------------------ */
/* GDN prepare                                                        */
/* ------------------------------------------------------------------ */

namespace {

/* One block per token. Every prepared head's gradient is written by the threads of
 * that token's block, and the duplicated key heads are reduced with atomics (the
 * group's value heads all feed one key head's input). */
__global__ void gdn_prepare_backward_kernel(
    float *__restrict__ d_conv_out, float *__restrict__ d_a, float *__restrict__ d_b,
    float *__restrict__ d_A_log, float *__restrict__ d_dt_bias, const float *__restrict__ d_q,
    const float *__restrict__ d_k, const float *__restrict__ d_v,
    const float *__restrict__ d_log_decay, const float *__restrict__ d_beta,
    const __nv_bfloat16 *__restrict__ conv_out, const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b, const __nv_bfloat16 *__restrict__ A_log,
    const __nv_bfloat16 *__restrict__ dt_bias, int key_heads, int value_heads, int head_dim,
    int accumulate) {
    const int t = blockIdx.x;
    const int group = value_heads / key_heads;
    const int q_base = 0;                        /* conv_out's Q block */
    const int k_base = key_heads * head_dim;     /* conv_out's K block */
    const int v_base = 2 * key_heads * head_dim; /* conv_out's V block */
    const float eps = 1e-6f;
    const long long conv_stride =
        2 * (long long)key_heads * head_dim + (long long)value_heads * head_dim;
    const __nv_bfloat16 *conv_row = conv_out + (long long)t * conv_stride;

    /* The output row is zeroed by the block that owns it, not by the host: a zeroing
     * loop in the wrapper would be a host write over device memory. */
    if (!accumulate) {
        for (long long i = threadIdx.x; i < conv_stride; i += blockDim.x) {
            d_conv_out[(long long)t * conv_stride + i] = 0.0f;
        }
        __syncthreads();
    }

    /* Phase A, one thread per key head: the reciprocal norm and the two dot products
     * the L2 backward needs. The dot can be summed over the group first, because every
     * element of a key head feeds all `group` value heads and the correction term
     * factorises: sum_h (r^3 q_d sum_e dq_he q_e) = r^3 q_d * sum_h sum_e dq_he q_e. */
    extern __shared__ float shared[];
    float *r_q = shared;                  /* [key_heads] */
    float *r_k = shared + key_heads;      /* [key_heads] */
    float *dot_q = shared + 2 * key_heads;
    float *dot_k = shared + 3 * key_heads;
    for (int kh = threadIdx.x; kh < key_heads; kh += blockDim.x) {
        const __nv_bfloat16 *row_q = conv_row + q_base + kh * head_dim;
        const __nv_bfloat16 *row_k = conv_row + k_base + kh * head_dim;
        float sq_q = 0.0f, sq_k = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float qv = __bfloat162float(row_q[d]);
            const float kv = __bfloat162float(row_k[d]);
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
        r_q[kh] = rsqrtf(sq_q + eps);
        r_k[kh] = rsqrtf(sq_k + eps);
        float acc_q = 0.0f, acc_k = 0.0f;
        for (int g = 0; g < group; ++g) {
            const int h = kh * group + g;
            for (int d = 0; d < head_dim; ++d) {
                acc_q += d_q[((long long)t * value_heads + h) * head_dim + d] *
                         __bfloat162float(row_q[d]);
                acc_k += d_k[((long long)t * value_heads + h) * head_dim + d] *
                         __bfloat162float(row_k[d]);
            }
        }
        dot_q[kh] = acc_q;
        dot_k[kh] = acc_k;
    }
    __syncthreads();

    /* Phase B: the element-wise gradients. The linear term is accumulated for every
     * group member; the L2 correction is subtracted *once per key head*, because
     * `dot_q[kh]` already is the sum over the group -- subtracting it inside the group
     * loop would over-correct by the group size, which is the bug this split avoids. */
    for (int g = 0; g < group; ++g) {
        for (int kh = 0; kh < key_heads; ++kh) {
            const int h = kh * group + g;
            const float rq = r_q[kh];
            const float rk = r_k[kh];
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                atomicAdd(&d_conv_out[(long long)t * conv_stride + q_base + kh * head_dim + d],
                          rq * d_q[((long long)t * value_heads + h) * head_dim + d]);
                atomicAdd(&d_conv_out[(long long)t * conv_stride + k_base + kh * head_dim + d],
                          rk * d_k[((long long)t * value_heads + h) * head_dim + d]);
            }
        }
    }
    for (int kh = 0; kh < key_heads; ++kh) {
        const float corr_q = r_q[kh] * r_q[kh] * r_q[kh] * dot_q[kh];
        const float corr_k = r_k[kh] * r_k[kh] * r_k[kh] * dot_k[kh];
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            atomicAdd(&d_conv_out[(long long)t * conv_stride + q_base + kh * head_dim + d],
                      -corr_q * __bfloat162float(conv_row[q_base + kh * head_dim + d]));
            atomicAdd(&d_conv_out[(long long)t * conv_stride + k_base + kh * head_dim + d],
                      -corr_k * __bfloat162float(conv_row[k_base + kh * head_dim + d]));
        }
    }
    for (int h = 0; h < value_heads; ++h) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float value = d_v[((long long)t * value_heads + h) * head_dim + d];
            float *dst = &d_conv_out[(long long)t * conv_stride + v_base + h * head_dim + d];
            if (accumulate) {
                atomicAdd(dst, value);
            } else {
                *dst = value;
            }
        }
    }

    /* The log-decay and beta chains. One thread per value head keeps the per-head
     * reductions (A_log, dt_bias) atomic and the per-token outputs plain. */
    for (int h = threadIdx.x; h < value_heads; h += blockDim.x) {
        const float a_val = __bfloat162float(a[(long long)t * value_heads + h]);
        const float b_val = __bfloat162float(b[(long long)t * value_heads + h]);
        const float a_log = __bfloat162float(A_log[h]);
        const float dt = __bfloat162float(dt_bias[h]);
        const float x = a_val + dt;
        /* softplus with the AOT kernel's stability branch, and its derivative: the
         * branch matters, because for x > 20 softplus(x) is x exactly. */
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        const float dsp = x > 20.0f ? 1.0f : sigmoidf_(x);
        const float exp_a = expf(a_log);
        const float d_g = d_log_decay[(long long)t * value_heads + h];
        d_a[(long long)t * value_heads + h] = d_g * (-exp_a) * dsp;
        atomicAdd(&d_dt_bias[h], d_g * (-exp_a) * dsp);
        /* d/d(A_log) of -exp(A_log)*softplus(...) is -exp(A_log)*softplus(...) itself. */
        atomicAdd(&d_A_log[h], d_g * (-exp_a) * sp);
        /* beta is stored as the FP32 value of a BF16-rounded sigmoid, and a cast is
         * identity for gradient propagation, so the sigmoid's own derivative is
         * evaluated at the pre-cast value. */
        const float s = sigmoidf_(b_val);
        d_b[(long long)t * value_heads + h] =
            d_beta[(long long)t * value_heads + h] * s * (1.0f - s);
    }
}

}  // namespace

void kernel_gdn_prepare_backward(float *d_conv_out, float *d_a, float *d_b, float *d_A_log,
                                 float *d_dt_bias, const float *d_q, const float *d_k,
                                 const float *d_v, const float *d_log_decay, const float *d_beta,
                                 const __nv_bfloat16 *conv_out, const __nv_bfloat16 *a,
                                 const __nv_bfloat16 *b, const __nv_bfloat16 *A_log,
                                 const __nv_bfloat16 *dt_bias, int tokens, int key_heads,
                                 int value_heads, int head_dim, int accumulate,
                                 cudaStream_t stream) {
    if (tokens < 0 || key_heads <= 0 || value_heads <= 0 || head_dim <= 0 ||
        value_heads % key_heads != 0) {
        throw std::runtime_error("kernel_gdn_prepare_backward: invalid shape or head count");
    }
    if (tokens == 0) return;
    if (!d_conv_out || !d_a || !d_b || !d_A_log || !d_dt_bias || !d_q || !d_k || !d_v ||
        !d_log_decay || !d_beta || !conv_out || !a || !b || !A_log || !dt_bias) {
        throw std::runtime_error("kernel_gdn_prepare_backward: null buffer");
    }
    const long long conv_stride = 2 * (long long)key_heads * head_dim + (long long)value_heads * head_dim;
    const int shared_bytes = 4 * key_heads * (int)sizeof(float);
    gdn_prepare_backward_kernel<<<tokens, 256, shared_bytes, stream>>>(
        d_conv_out, d_a, d_b, d_A_log, d_dt_bias, d_q, d_k, d_v, d_log_decay, d_beta, conv_out, a,
        b, A_log, dt_bias, key_heads, value_heads, head_dim, accumulate);
    check_launch("kernel_gdn_prepare_backward");
}
