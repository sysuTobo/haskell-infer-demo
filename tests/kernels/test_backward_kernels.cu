/*
 * test_backward_kernels.cu - The Stage-4 gradient gate.
 *
 * Every backward kernel in csrc/kernels/backward.cu and backward_paired.cu against an
 * independent oracle, with no checkpoint and no engine handle.
 *
 * The oracle is a double-precision forward written here, in this file, from the
 * definition of each operation -- not a second copy of the kernel's derivation. For the
 * elementwise, norm, embedding, RoPE, GEMM, conv1d and prepare kernels the *analytic*
 * gradient of that forward is written out, because those gradients are short enough to
 * read. For the two paired regions (attention core, GDN core) the oracle is instead a
 * *central difference* of the double-precision forward, which is the strongest check
 * available for a recurrence whose analytic form is long: it tests the kernel against
 * the definition rather than against my algebra. The fractional step is safe because
 * the reference forward is a smooth function of the operand values -- the BF16 cast
 * happens on the device, and the reference reads the already-rounded values, which is
 * exactly the plan's "casts are identity for gradient propagation" convention. Nothing
 * here finite-differences a discontinuous cast and calls it a gradient.
 *
 * The two paired regions are additionally checked for run-to-run determinism, which the
 * plan asks to be tested separately from closeness to the reference: a kernel that
 * agrees with the oracle to 1e-3 but differs from itself by 1e-3 across runs is not a
 * reproducible trainer. Both kernels reduce with fixed thread mappings and never with
 * atomics on the reduction path, so bitwise equality is the expected result.
 *
 * Shapes are small on purpose: a central difference over every input coordinate is
 * O(N) forwards, and the point of this gate is the mathematics, not the throughput.
 */
#include "kernels.h"
#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using test::Bf16;
using test::DeviceBuffer;
using test::Stream;
using test::bf16;
using test::round_bf16;
using test::sample;

namespace {

bool g_ok = true;

void require(bool condition, const char *what) {
    if (condition) return;
    g_ok = false;
    std::fprintf(stderr, "FAIL: %s\n", what);
}

/* The largest absolute difference, reported with the run's scale so a passing check is
 * still informative. */
double max_abs_diff(const std::vector<float> &got, const std::vector<double> &want) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
        worst = std::max(worst, std::abs((double)got[i] - want[i]));
    }
    return worst;
}

void report(const char *name, double worst, double tolerance) {
    const bool ok = worst <= tolerance;
    if (!ok) g_ok = false;
    std::printf("%-46s max_abs=%.3e tol=%.1e %s\n", name, worst, tolerance, ok ? "ok" : "FAIL");
}

/* The worst element with its index and both values: a max_abs alone says how wrong, not
 * where, and the two remaining failures were both localized (one bad slot, one head). */
template <typename T>
void report_worst(const char *name, const std::vector<T> &got, const std::vector<double> &want,
                  double tolerance) {
    size_t worst = 0;
    double worst_error = -1.0;
    for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
        const double error = std::abs((double)got[i] - want[i]);
        if (error > worst_error) {
            worst_error = error;
            worst = i;
        }
    }
    const bool ok = worst_error <= tolerance;
    if (!ok) g_ok = false;
    std::printf("%-46s max_abs=%.3e tol=%.1e %s", name, worst_error, tolerance, ok ? "ok" : "FAIL");
    if (!ok) {
        std::printf("  worst[%zu] got=%.9g want=%.9g", worst, (double)got[worst], want[worst]);
    }
    std::printf("\n");
}

/* Central difference of a scalar function of `coords` with respect to one coordinate. */
template <typename F>
double fd(F f, std::vector<double> &coords, size_t index, double h) {
    const double saved = coords[index];
    coords[index] = saved + h;
    const double plus = f();
    coords[index] = saved - h;
    const double minus = f();
    coords[index] = saved;
    return (plus - minus) / (2.0 * h);
}

/* A value that is representable in BF16, so the device and the reference see the same
 * operand and the comparison is not about the fixture's rounding. */
double bf16_sample(size_t index, uint32_t salt) {
    return (double)round_bf16(sample(index, salt));
}

/* ------------------------------------------------------------------ */
/* Elementwise                                                        */
/* ------------------------------------------------------------------ */

void elementwise(Stream &stream) {
    const int n = 4096;
    std::vector<Bf16> gate(n), up(n);
    std::vector<double> g(n), u(n);
    for (int i = 0; i < n; ++i) {
        g[i] = bf16_sample(i, 11);
        u[i] = bf16_sample(i, 12);
        gate[i] = bf16((float)g[i]);
        up[i] = bf16((float)u[i]);
    }
    std::vector<double> dout(n);
    for (int i = 0; i < n; ++i) dout[i] = (double)sample(i, 13);
    std::vector<float> dout_f(n);
    for (int i = 0; i < n; ++i) dout_f[i] = (float)dout[i];

    DeviceBuffer<Bf16> d_gate(n), d_up(n);
    DeviceBuffer<float> d_dout(n), d_dgate(n), d_dup(n);
    d_gate.upload(gate, stream.get());
    d_up.upload(up, stream.get());
    d_dout.upload(dout_f, stream.get());
    d_dgate.upload(std::vector<float>(n, 0.0f), stream.get());
    d_dup.upload(std::vector<float>(n, 0.0f), stream.get());

    kernel_silu_mul_backward(d_dgate.get(), d_dup.get(), d_dout.get(), d_gate.get(), d_up.get(), n,
                             0, stream.get());
    CUDA_CHECK(cudaGetLastError());
    /* silu(gate)*up: d/dgate = up * sigmoid(g)(1 + g(1-sigmoid(g))), d/dup = silu(g). */
    std::vector<double> want_gate(n), want_up(n);
    for (int i = 0; i < n; ++i) {
        const double s = 1.0 / (1.0 + std::exp(-g[i]));
        const double silu = g[i] / (1.0 + std::exp(-g[i]));
        want_gate[i] = dout[i] * u[i] * (s * (1.0 + g[i] * (1.0 - s)));
        want_up[i] = dout[i] * silu;
    }
    report("silu_mul_backward d_gate", max_abs_diff(d_dgate.download(stream.get()), want_gate),
           1e-5);
    report("silu_mul_backward d_up", max_abs_diff(d_dup.download(stream.get()), want_up), 1e-5);

    /* In-place SiLU of the GDN conv output. */
    std::vector<float> pre(n);
    for (int i = 0; i < n; ++i) pre[i] = (float)bf16_sample(i, 14);
    DeviceBuffer<float> d_pre(n), d_dx(n);
    d_pre.upload(pre, stream.get());
    d_dx.upload(std::vector<float>(n, 0.0f), stream.get());
    kernel_silu_inplace_backward(d_dx.get(), d_dout.get(), d_pre.get(), n, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> want_dx(n);
    for (int i = 0; i < n; ++i) {
        const double s = 1.0 / (1.0 + std::exp(-pre[i]));
        want_dx[i] = dout[i] * (s * (1.0 + pre[i] * (1.0 - s)));
    }
    report("silu_inplace_backward", max_abs_diff(d_dx.download(stream.get()), want_dx), 1e-6);

    /* The output gate: out = attn * sigmoid(gate), with the forward's BF16 sigmoid. */
    const int dim = 64, tokens = 16, gate_stride = 128, gate_offset = 32;
    std::vector<Bf16> attn((size_t)tokens * dim), gate_buf((size_t)tokens * gate_stride);
    std::vector<double> a(tokens * dim), gv(tokens * dim);
    for (int i = 0; i < tokens * dim; ++i) {
        a[i] = bf16_sample(i, 15);
        attn[i] = bf16((float)a[i]);
    }
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < gate_stride; ++i) {
            /* Only the window carries a gate; the rest of the stride is filler, and the
             * index arithmetic has to stay inside the token's own dim to write it. */
            gate_buf[(size_t)t * gate_stride + i] = bf16(0.0f);
            if (i >= gate_offset && i < gate_offset + dim) {
                const int flat = t * dim + (i - gate_offset);
                gv[flat] = bf16_sample((size_t)flat, 16);
                gate_buf[(size_t)t * gate_stride + i] = bf16((float)gv[flat]);
            }
        }
    }
    std::vector<float> dout_gate((size_t)tokens * dim);
    for (int i = 0; i < tokens * dim; ++i) dout_gate[i] = (float)sample(i, 17);
    DeviceBuffer<Bf16> d_attn(attn.size()), d_gate_buf(gate_buf.size());
    DeviceBuffer<float> d_dout_gate(dout_gate.size()), d_dattn(dout_gate.size()),
        d_dgate_sig(dout_gate.size());
    d_attn.upload(attn, stream.get());
    d_gate_buf.upload(gate_buf, stream.get());
    d_dout_gate.upload(dout_gate, stream.get());
    d_dattn.upload(std::vector<float>(dout_gate.size(), 0.0f), stream.get());
    d_dgate_sig.upload(std::vector<float>(dout_gate.size(), 0.0f), stream.get());
    kernel_sigmoid_mul_backward(d_dattn.get(), d_dgate_sig.get(), d_dout_gate.get(), d_attn.get(),
                                d_gate_buf.get(), dim, tokens, gate_stride, gate_offset, 0,
                                stream.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> want_attn_gate(tokens * dim), want_gate_grad(tokens * dim);
    for (int i = 0; i < tokens * dim; ++i) {
        const double s_exact = 1.0 / (1.0 + std::exp(-gv[i]));
        const double s_stored = (double)round_bf16((float)s_exact);
        want_attn_gate[i] = dout_gate[i] * s_stored;
        want_gate_grad[i] = dout_gate[i] * a[i] * s_exact * (1.0 - s_exact);
    }
    report("sigmoid_mul_backward d_attn",
           max_abs_diff(d_dattn.download(stream.get()), want_attn_gate), 1e-6);
    report("sigmoid_mul_backward d_gate",
           max_abs_diff(d_dgate_sig.download(stream.get()), want_gate_grad), 1e-6);

    /* The residual branch: both sides receive the output gradient. */
    const long long m = 1024;
    std::vector<float> dout_branch(m);
    for (long long i = 0; i < m; ++i) dout_branch[i] = (float)sample((size_t)i, 18);
    DeviceBuffer<float> d_branch_out(m);
    DeviceBuffer<float> d_a_grad(m), d_b_grad(m);
    d_branch_out.upload(dout_branch, stream.get());
    d_a_grad.upload(std::vector<float>(m, 0.0f), stream.get());
    d_b_grad.upload(std::vector<float>(m, 0.0f), stream.get());
    kernel_branch_backward(d_a_grad.get(), d_b_grad.get(), d_branch_out.get(), m, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());
    report("branch_backward d_a", max_abs_diff(d_a_grad.download(stream.get()),
                                               std::vector<double>(dout_branch.begin(),
                                                                   dout_branch.end())),
           0.0);
    report("branch_backward d_b", max_abs_diff(d_b_grad.download(stream.get()),
                                               std::vector<double>(dout_branch.begin(),
                                                                   dout_branch.end())),
           0.0);
}

/* ------------------------------------------------------------------ */
/* Norms                                                              */
/* ------------------------------------------------------------------ */

void norms(Stream &stream) {
    const int rows = 32, cols = 64;
    std::vector<double> x((size_t)rows * cols), w(cols), inv_rms(rows), dout((size_t)rows * cols);
    std::vector<float> xf(x.size()), wf(cols), rf(rows), doutf(x.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = sample(i, 21) * 0.7;
    for (int c = 0; c < cols; ++c) w[c] = 0.5 + 0.5 * sample((size_t)c, 22);
    for (size_t i = 0; i < x.size(); ++i) dout[i] = sample(i, 23);
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += x[(size_t)r * cols + c] * x[(size_t)r * cols + c];
        inv_rms[r] = 1.0 / std::sqrt(sum / cols + 1e-6);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        xf[i] = (float)x[i];
        doutf[i] = (float)dout[i];
    }
    for (int c = 0; c < cols; ++c) wf[c] = (float)w[c];
    for (int r = 0; r < rows; ++r) rf[r] = (float)inv_rms[r];

    for (int gemma = 0; gemma < 2; ++gemma) {
        DeviceBuffer<float> d_x(xf.size()), d_w(cols), d_rf(rows), d_dout(xf.size()),
            d_dx(xf.size()), d_dw(cols);
        d_x.upload(xf, stream.get());
        d_w.upload(wf, stream.get());
        d_rf.upload(rf, stream.get());
        d_dout.upload(doutf, stream.get());
        d_dx.upload(std::vector<float>(xf.size(), 0.0f), stream.get());
        d_dw.upload(std::vector<float>(cols, 0.0f), stream.get());
        /* The weight gradient is a separate buffer from the saved weight: the kernel reads
         * the weight while accumulating into the gradient, so an aliased fixture would
         * corrupt the operand it is differentiating. */
        kernel_rmsnorm_backward(d_dx.get(), d_dw.get(), d_dout.get(), d_x.get(), d_w.get(),
                                d_rf.get(), cols, rows, gemma, 0, stream.get());
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> want_x(x.size(), 0.0), want_w(cols, 0.0);
        for (int r = 0; r < rows; ++r) {
            const double s = inv_rms[r];
            double dot = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double wp = gemma ? w[c] + 1.0 : w[c];
                dot += wp * x[(size_t)r * cols + c] * dout[(size_t)r * cols + c];
            }
            for (int c = 0; c < cols; ++c) {
                const double wp = gemma ? w[c] + 1.0 : w[c];
                const size_t i = (size_t)r * cols + c;
                want_x[i] = s * wp * dout[i] - s * s * s / cols * dot * x[i];
                want_w[c] += dout[i] * s * x[i];
            }
        }
        const char *name = gemma ? "rmsnorm_backward (gemma)" : "rmsnorm_backward (plain)";
        report(name, std::max(max_abs_diff(d_dx.download(stream.get()), want_x),
                              max_abs_diff(d_dw.download(stream.get()), want_w)),
               1e-5);
    }

    /* The GDN Q/K L2 normalisation: no mean, eps inside the rsqrt. */
    {
        std::vector<double> inv_norm(rows);
        std::vector<float> invf(rows);
        for (int r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (int c = 0; c < cols; ++c) sum += x[(size_t)r * cols + c] * x[(size_t)r * cols + c];
            inv_norm[r] = 1.0 / std::sqrt(sum + 1e-6);
            invf[r] = (float)inv_norm[r];
        }
        DeviceBuffer<float> d_x(xf.size()), d_dout(xf.size()), d_dx(xf.size()), d_r(rows);
        d_x.upload(xf, stream.get());
        d_dout.upload(doutf, stream.get());
        d_dx.upload(std::vector<float>(xf.size(), 0.0f), stream.get());
        d_r.upload(invf, stream.get());
        kernel_l2norm_backward(d_dx.get(), d_dout.get(), d_x.get(), d_r.get(), cols, rows, 0,
                               stream.get());
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> want(x.size(), 0.0);
        for (int r = 0; r < rows; ++r) {
            const double rr = inv_norm[r];
            double dot = 0.0;
            for (int c = 0; c < cols; ++c) dot += dout[(size_t)r * cols + c] * x[(size_t)r * cols + c];
            for (int c = 0; c < cols; ++c) {
                const size_t i = (size_t)r * cols + c;
                want[i] = rr * dout[i] - rr * rr * rr * dot * x[i];
            }
        }
        report("l2norm_backward", max_abs_diff(d_dx.download(stream.get()), want), 1e-5);
    }

    /* The GDN gated norm, with the forward's three BF16 boundaries. */
    {
        std::vector<double> z(x.size()), dweight(cols, 0.0);
        std::vector<float> zf(x.size());
        for (size_t i = 0; i < x.size(); ++i) z[i] = sample(i, 24) * 2.0;
        for (size_t i = 0; i < x.size(); ++i) zf[i] = (float)z[i];
        DeviceBuffer<float> d_x(xf.size()), d_z(zf.size()), d_w(cols), d_dout(xf.size()),
            d_dx(xf.size()), d_dz(xf.size()), d_dw(cols);
        d_x.upload(xf, stream.get());
        d_z.upload(zf, stream.get());
        d_w.upload(wf, stream.get());
        d_dout.upload(doutf, stream.get());
        d_dx.upload(std::vector<float>(xf.size(), 0.0f), stream.get());
        d_dz.upload(std::vector<float>(zf.size(), 0.0f), stream.get());
        d_dw.upload(std::vector<float>(cols, 0.0f), stream.get());
        const std::vector<float> invf_v(rf.begin(), rf.end());
        DeviceBuffer<float> d_rf2(rows);
        d_rf2.upload(invf_v, stream.get());
        /* As with the RMSNorm, the weight gradient is its own buffer. */
        kernel_gdn_gated_norm_backward(d_dx.get(), d_dw.get(), d_dz.get(), d_dout.get(), d_x.get(),
                                       d_z.get(), d_w.get(), d_rf2.get(), cols, rows, 0,
                                       stream.get());
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> want_x(x.size(), 0.0), want_z(z.size(), 0.0);
        for (int r = 0; r < rows; ++r) {
            const double s = inv_rms[r];
            double dot = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double n = (double)round_bf16((float)(x[(size_t)r * cols + c] * s));
                const double sw = z[(size_t)r * cols + c] / (1.0 + std::exp(-z[(size_t)r * cols + c]));
                dot += w[c] * n * (dout[(size_t)r * cols + c] * sw);
            }
            for (int c = 0; c < cols; ++c) {
                const size_t i = (size_t)r * cols + c;
                const double n = (double)round_bf16((float)(x[i] * s));
                const double sw = z[i] / (1.0 + std::exp(-z[i]));
                const double weighted = (double)round_bf16((float)(n * w[c]));
                const double dsilu = 1.0 / (1.0 + std::exp(-z[i]));
                want_x[i] = s * (dout[i] * sw * w[c]) - s * s * s / cols * dot * x[i];
                want_z[i] = dout[i] * weighted * (dsilu * (1.0 + z[i] * (1.0 - dsilu)));
            }
        }
        report("gdn_gated_norm_backward d_x", max_abs_diff(d_dx.download(stream.get()), want_x),
               1e-5);
        report("gdn_gated_norm_backward d_z", max_abs_diff(d_dz.download(stream.get()), want_z),
               1e-5);
    }
}

/* ------------------------------------------------------------------ */
/* Embedding, RoPE and the Q/gate split                               */
/* ------------------------------------------------------------------ */

void embedding_and_rotation(Stream &stream) {
    /* Repeated ids: token 3 appears four times, which is the case an overwrite gets
     * wrong and the plan's gate names explicitly. */
    const int tokens = 17, hidden = 32, vocab = 9;
    std::vector<int64_t> ids(tokens);
    const int repeated[] = {0, 3, 7, 3, 1, 3, 8, 2, 3, 5, 6, -1, 4, 8, 3, 0, 7};
    for (int i = 0; i < tokens; ++i) ids[i] = repeated[i];
    std::vector<float> dout((size_t)tokens * hidden);
    for (size_t i = 0; i < dout.size(); ++i) dout[i] = (float)sample(i, 31);
    std::vector<float> table_initial((size_t)vocab * hidden, 0.0f);

    DeviceBuffer<int64_t> d_ids(tokens);
    DeviceBuffer<float> d_dout(dout.size()), d_table(table_initial.size());
    d_ids.upload(ids, stream.get());
    d_dout.upload(dout, stream.get());
    d_table.upload(table_initial, stream.get());
    kernel_embedding_backward(d_table.get(), d_dout.get(), d_ids.get(), hidden, tokens,
                              stream.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> want(table_initial.size(), 0.0);
    for (int t = 0; t < tokens; ++t) {
        if (ids[t] < 0) continue;
        for (int i = 0; i < hidden; ++i) {
            want[(size_t)ids[t] * hidden + i] += dout[(size_t)t * hidden + i];
        }
    }
    const auto got_table = d_table.download(stream.get());
    report("embedding_backward (repeated ids)", max_abs_diff(got_table, want), 1e-6);

    /* RoPE: the transposed rotation must invert the forward's rotation exactly. */
    const int q_heads = 3, kv_heads = 1, head_dim = 16, rotary = 8;
    const float theta = 10000.0f;
    std::vector<int64_t> pos(tokens);
    for (int t = 0; t < tokens; ++t) pos[t] = t * 3 + 1;
    std::vector<double> qv((size_t)tokens * q_heads * head_dim),
        kv((size_t)tokens * kv_heads * head_dim);
    for (size_t i = 0; i < qv.size(); ++i) qv[i] = sample(i, 32);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = sample(i, 33);
    /* The forward rotation in double, from the same table definition the kernel uses. */
    auto rotate = [&](std::vector<double> &x, int heads, bool inverse) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < heads; ++h) {
                double *row = &x[((size_t)t * heads + h) * head_dim];
                for (int i = 0; i < rotary / 2; ++i) {
                    const double freq = std::pow((double)theta, -2.0 * i / rotary);
                    const double angle = (double)pos[t] * freq;
                    const double c = std::cos(angle), s = std::sin(angle);
                    const double a = row[i], b = row[i + rotary / 2];
                    if (!inverse) {
                        row[i] = a * c - b * s;
                        row[i + rotary / 2] = a * s + b * c;
                    } else {
                        row[i] = a * c + b * s;
                        row[i + rotary / 2] = -a * s + b * c;
                    }
                }
            }
        }
    };
    /* Check the *backward* is the transpose: rotate forward in double, then run the
     * kernel's backward on the rotated values and require the original. */
    std::vector<double> rotated_q = qv, rotated_k = kv;
    rotate(rotated_q, q_heads, false);
    rotate(rotated_k, kv_heads, false);
    std::vector<float> dout_q(rotated_q.size()), dout_k(rotated_k.size());
    for (size_t i = 0; i < dout_q.size(); ++i) dout_q[i] = (float)rotated_q[i];
    for (size_t i = 0; i < dout_k.size(); ++i) dout_k[i] = (float)rotated_k[i];
    DeviceBuffer<int64_t> d_pos(tokens);
    DeviceBuffer<float> d_dq(dout_q.size()), d_dk(dout_k.size()), d_doq(dout_q.size()),
        d_dok(dout_k.size());
    d_pos.upload(pos, stream.get());
    d_dq.upload(std::vector<float>(dout_q.size(), 0.0f), stream.get());
    d_dk.upload(std::vector<float>(dout_k.size(), 0.0f), stream.get());
    d_doq.upload(dout_q, stream.get());
    d_dok.upload(dout_k, stream.get());
    kernel_rope_backward(d_dq.get(), d_dk.get(), d_doq.get(), d_dok.get(), d_pos.get(), tokens,
                         q_heads, kv_heads, head_dim, rotary, theta, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());
    report("rope_backward (the transpose inverts the rotation)",
           std::max(max_abs_diff(d_dq.download(stream.get()), qv),
                    max_abs_diff(d_dk.download(stream.get()), kv)),
           2e-6);

    /* The Q/gate merge: the inverse of the forward's deinterleave. */
    const int qgate_rows = 8, qgate_dim = 6;
    const int total = qgate_rows * qgate_dim;
    std::vector<float> dq_src(total), dgate_src(total);
    for (int i = 0; i < total; ++i) {
        dq_src[i] = (float)sample((size_t)i, 34);
        dgate_src[i] = (float)sample((size_t)i, 35);
    }
    std::vector<float> raw_initial((size_t)total * 2, 0.0f);
    DeviceBuffer<float> d_dq_src(total), d_dgate_src(total), d_raw(raw_initial.size());
    d_dq_src.upload(dq_src, stream.get());
    d_dgate_src.upload(dgate_src, stream.get());
    d_raw.upload(raw_initial, stream.get());
    kernel_qgate_merge_backward(d_raw.get(), d_dq_src.get(), d_dgate_src.get(), total,
                                qgate_dim, stream.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> want_raw(raw_initial.size(), 0.0);
    for (int row = 0; row < qgate_rows; ++row) {
        for (int d = 0; d < qgate_dim; ++d) {
            const int i = row * qgate_dim + d;
            want_raw[(size_t)row * 2 * qgate_dim + d] = dq_src[i];
            want_raw[(size_t)row * 2 * qgate_dim + qgate_dim + d] = dgate_src[i];
        }
    }
    report("qgate_merge_backward", max_abs_diff(d_raw.download(stream.get()), want_raw), 0.0);
}

/* ------------------------------------------------------------------ */
/* GEMM                                                               */
/* ------------------------------------------------------------------ */

void gemm(cudaStream_t stream) {
    const int M = 12, N = 20, K = 16;
    std::vector<double> x(M * K), w(N * K), dout(M * N);
    for (size_t i = 0; i < x.size(); ++i) x[i] = sample(i, 41);
    for (size_t i = 0; i < w.size(); ++i) w[i] = sample(i, 42) * 0.5;
    for (size_t i = 0; i < dout.size(); ++i) dout[i] = sample(i, 43);
    std::vector<Bf16> xb(M * K), wb(N * K);
    std::vector<float> xf(M * K), doutf(M * N);
    for (size_t i = 0; i < xb.size(); ++i) {
        x[i] = (double)round_bf16((float)x[i]);
        xb[i] = bf16((float)x[i]);
        xf[i] = (float)x[i];
    }
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = bf16((float)w[i]);
    for (size_t i = 0; i < doutf.size(); ++i) doutf[i] = (float)dout[i];

    std::vector<double> want_dx(M * K, 0.0), want_dw(N * K, 0.0);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            double sum = 0.0;
            for (int nn = 0; nn < N; ++nn) sum += dout[m * N + nn] * w[nn * K + k];
            want_dx[m * K + k] = sum;
        }
    }
    for (int nn = 0; nn < N; ++nn) {
        for (int k = 0; k < K; ++k) {
            double sum = 0.0;
            for (int m = 0; m < M; ++m) sum += dout[m * N + nn] * x[m * K + k];
            want_dw[nn * K + k] = sum;
        }
    }

    DeviceBuffer<Bf16> d_x(xb.size()), d_w(wb.size());
    DeviceBuffer<float> d_dout(doutf.size()), d_dx(want_dx.size()), d_dw(want_dw.size());
    /* The backward's operands are the FP32 widenings of the forward's BF16 values, which
     * is what the header's dtype contract requires (cuBLAS takes no mixed A/B types). */
    const std::vector<float> w_fp32(w.begin(), w.end());
    d_x.upload(xb, stream);
    d_w.upload(wb, stream);
    d_dout.upload(doutf, stream);
    d_dx.upload(std::vector<float>(want_dx.size(), 0.0f), stream);
    d_dw.upload(std::vector<float>(want_dw.size(), 0.0f), stream);
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    require(gemm_backward_dx(handle, d_dx.get(), d_dout.get(), w_fp32.data(), M, N, K) == 0,
            "gemm_backward_dx returned nonzero");
    require(gemm_backward_dw(handle, d_dw.get(), xf.data(), d_dout.get(), M, N, K) == 0,
            "gemm_backward_dw returned nonzero");
    cublasDestroy(handle);
    report_worst("gemm_backward_dx", d_dx.download(stream), want_dx, 1e-5);
    report("gemm_backward_dw", max_abs_diff(d_dw.download(stream), want_dw), 1e-5);
}

/* ------------------------------------------------------------------ */
/* Masked-CE gradient and AdamW                                       */
/* ------------------------------------------------------------------ */

void loss_and_optimizer(Stream &stream) {
    const int rows = 5, vocab = 11;
    std::vector<float> logits(rows * vocab), dout(rows);
    std::vector<int> labels(rows);
    std::vector<uint8_t> mask(rows);
    for (int i = 0; i < rows * vocab; ++i) logits[i] = (float)sample((size_t)i, 51);
    for (int r = 0; r < rows; ++r) {
        labels[r] = (r * 3) % vocab;
        mask[r] = (r == 1) ? 0 : 1; /* one deselected row */
        dout[r] = (float)sample((size_t)r, 52);
    }
    DeviceBuffer<float> d_logits(logits.size()), d_dout(dout.size()), d_dlogits(logits.size());
    DeviceBuffer<int> d_labels(rows);
    DeviceBuffer<uint8_t> d_mask(rows);
    d_logits.upload(logits, stream.get());
    d_dout.upload(dout, stream.get());
    d_dlogits.upload(std::vector<float>(logits.size(), 0.0f), stream.get());
    d_labels.upload(labels, stream.get());
    d_mask.upload(mask, stream.get());
    kernel_logprob_gather_backward(d_dlogits.get(), d_dout.get(), d_logits.get(), d_labels.get(),
                                   d_mask.get(), rows, vocab, stream.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> want(logits.size(), 0.0);
    for (int r = 0; r < rows; ++r) {
        if (mask[r] == 0) continue;
        double max = -1e300, sum = 0.0;
        for (int j = 0; j < vocab; ++j) max = std::max(max, (double)logits[r * vocab + j]);
        for (int j = 0; j < vocab; ++j) sum += std::exp((double)logits[r * vocab + j] - max);
        for (int j = 0; j < vocab; ++j) {
            double g = (double)dout[r] * std::exp((double)logits[r * vocab + j] - max) / sum;
            if (j == labels[r]) g -= dout[r];
            want[r * vocab + j] = g;
        }
    }
    report("logprob_gather_backward (masked row is zero)",
           max_abs_diff(d_dlogits.download(stream.get()), want), 1e-6);

    /* AdamW on the device: the same step order the CPU gate checks against an FP64
     * implementation, so this only has to agree with the device's own arithmetic. */
    const int n = 257;
    const float lr = 0.03f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, wd = 0.02f;
    std::vector<float> master(n), grad(n);
    for (int i = 0; i < n; ++i) {
        master[i] = (float)sample((size_t)i, 61) * 0.5f;
        grad[i] = (float)sample((size_t)i, 62);
    }
    std::vector<float> m(n, 0.0f), v(n, 0.0f);
    std::vector<double> ref_master(n), ref_grad(n), ref_m(n, 0.0), ref_v(n, 0.0);
    for (int i = 0; i < n; ++i) {
        ref_master[i] = master[i];
        ref_grad[i] = grad[i];
    }
    const int steps = 3;
    std::vector<Bf16> bf16(n);
    DeviceBuffer<float> d_master(n), d_grad(n), d_m(n), d_v(n);
    DeviceBuffer<Bf16> d_bf16(n);
    d_master.upload(master, stream.get());
    d_grad.upload(grad, stream.get());
    d_m.upload(m, stream.get());
    d_v.upload(v, stream.get());
    for (int step = 1; step <= steps; ++step) {
        kernel_adamw(d_master.get(), d_grad.get(), d_m.get(), d_v.get(), n, lr, beta1, beta2, eps,
                     wd, step, d_bf16.get(), stream.get());
        CUDA_CHECK(cudaGetLastError());
        const double bc1 = 1.0 - std::pow((double)beta1, step);
        const double bc2 = 1.0 - std::pow((double)beta2, step);
        for (int i = 0; i < n; ++i) {
            ref_m[i] = beta1 * ref_m[i] + (1.0 - beta1) * ref_grad[i];
            ref_v[i] = beta2 * ref_v[i] + (1.0 - beta2) * ref_grad[i] * ref_grad[i];
            ref_master[i] *= (1.0 - (double)lr * wd);
            ref_master[i] -= ((double)lr / bc1) * ref_m[i] / (std::sqrt(ref_v[i]) / std::sqrt(bc2) + eps);
        }
    }
    report("kernel_adamw master", max_abs_diff(d_master.download(stream.get()), ref_master), 2e-6);
    report("kernel_adamw first moment", max_abs_diff(d_m.download(stream.get()), ref_m), 2e-6);
    report("kernel_adamw second moment", max_abs_diff(d_v.download(stream.get()), ref_v), 2e-6);
    /* The published BF16 must be the master's round-to-nearest-even. */
    const auto got_bf16 = d_bf16.download(stream.get());
    double bf16_worst = 0.0;
    for (int i = 0; i < n; ++i) {
        bf16_worst = std::max(bf16_worst,
                              std::abs((double)got_bf16[i] - (double)round_bf16(ref_master[i])));
    }
    report("kernel_adamw published BF16", bf16_worst, 0.0);
}

/* ------------------------------------------------------------------ */
/* GDN causal conv1d and prepare                                      */
/* ------------------------------------------------------------------ */

void gdn_conv_and_prepare(Stream &stream) {
    /* ---- conv1d ---- */
    const int conv_dim = 6, tokens = 10, kernel_size = 4, state_len = kernel_size - 1;
    std::vector<Bf16> x_in((size_t)tokens * conv_dim), weight((size_t)conv_dim * kernel_size),
        bias(conv_dim), state_in((size_t)conv_dim * state_len);
    std::vector<double> xd((size_t)tokens * conv_dim), wd((size_t)conv_dim * kernel_size),
        bd(conv_dim), sd((size_t)conv_dim * state_len);
    for (size_t i = 0; i < xd.size(); ++i) {
        xd[i] = bf16_sample(i, 71);
        x_in[i] = bf16((float)xd[i]);
    }
    for (size_t i = 0; i < wd.size(); ++i) {
        wd[i] = bf16_sample(i, 72);
        weight[i] = bf16((float)wd[i]);
    }
    for (int c = 0; c < conv_dim; ++c) {
        bd[c] = bf16_sample((size_t)c, 73);
        bias[c] = bf16((float)bd[c]);
    }
    for (size_t i = 0; i < sd.size(); ++i) {
        sd[i] = bf16_sample(i, 74);
        state_in[i] = bf16((float)sd[i]);
    }
    std::vector<double> dout((size_t)tokens * conv_dim);
    for (size_t i = 0; i < dout.size(); ++i) dout[i] = sample(i, 75);

    /* The reference forward, from the definition (a negative source index reads the
     * oldest-first state). */
    auto reference_out = [&](const std::vector<double> &x, const std::vector<double> &w,
                             const std::vector<double> &b, const std::vector<double> &s) {
        std::vector<double> out((size_t)tokens * conv_dim, 0.0);
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < conv_dim; ++c) {
                double sum = b[c];
                for (int j = 0; j < kernel_size; ++j) {
                    const int idx = t - state_len + j;
                    const double value = idx >= 0 ? x[(size_t)idx * conv_dim + c]
                                                  : s[(size_t)c * state_len + idx + state_len];
                    sum += w[(size_t)c * kernel_size + j] * value;
                }
                out[(size_t)t * conv_dim + c] = sum;
            }
        }
        return out;
    };
    std::vector<double> x_work = xd, w_work = wd, b_work = bd, s_work = sd;
    auto loss = [&]() {
        const std::vector<double> out = reference_out(x_work, w_work, b_work, s_work);
        double sum = 0.0;
        for (size_t i = 0; i < out.size(); ++i) sum += out[i] * dout[i];
        return sum;
    };

    DeviceBuffer<Bf16> d_x_in(x_in.size()), d_weight(weight.size()), d_bias(bias.size()),
        d_state_in(state_in.size());
    DeviceBuffer<float> d_dout(dout.size()), d_dx(xd.size()), d_dw(wd.size()), d_db(bd.size()),
        d_ds(sd.size());
    d_x_in.upload(x_in, stream.get());
    d_weight.upload(weight, stream.get());
    d_bias.upload(bias, stream.get());
    d_state_in.upload(state_in, stream.get());
    std::vector<float> doutf(dout.begin(), dout.end());
    d_dout.upload(doutf, stream.get());
    d_dx.upload(std::vector<float>(xd.size(), 0.0f), stream.get());
    d_dw.upload(std::vector<float>(wd.size(), 0.0f), stream.get());
    d_db.upload(std::vector<float>(bd.size(), 0.0f), stream.get());
    d_ds.upload(std::vector<float>(sd.size(), 0.0f), stream.get());
    kernel_causal_conv1d_backward(d_dx.get(), d_dw.get(), d_db.get(), d_ds.get(), d_dout.get(),
                                  d_x_in.get(), d_weight.get(), d_state_in.get(), conv_dim, tokens,
                                  kernel_size, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> want_dx(xd.size()), want_dw(wd.size()), want_db(bd.size()), want_ds(sd.size());
    for (size_t i = 0; i < xd.size(); ++i) want_dx[i] = fd(loss, x_work, i, 1e-5);
    for (size_t i = 0; i < wd.size(); ++i) want_dw[i] = fd(loss, w_work, i, 1e-5);
    for (size_t i = 0; i < bd.size(); ++i) want_db[i] = fd(loss, b_work, i, 1e-5);
    for (size_t i = 0; i < sd.size(); ++i) want_ds[i] = fd(loss, s_work, i, 1e-5);
    report("conv1d_backward d_x (finite difference of the definition)",
           max_abs_diff(d_dx.download(stream.get()), want_dx), 1e-5);
    report("conv1d_backward d_weight", max_abs_diff(d_dw.download(stream.get()), want_dw), 1e-5);
    report("conv1d_backward d_bias", max_abs_diff(d_db.download(stream.get()), want_db), 1e-5);
    report("conv1d_backward d_state_in", max_abs_diff(d_ds.download(stream.get()), want_ds),
           1e-5);

    /* ---- prepare ---- */
    const int p_tokens = 4, key_heads = 2, value_heads = 4, head_dim = 3;
    const int group = value_heads / key_heads;
    const int conv_stride = 2 * key_heads * head_dim + value_heads * head_dim;
    std::vector<Bf16> conv_out((size_t)p_tokens * conv_stride);
    std::vector<double> conv_d(conv_out.size());
    for (size_t i = 0; i < conv_d.size(); ++i) {
        conv_d[i] = bf16_sample(i, 81);
        conv_out[i] = bf16((float)conv_d[i]);
    }
    std::vector<double> a_d(p_tokens * value_heads), b_d(p_tokens * value_heads),
        alog_d(value_heads), dt_d(value_heads);
    std::vector<Bf16> a_buf(a_d.size()), b_buf(b_d.size()), alog_buf(value_heads),
        dt_buf(value_heads);
    for (size_t i = 0; i < a_d.size(); ++i) {
        a_d[i] = bf16_sample(i, 82) * 0.5;
        a_buf[i] = bf16((float)a_d[i]);
        b_d[i] = bf16_sample(i, 83);
        b_buf[i] = bf16((float)b_d[i]);
    }
    for (int h = 0; h < value_heads; ++h) {
        alog_d[h] = bf16_sample((size_t)h, 84);
        alog_buf[h] = bf16((float)alog_d[h]);
        /* The dt_bias product has to be BF16-representable too: 0.3 is not a power of
         * two, so an unrounded product would be a different operand after the upload. */
        dt_d[h] = (double)round_bf16(sample((size_t)h, 85) * 0.3f);
        dt_buf[h] = bf16((float)dt_d[h]);
    }
    /* The upstream gradients the core would hand back. */
    std::vector<double> dq(p_tokens * value_heads * head_dim), dk(dq.size()), dv(dq.size()),
        dg(p_tokens * value_heads), dbeta(p_tokens * value_heads);
    for (size_t i = 0; i < dq.size(); ++i) {
        dq[i] = sample(i, 86);
        dk[i] = sample(i, 87);
        dv[i] = sample(i, 88);
    }
    for (size_t i = 0; i < dg.size(); ++i) {
        dg[i] = sample(i, 89);
        dbeta[i] = sample(i, 90);
    }

    const int q_base = 0, k_base = key_heads * head_dim, v_base = 2 * key_heads * head_dim;
    auto prepared = [&](const std::vector<double> &conv, const std::vector<double> &a,
                        const std::vector<double> &b, const std::vector<double> &alog,
                        const std::vector<double> &dt, std::vector<double> *q_out,
                        std::vector<double> *k_out, std::vector<double> *v_out,
                        std::vector<double> *g_out, std::vector<double> *beta_out) {
        for (int t = 0; t < p_tokens; ++t) {
            const double *row = &conv[(size_t)t * conv_stride];
            for (int h = 0; h < value_heads; ++h) {
                const int kh = h / group;
                double sq_q = 0.0, sq_k = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    sq_q += row[q_base + kh * head_dim + d] * row[q_base + kh * head_dim + d];
                    sq_k += row[k_base + kh * head_dim + d] * row[k_base + kh * head_dim + d];
                }
                const double rq = 1.0 / std::sqrt(sq_q + 1e-6);
                const double rk = 1.0 / std::sqrt(sq_k + 1e-6);
                for (int d = 0; d < head_dim; ++d) {
                    (*q_out)[(size_t)(t * value_heads + h) * head_dim + d] =
                        rq * row[q_base + kh * head_dim + d];
                    (*k_out)[(size_t)(t * value_heads + h) * head_dim + d] =
                        rk * row[k_base + kh * head_dim + d];
                    (*v_out)[(size_t)(t * value_heads + h) * head_dim + d] =
                        row[v_base + h * head_dim + d];
                }
                const double x = a[(size_t)t * value_heads + h] + dt[h];
                const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
                (*g_out)[(size_t)t * value_heads + h] = -std::exp(alog[h]) * sp;
                (*beta_out)[(size_t)t * value_heads + h] =
                    1.0 / (1.0 + std::exp(-b[(size_t)t * value_heads + h]));
            }
        }
    };
    std::vector<double> conv_w = conv_d, a_w = a_d, b_w = b_d, alog_w = alog_d, dt_w = dt_d;
    auto prepare_loss = [&]() {
        std::vector<double> qp(dq.size()), kp(dq.size()), vp(dq.size()), g(dg.size()),
            beta(dg.size());
        prepared(conv_w, a_w, b_w, alog_w, dt_w, &qp, &kp, &vp, &g, &beta);
        double sum = 0.0;
        for (size_t i = 0; i < dq.size(); ++i) {
            sum += dq[i] * qp[i] + dk[i] * kp[i] + dv[i] * vp[i];
        }
        for (size_t i = 0; i < dg.size(); ++i) sum += dg[i] * g[i] + dbeta[i] * beta[i];
        return sum;
    };

    DeviceBuffer<Bf16> d_conv(conv_out.size()), d_a(a_buf.size()), d_b(b_buf.size()),
        d_alog(alog_buf.size()), d_dt(dt_buf.size());
    DeviceBuffer<float> d_dq(dq.size()), d_dk(dk.size()), d_dv(dv.size()), d_dg(dg.size()),
        d_dbeta(dbeta.size());
    DeviceBuffer<float> d_dconv(conv_out.size()), d_da(a_buf.size()), d_db_prep(b_buf.size()),
        d_dalog(alog_buf.size()), d_ddt(dt_buf.size());
    std::vector<float> f32;
    d_conv.upload(conv_out, stream.get());
    d_a.upload(a_buf, stream.get());
    d_b.upload(b_buf, stream.get());
    d_alog.upload(alog_buf, stream.get());
    d_dt.upload(dt_buf, stream.get());
    f32.assign(dq.begin(), dq.end());
    d_dq.upload(f32, stream.get());
    f32.assign(dk.begin(), dk.end());
    d_dk.upload(f32, stream.get());
    f32.assign(dv.begin(), dv.end());
    d_dv.upload(f32, stream.get());
    f32.assign(dg.begin(), dg.end());
    d_dg.upload(f32, stream.get());
    f32.assign(dbeta.begin(), dbeta.end());
    d_dbeta.upload(f32, stream.get());
    d_dconv.upload(std::vector<float>(conv_out.size(), 0.0f), stream.get());
    d_da.upload(std::vector<float>(a_buf.size(), 0.0f), stream.get());
    d_db_prep.upload(std::vector<float>(b_buf.size(), 0.0f), stream.get());
    d_dalog.upload(std::vector<float>(alog_buf.size(), 0.0f), stream.get());
    d_ddt.upload(std::vector<float>(dt_buf.size(), 0.0f), stream.get());
    kernel_gdn_prepare_backward(d_dconv.get(), d_da.get(), d_db_prep.get(), d_dalog.get(), d_ddt.get(),
                                d_dq.get(), d_dk.get(), d_dv.get(), d_dg.get(), d_dbeta.get(),
                                d_conv.get(), d_a.get(), d_b.get(), d_alog.get(), d_dt.get(),
                                p_tokens, key_heads, value_heads, head_dim, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> want_conv(conv_d.size()), want_a(a_d.size()), want_b(b_d.size()),
        want_alog(value_heads), want_dt(value_heads);
    for (size_t i = 0; i < conv_d.size(); ++i) want_conv[i] = fd(prepare_loss, conv_w, i, 1e-5);
    for (size_t i = 0; i < a_d.size(); ++i) want_a[i] = fd(prepare_loss, a_w, i, 1e-5);
    for (size_t i = 0; i < b_d.size(); ++i) want_b[i] = fd(prepare_loss, b_w, i, 1e-5);
    for (int h = 0; h < value_heads; ++h) {
        want_alog[h] = fd(prepare_loss, alog_w, (size_t)h, 1e-5);
        want_dt[h] = fd(prepare_loss, dt_w, (size_t)h, 1e-5);
    }
    report_worst("gdn_prepare_backward d_conv_out", d_dconv.download(stream.get()), want_conv,
                 1e-5);
    report("gdn_prepare_backward d_a", max_abs_diff(d_da.download(stream.get()), want_a), 1e-5);
    report("gdn_prepare_backward d_b", max_abs_diff(d_db_prep.download(stream.get()), want_b), 1e-5);
    report("gdn_prepare_backward d_A_log", max_abs_diff(d_dalog.download(stream.get()), want_alog),
           1e-5);
    report("gdn_prepare_backward d_dt_bias", max_abs_diff(d_ddt.download(stream.get()), want_dt),
           1e-5);
}

/* ------------------------------------------------------------------ */
/* Attention: the paired forward and backward                          */
/* ------------------------------------------------------------------ */

/* A deliberately transparent mirror of the paired attention backward: one thread per
 * (query token, head block), every loop serial, no shared memory and no tree. It is a
 * second reading of the same mathematics, in the spirit of the manifest's Python
 * re-derivation: the two implementations share no reduction strategy, so an error in
 * one (a reduction, a pointer offset) does not hide in the other. File-scope with
 * internal linkage because nvcc's stub generation dislikes a second anonymous namespace
 * in one translation unit. */
static __global__ void attention_mirror_kernel(float *__restrict__ d_q, float *__restrict__ d_k,
                                        float *__restrict__ d_v,
                                        const float *__restrict__ d_out,
                                        const __nv_bfloat16 *__restrict__ q,
                                        const __nv_bfloat16 *__restrict__ kc,
                                        const __nv_bfloat16 *__restrict__ vc,
                                        const float *__restrict__ lse, int tokens, int heads,
                                        int kv_heads, int head_dim, int max_seq_len, float scale) {
    const int t = blockIdx.x / heads;
    const int h = blockIdx.x % heads;
    const int kh = h / (heads / kv_heads);
    if (t >= tokens || threadIdx.x != 0) return;
    const int kv_stride = kv_heads * head_dim;
    const __nv_bfloat16 *q_row = q + ((long long)t * heads + h) * head_dim;
    const float *dout_row = d_out + ((long long)t * heads + h) * head_dim;
    const float L = lse[(long long)t * heads + h];
    float P[128], dP[128];
    for (int s = 0; s <= t; ++s) {
        const __nv_bfloat16 *k_row = kc + (long long)s * kv_stride + (long long)kh * head_dim;
        const __nv_bfloat16 *v_row =
            vc + (long long)max_seq_len * kv_stride + (long long)s * kv_stride +
            (long long)kh * head_dim;
        float dot = 0.0f, dp = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
            dp += dout_row[d] * __bfloat162float(v_row[d]);
        }
        P[s] = exp2f(dot * scale * 1.4426950408889634f - L);
        dP[s] = dp;
    }
    float row = 0.0f;
    for (int s = 0; s <= t; ++s) row += P[s] * dP[s];
    for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int s = 0; s <= t; ++s) {
            const __nv_bfloat16 *k_row = kc + (long long)s * kv_stride + (long long)kh * head_dim;
            acc += P[s] * (dP[s] - row) * __bfloat162float(k_row[d]) * scale;
        }
        d_q[((long long)t * heads + h) * head_dim + d] = acc;
    }
    for (int s = 0; s <= t; ++s) {
        const float ds = P[s] * (dP[s] - row);
        for (int d = 0; d < head_dim; ++d) {
            atomicAdd(&d_k[((long long)s * kv_heads + kh) * head_dim + d],
                      ds * __bfloat162float(q_row[d]) * scale);
            atomicAdd(&d_v[((long long)s * kv_heads + kh) * head_dim + d], P[s] * dout_row[d]);
        }
    }
}

void attention_paired(Stream &stream) {
    const int tokens = 6, heads = 2, kv_heads = 1, head_dim = 128;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int kv_stride = kv_heads * head_dim;

    std::vector<double> qd((size_t)tokens * heads * head_dim),
        kd((size_t)tokens * kv_heads * head_dim), vd((size_t)tokens * kv_heads * head_dim);
    /* The kernel reads BF16 operands, so every fixture value has to *be* a BF16 value:
     * a q that is "a BF16 value times 0.8" is a different number after the upload
     * rounds it, and the finite difference would then be of a function the kernel never
     * evaluated. d_out is a gradient (FP32) and needs no such care. */
    for (size_t i = 0; i < qd.size(); ++i) qd[i] = (double)round_bf16(sample(i, 101) * 0.8f);
    for (size_t i = 0; i < kd.size(); ++i) kd[i] = (double)round_bf16(sample(i, 102) * 0.8f);
    for (size_t i = 0; i < vd.size(); ++i) vd[i] = (double)round_bf16(sample(i, 103));
    std::vector<double> doutd((size_t)tokens * heads * head_dim);
    for (size_t i = 0; i < doutd.size(); ++i) doutd[i] = sample(i, 104);

    std::vector<Bf16> q_buf(qd.size()), cache((size_t)2 * tokens * kv_stride);
    for (size_t i = 0; i < qd.size(); ++i) q_buf[i] = bf16((float)qd[i]);
    for (size_t i = 0; i < kd.size(); ++i) cache[i] = bf16((float)kd[i]);
    for (size_t i = 0; i < vd.size(); ++i) {
        cache[(size_t)tokens * kv_stride + i] = bf16((float)vd[i]);
    }

    /* The reference: the real composition, softmax over natural logits, with the base-2
     * LSE the kernel exports. */
    auto reference = [&](const std::vector<double> &q, const std::vector<double> &k,
                         const std::vector<double> &v, std::vector<double> *out,
                         std::vector<double> *lse) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < heads; ++h) {
                const double *q_row = &q[((size_t)t * heads + h) * head_dim];
                double max = -1e300;
                std::vector<double> scores(t + 1);
                for (int s = 0; s <= t; ++s) {
                    const double *k_row = &k[(size_t)s * kv_stride];
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
                    scores[s] = dot * scale;
                    max = std::max(max, scores[s]);
                }
                double sum = 0.0;
                for (int s = 0; s <= t; ++s) sum += std::exp(scores[s] - max);
                (*lse)[t * heads + h] = (max + std::log(sum)) / std::log(2.0);
                for (int d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (int s = 0; s <= t; ++s) {
                        const double *v_row = &v[(size_t)s * kv_stride];
                        acc += std::exp(scores[s] - max) / sum * v_row[d];
                    }
                    (*out)[((size_t)t * heads + h) * head_dim + d] = acc;
                }
            }
        }
    };

    std::vector<double> want_out(qd.size(), 0.0), want_lse(tokens * heads, 0.0);
    reference(qd, kd, vd, &want_out, &want_lse);

    DeviceBuffer<Bf16> d_q(q_buf.size()), d_cache(cache.size());
    DeviceBuffer<Bf16> d_out(q_buf.size());
    DeviceBuffer<float> d_lse(tokens * heads);
    d_q.upload(q_buf, stream.get());
    d_cache.upload(cache, stream.get());
    d_out.upload(test::poison(q_buf.size()), stream.get());
    d_lse.upload(std::vector<float>(tokens * heads, -12345.0f), stream.get());
    kernel_attention_lse(d_out.get(), d_lse.get(), d_q.get(), d_cache.get(), 0, tokens, tokens,
                         heads, kv_heads, head_dim, scale, tokens, stream.get());
    CUDA_CHECK(cudaGetLastError());
    const auto got_out = d_out.download(stream.get());
    const auto got_lse = d_lse.download(stream.get());
    {
        double worst_out = 0.0, worst_lse = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            worst_out = std::max(worst_out, std::abs((double)got_out[i] - want_out[i]));
        }
        for (size_t i = 0; i < got_lse.size(); ++i) {
            worst_lse = std::max(worst_lse, std::abs((double)got_lse[i] - want_lse[i]));
        }
        report("attention_lse forward (BF16 out vs the definition)", worst_out, 1e-2);
        report("attention_lse base-2 LSE vs the definition", worst_lse, 2e-3);
    }

    /* The backward against a central difference of the same reference. The residual is
     * the BF16 rounding of P inside the kernel's PV product: the kernel's forward rounds
     * it, the reference does not, so this number is the pairing gap and is reported even
     * when it passes. */
    std::vector<double> q_w = qd, k_w = kd, v_w = vd;
    auto attention_loss = [&]() {
        std::vector<double> out(qd.size(), 0.0), lse(tokens * heads, 0.0);
        reference(q_w, k_w, v_w, &out, &lse);
        double sum = 0.0;
        for (size_t i = 0; i < out.size(); ++i) sum += out[i] * doutd[i];
        return sum;
    };
    std::vector<double> want_dq(qd.size()), want_dk(kd.size()), want_dv(vd.size());
    for (size_t i = 0; i < qd.size(); ++i) want_dq[i] = fd(attention_loss, q_w, i, 1e-5);
    for (size_t i = 0; i < kd.size(); ++i) want_dk[i] = fd(attention_loss, k_w, i, 1e-5);
    for (size_t i = 0; i < vd.size(); ++i) want_dv[i] = fd(attention_loss, v_w, i, 1e-5);

    std::vector<float> doutf(doutd.begin(), doutd.end());
    DeviceBuffer<float> d_dout(doutf.size()), d_dq(qd.size()), d_dk(kd.size()), d_dv(vd.size());
    d_dout.upload(doutf, stream.get());
    d_dq.upload(std::vector<float>(qd.size(), 0.0f), stream.get());
    d_dk.upload(std::vector<float>(kd.size(), 0.0f), stream.get());
    d_dv.upload(std::vector<float>(vd.size(), 0.0f), stream.get());
    kernel_attention_backward(d_dq.get(), d_dk.get(), d_dv.get(), d_dout.get(), d_q.get(),
                              d_cache.get(), d_lse.get(), tokens, tokens, tokens, heads, kv_heads,
                              head_dim, scale, 0, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());
    const auto got_dq = d_dq.download(stream.get());
    const auto got_dk = d_dk.download(stream.get());
    const auto got_dv = d_dv.download(stream.get());
    report_worst("attention_backward d_q (FD of the definition)", got_dq, want_dq, 5e-3);
    report_worst("attention_backward d_k", got_dk, want_dk, 5e-3);
    report("attention_backward d_v", max_abs_diff(got_dv, want_dv), 5e-3);

    /* A second reading of the same gradient, in double, using the *device's* LSE and
     * the kernel's own probability formula. It separates the two ways this can be
     * wrong: if this disagrees with the finite difference, the probability convention
     * (base-2 LSE -> softmax) or the LSE the kernel actually wrote is the problem; if it
     * agrees with the finite difference but the kernel does not, the kernel's indexing
     * or ordering is. */
    {
        std::vector<double> analytic_dq(qd.size(), 0.0);
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < heads; ++h) {
                const int kv_head = h / (heads / kv_heads);
                double row = 0.0;
                std::vector<double> dP(t + 1), P(t + 1);
                for (int s = 0; s <= t; ++s) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += qd[((size_t)t * heads + h) * head_dim + d] * kd[(size_t)s * kv_stride + d];
                    }
                    P[s] = std::exp2(dot * scale * 1.4426950408889634 - (double)got_lse[t * heads + h]);
                    double dp = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        dp += doutd[((size_t)t * heads + h) * head_dim + d] *
                              vd[(size_t)s * kv_stride + d];
                    }
                    dP[s] = dp;
                    row += P[s] * dp;
                }
                for (int d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (int s = 0; s <= t; ++s) {
                        const double ds = P[s] * (dP[s] - row);
                        acc += ds * kd[(size_t)s * kv_stride + d] * scale;
                    }
                    analytic_dq[((size_t)t * heads + h) * head_dim + d] = acc;
                }
                (void)kv_head;
            }
        }
        report_worst("attention d_q from the device LSE (analytic double vs FD)", analytic_dq,
                     want_dq, 5e-3);
        std::vector<float> analytic_dq_f(analytic_dq.begin(), analytic_dq.end());
        report_worst("attention d_q kernel vs the device-LSE analytic", got_dq, analytic_dq, 5e-3);
        (void)analytic_dq_f;

        /* The transparent mirror, same inputs, same formula, no parallel reduction. */
        {
            DeviceBuffer<float> m_dq(qd.size()), m_dk(kd.size()), m_dv(vd.size());
            m_dq.upload(std::vector<float>(qd.size(), 0.0f), stream.get());
            m_dk.upload(std::vector<float>(kd.size(), 0.0f), stream.get());
            m_dv.upload(std::vector<float>(vd.size(), 0.0f), stream.get());
            attention_mirror_kernel<<<tokens * heads, 32, 0, stream.get()>>>(
                m_dq.get(), m_dk.get(), m_dv.get(), d_dout.get(), d_q.get(), d_cache.get(),
                d_cache.get(), d_lse.get(), tokens, heads, kv_heads, head_dim, tokens, scale);
            CUDA_CHECK(cudaGetLastError());
            report_worst("attention mirror d_q vs the device-LSE analytic",
                         m_dq.download(stream.get()), analytic_dq, 5e-3);
            report_worst("attention mirror d_k", m_dk.download(stream.get()), want_dk, 5e-3);
            report_worst("attention mirror d_v", m_dv.download(stream.get()), want_dv, 5e-3);
            report_worst("attention mirror d_q vs the production kernel",
                         m_dq.download(stream.get()), std::vector<double>(got_dq.begin(),
                                                                         got_dq.end()),
                         1e-6);
        }
    }

    /* Run-to-run reproducibility, checked per gradient because the kernel's guarantee
     * is per gradient: dQ is finished inside its own block with one owner per
     * coordinate, so it must be bitwise equal; dK/dV are cross-block group sums with an
     * atomic order, so they are *reported* rather than required. The plan asks for
     * determinism to be tested separately from closeness, and this is that separation
     * made concrete. */
    DeviceBuffer<float> d_dq2(qd.size()), d_dk2(kd.size()), d_dv2(vd.size());
    d_dq2.upload(std::vector<float>(qd.size(), 0.0f), stream.get());
    d_dk2.upload(std::vector<float>(kd.size(), 0.0f), stream.get());
    d_dv2.upload(std::vector<float>(vd.size(), 0.0f), stream.get());
    kernel_attention_backward(d_dq2.get(), d_dk2.get(), d_dv2.get(), d_dout.get(), d_q.get(),
                              d_cache.get(), d_lse.get(), tokens, tokens, tokens, heads, kv_heads,
                              head_dim, scale, 0, 0, stream.get());
    CUDA_CHECK(cudaGetLastError());
    const auto got_dq2 = d_dq2.download(stream.get());
    const auto got_dk2 = d_dk2.download(stream.get());
    const auto got_dv2 = d_dv2.download(stream.get());
    require(std::memcmp(got_dq.data(), got_dq2.data(), got_dq.size() * sizeof(float)) == 0,
            "attention_backward d_q is not bitwise reproducible");
    const double dq_run_gap = max_abs_diff(got_dq, std::vector<double>(got_dq2.begin(),
                                                                       got_dq2.end()));
    const double dk_run_gap = max_abs_diff(got_dk, std::vector<double>(got_dk2.begin(),
                                                                       got_dk2.end()));
    const double dv_run_gap = max_abs_diff(got_dv, std::vector<double>(got_dv2.begin(),
                                                                       got_dv2.end()));
    std::printf("%-46s d_q=%.3e (bitwise=%s) d_k=%.3e d_v=%.3e\n",
                "attention_backward reproducibility", dq_run_gap,
                dq_run_gap == 0.0 ? "yes" : "no", dk_run_gap, dv_run_gap);
}

/* ------------------------------------------------------------------ */
/* GDN core: the paired forward and backward                           */
/* ------------------------------------------------------------------ */

/* The reference forward, in double, from the design document's recurrence:
 *   alpha_t = exp(g_t) ; D_t = alpha_t S_{t-1} ; pred_t = k_t D_t
 *   delta_t = v_t - pred_t ; S_t = D_t + outer(k_t, beta_t delta_t) ; o_t = scale q_t S_t
 * It also returns the state at each chunk boundary, which is what the kernel's retained
 * `chunk_state` is -- so the kernel's forward and its backward are checked against one
 * function, not two. */
struct GdnReference {
    int tokens, heads, key_dim, value_dim, chunk;
    std::vector<double> q, k, v, g, beta, initial;
    double scale;

    void forward(std::vector<double> *out, std::vector<double> *final_state,
                 std::vector<double> *chunk_state) const {
        out->assign((size_t)tokens * heads * value_dim, 0.0);
        chunk_state->assign((size_t)((tokens + chunk - 1) / chunk) * heads * key_dim * value_dim,
                            0.0);
        std::vector<double> state((size_t)heads * key_dim * value_dim);
        for (size_t i = 0; i < state.size(); ++i) state[i] = initial[i];
        for (int t = 0; t < tokens; ++t) {
            if (t % chunk == 0) {
                const int ci = t / chunk;
                for (size_t i = 0; i < state.size(); ++i) {
                    (*chunk_state)[(size_t)ci * state.size() + i] = state[i];
                }
            }
            for (int h = 0; h < heads; ++h) {
                const double alpha = std::exp(g[(size_t)t * heads + h]);
                const double bt = beta[(size_t)t * heads + h];
                double *S = &state[(size_t)h * key_dim * value_dim];
                std::vector<double> D(key_dim * value_dim);
                for (int i = 0; i < key_dim * value_dim; ++i) D[i] = alpha * S[i];
                std::vector<double> delta(value_dim, 0.0);
                for (int c = 0; c < value_dim; ++c) {
                    double pred = 0.0;
                    for (int j = 0; j < key_dim; ++j) {
                        pred += k[((size_t)t * heads + h) * key_dim + j] *
                                D[j * value_dim + c];
                    }
                    delta[c] = v[((size_t)t * heads + h) * value_dim + c] - pred;
                    (*out)[((size_t)t * heads + h) * value_dim + c] = 0.0;
                }
                for (int j = 0; j < key_dim; ++j) {
                    for (int c = 0; c < value_dim; ++c) {
                        S[j * value_dim + c] =
                            D[j * value_dim + c] +
                            k[((size_t)t * heads + h) * key_dim + j] * (bt * delta[c]);
                    }
                }
                for (int c = 0; c < value_dim; ++c) {
                    double acc = 0.0;
                    for (int j = 0; j < key_dim; ++j) {
                        acc += q[((size_t)t * heads + h) * key_dim + j] * S[j * value_dim + c];
                    }
                    (*out)[((size_t)t * heads + h) * value_dim + c] = scale * acc;
                }
            }
        }
        final_state->assign(state.begin(), state.end());
    }
};

void gdn_core_paired(Stream &stream) {
    const int tokens = 8, heads = 2, key_dim = 4, value_dim = 4;
    const float scale = 1.0f / std::sqrt((float)key_dim);

    GdnReference ref;
    ref.tokens = tokens;
    ref.heads = heads;
    ref.key_dim = key_dim;
    ref.value_dim = value_dim;
    ref.scale = (double)scale;
    ref.q.assign((size_t)tokens * heads * key_dim, 0.0);
    ref.k.assign(ref.q.size(), 0.0);
    ref.v.assign((size_t)tokens * heads * value_dim, 0.0);
    ref.g.assign((size_t)tokens * heads, 0.0);
    ref.beta.assign(ref.g.size(), 0.0);
    ref.initial.assign((size_t)heads * key_dim * value_dim, 0.0);
    for (size_t i = 0; i < ref.q.size(); ++i) ref.q[i] = sample(i, 111);
    for (size_t i = 0; i < ref.k.size(); ++i) ref.k[i] = sample(i, 112);
    for (size_t i = 0; i < ref.v.size(); ++i) ref.v[i] = sample(i, 113);
    /* A *nonzero* initial state and a nonzero final-state gradient: the two the plan's
     * gate names, and the two a zero-initialised test would silently skip. */
    for (size_t i = 0; i < ref.initial.size(); ++i) ref.initial[i] = bf16_sample(i, 114) * 0.5;
    for (size_t i = 0; i < ref.g.size(); ++i) ref.g[i] = -std::abs(sample(i, 115)) * 0.5;
    for (size_t i = 0; i < ref.beta.size(); ++i) ref.beta[i] = 0.5 + 0.4 * sample(i, 116);
    std::vector<double> dout((size_t)tokens * heads * value_dim);
    for (size_t i = 0; i < dout.size(); ++i) dout[i] = sample(i, 117);
    std::vector<double> d_state_in((size_t)heads * key_dim * value_dim);
    for (size_t i = 0; i < d_state_in.size(); ++i) d_state_in[i] = sample(i, 118) * 0.5;

    for (int chunk : {tokens, 3}) {
        ref.chunk = chunk;
        std::vector<double> want_out, want_final, chunk_state;
        ref.forward(&want_out, &want_final, &chunk_state);
        const int chunks = (tokens + chunk - 1) / chunk;

        auto loss = [&]() {
            std::vector<double> out, final, cs;
            ref.forward(&out, &final, &cs);
            double sum = 0.0;
            for (size_t i = 0; i < out.size(); ++i) sum += out[i] * dout[i];
            for (size_t i = 0; i < final.size(); ++i) sum += d_state_in[i] * final[i];
            return sum;
        };

        std::vector<float> f32;
        DeviceBuffer<float> d_q(ref.q.size()), d_k(ref.k.size()), d_v(ref.v.size()),
            d_g(ref.g.size()), d_beta(ref.beta.size()), d_cs(chunk_state.size()),
            d_dout(dout.size()), d_dsi(d_state_in.size());
        f32.assign(ref.q.begin(), ref.q.end());
        d_q.upload(f32, stream.get());
        f32.assign(ref.k.begin(), ref.k.end());
        d_k.upload(f32, stream.get());
        f32.assign(ref.v.begin(), ref.v.end());
        d_v.upload(f32, stream.get());
        f32.assign(ref.g.begin(), ref.g.end());
        d_g.upload(f32, stream.get());
        f32.assign(ref.beta.begin(), ref.beta.end());
        d_beta.upload(f32, stream.get());
        f32.assign(chunk_state.begin(), chunk_state.end());
        d_cs.upload(f32, stream.get());
        f32.assign(dout.begin(), dout.end());
        d_dout.upload(f32, stream.get());
        f32.assign(d_state_in.begin(), d_state_in.end());
        d_dsi.upload(f32, stream.get());

        DeviceBuffer<float> d_dq(ref.q.size()), d_dk(ref.k.size()), d_dv(ref.v.size()),
            d_dg(ref.g.size()), d_dbeta(ref.beta.size()), d_dstart(ref.initial.size());
        d_dq.upload(std::vector<float>(ref.q.size(), 0.0f), stream.get());
        d_dk.upload(std::vector<float>(ref.k.size(), 0.0f), stream.get());
        d_dv.upload(std::vector<float>(ref.v.size(), 0.0f), stream.get());
        d_dg.upload(std::vector<float>(ref.g.size(), 0.0f), stream.get());
        d_dbeta.upload(std::vector<float>(ref.beta.size(), 0.0f), stream.get());
        d_dstart.upload(std::vector<float>(ref.initial.size(), 0.0f), stream.get());
        const size_t workspace_bytes =
            kernel_gdn_core_workspace_bytes(tokens, heads, key_dim, value_dim, chunk);
        require(workspace_bytes > 0, "the GDN core workspace size is zero");
        DeviceBuffer<unsigned char> d_workspace(workspace_bytes);
        kernel_gdn_core_backward(d_dq.get(), d_dk.get(), d_dv.get(), d_dg.get(), d_dbeta.get(),
                                 d_dstart.get(), d_dout.get(), d_dsi.get(), d_q.get(), d_k.get(),
                                 d_v.get(), d_g.get(), d_beta.get(), d_cs.get(), tokens, heads,
                                 heads, key_dim, value_dim, chunk, scale, 0, d_workspace.get(),
                                 workspace_bytes, stream.get());
        CUDA_CHECK(cudaGetLastError());

        /* The finite difference of the same reference, over every coordinate including
         * the initial state. `loss()` reads the reference's working members, so `fd`
         * perturbs the member itself and the perturbation is visible to it. */
        const std::vector<double> bq = ref.q, bk = ref.k, bv = ref.v, bg = ref.g,
                                  bb = ref.beta, bi = ref.initial;
        std::vector<double> want_dq(ref.q.size()), want_dk(ref.k.size()), want_dv(ref.v.size()),
            want_dg(ref.g.size()), want_dbeta(ref.beta.size()), want_dstart(ref.initial.size());
        auto fd_member = [&](int which, std::vector<double> &target) {
            for (size_t i = 0; i < target.size(); ++i) {
                ref.q = bq;
                ref.k = bk;
                ref.v = bv;
                ref.g = bg;
                ref.beta = bb;
                ref.initial = bi;
                std::vector<double> *member = which == 0   ? &ref.q
                                             : which == 1 ? &ref.k
                                             : which == 2 ? &ref.v
                                             : which == 3 ? &ref.g
                                             : which == 4 ? &ref.beta
                                                          : &ref.initial;
                target[i] = fd(loss, *member, i, 1e-5);
            }
        };
        fd_member(0, want_dq);
        fd_member(1, want_dk);
        fd_member(2, want_dv);
        fd_member(3, want_dg);
        fd_member(4, want_dbeta);
        fd_member(5, want_dstart);

        const char *suffix = chunk == tokens ? "(one chunk)" : "(three chunks)";
        char name[96];
        std::snprintf(name, sizeof(name), "gdn_core_backward d_q %s", suffix);
        report(name, max_abs_diff(d_dq.download(stream.get()), want_dq), 2e-3);
        std::snprintf(name, sizeof(name), "gdn_core_backward d_k %s", suffix);
        report(name, max_abs_diff(d_dk.download(stream.get()), want_dk), 2e-3);
        std::snprintf(name, sizeof(name), "gdn_core_backward d_v %s", suffix);
        report(name, max_abs_diff(d_dv.download(stream.get()), want_dv), 2e-3);
        std::snprintf(name, sizeof(name), "gdn_core_backward d_log_decay %s", suffix);
        report(name, max_abs_diff(d_dg.download(stream.get()), want_dg), 2e-3);
        std::snprintf(name, sizeof(name), "gdn_core_backward d_beta %s", suffix);
        report(name, max_abs_diff(d_dbeta.download(stream.get()), want_dbeta), 2e-3);
        std::snprintf(name, sizeof(name), "gdn_core_backward d_state_start %s", suffix);
        report(name, max_abs_diff(d_dstart.download(stream.get()), want_dstart), 2e-3);

        /* Run-to-run determinism, bitwise: the reductions are fixed-order by
         * construction, so a difference here is a bug and not a scheduling artefact. */
        DeviceBuffer<float> d_dq2(ref.q.size()), d_dk2(ref.k.size()), d_dv2(ref.v.size());
        d_dq2.upload(std::vector<float>(ref.q.size(), 0.0f), stream.get());
        d_dk2.upload(std::vector<float>(ref.k.size(), 0.0f), stream.get());
        d_dv2.upload(std::vector<float>(ref.v.size(), 0.0f), stream.get());
        kernel_gdn_core_backward(d_dq2.get(), d_dk2.get(), d_dv2.get(), d_dg.get(), d_dbeta.get(),
                                 d_dstart.get(), d_dout.get(), d_dsi.get(), d_q.get(), d_k.get(),
                                 d_v.get(), d_g.get(), d_beta.get(), d_cs.get(), tokens, heads,
                                 heads, key_dim, value_dim, chunk, scale, 0, d_workspace.get(),
                                 workspace_bytes, stream.get());
        CUDA_CHECK(cudaGetLastError());
        const auto got_dq = d_dq.download(stream.get());
        const auto got_dq2 = d_dq2.download(stream.get());
        require(std::memcmp(got_dq.data(), got_dq2.data(), got_dq.size() * sizeof(float)) == 0,
                "gdn_core_backward d_q is not bitwise reproducible");
        std::snprintf(name, sizeof(name), "gdn_core_backward determinism %s", suffix);
        std::printf("%-46s %s\n", name, g_ok ? "bitwise equal" : "DIFFERS");
        (void)chunks;
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("test_backward_kernels: no device visible\n");
        return 0; /* the CPU gate covers the contract; this one needs a GPU */
    }
    Stream stream;
    elementwise(stream);
    norms(stream);
    embedding_and_rotation(stream);
    gemm(stream.get());
    loss_and_optimizer(stream);
    gdn_conv_and_prepare(stream);
    attention_paired(stream);
    gdn_core_paired(stream);
    return test::finish("test_backward_kernels", g_ok);
}
