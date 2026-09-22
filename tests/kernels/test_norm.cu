/* RMSNorm variants: the Gemma form (weight + 1) and the plain form, each against
 * a CPU reference, at shapes that match the model's use (per-head q/k norm with
 * many rows and a narrow width, and the full hidden width). */
#include "flashinfer_ops.h"
#include "test_utils.h"

#include <cmath>

using test::Bf16;
using test::DeviceBuffer;

namespace {

std::vector<Bf16> reference_norm(const std::vector<Bf16> &input, const std::vector<Bf16> &weight,
                                 int rows, int cols, float eps, bool gemma) {
    std::vector<Bf16> out(input.size());
    for (int r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double x = (double)test::value(input[(size_t)r * cols + c]);
            sum_sq += x * x;
        }
        const double rms_rcp = 1.0 / std::sqrt(sum_sq / cols + eps);
        for (int c = 0; c < cols; ++c) {
            const double x = (double)test::value(input[(size_t)r * cols + c]);
            const double w = (double)test::value(weight[c]);
            const double scale = gemma ? (1.0 + w) : w;
            out[(size_t)r * cols + c] = test::bf16((float)(x * rms_rcp * scale));
        }
    }
    return out;
}

bool run_case(const char *name, int rows, int cols, bool gemma, test::Stream &stream) {
    std::vector<Bf16> input((size_t)rows * cols), weight(cols);
    for (size_t i = 0; i < input.size(); ++i) input[i] = test::bf16(test::sample(i, 23));
    for (int c = 0; c < cols; ++c) weight[c] = test::bf16(0.5f + test::sample(c, 29) * 0.5f);
    const float eps = 1e-6f;

    DeviceBuffer<Bf16> d_input(input.size()), d_weight(weight.size()), d_out(input.size());
    d_input.upload(input, stream.get());
    d_weight.upload(weight, stream.get());
    if (gemma) {
        kernel_gemma_rms_norm(d_out.get(), d_input.get(), d_weight.get(), cols, rows, eps,
                              stream.get());
    } else {
        kernel_rms_norm_plain(d_out.get(), d_input.get(), d_weight.get(), cols, rows, eps,
                              stream.get());
    }
    CUDA_CHECK(cudaGetLastError());
    const auto want = reference_norm(input, weight, rows, cols, eps, gemma);
    return test::compare(name, d_out.download(stream.get()), want, 2e-2, 2e-2);
}

}  // namespace

int main() {
    bool ok = true;
    test::Stream stream;
    /* Per-head q/k norm: 128 rows x 128 cols; hidden width: 4 rows x 256 cols. */
    ok &= run_case("plain q/k (128x128)", 128, 128, false, stream);
    ok &= run_case("gemma q/k (128x128)", 128, 128, true, stream);
    ok &= run_case("plain hidden (4x256)", 4, 256, false, stream);
    ok &= run_case("gemma hidden (4x256)", 4, 256, true, stream);
    ok &= run_case("plain single row (1x128)", 1, 128, false, stream);
    return test::finish("test_norm", ok);
}
