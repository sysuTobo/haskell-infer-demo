#ifndef HASKELL_INFER_TEST_UTILS_H
#define HASKELL_INFER_TEST_UTILS_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace test {

using Bf16 = __nv_bfloat16;

inline void check_cuda(cudaError_t status, const char *call,
                       const char *file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, call,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) ::test::check_cuda((call), #call, __FILE__, __LINE__)

// Host-only round-to-nearest-even reference, independent of device conversion.
inline float round_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if (bits & 0x007fffffu) bits |= 0x00400000u;  // Preserve NaN.
    } else {
        bits += 0x7fffu + ((bits >> 16) & 1u);
    }
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline Bf16 bf16(float value) { return __float2bfloat16_rn(value); }
inline double value(Bf16 x) { return __bfloat162float(x); }
inline double value(int x) { return x; }  // integer arrays (ids, indices) are comparable too
inline double value(float x) { return x; }
inline double value(double x) { return x; }

inline std::vector<Bf16> poison(size_t count) {
    return std::vector<Bf16>(count, bf16(std::numeric_limits<float>::quiet_NaN()));
}

// Index-based integer hash: fixed fixtures, no libc PRNG or random_device.
inline float sample(size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) + salt * 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    x ^= x >> 16;
    return (static_cast<int>(x & 0xffffu) - 32768) / 32768.0f;
}

class Stream {
public:
    cudaStream_t get() const { return stream_; }
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { CUDA_CHECK(cudaStreamDestroy(stream_)); }
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
private:
    cudaStream_t stream_ = nullptr;
};

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)));
    }
    ~DeviceBuffer() { CUDA_CHECK(cudaFree(data_)); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    T *get() const { return data_; }

    void upload(const std::vector<T> &host, cudaStream_t stream) {
        if (host.size() != count_) {
            std::fprintf(stderr, "DeviceBuffer upload size mismatch\n");
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMemcpyAsync(data_, host.data(), count_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
        // Also makes temporary host fixtures safe to destroy after upload.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> host(count_);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), data_, count_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return host;
    }
private:
    T *data_ = nullptr;
    size_t count_;
};

// Check every element, including nonfinites (NaN must never pass a tolerance
// comparison). Report a few examples, but keep counting through the whole tensor.
template <typename Actual, typename Expected>
bool compare(const std::string &name, const std::vector<Actual> &actual,
             const std::vector<Expected> &expected, double atol, double rtol) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: size mismatch %zu != %zu\n", name.c_str(),
                     actual.size(), expected.size());
        return false;
    }
    size_t errors = 0, nonfinite = 0;
    double max_error = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double got = value(actual[i]), want = value(expected[i]);
        const bool finite = std::isfinite(got) && std::isfinite(want);
        const double error = finite ? std::abs(got - want)
                                    : std::numeric_limits<double>::infinity();
        max_error = std::max(max_error, error);
        if (!finite) ++nonfinite;
        if (!finite || error > atol + rtol * std::abs(want)) {
            if (errors < 4) {
                std::fprintf(stderr, "%s[%zu]: got %.9g, expected %.9g (error %.9g)\n",
                             name.c_str(), i, got, want, error);
            }
            ++errors;
        }
    }
    std::printf("%s: %s, errors=%zu/%zu, nonfinite=%zu, max_abs=%.9g\n",
                name.c_str(), errors ? "FAIL" : "PASS", errors, actual.size(),
                nonfinite, max_error);
    return errors == 0;
}

inline int finish(const char *name, bool ok) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace test

#endif
