/* Cross-device primitives: event-ordered copies and the leader-based all-reduce.
 * Needs 2 GPUs; with fewer visible devices it reports SKIP and succeeds. */
#include "kernels.h"
#include "layers.h"
#include "test_utils.h"

using test::Bf16;
using test::DeviceBuffer;

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("test_collective: %d device(s) visible\n", device_count);
    if (device_count < 2) {
        std::printf("test_collective: SKIP (needs 2 GPUs)\n");
        return EXIT_SUCCESS;
    }
    bool ok = true;
    const int elements = 256;
    const size_t bytes = elements * sizeof(Bf16);

    cudaStream_t s0 = nullptr, s1 = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&e0, cudaEventDisableTiming));
    DeviceBuffer<Bf16> a(elements);          // device 0
    DeviceBuffer<Bf16> staging(elements);    // device 0 (all-reduce leader staging)
    DeviceBuffer<Bf16> values(elements);     // device 0 (async producer input)
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
    DeviceBuffer<Bf16> b(elements);          // device 1

    peer_probe_all((const int[]){0, 1}, 2, 0);

    // --- 1) Event-ordered copy: the value is produced asynchronously on the
    //        source stream, so a copy without host synchronization must still see
    //        the produced data.
    std::vector<Bf16> produced(elements);
    for (int i = 0; i < elements; ++i) produced[i] = test::bf16(static_cast<float>(i) * 0.5f);
    values.upload(produced, s0);
    CUDA_CHECK(cudaMemsetAsync(a.get(), 0, bytes, s0));
    CUDA_CHECK(cudaSetDevice(0));
    kernel_residual_add(a.get(), values.get(), elements, s0);
    CUDA_CHECK(cudaGetLastError());
    int status = copy_across_devices(0, s0, &e0, 1, s1, b.get(), a.get(), bytes);
    if (status != 0) {
        std::fprintf(stderr, "copy_across_devices failed: %s\n", cudaGetErrorString((cudaError_t)status));
        return EXIT_FAILURE;
    }
    ok &= test::compare("copy(producer visible without host sync)", b.download(s1), produced, 0.0, 0.0);

    // --- 2) All-reduce across the two devices: small integers stay exact in bf16.
    std::vector<Bf16> left(elements), right(elements), expected(elements);
    for (int i = 0; i < elements; ++i) {
        left[i] = test::bf16(static_cast<float>(i));
        right[i] = test::bf16(static_cast<float>(2 * i));
        expected[i] = test::bf16(static_cast<float>(3 * i));
    }
    a.upload(left, s0);
    b.upload(right, s1);
    __nv_bfloat16 *buffers[2] = {a.get(), b.get()};
    cudaStream_t streams[2] = {s0, s1};
    cudaEvent_t events[2] = {e0, e1};
    status = allreduce_sum_bf16((const int[]){0, 1}, streams, events, 2, buffers, staging.get(), elements);
    if (status != 0) {
        std::fprintf(stderr, "allreduce_sum_bf16 failed: %s\n", cudaGetErrorString((cudaError_t)status));
        return EXIT_FAILURE;
    }
    ok &= test::compare("allreduce on device 0", a.download(s0), expected, 1e-3, 1e-3);
    ok &= test::compare("allreduce on device 1", b.download(s1), expected, 1e-3, 1e-3);

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaEventDestroy(e1));
    return test::finish("test_collective", ok);
}
