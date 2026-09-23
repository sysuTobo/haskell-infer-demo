/* Cross-device primitives: event-ordered copies and the leader-based all-reduce.
 * Needs 2 GPUs; with fewer visible devices it reports SKIP and succeeds.
 *
 * The all-reduce is exercised back to back without host synchronization in
 * between, with the leader reusing its own buffer as soon as the call returns:
 * that reuse is exactly what the receiver completion events must order behind
 * the peer reads, so a missing dependency shows up as a corrupted broadcast
 * rather than as a silent assumption. */
#include "kernels.h"
#include "layers.h"
#include "test_utils.h"

#include <string>

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
    cudaEvent_t d0 = nullptr, d1 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&e0, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d0, cudaEventDisableTiming));
    DeviceBuffer<Bf16> a(elements);          // device 0
    DeviceBuffer<Bf16> staging(elements);    // device 0 (all-reduce leader staging)
    DeviceBuffer<Bf16> values(elements);     // device 0 (async producer input)
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&d1, cudaEventDisableTiming));
    DeviceBuffer<Bf16> b(elements);          // device 1

    const int reachable = peer_probe_all((const int[]){0, 1}, 2, 0);
    std::printf("test_collective: %s\n",
                reachable > 0 ? "peer access present: the async peer path is exercised"
                              : "no peer access: the driver's staging path is exercised");

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
    cudaEvent_t done[2] = {d0, d1};
    status = allreduce_sum_bf16((const int[]){0, 1}, streams, events, done, 2, buffers,
                                staging.get(), elements);
    if (status != 0) {
        std::fprintf(stderr, "allreduce_sum_bf16 failed: %s\n", cudaGetErrorString((cudaError_t)status));
        return EXIT_FAILURE;
    }
    ok &= test::compare("allreduce on device 0", a.download(s0), expected, 1e-3, 1e-3);
    ok &= test::compare("allreduce on device 1", b.download(s1), expected, 1e-3, 1e-3);

    // --- 3) Back-to-back rounds with the leader reusing its buffer immediately.
    //        The transfers are issued with cudaMemcpyAsync (DeviceBuffer's
    //        upload/download synchronize the stream, which would hide the race).
    //        A wide buffer keeps the copies in flight long enough for the reuse
    //        to overlap them.
    {
        const int wide = 1 << 20;  // 2 MiB of bf16 per rank
        const size_t wide_bytes = (size_t)wide * sizeof(Bf16);
        /* Each buffer must live on its rank's device; DeviceBuffer allocates on
         * whatever device is current when it is constructed. */
        CUDA_CHECK(cudaSetDevice(0));
        DeviceBuffer<Bf16> wide_a(wide), wide_staging(wide);
        CUDA_CHECK(cudaSetDevice(1));
        DeviceBuffer<Bf16> wide_b(wide);
        std::vector<Bf16> host_left(wide), host_right(wide), host_sum(wide), host_reuse(wide);
        std::vector<Bf16> got_left(wide), got_right(wide);
        __nv_bfloat16 *wide_buffers[2] = {wide_a.get(), wide_b.get()};

        for (int round = 0; round < 8; ++round) {
            for (int i = 0; i < wide; ++i) {
                const int v = (i + round) % 17;  // sums stay exact in bf16
                host_left[i] = test::bf16(static_cast<float>(v));
                host_right[i] = test::bf16(static_cast<float>(2 * v));
                host_sum[i] = test::bf16(static_cast<float>(3 * v));
                host_reuse[i] = test::bf16(static_cast<float>(v - 8));
            }
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(wide_a.get(), host_left.data(), wide_bytes,
                                       cudaMemcpyHostToDevice, s0));
            CUDA_CHECK(cudaSetDevice(1));
            CUDA_CHECK(cudaMemcpyAsync(wide_b.get(), host_right.data(), wide_bytes,
                                       cudaMemcpyHostToDevice, s1));
            status = allreduce_sum_bf16((const int[]){0, 1}, streams, events, done, 2,
                                        wide_buffers, wide_staging.get(), wide);
            if (status != 0) {
                std::fprintf(stderr, "round %d: allreduce failed: %s\n", round,
                             cudaGetErrorString((cudaError_t)status));
                ok = false;
                break;
            }
            // The leader reuses its own buffer at once: this write must be
            // ordered behind the receiver's read of it.
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(wide_a.get(), host_reuse.data(), wide_bytes,
                                       cudaMemcpyHostToDevice, s0));

            CUDA_CHECK(cudaSetDevice(1));
            CUDA_CHECK(cudaMemcpyAsync(got_right.data(), wide_b.get(), wide_bytes,
                                       cudaMemcpyDeviceToHost, s1));
            CUDA_CHECK(cudaStreamSynchronize(s1));
            ok &= test::compare("round " + std::to_string(round) + ": broadcast reached device 1",
                                got_right, host_sum, 0.0, 0.0);

            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(got_left.data(), wide_a.get(), wide_bytes,
                                       cudaMemcpyDeviceToHost, s0));
            CUDA_CHECK(cudaStreamSynchronize(s0));
            ok &= test::compare("round " + std::to_string(round) + ": leader buffer reused",
                                got_left, host_reuse, 0.0, 0.0);
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(d0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(d1));
    return test::finish("test_collective", ok);
}
