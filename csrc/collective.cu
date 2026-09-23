/**
 * collective.cu - Cross-device primitives for multi-GPU execution.
 *
 * Single process, one stream per device, host-issued ordering: instead of
 * synchronizing the producer stream after every kernel (which idles the GPU), an
 * event recorded on the producer stream is waited on by the consumer stream.
 * No NCCL: at 2-8 devices inside one process the copies and the reduction are
 * cheaper than a communicator and the demo stays dependency-free.
 *
 * Peer access is probed once at startup and reported; when it is unavailable the
 * copies still work (CUDA stages them through the host).
 */
#include "layers.h"
#include "kernels.h"

#include <cstdio>

int peer_probe_all(const int *devices, int count, int enable) {
    int reachable = 0;
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            if (i == j) continue;
            int can_access = 0;
            cudaError_t error = cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]);
            if (error != cudaSuccess) {
                fprintf(stderr, "[engine] cudaDeviceCanAccessPeer(%d,%d) failed: %s\n",
                        devices[i], devices[j], cudaGetErrorString(error));
                continue;
            }
            if (!can_access) continue;
            ++reachable;
            if (enable) {
                cudaSetDevice(devices[i]);
                cudaError_t enabled = cudaDeviceEnablePeerAccess(devices[j], 0);
                if (enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "[engine] peer access %d->%d not enabled: %s\n",
                            devices[i], devices[j], cudaGetErrorString(enabled));
                }
                /* Peer access is process-wide and a second engine re-enables the
                 * same pairs; that "already enabled" code is sticky and would be
                 * picked up by the next cudaGetLastError() check, so consume it. */
                (void)cudaGetLastError();
            }
        }
    }
    fprintf(stderr, "[engine] peer access: %d of %d ordered device pairs reachable%s\n",
            reachable, count * (count - 1),
            reachable ? "" : " (cross-device copies go through the host)");
    /* Probing reports through stderr, never through the sticky error state: an
     * expected "already enabled" code or a failed capability query would
     * otherwise surface as a misleading error on the next checked CUDA call. */
    (void)cudaGetLastError();
    return reachable;
}

int copy_across_devices(int from_device, cudaStream_t from_stream, cudaEvent_t *from_event,
                        int to_device, cudaStream_t to_stream,
                        void *dst, const void *src, size_t bytes) {
    if (from_device == to_device) {
        cudaError_t error = cudaSetDevice(to_device);
        if (error != cudaSuccess) return (int)error;
        return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, to_stream);
    }
    cudaError_t error = cudaSetDevice(from_device);
    if (error != cudaSuccess) return (int)error;
    error = cudaEventRecord(*from_event, from_stream);
    if (error != cudaSuccess) return (int)error;

    error = cudaSetDevice(to_device);
    if (error != cudaSuccess) return (int)error;
    /* The destination stream alone does not order reads after the producer's
     * kernels; the recorded event supplies that ordering without a host sync. */
    error = cudaStreamWaitEvent(to_stream, *from_event, 0);
    if (error != cudaSuccess) return (int)error;
    return (int)cudaMemcpyPeerAsync(dst, to_device, src, from_device, bytes, to_stream);
}

int allreduce_sum_bf16(const int *devices, cudaStream_t *streams, cudaEvent_t *events,
                       cudaEvent_t *done_events, int count, __nv_bfloat16 **buffers,
                       __nv_bfloat16 *leader_staging, size_t elements) {
    if (count < 2) return 0;
    const size_t bytes = elements * sizeof(__nv_bfloat16);

    /* Reduce: each rank's buffer is read on the leader stream, ordered by the
     * event its own stream recorded when it produced the data. */
    for (int i = 1; i < count; ++i) {
        int status = copy_across_devices(devices[i], streams[i], &events[i],
                                         devices[0], streams[0],
                                         leader_staging, buffers[i], bytes);
        if (status != 0) return status;
        /* Runs on the leader stream, ordered after the copy by the event above. */
        cudaError_t error = cudaSetDevice(devices[0]);
        if (error != cudaSuccess) return (int)error;
        kernel_residual_add(buffers[0], leader_staging, (int)elements, streams[0]);
        error = cudaGetLastError();
        if (error != cudaSuccess) return (int)error;
    }

    /* Broadcast: every rank copies the leader's buffer. The copy is enqueued on
     * the receiver's stream but *reads* devices[0]'s memory, so the leader stream
     * has no ordering with it. Each receiver therefore records a completion
     * event, and the leader waits on all of them afterwards: a later write to
     * buffers[0] (the next sublayer's partial sum, or the next round's reduce)
     * is then ordered after every receiver finished reading. */
    for (int i = 1; i < count; ++i) {
        int status = copy_across_devices(devices[0], streams[0], &events[0],
                                         devices[i], streams[i],
                                         buffers[i], buffers[0], bytes);
        if (status != 0) return status;
        cudaError_t error = cudaSetDevice(devices[i]);
        if (error != cudaSuccess) return (int)error;
        error = cudaEventRecord(done_events[i], streams[i]);
        if (error != cudaSuccess) return (int)error;
    }
    cudaError_t error = cudaSetDevice(devices[0]);
    if (error != cudaSuccess) return (int)error;
    for (int i = 1; i < count; ++i) {
        /* Recording and waiting happen in host order, so a wait always refers to
         * the copy it follows and cannot form a wait cycle across rounds. */
        error = cudaStreamWaitEvent(streams[0], done_events[i], 0);
        if (error != cudaSuccess) return (int)error;
    }
    return 0;
}
