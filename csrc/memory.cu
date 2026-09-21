/**
 * memory.cu - GPU memory management utilities.
 *
 * Thin wrappers around CUDA memory APIs with error checking.
 * All allocations are device-local; cross-device copies use
 * cudaMemcpyPeerAsync.
 */

#include "engine.h"
#include <cuda_runtime.h>
#include <cstdio>

/* ------------------------------------------------------------------ */
/*  Internal error reporting (shared with engine.cu via engine.h)     */
/* ------------------------------------------------------------------ */

/* Defined in engine.cu */
extern const char *engine_last_error(void);

/* ------------------------------------------------------------------ */
/*  Allocation helpers                                                */
/* ------------------------------------------------------------------ */

int mem_alloc_device(void **ptr, size_t bytes, int device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        *ptr = nullptr;
        return ENGINE_ERR_ALLOC;
    }
    return ENGINE_OK;
}

int mem_alloc_device_zero(void **ptr, size_t bytes, int device) {
    int rc = mem_alloc_device(ptr, bytes, device);
    if (rc != ENGINE_OK) return rc;

    cudaError_t err = cudaMemset(*ptr, 0, bytes);
    if (err != cudaSuccess) {
        cudaFree(*ptr);
        *ptr = nullptr;
        return ENGINE_ERR_CUDA;
    }
    return ENGINE_OK;
}

void mem_free_device(void *ptr) {
    if (ptr) cudaFree(ptr);
}

/* ------------------------------------------------------------------ */
/*  Copy helpers                                                      */
/* ------------------------------------------------------------------ */

int mem_copy_h2d(void *dst, const void *src, size_t bytes, int device,
                 cudaStream_t stream) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return (err == cudaSuccess) ? ENGINE_OK : ENGINE_ERR_CUDA;
}

int mem_copy_d2h(void *dst, const void *src, size_t bytes, int device,
                 cudaStream_t stream) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    return (err == cudaSuccess) ? ENGINE_OK : ENGINE_ERR_CUDA;
}

int mem_copy_d2d_peer(void *dst, int dst_device, const void *src,
                      int src_device, size_t bytes, cudaStream_t stream) {
    /* cudaMemcpyPeerAsync handles cross-device copies.
     * The stream should belong to the destination device. */
    cudaError_t err = cudaSetDevice(dst_device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
    return (err == cudaSuccess) ? ENGINE_OK : ENGINE_ERR_CUDA;
}

/* ------------------------------------------------------------------ */
/*  Synchronization                                                   */
/* ------------------------------------------------------------------ */

int mem_sync_device(int device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? ENGINE_OK : ENGINE_ERR_CUDA;
}

int mem_sync_stream(int device, cudaStream_t stream) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaStreamSynchronize(stream);
    return (err == cudaSuccess) ? ENGINE_OK : ENGINE_ERR_CUDA;
}

/* ------------------------------------------------------------------ */
/*  Stream management                                                 */
/* ------------------------------------------------------------------ */

int mem_create_stream(cudaStream_t *stream, int device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    err = cudaStreamCreate(stream);
    return (err == cudaSuccess) ? ENGINE_OK : ENGINE_ERR_CUDA;
}

void mem_destroy_stream(cudaStream_t stream, int device) {
    cudaSetDevice(device);
    cudaStreamDestroy(stream);
}

/* ------------------------------------------------------------------ */
/*  Peer access                                                       */
/* ------------------------------------------------------------------ */

int mem_enable_peer_access(int device, int peer_device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return ENGINE_ERR_CUDA;

    int can_access = 0;
    err = cudaDeviceCanAccessPeer(&can_access, device, peer_device);
    if (err != cudaSuccess || !can_access) return ENGINE_ERR_CUDA;

    err = cudaDeviceEnablePeerAccess(peer_device, 0);
    /* cudaErrorPeerAccessAlreadyEnabled is not fatal */
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
        return ENGINE_ERR_CUDA;

    return ENGINE_OK;
}
