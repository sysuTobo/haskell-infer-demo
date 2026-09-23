/**
 * safetensors_loader.cu - Upload tensors from safetensors files to the GPU.
 *
 * Header parsing and all request validation live in safetensors.cpp, which is
 * CUDA-free and covered by tests/safetensors_test.cpp. This file only moves
 * bytes: it re-checks the byte range against the real file size, re-checks what
 * it read, and reports file and CUDA failures instead of letting a short read
 * or a failed copy pass as success.
 */
#include "safetensors.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <memory>
#include <vector>

namespace {

struct FileCloser {
    void operator()(FILE *file) const {
        if (file != nullptr) fclose(file);
    }
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;

/* Open the file and confirm [offset, offset+bytes) really lies inside it before
 * a single byte is read. */
int open_range(const std::string &path, int64_t offset, int64_t bytes, FilePtr *file) {
    FILE *raw = fopen(path.c_str(), "rb");
    if (raw == nullptr) {
        safetensors_set_error("cannot open " + path + ": " + strerror(errno));
        return -1;
    }
    file->reset(raw);
    struct stat info;
    if (fstat(fileno(raw), &info) != 0 || !S_ISREG(info.st_mode)) {
        safetensors_set_error("cannot stat " + path);
        return -1;
    }
    const int64_t file_size = (int64_t)info.st_size;
    if (offset > file_size || bytes > file_size - offset) {
        safetensors_set_error("reading " + std::to_string(bytes) + " bytes at " +
                              std::to_string(offset) + " runs past the end of " + path);
        return -1;
    }
    if (fseek(raw, (long)offset, SEEK_SET) != 0) {
        safetensors_set_error("cannot seek in " + path);
        return -1;
    }
    return 0;
}

int upload(const void *host, void *dst, int64_t bytes, int device) {
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        safetensors_set_error(cudaGetErrorString(error));
        return -1;
    }
    if (bytes > 0) {
        error = cudaMemcpy(dst, host, (size_t)bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            safetensors_set_error(cudaGetErrorString(error));
            return -1;
        }
    }
    return 0;
}

int read_exactly(const std::string &path, FILE *file, char *buffer, int64_t bytes) {
    if (bytes > 0 && fread(buffer, 1, (size_t)bytes, file) != (size_t)bytes) {
        safetensors_set_error("short read from " + path);
        return -2;
    }
    return 0;
}

}  // namespace

int safetensors_load_tensor(const TensorInfo &ti, void *dst, int64_t dst_capacity, int device) {
    if (dst == nullptr) {
        safetensors_set_error("tensor " + ti.name + ": no destination");
        return -3;
    }
    SafetensorsPlan plan;
    int status = safetensors_plan_whole(ti, dst_capacity, &plan);
    if (status != 0) return status;

    FilePtr file;
    status = open_range(ti.file_path, plan.file_offset, plan.file_bytes, &file);
    if (status != 0) return status;

    std::vector<char> host((size_t)plan.file_bytes);
    status = read_exactly(ti.file_path, file.get(), host.data(), plan.file_bytes);
    if (status != 0) return status;
    return upload(host.data(), dst, plan.dst_bytes, device);
}

int safetensors_load_tensor_rows(const TensorInfo &ti, void *dst, int64_t dst_capacity,
                                 int device, const int *order, int64_t rows, int64_t row_bytes) {
    if (dst == nullptr) {
        safetensors_set_error("tensor " + ti.name + ": no destination");
        return -3;
    }
    SafetensorsPlan plan;
    int status = safetensors_plan_rows(ti, dst_capacity, order, rows, row_bytes, &plan);
    if (status != 0) return status;

    FilePtr file;
    status = open_range(ti.file_path, plan.file_offset, plan.file_bytes, &file);
    if (status != 0) return status;

    std::vector<char> host((size_t)plan.file_bytes);
    status = read_exactly(ti.file_path, file.get(), host.data(), plan.file_bytes);
    if (status != 0) return status;

    std::vector<char> gathered((size_t)plan.dst_bytes);
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t source = order[row];
        memcpy(gathered.data() + (size_t)(row * row_bytes),
               host.data() + (size_t)(source * row_bytes), (size_t)row_bytes);
    }
    return upload(gathered.data(), dst, plan.dst_bytes, device);
}

int safetensors_load_tensor_slice(const TensorInfo &ti, void *dst, int64_t dst_capacity,
                                  int device, int64_t row_off, int64_t rows,
                                  int64_t col_off, int64_t cols) {
    if (dst == nullptr) {
        safetensors_set_error("tensor " + ti.name + ": no destination");
        return -3;
    }
    SafetensorsPlan plan;
    int status = safetensors_plan_slice(ti, dst_capacity, row_off, rows, col_off, cols, &plan);
    if (status != 0) return status;

    FilePtr file;
    status = open_range(ti.file_path, plan.file_offset, plan.file_bytes, &file);
    if (status != 0) return status;

    const int64_t src_row_bytes = ti.shape[1] * ti.element_bytes();
    const int64_t dst_row_bytes = plan.dst_row_bytes;

    /* Whole rows are one contiguous read; a column window stages a bounded
     * number of rows through the host. */
    if (col_off == 0 && cols == ti.shape[1]) {
        const int64_t start = plan.file_offset + row_off * src_row_bytes;
        if (fseek(file.get(), (long)start, SEEK_SET) != 0) {
            safetensors_set_error("tensor " + ti.name + ": cannot seek to the row window");
            return -2;
        }
        std::vector<char> host((size_t)plan.dst_bytes);
        status = read_exactly(ti.file_path, file.get(), host.data(), plan.dst_bytes);
        if (status != 0) return status;
        return upload(host.data(), dst, plan.dst_bytes, device);
    }

    const int64_t chunk_rows = rows < 64 ? rows : 64;
    const int64_t element = ti.element_bytes();
    std::vector<char> host((size_t)(chunk_rows * dst_row_bytes));
    for (int64_t done = 0; done < rows; done += chunk_rows) {
        const int64_t take = (chunk_rows < rows - done) ? chunk_rows : rows - done;
        for (int64_t r = 0; r < take; ++r) {
            /* Every position stays inside [file_offset, file_offset + file_bytes),
             * which open_range already checked against the real file size. */
            const int64_t position = plan.file_offset + (row_off + done + r) * src_row_bytes +
                                     col_off * element;
            if (fseek(file.get(), (long)position, SEEK_SET) != 0) {
                safetensors_set_error("tensor " + ti.name + ": cannot seek inside the row");
                return -2;
            }
            status = read_exactly(ti.file_path, file.get(),
                                  host.data() + (size_t)(r * dst_row_bytes), dst_row_bytes);
            if (status != 0) return status;
        }
        status = upload(host.data(), (char *)dst + (size_t)(done * dst_row_bytes),
                        take * dst_row_bytes, device);
        if (status != 0) return status;
    }
    return 0;
}
