/**
 * safetensors.h - safetensors metadata parsing and file-boundary validation.
 *
 * Deliberately free of CUDA: safetensors.cpp and the CPU test
 * (tests/safetensors_test.cpp) compile from this header alone, so a malformed
 * checkpoint is rejected by a test that needs no GPU. The upload helpers that
 * need a device live in safetensors_loader.cu.
 *
 * Format: [8-byte little-endian header size][JSON header][raw tensor data].
 * The JSON header maps a tensor name to
 * {"dtype": "BF16"|"F32"|"F16", "shape": [...], "data_offsets": [start, end]}
 * with offsets relative to the start of the data section.
 */
#ifndef HASKELL_INFER_SAFETENSORS_H
#define HASKELL_INFER_SAFETENSORS_H

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

/* Element types the loader understands (the checkpoint's dtype tag). */
enum {
    SAFETENSORS_BF16 = 0,
    SAFETENSORS_F32 = 1,
    SAFETENSORS_F16 = 2
};

/* Highest tensor rank the parser stores. Real checkpoints carry tensors the text
 * model never loads -- Qwen3.8-27B ships a rank-5 Conv3D vision patch embedding
 * next to 1198 text tensors -- so this bound keeps TensorInfo a fixed size and
 * the parse finite; it is not a claim about the role templates. The engine
 * validates the rank of a tensor it actually requests against the role's shape. */
static const int SAFETENSORS_MAX_DIMS = 8;

struct TensorInfo {
    std::string name;
    int dtype;                              /* SAFETENSORS_* */
    int ndim;
    int64_t shape[SAFETENSORS_MAX_DIMS];
    int64_t data_start;                     /* byte offset inside the data section */
    int64_t data_end;                       /* exclusive */
    std::string file_path;
    int64_t file_data_offset;               /* where the data section starts in the file */

    /* Bytes per element, or -1 for an unknown dtype. */
    int64_t element_bytes() const {
        switch (dtype) {
            case SAFETENSORS_BF16:
            case SAFETENSORS_F16:
                return 2;
            case SAFETENSORS_F32:
                return 4;
            default:
                return -1;
        }
    }
    int64_t data_bytes() const { return data_end - data_start; }
};

/* Human-readable reason the last safetensors call failed (never null). */
const char *safetensors_last_error(void);

/* Record a failure reason for safetensors_last_error(). The upload helpers in
 * safetensors_loader.cu report through the same channel so a caller sees one
 * message whatever stage failed. */
void safetensors_set_error(const std::string &message);

/* Byte count the tensor's shape declares, with overflow checking.
 * Returns 0 and fills @bytes, or -1 (see safetensors_last_error). */
int safetensors_tensor_bytes(const TensorInfo &ti, int64_t *bytes);

/* Parse one safetensors file's header into @out. Validates the header against
 * the actual file size, the shape against the dtype, and the offsets against
 * both the declared byte count and the data section. Returns 0, or a negative
 * error code with the reason in safetensors_last_error(). */
int safetensors_read_header(const char *path, std::vector<TensorInfo> &out);

/* Parse every *.safetensors file in a directory into one index, sorted by file
 * name. Returns 0 on success. Any unreadable or invalid shard fails the whole
 * scan: @index is left empty rather than partially populated. */
int safetensors_scan_dir(const char *model_dir, std::map<std::string, TensorInfo> &index);

/* ------------------------------------------------------------------ */
/* Transfer planning (device-free, so the CPU test covers the           */
/* arithmetic)                                                          */
/* ------------------------------------------------------------------ */

/* What a transfer will read and write, once the request has been validated
 * against the tensor, the file layout and the destination capacity. */
struct SafetensorsPlan {
    int64_t file_offset;    /* first byte to read in the file */
    int64_t file_bytes;     /* bytes readable from there (the bound to re-check) */
    int64_t dst_bytes;      /* bytes the destination receives */
    int64_t dst_row_bytes;  /* bytes per destination row (slices only) */
};

/* Whole tensor. Returns 0 or a negative code with the reason in
 * safetensors_last_error(). */
int safetensors_plan_whole(const TensorInfo &ti, int64_t dst_capacity, SafetensorsPlan *plan);

/* Row gather: row i of the destination comes from row order[i] of the tensor.
 * Rejects a row size that does not divide the tensor and an index outside it. */
int safetensors_plan_rows(const TensorInfo &ti, int64_t dst_capacity, const int *order,
                          int64_t rows, int64_t row_bytes, SafetensorsPlan *plan);

/* Rectangle of a 2-D tensor: rows [row_off, row_off+rows) and, inside each row,
 * elements [col_off, col_off+cols). */
int safetensors_plan_slice(const TensorInfo &ti, int64_t dst_capacity, int64_t row_off,
                           int64_t rows, int64_t col_off, int64_t cols,
                           SafetensorsPlan *plan);

/* ------------------------------------------------------------------ */
/* Upload helpers (safetensors_loader.cu; need a CUDA device)          */
/* ------------------------------------------------------------------ */

/* Each helper takes the destination capacity in bytes and refuses a transfer
 * that would overrun it. Return 0, or a negative code (see
 * safetensors_last_error). */

/* The whole tensor. */
int safetensors_load_tensor(const TensorInfo &ti, void *dst, int64_t dst_capacity, int device);

/* Row i of @dst comes from row order[i] of the file (row_bytes per row). */
int safetensors_load_tensor_rows(const TensorInfo &ti, void *dst, int64_t dst_capacity,
                                 int device, const int *order, int64_t rows, int64_t row_bytes);

/* Rows [row_off, row_off+rows) and, inside each row, elements
 * [col_off, col_off+cols) of a 2-D tensor. */
int safetensors_load_tensor_slice(const TensorInfo &ti, void *dst, int64_t dst_capacity,
                                  int device, int64_t row_off, int64_t rows,
                                  int64_t col_off, int64_t cols);

#endif /* HASKELL_INFER_SAFETENSORS_H */
