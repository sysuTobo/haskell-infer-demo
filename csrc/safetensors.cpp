/**
 * safetensors.cpp - strictly bounded parsing of safetensors headers.
 *
 * The previous parser used strstr to guess where an object started and ended,
 * defaulted an unknown dtype to BF16, and never compared the declared byte
 * count against the file. This one walks the header with a cursor that is
 * bounded by the header length read from the file, rejects anything it cannot
 * account for, and cross-checks every offset against the real file size.
 *
 * No CUDA, no engine dependency: the CPU test links this file directly.
 */
#include "safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <algorithm>
#include <memory>
#include <set>

namespace {

/* A header is a JSON object; the format's own guidance is that it stays well
 * under 100 MB, and a real checkpoint's is a few MB at most. */
const int64_t kMaxHeaderBytes = 100ll * 1024 * 1024;
/* Upper bound on a single tensor, so a shape product cannot be mistaken for a
 * plausible allocation. */
const int64_t kMaxTensorBytes = 1ll << 48;

std::string g_error;

void set_error(const std::string &message) { g_error = message; }

struct FileCloser {
    void operator()(FILE *file) const {
        if (file != nullptr) fclose(file);
    }
};

struct DirCloser {
    void operator()(DIR *dir) const {
        if (dir != nullptr) closedir(dir);
    }
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;
using DirPtr = std::unique_ptr<DIR, DirCloser>;

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* A forward-only cursor over the header text. Every step either consumes at
 * least one character or fails, so no loop can stall on malformed input. */
class Cursor {
public:
    Cursor(const char *begin, const char *end) : pos_(begin), end_(end) {}

    bool at_end() const { return pos_ >= end_; }

    void skip_ws() {
        while (pos_ < end_ &&
               (*pos_ == ' ' || *pos_ == '\n' || *pos_ == '\r' || *pos_ == '\t')) {
            ++pos_;
        }
    }

    bool consume(char c) {
        if (pos_ < end_ && *pos_ == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    /* A JSON string, starting on the opening quote. */
    bool parse_string(std::string *out) {
        if (!consume('"')) return false;
        out->clear();
        while (pos_ < end_) {
            const unsigned char c = (unsigned char)*pos_++;
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= end_) return false;
                const char escape = *pos_++;
                switch (escape) {
                    case '"': out->push_back('"'); break;
                    case '\\': out->push_back('\\'); break;
                    case '/': out->push_back('/'); break;
                    case 'b': out->push_back('\b'); break;
                    case 'f': out->push_back('\f'); break;
                    case 'n': out->push_back('\n'); break;
                    case 'r': out->push_back('\r'); break;
                    case 't': out->push_back('\t'); break;
                    case 'u': {
                        if (end_ - pos_ < 4) return false;
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            const int digit = hex_digit(*pos_++);
                            if (digit < 0) return false;
                            code = (code << 4) | (unsigned)digit;
                        }
                        /* Tensor names and dtype tags are ASCII; a wider escape
                         * would need an encoding this parser does not claim. */
                        if (code >= 0x80) return false;
                        out->push_back((char)code);
                        break;
                    }
                    default: return false;
                }
            } else if (c < 0x20) {
                return false;  /* raw control characters are not valid JSON */
            } else {
                out->push_back((char)c);
            }
        }
        return false;  /* unterminated string */
    }

    /* A non-negative JSON integer with no leading sign, fraction or exponent. */
    bool parse_int(int64_t *out) {
        if (pos_ >= end_ || *pos_ < '0' || *pos_ > '9') return false;
        int64_t value = 0;
        while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
            const int digit = *pos_++ - '0';
            if (value > (INT64_MAX - digit) / 10) return false;  /* would overflow */
            value = value * 10 + digit;
        }
        *out = value;
        return true;
    }

private:
    const char *pos_;
    const char *end_;
};

bool parse_shape(Cursor &cursor, const std::string &name, int64_t *shape, int *ndim) {
    if (!cursor.consume('[')) {
        set_error("tensor " + name + ": shape is not an array");
        return false;
    }
    cursor.skip_ws();
    if (cursor.consume(']')) {
        *ndim = 0;
        return true;
    }
    int count = 0;
    for (;;) {
        int64_t dim = 0;
        if (!cursor.parse_int(&dim)) {
            set_error("tensor " + name + ": shape has a dimension that is not a "
                      "non-negative integer");
            return false;
        }
        if (count >= SAFETENSORS_MAX_DIMS) {
            set_error("tensor " + name + ": shape has more than " +
                      std::to_string(SAFETENSORS_MAX_DIMS) + " dimensions");
            return false;
        }
        shape[count++] = dim;
        cursor.skip_ws();
        if (cursor.consume(',')) {
            cursor.skip_ws();
            continue;
        }
        if (cursor.consume(']')) break;
        set_error("tensor " + name + ": shape is malformed");
        return false;
    }
    *ndim = count;
    return true;
}

bool parse_offsets(Cursor &cursor, const std::string &name, int64_t *start, int64_t *stop) {
    if (!cursor.consume('[')) {
        set_error("tensor " + name + ": data_offsets is not an array");
        return false;
    }
    cursor.skip_ws();
    if (!cursor.parse_int(start)) {
        set_error("tensor " + name + ": data_offsets[0] is not a non-negative integer");
        return false;
    }
    cursor.skip_ws();
    if (!cursor.consume(',')) {
        set_error("tensor " + name + ": data_offsets needs exactly two entries");
        return false;
    }
    cursor.skip_ws();
    if (!cursor.parse_int(stop)) {
        set_error("tensor " + name + ": data_offsets[1] is not a non-negative integer");
        return false;
    }
    cursor.skip_ws();
    if (!cursor.consume(']')) {
        set_error("tensor " + name + ": data_offsets needs exactly two entries");
        return false;
    }
    return true;
}

bool parse_metadata(Cursor &cursor) {
    if (!cursor.consume('{')) {
        set_error("__metadata__ is not an object");
        return false;
    }
    cursor.skip_ws();
    if (cursor.consume('}')) return true;
    for (;;) {
        std::string key;
        std::string value;
        if (!cursor.parse_string(&key)) {
            set_error("__metadata__ has a key that is not a string");
            return false;
        }
        cursor.skip_ws();
        if (!cursor.consume(':')) {
            set_error("__metadata__ key " + key + " is missing ':'");
            return false;
        }
        cursor.skip_ws();
        if (!cursor.parse_string(&value)) {
            set_error("__metadata__ value for " + key + " is not a string");
            return false;
        }
        cursor.skip_ws();
        if (cursor.consume(',')) {
            cursor.skip_ws();
            continue;
        }
        if (cursor.consume('}')) return true;
        set_error("__metadata__ is malformed");
        return false;
    }
}

bool parse_tensor_entry(Cursor &cursor, const std::string &name, int64_t data_bytes,
                        TensorInfo *out) {
    if (!cursor.consume('{')) {
        set_error("tensor " + name + " is not an object");
        return false;
    }
    bool have_dtype = false;
    bool have_shape = false;
    bool have_offsets = false;
    int dtype = -1;
    int ndim = 0;
    int64_t shape[SAFETENSORS_MAX_DIMS] = {0, 0, 0, 0};
    int64_t start = 0;
    int64_t stop = 0;

    cursor.skip_ws();
    if (!cursor.consume('}')) {
        for (;;) {
            std::string field;
            if (!cursor.parse_string(&field)) {
                set_error("tensor " + name + ": expected a field name");
                return false;
            }
            cursor.skip_ws();
            if (!cursor.consume(':')) {
                set_error("tensor " + name + ": field " + field + " is missing ':'");
                return false;
            }
            cursor.skip_ws();
            if (field == "dtype") {
                if (have_dtype) {
                    set_error("tensor " + name + ": duplicate dtype");
                    return false;
                }
                have_dtype = true;
                std::string tag;
                if (!cursor.parse_string(&tag)) {
                    set_error("tensor " + name + ": dtype is not a string");
                    return false;
                }
                if (tag == "BF16") {
                    dtype = SAFETENSORS_BF16;
                } else if (tag == "F32") {
                    dtype = SAFETENSORS_F32;
                } else if (tag == "F16") {
                    dtype = SAFETENSORS_F16;
                } else {
                    set_error("tensor " + name + ": unsupported dtype " + tag);
                    return false;
                }
            } else if (field == "shape") {
                if (have_shape) {
                    set_error("tensor " + name + ": duplicate shape");
                    return false;
                }
                have_shape = true;
                if (!parse_shape(cursor, name, shape, &ndim)) return false;
            } else if (field == "data_offsets") {
                if (have_offsets) {
                    set_error("tensor " + name + ": duplicate data_offsets");
                    return false;
                }
                have_offsets = true;
                if (!parse_offsets(cursor, name, &start, &stop)) return false;
            } else {
                set_error("tensor " + name + ": unknown field " + field);
                return false;
            }
            cursor.skip_ws();
            if (cursor.consume(',')) {
                cursor.skip_ws();
                continue;
            }
            if (cursor.consume('}')) break;
            set_error("tensor " + name + ": expected ',' or '}'");
            return false;
        }
    }

    if (!have_dtype || !have_shape || !have_offsets) {
        set_error("tensor " + name + " must carry dtype, shape and data_offsets");
        return false;
    }
    if (stop < start) {
        set_error("tensor " + name + ": data_offsets are reversed");
        return false;
    }
    if (stop > data_bytes) {
        set_error("tensor " + name + ": data_offsets run past the end of the data section");
        return false;
    }

    out->name = name;
    out->dtype = dtype;
    out->ndim = ndim;
    for (int i = 0; i < SAFETENSORS_MAX_DIMS; ++i) out->shape[i] = shape[i];
    out->data_start = start;
    out->data_end = stop;
    out->file_path.clear();
    out->file_data_offset = 0;

    int64_t declared = 0;
    if (safetensors_tensor_bytes(*out, &declared) != 0) {
        set_error("tensor " + name + ": shape does not describe a loadable byte count");
        return false;
    }
    if (declared != stop - start) {
        set_error("tensor " + name + ": data_offsets cover " + std::to_string(stop - start) +
                  " bytes but the shape needs " + std::to_string(declared));
        return false;
    }
    return true;
}

int parse_header(const char *text, int64_t header_bytes, int64_t data_bytes,
                 std::vector<TensorInfo> &out) {
    out.clear();
    Cursor cursor(text, text + header_bytes);
    cursor.skip_ws();
    if (!cursor.consume('{')) {
        set_error("header is not a JSON object");
        return -1;
    }

    std::set<std::string> seen;
    cursor.skip_ws();
    if (!cursor.consume('}')) {
        for (;;) {
            std::string key;
            if (!cursor.parse_string(&key)) {
                set_error("expected a tensor name string");
                return -1;
            }
            cursor.skip_ws();
            if (!cursor.consume(':')) {
                set_error("key " + key + " is missing ':'");
                return -1;
            }
            cursor.skip_ws();
            if (key == "__metadata__") {
                if (!parse_metadata(cursor)) return -1;
            } else {
                if (!seen.insert(key).second) {
                    set_error("duplicate tensor " + key);
                    return -1;
                }
                TensorInfo info;
                if (!parse_tensor_entry(cursor, key, data_bytes, &info)) return -1;
                out.push_back(info);
            }
            cursor.skip_ws();
            if (cursor.consume(',')) {
                cursor.skip_ws();
                continue;
            }
            if (cursor.consume('}')) break;
            set_error("expected ',' or '}' in the header");
            return -1;
        }
    }
    cursor.skip_ws();
    if (!cursor.at_end()) {
        set_error("trailing bytes after the header object");
        return -1;
    }
    return 0;
}

bool mul_overflows(int64_t a, int64_t b, int64_t *out) {
    if (a < 0 || b < 0) return true;
    if (a != 0 && b > INT64_MAX / a) return true;
    *out = a * b;
    return false;
}

bool add_overflows(int64_t a, int64_t b, int64_t *out) {
    if (a < 0 || b < 0 || a > INT64_MAX - b) return true;
    *out = a + b;
    return false;
}

/* The tensor's data as (file offset, byte count), agreeing with its shape. */
bool data_range(const TensorInfo &ti, int64_t *offset, int64_t *bytes) {
    if (safetensors_tensor_bytes(ti, bytes) != 0) return false;
    if (*bytes != ti.data_bytes()) {
        set_error("tensor " + ti.name + ": shape and data_offsets disagree");
        return false;
    }
    if (add_overflows(ti.file_data_offset, ti.data_start, offset)) {
        set_error("tensor " + ti.name + ": file offset overflows");
        return false;
    }
    if ((uint64_t)*bytes > (uint64_t)SIZE_MAX) {
        set_error("tensor " + ti.name + ": byte count exceeds the host address space");
        return false;
    }
    return true;
}

}  // namespace

const char *safetensors_last_error(void) { return g_error.c_str(); }

void safetensors_set_error(const std::string &message) { g_error = message; }

int safetensors_tensor_bytes(const TensorInfo &ti, int64_t *bytes) {
    const int64_t element = ti.element_bytes();
    if (element < 0) {
        set_error("unknown dtype tag");
        return -1;
    }
    int64_t total = 1;
    for (int i = 0; i < ti.ndim; ++i) {
        if (ti.shape[i] < 0) {
            set_error("negative dimension");
            return -1;
        }
        if (ti.shape[i] != 0 && total > kMaxTensorBytes / ti.shape[i]) {
            set_error("shape product overflows");
            return -1;
        }
        total *= ti.shape[i];
    }
    if (total > kMaxTensorBytes / element) {
        set_error("tensor byte count overflows");
        return -1;
    }
    *bytes = total * element;
    return 0;
}

int safetensors_plan_whole(const TensorInfo &ti, int64_t dst_capacity, SafetensorsPlan *plan) {
    int64_t offset = 0;
    int64_t bytes = 0;
    if (!data_range(ti, &offset, &bytes)) return -3;
    if (dst_capacity < bytes) {
        set_error("tensor " + ti.name + ": destination holds " + std::to_string(dst_capacity) +
                  " bytes but the tensor needs " + std::to_string(bytes));
        return -3;
    }
    plan->file_offset = offset;
    plan->file_bytes = bytes;
    plan->dst_bytes = bytes;
    plan->dst_row_bytes = 0;
    return 0;
}

int safetensors_plan_rows(const TensorInfo &ti, int64_t dst_capacity, const int *order,
                          int64_t rows, int64_t row_bytes, SafetensorsPlan *plan) {
    if (order == nullptr || rows <= 0 || row_bytes <= 0) {
        set_error("tensor " + ti.name + ": invalid row gather");
        return -3;
    }
    int64_t offset = 0;
    int64_t bytes = 0;
    if (!data_range(ti, &offset, &bytes)) return -3;
    if (row_bytes > bytes || bytes % row_bytes != 0) {
        set_error("tensor " + ti.name + ": row size does not divide the tensor");
        return -3;
    }
    const int64_t source_rows = bytes / row_bytes;
    int64_t out_bytes = 0;
    if (mul_overflows(rows, row_bytes, &out_bytes)) {
        set_error("tensor " + ti.name + ": row gather overflows");
        return -3;
    }
    if (dst_capacity < out_bytes) {
        set_error("tensor " + ti.name + ": destination holds " + std::to_string(dst_capacity) +
                  " bytes but the gather needs " + std::to_string(out_bytes));
        return -3;
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (order[row] < 0 || (int64_t)order[row] >= source_rows) {
            set_error("tensor " + ti.name + ": row index " + std::to_string(order[row]) +
                      " is outside " + std::to_string(source_rows) + " rows");
            return -4;
        }
    }
    plan->file_offset = offset;
    plan->file_bytes = bytes;
    plan->dst_bytes = out_bytes;
    plan->dst_row_bytes = row_bytes;
    return 0;
}

int safetensors_plan_slice(const TensorInfo &ti, int64_t dst_capacity, int64_t row_off,
                           int64_t rows, int64_t col_off, int64_t cols,
                           SafetensorsPlan *plan) {
    if (ti.ndim != 2 || rows <= 0 || cols <= 0) {
        set_error("tensor " + ti.name +
                  ": slice needs a 2-D tensor and a non-empty rectangle");
        return -3;
    }
    const int64_t src_rows = ti.shape[0];
    const int64_t src_cols = ti.shape[1];
    if (row_off < 0 || col_off < 0 || row_off > src_rows || rows > src_rows - row_off ||
        col_off > src_cols || cols > src_cols - col_off) {
        set_error("tensor " + ti.name + ": slice is outside the tensor");
        return -3;
    }
    const int64_t element = ti.element_bytes();
    if (element < 0) {
        set_error("tensor " + ti.name + ": unknown dtype tag");
        return -3;
    }
    int64_t offset = 0;
    int64_t bytes = 0;
    if (!data_range(ti, &offset, &bytes)) return -3;

    int64_t src_row_bytes = 0;
    int64_t dst_row_bytes = 0;
    int64_t dst_bytes = 0;
    int64_t expected = 0;
    if (mul_overflows(src_cols, element, &src_row_bytes) ||
        mul_overflows(cols, element, &dst_row_bytes) ||
        mul_overflows(dst_row_bytes, rows, &dst_bytes) ||
        mul_overflows(src_row_bytes, src_rows, &expected)) {
        set_error("tensor " + ti.name + ": slice size overflows");
        return -3;
    }
    if (expected != bytes) {
        set_error("tensor " + ti.name + ": shape and data_offsets disagree");
        return -3;
    }
    if (dst_capacity < dst_bytes) {
        set_error("tensor " + ti.name + ": destination holds " + std::to_string(dst_capacity) +
                  " bytes but the slice needs " + std::to_string(dst_bytes));
        return -3;
    }
    plan->file_offset = offset;
    plan->file_bytes = bytes;
    plan->dst_bytes = dst_bytes;
    plan->dst_row_bytes = dst_row_bytes;
    return 0;
}

int safetensors_read_header(const char *path, std::vector<TensorInfo> &out) {
    out.clear();
    if (path == nullptr) {
        set_error("no path");
        return -1;
    }

    FilePtr file(fopen(path, "rb"));
    if (!file) {
        set_error(std::string("cannot open ") + path + ": " + strerror(errno));
        return -1;
    }
    struct stat info;
    if (fstat(fileno(file.get()), &info) != 0 || !S_ISREG(info.st_mode)) {
        set_error(std::string("cannot stat ") + path);
        return -1;
    }
    const int64_t file_size = (int64_t)info.st_size;
    if (file_size < 8) {
        set_error(std::string(path) + ": shorter than the 8-byte header size");
        return -1;
    }

    unsigned char raw[8];
    if (fread(raw, 1, sizeof raw, file.get()) != sizeof raw) {
        set_error(std::string(path) + ": cannot read the header size");
        return -1;
    }
    uint64_t header_bytes = 0;
    for (int i = 7; i >= 0; --i) header_bytes = (header_bytes << 8) | raw[i];

    /* Bound the allocation by the real file size and by the format's own
     * sanity limit before reserving anything. */
    if (header_bytes > (uint64_t)(file_size - 8)) {
        set_error(std::string(path) + ": header size " + std::to_string(header_bytes) +
                  " exceeds the file size " + std::to_string(file_size));
        return -1;
    }
    if ((int64_t)header_bytes > kMaxHeaderBytes) {
        set_error(std::string(path) + ": header size " + std::to_string(header_bytes) +
                  " exceeds the sanity limit");
        return -1;
    }

    std::vector<char> header((size_t)header_bytes + 1);
    if (header_bytes > 0 &&
        fread(header.data(), 1, (size_t)header_bytes, file.get()) != (size_t)header_bytes) {
        set_error(std::string(path) + ": header is truncated");
        return -1;
    }
    header[(size_t)header_bytes] = '\0';

    const int64_t data_offset = 8 + (int64_t)header_bytes;
    const int64_t data_bytes = file_size - data_offset;
    if (parse_header(header.data(), (int64_t)header_bytes, data_bytes, out) != 0) {
        out.clear();
        return -1;
    }
    for (TensorInfo &ti : out) {
        ti.file_path = path;
        ti.file_data_offset = data_offset;
    }
    return 0;
}

int safetensors_scan_dir(const char *model_dir, std::map<std::string, TensorInfo> &index) {
    index.clear();
    if (model_dir == nullptr) {
        set_error("no model directory");
        return -1;
    }
    DirPtr dir(opendir(model_dir));
    if (!dir) {
        set_error(std::string("cannot open directory ") + model_dir + ": " + strerror(errno));
        return -1;
    }

    static const size_t kSuffixLength = 12;  /* ".safetensors" */
    std::vector<std::string> files;
    while (struct dirent *entry = readdir(dir.get())) {
        const std::string name = entry->d_name;
        if (name.size() > kSuffixLength &&
            name.compare(name.size() - kSuffixLength, kSuffixLength, ".safetensors") == 0) {
            files.push_back(std::string(model_dir) + "/" + name);
        }
    }
    dir.reset();
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        set_error(std::string("no .safetensors files in ") + model_dir);
        return -1;
    }

    for (const std::string &file : files) {
        std::vector<TensorInfo> tensors;
        if (safetensors_read_header(file.c_str(), tensors) != 0) {
            /* Keep the per-file message and report the scan as failed: a partial
             * index would silently drop weights. */
            index.clear();
            return -1;
        }
        for (const TensorInfo &ti : tensors) {
            if (index.find(ti.name) != index.end()) {
                set_error("tensor " + ti.name + " appears in more than one shard");
                index.clear();
                return -1;
            }
            index[ti.name] = ti;
        }
    }
    if (index.empty()) {
        set_error(std::string("no tensors found in ") + model_dir);
        return -1;
    }
    return 0;
}
