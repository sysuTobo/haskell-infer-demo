/**
 * safetensors_loader.cu - Load weights from safetensors files.
 *
 * Safetensors format: [8B header_size LE][JSON header][raw tensor data]
 * JSON header: {"name": {"dtype":"BF16","shape":[N,M],"data_offsets":[s,e]}, ...}
 *
 * Strategy: read header JSON, parse with minimal string matching,
 * then mmap the file and cudaMemcpy tensors to the correct device.
 */

#include "engine.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

struct TensorInfo {
    std::string name;
    int dtype;          // 0=BF16, 1=F32, 2=F16
    int shape[4];
    int ndim;
    long long data_start;  // byte offset within data section
    long long data_end;
    std::string file_path;
    long long file_data_offset;  // where data section starts in the file
};

// Minimal JSON value extractor for safetensors headers
static const char *find_key(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    return strstr(json, pattern);
}

static long long parse_int_after(const char *p, const char *label) {
    const char *q = find_key(p, label);
    if (!q) return -1;
    q = strchr(q + strlen(label) + 2, ':');
    if (!q) return -1;
    q++;
    while (*q == ' ' || *q == '[') q++;
    return strtoll(q, nullptr, 10);
}

static int parse_shape(const char *entry, int *shape) {
    const char *p = strstr(entry, "\"shape\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    int n = 0;
    p++;
    while (*p && *p != ']' && n < 4) {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        shape[n++] = (int)strtol(p, (char **)&p, 10);
    }
    return n;
}

static int parse_offsets(const char *entry, long long *start, long long *end) {
    const char *p = strstr(entry, "\"data_offsets\"");
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    p++;
    *start = strtoll(p, (char **)&p, 10);
    while (*p && *p != ',' && *p != ']') p++;
    if (*p == ',') p++;
    *end = strtoll(p, nullptr, 10);
    return 0;
}

/**
 * Parse a safetensors file header and return tensor metadata.
 */
int safetensors_parse_header(const char *path, std::vector<TensorInfo> &tensors) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint64_t hdr_size = 0;
    fread(&hdr_size, 8, 1, f);

    std::vector<char> hdr_buf(hdr_size + 1);
    fread(hdr_buf.data(), 1, hdr_size, f);
    hdr_buf[hdr_size] = '\0';
    long long data_offset = 8 + (long long)hdr_size;
    fclose(f);

    const char *json = hdr_buf.data();
    // Parse each top-level key-value pair
    // Format: {"name": {...}, "name2": {...}, ...}
    const char *p = json;
    while ((p = strchr(p, '"')) != nullptr) {
        // Check if this is a tensor name (followed by ": {")
        const char *name_start = p + 1;
        const char *name_end = strchr(name_start, '"');
        if (!name_end) break;

        std::string name(name_start, name_end - name_start);
        p = name_end + 1;

        // Skip metadata key
        if (name == "__metadata__") {
            // Skip to next entry
            const char *next = strchr(p, '}');
            if (next) p = next + 1;
            continue;
        }

        // Find the opening brace of this entry
        const char *entry_start = strchr(p, '{');
        if (!entry_start) break;

        // Find matching closing brace (no nested objects in safetensors)
        const char *entry_end = strchr(entry_start, '}');
        if (!entry_end) break;

        // Extract fields from this entry
        std::string entry(entry_start, entry_end - entry_start + 1);

        TensorInfo ti;
        ti.name = name;
        ti.file_path = path;
        ti.file_data_offset = data_offset;

        // dtype
        if (entry.find("\"BF16\"") != std::string::npos) ti.dtype = 0;
        else if (entry.find("\"F32\"") != std::string::npos) ti.dtype = 1;
        else if (entry.find("\"F16\"") != std::string::npos) ti.dtype = 2;
        else ti.dtype = 0;

        ti.ndim = parse_shape(entry.c_str(), ti.shape);
        parse_offsets(entry.c_str(), &ti.data_start, &ti.data_end);

        tensors.push_back(ti);
        p = entry_end + 1;
    }
    return 0;
}

/**
 * Load a tensor from a safetensors file to GPU memory.
 */
int safetensors_load_tensor(const TensorInfo &ti, void *dst, int device) {
    long long size = ti.data_end - ti.data_start;
    long long file_offset = ti.file_data_offset + ti.data_start;

    // Read from file to host, then copy to device
    std::vector<char> host_buf(size);
    FILE *f = fopen(ti.file_path.c_str(), "rb");
    if (!f) return -1;
    fseek(f, file_offset, SEEK_SET);
    size_t read = fread(host_buf.data(), 1, size, f);
    fclose(f);
    if ((long long)read != size) return -2;

    cudaSetDevice(device);
    cudaMemcpy(dst, host_buf.data(), size, cudaMemcpyHostToDevice);
    return 0;
}

/**
 * Scan model directory for all safetensors files and build a tensor index.
 */
int safetensors_scan_dir(const char *model_dir, std::map<std::string, TensorInfo> &index) {
    std::vector<TensorInfo> all_tensors;
    DIR *dir = opendir(model_dir);
    if (!dir) return -1;

    std::vector<std::string> files;
    struct dirent *ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
            files.push_back(std::string(model_dir) + "/" + name);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());

    for (const auto &f : files) {
        safetensors_parse_header(f.c_str(), all_tensors);
    }

    for (const auto &ti : all_tensors) {
        index[ti.name] = ti;
    }
    return (int)all_tensors.size();
}
