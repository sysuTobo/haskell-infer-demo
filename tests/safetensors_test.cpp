/*
 * safetensors_test.cpp - CPU tests for the safetensors loader's parsing and
 * request validation.
 *
 * Every fixture is a small file written at run time into a private directory,
 * so the cases that matter for a bad checkpoint (a lying header size, offsets
 * past the data section, a shape that does not match the payload, a duplicate
 * key) are exercised without a GPU or a real model.
 */
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

static int failures = 0;
static int checks = 0;

static void check(bool condition, const std::string &what) {
    ++checks;
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

static void check_ok(int status, const std::string &what) {
    check(status == 0, what + " (status " + std::to_string(status) + ": " +
                           safetensors_last_error() + ")");
}

/* Rejected, and the message mentions the expected reason. */
static void check_rejected(int status, const std::string &needle, const std::string &what) {
    const std::string message = safetensors_last_error();
    check(status != 0, what + " must be rejected");
    if (status == 0) return;
    check(message.find(needle) != std::string::npos,
          what + ": message \"" + message + "\" should mention \"" + needle + "\"");
}

static std::string g_dir;

static std::string path_of(const std::string &name) { return g_dir + "/" + name; }

/* A safetensors file: [8-byte little-endian header size][header][payload]. The
 * declared size is a parameter so a lying header can be built. */
static void write_shard(const std::string &name, const std::string &header,
                        const std::string &payload, int64_t declared_header_size = -1) {
    const int64_t declared = declared_header_size < 0 ? (int64_t)header.size()
                                                      : declared_header_size;
    std::string bytes;
    for (int i = 0; i < 8; ++i) {
        bytes.push_back((char)((uint64_t)declared >> (8 * i)) & 0xFF);
    }
    bytes += header;
    bytes += payload;
    FILE *file = fopen(path_of(name).c_str(), "wb");
    if (file == nullptr) {
        fprintf(stderr, "cannot create fixture %s\n", name.c_str());
        exit(2);
    }
    fwrite(bytes.data(), 1, bytes.size(), file);
    fclose(file);
}

static int read_shard(const std::string &name, std::vector<TensorInfo> *out) {
    return safetensors_read_header(path_of(name).c_str(), *out);
}

static std::string payload(int bytes, char value) { return std::string((size_t)bytes, value); }

static void test_valid_dtypes(void) {
    const std::string header =
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[12,24]},"
        "\"c\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[24,26]}}";
    write_shard("valid.safetensors", header, payload(26, 0x11));
    std::vector<TensorInfo> tensors;
    check_ok(read_shard("valid.safetensors", &tensors), "a well-formed shard parses");
    check(tensors.size() == 3, "three tensors are indexed");
    if (tensors.size() == 3) {
        check(tensors[0].name == "a" && tensors[0].ndim == 2 && tensors[0].shape[0] == 2 &&
                  tensors[0].shape[1] == 3,
              "BF16 shape is preserved");
        check(tensors[0].dtype == SAFETENSORS_BF16, "BF16 dtype is recognized");
        check(tensors[1].dtype == SAFETENSORS_F32 && tensors[1].element_bytes() == 4,
              "F32 element size is 4");
        check(tensors[2].dtype == SAFETENSORS_F16 && tensors[2].element_bytes() == 2,
              "F16 element size is 2");
        check(tensors[0].file_data_offset == 8 + (int64_t)header.size(),
              "the data section starts after the header");
        check(tensors[2].file_data_offset == 8 + (int64_t)header.size(),
              "every tensor shares the file's data section");
        check(tensors[0].file_path == path_of("valid.safetensors"), "the file path is recorded");
        int64_t bytes = 0;
        check(safetensors_tensor_bytes(tensors[0], &bytes) == 0 && bytes == 12,
              "the declared byte count follows the shape and dtype");
    }
}

/* A checkpoint may carry tensors the text model never loads: Qwen3.8-27B ships a
 * rank-5 Conv3D vision patch embedding alongside 1198 text tensors. The parser
 * must index it (and check its declared byte count exactly) instead of failing
 * the whole file, which would make the model unloadable. */
static void test_higher_rank_tensor_is_indexed(void) {
    const std::string header =
        "{\"visual.patch_embed.proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[4,3,2,2,2],\"data_offsets\":[0,192]},"
        "\"text.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[192,200]}}";
    write_shard("vision.safetensors", header, payload(200, 0x5A));
    std::vector<TensorInfo> tensors;
    check_ok(read_shard("vision.safetensors", &tensors), "a rank-5 tensor does not fail the scan");
    check(tensors.size() == 2, "both tensors are indexed");
    if (tensors.size() == 2) {
        check(tensors[0].ndim == 5 && tensors[0].shape[4] == 2, "the rank-5 shape is kept");
        int64_t bytes = 0;
        check(safetensors_tensor_bytes(tensors[0], &bytes) == 0 && bytes == 192,
              "its declared byte count is validated exactly (96 elements x 2 bytes)");
        check(tensors[1].name == "text.weight" && tensors[1].data_start == 192,
              "a tensor that follows it is still indexed");
    }
}

static void test_metadata_is_allowed(void) {
    write_shard("metadata.safetensors",
                "{\"__metadata__\":{\"format\":\"pt\"},"
                "\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
                payload(2, 0x22));
    std::vector<TensorInfo> tensors;
    check_ok(read_shard("metadata.safetensors", &tensors), "__metadata__ is accepted");
    check(tensors.size() == 1 && tensors[0].name == "a", "metadata is not a tensor");
}

static void test_rejects_bad_layout(void) {
    std::vector<TensorInfo> tensors;

    write_shard("short.safetensors", "", payload(4, 0x33), 4096);
    check_rejected(read_shard("short.safetensors", &tensors), "exceeds the file size",
                   "a header longer than the file");

    write_shard("truncated.safetensors",
                "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", "",
                4096);
    check_rejected(read_shard("truncated.safetensors", &tensors), "exceeds the file size",
                   "a header size past the end of the file");

    /* A sparse file big enough for the declared header, so only the header
     * sanity limit can reject it (the file-size bound passes). */
    {
        const uint64_t declared = 200ull * 1024 * 1024;
        FILE *file = fopen(path_of("huge.safetensors").c_str(), "wb");
        unsigned char raw[8];
        for (int i = 0; i < 8; ++i) raw[i] = (unsigned char)((declared >> (8 * i)) & 0xFF);
        fwrite(raw, 1, 8, file);
        if (ftruncate(fileno(file), (off_t)(declared + 8)) != 0) {
            fprintf(stderr, "cannot create the sparse fixture\n");
        }
        fclose(file);
    }
    check_rejected(read_shard("huge.safetensors", &tensors), "sanity limit",
                   "a header above the sanity limit");

    /* Four bytes: fewer than the 8-byte size field itself. */
    FILE *tiny = fopen(path_of("tiny.bin").c_str(), "wb");
    fwrite("1234", 1, 4, tiny);
    fclose(tiny);
    check_rejected(safetensors_read_header(path_of("tiny.bin").c_str(), tensors),
                   "shorter than the 8-byte header size", "a file with no header at all");
}

static void test_rejects_bad_entries(void) {
    std::vector<TensorInfo> tensors;
    const std::string prefix = "{\"a\":";
    const std::string suffix = "}";
    auto reject = [&](const std::string &entry, const std::string &needle,
                      const std::string &what) {
        write_shard("bad.safetensors", prefix + entry + suffix, payload(64, 0x44));
        check_rejected(read_shard("bad.safetensors", &tensors), needle, what);
    };

    reject("{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}", "unsupported dtype",
           "an unknown dtype");
    reject("{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,2],\"extra\":1}",
           "unknown field", "an unknown field");
    reject("{\"dtype\":\"BF16\",\"data_offsets\":[0,2]}", "must carry dtype, shape and data_offsets",
           "a missing shape");
    reject("{\"dtype\":\"BF16\",\"shape\":[1]}", "must carry dtype, shape and data_offsets",
           "missing data_offsets");
    reject("{\"shape\":[1],\"data_offsets\":[0,2]}", "must carry dtype, shape and data_offsets",
           "a missing dtype");
    reject("{\"dtype\":\"BF16\",\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}",
           "duplicate dtype", "a duplicated field");
    reject("{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2],\"shape\":[1]}",
           "duplicate shape", "a duplicated shape");
    reject("{\"dtype\":\"BF16\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,2]}",
           "more than 8 dimensions", "a rank beyond the parser bound");
    reject("{\"dtype\":\"BF16\",\"shape\":[-1],\"data_offsets\":[0,2]}", "non-negative integer",
           "a negative dimension");
    reject("{\"dtype\":\"BF16\",\"shape\":[1.5],\"data_offsets\":[0,2]}", "shape is malformed",
           "a fractional dimension");
    reject("{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,0]}", "reversed",
           "reversed offsets");
    reject("{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,-2]}", "non-negative integer",
           "a negative offset");
    reject("{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,4096]}",
           "run past the end of the data section", "offsets past the data section");
    reject("{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[0,2]}",
           "data_offsets cover 2 bytes but the shape needs 6", "a payload shorter than the shape");
    reject("{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,16]}",
           "data_offsets cover 16 bytes but the shape needs 4", "a payload longer than the shape");
    /* A zero-element tensor is legal, so shape[0] == 0 must not be rejected. */
    write_shard("empty.safetensors",
                "{\"a\":{\"dtype\":\"BF16\",\"shape\":[0],\"data_offsets\":[0,0]}}", "");
    check_ok(read_shard("empty.safetensors", &tensors), "a zero-element tensor is legal");
}

static void test_rejects_overflow_and_malformed_json(void) {
    std::vector<TensorInfo> tensors;
    write_shard("overflow.safetensors",
                "{\"a\":{\"dtype\":\"BF16\",\"shape\":[4611686018427387904,4],"
                "\"data_offsets\":[0,8]}}",
                payload(8, 0x55));
    check_rejected(read_shard("overflow.safetensors", &tensors),
                   "does not describe a loadable byte count", "a shape product that overflows");

    write_shard("dupkey.safetensors",
                "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},"
                "\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
                payload(2, 0x66));
    check_rejected(read_shard("dupkey.safetensors", &tensors), "duplicate tensor",
                   "a duplicate tensor name");

    write_shard("trailing.safetensors",
                "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}} trailing",
                payload(2, 0x77));
    check_rejected(read_shard("trailing.safetensors", &tensors), "trailing bytes",
                   "junk after the header object");

    write_shard("notjson.safetensors", "[]", payload(2, 0x88));
    check_rejected(read_shard("notjson.safetensors", &tensors), "not a JSON object",
                   "a non-object header");

    write_shard("unclosed.safetensors",
                "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}",
                payload(2, 0x99));
    check_rejected(read_shard("unclosed.safetensors", &tensors), "expected ',' or '}'",
                   "an unterminated entry");
}

static void test_plan_rows(void) {
    write_shard("rows.safetensors",
                "{\"w\":{\"dtype\":\"BF16\",\"shape\":[4,3],\"data_offsets\":[0,24]}}",
                payload(24, 0xAB));
    std::vector<TensorInfo> tensors;
    check_ok(read_shard("rows.safetensors", &tensors), "the row fixture parses");
    if (tensors.empty()) return;
    const TensorInfo &ti = tensors[0];
    const int64_t row_bytes = 6;  /* 3 BF16 elements */

    SafetensorsPlan plan;
    const std::vector<int> order = {3, 0, 2, 1};
    check_ok(safetensors_plan_rows(ti, 24, order.data(), 4, row_bytes, &plan),
             "a valid row permutation is accepted");
    check(plan.dst_bytes == 24 && plan.file_bytes == 24, "the gather moves the whole tensor");

    const std::vector<int> out_of_range = {0, 1, 2, 4};
    check_rejected(safetensors_plan_rows(ti, 24, out_of_range.data(), 4, row_bytes, &plan),
                   "outside 4 rows", "a gather index past the last row");
    const std::vector<int> negative = {0, 1, 2, -1};
    check_rejected(safetensors_plan_rows(ti, 24, negative.data(), 4, row_bytes, &plan),
                   "outside 4 rows", "a negative gather index");
    check_rejected(safetensors_plan_rows(ti, 24, order.data(), 4, 5, &plan),
                   "does not divide the tensor", "a row size that does not divide the tensor");
    check_rejected(safetensors_plan_rows(ti, 23, order.data(), 4, row_bytes, &plan),
                   "destination holds 23 bytes", "a destination smaller than the gather");
    check_rejected(safetensors_plan_rows(ti, 24, nullptr, 4, row_bytes, &plan),
                   "invalid row gather", "a null row order");
    check_rejected(safetensors_plan_rows(ti, 24, order.data(), 0, row_bytes, &plan),
                   "invalid row gather", "an empty gather");
}

static void test_plan_slice(void) {
    write_shard("slice.safetensors",
                "{\"w\":{\"dtype\":\"BF16\",\"shape\":[4,8],\"data_offsets\":[0,64]}}",
                payload(64, 0xCD));
    std::vector<TensorInfo> tensors;
    check_ok(read_shard("slice.safetensors", &tensors), "the slice fixture parses");
    if (tensors.empty()) return;
    const TensorInfo &ti = tensors[0];

    SafetensorsPlan plan;
    check_ok(safetensors_plan_slice(ti, 32, 1, 2, 0, 8, &plan), "a whole-row window is accepted");
    check(plan.dst_bytes == 32 && plan.dst_row_bytes == 16, "the row window moves 32 bytes");

    check_ok(safetensors_plan_slice(ti, 16, 0, 2, 2, 4, &plan), "a column window is accepted");
    check(plan.dst_row_bytes == 8 && plan.dst_bytes == 16, "the column window is 4 columns wide");

    check_ok(safetensors_plan_slice(ti, 64, 0, 4, 0, 8, &plan), "the whole tensor is a valid window");

    check_rejected(safetensors_plan_slice(ti, 32, 3, 2, 0, 8, &plan), "outside the tensor",
                   "a row window past the last row");
    check_rejected(safetensors_plan_slice(ti, 32, 0, 2, 5, 4, &plan), "outside the tensor",
                   "a column window past the last column");
    check_rejected(safetensors_plan_slice(ti, 32, -1, 2, 0, 8, &plan), "outside the tensor",
                   "a negative row offset");
    check_rejected(safetensors_plan_slice(ti, 8, 0, 2, 0, 8, &plan), "destination holds",
                   "a destination smaller than the window");
    check_rejected(safetensors_plan_slice(ti, 32, 0, 0, 0, 8, &plan), "non-empty rectangle",
                   "an empty window");
    write_shard("vector.safetensors",
                "{\"v\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[0,16]}}",
                payload(16, 0xEE));
    check_ok(read_shard("vector.safetensors", &tensors), "the 1-D fixture parses");
    if (!tensors.empty()) {
        check_rejected(safetensors_plan_slice(tensors[0], 16, 0, 1, 0, 8, &plan), "2-D tensor",
                       "a slice of a 1-D tensor");
    }
}

static void test_scan_dir(void) {
    const std::string dir = g_dir + "/broken";
    mkdir(dir.c_str(), 0755);

    std::map<std::string, TensorInfo> index;
    write_shard("broken/aa.safetensors",
                "{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
                payload(2, 0x01));
    check_ok(safetensors_scan_dir(dir.c_str(), index), "a directory of good shards scans");
    check(index.size() == 1 && index.count("x") == 1, "the shard's tensor is indexed");

    /* One valid shard plus one broken one: the whole scan must fail and leave
     * no partial index behind. */
    {
        FILE *file = fopen((dir + "/zz.safetensors").c_str(), "wb");
        fwrite("not a safetensors file", 1, 22, file);
        fclose(file);
    }
    check_rejected(safetensors_scan_dir(dir.c_str(), index), "exceeds the file size",
                   "a broken shard");
    check(index.empty(), "a failed scan leaves no partial index");

    const std::string empty_dir = g_dir + "/empty";
    mkdir(empty_dir.c_str(), 0755);
    check_rejected(safetensors_scan_dir(empty_dir.c_str(), index), "no .safetensors files",
                   "a directory without shards");
}

static void test_scan_dir_merges_shards(void) {
    const std::string dir = g_dir + "/shards";
    mkdir(dir.c_str(), 0755);
    auto write_into = [&](const std::string &name, const std::string &header,
                          const std::string &payload) {
        std::string bytes;
        const int64_t size = (int64_t)header.size();
        for (int i = 0; i < 8; ++i) bytes.push_back((char)((uint64_t)size >> (8 * i)) & 0xFF);
        bytes += header;
        bytes += payload;
        FILE *file = fopen((dir + "/" + name).c_str(), "wb");
        fwrite(bytes.data(), 1, bytes.size(), file);
        fclose(file);
    };
    write_into("model-00001-of-00002.safetensors",
               "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", payload(4, 1));
    write_into("model-00002-of-00002.safetensors",
               "{\"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", payload(4, 2));
    /* A non-safetensors file is ignored, not an error. */
    FILE *other = fopen((dir + "/config.json").c_str(), "wb");
    fwrite("{}", 1, 2, other);
    fclose(other);

    std::map<std::string, TensorInfo> index;
    check_ok(safetensors_scan_dir(dir.c_str(), index), "multiple shards are merged");
    check(index.size() == 2 && index.count("a") == 1 && index.count("b") == 1,
          "both shards contribute their tensors");

    /* The same tensor in two shards is ambiguous: reject it. */
    write_into("model-00003-of-00003.safetensors",
               "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", payload(4, 3));
    check_rejected(safetensors_scan_dir(dir.c_str(), index), "more than one shard",
                   "a tensor duplicated across shards");
    check(index.empty(), "the failed merge leaves no partial index");
}

int main(void) {
    char template_dir[] = "/tmp/safetensors-test-XXXXXX";
    char *created = mkdtemp(template_dir);
    if (created == nullptr) {
        fprintf(stderr, "cannot create a temporary directory\n");
        return 2;
    }
    g_dir = created;

    test_valid_dtypes();
    test_higher_rank_tensor_is_indexed();
    test_metadata_is_allowed();
    test_rejects_bad_layout();
    test_rejects_bad_entries();
    test_rejects_overflow_and_malformed_json();
    test_plan_rows();
    test_plan_slice();
    test_scan_dir();
    test_scan_dir_merges_shards();

    fprintf(stderr, "safetensors: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
