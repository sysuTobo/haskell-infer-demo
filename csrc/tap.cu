/**
 * tap.cu - Debug taps: dump one sub-layer's activation as raw float32
 * [tokens, hidden] to disk so it can be compared against the transformers
 * reference (tests/synth/taps.py).
 *
 * Slow on purpose (a host sync per tap) -- this exists to localize numerical
 * differences between the engine and the reference implementation.
 */
#include "layers.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

void tap_parse_env(TapConfig *taps) {
    const char *layers = getenv("INFER_TAP_LAYERS");
    if (layers == nullptr) return;
    const char *dir = getenv("INFER_TAP_DIR");
    if (dir == nullptr || !*dir) {
        fprintf(stderr, "[tap] INFER_TAP_LAYERS set without INFER_TAP_DIR\n");
        return;
    }
    taps->dir = dir;
    const char *cursor = layers;
    while (*cursor) {
        char *end = nullptr;
        const long value = strtol(cursor, &end, 10);
        if (end == cursor) break;
        taps->layers.push_back((int)value);
        cursor = (*end == ',') ? end + 1 : end;
    }
    fprintf(stderr, "[tap] dumping %zu layer(s) to %s\n", taps->layers.size(), taps->dir.c_str());
}

void tap_dump(const TapConfig *taps, const char *kind, int layer, int device,
              const __nv_bfloat16 *data, int tokens, const ModelDims *dims) {
    if (taps == nullptr || taps->dir.empty()) return;
    if (std::find(taps->layers.begin(), taps->layers.end(), layer) == taps->layers.end())
        return;
    const size_t elements = (size_t)tokens * dims->hidden_size;
    std::vector<__nv_bfloat16> staged(elements);
    cudaSetDevice(device);
    cudaError_t status = cudaMemcpy(staged.data(), data, elements * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "[tap] %s layer %d copy failed: %s\n", kind, layer,
                cudaGetErrorString(status));
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s_%02d_seq0_tok%d.f32", taps->dir.c_str(), kind, layer,
             tokens);
    FILE *file = fopen(path, "wb");
    if (file == nullptr) {
        fprintf(stderr, "[tap] cannot open %s\n", path);
        return;
    }
    std::vector<float> expanded(elements);
    for (size_t i = 0; i < elements; ++i) expanded[i] = __bfloat162float(staged[i]);
    fwrite(expanded.data(), sizeof(float), elements, file);
    fclose(file);
}
