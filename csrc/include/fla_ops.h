#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "layers.h"

size_t kernel_fla_workspace_size(int tokens, int value_heads);

void kernel_fla_recurrent(__nv_bfloat16 *out, const __nv_bfloat16 *q,
                          const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                          const float *log_decay, const float *beta,
                          float *state, cudaStream_t stream);

void kernel_fla_gdn(__nv_bfloat16 *out, const __nv_bfloat16 *qkv,
                    const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                    const __nv_bfloat16 *A_log, const __nv_bfloat16 *dt_bias,
                    float *state, void *workspace, int tokens,
                    int key_heads, int value_heads, cudaStream_t stream,
                    const GdnTapSites *taps = nullptr);
