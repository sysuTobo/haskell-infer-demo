#include "kernels.h"
#include "causal_conv1d.h"

#include <stdexcept>
#include <string>

static void check_conv_launch() {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("causal-conv1d: ") + cudaGetErrorString(status));
}

#include "causal_conv1d_update.cuh"

void kernel_causal_conv1d(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                          const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
                          __nv_bfloat16 *conv_state, int conv_dim, int tokens,
                          int kernel_size, cudaStream_t stream) {
    if (kernel_size != 4)
        throw std::invalid_argument("causal-conv1d requires width 4");
    ConvParamsBase params{};
    params.batch = 1;
    params.dim = conv_dim;
    params.seqlen = tokens;
    params.width = kernel_size;
    params.x_batch_stride = tokens * conv_dim;
    params.x_c_stride = 1;
    params.x_l_stride = conv_dim;
    params.out_batch_stride = tokens * conv_dim;
    params.out_c_stride = 1;
    params.out_l_stride = conv_dim;
    params.weight_c_stride = kernel_size;
    params.weight_width_stride = 1;
    params.conv_state_len = kernel_size - 1;
    params.conv_state_batch_stride = conv_dim * (kernel_size - 1);
    params.conv_state_c_stride = kernel_size - 1;
    params.conv_state_l_stride = 1;
    params.x_ptr = const_cast<__nv_bfloat16 *>(x);
    params.weight_ptr = const_cast<__nv_bfloat16 *>(weight);
    params.bias_ptr = const_cast<__nv_bfloat16 *>(bias);
    params.out_ptr = out;
    params.conv_state_ptr = conv_state;
    causal_conv1d_update_cuda<__nv_bfloat16, __nv_bfloat16>(params, stream);
}
