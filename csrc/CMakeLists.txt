cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture")
project(infer-engine LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
find_path(FLASHINFER_INCLUDE_DIR flashinfer/norm.cuh HINTS ENV FLASHINFER_INCLUDE REQUIRED)
find_path(CAUSAL_CONV1D_INCLUDE_DIR causal_conv1d_update.cu
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/causal_conv1d" REQUIRED)

set(GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_command(
    OUTPUT "${GENERATED}/aot_kernels.cpp" "${GENERATED}/aot_kernels.h"
    COMMAND ${CMAKE_COMMAND} -E env "TRITON_PTXAS_PATH=${CUDAToolkit_BIN_DIR}/ptxas"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/triton/build_aot.py"
            --output "${GENERATED}" --arch ${CMAKE_CUDA_ARCHITECTURES}
    DEPENDS triton/build_aot.py triton/gated_delta_rule_chunkwise.py
    VERBATIM
)
add_custom_command(
    OUTPUT "${GENERATED}/causal_conv1d.h" "${GENERATED}/causal_conv1d_update.cuh"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_conv.py"
            --source "${CAUSAL_CONV1D_INCLUDE_DIR}" --output "${GENERATED}"
    DEPENDS ../scripts/generate_conv.py
            "${CAUSAL_CONV1D_INCLUDE_DIR}/causal_conv1d.h"
            "${CAUSAL_CONV1D_INCLUDE_DIR}/causal_conv1d_update.cu"
    VERBATIM
)

add_library(engine SHARED
    engine.cu memory.cu safetensors_loader.cu model_desc.c
    kernels/gemm.cu kernels/layers.cu kernels/flashinfer_norm.cu
    kernels/attention.cu kernels/gdn_conv.cu kernels/fla_gdn.cu
    kernels/gdn_norm.cu kernels/silu.cu kernels/embedding.cu
    "${GENERATED}/aot_kernels.cpp" "${GENERATED}/aot_kernels.h"
    "${GENERATED}/causal_conv1d.h" "${GENERATED}/causal_conv1d_update.cuh"
)
target_include_directories(engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(engine PRIVATE "${GENERATED}")
target_include_directories(engine SYSTEM PRIVATE "${FLASHINFER_INCLUDE_DIR}")
target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(engine PRIVATE FLASHINFER_ENABLE_F16 FLASHINFER_ENABLE_BF16
    FLASHINFER_ENABLE_FP8_E4M3 FLASHINFER_ENABLE_FP8_E5M2 ENABLE_BF16 ENABLE_FP8)
target_link_libraries(engine PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

enable_testing()

# CPU-only: descriptor parsing/validation, no GPU required.
add_executable(test_model_desc ../tests/model_desc_test.c model_desc.c)
target_include_directories(test_model_desc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME test_model_desc COMMAND test_model_desc)

option(BUILD_TESTS "Build GPU numerical tests" OFF)
if(BUILD_TESTS)
    add_executable(test_attention ../tests/kernels/test_attention.cu)
    target_link_libraries(test_attention PRIVATE engine)
    add_test(NAME test_attention COMMAND test_attention)
    add_executable(test_gdn ../tests/kernels/test_gdn.cu)
    target_link_libraries(test_gdn PRIVATE engine)
    add_test(NAME test_gdn COMMAND test_gdn)
    add_library(kernel_test_bridge SHARED ../tests/kernels/kernel_bridge.cu)
    target_link_libraries(kernel_test_bridge PRIVATE engine)
    add_test(NAME test_library_ops COMMAND ${Python3_EXECUTABLE}
             "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_library_ops.py"
             --library $<TARGET_FILE:kernel_test_bridge>)
endif()
