cmake_minimum_required(VERSION 3.18)
project(infer-engine LANGUAGES CXX CUDA)

# Target L20 (sm_89). Override with -DCMAKE_CUDA_ARCHITECTURES=...
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA toolkit components
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Static library: libengine.a
# ---------------------------------------------------------------------------
add_library(engine STATIC
    engine.cu
    memory.cu
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/attention.cu
    kernels/gdn_conv.cu
    kernels/gdn_delta_rule.cu
    kernels/gdn_norm.cu
    kernels/silu.cu
    kernels/embedding.cu
    kernels/argmax.cu
)

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine PUBLIC
    CUDA::cudart
    CUDA::cublas
)

# ---------------------------------------------------------------------------
# Kernel unit tests (optional, built with -DBUILD_TESTS=ON)
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build CUDA kernel unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_rmsnorm ../tests/kernels/test_rmsnorm.cu)
    target_link_libraries(test_rmsnorm PRIVATE engine)
    add_test(NAME test_rmsnorm COMMAND test_rmsnorm)

    add_executable(test_attention ../tests/kernels/test_attention.cu)
    target_link_libraries(test_attention PRIVATE engine)
    add_test(NAME test_attention COMMAND test_attention)

    add_executable(test_gdn ../tests/kernels/test_gdn.cu)
    target_link_libraries(test_gdn PRIVATE engine)
    add_test(NAME test_gdn COMMAND test_gdn)
endif()
