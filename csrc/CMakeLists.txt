cmake_minimum_required(VERSION 3.24)
# SASS for every deployment target plus PTX for compute_90 (the only forward
# compatibility we have: it lets a newer device JIT the nvcc-compiled kernels).
set(CMAKE_CUDA_ARCHITECTURES "86;89;90a;90-virtual" CACHE STRING "CUDA SASS/PTX targets")
# Triton AOT cubins have no PTX fallback, so they are built per architecture and
# selected at runtime by compute capability (see triton/build_aot.py).
set(ENGINE_TRITON_ARCHS "86;89;90" CACHE STRING "Triton AOT targets")
project(infer-engine VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
find_path(FLASHINFER_INCLUDE_DIR flashinfer/norm.cuh HINTS ENV FLASHINFER_INCLUDE REQUIRED)
find_path(CAUSAL_CONV1D_INCLUDE_DIR causal_conv1d_update.cu
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/causal_conv1d" REQUIRED)

set(GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
# aot_versions.txt is declared as an output too: the build-info step reads it, and
# a file a custom command *writes* has to be listed as an output or a fresh build
# directory has no rule for it (make fails with "no rule to make target").
add_custom_command(
    OUTPUT "${GENERATED}/aot_kernels.cpp" "${GENERATED}/aot_kernels.h"
           "${GENERATED}/aot_versions.txt"
    COMMAND ${CMAKE_COMMAND} -E env "TRITON_PTXAS_PATH=${CUDAToolkit_BIN_DIR}/ptxas"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/triton/build_aot.py"
            --output "${GENERATED}" --arch "${ENGINE_TRITON_ARCHS}"
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

# Build provenance for the execution manifest. Generated at build time from the
# artefacts and the tools that produced them (git revision, the Triton/FLA
# versions the AOT generator asserted, the hashes of the generated kernels and of
# the FlashInfer header compiled against, the toolkit/arch/flag configuration), so
# engine_manifest() reports what this build actually is. List-valued arguments are
# '|'-joined: a ';' or a space would be split by the command line.
string(REPLACE ";" "|" ENGINE_CUDA_ARCHS_ARG "${CMAKE_CUDA_ARCHITECTURES}")
string(REPLACE ";" "|" ENGINE_TRITON_ARCHS_ARG "${ENGINE_TRITON_ARCHS}")
set(ENGINE_NVCC_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
string(REPLACE " " "|" ENGINE_NVCC_FLAGS_ARG "${ENGINE_NVCC_FLAGS}")
set(BUILD_INFO_HEADER "${GENERATED}/build_info.h")
add_custom_command(
    OUTPUT "${BUILD_INFO_HEADER}"
    COMMAND ${CMAKE_COMMAND}
            "-DGENERATED=${GENERATED}"
            "-DOUTPUT=${BUILD_INFO_HEADER}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/.."
            "-DENGINE_VERSION=${PROJECT_VERSION}"
            "-DCUDA_TOOLKIT=${CUDAToolkit_VERSION}"
            "-DCUDA_ARCHS_ARG=${ENGINE_CUDA_ARCHS_ARG}"
            "-DTRITON_ARCHS_ARG=${ENGINE_TRITON_ARCHS_ARG}"
            "-DFLASHINFER_HEADER=${FLASHINFER_INCLUDE_DIR}/flashinfer/norm.cuh"
            "-DNVCC_FLAGS_ARG=${ENGINE_NVCC_FLAGS_ARG}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/gen_build_info.cmake"
    DEPENDS "${GENERATED}/aot_kernels.cpp" "${GENERATED}/aot_versions.txt"
            "${GENERATED}/causal_conv1d_update.cuh" gen_build_info.cmake
    VERBATIM
)

add_library(engine SHARED
    engine.cu memory.cu safetensors_loader.cu safetensors.cpp model_desc.c manifest.c sha256.c
    regions.c train.c backward.c train_loop.c alignment.c rollout_queue.c
    layer_dispatch.cu collective.cu moe.cu tap.cu profile.cu
    kernels/gemm.cu kernels/layers.cu kernels/flashinfer_norm.cu
    kernels/attention.cu kernels/gdn_conv.cu kernels/fla_gdn.cu
    kernels/gdn_norm.cu kernels/silu.cu kernels/embedding.cu kernels/mla.cu
    kernels/logprob.cu kernels/backward.cu kernels/backward_paired.cu
    backward_layers.cu
    "${GENERATED}/aot_kernels.cpp" "${GENERATED}/aot_kernels.h"
    "${GENERATED}/causal_conv1d.h" "${GENERATED}/causal_conv1d_update.cuh"
    "${BUILD_INFO_HEADER}"
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

# CPU-only: safetensors header parsing and transfer validation, no GPU required.
# safetensors.cpp is CUDA-free, so a malformed checkpoint is rejected by a test
# that runs anywhere.
add_executable(test_safetensors ../tests/safetensors_test.cpp safetensors.cpp)
target_include_directories(test_safetensors PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME test_safetensors COMMAND test_safetensors)

# CPU-only: the execution manifest. SHA-256 against the FIPS vectors, canonical
# format determinism and the identity matrix (a semantic/numerical change moves
# the matching id only, a deployment-only change moves deployment_id only, and a
# provenance or weight change moves no identity). No GPU and no weights.
add_executable(test_manifest ../tests/manifest_test.c manifest.c sha256.c model_desc.c)
target_include_directories(test_manifest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME test_manifest COMMAND test_manifest)

# The same gate checked by a second implementation: Python re-canonicalizes every
# emitted document and re-derives each digest with hashlib, so the emitter cannot
# certify itself.
add_test(NAME test_manifest_hashes COMMAND ${Python3_EXECUTABLE}
         "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_manifest_hashes.py"
         --emitter $<TARGET_FILE:test_manifest>)

# CPU-only: the Stage-1 region inventory. It checks that the inventory covers the
# plan's in-scope forward operations and nothing else, that it agrees with the
# manifest registry in both directions (region names and the determinism that
# backs an `exact` verdict), that every registered verdict carries the evidence it
# needs, and that the trainer traversal cases are available from no region. No GPU
# and no weights.
add_executable(test_region_inventory ../tests/region_inventory_test.c
               regions.c manifest.c sha256.c model_desc.c)
target_include_directories(test_region_inventory PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME test_region_inventory COMMAND test_region_inventory)

# CPU-only: the trainable runtime's ownership objects (plan Stage 3). The parameter
# store's buffers are opaque slots, so tying, frozen parameters, the borrow/update
# lifetime, derived-copy publication and the teacher-forcing plan are tested exactly
# as they run in the engine; the tying case is the real Qwen3-4B descriptor, whose
# lmHead role maps onto the embedding tensor.
add_executable(test_train ../tests/train_test.c train.c model_desc.c)
target_include_directories(test_train PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME test_train COMMAND test_train
         "${CMAKE_CURRENT_SOURCE_DIR}/../descriptors/qwen3-4b.json")

# CPU-only: the Stage-5 training/rollout baseline. The phase budgets, the phase machine
# (a rollout's borrow is what makes the store refuse an update), the version-bound
# selection records, the exactly-one ratio at unchanged parameters and the host FP64
# sampler's measured frequencies. No GPU and no weights.
add_executable(test_train_loop ../tests/train_loop_test.c train_loop.c train.c backward.c
               model_desc.c sha256.c)
target_include_directories(test_train_loop PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(test_train_loop PRIVATE m)
add_test(NAME test_train_loop COMMAND test_train_loop)

# CPU-only: the Stage-Q INT4 weight-only format (plan Q0). CUDA-free, like the other CPU
# gates: the layout's refusals, BF16 round-to-nearest-even, the two's-complement nibble packing
# (checked against a hand-computed byte pattern, not only a round trip), the reserved code, the
# all-zero group's scale, the signed extrema and the clipping, and the reference dequantizer a
# kernel is later admitted against (Q2). The second test re-derives the same fixture from the
# format's definition in numpy, so the C reference cannot certify itself.
add_executable(test_quantization_format ../tests/quantization_format_test.c linear_weight.cpp)
target_include_directories(test_quantization_format PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(test_quantization_format PRIVATE m)
add_test(NAME test_quantization_format COMMAND test_quantization_format)

add_test(NAME test_quantization_format_python COMMAND ${Python3_EXECUTABLE}
         "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_quantization_format.py"
         --emitter $<TARGET_FILE:test_quantization_format>)

# CPU-only: the Stage-6 numerical-alignment decision. The verdicts are derived from the
# Stage-1 inventory (so the two cannot drift), an exception must carry a finite measured
# bound, an unverified region must name the work that would remove it, and a policy
# change or sampler difference is refused as a numerical comparison rather than scored
# against a numerical bound. The region/case comparisons themselves and the
# gradient/update regression stay Stage 2's and Stage 4's device/CPU gates.
add_executable(test_alignment ../tests/alignment_test.c alignment.c regions.c manifest.c
               sha256.c model_desc.c)
target_include_directories(test_alignment PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(test_alignment PRIVATE m)
add_test(NAME test_alignment COMMAND test_alignment)

# CPU-only: the Stage-7 group objectives (plan Stage 7). An independent FP64 reference in
# the test computes the GSPO and GRPO objectives, and the analytic gradient the
# implementation returns is checked against the reference's central difference, so an
# implementation and a reference that share a formula cannot certify each other. The
# fixture is unequal-length on purpose: GSPO weights responses equally and GRPO tokens,
# and the sequence and token clipping statistics are asserted separately.
add_executable(test_gspo ../tests/gspo_test.c backward.c regions.c manifest.c sha256.c
               model_desc.c)
target_include_directories(test_gspo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(test_gspo PRIVATE m)
add_test(NAME test_gspo COMMAND test_gspo)

# CPU-only: the Stage-8 bounded-staleness admission protocol. Whole completed groups only,
# bounded in groups, tokens and live behavior versions; lag enforced at admission; a
# consumed-group ledger so a retry cannot count a response twice. It also does the plan's
# async gate 1 (lag zero reproduces the synchronous objective and gradient bitwise) and
# gate 3 (lag 0/1/2 injection, reported as a policy change through Stage 6's classifier)
# at the level the protocol can carry them. The GPU half - snapshots, device leases and
# publication transfer - is not here and is recorded as such.
add_executable(test_rollout_queue ../tests/rollout_queue_test.c rollout_queue.c backward.c
               alignment.c regions.c manifest.c sha256.c model_desc.c)
target_include_directories(test_rollout_queue PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(test_rollout_queue PRIVATE m)
add_test(NAME test_rollout_queue COMMAND test_rollout_queue)

# CPU-only: the Stage-4 differentiation contract, the losses, AdamW and the checkpoint
# format. The losses and the optimizer are compared against an independent FP64
# implementation in the test itself, and the region table is checked in both directions
# against the Stage-1 inventory, so neither can drift silently.
add_executable(test_backward ../tests/backward_test.c backward.c regions.c manifest.c
               sha256.c model_desc.c)
target_include_directories(test_backward PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
# The losses and AdamW use libm (exp, log, sqrt, pow, fminf); the other CPU gates do not.
target_link_libraries(test_backward PRIVATE m)
add_test(NAME test_backward COMMAND test_backward)

option(BUILD_TESTS "Build GPU numerical tests" OFF)
if(BUILD_TESTS)
    add_executable(test_attention ../tests/kernels/test_attention.cu)
    target_link_libraries(test_attention PRIVATE engine)
    add_test(NAME test_attention COMMAND test_attention)
    add_executable(test_gdn ../tests/kernels/test_gdn.cu)
    target_link_libraries(test_gdn PRIVATE engine)
    add_test(NAME test_gdn COMMAND test_gdn)
    add_executable(test_collective ../tests/kernels/test_collective.cu)
    target_link_libraries(test_collective PRIVATE engine)
    add_test(NAME test_collective COMMAND test_collective)
    add_executable(test_moe ../tests/kernels/test_moe.cu)
    target_link_libraries(test_moe PRIVATE engine)
    add_test(NAME test_moe COMMAND test_moe)
    add_executable(test_norm ../tests/kernels/test_norm.cu)
    target_link_libraries(test_norm PRIVATE engine)
    add_test(NAME test_norm COMMAND test_norm)
    add_executable(test_rope ../tests/kernels/test_rope.cu)
    target_link_libraries(test_rope PRIVATE engine)
    add_test(NAME test_rope COMMAND test_rope)
    add_executable(test_mla ../tests/kernels/test_mla.cu)
    target_link_libraries(test_mla PRIVATE engine)
    add_test(NAME test_mla COMMAND test_mla)
    # Stage-1 cross-case harness: identical inputs and persistent state under each
    # applicable execution case, adjudicated against the committed inventory, plus
    # the unsupported-shape/case rejections and the region entry-point cost.
    add_executable(test_region_cases ../tests/kernels/test_region_cases.cu)
    target_link_libraries(test_region_cases PRIVATE engine)
    add_test(NAME test_region_cases COMMAND test_region_cases)
    # Stage-4 gradient gate: every backward kernel against a host FP64 reference, the
    # paired attention backward against autograd of the same composition, the GDN core
    # backward with a nonzero initial state and across chunk boundaries, and the
    # run-to-run determinism of both paired kernels.
    add_executable(test_backward_kernels ../tests/kernels/test_backward_kernels.cu)
    target_link_libraries(test_backward_kernels PRIVATE engine)
    add_test(NAME test_backward_kernels COMMAND test_backward_kernels)
    # Stage-2 invariance experiments (plan claims B, C, D and E's forward half).
    # Each one measures a matrix of tilings/decompositions/shapes and prints the
    # bound the region has to carry; the claim's decision is asserted at the end.
    add_executable(test_gdn_invariance ../tests/kernels/test_gdn_invariance.cu)
    target_link_libraries(test_gdn_invariance PRIVATE engine)
    add_test(NAME test_gdn_invariance COMMAND test_gdn_invariance)
    add_executable(test_attention_invariance ../tests/kernels/test_attention_invariance.cu)
    target_link_libraries(test_attention_invariance PRIVATE engine)
    add_test(NAME test_attention_invariance COMMAND test_attention_invariance)
    add_executable(test_gemm_invariance ../tests/kernels/test_gemm_invariance.cu)
    target_link_libraries(test_gemm_invariance PRIVATE engine)
    add_test(NAME test_gemm_invariance COMMAND test_gemm_invariance)
    # Claim E's forward half calls FlashInfer's dispatcher directly (with a real
    # LSE buffer, which the engine's own entry point never asks for), so unlike the
    # others this test needs the FlashInfer include path and feature defines.
    add_executable(test_attention_lse ../tests/kernels/test_attention_lse.cu)
    target_link_libraries(test_attention_lse PRIVATE engine)
    target_include_directories(test_attention_lse SYSTEM PRIVATE "${FLASHINFER_INCLUDE_DIR}")
    target_compile_definitions(test_attention_lse PRIVATE FLASHINFER_ENABLE_F16
        FLASHINFER_ENABLE_BF16 FLASHINFER_ENABLE_FP8_E4M3 FLASHINFER_ENABLE_FP8_E5M2 ENABLE_BF16
        ENABLE_FP8)
    add_test(NAME test_attention_lse COMMAND test_attention_lse)
    # Stage 3's device gate: the all-position teacher-forced forward against an
    # independent transformers reference, the tied-role parameter store, a synthetic
    # update that must refresh both readers and the derived FP32 copy, and the step
    # lifetime. It needs the synthetic checkpoint and torch, so it skips itself when
    # the checkpoint is not there (like test_engine_resources without a device).
    set(ENGINE_SYNTH_MODEL_DIR "/var/pony/cache/bohaotu-haskell/synth-qwen3next"
        CACHE STRING "Synthetic Qwen3-Next checkpoint for the Stage-3 gate")
    add_test(NAME test_train_forward COMMAND ${Python3_EXECUTABLE}
             "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_train_forward.py"
             --library $<TARGET_FILE:engine>
             --model-dir "${ENGINE_SYNTH_MODEL_DIR}"
             --desc "${CMAKE_CURRENT_SOURCE_DIR}/../descriptors/qwen3-next-synth.json")
    # Stage 5's device gate: the SFT step on the tiny dense checkpoint, against a
    # transformers training run (the loss, the gradient's direction per parameter, the
    # overfit), plus determinism, a bitwise state round-trip and the refusals. It needs
    # the synthetic checkpoint (tests/synth/make_qwen3_dense.py) and torch, so it skips
    # itself when the checkpoint is not there, like test_train_forward.
    set(ENGINE_SFT_MODEL_DIR "/var/pony/cache/bohaotu-haskell/synth-qwen3-dense"
        CACHE STRING "Synthetic dense Qwen3 checkpoint for the Stage-5 gate")
    add_test(NAME test_sft COMMAND ${Python3_EXECUTABLE}
             "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_sft.py"
             --library $<TARGET_FILE:engine>
             --model-dir "${ENGINE_SFT_MODEL_DIR}"
             --desc "${CMAKE_CURRENT_SOURCE_DIR}/../descriptors/qwen3-dense-synth.json"
             --torch)
    # Initialization resource safety: a checkpoint that cannot be loaded must
    # leave no handle and no device memory behind. Skips itself when no GPU is
    # visible.
    add_test(NAME test_engine_resources COMMAND ${Python3_EXECUTABLE}
             "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_engine_resources.py"
             --library $<TARGET_FILE:engine>)
    add_library(kernel_test_bridge SHARED ../tests/kernels/kernel_bridge.cu)
    target_link_libraries(kernel_test_bridge PRIVATE engine)
    add_test(NAME test_library_ops COMMAND ${Python3_EXECUTABLE}
             "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_library_ops.py"
             --library $<TARGET_FILE:kernel_test_bridge>)
endif()
