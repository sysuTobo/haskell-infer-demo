import argparse
import importlib.metadata
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import gated_delta_rule_chunkwise as chunk
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel


SPECS = [
    ("prepare", "gdr_prepare_qkv_gbeta_qwen35_kernel", "*bf16,*bf16,*bf16,*bf16,*bf16,*bf16,*bf16,*bf16,*fp32,*fp32,i32,i32,i32,i32,128,128", ("seq_len", "num_value_heads", "1"), 4, 2),
    ("cumsum", "gdr_chunk_local_cumsum_qwen35_kernel", "*fp32,*fp32,i32,i32,64", ("(seq_len+63)/64", "num_value_heads", "1"), 1, 1),
    ("kkt", "gdr_chunk_scaled_dot_kkt_qwen35_kernel", "*bf16,*fp32,*fp32,*fp32,i32,i32,64,64,128", ("(seq_len+63)/64", "num_value_heads", "1"), 4, 2),
    ("solve", "gdr_solve_tril_64_qwen35_kernel", "*fp32,*bf16,i32,i32", ("(seq_len+63)/64", "num_value_heads", "1"), 4, 2),
    ("recompute", "gdr_recompute_w_u_qwen35_kernel", "*bf16,*bf16,*fp32,*bf16,*bf16,*bf16,*fp32,i32,i32,128,128,64,64,64", ("(seq_len+63)/64", "num_value_heads", "1"), 4, 2),
    ("state", "gdr_chunk_state_qwen35_kernel", "*bf16,*bf16,*bf16,*fp32,*fp32,*fp32,*bf16,*fp32,i32,i32,32,64,128,128,64", ("4", "num_value_heads", "1"), 4, 2),
    ("output", "gdr_chunk_o_qwen35_kernel", "*bf16,*bf16,*bf16,*fp32,*fp32,*bf16,i32,i32,fp32,64,32,64,128,128", ("4", "(seq_len+63)/64", "num_value_heads"), 4, 2),
]


PREAMBLE = r'''#include "aot_kernels.h"
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

static void check_driver(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}

struct Kernel {
    const unsigned char *image;
    const char *name;
    unsigned shared;
    std::mutex mutex;
    std::map<CUcontext, std::pair<CUmodule, CUfunction>> contexts;

    CUfunction get() {
        CUcontext context;
        check_driver(cuCtxGetCurrent(&context));
        if (!context) throw std::runtime_error("FLA requires a current CUDA context");
        std::lock_guard<std::mutex> guard(mutex);
        auto found = contexts.find(context);
        if (found != contexts.end()) return found->second.second;
        CUmodule module;
        check_driver(cuModuleLoadData(&module, image));
        CUfunction function;
        CUresult status = cuModuleGetFunction(&function, module, name);
        if (status != CUDA_SUCCESS) {
            cuModuleUnload(module);
            check_driver(status);
        }
        if (shared > 49152)
            check_driver(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
        contexts.emplace(context, std::make_pair(module, function));
        return function;
    }
};
'''


def emit(name, fn, signature, constants, grid, warps, stages, target):
    compiled = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants),
        target=target, options={"num_warps": warps, "num_stages": stages},
    )
    if compiled.metadata.global_scratch_size:
        raise RuntimeError(f"{name} unexpectedly requires global scratch")
    params = []
    names = []
    for arg in fn.arg_names:
        if arg not in signature:
            continue
        dtype = signature[arg]
        ctype = "const void *" if dtype.startswith("*") else {"i32": "int32_t ", "fp32": "float "}[dtype]
        params.append(ctype + arg)
        names.append(arg)
    declaration = f"void aot_{name}(CUstream stream, " + ", ".join(params) + ")"
    data = ",".join(str(x) for x in compiled.asm["cubin"])
    source = f"static const unsigned char image_{name}[] = {{{data}}};\n"
    source += f'static Kernel kernel_{name}{{image_{name}, "{compiled.name}", {compiled.metadata.shared}}};\n'
    source += declaration + " {\n    void *scratch = nullptr;\n"
    source += "    void *args[] = {" + ", ".join("&" + x for x in names) + ", &scratch};\n"
    source += f"    check_driver(cuLaunchKernel(kernel_{name}.get(), {', '.join(grid)}, {warps * 32}, 1, 1, {compiled.metadata.shared}, stream, args, nullptr));\n}}\n"
    print(f"{name}: {len(compiled.asm['cubin'])} bytes, shared={compiled.metadata.shared}", flush=True)
    return declaration + ";\n", source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", type=int, default=86)
    args = parser.parse_args()
    if triton.__version__ != "3.4.0" or importlib.metadata.version("fla-core") != "0.5.2":
        raise RuntimeError("AOT requires triton==3.4.0 and fla-core==0.5.2")
    target = GPUTarget("cuda", args.arch, 32)
    declarations = ["#pragma once\n#include <cuda.h>\n#include <cstdint>\n"]
    sources = [PREAMBLE]
    for name, symbol, spec, grid, warps, stages in SPECS:
        fn = getattr(chunk, symbol)
        signature, constants = {}, {}
        for arg, dtype in zip(fn.arg_names, spec.split(","), strict=True):
            if dtype.isdigit():
                constants[arg] = int(dtype)
            else:
                signature[arg] = dtype
        h, cc = emit(name, fn, signature, constants, grid, warps, stages, target)
        declarations.append(h)
        sources.append(cc)
    fn = fused_recurrent_gated_delta_rule_fwd_kernel
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = fn.fn
    signature = {
        "q": "*bf16", "k": "*bf16", "v": "*bf16", "g": "*fp32",
        "gk": "*fp32", "gv": "*fp32", "beta": "*fp32", "A_log": "*fp32",
        "dt_bias": "*fp32", "o": "*bf16", "h0": "*fp32", "ht": "*fp32",
        "cu_seqlens": "*i64", "scale": "fp32", "T": "i32",
    }
    constants = {
        "H": 48, "HV": 48, "K": 128, "V": 128, "BK": 128, "BV": 8,
        "USE_G": True, "USE_GK": False, "USE_GV": False,
        "USE_QK_L2NORM_IN_KERNEL": False, "IS_BETA_HEADWISE": True,
        "USE_INITIAL_STATE": True, "STORE_FINAL_STATE": True,
        "STATE_V_FIRST": False, "IS_VARLEN": False,
        "USE_GATE_IN_KERNEL": False, "HAS_DT_BIAS": False,
        "APPLY_BETA_SIGMOID": False, "ALLOW_NEG_EIGVAL": False,
    }
    for heads in (16, 48):
        constants["H"] = heads
        h, cc = emit(f"recurrent{heads}", fn, signature, constants, ("16", "48", "1"), 1, 3, target)
        declarations.append(h)
        sources.append(cc)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aot_kernels.h").write_text("".join(declarations))
    (output / "aot_kernels.cpp").write_text("\n".join(sources))


if __name__ == "__main__":
    main()
