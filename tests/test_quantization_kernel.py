#!/usr/bin/env python3
"""Q2's gate: the weight-only INT4 GEMM against an independent dequantized-weight reference.

The plan's Q gates separate two comparisons: "first verify the kernel against an independent
dequantized-weight reference, then assess the quantizer against the original BF16 model. These
are different comparisons." This runner makes both, and asserts only on the first:

  * **kernel vs the dequantized reference** - the weight is dequantized here, in numpy, from
    the packed bytes the Q0 reference produced, then multiplied in float64. The kernel has to
    match that to within the BF16 output's own rounding, which is a bound a wrong nibble, a
    wrong scale or a wrong group index cannot pass.
  * **dequantized vs the original weight** - the quantization error, reported rather than
    asserted: it is the quantizer's quality, which the plan assesses at the model level (logits
    RMS, top-1 agreement, held-out NLL) once the kernel is routed into the engine, not here.

`--time` also reports the kernel's wall time at M = 1 and a batched M, because the plan asks
for those to be measured separately before choosing specializations.

Usage:
    python3 tests/test_quantization_kernel.py --library csrc/build-libs/libkernel_test_bridge.so \\
        --quantize-reference csrc/build-libs/test_quantization_format [--time]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

GROUP = 128
# The dense FFN shapes of the families this repository loads, plus the group-boundary case.
SHAPES = [(256, 128), (1024, 256), (4096, 5120)]
M_VALUES = [1, 2, 8, 64]


def quantize_reference(reference, weights, workdir, tag):
    """Quantize with the Q0 reference binary and read back the payload and the scales."""
    n, k = weights.shape
    in_path = os.path.join(workdir, f"{tag}.f32")
    packed_path = os.path.join(workdir, f"{tag}.packed")
    scales_path = os.path.join(workdir, f"{tag}.scales")
    with open(in_path, "wb") as handle:
        handle.write(np.ascontiguousarray(weights, dtype=np.float32).tobytes())
    proc = subprocess.run([reference, "--quantize", in_path, packed_path, scales_path,
                           str(n), str(k)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"test_quantization_kernel: the reference refused {n}x{k}: "
                         f"{proc.stdout}{proc.stderr}")
    with open(packed_path, "rb") as handle:
        packed = np.frombuffer(handle.read(), dtype=np.uint8)
    with open(scales_path, "rb") as handle:
        scales = np.frombuffer(handle.read(), dtype=np.uint16)
    return packed, scales


def dequantize_independent(packed, scales, n, k):
    """The Q0 format read back here, from the definition: `w = code * bf16_to_f32(scale)`."""
    codes = np.empty((n, k), dtype=np.int32)
    low = (packed & 0x0F).astype(np.int32).reshape(n, k // 2)
    high = (packed >> 4).astype(np.int32).reshape(n, k // 2)
    codes[:, 0::2] = np.where(low >= 8, low - 16, low)
    codes[:, 1::2] = np.where(high >= 8, high - 16, high)
    scale_f32 = ((scales.astype(np.uint32) << 16).view(np.float32)).reshape(n, k // GROUP)
    return (codes.astype(np.float32) *
            np.repeat(scale_f32, GROUP, axis=1)).reshape(n, k)


def bf16_round(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bias = np.uint64(0x7FFF) + ((bits >> 16) & 1)
    return (((bits + bias) & np.uint64(0xFFFFFFFF)) >> np.uint64(16)).astype(np.uint16)


def bf16(values):
    return (bf16_round(values).astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--quantize-reference", required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--time", action="store_true", help="also report M=1 vs batched timing")
    args = parser.parse_args()

    library = ctypes.CDLL(args.library)
    library.test_gemm_int4.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    if not hasattr(library, "test_gemm_int4"):
        print("test_quantization_kernel: skipped (the library has no test_gemm_int4)")
        return 0

    rng = np.random.RandomState(11)
    failures = 0
    with tempfile.TemporaryDirectory() as workdir:
        for n, k in SHAPES:
            weights = (rng.randn(n, k) * 0.05).astype(np.float32)
            # A per-group scale structure, so a kernel that ignored the group index or reused
            # one scale would be wrong in a way a single-scale fixture could hide.
            weights *= (1.0 + 0.5 * rng.rand(n, k // GROUP).astype(np.float32)
                        ).repeat(GROUP, axis=1)
            packed, scales = quantize_reference(args.quantize_reference, weights, workdir,
                                                f"w{n}x{k}")
            dequant = dequantize_independent(packed, scales, n, k)

            for m in M_VALUES:
                x = (rng.randn(m, k) * 0.5).astype(np.float32)
                # The kernel reads a *BF16* activation buffer, so the fixture hands it the raw
                # 16-bit patterns; the reference multiplies the same rounded values.
                a_bits = np.ascontiguousarray(bf16_round(x))
                a_values = (a_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
                reference = a_values.astype(np.float64) @ dequant.astype(np.float64).T
                # The kernel writes a BF16 [M, N] output: the buffer is 16-bit and the values
                # are widened here for the comparison, exactly as the activation is narrowed.
                out = np.empty((m, n), dtype=np.uint16)
                status = library.test_gemm_int4(
                    ctypes.c_void_p(out.ctypes.data),
                    ctypes.c_void_p(a_bits.ctypes.data),
                    ctypes.c_void_p(packed.ctypes.data),
                    ctypes.c_void_p(scales.ctypes.data), m, n, k, GROUP)
                if status != 0:
                    print(f"test_quantization_kernel: FAIL: the kernel returned {status} for "
                          f"M={m} N={n} K={k}")
                    failures += 1
                    continue
                got = (out.astype(np.uint32) << np.uint32(16)).view(np.float32)
                error = np.abs(got.astype(np.float64) - reference)
                scale = max(float(np.abs(reference).max()), 1e-9)
                max_rel = float(error.max()) / scale
                rms_rel = float(np.sqrt((error ** 2).mean())) / scale
                # The output is BF16, so one ULP is 2^-8 = 3.9e-3 of the magnitude; a wrong
                # nibble, scale or group index lands far above that.
                ok = max_rel <= 8e-3 and rms_rel <= 4e-3
                print(f"{'PASS' if ok else 'FAIL'} M={m:<4} N={n:<6} K={k:<6} "
                      f"kernel-vs-reference max_rel={max_rel:.3g} rms_rel={rms_rel:.3g}")
                if not ok:
                    failures += 1

            quant_error = np.abs(weights - dequant)
            print(f"     quantizer-vs-original N={n} K={k}: max_abs {quant_error.max():.5g}, "
                  f"rms {np.sqrt((quant_error ** 2).mean()):.5g} "
                  f"(of |w| max {np.abs(weights).max():.4g}) - reported, not gated here")

        # A shape the format cannot represent must be refused by the host wrapper, not run.
        n, k = 128, 192
        bad = np.zeros(n * k // 2, dtype=np.uint8)
        bscales = np.zeros(n * (k // GROUP), dtype=np.uint16)
        out = np.zeros((1, n), dtype=np.uint16)
        status = library.test_gemm_int4(ctypes.c_void_p(out.ctypes.data),
                                        ctypes.c_void_p(out.ctypes.data),
                                        ctypes.c_void_p(bad.ctypes.data),
                                        ctypes.c_void_p(bscales.ctypes.data), 1, n, k, GROUP)
        if status == 0:
            print("test_quantization_kernel: FAIL: K=192 (1.5 groups) was accepted")
            failures += 1
        else:
            print(f"PASS K=192 refused ({status})")

        if args.time:
            n, k = 4096, 5120
            weights = (rng.randn(n, k) * 0.05).astype(np.float32)
            packed, scales = quantize_reference(args.quantize_reference, weights, workdir, "tw")
            for m in (1, 64):
                x_bits = np.ascontiguousarray(bf16_round((rng.randn(m, k) * 0.5).astype(np.float32)))
                out = np.empty((m, n), dtype=np.uint16)
                for _ in range(3):
                    library.test_gemm_int4(ctypes.c_void_p(out.ctypes.data),
                                           ctypes.c_void_p(x_bits.ctypes.data),
                                           ctypes.c_void_p(packed.ctypes.data),
                                           ctypes.c_void_p(scales.ctypes.data), m, n, k, GROUP)
                runs = []
                for _ in range(10):
                    start = time.perf_counter()
                    library.test_gemm_int4(ctypes.c_void_p(out.ctypes.data),
                                           ctypes.c_void_p(x_bits.ctypes.data),
                                           ctypes.c_void_p(packed.ctypes.data),
                                           ctypes.c_void_p(scales.ctypes.data), m, n, k, GROUP)
                    runs.append((time.perf_counter() - start) * 1e6)
                weight_bytes = n * k / 2 + n * (k // GROUP) * 2
                median = sorted(runs)[len(runs) // 2]
                line = (f"timing M={m:<3} N={n} K={k}: median {median:.1f} us, "
                        f"{weight_bytes / (median * 1e-6) / 1e9:.1f} GB/s of packed weight "
                        f"({weight_bytes / 1e6:.1f} MB read)")
                # The comparison that matters is against the *same values* in BF16, so the
                # time gap is the format's and not the fixture's. torch is only needed here.
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = torch.device("cuda", int(args.devices.split(",")[0]))
                        w_bf16 = torch.from_numpy(
                            (bf16_round(dequant).astype(np.uint32) << 16).view(np.float32)
                        ).to(device).to(torch.bfloat16)
                        xb = torch.from_numpy(
                            (bf16_round(x).astype(np.uint32) << 16).view(np.float32)
                        ).to(device).to(torch.bfloat16)
                        for _ in range(3):
                            torch.matmul(xb, w_bf16.t())
                        torch.cuda.synchronize()
                        samples = []
                        for _ in range(10):
                            start = time.perf_counter()
                            torch.matmul(xb, w_bf16.t())
                            torch.cuda.synchronize()
                            samples.append((time.perf_counter() - start) * 1e6)
                        bf16_median = sorted(samples)[len(samples) // 2]
                        line += (f"; the same values as BF16: {bf16_median:.1f} us "
                                 f"(2x the bytes read, and {bf16_median / median:.2f}x the time)")
                except ImportError:
                    line += "; BF16 comparison skipped (no torch)"
                print(line)

    if failures:
        print(f"test_quantization_kernel: {failures} check(s) failed")
        return 1
    print("test_quantization_kernel: the INT4 GEMM matches an independently dequantized-weight "
          "reference within the BF16 output's rounding and refuses shapes the format cannot hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
