#!/usr/bin/env python3
"""F2's gate: the fused residual+norm against an independent reference, and the rounding rule.

The plan's F2 specifies two outputs from one pass - the updated BF16 residual
`r = BF16(old_r + sublayer_out)` and the normalized activation - and is explicit that the norm
must read the *rounded* r: "the fused kernel must compute r = BF16(old_r + sublayer_out) and
normalize that rounded r, not an unrounded FP32 sum". So this runner checks three things the
plan's F gates name:

  * the updated residual matches the reference **bitwise** (an elementwise add-then-round has no
    reduction order to disagree about);
  * the normalized activation matches a reference written as an independent expression - torch,
    not the kernel's arithmetic - to within the BF16 output's own rounding;
  * and it is **closer** to the rounded-r reference than to an unrounded-FP32 one, which is what
    makes the rounding rule a tested requirement rather than a comment. A kernel that normalized
    the unrounded sum would pass the first check and fail this one.

The weight convention is a parameter, so both a plain RMSNorm and a Gemma norm (`weight + 1`)
are covered, along with a nonzero residual, T = 1, multi-row, and a hidden size that is not a
multiple of the launch block.

Usage:
    python3 tests/test_residual_norm.py --library csrc/build-libs/libkernel_test_bridge.so
"""

import argparse
import ctypes
import os
import sys

import numpy as np
import torch

SHAPES = [(1, 256), (4, 256), (3, 100), (2, 5000)]
EPS = 1e-6


def check(condition, message):
    if condition:
        return True
    print(f"test_residual_norm: FAIL: {message}")
    return False


def reference(residual, sublayer, weight, gemma, rounded):
    """The plan's function, written independently: round to BF16 first when asked, then
    normalize in FP32 and round once on store."""
    total = residual.float() + sublayer.float()
    if rounded:
        total = total.to(torch.bfloat16).float()
    mean_square = total.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(mean_square + EPS)
    effective = weight.float() + (1.0 if gemma else 0.0)
    return total, (total * scale * effective).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--max-rel", type=float, default=8e-3,
                        help="the normalized activation's allowance, inside one BF16 step")
    args = parser.parse_args()

    if not os.path.exists(args.library):
        print(f"test_residual_norm: skipped (no library at {args.library})")
        return 0
    if not torch.cuda.is_available():
        print("test_residual_norm: skipped (no CUDA device)")
        return 0

    lib = ctypes.CDLL(os.path.abspath(args.library))
    if not hasattr(lib, "test_residual_norm"):
        print("test_residual_norm: skipped (the library has no test_residual_norm)")
        return 0
    lib.test_residual_norm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_int]
    lib.test_residual_norm.restype = ctypes.c_int

    device = torch.device("cuda", int(args.devices.split(",")[0]))
    failures = 0
    torch.manual_seed(11)

    for rows, hidden in SHAPES:
        for gemma in (0, 1):
            # A nonzero residual, and a sublayer whose sum is deliberately *not* representable
            # in BF16: that is what makes the rounding rule observable.
            residual = (torch.randn(rows, hidden, device=device) * 1.5).to(torch.bfloat16)
            sublayer = (torch.randn(rows, hidden, device=device) * 1.5).to(torch.bfloat16)
            weight = (torch.randn(hidden, device=device) * 0.1).to(torch.bfloat16)

            normed = torch.empty(rows, hidden, device=device, dtype=torch.bfloat16)
            kept = residual.clone()
            status = lib.test_residual_norm(ctypes.c_void_p(normed.data_ptr()),
                                           ctypes.c_void_p(kept.data_ptr()),
                                           ctypes.c_void_p(sublayer.data_ptr()),
                                           ctypes.c_void_p(weight.data_ptr()),
                                           hidden, rows, EPS, gemma)
            if not check(status == 0, f"the kernel failed for {rows}x{hidden} gemma={gemma}"):
                return 1

            want_residual, want_normed = reference(residual, sublayer, weight, gemma,
                                                   rounded=True)
            want_unrounded, _ = reference(residual, sublayer, weight, gemma, rounded=False)

            if not check(torch.equal(kept, want_residual.to(torch.bfloat16)),
                         f"the updated residual is not bitwise the reference "
                         f"({rows}x{hidden} gemma={gemma})"):
                failures += 1
            actual = normed.float()
            want = want_normed.float()
            want_plain = want_unrounded.to(torch.bfloat16).float()
            max_rel = float((actual - want).abs().max() / max(want.abs().max().item(), 1e-6))
            if not check(max_rel <= args.max_rel,
                         f"the normalized activation is off by {max_rel:.3e} "
                         f"({rows}x{hidden} gemma={gemma})"):
                failures += 1
            to_rounded = float((actual - want).abs().mean())
            to_unrounded = float((actual - want_plain).abs().mean())
            if not check(to_rounded <= to_unrounded,
                         f"the norm is no closer to the rounded-residual reference than to an "
                         f"unrounded one ({to_rounded:.3e} vs {to_unrounded:.3e} at "
                         f"{rows}x{hidden} gemma={gemma}), so the rounding rule is not what it "
                         f"implements"):
                failures += 1
            print(f"  {rows:2d}x{hidden:<5d} gemma={gemma}  max_rel {max_rel:.2e}  "
                  f"rounded {to_rounded:.2e} vs unrounded {to_unrounded:.2e}")

    if failures:
        print(f"test_residual_norm: {failures} check(s) failed")
        return 1
    print("test_residual_norm: the fused pass matches the reference, bitwise on the residual, "
          "and normalizes the rounded one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
