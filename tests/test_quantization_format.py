#!/usr/bin/env python3
"""Re-derive the INT4 format's fixture independently (plan Q0).

`tests/quantization_format_test.c --emit` writes what the C reference produced for a
deterministic [8, 256] weight: the source FP32 bit patterns, the packed payload, the BF16
scales and the reference dequantization. This runner re-implements the format **from the
plan's definition** in numpy - the BF16 rounding, the per-group scale, the round-to-nearest-
even code, the clipping, the nibble packing and the dequantized value - and requires an exact
match, so the C reference cannot certify itself.

This is the same shape as `test_manifest_hashes.py`: the emitter's own claim is checked by a
second implementation in another language.

Usage:
    python3 tests/test_quantization_format.py --emitter <path to test_quantization_format>
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

GROUP = 128
QMIN, QMAX = -7, 7


def bf16_round(values):
    """Round FP32 to BF16 (round-to-nearest-even on the 16 discarded bits), as the format's
    scale rule requires. Implemented here from the rule, not called from the C side."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    rounding_bias = np.uint64(0x7FFF) + lsb
    rounded = (bits + rounding_bias) & np.uint64(0xFFFFFFFF)
    return (rounded >> np.uint64(16)).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def rederive(weights, n, k, group):
    """The format, written here from the plan's text: s = BF16(max|group|/7) with s = 1 for an
    all-zero group, q = round-nearest-even(w/s) clipped to [-7, 7], two's-complement nibbles
    with the lower K index in the low nibble."""
    assert k % group == 0, "the fixture must be a whole number of groups"
    rows = weights.reshape(n, k).astype(np.float32)
    groups = rows.reshape(n, k // group, group)
    max_abs = np.abs(groups).max(axis=2)
    scale_f32 = np.where(max_abs == 0.0, np.float32(1.0), (max_abs / np.float32(QMAX)).astype(np.float32))
    scale_bits = bf16_round(scale_f32)
    scale = bf16_to_f32(scale_bits)
    codes = np.rint(groups / scale[:, :, None]).astype(np.int32)
    codes = np.clip(codes, QMIN, QMAX)
    assert (codes >= QMIN).all() and (codes <= QMAX).all(), "a code escaped the valid range"
    assert not (codes == -8).any(), "the reserved code -8 must never be produced"

    flat = codes.reshape(n, k)
    low = (flat[:, 0::2] & 0x0F).astype(np.uint8)
    high = ((flat[:, 1::2] & 0x0F) << 4).astype(np.uint8)
    packed = low | high
    dequantized = (flat.astype(np.float32) * scale.repeat(group, axis=1)).reshape(-1)
    return packed.reshape(-1), scale_bits.reshape(-1), dequantized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emitter", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fixture.json")
        proc = subprocess.run([args.emitter, "--emit", path], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"test_quantization_format: the emitter failed: {proc.stdout}{proc.stderr}")
            return 1
        with open(path) as handle:
            fixture = json.load(handle)

    n, k, group = fixture["n"], fixture["k"], fixture["group"]
    weights = np.frombuffer(
        np.array(fixture["weights_f32_bits"], dtype=np.uint32).tobytes(), dtype=np.float32)
    packed_c = np.array(fixture["packed"], dtype=np.uint8)
    scales_c = np.array(fixture["scales_bf16"], dtype=np.uint16)
    dequant_c = np.frombuffer(
        np.array(fixture["dequantized_f32_bits"], dtype=np.uint32).tobytes(), dtype=np.float32)

    if group != GROUP:
        print(f"test_quantization_format: the fixture's group is {group}, expected {GROUP}")
        return 1
    if fixture["packed_bytes"] != n * k // 2 or fixture["scale_count"] != n * (k // group):
        print("test_quantization_format: the fixture's extents disagree with the format")
        return 1

    packed_py, scales_py, dequant_py = rederive(weights, n, k, group)
    ok = True

    if not np.array_equal(packed_py, packed_c):
        bad = int((packed_py != packed_c).sum())
        print(f"test_quantization_format: the payload differs in {bad} of {packed_c.size} bytes")
        ok = False
    if not np.array_equal(scales_py, scales_c):
        bad = int((scales_py != scales_c).sum())
        print(f"test_quantization_format: the scales differ in {bad} of {scales_c.size} values")
        ok = False
    if not np.array_equal(dequant_py.view(np.uint32), dequant_c.view(np.uint32)):
        bad = int((dequant_py.view(np.uint32) != dequant_c.view(np.uint32)).sum())
        print(f"test_quantization_format: the reference dequantization differs in {bad} of "
              f"{dequant_c.size} values")
        ok = False

    error = np.abs(weights - dequant_c)
    print(f"test_quantization_format: {n}x{k}, {group}-wide groups, block max_abs error "
          f"{error.max():.6g}, rms {np.sqrt((error ** 2).mean()):.6g}")
    if not ok:
        return 1
    print("test_quantization_format: the C reference and the numpy re-derivation agree exactly "
          "on the payload, the scales and the dequantization")
    return 0


if __name__ == "__main__":
    sys.exit(main())
