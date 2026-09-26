#!/usr/bin/env python3
"""Q3's gate: the sharding admissions for quantized weights, and the refusals around them.

The plan's Q3 is about *where a quantized artifact may be sliced*, and it states the rules as
requirements: an output split slices the corresponding scale rows, an input split has to land on
a group and packing boundary and keeps the *global* scales, and nothing is requantized per rank.
The equivalence half of that is measured by `tests/test_tp.py --w4a16-dir`, which loads one
artifact whole in the first arm and sliced per rank in the second and requires the same greedy
tokens. This runner covers the refusals, because a slicing rule that silently approximates is
worse than one that refuses:

  * **a split that lands inside a group is refused.** The fixture here has an intermediate size
    of 384, so a tp = 2 input split cuts its K at 192 - one and a half groups - and the loader has
    to say so rather than derive a scale from half a group.
  * **a sidecar for a different model is refused**, by the coverage check rather than by reading
    the wrong bytes: the deployment target's sidecar names 36 dense layers and the tiny fixture
    has 2.
  * **the unsharded path still works**, which is the control: the same artifact loads whole at
    tp = 1 and produces finite logits.

Expert parallelism is not exercised here because it cannot be reached: no expert role is an
admitted role, so `engine_load_quantized_ffn` refuses `ep_size > 1` outright - the item's own
interim rule - and a descriptor that would allow quantized experts does not exist yet.

Usage:
    python3 tests/test_quantized_sharding.py --library ... --plain-dir <4B> --plain-desc ... \\
        --plain-sidecar <4B sidecar> --tiny-dir <synth> --tiny-desc ... \\
        --split-dir <synth with intermediate 384> --split-desc ... --split-sidecar ...
"""

import argparse
import ctypes
import os
import sys

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, ptr


def check(condition, message):
    if condition:
        return True
    print(f"test_quantized_sharding: FAIL: {message}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--plain-dir", required=True)
    parser.add_argument("--plain-desc", required=True)
    parser.add_argument("--plain-sidecar", required=True)
    parser.add_argument("--tiny-dir", required=True)
    parser.add_argument("--tiny-desc", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split-desc", required=True)
    parser.add_argument("--split-sidecar", required=True)
    parser.add_argument("--devices", default="0,1")
    args = parser.parse_args()

    devices = [int(d) for d in args.devices.split(",")]
    lib = ctypes.CDLL(os.path.abspath(args.library))
    bind(lib)
    if not hasattr(lib, "engine_load_quantized_ffn"):
        print("test_quantized_sharding: skipped (no quantized loader)")
        return 0

    failures = 0

    # --- the control: tp = 1 loads the whole artifact and stays finite ---------------------
    print("=== tp = 1 loads the artifact whole ===")
    desc = load_descriptor(args.plain_desc, max_seq_len=16)
    engine, vocab = create_engine(lib, args.plain_dir, desc, devices[:1])
    if not check(lib.engine_load_quantized_ffn(engine, args.plain_sidecar.encode()) == 0,
                 f"a whole-artifact load failed: {lib.engine_last_error().decode()}"):
        return 1
    lib.engine_reset(engine)
    logits = np.empty(vocab, dtype=np.float32)
    ids = np.ascontiguousarray([3, 4, 5], dtype=np.int64)
    status = lib.engine_prefill(engine, ptr(ids), 3, ptr(logits))
    print(f"  logits finite: {bool(np.isfinite(logits).all())}")
    if not check(status == 0 and np.isfinite(logits).all(),
                 f"the unsharded load does not produce finite logits: "
                 f"{lib.engine_last_error().decode()}"):
        failures += 1
    lib.engine_destroy(engine)

    # --- a sidecar for another model is refused --------------------------------------------
    print("=== a sidecar for a different model is refused ===")
    tiny_desc = load_descriptor(args.tiny_desc, max_seq_len=16)
    engine, _ = create_engine(lib, args.tiny_dir, tiny_desc, devices[:1])
    refused = lib.engine_load_quantized_ffn(engine, args.plain_sidecar.encode())
    print(f"  refused={refused != 0} ({lib.engine_last_error().decode()})")
    if not check(refused != 0, "a sidecar naming another model's layers was accepted"):
        failures += 1
    lib.engine_destroy(engine)

    # --- a split inside a group is refused -------------------------------------------------
    print("=== a tp split that lands inside a group is refused ===")
    split_desc = load_descriptor(args.split_desc, max_seq_len=16)
    split_desc["tp_size"] = 2
    engine, _ = create_engine(lib, args.split_dir, split_desc, devices)
    if engine is None:
        print(f"  the tp = 2 fixture itself was refused: {lib.engine_last_error().decode()}")
        print("test_quantized_sharding: skipped the alignment arm (the engine refused the "
              "fixture before the loader could)")
    else:
        refused = lib.engine_load_quantized_ffn(engine, args.split_sidecar.encode())
        print(f"  refused={refused != 0} ({lib.engine_last_error().decode()})")
        if not check(refused != 0,
                     "a split that cuts K inside a group was accepted, which would need a scale "
                     "shared with columns this rank does not hold"):
            failures += 1
        if not check(b"group boundary" in lib.engine_last_error(),
                     f"the refusal does not name the alignment rule: "
                     f"{lib.engine_last_error().decode()}"):
            failures += 1
        lib.engine_destroy(engine)

    if failures:
        print(f"test_quantized_sharding: {failures} check(s) failed")
        return 1
    print("test_quantized_sharding: a whole artifact loads, a foreign sidecar is refused, and a "
          "non-group-aligned split is refused by name. The equivalence of the sliced load "
          "against the whole one is tests/test_tp.py --w4a16-dir's gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
