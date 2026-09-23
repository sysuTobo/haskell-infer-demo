"""Capture engine logits as a golden reference, and compare two captures bitwise.

Capture (needs weights):
  python tests/capture_logits.py --library csrc/build-libs/libengine.so \
      --model-dir "$MODEL_DIR" --output /path/golden.npz

Verify a refactor did not change numerics (same build, same prompt => deterministic):
  python tests/capture_logits.py --compare old.npz new.npz

The compare gate is max_abs == 0 on every array: the engine is deterministic for
a fixed build, so any difference means behavior changed.
"""

import argparse
import ctypes
import json
import sys
import time

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, ptr

# "The capital of France is" -> Qwen tokenizer ids (verified in the worklog).
PROMPT_MAIN = [760, 6511, 314, 9564, 369]

# The engine chunks prefill at 128 tokens; 129 crosses a chunk boundary.
LONG_PROMPT_TOKENS = 129


def capture(args):
    lib = bind(ctypes.CDLL(args.library))
    devices = [int(x) for x in args.devices.split(",")]
    descriptor = load_descriptor(args.desc, max_seq_len=args.max_seq_len)

    start = time.monotonic()
    engine, vocab = create_engine(lib, args.model_dir, descriptor, devices)
    print(f"Loaded in {time.monotonic() - start:.1f}s (vocab {vocab})", flush=True)
    logits = np.empty(vocab, dtype=np.float32)
    records = {}

    def status(code):
        assert code == 0, lib.engine_last_error().decode()

    def run_case(name, prompt):
        lib.engine_reset(engine)
        assert lib.engine_seq_len(engine) == 0
        prompt = np.ascontiguousarray(prompt, dtype=np.int64)
        status(lib.engine_prefill(engine, ptr(prompt), len(prompt), ptr(logits)))
        records[f"prompt_{name}"] = prompt
        tokens = []
        for step in range(args.steps):
            assert np.isfinite(logits).all(), f"{name} step {step}: non-finite logits"
            records[f"logits_{name}_{step}"] = logits.copy()
            token = int(logits.argmax())
            tokens.append(token)
            print(f"{name} step {step}: top1={token} max={logits.max():.4f}", flush=True)
            if step + 1 < args.steps:
                status(lib.engine_decode(engine, token, ptr(logits)))
        records[f"tokens_{name}"] = np.array(tokens, dtype=np.int64)
        # The sequence grows across steps; verify no drift.
        assert lib.engine_seq_len(engine) == len(prompt) + len(tokens) - 1, "seq_len drift"

    try:
        run_case("main", PROMPT_MAIN)
        run_case("long", np.resize(np.array(PROMPT_MAIN, dtype=np.int64), LONG_PROMPT_TOKENS))
    finally:
        lib.engine_destroy(engine)

    meta = {"vocab": int(vocab), "devices": devices, "steps": args.steps,
            "desc": args.desc, "prompt_main": PROMPT_MAIN}
    records["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(args.output, **records)
    print(f"Wrote {args.output}", flush=True)


def compare(args):
    left = np.load(args.compare[0])
    right = np.load(args.compare[1])
    left_keys, right_keys = set(left.files), set(right.files)
    if left_keys != right_keys:
        print(f"key mismatch: only-left={sorted(left_keys - right_keys)} "
              f"only-right={sorted(right_keys - left_keys)}")
        return 1
    failed = False
    # Two captures are only comparable when they were taken the same way: a
    # different prompt, descriptor or device set means the logits differ for
    # reasons that have nothing to do with the change under test.
    if "meta" in left_keys:
        if left["meta"].item() != right["meta"].item():
            print(f"meta mismatch:\n  old: {left['meta'].item()}\n  new: {right['meta'].item()}")
            failed = True
    worst_key, worst_abs, worst_rms, differing = None, 0.0, 0.0, 0
    for key in sorted(left_keys):
        if key == "meta":
            continue
        a, b = left[key], right[key]
        if a.shape != b.shape:
            print(f"{key}: shape {a.shape} vs {b.shape}")
            failed = True
            continue
        # A NaN or infinity is never a match and would otherwise slip through the
        # max() below (NaN compares false, leaving the array out of worst_key),
        # so a capture with non-finite values fails instead of reporting success.
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            bad = int(np.count_nonzero(~np.isfinite(a))) + int(np.count_nonzero(~np.isfinite(b)))
            print(f"{key}: {bad} non-finite value(s); a capture must be finite")
            failed = True
            continue
        if not np.array_equal(a, b):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            count = int(np.count_nonzero(a != b))
            differing += count
            rms = float(np.sqrt(np.mean(diff ** 2)))
            print(f"{key}: differing={count} max_abs={diff.max():.6g} rms={rms:.6g}")
            if diff.max() > worst_abs:
                worst_key, worst_abs, worst_rms = key, float(diff.max()), rms
    if worst_key is None and not failed:
        print("bitwise identical (max_abs == 0 on every array)")
        return 0
    if worst_key is not None:
        print(f"NOT bitwise identical: worst={worst_key} max_abs={worst_abs:.6g} "
              f"rms={worst_rms:.6g} differing_elements={differing}")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--model-dir")
    parser.add_argument("--desc", default="descriptors/qwen38-27b.json")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="override the descriptor's max_seq_len")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    args = parser.parse_args()
    if args.compare:
        return compare(args)
    if not args.library or not args.model_dir or not args.output:
        parser.error("--library, --model-dir and --output are required for capture")
    capture(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
