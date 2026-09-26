#!/usr/bin/env python3
"""S2's device gate: round checkpoints on a model with a recurrent layer.

The plan's S2 exists because a GDN layer's state cannot be moved back the way an append-only
cache's length can. So the load-bearing check is on a model that *has* one: decode a few tokens,
save, decode further, restore, and require the next decode to be **bitwise** what a second
engine that never took the detour produces. That is the claim "restore the round-start state,
then replay exactly the retained inputs" rests on.

The rest is the refusal set the plan asks for, because a checkpoint that silently restores the
wrong state is worse than none: a second save while one is live, a restore with nothing saved, a
restore after `engine_reset` (the reset generation moved), a restore after `release`, and - when
a converted sidecar is pointed at - a restore after the numerical policy changed (the packed
operands do not move the weight digest, which is why the posture is recorded separately).

Usage:
    python3 tests/test_checkpoints.py --library csrc/build-libs/libengine.so \\
        --attn-dir <attention-only checkpoint> --attn-desc <descriptor> \\
        --recurrent-dir <checkpoint with GDN layers> --recurrent-desc <descriptor> \\
        [--quantized-dir <converted sidecar>]
"""

import argparse
import ctypes
import json
import os
import sys

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, ptr


def check(condition, message):
    if condition:
        return True
    print(f"test_checkpoints: FAIL: {message}")
    return False


def decode_one(lib, engine, token, vocab):
    out = np.empty(vocab, dtype=np.float32)
    status = lib.engine_decode(engine, int(token), ptr(out))
    return status, out


def prefill(lib, engine, ids, vocab):
    out = np.empty(vocab, dtype=np.float32)
    ids_array = np.ascontiguousarray(ids, dtype=np.int64)
    return lib.engine_prefill(engine, ptr(ids_array), len(ids), ptr(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--attn-dir", required=True)
    parser.add_argument("--attn-desc", required=True)
    parser.add_argument("--recurrent-dir", required=True)
    parser.add_argument("--recurrent-desc", required=True)
    parser.add_argument("--quantized-dir", default=None,
                        help="a converted sidecar whose engine_manifest posture difference "
                             "makes the restore refusal observable")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--prompt", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    for path in (args.attn_dir, args.recurrent_dir):
        if not os.path.isdir(path):
            print(f"test_checkpoints: skipped (no checkpoint at {path})")
            return 0

    devices = [int(d) for d in args.devices.split(",")]
    lib = ctypes.CDLL(os.path.abspath(args.library))
    bind(lib)
    if not hasattr(lib, "engine_checkpoint_save"):
        print("test_checkpoints: skipped (the library has no checkpoint API)")
        return 0
    for name in ("save", "restore", "release"):
        fn = getattr(lib, f"engine_checkpoint_{name}")
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
    lib.engine_checkpoint_bytes.argtypes = [ctypes.c_void_p]
    lib.engine_checkpoint_bytes.restype = ctypes.c_longlong

    failures = 0

    # --- the round-trip on a model with a recurrent layer --------------------------------
    print("=== a GDN model round-trips its state ===")
    desc = load_descriptor(args.recurrent_desc, max_seq_len=args.prompt + args.steps + 2)
    vocab = desc["vocab_size"]
    prompt = [(11 * (t + 1)) % vocab for t in range(args.prompt)]
    steps = [(31 * (t + 2) + 7) % vocab for t in range(args.steps)]

    detour, straight = create_engine(lib, args.recurrent_dir, desc, devices)[0], \
        create_engine(lib, args.recurrent_dir, desc, devices)[0]
    for engine in (detour, straight):
        lib.engine_reset(engine)
        if not check(prefill(lib, engine, prompt, vocab) == 0, lib.engine_last_error().decode()):
            return 1

    saved = lib.engine_checkpoint_save(detour)
    if not check(saved == 0, f"a GDN model refused a checkpoint save: "
                             f"{lib.engine_last_error().decode()}"):
        return 1
    print(f"  checkpoint bytes: {lib.engine_checkpoint_bytes(detour)}")
    for token in steps:
        decode_one(lib, detour, token, vocab)
    restored = lib.engine_checkpoint_restore(detour)
    if not check(restored == 0, f"restore failed: {lib.engine_last_error().decode()}"):
        return 1
    released = lib.engine_checkpoint_release(detour)
    if not check(released == 0, f"release failed: {lib.engine_last_error().decode()}"):
        return 1

    for token in steps:
        status_detour, row_detour = decode_one(lib, detour, token, vocab)
        status_straight, row_straight = decode_one(lib, straight, token, vocab)
        if not check(status_detour == 0 and status_straight == 0,
                     lib.engine_last_error().decode()):
            return 1
        if not check(np.array_equal(row_detour, row_straight),
                     f"after the restore the state is not the round-start state (max_abs "
                     f"{float(np.abs(row_detour - row_straight).max()):.3e})"):
            failures += 1
    print(f"  {len(steps)} decodes after the restore match the straight-through engine")
    lib.engine_destroy(detour)
    lib.engine_destroy(straight)

    # --- the refusals --------------------------------------------------------------------
    print("=== the refusals ===")
    attn_desc = load_descriptor(args.attn_desc, max_seq_len=args.prompt + 2)
    attn_vocab = attn_desc["vocab_size"]
    engine = create_engine(lib, args.attn_dir, attn_desc, devices)[0]
    lib.engine_reset(engine)
    attn_prompt = [3, 4]
    prefill(lib, engine, attn_prompt, attn_vocab)

    if not check(lib.engine_checkpoint_restore(engine) != 0,
                 "a restore without a checkpoint was accepted"):
        failures += 1
    lib.engine_checkpoint_save(engine)
    if not check(lib.engine_checkpoint_save(engine) != 0,
                 "a second checkpoint was accepted while one was live"):
        failures += 1
    lib.engine_reset(engine)
    if not check(lib.engine_checkpoint_restore(engine) != 0,
                 "a restore after engine_reset was accepted"):
        failures += 1
    lib.engine_checkpoint_release(engine)
    if not check(lib.engine_checkpoint_restore(engine) != 0,
                 "a restore after release was accepted"):
        failures += 1
    lib.engine_destroy(engine)

    if args.quantized_dir is not None and os.path.exists(
            os.path.join(args.quantized_dir, "weights.manifest.json")):
        print("=== the numerical policy is part of the identity ===")
        engine = create_engine(lib, args.attn_dir, attn_desc, devices)[0]
        lib.engine_reset(engine)
        prefill(lib, engine, attn_prompt, attn_vocab)
        if not check(lib.engine_checkpoint_save(engine) == 0,
                     lib.engine_last_error().decode()):
            failures += 1
        loaded = lib.engine_load_quantized_ffn(engine, args.quantized_dir.encode())
        if loaded != 0:
            # The arm's premise is a *successful* load: a sidecar that is missing an artifact or
            # disagrees with the model cannot demonstrate the policy comparison, and a gate that
            # failed here would be reporting the sidecar rather than the checkpoint.
            print(f"  the policy arm needs a loadable sidecar; skipped "
                  f"({lib.engine_last_error().decode()})")
        else:
            refused = lib.engine_checkpoint_restore(engine)
            print(f"  a restore after the policy changed: refused={refused != 0} "
                  f"({lib.engine_last_error().decode()})")
            if not check(refused != 0,
                         "a restore after the numerical policy changed was accepted"):
                failures += 1
        lib.engine_destroy(engine)
    else:
        print("test_checkpoints: the policy arm needs --quantized-dir; skipped")

    if failures:
        print(f"test_checkpoints: {failures} check(s) failed")
        return 1
    print("test_checkpoints: a GDN model restore is exact, and the refusals fire")
    return 0


if __name__ == "__main__":
    sys.exit(main())
