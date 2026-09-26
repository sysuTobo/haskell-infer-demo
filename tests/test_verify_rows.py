#!/usr/bin/env python3
"""S1's device gate: the bounded all-position verification against serial execution.

The plan's S1 adds `engine_verify_rows`, and the plan's S3 warns that admitting it needs more
than "it returned rows": every row has to decide what serial execution would have decided, with
the differences reported rather than hidden behind a tolerance. So this compares two engines'
views of the same state:

  * the batch engine consumes `prompt` and then verifies a window of n ids in one call;
  * the serial engine consumes `prompt` and then decodes those same n ids one at a time.

`n = 1` must be **bitwise** equal to a decode (one token is one execution case, so a difference
there is a bug, not a tolerance). For `n > 1` the two are different execution cases - Stage 2
measured that chunked and recurrent execution differ - so the rows are reported as
max_abs/rms and the plan's claim is checked where it matters: **every row's argmax must be the
serial argmax**, because a round that verified a token against a different decision would
commit a token the target never chose.

The rollback is checked too: `engine_truncate` must put the batch engine back where the serial
one is, so a subsequent decode agrees; and it must be *refused* for a model with a recurrent
layer, because a GDN state cannot be rewound (that is S2's checkpoint work).

Usage:
    python3 tests/test_verify_rows.py --library csrc/build-libs/libengine.so \\
        --attn-dir <attention-only checkpoint> --attn-desc <descriptor> \\
        --recurrent-dir <checkpoint with GDN layers> --recurrent-desc <descriptor>
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
    print(f"test_verify_rows: FAIL: {message}")
    return False


def flat_rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=np.float64) -
                                  np.asarray(b, dtype=np.float64)) ** 2)))


def all_rows(lib, engine, ids, vocab):
    rows = np.empty((len(ids), vocab), dtype=np.float32)
    # The id array has to outlive the call: ptr() hands the C side a bare address, and a
    # temporary numpy array is freed as soon as its expression's refcount drops.
    id_array = np.ascontiguousarray(ids, dtype=np.int64)
    status = lib.engine_verify_rows(engine, ptr(id_array), len(ids), ptr(rows), rows.size)
    return status, rows


def serial_rows(lib, engine, ids, vocab):
    rows = []
    for token in ids:
        row = np.empty(vocab, dtype=np.float32)
        status = lib.engine_decode(engine, int(token), ptr(row))
        if status != 0:
            return status, None
        rows.append(row)
    return 0, np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--attn-dir", required=True,
                        help="an attention-only checkpoint, where truncation is admitted")
    parser.add_argument("--attn-desc", required=True)
    parser.add_argument("--recurrent-dir", required=True,
                        help="a checkpoint with a recurrent layer (GDN), where it is refused")
    parser.add_argument("--recurrent-desc", required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--prompt", type=int, default=6)
    parser.add_argument("--window", type=int, default=4)
    args = parser.parse_args()

    for path in (args.attn_dir, args.recurrent_dir):
        if not os.path.isdir(path):
            print(f"test_verify_rows: skipped (no checkpoint at {path})")
            return 0
    for path in (args.attn_desc, args.recurrent_desc):
        if not os.path.exists(path):
            print(f"test_verify_rows: skipped (no descriptor at {path})")
            return 0

    devices = [int(d) for d in args.devices.split(",")]
    lib = ctypes.CDLL(os.path.abspath(args.library))
    bind(lib)
    if not hasattr(lib, "engine_verify_rows"):
        print("test_verify_rows: skipped (the library has no engine_verify_rows)")
        return 0
    lib.engine_verify_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_longlong]
    lib.engine_verify_rows.restype = ctypes.c_int
    lib.engine_truncate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.engine_truncate.restype = ctypes.c_int

    failures = 0
    # The synthetic checkpoints cap positions well below their descriptors' declared context,
    # so the runtime context is declared here as the other gates do.
    desc = load_descriptor(args.attn_desc, max_seq_len=args.prompt + args.window + 1)
    vocab = desc["vocab_size"]
    prompt = [(7 * (t + 1)) % vocab for t in range(args.prompt)]
    window = [(29 * (t + 3) + 5) % vocab for t in range(args.window)]

    print(f"=== a window of {len(window)} against serial execution ===")
    batch, _ = create_engine(lib, args.attn_dir, desc, devices)
    serial, _ = create_engine(lib, args.attn_dir, desc, devices)
    prompt_ids = np.ascontiguousarray(prompt, dtype=np.int64)
    for engine in (batch, serial):
        lib.engine_reset(engine)
        logits = np.empty(vocab, dtype=np.float32)
        status = lib.engine_prefill(engine, ptr(prompt_ids), len(prompt), ptr(logits))
        if not check(status == 0, lib.engine_last_error().decode()):
            return 1

    # n = 1 is one execution case on both sides: bitwise, or the API is wrong.
    status, one_row = all_rows(lib, batch, window[:1], vocab)
    if not check(status == 0, lib.engine_last_error().decode()):
        return 1
    status, serial_one = serial_rows(lib, serial, window[:1], vocab)
    if not check(status == 0, lib.engine_last_error().decode()):
        return 1
    if not check(np.array_equal(one_row[0], serial_one[0]),
                 f"a one-token verification is not a decode (max_abs "
                 f"{float(np.abs(one_row[0] - serial_one[0]).max()):.3e})"):
        failures += 1

    # Roll back both to the prompt, then take the real window on each side.
    for engine in (batch, serial):
        status = lib.engine_truncate(engine, len(prompt))
        if not check(status == 0, f"truncate back to the prompt: {lib.engine_last_error().decode()}"):
            failures += 1
    status, rows = all_rows(lib, batch, window, vocab)
    if not check(status == 0, lib.engine_last_error().decode()):
        return 1
    status, reference = serial_rows(lib, serial, window, vocab)
    if not check(status == 0, lib.engine_last_error().decode()):
        return 1
    if not check(rows.shape == reference.shape, "the batch returned a different number of rows"):
        return 1

    print("  row   max_abs      rms        batch argmax  serial argmax  agree")
    agree = 0
    for i in range(len(window)):
        same = int(rows[i].argmax()) == int(reference[i].argmax())
        agree += same
        print(f"  {i:3d}   {float(np.abs(rows[i] - reference[i]).max()):.3e}  "
              f"{flat_rms(rows[i], reference[i]):.3e}  {int(rows[i].argmax()):12d}  "
              f"{int(reference[i].argmax()):12d}  {'yes' if same else 'NO'}")
    if not check(agree == len(window),
                 f"the batch decided differently from serial execution in "
                 f"{len(window) - agree} of {len(window)} rows"):
        failures += 1

    # The rollback has to leave the batch engine exactly where the prompt's state is: decoding
    # the same token afterwards must give the row a fresh prefill-and-decode gives, at the same
    # position.
    status = lib.engine_truncate(batch, len(prompt))
    if not check(status == 0, lib.engine_last_error().decode()):
        failures += 1
    tail_batch = np.empty(vocab, dtype=np.float32)
    tail_serial = np.empty(vocab, dtype=np.float32)
    lib.engine_reset(serial)
    status = lib.engine_prefill(serial, ptr(prompt_ids), len(prompt), ptr(tail_serial))
    if not check(status == 0, lib.engine_last_error().decode()):
        failures += 1
    lib.engine_decode(batch, int(window[0]), ptr(tail_batch))
    lib.engine_decode(serial, int(window[0]), ptr(tail_serial))
    if not check(np.array_equal(tail_batch, tail_serial),
                 f"after the rollback the batch engine is not where a clean prompt is (max_abs "
                 f"{float(np.abs(tail_batch - tail_serial).max()):.3e})"):
        failures += 1

    # The refusals: a short buffer, a window past max_chunk, and a truncation past the sequence.
    short = np.empty(vocab, dtype=np.float32)
    window_ids = np.ascontiguousarray(window, dtype=np.int64)
    status = lib.engine_verify_rows(batch, ptr(window_ids), len(window), ptr(short), short.size)
    if not check(status != 0, "a buffer that cannot hold every row was accepted"):
        failures += 1
    status = lib.engine_verify_rows(batch, ptr(window_ids), int(desc["max_chunk"]) + 1,
                                    ptr(short), short.size)
    if not check(status != 0, "a window past max_chunk was accepted"):
        failures += 1
    status = lib.engine_truncate(batch, lib.engine_seq_len(batch) + 1)
    if not check(status != 0, "a truncation past the sequence was accepted"):
        failures += 1
    lib.engine_destroy(batch)
    lib.engine_destroy(serial)

    print("=== a recurrent model refuses the rollback ===")
    recurrent_desc = load_descriptor(args.recurrent_desc,
                                     max_seq_len=args.prompt + args.window + 1)
    recurrent, _ = create_engine(lib, args.recurrent_dir, recurrent_desc, devices)
    recurrent_vocab = recurrent_desc["vocab_size"]
    lib.engine_reset(recurrent)
    logits = np.empty(recurrent_vocab, dtype=np.float32)
    recurrent_ids = np.ascontiguousarray([3, 4], dtype=np.int64)
    lib.engine_prefill(recurrent, ptr(recurrent_ids), 2, ptr(logits))
    refused = lib.engine_truncate(recurrent, 1)
    print(f"  a GDN model: refused={refused != 0} ({lib.engine_last_error().decode()})")
    if not check(refused != 0, "a recurrent model accepted a truncation"):
        failures += 1
    lib.engine_destroy(recurrent)

    if failures:
        print(f"test_verify_rows: {failures} check(s) failed")
        return 1
    print("test_verify_rows: the batch decides what serial execution decides, the rollback "
          "restores the retained prefix, and a recurrent model is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
