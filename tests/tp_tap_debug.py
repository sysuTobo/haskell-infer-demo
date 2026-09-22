"""Localize where the tp=1 and tp=2 arms diverge, layer by layer.

Runs one engine per arm with INFER_TAP_LAYERS/INFER_TAP_DIR set, prefills the
same prompt (default: 129 tokens, i.e. the 128-token chunk plus the boundary
step), then compares the layer taps (float32 [tokens, hidden]) and the final
logits. A structurally wrong placement shows a jump at the first affected layer;
benign BF16 path differences grow smoothly from bitwise-identical GDN internals
(layer taps use the producing stream, so no tap races).

    python tests/tp_tap_debug.py --library csrc/build-libs/libengine.so \
        --model-dir "$MODEL_DIR" [--desc descriptors/qwen38-27b.json] [--tokens 129]
"""

import argparse
import ctypes
import os
import sys

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, ptr

PROMPT_MAIN = [760, 6511, 314, 9564, 369]
TAP_LAYERS = [0, 1, 3, 4, 5, 63]


def run(lib, model_dir, desc, devices, prompt, tap_dir):
    os.environ["INFER_TAP_LAYERS"] = ",".join(str(x) for x in TAP_LAYERS)
    os.environ["INFER_TAP_DIR"] = tap_dir
    os.makedirs(tap_dir, exist_ok=True)
    engine, vocab = create_engine(lib, model_dir, desc, devices)
    try:
        logits = np.empty(vocab, dtype=np.float32)
        ids = np.ascontiguousarray(prompt, dtype=np.int64)
        rc = lib.engine_prefill(engine, ptr(ids), len(ids), ptr(logits))
        assert rc == 0, lib.engine_last_error().decode()
        return logits.copy()
    finally:
        lib.engine_destroy(engine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", default="descriptors/qwen38-27b.json")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--tokens", type=int, default=129,
                        help="prompt length (the five-token prompt repeated)")
    parser.add_argument("--tap-a", default="/tmp/tp-tap-a")
    parser.add_argument("--tap-b", default="/tmp/tp-tap-b")
    args = parser.parse_args()

    lib = bind(ctypes.CDLL(args.library))
    devices = [int(x) for x in args.devices.split(",")]
    prompt = np.resize(PROMPT_MAIN, args.tokens).astype(int).tolist()
    base = load_descriptor(args.desc, max_seq_len=256)
    tp2 = dict(base)
    tp2["tp_size"] = 2

    logits_a = run(lib, args.model_dir, base, devices, prompt, args.tap_a)
    print("tp1 done", flush=True)
    logits_b = run(lib, args.model_dir, tp2, devices, prompt, args.tap_b)
    print("tp2 done", flush=True)

    print(f"logits: rms={float(np.sqrt(np.mean((logits_a - logits_b) ** 2))):.4g} "
          f"top1={int(logits_a.argmax())}/{int(logits_b.argmax())}")
    for layer in TAP_LAYERS:
        # Chunked prefill writes one tap file per chunk (e.g. tok128 and tok1).
        for tokens in sorted({min(args.tokens, 128), 1 if args.tokens > 128 else args.tokens}):
            name = f"layer_{layer:02d}_seq0_tok{tokens}.f32"
            path_a = os.path.join(args.tap_a, name)
            path_b = os.path.join(args.tap_b, name)
            if not (os.path.exists(path_a) and os.path.exists(path_b)):
                continue
            a = np.fromfile(path_a, dtype=np.float32)
            b = np.fromfile(path_b, dtype=np.float32)
            if a.size != b.size:
                print(f"{name}: size mismatch {a.size} vs {b.size}")
                continue
            print(f"{name}: max_abs={np.abs(a - b).max():.4g} "
                  f"rms={float(np.sqrt(np.mean((a - b) ** 2))):.4g} "
                  f"scale={float(a.std()):.3g}")


if __name__ == "__main__":
    main()
