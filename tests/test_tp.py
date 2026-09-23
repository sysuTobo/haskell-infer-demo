"""Tensor-parallel equivalence: one model, two placements, same answers.

Run A is the pipelined layer split (tp_size 1, each device owns half the
layers); run B is the replicated tensor-parallel policy (tp_size 2, every rank
holds every layer with its shard of the weights). Both must produce the same
greedy tokens; the logits differ only by the split-K GEMM sums and the BF16
all-reduce, so a small RMS gate applies (the engine-vs-PyTorch gate for this
family is 0.1 with a logit std near 2, so 0.05 stays well inside it).

The second case crosses the engine's 128-token chunk boundary with a natural
prompt. Note that a *degenerate* prompt (e.g. a five-token sequence repeated to
129) is not a meaningful equivalence target: the model amplifies BF16-level path
differences to O(1) logits there while the greedy tokens still match -- the same
regime tests/test_engine.py documents for different prefill chunkings. The
natural prompts used here stay inside the gate (128 tokens: 0.031, 129 tokens
with the boundary: 0.029, measured on Qwen3.8-27B, 2x A40).

    python tests/test_tp.py --library csrc/build-libs/libengine.so \
        --model-dir "$MODEL_DIR" --devices 0,1

With `--ep N` the second arm is expert parallelism instead of tensor parallelism
(the layer split stays the baseline), which is how the MoE families are checked:
`--desc descriptors/qwen3-30b-a3b.json --ep 2`.
"""

import argparse
import ctypes
import sys
import time

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, ptr
from test_longseq import PROMPT as LONG_PASSAGE

# "The capital of France is" (ids verified in the worklog).
PROMPT_MAIN = [760, 6511, 314, 9564, 369]

RMS_GATE = 0.05
STEPS = 10


def run_case(lib, model_dir, descriptor, devices, prompt, steps):
    """Greedy-decode `steps` tokens and return the logits seen at each step."""
    engine, vocab = create_engine(lib, model_dir, descriptor, devices)
    try:
        logits = np.empty(vocab, dtype=np.float32)
        ids = np.ascontiguousarray(prompt, dtype=np.int64)
        rc = lib.engine_prefill(engine, ptr(ids), len(ids), ptr(logits))
        assert rc == 0, lib.engine_last_error().decode()
        seen, greedy = [], []
        for step in range(steps):
            seen.append(logits.copy())
            token = int(logits.argmax())
            greedy.append(token)
            if step + 1 < steps:
                rc = lib.engine_decode(engine, token, ptr(logits))
                assert rc == 0, lib.engine_last_error().decode()
        assert lib.engine_seq_len(engine) == len(prompt) + steps - 1
        return greedy, seen
    finally:
        lib.engine_destroy(engine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", default="descriptors/qwen38-27b.json")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--rms-gate", type=float, default=RMS_GATE)
    parser.add_argument("--ep", type=int, default=1,
                        help="expert-parallel ranks for the second arm "
                             "(> 1 compares the layer split against EP on MoE models)")
    args = parser.parse_args()

    lib = bind(ctypes.CDLL(args.library))
    devices = [int(x) for x in args.devices.split(",")]
    if args.ep > 1:
        if len(devices) != args.ep:
            print(f"expert parallel needs exactly --ep {args.ep} devices", file=sys.stderr)
            return 1
    elif len(devices) != 2:
        print("this test compares one two-device placement against another", file=sys.stderr)
        return 1
    base = load_descriptor(args.desc, max_seq_len=256)
    if base.get("tp_size", 1) != 1 or base.get("ep_size", 1) != 1:
        print("the descriptor snapshot must be single-rank; it is the first arm",
              file=sys.stderr)
        return 1

    other = dict(base)
    if args.ep > 1:
        other["ep_size"] = args.ep
        label = f"ep={args.ep}"
    else:
        other["tp_size"] = 2
        label = "tp=2"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    long_prompt = tokenizer(LONG_PASSAGE, return_tensors="np").input_ids[0]
    long_prompt = long_prompt[:129].astype(int).tolist()

    failures = []
    cases = [("main", PROMPT_MAIN), ("boundary129", long_prompt)]
    for name, prompt in cases:
        start = time.monotonic()
        tokens_a, logits_a = run_case(lib, args.model_dir, base, devices, prompt, args.steps)
        print(f"[{name}] layer split: {tokens_a}  ({time.monotonic() - start:.0f}s)", flush=True)
        start = time.monotonic()
        tokens_b, logits_b = run_case(lib, args.model_dir, other, devices, prompt, args.steps)
        print(f"[{name}] replicated ({label}): {tokens_b}  ({time.monotonic() - start:.0f}s)",
              flush=True)

        if tokens_a != tokens_b:
            failures.append({"case": name, "greedy": [tokens_a, tokens_b]})
        for step, (a, b) in enumerate(zip(logits_a, logits_b)):
            rms = float(np.sqrt(np.mean((a - b) ** 2)))
            std = float(a.std())
            print(f"[{name}] step {step}: rms={rms:.4g} (logit std={std:.3g}, "
                  f"top1 {int(a.argmax())}/{int(b.argmax())})", flush=True)
            if not np.isfinite(rms) or rms > args.rms_gate:
                failures.append({"case": name, "step": step, "rms": rms})

    assert not failures, failures
    print(f"placement equivalence passed ({label} vs layer split: rms <= {args.rms_gate}, "
          "greedy tokens identical)")


if __name__ == "__main__":
    main()
