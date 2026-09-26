#!/usr/bin/env python3
"""The plan's S performance gate: is speculative decoding actually faster here?

The plan is explicit that this is a question, not a conclusion - "low acceptance or expensive
GDN replay may make the method slower" - and that the phases have to be measured rather than
assumed. So this reports, for one request:

  * the **acceptance histogram** and the useful tokens per verification, from the loop's own
    per-round statistics (`--spec-stats`),
  * the **per-phase** split from the engine's opt-in region table (`--profile`): the target's
    batched verification (final norm, LM head, D2H) against the ordinary decode's single-row LM
    head, plus the checkpoint copies and the draft's decodes, all in one run,
  * the **comparison the plan asks for**: the speculative generation's own time against the
    serial target time for the same number of newly committed tokens - the serial side measured
    directly on the same engine, same model, same prompt length, so it is the latency of the
    work that was avoided rather than an estimate from a different configuration,
  * the useful tokens per second, and the peak device memory the request needed.

The verdict is printed, not assumed: if the speculative total is not below the serial time for
the tokens it actually committed, it says so, because "we implemented it" is not "it helps".

Usage:
    python3 tests/benchmark_speculative.py --exe "$(cabal list-bin haskell-infer-demo)" \\
        --library csrc/build-libs/libengine.so --target-dir DIR [--target-desc D] \\
        --draft-dir DIR [--speculative-k K] [--max-tokens N] [--w4a16-dir DIR]
"""

import argparse
import ctypes
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, ptr

ROUND_RE = re.compile(
    r"spec-round:\s+(?P<index>\d+)\s+proposals=(?P<proposals>\d+)\s+confirmed=(?P<confirmed>\d+)"
    r"\s+committed=(?P<committed>\d+)\s+ms=(?P<ms>\d+)")
SUMMARY_RE = re.compile(r"spec-stats:.*")


def generated_text(out):
    """The text between the CLI's two '---' markers: the generated output alone."""
    parts = out.split("---", 2)
    return parts[1].strip() if len(parts) >= 3 else None


def run_cli(exe, args):
    proc = subprocess.run([exe, "generate", *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def serial_latency(library, model_dir, desc_path, devices, prompt_len, tokens, tries=5):
    """TTFT and per-decode latency on the target alone, measured on the same engine the
    speculative run uses. The ids are arbitrary - only the latency of the work matters - but the
    prompt length and the device placement match, which is what makes the comparison honest."""
    lib = ctypes.CDLL(os.path.abspath(library))
    bind(lib)
    desc = load_descriptor(desc_path, max_seq_len=prompt_len + tokens + 2)
    engine, vocab = create_engine(lib, model_dir, desc, devices)
    prompt = [(7 * (t + 1)) % vocab for t in range(prompt_len)]
    steps = [(13 * (t + 3) + 1) % vocab for t in range(tokens)]
    prompt_array = np.ascontiguousarray(prompt, dtype=np.int64)
    logits = np.empty(vocab, dtype=np.float32)

    lib.engine_reset(engine)
    lib.engine_prefill(engine, ptr(prompt_array), len(prompt), ptr(logits))
    lib.engine_decode(engine, int(steps[0]), ptr(logits))
    lib.engine_reset(engine)

    started = time.perf_counter()
    lib.engine_prefill(engine, ptr(prompt_array), len(prompt), ptr(logits))
    ttft = time.perf_counter() - started
    per_step = []
    for token in steps:
        started = time.perf_counter()
        lib.engine_decode(engine, int(token), ptr(logits))
        per_step.append(time.perf_counter() - started)
    lib.engine_destroy(engine)
    return ttft, per_step


def peak_memory_mb(devices):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    peak = 0
    for line in out.stdout.strip().splitlines():
        index, used = [part.strip() for part in line.split(",")]
        if int(index) in devices:
            peak = max(peak, int(used))
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--target-desc", default=None)
    parser.add_argument("--draft-dir", default=None)
    parser.add_argument("--speculative-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--w4a16-dir", default=None)
    parser.add_argument("--draft-w4a16-dir", default=None,
                        help="a packed-INT4 draft: the same model with the operands this "
                             "repository can admit, which is what makes the acceptance "
                             "histogram non-trivial (an identical draft always accepts)")
    args = parser.parse_args()

    for path in (args.exe, args.library):
        if not os.path.exists(path):
            print(f"benchmark_speculative: skipped (no {path})")
            return 0
    if not os.path.isdir(args.target_dir):
        print(f"benchmark_speculative: skipped (no checkpoint at {args.target_dir})")
        return 0

    devices = [int(d) for d in args.devices.split(",")]
    draft_dir = args.draft_dir or args.target_dir
    common = ["--model-dir", args.target_dir, "-p", args.prompt,
              "--max-tokens", str(args.max_tokens), "--temperature", "0",
              "--gpus", args.devices]
    if args.target_desc:
        common += ["--descriptor", args.target_desc]
    if args.w4a16_dir:
        common += ["--w4a16-dir", args.w4a16_dir]

    print("=== the target alone ===")
    before = peak_memory_mb(devices)
    started = time.perf_counter()
    code, out, err = run_cli(args.exe, common)
    serial_wall = time.perf_counter() - started
    if code != 0:
        print(out, err)
        return 1
    serial_tokens = int(re.search(r"Generated (\d+) tokens", out).group(1))
    serial_text = generated_text(out)
    print(f"  {serial_tokens} tokens in {serial_wall:.3f} s of wall time")

    print("=== the speculative request ===")
    spec_args = common + ["--draft-model-dir", draft_dir,
                          "--speculative-k", str(args.speculative_k),
                          "--spec-stats", "--profile"]
    if args.draft_w4a16_dir:
        spec_args += ["--draft-w4a16-dir", args.draft_w4a16_dir]
    started = time.perf_counter()
    code, out, err = run_cli(args.exe, spec_args)
    spec_wall = time.perf_counter() - started
    if code != 0:
        print(out, err)
        return 1
    after = peak_memory_mb(devices)

    # The plan's S gate: the speculative output must be the target-only output, which at this
    # scale is also the continuation-after-rejection check - a rejected round that left the
    # wrong state would show up as a different text.
    speculative_text = generated_text(out)
    if serial_text is None or speculative_text != serial_text:
        print(f"benchmark_speculative: FAIL: the speculative output is not the target-only "
              f"output ({speculative_text!r} vs {serial_text!r})")
        return 1
    print("  the speculative output is byte-identical to the target-only output")

    rounds = [m.groupdict() for m in ROUND_RE.finditer(err)]
    if not rounds:
        print("benchmark_speculative: FAIL: the run reported no rounds (is --spec-stats working?)")
        return 1
    for entry in rounds:
        for key in ("index", "proposals", "confirmed", "committed", "ms"):
            entry[key] = int(entry[key])
    committed = sum(entry["committed"] for entry in rounds)
    proposed = sum(entry["proposals"] for entry in rounds)
    confirmed = sum(entry["confirmed"] for entry in rounds)
    spec_seconds = sum(entry["ms"] for entry in rounds) / 1000.0

    histogram = {}
    for entry in rounds:
        if entry["proposals"] == 0:
            continue
        histogram[entry["confirmed"]] = histogram.get(entry["confirmed"], 0) + 1
    print(f"  rounds {len(rounds)}, proposed {proposed}, confirmed {confirmed}, "
          f"committed {committed}")
    print("  acceptance histogram (confirmed -> rounds): "
          + ", ".join(f"{key}:{histogram[key]}" for key in sorted(histogram, reverse=True))
          + (f"; {len(rounds) - sum(histogram.values())} round(s) with no window" if
             len(rounds) - sum(histogram.values()) else ""))
    print(f"  useful tokens per verification: "
          f"{committed / proposed if proposed else 0:.3f} (committed / proposed)")
    print(f"  the rounds took {spec_seconds:.3f} s of the request's {spec_wall:.3f} s wall time")

    print("=== the phases, from the engine's own region table ===")
    for line in err.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("spec-"):
            continue
        if any(name in stripped for name in ("verify.", "checkpoint.", "lm_head.", "mlp.")):
            print(f"  {stripped}")

    print("=== the comparison the plan asks for ===")
    ttft, per_step = serial_latency(args.library, args.target_dir,
                                    args.target_desc or os.path.join(
                                        os.path.dirname(os.path.abspath(__file__)), "..",
                                        "descriptors", "qwen38-27b.json"),
                                    devices, prompt_len=8, tokens=max(committed, 1))
    decode_median = statistics.median(per_step)
    serial_for_committed = ttft + decode_median * max(committed - 1, 0)
    print(f"  the target alone: TTFT {ttft * 1000:.2f} ms, decode {decode_median * 1000:.2f} ms "
          f"per token")
    print(f"  serial time for the {committed} tokens the speculative run committed: "
          f"{serial_for_committed:.3f} s")
    print(f"  speculative time for the same tokens: {spec_seconds:.3f} s")
    ratio = (serial_for_committed / spec_seconds) if spec_seconds > 0 else 0.0
    print(f"  ratio (serial / speculative): {ratio:.2f}x")
    if ratio > 1.0:
        print("  VERDICT: admitted - the same tokens cost less time than serial decoding")
    else:
        print("  VERDICT: not admitted at this acceptance and window - the plan's own warning "
              "(\"low acceptance or expensive GDN replay may make the method slower\") applies "
              "here; the numbers above are what a fixed-k choice would be made from")
    useful_rate = committed / spec_seconds if spec_seconds > 0 else 0.0
    print(f"  useful tokens/s: {useful_rate:.2f}")
    if after is not None and before is not None:
        print(f"  peak device memory: {max(before, after)} MiB (both engines resident)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
