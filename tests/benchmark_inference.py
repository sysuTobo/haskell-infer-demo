#!/usr/bin/env python3
"""F0: a costed baseline before choosing any fusion (plan "Inference optimization track").

docs/plan-numeric-contract.md's F0 asks for per-region CUDA time, launch count, host
synchronization and memory traffic for decode M=1 and representative prefill M=2/64/128,
"within each descriptor's limits", including layer placement and full request wall time, with
timing run "after warm-up, and synchronize only at measurement boundaries".

What this runner produces:

* **full request wall time** for prefill at several M and for single-token decode, as the
  min/median/max over repetitions plus a standard deviation - the plan is explicit that a
  single best timing is not the measurement ("record repeated warm runs and dispersion, not
  only the best timing");
* **per-region CUDA time and launch count** from the opt-in `profile_scope_*` facility in
  `csrc/profile.cu`, which records one event pair per region invocation and synchronizes once
  per report (the measurement boundary);
* **provenance**: the descriptor the engine parsed, the device list, the GPU model/clocks/
  temperature at measurement time and the execution manifest's identities where available,
  because the plan's own checkpoint list says to hold those fixed across baseline comparisons.

What it does **not** produce, and says so rather than estimating: per-region **memory
traffic** and per-region **host-synchronization counts**. Both need a profiler, not events.

The measured call already synchronizes (each engine call downloads its logits), so a wall
clock around it is a complete request time and no extra synchronization is added here.

Usage (on a device, with the built library):

    python3 tests/benchmark_inference.py --library csrc/build-libs/libengine.so \
        --model-dir "$MODEL_DIR" --desc "$DESC" --devices 0,1 [--json out.json]
"""

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_bindings import bind, create_engine, load_descriptor, ptr  # noqa: E402


def bind_profile(lib):
    lib.profile_set_enabled.argtypes = [ctypes.c_int]
    lib.profile_set_enabled.restype = None
    lib.profile_enabled.argtypes = []
    lib.profile_enabled.restype = ctypes.c_int
    lib.profile_reset.argtypes = []
    lib.profile_reset.restype = None
    lib.profile_scope_count.argtypes = []
    lib.profile_scope_count.restype = ctypes.c_int
    lib.profile_scope_overflowed.argtypes = []
    lib.profile_scope_overflowed.restype = ctypes.c_int
    lib.profile_report.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.profile_report.restype = ctypes.c_int
    lib.profile_last_error.argtypes = []
    lib.profile_last_error.restype = ctypes.c_char_p


def gpu_provenance():
    """The devices' model, clocks and thermal state, which the plan lists as fixed inputs."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,"
             "temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return {"error": str(err)}
    return {"devices": [line.strip() for line in out.splitlines()]}


def manifest_identities(lib, engine):
    """The execution manifest's identities, when the library exposes them."""
    if not hasattr(lib, "engine_manifest"):
        return None
    buf = ctypes.create_string_buffer(256 * 1024)
    if lib.engine_manifest(engine, buf, len(buf)) != 0:
        return None
    try:
        doc = json.loads(buf.value.decode())
    except (ValueError, UnicodeDecodeError):
        return None
    return {"identities": doc.get("identities"), "build": doc.get("provenance", {}).get("build")}


def summarise(samples_ms):
    """Dispersion, not a single best number (the plan's own rule)."""
    if not samples_ms:
        return {}
    return {
        "repeats": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "max_ms": max(samples_ms),
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def time_call(repeats, warmup, fn):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def parse_report(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("region") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            continue
        rows.append({"region": parts[0], "count": int(parts[1]), "total_ms": float(parts[2]),
                     "mean_ms": float(parts[3]), "max_ms": float(parts[4]), "depth": int(parts[5])})
    return rows


def profile_one_call(lib, call, label, baseline_ms=None):
    """Reset, run one call with recording on, and report. One synchronization per call.

    The recording itself is not free: every scope records two events on the stream, so the
    instrumented call carries host-side launch overhead the uninstrumented one does not, and
    the device can starve behind it. That is why the same call is timed with the wall clock
    here too: `overhead_ms` is the price of the measurement, and a reader who wants the
    uninstrumented cost should use the wall-clock table above rather than this one's sum.
    """
    lib.profile_set_enabled(1)
    lib.profile_reset()
    start = time.perf_counter()
    call()
    wall_ms = (time.perf_counter() - start) * 1000.0
    buf = ctypes.create_string_buffer(256 * 1024)
    needed = lib.profile_report(buf, len(buf))
    rows = parse_report(buf.value.decode()) if needed > 0 else []
    result = {
        "scopes": lib.profile_scope_count(),
        "overflowed": bool(lib.profile_scope_overflowed()),
        "regions": rows,
        "wall_ms": wall_ms,
    }
    if baseline_ms:
        result["baseline_ms"] = baseline_ms
        result["overhead_ms"] = wall_ms - baseline_ms
    lib.profile_set_enabled(0)
    if not rows:
        result["error"] = lib.profile_last_error().decode()
    print(f"\n--- per-region CUDA time: {label} "
          f"({result['scopes']} scopes recorded"
          f"{', OVERFLOWED' if result['overflowed'] else ''})"
          + (f"; instrumented wall {wall_ms:.2f} ms vs uninstrumented "
             f"{baseline_ms:.2f} ms ({wall_ms - baseline_ms:+.2f} ms of recording)"
             if baseline_ms else ""))
    print(f"{'region':<24}{'count':>7}{'total_ms':>11}{'mean_ms':>10}{'max_ms':>10}{'depth':>6}")
    for row in rows:
        print(f"{row['region']:<24}{row['count']:>7}{row['total_ms']:>11.3f}"
              f"{row['mean_ms']:>10.4f}{row['max_ms']:>10.4f}{row['depth']:>6}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prefill-lengths", default="2,64,128")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", default=None, help="also write the measurements here")
    args = parser.parse_args()

    devices = [int(d) for d in args.devices.split(",") if d.strip()]
    prefill_lengths = [int(m) for m in args.prefill_lengths.split(",") if m.strip()]

    lib = ctypes.CDLL(args.library)
    bind(lib)
    if not hasattr(lib, "profile_set_enabled"):
        print("benchmark_inference: skipped (the library has no profile_scope_* entry points)")
        return 0
    bind_profile(lib)

    desc = load_descriptor(args.desc)
    engine, vocab = create_engine(lib, args.model_dir, desc, devices)

    report = {
        "provenance": {
            "model_dir": args.model_dir,
            "descriptor": args.desc,
            "devices": devices,
            "vocab_size": int(vocab),
            "max_seq_len": int(desc.get("max_seq_len", 0)),
            "num_layers": int(desc.get("num_layers", 0)),
            "gpu": gpu_provenance(),
            "manifest": manifest_identities(lib, engine),
            "method": {
                "repeats": args.repeats,
                "warmup": args.warmup,
                "prefill_lengths": prefill_lengths,
                "decode_tokens": args.decode_tokens,
                "clock": "perf_counter around one engine call, which downloads its own logits",
            },
        },
        "prefill": {},
        "decode": {},
    }

    logits = np.empty((vocab,), dtype=np.float32)
    prompt_len = min(8, max(prefill_lengths))

    try:
        for length in prefill_lengths:
            if length > desc.get("max_seq_len", length):
                report["prefill"][str(length)] = {"skipped": "beyond max_seq_len"}
                continue

            def one_prefill(length=length):
                lib.engine_reset(engine)
                ids = np.arange(1, length + 1, dtype=np.int64)
                status = lib.engine_prefill(engine, ptr(ids), length, ptr(logits))
                assert status == 0, lib.engine_last_error().decode()

            report["prefill"][str(length)] = summarise(
                time_call(args.repeats, args.warmup, one_prefill))
            print(f"prefill M={length:<4} {report['prefill'][str(length)]}")

        # Decode: one prefill, then one call per token. Each engine call returns logits, so its
        # wall time is a complete single-token step (forward + download).
        prompt = np.arange(1, prompt_len + 1, dtype=np.int64)

        def decode_steps(count):
            lib.engine_reset(engine)
            assert lib.engine_prefill(engine, ptr(prompt), prompt_len, ptr(logits)) == 0, \
                lib.engine_last_error().decode()
            samples = []
            for step in range(count):
                start = time.perf_counter()
                status = lib.engine_decode(engine, int(prompt[step % prompt_len]), ptr(logits))
                samples.append((time.perf_counter() - start) * 1000.0)
                assert status == 0, lib.engine_last_error().decode()
            return samples

        for _ in range(args.warmup):
            decode_steps(prompt_len)
        per_token = []
        for _ in range(args.repeats):
            per_token.extend(decode_steps(args.decode_tokens))
        step = summarise(per_token)
        step["tokens_per_s"] = 1000.0 / step["median_ms"] if step.get("median_ms") else 0.0
        report["decode"]["m1"] = step
        print(f"decode  M=1  {step}")

        # Per-region numbers, prefill and decode measured apart: a decode step reuses the
        # prefill's cache, so the cache is built with recording off and only the step itself
        # is recorded. Aggregating the two would average two different shapes.
        report["profile"] = {}
        profile_prefill_len = min(max(prefill_lengths), desc.get("max_seq_len", 0))
        profile_ids = np.arange(1, profile_prefill_len + 1, dtype=np.int64)
        report["profile"]["prefill"] = profile_one_call(
            lib, lambda: (lib.engine_reset(engine),
                          lib.engine_prefill(engine, ptr(profile_ids), profile_prefill_len,
                                             ptr(logits))),
            f"prefill M={profile_prefill_len}", report["prefill"][str(profile_prefill_len)].get("median_ms"))
        lib.engine_reset(engine)
        assert lib.engine_prefill(engine, ptr(prompt), prompt_len, ptr(logits)) == 0, \
            lib.engine_last_error().decode()
        report["profile"]["decode"] = profile_one_call(
            lib, lambda: lib.engine_decode(engine, int(prompt[0]), ptr(logits)),
            "one decode step (M=1)", report["decode"]["m1"].get("median_ms"))
    finally:
        lib.engine_destroy(engine)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
