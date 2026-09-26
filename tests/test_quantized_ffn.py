#!/usr/bin/env python3
"""Q2's model-quality gate: the dense FFN's weight-only INT4 decode path against BF16.

The plan's Q2 asks for the packed kernel to be admitted *and* for the model-quality gates -
logits RMS, top-1 agreement and a held-out NLL - that compare a quantized model against the
BF16 baseline. This runs both on the synthetic dense checkpoint, where they are cheap:

  * the BF16 baseline and the INT4 run share one engine: the BF16 weights are loaded first and
    `engine_load_quantized_ffn` adds the packed operands beside them, so prefill (M > 1) keeps
    reading BF16 and only decode (M = 1) reads INT4 - the split the Q2 measurement supports.
  * both runs prefill the same prompt and then decode the same token ids, so every comparison is
    between two runs in the same state rather than a cascade of greedy choices.
  * the prefill logits are required to be **bitwise equal**, which is what shows the BF16 path
    was not disturbed by the load; the decode logits are then where the quantization shows up.
  * the execution manifest is compared too: a packed operand is a *numerical policy* change, so
    `numerical_policy_id` must move while `semantic_id` and `deployment_id` must not.

Refusals are checked through the same entry point: a corrupted artifact (a byte of its payload
flipped) and a tampered format block must both make the load fail rather than reach the kernel.

Usage:
    python3 tests/test_quantized_ffn.py --library csrc/build-libs/libengine.so \\
        --model-dir <synth dense checkpoint> --desc descriptors/qwen3-dense-synth.json \\
        --reference csrc/build-libs/test_quantization_format
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from engine_bindings import bind, create_engine, load_descriptor, manifest, ptr

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "..", "scripts", "quantize_weights.py")


def check(condition, message):
    if condition:
        return True
    print(f"test_quantized_ffn: FAIL: {message}")
    return False


def run(args):
    return subprocess.run(args, capture_output=True, text=True)


def log_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def warm_up(lib, engine, vocab, prompt):
    """One throwaway prefill and decode: the first call on a fresh engine pays context and
    allocation setup, which is not TTFT and would otherwise land in whichever run went first."""
    lib.engine_reset(engine)
    logits = np.empty(vocab, dtype=np.float32)
    lib.engine_prefill(engine, ptr(np.ascontiguousarray(prompt, dtype=np.int64)), len(prompt),
                       ptr(logits))
    lib.engine_decode(engine, int(prompt[0]), ptr(logits))
    lib.engine_reset(engine)


def decode_sequence(lib, engine, vocab, prompt, steps):
    """Reset, prefill the prompt, then decode `steps` tokens.

    Returns (prefill logits, [steps], prefill seconds, [step seconds]) - the times are taken
    around the calls so the report's TTFT and decode latency come from the same run as the
    quality numbers rather than a second, differently-warmed one."""
    lib.engine_reset(engine)
    prefill_logits = np.empty(vocab, dtype=np.float32)
    started = time.perf_counter()
    status = lib.engine_prefill(engine, ptr(np.ascontiguousarray(prompt, dtype=np.int64)),
                               len(prompt), ptr(prefill_logits))
    prefill_seconds = time.perf_counter() - started
    if not check(status == 0, lib.engine_last_error().decode()):
        return None, None, 0.0, []
    out = []
    step_seconds = []
    for token in steps:
        logits = np.empty(vocab, dtype=np.float32)
        started = time.perf_counter()
        status = lib.engine_decode(engine, int(token), ptr(logits))
        step_seconds.append(time.perf_counter() - started)
        if not check(status == 0, lib.engine_last_error().decode()):
            return None, None, 0.0, []
        out.append(logits)
    return prefill_logits, out, prefill_seconds, step_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", required=True)
    parser.add_argument("--reference", required=True,
                        help="the Q0 reference binary the converter quantizes with")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--rms-tolerance", type=float, default=0.05,
                        help="the per-model quality budget for the decode logits' rms; the plan "
                             "requires the budget to be set per model before admission, so a "
                             "real-model run passes its own")
    parser.add_argument("--nll-tolerance", type=float, default=0.05,
                        help="the per-model budget for the held-out NLL's movement, in nats")
    parser.add_argument("--prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--quantized-dir", default=None,
                        help="convert here and reuse it if it already holds a manifest; also the "
                             "place the report's persistent-byte count is measured from")
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the temporary directory (implies the report can be "
                             "re-run against the same artifacts)")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.desc):
        args.desc = os.path.join(repo, args.desc)
    if not os.path.isdir(args.model_dir):
        print(f"test_quantized_ffn: skipped (no checkpoint at {args.model_dir})")
        return 0
    if not os.path.exists(args.reference):
        print(f"test_quantized_ffn: skipped (no reference binary at {args.reference})")
        return 0

    if args.quantized_dir is not None:
        workdir = args.quantized_dir
        out_dir = os.path.join(workdir, "quantized")
        os.makedirs(out_dir, exist_ok=True)
        temporary = False
    else:
        workdir = tempfile.mkdtemp(prefix="quantized-ffn-")
        out_dir = os.path.join(workdir, "quantized")
        temporary = not args.keep
    failures = 0

    manifest_path = os.path.join(out_dir, "weights.manifest.json")
    if os.path.exists(manifest_path):
        print(f"  reusing the converted sidecar at {out_dir}")
    else:
        convert_started = time.time()
        conversion = run([sys.executable, CONVERTER, "--model-dir", args.model_dir,
                          "--desc", args.desc, "--out-dir", out_dir,
                          "--reference", args.reference])
        if conversion.returncode != 0:
            print(conversion.stdout, conversion.stderr)
            print("test_quantized_ffn: FAIL: the conversion failed")
            return 1
        print(f"  converted in {time.time() - convert_started:.1f} s")
        print(conversion.stdout.strip())

    descriptor = load_descriptor(args.desc, max_seq_len=args.prompt + args.steps + 1)
    devices = [int(d) for d in args.devices.split(",")]
    vocab = descriptor["vocab_size"]
    lib = ctypes.CDLL(os.path.abspath(args.library))
    bind(lib)

    # A fixed token stream: both runs prefill it and then decode the same ids, so a difference in
    # one step cannot be caused by a different choice in the previous one.
    prompt = [(17 * (t + 1)) % vocab for t in range(args.prompt)]
    steps = [(29 * (t + 3) + 5) % vocab for t in range(args.steps)]

    engine, engine_vocab = create_engine(lib, args.model_dir, descriptor, devices)
    if not check(engine_vocab == vocab, f"the engine reports vocab {engine_vocab}, expected {vocab}"):
        return 1

    warm_up(lib, engine, vocab, prompt)
    prefill_bf16, logits_bf16, prefill_bf16_s, decode_bf16_s = decode_sequence(
        lib, engine, vocab, prompt, steps)
    if prefill_bf16 is None:
        return 1
    manifest_bf16 = json.loads(manifest(lib, engine))
    print(f"  BF16: weight_quantization={manifest_bf16['numerical_policy']['fields']['weight_quantization']}")

    load_started = time.perf_counter()
    loaded = lib.engine_load_quantized_ffn(engine, out_dir.encode())
    load_seconds = time.perf_counter() - load_started
    if not check(loaded == 0, f"engine_load_quantized_ffn: {lib.engine_last_error().decode()}"):
        lib.engine_destroy(engine)
        return 1

    warm_up(lib, engine, vocab, prompt)
    prefill_int4, logits_int4, prefill_int4_s, decode_int4_s = decode_sequence(
        lib, engine, vocab, prompt, steps)
    if prefill_int4 is None:
        return 1
    manifest_int4 = json.loads(manifest(lib, engine))

    # Prefill is M > 1, so it still reads the BF16 weights: bitwise equal, not "close".
    if not check(np.array_equal(prefill_bf16, prefill_int4),
                 f"prefill moved after the INT4 load (max_abs "
                 f"{float(np.abs(prefill_bf16 - prefill_int4).max()):.3e}); the BF16 path was "
                 f"disturbed"):
        failures += 1

    rms = [float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))
           for a, b in zip(logits_int4, logits_bf16)]
    top1 = [int(a.argmax()) == int(b.argmax()) for a, b in zip(logits_int4, logits_bf16)]
    # A greedy decision is only a decision if it has a margin. On a fixture this small the
    # distribution is nearly flat (the held-out NLL below is close to ln(vocab)), so a top-1
    # flip can be a tie-break rather than a quality regression; the test therefore requires
    # every flip to be one the perturbation could have caused - the BF16 top-1 margin must not
    # exceed the step's own max_abs logit difference. A flip outside that is a real regression,
    # and a sharp model (the deployment target) has margins far above the error, so the
    # criterion costs nothing where it matters.
    margin = [float(np.sort(b)[-1] - np.sort(b)[-2]) for b in logits_bf16]
    max_abs = [float(np.abs(a - b).max()) for a, b in zip(logits_int4, logits_bf16)]
    unexplained = [i for i in range(len(top1)) if not top1[i] and margin[i] > max_abs[i]]
    # The held-out NLL: how much probability each variant put on the token that actually
    # followed, which is a quality number a top-1 count cannot see.
    targets = steps[1:] + [steps[-1]]
    nll_bf16 = float(np.mean([-log_softmax(b)[t] for b, t in zip(logits_bf16, targets)]))
    nll_int4 = float(np.mean([-log_softmax(b)[t] for b, t in zip(logits_int4, targets)]))
    worst = max(rms)
    print("  step   rms        top-1  margin     max_abs")
    for i, (r, same) in enumerate(zip(rms, top1)):
        print(f"  {i:4d}   {r:.3e}  {'yes' if same else 'NO ':>5}  {margin[i]:.3e}  "
              f"{max_abs[i]:.3e}")
    print(f"  worst rms {worst:.5f} over {len(rms)} decode steps, "
          f"top-1 agreement {sum(top1)}/{len(top1)} "
          f"({len(top1) - sum(top1)} tie-break(s) within the perturbation, {len(unexplained)} "
          f"unexplained)")
    print(f"  held-out NLL: bf16 {nll_bf16:.5f}, int4 {nll_int4:.5f}, "
          f"delta {nll_int4 - nll_bf16:+.5f} nats")
    print(f"  quality budget: rms <= {args.rms_tolerance}, |NLL delta| <= {args.nll_tolerance} "
          f"nats (per model, set before admission; the plan does not require token identity)")

    if not check(not unexplained,
                 f"the INT4 decode reversed top-1 at step(s) {unexplained} by more than the "
                 f"perturbation"):
        failures += 1
    if not check(worst <= args.rms_tolerance,
                 f"the INT4 decode moved the logits by rms {worst:.5f} > {args.rms_tolerance}"):
        failures += 1
    if not check(abs(nll_int4 - nll_bf16) <= args.nll_tolerance,
                 f"the held-out NLL moved by {nll_int4 - nll_bf16:+.5f} nats > "
                 f"{args.nll_tolerance}"):
        failures += 1

    # The identity discipline: a packed operand is a numerical-policy change and nothing else.
    fields_bf16 = manifest_bf16["numerical_policy"]["fields"]
    fields_int4 = manifest_int4["numerical_policy"]["fields"]
    if not check(fields_bf16["weight_quantization"] == "none" and
                 fields_int4["weight_quantization"] ==
                 "int4_symmetric_group_bf16_scale_decode_only",
                 f"weight_quantization is {fields_bf16['weight_quantization']} then "
                 f"{fields_int4['weight_quantization']}"):
        failures += 1
    if not check(manifest_bf16["numerical_policy"]["numerical_policy_id"] !=
                 manifest_int4["numerical_policy"]["numerical_policy_id"],
                 "the INT4 load did not move numerical_policy_id"):
        failures += 1
    for block, key in (("semantic", "semantic_id"), ("deployment", "deployment_id")):
        if not check(manifest_bf16[block][key] == manifest_int4[block][key],
                     f"the INT4 load moved {key}"):
            failures += 1

    # The cost side, from the same run: what the operands add to persistent memory, how long the
    # load takes, and the TTFT/decode latency of both paths. The pair duplicates its members on
    # disk, so the engine's own residency is counted from what it actually loads (the pair plus
    # every mlpDown), not from the sidecar's total.
    sidecar = json.load(open(manifest_path))
    engine_bytes = sum(pair["packed"]["bytes"] + pair["scales"]["bytes"]
                       for pair in sidecar["pairs"])
    engine_bytes += sum(entry["packed"]["bytes"] + entry["scales"]["bytes"]
                        for entry in sidecar["entries"] if entry["role"] == "mlpDown")
    on_disk = 0
    for root, _dirs, names in os.walk(out_dir):
        for name in names:
            on_disk += os.path.getsize(os.path.join(root, name))
    mib = 1024.0 * 1024.0

    def rates(seconds):
        if not seconds:
            return (0.0, 0.0)
        return (min(seconds), 1.0 / (sum(seconds) / len(seconds)))

    bf16_min, bf16_rate = rates(decode_bf16_s)
    int4_min, int4_rate = rates(decode_int4_s)
    print(f"  performance (same engine, same run, warm): sidecar on disk "
          f"{on_disk / mib:.1f} MiB, loaded into the engine {engine_bytes / mib:.1f} MiB, "
          f"load {load_seconds:.2f} s")
    print(f"    TTFT (prefill {len(prompt)} tokens): bf16 {prefill_bf16_s * 1000:.3f} ms | "
          f"int4 {prefill_int4_s * 1000:.3f} ms")
    print(f"    decode per step: bf16 {bf16_min * 1000:.3f} ms min ({bf16_rate:.1f} tok/s) | "
          f"int4 {int4_min * 1000:.3f} ms min ({int4_rate:.1f} tok/s) | "
          f"speedup {bf16_min / int4_min if int4_min else 0:.2f}x")

    lib.engine_destroy(engine)

    # The two postures are mutually exclusive in both directions: a store's publication writes
    # the BF16 weights and would leave the packed operands describing the old ones, and an INT4
    # engine has no way to publish into them. Either order must be refused rather than run.
    # `allocate_training_state = 0`: this is about the store's *existence*, and allocating
    # masters, gradients and optimizer moments for a real model would need several times its
    # BF16 size (a 4B's is 60+ GiB) to test a refusal that happens before any allocation.
    class AttachOptions(ctypes.Structure):
        _fields_ = [("allocate_training_state", ctypes.c_int),
                    ("frozen_roles", ctypes.c_void_p), ("frozen_role_count", ctypes.c_int)]

    options = AttachOptions(0, None, 0)
    engine, _ = create_engine(lib, args.model_dir, descriptor, devices)
    if not check(lib.engine_load_quantized_ffn(engine, out_dir.encode()) == 0,
                 lib.engine_last_error().decode()):
        return 1
    store = lib.engine_train_attach(engine, ctypes.byref(options))
    print(f"  a training store on an INT4 engine: refused={not store} "
          f"({lib.engine_last_error().decode()})")
    if not check(not store, "a training store attached to an INT4 engine"):
        failures += 1
    lib.engine_destroy(engine)

    engine, _ = create_engine(lib, args.model_dir, descriptor, devices)
    store = lib.engine_train_attach(engine, ctypes.byref(options))
    if not check(bool(store), f"a store should attach to a BF16 engine: "
                              f"{lib.train_last_error().decode()}"):
        failures += 1
    blocked = lib.engine_load_quantized_ffn(engine, out_dir.encode())
    print(f"  the INT4 load onto a training engine: refused={blocked != 0} "
          f"({lib.engine_last_error().decode()})")
    if not check(blocked != 0, "the INT4 load was accepted on an engine with a training store"):
        failures += 1
    lib.engine_destroy(engine)

    # Refusals through the same entry point: a corrupted payload and a tampered format block
    # must not reach the kernel. The artifact is the one the engine actually reads - the F1
    # pair's packed payload, not a member's (the engine loads the concatenated operand).
    manifest_json = json.load(open(os.path.join(out_dir, "weights.manifest.json")))
    packed_path = os.path.join(out_dir, manifest_json["pairs"][0]["packed"]["file"])
    with open(packed_path, "r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
    engine, _ = create_engine(lib, args.model_dir, descriptor, devices)
    corrupted = lib.engine_load_quantized_ffn(engine, out_dir.encode())
    print(f"  a corrupted artifact: refused={corrupted != 0} "
          f"({lib.engine_last_error().decode()})")
    if not check(corrupted != 0, "a corrupted artifact was accepted"):
        failures += 1
    lib.engine_destroy(engine)

    tampered_dir = os.path.join(workdir, "tampered")
    shutil.copytree(out_dir, tampered_dir)
    tampered_path = os.path.join(tampered_dir, "weights.manifest.json")
    tampered = json.load(open(tampered_path))
    tampered["format"] = dict(tampered["format"], group=64)
    with open(tampered_path, "w") as handle:
        json.dump(tampered, handle)
    engine, _ = create_engine(lib, args.model_dir, descriptor, devices)
    rejected = lib.engine_load_quantized_ffn(engine, tampered_dir.encode())
    print(f"  a tampered format block: refused={rejected != 0} "
          f"({lib.engine_last_error().decode()})")
    if not check(rejected != 0, "a tampered format block was accepted"):
        failures += 1
    lib.engine_destroy(engine)

    if temporary:
        shutil.rmtree(workdir, ignore_errors=True)
    if failures:
        print(f"test_quantized_ffn: {failures} check(s) failed")
        return 1
    print("test_quantized_ffn: the decode path reads the packed operands, prefill is bitwise "
          "untouched, every top-1 flip is a tie-break the perturbation could explain, the "
          "held-out NLL agrees with BF16, the load moves only numerical_policy_id, a corrupted "
          "or tampered sidecar is refused, and the training and INT4 postures exclude each other")
    return 0


if __name__ == "__main__":
    sys.exit(main())
