"""Capture engine logits as a golden reference, and compare two captures.

Capture (needs weights):
  python tests/capture_logits.py --library csrc/build-libs/libengine.so \
      --model-dir "$MODEL_DIR" --output /path/golden.npz

Verify a refactor did not change numerics (same build, same prompt, same weights
=> the engine is deterministic, so any difference means behaviour changed):
  python tests/capture_logits.py --compare old.npz new.npz

A capture records the execution manifest it was taken under, so a comparison has
three outcomes and only one of them is a pass:

  strict (default when both captures carry a manifest)
      the semantic, numerical-policy and parameter identities agree, the
      placement agrees (or the caller declares a scope covering it, and then it is
      reported as scoped rather than as an identity-level pass), the provenance is
      established and equal, and every numeric array is bitwise identical.
      Exit 0 admitted, 1 rejected.
  diagnostic (--compare-mode diagnostic)
      a deliberate manifest difference is reported for attribution and never
      called a contract pass. Exit 3.
  legacy (one side has no manifest, e.g. an older golden)
      the numeric arrays stay comparable, but no identity is established for the
      document and none is invented. Exit 2, never 0.

The admission rules implemented here mirror Infer.Manifest (the authoritative
interface, exercised by tests/ManifestSpec.hs and the `manifest-compare` CLI).
"""

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_bindings import bind, create_engine, load_descriptor, manifest, ptr  # noqa: E402
import manifest_check  # noqa: E402

# "The capital of France is" -> Qwen tokenizer ids (verified in the worklog).
PROMPT_MAIN = [760, 6511, 314, 9564, 369]

# The engine chunks prefill at 128 tokens; 129 crosses a chunk boundary.
LONG_PROMPT_TOKENS = 129

# Keys that describe *how* the capture was taken rather than a numeric array.
PROVENANCE_KEYS = ("meta", "provenance")


def capture(args):
    lib = bind(ctypes.CDLL(args.library))
    devices = [int(x) for x in args.devices.split(",")]
    descriptor = load_descriptor(args.desc, max_seq_len=args.max_seq_len)

    start = time.monotonic()
    engine, vocab = create_engine(lib, args.model_dir, descriptor, devices)
    print(f"Loaded in {time.monotonic() - start:.1f}s (vocab {vocab})", flush=True)
    logits = np.empty(vocab, dtype=np.float32)
    records = {}
    cases = {}

    def status(code):
        assert code == 0, lib.engine_last_error().decode()

    def run_case(name, prompt):
        lib.engine_reset(engine)
        assert lib.engine_seq_len(engine) == 0
        prompt = np.ascontiguousarray(prompt, dtype=np.int64)
        status(lib.engine_prefill(engine, ptr(prompt), len(prompt), ptr(logits)))
        records[f"prompt_{name}"] = prompt
        cases[name] = {"prompt_tokens": [int(t) for t in prompt], "prompt_len": len(prompt),
                       "steps": args.steps}
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
        # The manifest is read while the engine is alive and verified by
        # manifest_check before anything is written, so a capture never embeds a
        # manifest whose digests do not re-derive.
        manifest_text = manifest(lib, engine)
    finally:
        lib.engine_destroy(engine)

    # Kept for older tooling that reads the flat meta block. The provenance block
    # below is the authoritative record; `meta` is not used for admission.
    meta = {"vocab": int(vocab), "devices": devices, "steps": args.steps,
            "desc": args.desc, "prompt_main": PROMPT_MAIN}
    records["meta"] = np.array(json.dumps(meta, sort_keys=True))

    provenance = {
        "manifest": manifest_text,
        "manifest_ids": manifest_check.identities(manifest_text),
        "cases": cases,
        # The trainer's Stage 1 inventory needs the layout conventions, not just
        # the values: which row the logits belong to and how positions count.
        "mask_convention": "causal; the capture records the last row's logits",
        "position_convention": "0..n-1 from the start of each request; reset before each case",
        "sampling": {
            "mode": "greedy",
            "temperature": 0.0,
            "seed": None,
            "transform": "argmax_lowest_token_id",
            "rng": "none",
            "draw_mapping": "one RNG word per sampled token at T>0; greedy consumes none",
        },
        # A capture is replay data, not a training artifact: there is no optimizer
        # version yet, so the run identity defaults to the immutable parameter
        # manifest the manifest already carries. The trainer's stages add the
        # monotonic version.
        "run": {
            "run_id": args.run_id,
            "parameter_version": args.parameter_version,
            "parameter_manifest_sha256": manifest_check.identities(manifest_text)[
                "parameter_manifest_sha256"],
        },
    }
    records["provenance"] = np.array(json.dumps(provenance, sort_keys=True))

    np.savez(args.output, **records)
    print(f"Wrote {args.output}", flush=True)
    print("  semantic_id:  " + provenance["manifest_ids"]["semantic_id"], flush=True)
    print("  numerical_id: " + provenance["manifest_ids"]["numerical_policy_id"], flush=True)
    print("  deployment:   " + provenance["manifest_ids"]["deployment_id"], flush=True)


def provenance_of(capture_file):
    """The parsed provenance block, or None for a legacy capture."""
    if "provenance" not in capture_file.files:
        return None
    return json.loads(capture_file["provenance"].item())


def compare_manifests(left_text, right_text, mode, deployment_scoped):
    """Returns (verdict, note). Mirrors Infer.Manifest.compareManifests."""
    left_ids = manifest_check.identities(left_text)
    right_ids = manifest_check.identities(right_text)
    identity_keys = ("semantic_id", "numerical_policy_id", "parameter_manifest_sha256")
    identity_diffs = [key for key in identity_keys if left_ids[key] != right_ids[key]]
    # The descriptor reference is scoped rather than an identity: its canonical
    # text carries the placement fields.
    placement_diffs = [key for key in ("deployment_id",) if left_ids[key] != right_ids[key]]
    left = json.loads(left_text)
    right = json.loads(right_text)
    if left["descriptor"]["sha256"] != right["descriptor"]["sha256"]:
        placement_diffs.append("descriptor_sha256")
    provenance_diffs = manifest_check.provenance_differences(left_text, right_text)
    unestablished = (manifest_check.unestablished_paths(left_text)
                     + manifest_check.unestablished_paths(right_text))

    if mode == "diagnostic":
        return "diagnostic-only", ("reported for attribution; this is NOT a strict contract pass"
                                   + (f"; differing identities: {identity_diffs}" if identity_diffs else "")
                                   + (f"; placement: {placement_diffs}" if placement_diffs else ""))
    if identity_diffs:
        return "rejected", f"identity mismatch: {identity_diffs}"
    if placement_diffs and not deployment_scoped:
        return "rejected", ("placement differs (" + ", ".join(placement_diffs)
                            + "); pass --deployment-scoped to declare a scoped invariance claim")
    if unestablished:
        return "rejected", ("provenance is not established: " + ", ".join(unestablished))
    if provenance_diffs:
        return "rejected", ("provenance differs: "
                            + ", ".join(path for path, _, _ in provenance_diffs))
    if placement_diffs:
        return "admitted-with-declared-deployment-scope", ("placement differs ("
                                                          + ", ".join(placement_diffs) + ")")
    return "admitted", "identities, parameter identity and provenance agree"


def compare(args):
    left = np.load(args.compare[0])
    right = np.load(args.compare[1], allow_pickle=False)
    left_numeric = sorted(set(left.files) - set(PROVENANCE_KEYS))
    right_numeric = sorted(set(right.files) - set(PROVENANCE_KEYS))
    if left_numeric != right_numeric:
        print(f"numeric key mismatch: only-left={sorted(set(left_numeric) - set(right_numeric))} "
              f"only-right={sorted(set(right_numeric) - set(left_numeric))}")
        return 1

    left_provenance = provenance_of(left)
    right_provenance = provenance_of(right)
    mode = args.compare_mode
    if mode == "auto":
        mode = "strict" if left_provenance and right_provenance else "legacy"

    failed = False
    if mode == "legacy":
        verdict, note = "legacy-unverified", ("at least one capture has no manifest; the numeric "
                                              "arrays are compared, but no identity is established")
    elif left_provenance is None or right_provenance is None:
        print("ERROR: a capture without a manifest cannot be compared in --compare-mode strict")
        return 1
    else:
        verdict, note = compare_manifests(left_provenance["manifest"], right_provenance["manifest"],
                                          mode, args.deployment_scoped)

    # The numeric gate is unchanged and stays absolute: bitwise identity on every
    # array. A manifest verdict never loosens it.
    worst_key, worst_abs, worst_rms, differing = None, 0.0, 0.0, 0
    for key in left_numeric:
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
    elif worst_key is not None:
        print(f"NOT bitwise identical: worst={worst_key} max_abs={worst_abs:.6g} "
              f"rms={worst_rms:.6g} differing_elements={differing}")
        failed = True

    print(f"manifest verdict: {verdict}")
    print(f"  {note}")
    if failed:
        return 1
    if verdict == "diagnostic-only":
        return 3
    if verdict == "legacy-unverified":
        return 2
    return 0


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
    parser.add_argument("--run-id", default="inference",
                        help="run identifier recorded with the capture")
    parser.add_argument("--parameter-version", type=int, default=0,
                        help="monotonic parameter version (no optimizer yet, so 0)")
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--compare-mode", choices=("auto", "strict", "diagnostic"), default="auto",
                        help="auto: strict when both captures carry a manifest, else legacy")
    parser.add_argument("--deployment-scoped", action="store_true",
                        help="declare that a scoped invariance claim covers a placement-only difference")
    args = parser.parse_args()
    if args.compare:
        return compare(args)
    if not args.library or not args.model_dir or not args.output:
        parser.error("--library, --model-dir and --output are required for capture")
    capture(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
