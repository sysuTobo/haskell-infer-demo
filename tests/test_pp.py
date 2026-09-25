#!/usr/bin/env python3
"""Stage 2 claim A: pipeline-parallel inertness.

docs/plan-numeric-contract.md, claim A: "A model that fits one GPU, fixed
weights/prompt/build, one-device versus two-device layer split on matching GPU
architectures. Compare intermediate boundaries and logits bitwise; scope any pass
to tested configurations. The 27B checkpoint is not the single-A40 fixture."

This runs the same checkpoint twice -- once with every layer on device 0 and once
with the layer split across devices 0 and 1 -- and compares

  * the logits, through the Stage-0 capture machinery: a placement-only difference
    is exactly what `--deployment-scoped` exists for, so the comparison is reported
    as a *scoped* exception and the numeric arrays must still be bitwise identical;
  * every intermediate boundary, through the engine's debug taps
    (INFER_TAP_LAYERS/INFER_TAP_DIR): each layer's residual stream plus its mixer
    and ffn outputs, compared byte for byte.

The fixture is Qwen3-4B (dense attention, 36 layers, ~8 GB bf16): a model that fits
one A40, which the 27B does not. A pass is scoped to the tested configuration --
same architecture, same build, two A40s -- and says nothing about a different
device mix.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, env=None):
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)


def capture(args, devices, tap_dir, output):
    env = dict(os.environ)
    env["INFER_TAP_DIR"] = tap_dir
    env["INFER_TAP_LAYERS"] = ",".join(str(i) for i in range(args.layers))
    os.makedirs(tap_dir, exist_ok=True)
    result = run([sys.executable, "tests/capture_logits.py", "--library", args.library,
                  "--model-dir", args.model_dir, "--desc", args.desc,
                  "--devices", devices, "--steps", str(args.steps),
                  "--run-id", "pp-" + devices.replace(",", "_"), "--output", output], env=env)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"capture on devices {devices} failed ({result.returncode})")
    return [f for f in sorted(os.listdir(tap_dir)) if f.endswith(".f32")]


def compare_taps(dir_a, dir_b):
    """Byte-for-byte comparison of the two tap dumps. The dumps are raw float32
    views of the bf16 activations, so equal bytes is equal activations."""
    files_a = sorted(f for f in os.listdir(dir_a) if f.endswith(".f32"))
    files_b = sorted(f for f in os.listdir(dir_b) if f.endswith(".f32"))
    if files_a != files_b:
        only_a = sorted(set(files_a) - set(files_b))
        only_b = sorted(set(files_b) - set(files_a))
        print(f"FAIL tap sets differ: only in placement A: {only_a}, only in B: {only_b}")
        return False, 0
    differing = []
    for name in files_a:
        with open(os.path.join(dir_a, name), "rb") as fa, open(os.path.join(dir_b, name), "rb") as fb:
            if fa.read() != fb.read():
                differing.append(name)
    print(f"taps: {len(files_a)} files compared, {len(differing)} differ")
    for name in differing[:10]:
        print(f"  DIFFERS: {name}")
    return not differing, len(files_a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", default="descriptors/qwen3-4b.json")
    parser.add_argument("--layers", type=int, default=36,
                        help="layer count to tap (from the descriptor)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--work-dir", default="/tmp/pp_inertness")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    single = os.path.join(args.work_dir, "one_device")
    split = os.path.join(args.work_dir, "two_device")

    print("=== placement A: every layer on device 0 ===")
    capture(args, "0", single, os.path.join(args.work_dir, "one.npz"))
    print("=== placement B: layer split across devices 0,1 ===")
    capture(args, "0,1", split, os.path.join(args.work_dir, "two.npz"))

    taps_ok, tap_count = compare_taps(single, split)

    print("=== logits: placement-only difference, declared as a scoped claim ===")
    one = os.path.join(args.work_dir, "one.npz")
    two = os.path.join(args.work_dir, "two.npz")
    result = run([sys.executable, "tests/capture_logits.py", "--compare", one, two,
                  "--deployment-scoped"])
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    bitwise = "bitwise identical" in result.stdout and "max_abs == 0" in result.stdout

    # The manifest verdict is a separate statement from the numeric one, and on
    # this claim it is expected to be `rejected` rather than `admitted`: running the
    # layers on one device instead of two changes the runtime provenance (the device
    # list and every per-device fact), and strict admission requires provenance to
    # agree -- a scoped placement claim relaxes the placement, not the device set.
    # What has to hold is narrower, and is read straight out of both captures: the
    # semantic, numerical and parameter identities are equal and the deployment
    # identity is the one that moved.
    ids = {}
    for label, path in (("one_device", one), ("two_device", two)):
        with np.load(path) as archive:
            provenance = json.loads(str(archive["provenance"]))
        ids[label] = provenance["manifest_ids"]
    same_semantic = ids["one_device"]["semantic_id"] == ids["two_device"]["semantic_id"]
    same_numerical = (ids["one_device"]["numerical_policy_id"]
                      == ids["two_device"]["numerical_policy_id"])
    same_weights = (ids["one_device"]["parameter_manifest_sha256"]
                    == ids["two_device"]["parameter_manifest_sha256"])
    deployment_differs = (ids["one_device"]["deployment_id"]
                          != ids["two_device"]["deployment_id"])
    print(f"identities: semantic {'equal' if same_semantic else 'DIFFER'}, "
          f"numerical_policy {'equal' if same_numerical else 'DIFFER'}, "
          f"parameter {'equal' if same_weights else 'DIFFER'}, "
          f"deployment {'differs as expected' if deployment_differs else 'DID NOT DIFFER'}")
    verdict = [l for l in result.stdout.splitlines() if "manifest verdict" in l]
    print(f"manifest verdict: {verdict[0].strip() if verdict else 'not reported'} "
          f"(exit {result.returncode}); the numeric gate is unchanged either way")

    tap_verdict = "bitwise identical" if taps_ok else "DIFFER"
    logits_verdict = "bitwise identical" if bitwise else "DIFFER"
    identities_ok = same_semantic and same_numerical and same_weights and deployment_differs
    print()
    print(f"claim A summary: taps {tap_verdict} ({tap_count} files), logits {logits_verdict}, "
          f"identities {'clean placement-only difference' if identities_ok else 'MISMATCH'}")
    if not (taps_ok and bitwise and identities_ok):
        print("test_pp: FAIL")
        return 1
    print("test_pp: PASS (scoped to the tested configuration: same build, two A40 sm_86; the "
          "strict manifest verdict refuses the device-set difference, which is the contract "
          "working as intended)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
