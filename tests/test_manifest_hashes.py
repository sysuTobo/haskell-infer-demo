#!/usr/bin/env python3
"""ctest gate: verify the emitted execution manifest in a second implementation.

The C test (tests/manifest_test.c) checks the identity matrix from the inside.
This runner re-derives every digest from the parsed document with hashlib and
re-checks the same matrix across separately emitted documents, so a bug in the
emitter cannot certify itself. No GPU and no weights are needed.

    python tests/test_manifest_hashes.py --emitter csrc/build-libs/test_manifest
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifest_check  # noqa: E402

# variant -> the identities that must move relative to the baseline
MATRIX = {
    "semantic": {"semantic_id"},
    "projected_constant": {"semantic_id", "numerical_policy_id"},
    "numerical": {"numerical_policy_id"},
    "regions": {"numerical_policy_id"},
    "deployment": {"deployment_id"},
    "provenance": set(),
    "weights": set(),
}

ALL_IDS = ("semantic_id", "numerical_policy_id", "deployment_id")


def emit(emitter, path, variant=None):
    command = [emitter, "--emit", path]
    if variant is not None:
        command = [emitter, "--emit-variant", path, variant]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed: {result.stderr.strip()}")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def check(label, condition, problems, detail=""):
    if not condition:
        problems.append(f"{label}: {detail}" if detail else label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emitter", required=True)
    args = parser.parse_args()

    problems = []
    with tempfile.TemporaryDirectory() as work:
        baseline = emit(args.emitter, os.path.join(work, "baseline.json"))
        documents = {"baseline": baseline}
        for variant in MATRIX:
            documents[variant] = emit(args.emitter, os.path.join(work, variant + ".json"),
                                      variant)

        # 1. Canonical encoding and every digest, re-derived independently.
        for name, text in sorted(documents.items()):
            for problem in manifest_check.verify(text):
                problems.append(f"{name}: {problem}")

        # 2. The identity matrix.
        base_ids = manifest_check.identities(baseline)
        for variant, expected in sorted(MATRIX.items()):
            variant_ids = manifest_check.identities(documents[variant])
            for key in ALL_IDS:
                moved = base_ids[key] != variant_ids[key]
                if moved and key not in expected:
                    problems.append(f"{variant}: {key} moved but should not have")
                if not moved and key in expected:
                    problems.append(f"{variant}: {key} did not move but should have")

        # 3. The manifest has to record enough for a strict comparison: build and
        # runtime provenance, the parameter identity, the region table with an
        # explicit backward record, and the sampling policy.
        import json

        parsed = json.loads(baseline)
        provenance = parsed.get("provenance", {})
        build = provenance.get("build", {})
        check("provenance.build is present", isinstance(build, dict) and bool(build), problems)
        for key in ("cuda_archs", "git_commit", "triton_version", "fla_version",
                    "generated_kernels_sha256", "nvcc_flags_sha256"):
            check(f"provenance.build records {key}", bool(build.get(key)), problems)
        runtime = provenance.get("runtime", {})
        for key in ("cuda_runtime_version", "cuda_driver_version", "cublas_version"):
            check(f"provenance.runtime records {key}", bool(runtime.get(key)), problems)
        devices = runtime.get("devices", [])
        check("provenance.runtime.devices is present", bool(devices), problems)
        for key in ("kernel_path", "uuid", "compute_capability", "triton_cubin_arch"):
            check(f"provenance.runtime.devices records {key}",
                  bool(devices) and bool(devices[0].get(key)), problems)
        weights = parsed.get("weights", {})
        check("weights.parameter_manifest_sha256 is present",
              bool(weights.get("parameter_manifest_sha256")), problems)
        regions = {entry.get("region") for entry in parsed.get("regions", [])}
        for required in ("attention_core", "gdn_core", "gemm_bf16", "backward"):
            check(f"the region table records {required}", required in regions, problems)
        backward = [entry for entry in parsed.get("regions", [])
                    if entry.get("region") == "backward"]
        check("the backward region records determinism separately",
              bool(backward) and bool(backward[0].get("determinism"))
              and bool(backward[0].get("mechanism")), problems)
        sampling = parsed.get("sampling", {})
        check("sampling records the greedy policy and that it consumes no RNG",
              sampling.get("mode") == "greedy" and sampling.get("rng") == "none", problems)

        # 4. An unestablished fact must be visible as such, not defaulted away.
        unestablished = manifest_check.unestablished_paths(baseline)
        check("a fully specified test manifest reports no unestablished provenance",
              not unestablished, problems, f"unestablished: {unestablished}")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        print(f"test_manifest_hashes: {len(problems)} problem(s)")
        return 1
    print("test_manifest_hashes: canonical encoding, digests and the identity matrix "
          "verify in a second implementation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
