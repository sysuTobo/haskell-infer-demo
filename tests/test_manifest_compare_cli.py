#!/usr/bin/env python3
"""CLI gate for `manifest-compare`: the real executable's admission verdicts.

The comparison rules are unit-tested in tests/ManifestSpec.hs; this runner checks
that the shipped command wires them to the exit codes the contract fixes:

    0 admitted (strict, or scoped with a declaration)
    1 rejected
    2 legacy/unverified
    3 diagnostic-only

The manifests are built here and their digests re-derived with manifest_check, so
the documents the CLI reads are valid ones rather than hand-written strings.

    python tests/test_manifest_compare_cli.py \
        --exe dist-newstyle/build/.../haskell-infer-demo
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifest_check  # noqa: E402

REGIONS = [
    {"cases": "prefill,decode", "determinism": "unverified",
     "implementation": "cuBLAS BF16 in, FP32 accumulate, BF16 out",
     "mechanism": "cuBLAS algorithm selection is not pinned", "region": "gemm_bf16",
     "rng_dependency": False},
    {"cases": "not_implemented", "determinism": "not_implemented",
     "implementation": "no backward region is implemented",
     "mechanism": "the trainer plan stages 3-4 define this", "region": "backward",
     "rng_dependency": False},
]

BASE_PROVENANCE = {
    "build": {"cuda_toolkit": "12.9", "git_commit": "abc", "triton_version": "3.4.0",
              "fla_version": "0.5.2"},
    "runtime": {"cublas_version": "12.9.0", "cuda_driver_version": "580.65.06",
                "cuda_runtime_version": "12.9",
                "devices": [{"compute_capability": "8.6", "cuda_ordinal": 0,
                             "kernel_path": "sass", "name": "NVIDIA A40",
                             "triton_cubin_arch": "86",
                             "uuid": "GPU-00000000-0000-0000-0000-000000000001"}]},
}

BASE_SAMPLING = {"arithmetic": "host_fp32_logits_exact_compare", "mode": "greedy",
                 "rng": "none", "transform": "argmax_lowest_token_id",
                 "transform_version": 1}


def build(name, descriptor_sha="desc", provenance=None, sampling=None,
          semantic_fields=None, numerical_fields=None, semantic_id=None,
          numerical_id=None, deployment_id=None, weights_sha="w"):
    """A valid canonical manifest whose digests are re-derived here."""
    sem_fields = semantic_fields if semantic_fields is not None else {
        "hidden_size": 256, "num_layers": 2, "tied_roles": ["embed=lmHead"]}
    num_fields = numerical_fields if numerical_fields is not None else {
        "max_chunk": 64, "regions_sha256": manifest_check.digest(REGIONS)}
    deployment_fields = {"devices": [0, 1], "placement": "layer_split", "tensor_count": 2}
    doc = {
        "manifest_version": 1,
        "descriptor": {"bytes": 100, "desc_version": 1, "sha256": descriptor_sha},
        "semantic": {"fields": sem_fields,
                     "semantic_id": semantic_id or manifest_check.digest(sem_fields)},
        "numerical_policy": {"fields": num_fields,
                             "numerical_policy_id": numerical_id
                             or manifest_check.digest(num_fields)},
        "deployment": {"fields": deployment_fields,
                       "deployment_id": deployment_id
                       or manifest_check.digest(deployment_fields)},
        "provenance": provenance if provenance is not None else BASE_PROVENANCE,
        "regions": REGIONS,
        "sampling": sampling if sampling is not None else BASE_SAMPLING,
        "weights": {"content_sha256": None, "parameter_manifest_sha256": weights_sha,
                    "shards_sha256": "s", "tensor_count": 7},
    }
    return manifest_check.canonical(doc)


def run(exe, left, right, extra=None):
    command = [exe, "manifest-compare", left, right] + (extra or [])
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()

    problems = []

    def check(label, expected_code, expected_text, left_text, right_text, extra=None, work=None):
        left = os.path.join(work, "left.json")
        right = os.path.join(work, "right.json")
        with open(left, "w", encoding="utf-8") as handle:
            handle.write(left_text)
        with open(right, "w", encoding="utf-8") as handle:
            handle.write(right_text)
        code, output = run(args.exe, left, right, extra)
        if code != expected_code:
            problems.append(f"{label}: exit {code}, expected {expected_code}\n{output}")
        elif expected_text not in output:
            problems.append(f"{label}: output does not mention {expected_text!r}\n{output}")
        return output

    with tempfile.TemporaryDirectory() as work:
        baseline = build("baseline")
        check("identical captures", 0, "admitted", baseline, baseline, work=work)

        changed_semantic = build("semantic", semantic_fields={
            "hidden_size": 320, "num_layers": 2, "tied_roles": ["embed=lmHead"]})
        check("a semantic change rejects", 1, "rejected", changed_semantic, baseline, work=work)

        changed_numerical = build("numerical", numerical_fields={
            "max_chunk": 32, "regions_sha256": manifest_check.digest(REGIONS)})
        check("a numerical change rejects", 1, "rejected", changed_numerical, baseline, work=work)
        check("diagnostic mode reports it without passing",
              3, "diagnostic-only", changed_numerical, baseline,
              extra=["--mode", "diagnostic"], work=work)

        changed_weights = build("weights", weights_sha="w2")
        check("a weight change rejects", 1, "rejected", changed_weights, baseline, work=work)

        # Placement: the deployment identity and the descriptor reference both
        # carry it, so both need the declared scope.
        moved_deployment = build("deployment", descriptor_sha="desc2",
                                 deployment_id="moved")
        check("placement alone rejects without a declaration",
              1, "rejected", moved_deployment, baseline, work=work)
        check("placement alone is admitted with a declaration",
              0, "deployment scope",
              moved_deployment, baseline, extra=["--deployment-scoped"], work=work)

        # Provenance: a different runtime is not a different identity, but it is
        # not comparable either.
        other_build = json.loads(json.dumps(BASE_PROVENANCE))
        other_build["build"]["cuda_toolkit"] = "13.0"
        check("differing provenance rejects", 1, "rejected",
              build("other-build", provenance=other_build), baseline, work=work)

        # A fact the build could not establish must refuse, even when both sides
        # agree on the spelling.
        unknown_build = json.loads(json.dumps(BASE_PROVENANCE))
        unknown_build["build"]["cuda_toolkit"] = "unknown"
        unknown = build("unknown", provenance=unknown_build)
        check("unestablished provenance rejects", 1, "not established", unknown, unknown, work=work)

        # A document without a manifest version is legacy, never a pass.
        legacy = json.dumps({"meta": "a pre-manifest capture"}, sort_keys=True,
                            separators=(",", ":"))
        check("a legacy document is unverified", 2, "legacy-unverified", legacy, baseline,
              work=work)

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        print(f"test_manifest_compare_cli: {len(problems)} problem(s)")
        return 1
    print("test_manifest_compare_cli: strict, scoped, diagnostic and legacy verdicts "
          "match the contract's exit codes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
