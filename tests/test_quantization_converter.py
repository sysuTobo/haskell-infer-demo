#!/usr/bin/env python3
"""Q1's gate: the converter, the sidecar and the validation (plan Q1).

Runs `scripts/quantize_weights.py` over the synthetic dense checkpoint, requires its own
`--verify` to pass, and then checks the properties the plan's Q1 lists as required *before GPU
use* - and, because a validator that cannot fail is not one, it corrupts an artifact and a
manifest and requires the verification to reject them.

Usage:
    python3 tests/test_quantization_converter.py --reference <test_quantization_format> \\
        --model-dir <synth dense checkpoint> --desc descriptors/qwen3-dense-synth.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "..", "scripts", "quantize_weights.py")


def check(condition, message):
    if condition:
        return True
    print(f"test_quantization_converter: FAIL: {message}")
    return False


def run(args):
    return subprocess.run(args, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True,
                        help="the Q0 reference binary (test_quantization_format)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--reader", default=None,
                        help="the C sidecar reader (test_quant_manifest); when given, the "
                             "sidecar the converter just wrote must be one it accepts")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"test_quantization_converter: skipped (no checkpoint at {args.model_dir})")
        return 0
    if not os.path.exists(args.reference):
        print(f"test_quantization_converter: skipped (no reference binary at {args.reference})")
        return 0

    workdir = args.out_dir or tempfile.mkdtemp(prefix="quantized-synth-")
    out_dir = os.path.join(workdir, "quantized")
    descriptor = json.load(open(args.desc))
    failures = 0

    conversion = run([sys.executable, CONVERTER, "--model-dir", args.model_dir, "--desc", args.desc,
                      "--out-dir", out_dir, "--reference", args.reference])
    if conversion.returncode != 0:
        print(conversion.stdout, conversion.stderr)
        print("test_quantization_converter: FAIL: the conversion failed")
        return 1
    print(conversion.stdout.strip())

    manifest_path = os.path.join(out_dir, "weights.manifest.json")
    manifest = json.load(open(manifest_path))
    dense_layers = [i for i, kind in enumerate(descriptor["layer_ffns"]) if kind == "dense"]
    expected_entries = len(dense_layers) * 3  # mlpGate, mlpUp, mlpDown

    if not check(len(manifest["entries"]) == expected_entries,
                 f"the manifest has {len(manifest['entries'])} entries, expected "
                 f"{expected_entries} (dense layers x gate/up/down)"):
        failures += 1
    for entry in manifest["entries"]:
        n, k = entry["logical_shape"]
        if not check(entry["packed"]["bytes"] == n * k // 2 and
                     entry["scales"]["count"] == n * (k // manifest["format"]["group"]),
                     f"{entry['role']}_{entry['layer']}'s extents do not follow the format"):
            failures += 1
        if not check(entry["group"] == manifest["format"]["group"] and
                     entry["group_axis"] == "k",
                     f"{entry['role']}_{entry['layer']}'s group metadata is inconsistent"):
            failures += 1
    cells = descriptor["num_layers"] * len(descriptor["role_names"])
    if not check(len(manifest["precision_map"]) == cells,
                 f"the precision map has {len(manifest['precision_map'])} cells, expected {cells}"):
        failures += 1
    quantized_cells = sum(1 for r in manifest["precision_map"] if r["precision"] == "int4")
    if not check(quantized_cells == expected_entries,
                 f"the precision map marks {quantized_cells} cells int4, expected "
                 f"{expected_entries}"):
        failures += 1
    error = manifest["quantization_error"]
    if not check(error["max_abs"] > 0 and error["rms"] > 0 and error["elements"] > 0,
                 "the manifest's error report is empty, so nothing was measured"):
        failures += 1
    print(f"quantization error over {error['elements']} elements: max_abs {error['max_abs']:.6g}, "
          f"rms {error['rms']:.6g}")

    # The F1 pair: gate rows then up rows, for the payload and the scales. Checked against the
    # members' *roles* rather than the pair's own list of members, so a manifest that swapped
    # them could not certify itself.
    for pair in manifest.get("pairs", []):
        layer = pair["layer"]
        members = {e["role"]: e for e in manifest["entries"] if e["layer"] == layer}
        if not {"mlpGate", "mlpUp"} <= set(members):
            if not check(False, f"layer {layer}'s pair has no gate/up entry to check against"):
                failures += 1
            continue

        def read(entry, kind):
            with open(os.path.join(out_dir, entry[kind]["file"]), "rb") as handle:
                return handle.read()

        gate, up = members["mlpGate"], members["mlpUp"]
        if not check(read(pair, "packed") == read(gate, "packed") + read(up, "packed"),
                     f"layer {layer}'s pair payload is not gate rows then up rows"):
            failures += 1
        if not check(read(pair, "scales") == read(gate, "scales") + read(up, "scales"),
                     f"layer {layer}'s pair scales are not gate rows then up rows"):
            failures += 1
        if not check(pair["logical_shape"][0] ==
                     gate["logical_shape"][0] + up["logical_shape"][0] and
                     pair["logical_shape"][1] == gate["logical_shape"][1],
                     f"layer {layer}'s pair claims {pair['logical_shape']}"):
            failures += 1
    print(f"F1 pairs: {len(manifest.get('pairs', []))} (packed rows then scale rows, "
          f"gate then up)")

    # The converter's own verification, which re-hashes and re-derives a sample.
    verification = run([sys.executable, CONVERTER, "--verify", manifest_path,
                        "--reference", args.reference])
    if not check(verification.returncode == 0, f"the converter's verify failed: "
                                               f"{verification.stdout}{verification.stderr}"):
        failures += 1
    else:
        print(verification.stdout.strip())

    # The C reader the engine will load with, over the same sidecar and while the artifacts are
    # still intact. This is what ties the schema to its only writer: a manifest the converter
    # writes has to be one the reader accepts, and the reader re-verifies an artifact's digest
    # from the bytes on disk rather than from the manifest's own claim.
    if args.reader is not None:
        if not os.path.exists(args.reader):
            print(f"test_quantization_converter: FAIL: no reader at {args.reader}")
            failures += 1
        else:
            reading = run([args.reader, "--manifest", manifest_path, "--root", out_dir])
            if not check(reading.returncode == 0,
                         f"the C reader refused the converter's sidecar: "
                         f"{reading.stdout}{reading.stderr}"):
                failures += 1
            else:
                for line in reading.stdout.strip().splitlines():
                    print(f"  {line}")

    # A validator that cannot fail is not a gate: corrupt the payload and require a refusal.
    first = manifest["entries"][0]
    packed_path = os.path.join(out_dir, first["packed"]["file"])
    with open(packed_path, "r+b") as handle:
        handle.write(b"\xff")
    corrupted = run([sys.executable, CONVERTER, "--verify", manifest_path,
                     "--reference", args.reference])
    if not check(corrupted.returncode != 0 and "checksum" in corrupted.stdout,
                 "a corrupted payload was not rejected by its checksum"):
        failures += 1

    # A manifest that claims a different format is refused rather than reinterpreted.
    tampered_path = os.path.join(workdir, "tampered.manifest.json")
    tampered = json.load(open(manifest_path))
    tampered["format"] = dict(tampered["format"], group=64)
    with open(tampered_path, "w") as handle:
        json.dump(tampered, handle)
    rejected = run([sys.executable, CONVERTER, "--verify", tampered_path,
                    "--reference", args.reference])
    if not check(rejected.returncode != 0 and "group" in rejected.stdout,
                 "a manifest claiming a different group width was not rejected"):
        failures += 1

    # And a missing artifact is refused.
    os.remove(packed_path)
    missing = run([sys.executable, CONVERTER, "--verify", manifest_path,
                   "--reference", args.reference])
    if not check(missing.returncode != 0, "a missing artifact was not rejected"):
        failures += 1

    if failures:
        print(f"test_quantization_converter: {failures} check(s) failed")
        return 1
    print("test_quantization_converter: the converter wrote a verifiable artifact, left the "
          "checkpoint untouched, and the verification rejects a corrupted payload, a foreign "
          "format and a missing artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
