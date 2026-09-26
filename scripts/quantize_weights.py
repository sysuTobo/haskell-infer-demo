#!/usr/bin/env python3
"""Q1: the offline converter and the `weights.manifest.json` sidecar (plan Q1).

docs/plan-numeric-contract.md's Q1 asks the converter to write quantized artifacts into a
*separate* output directory without touching the BF16 checkpoint, and to record enough in a
strict, versioned sidecar that a loader can validate the artifact before a GPU sees it:
source checkpoint and checksum, converter and config version, the per-layer/role mapping, the
logical [N, K], the packed layout/shape/dtype, the group axis and size, the scale
tensor/shape/dtype, the zero-point convention, artifact hashes, and an explicit precision map
of quantized versus retained-BF16 roles.

Two design points are worth stating, because they are what make the artifact checkable:

* **the quantization arithmetic is not re-implemented here.** Each tensor is handed to the Q0
  reference (`tests/quantization_format_test.c --quantize`, i.e. `csrc/linear_weight.cpp`),
  so a converted artifact is produced by the same code a kernel is admitted against (Q2) and
  the converter cannot drift from it. This file is I/O, layout and bookkeeping.
* **`--verify` re-derives rather than re-reads.** It re-hashes every artifact, re-checks the
  extents against the format, requires the precision map to cover every role of every layer
  exactly once, and re-quantizes a sample of entries from the *source checkpoint* to confirm
  the artifact is reproducible. A manifest that has drifted from its artifacts fails.

Usage:
    python3 scripts/quantize_weights.py --model-dir DIR --desc descriptors/x.json \\
        --out-dir OUT --reference csrc/build-libs/test_quantization_format
    python3 scripts/quantize_weights.py --verify OUT/weights.manifest.json \\
        --reference csrc/build-libs/test_quantization_format
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

FORMAT = {
    "name": "int4_symmetric_group_bf16_scale",
    "abi_version": 1,
    "group": 128,
    "qmin": -7,
    "qmax": 7,
    "invalid_code": -8,
    "zero_point": 0,
    "packed_dtype": "u8",
    "scale_dtype": "bf16",
    "packing": "two_twos_complement_nibbles_per_byte_lower_k_index_in_low_nibble",
    "arithmetic": "csrc/linear_weight.cpp (the Q0 reference)",
}
MANIFEST_VERSION = 1
CONVERTER = {"name": "scripts/quantize_weights.py", "version": 1}
# The plan's initial scope: dense FFN gate/up/down only, BF16 elsewhere.
DEFAULT_ROLES = ("mlpGate", "mlpUp", "mlpDown")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_index(model_dir):
    """name -> (file, dtype, shape, start, stop) over every shard in the directory."""
    index = {}
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, name)
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        for tensor, info in header.items():
            if tensor == "__metadata__":
                continue
            start, stop = info["data_offsets"]
            if tensor in index:
                raise SystemExit(f"quantize_weights: {tensor} appears in two shards")
            index[tensor] = (path, info["dtype"], list(info["shape"]), data_start + start,
                             data_start + stop)
    return index


def read_tensor_f32(entry):
    """Read a BF16 tensor as FP32. BF16 -> FP32 is exact, and the plan's quantizer consumes
    FP32 source values, so this is the widened value and not a rounded one."""
    path, dtype, shape, start, stop = entry
    if dtype != "BF16":
        raise SystemExit(f"quantize_weights: only BF16 sources are admitted, got {dtype}")
    with open(path, "rb") as handle:
        handle.seek(start)
        raw = handle.read(stop - start)
    bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def layer_roles(descriptor):
    """(role index, role name, template) for the roles the descriptor names."""
    names = descriptor["role_names"]
    templates = descriptor["role_templates"]
    if len(names) != len(templates):
        raise SystemExit("quantize_weights: the descriptor's role names and templates differ")
    return list(enumerate(zip(names, templates)))


def admitted_instances(descriptor, roles):
    """Every (layer, role index, role name, tensor name, shape) the converter will quantize:
    the requested roles on the layers whose feed-forward is dense."""
    wanted = set(roles)
    out = []
    for layer in range(descriptor["num_layers"]):
        if descriptor["layer_ffns"][layer] != "dense":
            continue
        for role_index, (name, template) in layer_roles(descriptor):
            if name not in wanted:
                continue
            tensor = template % layer if "%d" in template else template
            out.append((layer, role_index, name, tensor))
    return out


def quantize_with_reference(reference, tensor, workdir):
    """Hand one tensor to the Q0 reference and return its payload and scales."""
    n, k = int(tensor.shape[0]), int(tensor.shape[1])
    in_path = os.path.join(workdir, "in.f32")
    packed_path = os.path.join(workdir, "out.packed")
    scales_path = os.path.join(workdir, "out.scales")
    with open(in_path, "wb") as handle:
        handle.write(np.ascontiguousarray(tensor, dtype=np.float32).tobytes())
    proc = subprocess.run([reference, "--quantize", in_path, packed_path, scales_path,
                           str(n), str(k)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"quantize_weights: the reference refused [{n}, {k}]: "
                         f"{proc.stdout}{proc.stderr}")
    with open(packed_path, "rb") as handle:
        packed = handle.read()
    with open(scales_path, "rb") as handle:
        scales = handle.read()
    return packed, scales


def convert(args):
    descriptor = json.load(open(args.desc))
    os.makedirs(args.out_dir, exist_ok=True)
    artifact_dir = os.path.join(args.out_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    sources = [os.path.join(args.model_dir, name)
               for name in sorted(os.listdir(args.model_dir))
               if name.endswith(".safetensors")]
    before = {os.path.basename(p): (os.path.getsize(p), sha256_file(p)) for p in sources}
    index = read_safetensors_index(args.model_dir)

    instances = admitted_instances(descriptor, args.roles)
    if args.limit_layers is not None:
        instances = [i for i in instances if i[0] < args.limit_layers]

    entries = []
    error_max = 0.0
    error_sq = 0.0
    error_count = 0
    with tempfile.TemporaryDirectory() as workdir:
        for layer, role_index, role_name, tensor_name in instances:
            if tensor_name not in index:
                raise SystemExit(f"quantize_weights: {tensor_name} is not in the checkpoint")
            source = read_tensor_f32(index[tensor_name])
            if source.ndim != 2:
                raise SystemExit(f"quantize_weights: {tensor_name} is not a 2-D weight")
            packed, scales = quantize_with_reference(args.reference, source, workdir)
            n, k = int(source.shape[0]), int(source.shape[1])
            expected_packed = n * k // 2
            expected_scales = 2 * n * (k // FORMAT["group"])
            if len(packed) != expected_packed or len(scales) != expected_scales:
                raise SystemExit(f"quantize_weights: {tensor_name} produced {len(packed)}/"
                                 f"{len(scales)} bytes, expected {expected_packed}/{expected_scales}")

            stem = f"{role_name}_layer{layer}"
            packed_name = f"{stem}.packed"
            scales_name = f"{stem}.scales"
            with open(os.path.join(artifact_dir, packed_name), "wb") as handle:
                handle.write(packed)
            with open(os.path.join(artifact_dir, scales_name), "wb") as handle:
                handle.write(scales)

            # The error the *reference* implies, so the manifest carries the quantity a quality
            # gate will compare against the BF16 baseline rather than leaving it to be guessed.
            scale_values = np.frombuffer(scales, dtype=np.uint16).astype(np.uint32) << 16
            scale_f32 = scale_values.view(np.float32)
            codes = np.frombuffer(packed, dtype=np.uint8)
            low = (codes & 0x0F).astype(np.int32)
            high = (codes >> 4).astype(np.int32)
            code = np.empty((n, k), dtype=np.int32)
            code[:, 0::2] = np.where(low.reshape(n, k // 2) >= 8,
                                     low.reshape(n, k // 2) - 16, low.reshape(n, k // 2))
            code[:, 1::2] = np.where(high.reshape(n, k // 2) >= 8,
                                     high.reshape(n, k // 2) - 16, high.reshape(n, k // 2))
            scale_per_element = np.repeat(
                scale_f32.reshape(n, k // FORMAT["group"]), FORMAT["group"], axis=1)
            dequant = code.astype(np.float32) * scale_per_element
            difference = np.abs(source - dequant)
            error_max = max(error_max, float(difference.max()))
            error_sq += float((difference.astype(np.float64) ** 2).sum())
            error_count += difference.size

            entries.append({
                "role": role_name,
                "role_index": role_index,
                "layer": layer,
                "checkpoint_tensor": tensor_name,
                "logical_shape": [n, k],
                "local_shape": [n, k],
                "packed": {"file": f"artifacts/{packed_name}", "dtype": FORMAT["packed_dtype"],
                           "bytes": len(packed), "sha256": sha256_file(
                               os.path.join(artifact_dir, packed_name))},
                "scales": {"file": f"artifacts/{scales_name}", "dtype": FORMAT["scale_dtype"],
                           "count": len(scales) // 2, "bytes": len(scales),
                           "sha256": sha256_file(os.path.join(artifact_dir, scales_name))},
                "group_axis": "k",
                "group": FORMAT["group"],
            })

    # The precision map: every role of every layer, quantized or retained.
    admitted_keys = {(layer, name) for layer, _, name, _ in instances}
    precision_map = []
    for layer in range(descriptor["num_layers"]):
        for _role_index, (name, template) in layer_roles(descriptor):
            tensor = template % layer if "%d" in template else template
            precision_map.append({
                "role": name,
                "layer": layer,
                "precision": "int4" if (layer, name) in admitted_keys else "bf16",
                "checkpoint_tensor": tensor,
            })

    after = {os.path.basename(p): (os.path.getsize(p), sha256_file(p)) for p in sources}
    if before != after:
        raise SystemExit("quantize_weights: the source checkpoint changed during conversion")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "format": FORMAT,
        "converter": CONVERTER,
        "source": {
            "model_dir": os.path.abspath(args.model_dir),
            "files": [{"name": name, "bytes": size, "sha256": digest}
                      for name, (size, digest) in sorted(before.items())],
        },
        "descriptor": {"path": os.path.abspath(args.desc), "sha256": sha256_file(args.desc)},
        "quantization_error": {
            "max_abs": error_max,
            "rms": (error_sq / error_count) ** 0.5 if error_count else 0.0,
            "elements": error_count,
            "note": "measured against the source BF16 checkpoint's FP32 values, per element",
        },
        "precision_map": precision_map,
        "entries": entries,
    }
    path = os.path.join(args.out_dir, "weights.manifest.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"quantize_weights: {len(entries)} role instance(s) quantized; "
          f"block max_abs {error_max:.6g}, rms {(error_sq / error_count) ** 0.5 if error_count else 0:.6g}")
    print(f"quantize_weights: wrote {path}")
    return 0


def verify(args):
    """Re-derive the artifact's claims rather than trusting the manifest."""
    manifest = json.load(open(args.verify))
    problems = []

    def require(condition, message):
        if not condition:
            problems.append(message)

    require(manifest.get("manifest_version") == MANIFEST_VERSION,
            f"unknown manifest version {manifest.get('manifest_version')}")
    for key, value in FORMAT.items():
        require(manifest.get("format", {}).get(key) == value,
                f"the format's {key} is {manifest.get('format', {}).get(key)}, expected {value}")

    out_dir = os.path.dirname(os.path.abspath(args.verify))
    source_dir = manifest["source"]["model_dir"]
    for record in manifest["source"]["files"]:
        path = os.path.join(source_dir, record["name"])
        require(os.path.exists(path), f"the source file {record['name']} is gone")
        if os.path.exists(path):
            require(sha256_file(path) == record["sha256"],
                    f"the source file {record['name']} no longer matches its checksum")

    seen = set()
    entries = manifest["entries"]
    for entry in entries:
        key = (entry["layer"], entry["role"])
        require(key not in seen, f"{key} appears twice")
        seen.add(key)
        n, k = entry["logical_shape"]
        require(n % 8 == 0 and k % FORMAT["group"] == 0,
                f"{key} has a shape the format does not admit: {n}x{k}")
        packed = os.path.join(out_dir, entry["packed"]["file"])
        scales = os.path.join(out_dir, entry["scales"]["file"])
        require(os.path.exists(packed), f"{key} has no payload at {entry['packed']['file']}")
        require(os.path.exists(scales), f"{key} has no scales at {entry['scales']['file']}")
        if os.path.exists(packed):
            require(os.path.getsize(packed) == n * k // 2,
                    f"{key} packs to {os.path.getsize(packed)} bytes, expected {n * k // 2}")
            require(sha256_file(packed) == entry["packed"]["sha256"],
                    f"{key} payload does not match its checksum")
        if os.path.exists(scales):
            expected = 2 * n * (k // FORMAT["group"])
            require(os.path.getsize(scales) == expected,
                    f"{key} scales are {os.path.getsize(scales)} bytes, expected {expected}")
            require(sha256_file(scales) == entry["scales"]["sha256"],
                    f"{key} scales do not match their checksum")

    # The precision map must cover every role of every layer exactly once, and agree with the
    # entries about which ones are quantized.
    descriptor = json.load(open(manifest["descriptor"]["path"]))
    require(sha256_file(manifest["descriptor"]["path"]) == manifest["descriptor"]["sha256"],
            "the descriptor no longer matches its checksum")
    covered = set()
    for record in manifest["precision_map"]:
        key = (record["layer"], record["role"])
        require(key not in covered, f"the precision map lists {key} twice")
        covered.add(key)
        require((record["precision"] == "int4") == (key in seen),
                f"the precision map and the entries disagree about {key}")
    for layer in range(descriptor["num_layers"]):
        for _role_index, (name, _template) in layer_roles(descriptor):
            require((layer, name) in covered,
                    f"the precision map does not cover layer {layer}'s {name}")

    # Re-derive a sample: the artifact must be reproducible from the source by the reference.
    sample = entries[:max(1, min(2, len(entries)))]
    index = read_safetensors_index(source_dir)
    with tempfile.TemporaryDirectory() as workdir:
        for entry in sample:
            # An entry whose artifacts are missing or misplaced is already a problem above;
            # re-deriving it would only raise, and the report is what the caller needs.
            packed_path = os.path.join(out_dir, entry["packed"]["file"])
            scales_path = os.path.join(out_dir, entry["scales"]["file"])
            if not (os.path.exists(packed_path) and os.path.exists(scales_path)):
                continue
            source = read_tensor_f32(index[entry["checkpoint_tensor"]])
            packed, scales = quantize_with_reference(args.reference, source, workdir)
            with open(packed_path, "rb") as handle:
                require(handle.read() == packed,
                        f"{entry['role']}_{entry['layer']} does not re-derive from the source")
            with open(scales_path, "rb") as handle:
                require(handle.read() == scales,
                        f"{entry['role']}_{entry['layer']} scales do not re-derive from the source")

    if problems:
        for problem in problems:
            print(f"quantize_weights: verify: {problem}")
        return 1
    print(f"quantize_weights: verify: {len(entries)} entries, the precision map covers "
          f"{len(covered)} (role, layer) cells, the source is unchanged and the sampled "
          f"artifacts re-derive from it")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--desc")
    parser.add_argument("--out-dir")
    parser.add_argument("--reference", required=True,
                        help="the Q0 reference binary (test_quantization_format)")
    parser.add_argument("--roles", default=",".join(DEFAULT_ROLES),
                        help="the roles to quantize; the rest stay BF16")
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="only convert the first N layers (a smoke conversion)")
    parser.add_argument("--verify", metavar="MANIFEST",
                        help="validate an existing manifest and its artifacts instead")
    args = parser.parse_args()
    args.roles = tuple(r for r in args.roles.split(",") if r)
    if args.verify:
        return verify(args)
    if not (args.model_dir and args.desc and args.out_dir):
        parser.error("--model-dir, --desc and --out-dir are required to convert")
    return convert(args)


if __name__ == "__main__":
    sys.exit(main())
