#!/usr/bin/env python3
"""Initialization resource safety: a checkpoint that cannot be loaded must leave
nothing behind.

The engine registers every device buffer with its owner as soon as it is
allocated, so engine_create failing on a late missing or mis-shaped tensor must
report a clear error, return no handle, and release everything it had already
allocated. The test builds two tiny checkpoints (a dense one and a MoE one,
where one failed layer used to strand its fused expert buffers) in a temporary
directory, breaks them deliberately in their late layers, repeats the failing
creations, and compares device memory before and after.

  python tests/test_engine_resources.py --library csrc/build-libs/libengine.so [--device 0]

The models are a few tens of MB -- small enough to run anywhere, and never an
out-of-memory experiment.
"""

import argparse
import ctypes
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_bindings as bindings  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Memory checks: after the context and cuBLAS are warm, failing creations must
# not move the free-memory reading. The slack covers allocator fragmentation,
# and is far below the tens of megabytes a stranded buffer would show up as.
MEMORY_SLACK_BYTES = 16 * 1024 * 1024
FAILURE_ROUNDS = 4


def scaled_descriptor(source, **overrides):
    """A descriptor for the same family as `source`, scaled down."""
    with open(REPO / "descriptors" / source) as handle:
        desc = json.load(handle)
    desc.update(overrides)
    return desc


def dense_descriptor():
    return scaled_descriptor(
        "qwen3-4b.json",
        num_layers=2,
        layer_mixers=["full_attn", "full_attn"],
        layer_ffns=["dense", "dense"],
        hidden_size=256,
        intermediate_size=512,
        vocab_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        rotary_dim=64,
        max_seq_len=128,
        eos_tokens=[1],
    )


def moe_descriptor():
    return scaled_descriptor(
        "qwen3-30b-a3b.json",
        num_layers=2,
        layer_mixers=["full_attn", "full_attn"],
        layer_ffns=["moe", "moe"],
        hidden_size=512,
        intermediate_size=256,
        moe_intermediate_size=256,
        vocab_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        rotary_dim=64,
        moe_num_experts=32,
        moe_top_k=2,
        moe_num_shared_experts=0,
        moe_shared_intermediate_size=0,
        max_seq_len=128,
        eos_tokens=[1],
    )


def shape_for(role, desc):
    """The shape the engine's role table insists on (mirrors expected_shape)."""
    hidden = desc["hidden_size"]
    heads = desc["num_heads"]
    head_dim = desc["head_dim"]
    kv_heads = desc["num_kv_heads"]
    mlp = desc["intermediate_size"]
    moe_inner = desc["moe_intermediate_size"]
    q_rows = heads * head_dim * (2 if desc["attn_output_gate"] else 1)
    table = {
        "embed": [desc["vocab_size"], hidden],
        "lmHead": [desc["vocab_size"], hidden],
        "finalNorm": [hidden],
        "inputNorm": [hidden],
        "postNorm": [hidden],
        "mlpGate": [mlp, hidden],
        "mlpUp": [mlp, hidden],
        "mlpDown": [hidden, mlp],
        "attnQ": [q_rows, hidden],
        "attnK": [kv_heads * head_dim, hidden],
        "attnV": [kv_heads * head_dim, hidden],
        "attnO": [hidden, heads * head_dim],
        "attnQNorm": [head_dim],
        "attnKNorm": [head_dim],
        "moeRouter": [desc["moe_num_experts"], hidden],
        "moeExpertGate": [moe_inner, hidden],
        "moeExpertUp": [moe_inner, hidden],
        "moeExpertDown": [hidden, moe_inner],
    }
    return table[role]


def tensor_names(desc):
    """(name, role, layer, expert) for every tensor the checkpoint must carry."""
    layer_roles = {"inputNorm", "postNorm", "attnQ", "attnK", "attnV", "attnO",
                   "attnQNorm", "attnKNorm", "mlpGate", "mlpUp", "mlpDown",
                   "moeRouter", "moeExpertGate", "moeExpertUp", "moeExpertDown"}
    global_roles = {"embed", "lmHead", "finalNorm"}
    entries = []
    for role, template in zip(desc["role_names"], desc["role_templates"]):
        if role in global_roles:
            entries.append((template, role, None, None))
        elif role in layer_roles and "%d" in template:
            for layer in range(desc["num_layers"]):
                if role.startswith("moeExpert"):
                    for expert in range(desc["moe_num_experts"]):
                        name = template.replace("%d", str(layer)).replace("%e", str(expert))
                        entries.append((name, role, layer, expert))
                else:
                    entries.append((template.replace("%d", str(layer)), role, layer, None))
        elif role in layer_roles:
            raise AssertionError(f"layer role {role} has no layer placeholder: {template}")
    return entries


def to_bf16(values):
    """Round-to-nearest-even BF16 bytes; values are irrelevant to this test."""
    f32 = np.asarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype("<u2").tobytes()


def tensor_values(name, role, shape):
    if role.endswith("Norm"):
        return np.ones(shape, dtype=np.float32)
    count = int(np.prod(shape)) if shape else 1
    return (((np.arange(count, dtype=np.float32) % 7) - 3) * 0.125).reshape(shape)


def write_checkpoint(model_dir, desc, drop=(), wrong_shape=()):
    """Write one safetensors file holding every tensor, minus the broken ones."""
    drop = set(drop)
    wrong_shape = dict(wrong_shape)
    header = {}
    payload = bytearray()
    for name, role, _, _ in tensor_names(desc):
        if name in drop:
            continue
        shape = list(wrong_shape.get(name, shape_for(role, desc)))
        raw = to_bf16(tensor_values(name, role, shape))
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload += raw
    model_dir.mkdir(parents=True, exist_ok=True)
    for stale in model_dir.glob("*.safetensors"):
        stale.unlink()
    blob = json.dumps(header).encode()
    with open(model_dir / "model.safetensors", "wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        handle.write(bytes(payload))


def try_create(lib, model_dir, desc, device):
    """Create an engine, returning (handle, error message). Never raises."""
    layer_devices = bindings.contiguous_assignment(desc["num_layers"], [device])
    devices = (ctypes.c_int * 1)(device)
    layers = (ctypes.c_int * len(layer_devices))(*layer_devices)
    payload = json.dumps(desc).encode()
    handle = lib.engine_create(str(model_dir).encode(), payload, 1, devices, layers)
    error = lib.engine_last_error().decode() if not handle else ""
    return handle, error


def cuda_runtime(library_path):
    """The CUDA runtime, for the free-memory reading."""
    candidates = ["libcudart.so.12", "libcudart.so", library_path]
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError("cannot load the CUDA runtime to read device memory")


def free_bytes(runtime, device):
    import ctypes as ct
    runtime.cudaSetDevice.argtypes = [ct.c_int]
    runtime.cudaMemGetInfo.argtypes = [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
    runtime.cudaSetDevice(device)
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if status != 0:
        raise RuntimeError(f"cudaMemGetInfo failed with {status}")
    return free.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="csrc/build-libs/libengine.so")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cudart", default=None, help="path to libcudart, if not on the loader path")
    args = parser.parse_args()

    lib = bindings.bind(ctypes.CDLL(args.library))
    runtime = cuda_runtime(args.cudart)
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        print("test_engine_resources: SKIP (no CUDA device visible)")
        return 0
    failures = []

    def check(condition, what):
        status = "PASS" if condition else "FAIL"
        print(f"{status}: {what}")
        if not condition:
            failures.append(what)

    dense = dense_descriptor()
    moe = moe_descriptor()

    with tempfile.TemporaryDirectory(prefix="engine-resources-") as work:
        root = Path(work)
        dense_dir = root / "dense"
        moe_dir = root / "moe"

        # --- 1) A whole checkpoint builds, so a later failure is the only variable.
        write_checkpoint(dense_dir, dense)
        handle, error = try_create(lib, dense_dir, dense, args.device)
        check(handle is not None, f"a complete dense checkpoint builds ({error})")
        if handle:
            lib.engine_destroy(handle)
        write_checkpoint(moe_dir, moe)
        handle, error = try_create(lib, moe_dir, moe, args.device)
        check(handle is not None, f"a complete MoE checkpoint builds ({error})")
        if handle:
            lib.engine_destroy(handle)

        # --- 2) Failures are reported, not absorbed. The late tensors are the
        #        ones that leave earlier allocations behind if the engine
        #        registers them late.
        late_dense = f"model.layers.{dense['num_layers'] - 1}.mlp.down_proj.weight"
        late_expert = (f"model.layers.{moe['num_layers'] - 1}.mlp.experts."
                       f"{moe['moe_num_experts'] - 1}.down_proj.weight")

        write_checkpoint(dense_dir, dense, drop=[late_dense])
        handle, error = try_create(lib, dense_dir, dense, args.device)
        check(handle is None, "a missing late dense tensor yields no handle")
        check(late_dense in error, f"the error names the missing tensor ({error})")

        write_checkpoint(dense_dir, dense, wrong_shape={late_dense: [3, 5]})
        handle, error = try_create(lib, dense_dir, dense, args.device)
        check(handle is None, "a mis-shaped late dense tensor yields no handle")
        check("Unexpected shape" in error, f"the error explains the shape ({error})")

        write_checkpoint(moe_dir, moe, drop=[late_expert])
        handle, error = try_create(lib, moe_dir, moe, args.device)
        check(handle is None, "a missing late expert tensor yields no handle")
        check(late_expert in error, f"the error names the missing expert tensor ({error})")

        write_checkpoint(moe_dir, moe, wrong_shape={late_expert: [3, 5]})
        handle, error = try_create(lib, moe_dir, moe, args.device)
        check(handle is None, "a mis-shaped late expert tensor yields no handle")

        # --- 3) Repeating the failures must not accumulate device memory. The
        #        context and cuBLAS are warm by now, so the reading is stable.
        baseline = free_bytes(runtime, args.device)
        for _ in range(FAILURE_ROUNDS):
            for model_dir, desc, kwargs in (
                (dense_dir, dense, {"drop": [late_dense]}),
                (dense_dir, dense, {"wrong_shape": {late_dense: [3, 5]}}),
                (moe_dir, moe, {"drop": [late_expert]}),
                (moe_dir, moe, {"wrong_shape": {late_expert: [3, 5]}}),
            ):
                write_checkpoint(model_dir, desc, **kwargs)
                handle, _ = try_create(lib, model_dir, desc, args.device)
                if handle:
                    lib.engine_destroy(handle)
                    check(False, "a broken checkpoint must not create an engine")
        after = free_bytes(runtime, args.device)
        drift = baseline - after
        check(drift <= MEMORY_SLACK_BYTES,
              f"{FAILURE_ROUNDS} failing rounds moved free memory by "
              f"{drift / (1024 * 1024):.1f} MiB (slack {MEMORY_SLACK_BYTES // (1024 * 1024)} MiB)")

        # --- 4) A valid model still builds once the failures stop.
        write_checkpoint(dense_dir, dense)
        handle, error = try_create(lib, dense_dir, dense, args.device)
        check(handle is not None, f"a valid model builds after the failures ({error})")
        if handle:
            lib.engine_destroy(handle)
        write_checkpoint(moe_dir, moe)
        handle, error = try_create(lib, moe_dir, moe, args.device)
        check(handle is not None, f"a valid MoE model builds after the failures ({error})")
        if handle:
            lib.engine_destroy(handle)

    if failures:
        print(f"test_engine_resources: FAIL ({len(failures)} check(s))")
        return 1
    print("test_engine_resources: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
