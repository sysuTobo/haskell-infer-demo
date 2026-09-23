"""Per-stage MLA probe: engine taps vs the transformers oracle's hooks.

Runs the DeepSeek checkpoint through transformers with forward hooks on layer 0's
MLA submodules, dumps the same stages the engine's INFER_TAP_* taps produce, and
compares them stage by stage so the first diverging stage localizes the bug.

  python tests/ds_probe.py --model-dir /path/DeepSeek-V2-Lite --engine-lib csrc/build-libs/libengine.so \
      --desc descriptors/deepseek-v2-lite.json --tap-dir /tmp/mla-eng
"""

import argparse
import ctypes
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_bindings import bind, create_engine, load_descriptor, ptr  # noqa: E402

# Engine tap kind -> oracle stage name.
STAGES = {
    "mla_in": "input_layernorm",
    "mla_q_pre": "q_proj",
    "mla_kv_pre": "kv_a_proj",
    "mla_latent": "kv_a_layernorm",
    "mla_knope_v": "kv_b_proj",
}


def oracle_stages(model_dir, prompt, layer=0):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    block = model.model.layers[layer]
    attn = block.self_attn
    wanted = {
        "input_layernorm": block.input_layernorm,
        "q_proj": attn.q_proj,
        "kv_a_proj": attn.kv_a_proj_with_mqa,
        "kv_a_layernorm": attn.kv_a_layernorm,
        "kv_b_proj": attn.kv_b_proj,
        "attn_block": attn,
    }
    captured = {}

    def hook(name):
        def fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            array = tensor.detach().float().cpu().numpy().reshape(-1, tensor.shape[-1])
            captured[name] = array
            # Save immediately: a late kernel failure must not lose the stages.
            np.save(f"/tmp/oracle-{name}.npy", array)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in wanted.items()]
    with torch.inference_mode():
        model(input_ids=ids.to(model.device), use_cache=True)
    for h in handles:
        h.remove()
    return ids[0].tolist(), captured


def engine_stages(args, ids):
    os.environ["INFER_TAP_LAYERS"] = "0"
    os.environ["INFER_TAP_DIR"] = args.tap_dir
    os.makedirs(args.tap_dir, exist_ok=True)
    lib = bind(ctypes.CDLL(args.engine_lib))
    desc = load_descriptor(args.desc, max_seq_len=32)
    engine, _ = create_engine(lib, args.model_dir, desc, [int(x) for x in args.devices.split(",")])
    try:
        logits = np.empty(lib.engine_vocab_size(engine), dtype=np.float32)
        tokens = np.ascontiguousarray(ids, dtype=np.int64)
        rc = lib.engine_prefill(engine, ptr(tokens), len(tokens), ptr(logits))
        assert rc == 0, lib.engine_last_error().decode()
    finally:
        lib.engine_destroy(engine)


def load_tap(path, tokens):
    # tap_dump_rows writes raw float32 [tokens, cols].
    if not os.path.exists(path):
        return None
    flat = np.fromfile(path, dtype=np.float32)
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engine-lib", required=True)
    parser.add_argument("--desc", default="descriptors/deepseek-v2-lite.json")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--tap-dir", default="/tmp/mla-eng")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--mode", choices=["oracle", "engine", "compare"], default="compare")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()[: args.tokens]
    np.save("/tmp/probe-ids.npy", np.array(ids))
    print(f"prompt ids: {ids}", flush=True)

    if args.mode == "oracle":
        ids, captured = oracle_stages(args.model_dir, args.prompt)
        print("oracle stages:", {k: v.shape for k, v in captured.items()}, flush=True)
        return
    if args.mode == "engine":
        ids = np.load("/tmp/probe-ids.npy").tolist()
        engine_stages(args, ids)
        print(f"engine taps in {args.tap_dir}", flush=True)
        return

    # compare
    ids = np.load("/tmp/probe-ids.npy").tolist()
    tok = len(ids)
    for tap_kind, oracle_name in STAGES.items():
        path = os.path.join(args.tap_dir, f"{tap_kind}_00_seq0_tok{tok}.f32")
        got = load_tap(path, tok)
        if got is None:
            print(f"{tap_kind:12s} (no tap file)")
            continue
        oracle_path = f"/tmp/oracle-{oracle_name}.npy"
        if not os.path.exists(oracle_path):
            print(f"{tap_kind:12s} got {got.size} values; no oracle dump")
            continue
        want = np.load(oracle_path).reshape(-1).astype(np.float64)
        # The oracle may run a longer prompt; compare the engine's prefix.
        if got.size <= want.size:
            want = want[: got.size]
        got = got.astype(np.float64)
        if got.size != want.size:
            print(f"{tap_kind:12s} size mismatch: engine {got.size} vs oracle {want.size}")
            continue
        diff = np.abs(got - want)
        print(f"{tap_kind:12s} max_abs={diff.max():.4g} rms={np.sqrt((diff ** 2).mean()):.4g} "
              f"scale={np.abs(want).mean():.4g} (oracle {oracle_name})")


if __name__ == "__main__":
    main()
