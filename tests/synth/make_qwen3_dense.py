"""Build a tiny *dense* Qwen3-Next checkpoint for the Stage-5 SFT bring-up.

The existing synthetic checkpoint (`make_qwen3_next.py`) keeps the real model's
sparse FFN, which the first trainer allowlist deliberately excludes: MoE routing,
per-expert GEMMs and the routed combine are outside the dense/dense-hybrid path
(`csrc/regions.c`, the `moe_*` exclusions). Stage 5 has to bring SFT up on a model
whose every region has a backward, so this generator produces the same family with
a **dense** MLP in every layer (`mlp_only_layers`), which is exactly the Qwen3-Next
design's own dense-layer mode rather than a new architecture.

`--mixers` selects what the model trains:

  attn    every layer is full attention (`full_attention_interval=1`), so the step
          exercises attention + dense MLP + the norms and nothing else. This is the
          smallest model that still has the real attention path (qk-norm, partial
          RoPE, the output gate, GQA) and is Stage 5's first SFT target.
  hybrid  the real 4:1 mix (three GDN layers then a full-attention layer), which
          adds the GDN mixer's backward.

Both variants are dense in the FFN, so the trainer's `train_forward` step has a
backward for every region it visits. The reference for the *gradients* is the
torch training run in `tests/test_sft.py`, which re-builds the same model from
this checkpoint and steps it with `torch.optim.AdamW` on the same data.

  python tests/synth/make_qwen3_dense.py --out-dir DIR --mixers attn
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def tiny_config(layers: int):
    """A ~1M-parameter dense Qwen3: the family the adapter already maps, with the
    attention shapes the kernels dispatch on (head dim 128, GQA, qk-norm, partial RoPE)
    and **tied embeddings**, which is the case the plan's gate names explicitly.

    Qwen3-Next would be the closer relative of the deployment model, but its
    transformers cache requires at least one linear-attention layer, so a
    full-attention-only Next config cannot even run a reference forward here; a dense
    Qwen3 has no such constraint and exercises the same attention and MLP regions.
    """
    return Qwen3Config(
        vocab_size=1024,
        bos_token_id=1,
        eos_token_id=0,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.25,
        rope_theta=1e6,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        max_position_embeddings=512,
        attention_dropout=0.0,
        attn_implementation="eager",
    )


def structural_report(model, config, out_dir):
    """Print what the adapter must encode, read from the saved checkpoint."""
    layer_types = list(getattr(config, "layer_types", []))
    shard = sorted(Path(out_dir).glob("*.safetensors"))[0]
    with open(shard, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    names = sorted(header)
    has_lm_head = any(name == "lm_head.weight" for name in names)
    lines = [
        f"model_type            : {config.model_type}",
        f"layer_types           : {layer_types}",
        f"tied embeddings       : {config.tie_word_embeddings}",
        f"checkpoint has lm_head: {has_lm_head} (a tied model saves only the embedding)",
        f"qk-norm               : "
        f"{any(name.endswith('self_attn.q_norm.weight') for name in names)}",
        f"saved tensors         : {len(names)}",
    ]
    lines += [f"  {name}" for name in names]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    config = tiny_config(args.layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3ForCausalLM(config).eval()
    # The engine loads BF16 weights only, and a training run has to start from the
    # same numbers, so the checkpoint is BF16.
    model = model.to(device=device, dtype=torch.bfloat16)
    model.save_pretrained(out)
    print(structural_report(model, config, out), flush=True)
    (out / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}) + "\n")

    # A forward reference at the initial weights, so the engine's training forward
    # can first be checked against the architecture's own math before any step.
    prompts = ([1, 2, 3, 4, 5], [7, 8, 9])
    arrays = {}
    with torch.inference_mode():
        for case_id, prompt in enumerate(prompts):
            ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
            result = model(input_ids=ids)
            arrays[f"prompt_{case_id}"] = np.array(prompt, dtype=np.int64)
            arrays[f"logits_{case_id}"] = result.logits[0, -1].float().cpu().numpy()
    np.savez(Path(str(out) + "-ref.npz"), **arrays)
    total = sum(p.numel() for p in model.parameters())
    print(f"saved dense Qwen3 checkpoint ({total/1e6:.2f}M params) to {out}", flush=True)


if __name__ == "__main__":
    main()
