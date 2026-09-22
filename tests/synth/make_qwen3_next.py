"""Build a tiny Qwen3-Next checkpoint plus its PyTorch reference.

Qwen3-Next-80B-A3B is 160 GB in BF16, which does not fit the demo's 2x A40, so
the family is exercised with a scaled-down checkpoint that keeps the shapes the
kernels care about: GDN head dims (128) and head counts (16 keys / 32 values,
like the real model), attention head dim 256, and a sparse FFN with a shared
expert. The reference logits come from the same transformers implementation, so
the engine is compared against the architecture's own math rather than a second
hand-written model.

  python tests/synth/make_qwen3_next.py --out-dir DIR [--steps 8]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3NextConfig, Qwen3NextForCausalLM


def tiny_config():
    return Qwen3NextConfig(
        vocab_size=1024,
        bos_token_id=1,
        eos_token_id=0,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        rms_norm_eps=1e-6,
        # GDN: the kernels fix the head dim at 128; the head counts match the real
        # model (16 keys / 32 values), which is the layout the AOT set covers.
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        full_attention_interval=4,
        # Sparse FFN with a shared expert, small enough to run on two GPUs.
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        tie_word_embeddings=False,
        max_position_embeddings=512,
        attn_implementation="eager",
    )


def structural_report(model, config, out_dir):
    """Print what the adapter must encode: attention dims and the saved tensor
    names. Read from the saved checkpoint because that is what the engine sees
    (transformers 5.x stores fused modules that may serialize either way)."""
    import re
    import struct
    attn = model.model.layers[3].self_attn
    shard = sorted(Path(out_dir).glob("*.safetensors"))[0]
    with open(shard, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    patterns = sorted({re.sub(r"layers\.\d+\.experts\.\d+\.", "layers.N.experts.E.", key)
                       for key in header})
    lines = [
        f"model_type            : {config.model_type}",
        f"layer_types           : {config.layer_types}",
        f"q_proj out_features   : {attn.q_proj.out_features} "
        f"(plain {config.num_attention_heads * config.head_dim}, "
        f"gated {2 * config.num_attention_heads * config.head_dim})",
        f"o_proj in_features    : {attn.o_proj.in_features}",
        f"q_norm                : {tuple(attn.q_norm.weight.shape)}",
        f"saved tensors         : {len(header)}",
    ]
    lines += [f"  {name}" for name in patterns]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    config = tiny_config()
    # The venv's causal-conv1d fast path needs CUDA tensors, so the tiny model
    # lives on the GPU as well.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3NextForCausalLM(config).eval()
    # The engine loads BF16 weights only, and the reference should use the same
    # dtype as the engine for a meaningful comparison.
    model = model.to(device=device, dtype=torch.bfloat16)
    model.save_pretrained(out)
    print(structural_report(model, config, out), flush=True)
    # A tokenizer is not needed by the engine test (it drives token ids), but the
    # adapter reads tokenizer_config.json when present, so write a minimal one.
    (out / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}) + "\n")
    print(f"saved checkpoint to {out}", flush=True)

    prompts = ([1, 2, 3, 4, 5], [7, 8, 9])
    arrays = {}
    report = []
    with torch.inference_mode():
        for case_id, prompt in enumerate(prompts):
            ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
            arrays[f"prompt_{case_id}"] = np.array(prompt, dtype=np.int64)
            cache = None
            generated = []
            for step in range(args.steps):
                result = model(input_ids=ids, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits = result.logits[0, -1].float()
                assert torch.isfinite(logits).all(), f"reference produced non-finite logits at {step}"
                arrays[f"logits_{case_id}_{step}"] = logits.cpu().numpy()
                token = int(logits.argmax())
                generated.append(token)
                ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
            arrays[f"tokens_{case_id}"] = np.array(generated, dtype=np.int64)
            report.append({"prompt": prompt, "tokens": generated})
            print(json.dumps(report[-1]), flush=True)

    np.savez(Path(str(out) + "-ref.npz"), **arrays)
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote reference for {len(prompts)} prompts x {args.steps} steps", flush=True)


if __name__ == "__main__":
    main()
