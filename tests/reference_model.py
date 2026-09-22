import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_type_of(model_dir):
    config = json.loads((Path(model_dir) / "config.json").read_text())
    text_config = config.get("text_config", config)
    return text_config.get("model_type", config.get("model_type", ""))


def patch_gdn_fallbacks(model):
    """Qwen3.5's GatedDeltaNet needs the pytorch chunk/recurrent implementations:
    the fused kernels are optional and absent here, and without the patch the
    remote-code fallback path is what the engine is compared against."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(0)
    family = model_type_of(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    if family.startswith("qwen3_5"):
        patch_gdn_fallbacks(model)
    elif family.startswith("qwen3") or family == "mixtral":
        pass  # dense attention and/or sparse FFN: nothing to patch
    else:
        print(f"warning: no known adapter for model_type {family!r}; "
              "using the transformers implementation as-is", flush=True)
    arrays = {}
    cases = []
    with torch.inference_mode():
        for case_id, prompt in enumerate(("The capital of France is", "1 + 1 =")):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            arrays[f"prompt_{case_id}"] = ids.cpu().numpy()[0]
            cache = None
            generated = []
            for step in range(args.max_tokens):
                result = model(input_ids=ids, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits = result.logits[0, -1].float()
                arrays[f"logits_{case_id}_{step}"] = logits.cpu().numpy()
                token = int(logits.argmax())
                generated.append(token)
                ids = torch.tensor([[token]], device=model.device)
            arrays[f"tokens_{case_id}"] = np.array(generated, dtype=np.int64)
            text = tokenizer.decode(generated)
            cases.append({"prompt": prompt, "tokens": generated, "text": text})
            print(json.dumps(cases[-1]), flush=True)
    output = Path(args.output)
    np.savez(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
