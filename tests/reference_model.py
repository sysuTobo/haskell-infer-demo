import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
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
