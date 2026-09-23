"""Does the oracle itself agree across prefill chunkings? (MLA boundary check)

Runs the transformers reference twice on the same 129 tokens: once as one call
(the library chunks internally) and once as 64 + 65, then compares the logits.
An engine/engine difference of the same size means the model is boundary
sensitive, not that the engine loses state.

  python tests/ds_boundary_oracle.py --model-dir /path/DeepSeek-V2-Lite
"""

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokens", type=int, default=129)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids[0]
    long_ids = torch.cat([ids] * (args.tokens // len(ids) + 1))[: args.tokens].unsqueeze(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()

    def prefill(chunks):
        cache = None
        logits = None
        with torch.inference_mode():
            for chunk in chunks:
                result = model(input_ids=chunk.to(model.device), past_key_values=cache,
                               use_cache=True)
                cache = result.past_key_values
                logits = result.logits[0, -1].float()
        return logits.cpu().numpy()

    whole = prefill([long_ids])
    split = prefill([long_ids[:, :64], long_ids[:, 64:]])
    rms = float(np.sqrt(np.mean((whole - split) ** 2)))
    print(f"oracle 129 one-shot vs 64+65: rms={rms:.4g} (std={whole.std():.3g}) "
          f"top1={int(whole.argmax())}/{int(split.argmax())}", flush=True)


if __name__ == "__main__":
    main()
