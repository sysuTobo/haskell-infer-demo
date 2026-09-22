"""Layer-tap utilities: dump hidden states from the transformers reference, and
compare them against the engine's taps (`INFER_TAP_LAYERS` / `INFER_TAP_DIR`).

  python tests/synth/taps.py dump --model-dir DIR --out DIR --prompt 1,2,3,4,5
  python tests/synth/taps.py compare --ours DIR --theirs DIR

Both sides write raw float32 [tokens, hidden] as layer_<i>_seq<s>_tok<n>.f32, so
the first layer whose numbers diverge points at the sublayer to fix.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def dump(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    prompt = [int(x) for x in args.prompt.split(",")]
    tokens = len(prompt)
    captured = {}

    def hook(index):
        def run(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:  # (batch, tokens, hidden)
                hidden = hidden[0]
            captured[index] = hidden.detach().float().cpu().numpy()

        return run

    layers = model.model.layers
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    with torch.inference_mode():
        model(input_ids=torch.tensor([prompt], dtype=torch.long, device=model.device))
    for handle in handles:
        handle.remove()

    for index, hidden in sorted(captured.items()):
        assert hidden.shape == (tokens, model.config.hidden_size), hidden.shape
        path = out / f"layer_{index:02d}_seq0_tok{tokens}.f32"
        path.write_bytes(hidden.astype(np.float32).tobytes())
        print(f"wrote {path} rms={float(np.sqrt((hidden ** 2).mean())):.5f}")
    print(f"dumped {len(captured)} layers for prompt {prompt}")


def compare(args):
    ours, theirs = Path(args.ours), Path(args.theirs)
    names = sorted({p.name for p in ours.glob("layer_*.f32")} &
                   {p.name for p in theirs.glob("layer_*.f32")})
    if not names:
        print("no overlapping tap files")
        return 1
    worst = None
    for name in names:
        a = np.fromfile(ours / name, dtype=np.float32)
        b = np.fromfile(theirs / name, dtype=np.float32)
        if a.shape != b.shape:
            print(f"{name}: shape {a.shape} vs {b.shape}")
            return 1
        diff = np.abs(a - b)
        rms = float(np.sqrt((diff ** 2).mean()))
        # Scale-relative measure: a random small model has small hidden states.
        scale = float(np.sqrt((b ** 2).mean())) or 1.0
        print(f"{name}: max_abs={diff.max():.5f} rms={rms:.5f} "
              f"(reference rms={scale:.5f}, ratio={rms / scale:.2%})")
        if worst is None and rms / scale > 0.02:
            worst = name
    if worst is None:
        print("all layers agree within 2% of the reference scale")
        return 0
    print(f"first layer diverging beyond 2%: {worst}")
    return 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model-dir", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--prompt", default="1,2,3,4,5")
    d.set_defaults(func=dump)
    c = sub.add_parser("compare")
    c.add_argument("--ours", required=True)
    c.add_argument("--theirs", required=True)
    c.set_defaults(func=compare)
    args = parser.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
