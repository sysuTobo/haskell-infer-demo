"""Layer-tap utilities: dump hidden states from the transformers reference, and
compare them against the engine's taps (`INFER_TAP_LAYERS` / `INFER_TAP_DIR`).

  python tests/synth/taps.py dump --model-dir DIR --out DIR --prompt 1,2,3,4,5
  python tests/synth/taps.py stages --model-dir DIR --out DIR --prompt 1,2,3,4,5
  python tests/synth/taps.py compare --ours DIR --theirs DIR

All sides write raw float32 token-major rows as <kind>_<i>_seq<s>_tok<n>.f32,
with kind `layer` (residual stream after the decoder layer), `mixer` (the token
mixer's output, before its residual add) or `ffn` (the feed-forward's output),
so the first tap whose numbers diverge points at the sub-layer to fix.

`stages` additionally reimplements the GDN sub-layer stage by stage (kinds
`gdn_in`, `gdn_conv`, `gdn_q`, `gdn_k`, `gdn_v`, `gdn_z`, `gdn_a`, `gdn_b`,
`gdn_delta`, `gdn_gated`) from the checkpoint weights, following the reference's
`Qwen3NextGatedDeltaNet.forward`; it asserts its own final stage against the
model's `linear_attn` output before writing anything, because a staged
reimplementation is only trustworthy once it reproduces the reference.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Order in which a layer computes them, so `compare` reports the first stage
# that diverges rather than the first name in the directory listing.
KINDS = ("gdn_in", "gdn_conv", "gdn_q", "gdn_k", "gdn_v", "gdn_z", "gdn_a",
         "gdn_b", "gdn_delta", "gdn_gated", "mixer", "ffn", "layer")
TAP_NAME = re.compile(r"^(" + "|".join(KINDS) + r")_(\d+)_seq\d+_tok\d+\.f32$")


def load_model(model_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()


def write_tap(out, kind, index, tokens, value):
    """value: [tokens, cols] tensor or array, float32 on disk."""
    array = np.asarray(value)
    assert array.shape[0] == tokens, array.shape
    path = out / f"{kind}_{index:02d}_seq0_tok{tokens}.f32"
    path.write_bytes(array.reshape(tokens, -1).astype(np.float32).tobytes())
    print(f"wrote {path} shape={array.reshape(tokens, -1).shape} "
          f"rms={float(np.sqrt((array.astype(np.float32) ** 2).mean())):.5f}")


def dump(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir)
    prompt = [int(x) for x in args.prompt.split(",")]
    tokens = len(prompt)
    captured = {}

    def hook(name):
        def run(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:  # (batch, tokens, hidden)
                hidden = hidden[0]
            captured[name] = hidden.detach().float().cpu().numpy()

        return run

    layers = model.model.layers
    handles = []
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(f"layer_{index:02d}")))
        # GDN layers mix tokens with `linear_attn`, attention layers with `self_attn`.
        mixer = getattr(layer, "linear_attn", None) or layer.self_attn
        handles.append(mixer.register_forward_hook(hook(f"mixer_{index:02d}")))
        handles.append(layer.mlp.register_forward_hook(hook(f"ffn_{index:02d}")))
    with torch.inference_mode():
        model(input_ids=torch.tensor([prompt], dtype=torch.long, device=model.device))
    for handle in handles:
        handle.remove()

    for name, hidden in sorted(captured.items()):
        assert hidden.shape == (tokens, model.config.hidden_size), hidden.shape
        path = out / f"{name}_seq0_tok{tokens}.f32"
        path.write_bytes(hidden.astype(np.float32).tobytes())
        print(f"wrote {path} rms={float(np.sqrt((hidden ** 2).mean())):.5f}")
    print(f"dumped {len(captured)} taps for prompt {prompt}")


def gdn_stages(gdn, hidden, rule, state=None):
    """Rerun one GDN sub-layer stage by stage on its checkpoint weights.

    Follows `Qwen3NextGatedDeltaNet.forward` exactly, including the fused
    in_proj_qkvz/in_proj_ba layout it derives with `fix_query_key_value_ordering`
    (per-key-head [q, k, v, z] blocks), and returns the stages the engine taps.
    `state` holds the reference's cached conv/recurrent state for a decode step.
    """
    tokens = hidden.shape[1]
    hk, hv = gdn.head_k_dim, gdn.head_v_dim
    nk, nv = gdn.num_k_heads, gdn.num_v_heads
    repeat = nv // nk

    qkvz = gdn.in_proj_qkvz(hidden).view(1, tokens, nk, 2 * hk + 2 * hv * repeat)
    ba = gdn.in_proj_ba(hidden).view(1, tokens, nk, 2 * repeat)
    q, k, v, z = torch.split(qkvz, [hk, hk, hv * repeat, hv * repeat], dim=3)
    b, a = torch.split(ba, [repeat, repeat], dim=3)
    v = v.reshape(1, tokens, nv, hv)
    z = z.reshape(1, tokens, nv, hv)
    b = b.reshape(1, tokens, nv)
    a = a.reshape(1, tokens, nv)
    q, k = q.reshape(1, tokens, -1), k.reshape(1, tokens, -1)

    mixed = torch.cat((q, k, v.reshape(1, tokens, -1)), dim=-1)
    if state is None:
        # The reference convolves in (B, D, T), left-padded, and silus the result.
        mixed = F.pad(mixed.transpose(1, 2), (gdn.conv_kernel_size - 1, 0))
        conv = F.conv1d(mixed, gdn.conv1d.weight, gdn.conv1d.bias, groups=gdn.conv_dim)
        conv = F.silu(conv).transpose(1, 2).to(q.dtype)
    else:
        from transformers.models.qwen3_next.modeling_qwen3_next import torch_causal_conv1d_update

        conv = torch_causal_conv1d_update(
            mixed.transpose(1, 2), state["conv"].clone(), gdn.conv1d.weight.squeeze(1),
            gdn.conv1d.bias, gdn.activation).transpose(1, 2)

    stages = {"gdn_in": hidden[0], "gdn_conv": conv[0]}
    query, key, value = torch.split(conv, [nk * hk, nk * hk, nv * hv], dim=-1)
    query = query.reshape(1, tokens, nk, hk)
    key = key.reshape(1, tokens, nk, hk)
    value = value.reshape(1, tokens, nv, hv)

    # What the reference's chunk rule normalizes in kernel, for the taps.
    from transformers.models.qwen3_next.modeling_qwen3_next import l2norm

    norm_q = l2norm(query.float(), dim=-1, eps=1e-6).to(query.dtype)
    norm_k = l2norm(key.float(), dim=-1, eps=1e-6).to(key.dtype)
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
        norm_q = norm_q.repeat_interleave(repeat, dim=2)
        norm_k = norm_k.repeat_interleave(repeat, dim=2)
    stages["gdn_q"] = norm_q.reshape(tokens, -1)
    stages["gdn_k"] = norm_k.reshape(tokens, -1)
    stages["gdn_v"] = value[0]
    stages["gdn_z"] = z[0]
    stages["gdn_a"] = a[0]
    stages["gdn_b"] = b[0]

    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    if state is None:
        core, _ = rule(query, key, value, g=g, beta=beta, initial_state=None,
                       output_final_state=False, use_qk_l2norm_in_kernel=True)
    else:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            torch_recurrent_gated_delta_rule,
        )

        core, _ = torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, state["recurrent"].clone(), True,
            use_qk_l2norm_in_kernel=True)
    stages["gdn_delta"] = core[0]

    gated = gdn.norm(core.reshape(-1, hv), z.reshape(-1, hv)).reshape(1, tokens, -1)
    stages["gdn_gated"] = gated[0]
    return stages, gdn.out_proj(gated)


def stages(args):
    from transformers.models.qwen3_next import modeling_qwen3_next as reference

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir)
    # The reference's own eager chunk rule: the same math its fast path uses, but
    # without the triton autotuner (which re-benchmarks on this staged call).
    rule = reference.torch_chunk_gated_delta_rule
    print(f"delta rule: {rule.__module__}.{rule.__name__}")

    captured = {}

    def capture(index):
        def run(_module, inputs, kwargs, output):
            hidden = kwargs["hidden_states"] if kwargs else inputs[0]
            out = output[0] if isinstance(output, tuple) else output
            captured[index] = {"hidden": hidden.detach(),
                               "out": out.detach(),
                               "state": captured.get(index, {}).get("state")}

        return run

    def capture_state(index):
        """Snapshot the incoming conv/recurrent state (what a decode step uses)."""
        def run(_module, _inputs, kwargs):
            cache = kwargs.get("cache_params")
            # Empty during the prefill pass, filled in by it for the decode steps.
            if cache is not None and cache.conv_states[index] is not None:
                captured.setdefault(index, {})["state"] = {
                    "conv": cache.conv_states[index].clone(),
                    "recurrent": cache.recurrent_states[index].clone(),
                }

        return run

    gdn_layers = sorted(index for index, layer in enumerate(model.model.layers)
                        if getattr(layer, "linear_attn", None) is not None)
    handles = []
    for index in gdn_layers:
        mixer = model.model.layers[index].linear_attn
        handles.append(mixer.register_forward_hook(capture(index), with_kwargs=True))
        handles.append(mixer.register_forward_pre_hook(capture_state(index), with_kwargs=True))

    forced = [int(x) for x in args.decode.split(",")] if args.decode else []
    with torch.inference_mode():
        ids = torch.tensor([[int(x) for x in args.prompt.split(",")]], dtype=torch.long,
                           device=model.device)
        cache = None
        for step in range(args.steps):
            captured.clear()
            tokens = ids.shape[1]
            result = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            for index in gdn_layers:
                gdn = model.model.layers[index].linear_attn
                entry = captured[index]
                state = entry.get("state")
                stage_values, final = gdn_stages(gdn, entry["hidden"], rule, state)
                diff = (final - entry["out"]).float()
                scale = float(entry["out"].float().pow(2).mean().sqrt())
                rms = float(diff.pow(2).mean().sqrt())
                cos = float(F.cosine_similarity(final.float().reshape(-1),
                                                entry["out"].float().reshape(-1), dim=0))
                stage = "decode" if state is not None else "prefill"
                print(f"layer {index} {stage} step {step}: staged out_proj vs linear_attn hook: "
                      f"rms={rms:.6f} (reference rms={scale:.6f}, ratio={rms / scale:.2%}) "
                      f"cos={cos:.6f}")
                # The decode stages mirror the reference's own recurrence, so they
                # only carry bf16-reassociation differences; the prefill stages are
                # the ones the group-layout check hinges on.
                if cos < 0.99 or rms / scale > (0.05 if state is not None else 0.02):
                    raise SystemExit(f"staged reference does not reproduce the hook output "
                                     f"on layer {index} ({stage})")
                for kind, value in stage_values.items():
                    write_tap(out, kind, index, tokens, value.detach().float().cpu().numpy())
            token = forced[step] if step < len(forced) else int(result.logits[0, -1].argmax())
            ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
    for handle in handles:
        handle.remove()
    print(f"verified {len(gdn_layers)} GDN layer(s) x {args.steps} step(s) "
          f"against the reference's own hook output")


def tap_rank(name):
    """Order taps layer-major, in the order a layer computes them."""
    match = TAP_NAME.match(name)
    return int(match.group(2)), KINDS.index(match.group(1))


def compare(args):
    ours, theirs = Path(args.ours), Path(args.theirs)
    names = sorted(({p.name for p in ours.glob("*.f32")} &
                    {p.name for p in theirs.glob("*.f32")}),
                   key=lambda name: tap_rank(name) if TAP_NAME.match(name) else (1 << 30, 0))
    names = [name for name in names if TAP_NAME.match(name)]
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
        print("all taps agree within 2% of the reference scale")
        return 0
    print(f"first tap diverging beyond 2%: {worst}")
    return 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model-dir", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--prompt", default="1,2,3,4,5")
    d.set_defaults(func=dump)
    s = sub.add_parser("stages")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--prompt", default="1,2,3,4,5")
    s.add_argument("--steps", type=int, default=1,
                   help="forward passes: 1 = prefill only, then one decode per extra step")
    s.add_argument("--decode", default="",
                   help="comma-separated tokens to feed instead of the reference's own greedy ones")
    s.set_defaults(func=stages)
    c = sub.add_parser("compare")
    c.add_argument("--ours", required=True)
    c.add_argument("--theirs", required=True)
    c.set_defaults(func=compare)
    args = parser.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
