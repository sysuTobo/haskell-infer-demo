#!/usr/bin/env python3
"""Stage 2 claim E, backward half: is there a usable forward/backward pair?

docs/plan-numeric-contract.md, claim E: "Determine a usable paired implementation
or an explicitly compatible saved-state/recomputation adapter, including LSE,
masks, GQA and determinism. Current forward passes lse=nullptr; a backward from
another library cannot just be plugged in without a compatibility/gradient check."

`test_attention_lse` (ctest) establishes the forward half:

  * asking FlashInfer's prefill for the LSE leaves the attention output bitwise
    unchanged, so producing it costs nothing numerically;
  * the LSE it writes is log2(SUM_j e^{s_j}) -- the *natural* log-sum-exp expressed
    in base 2, because the kernel folds log2(e) into the scores so it can use exp2.
    It is neither ln(SUM_j e^{s_j}) nor log2(SUM_j 2^{s_j});
  * layout is [qo_idx * num_heads + head] float32, every element written.

This script establishes the backward half: what a backward consuming that saved
state has to compute, and how well it can be expected to agree with a paired
library.

  1. formula -- the analytic backward from (q, k, v, lse, dout) is compared against
     torch.autograd on the same math in float64, including the GQA head grouping.
     The natural domain is used there (lse_nat = ln SUM e^s), which is where the
     standard flash-attention backward formulas live.
  2. the compatibility trap -- the same backward fed the kernel's value *as if* it
     were the natural LSE. The kernel's value is the natural LSE divided by ln 2, so
     this scales the probabilities and the gradients; the size of the error is
     measured, because it is a silent failure, not a crash.
  3. paired-library compatibility -- torch's scaled_dot_product_attention (which does
     have a backward) is run in bf16 on the same q/k/v and its gradients are compared
     against the adapter's. That gap is the tolerance a trainer accepts by bolting a
     library backward onto this forward, and it is measured between two *different
     functions* (our forward rounds probabilities to bf16 and accumulates in fp32),
     which is why the pair has to be validated as a pair.
  4. resource estimate for the deployment shape.

Pure torch: no engine, no checkpoint, no GPU required (it uses one if present).
Run: python tests/attention_backward_feasibility.py
"""
import argparse
import math

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
LOG2E = 1.0 / LN2


def causal_mask(tokens, device):
    return torch.triu(torch.ones(tokens, tokens, dtype=torch.bool, device=device), 1)


def natural_forward(q, k, v, scale, group):
    """Attention in the natural domain: lse_nat = ln SUM e^s, P = e^{s - lse_nat}."""
    k_full = k.repeat_interleave(group, dim=1)
    v_full = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k_full) * scale
    scores = scores.masked_fill(causal_mask(q.shape[0], q.device).unsqueeze(0), float("-inf"))
    lse_nat = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse_nat.unsqueeze(-1))
    out = torch.einsum("hts,shd->thd", probs, v_full)
    return out, lse_nat, probs


def analytic_backward(q, k, v, lse_nat, dout, scale, group):
    """The backward a custom kernel computes from the saved (q, k, v, lse).

    Standard flash-attention backward in the natural domain. k/v are expanded to
    full heads for the math, and the gradients are summed back over the group --
    that grouping is the GQA requirement, not an implementation detail.
    """
    k_full = k.repeat_interleave(group, dim=1)
    v_full = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k_full) * scale
    scores = scores.masked_fill(causal_mask(q.shape[0], q.device).unsqueeze(0), float("-inf"))
    probs = torch.exp(scores - lse_nat.unsqueeze(-1))
    dv_full = torch.einsum("hts,thd->shd", probs, dout)
    dp = torch.einsum("thd,shd->hts", dout, v_full)
    row = (dp * probs).sum(dim=-1, keepdim=True)
    d_scores = probs * (dp - row)
    dq = torch.einsum("hts,shd->thd", d_scores, k_full) * scale
    dk_full = torch.einsum("hts,thd->shd", d_scores, q) * scale
    dk = dk_full.reshape(dk_full.shape[0], dk_full.shape[1] // group, group, dk_full.shape[2]).sum(2)
    dv = dv_full.reshape(dv_full.shape[0], dv_full.shape[1] // group, group, dv_full.shape[2]).sum(2)
    return dq, dk, dv


def report(name, got, want):
    diff = (got - want).abs()
    rel = (diff / want.abs().clamp_min(1e-12)).max().item()
    print(f"  {name}: max_abs={diff.max().item():.3e} max_rel={rel:.3e} "
          f"rms={diff.pow(2).mean().sqrt().item():.3e}")
    return diff.max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    args = parser.parse_args()
    if args.heads % args.kv_heads:
        raise SystemExit("heads must be a multiple of kv_heads")
    group = args.heads // args.kv_heads
    dev, T, H, Hk, D = args.device, args.tokens, args.heads, args.kv_heads, args.head_dim
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    q = (torch.randn(T, H, D, dtype=torch.float64, device=dev) * 0.3).requires_grad_(True)
    k = (torch.randn(T, Hk, D, dtype=torch.float64, device=dev) * 0.3).requires_grad_(True)
    v = (torch.randn(T, Hk, D, dtype=torch.float64, device=dev) * 0.3).requires_grad_(True)
    dout = torch.randn(T, H, D, dtype=torch.float64, device=dev)

    print(f"claim E: T={T} heads={H} kv_heads={Hk} group={group} head_dim={D} on {dev}")
    print("1. formula: analytic backward from (q,k,v,lse) vs torch.autograd, float64")
    out, lse_nat, probs = natural_forward(q, k, v, scale, group)
    print(f"  P rows sum to 1: worst |1 - sum| = {(probs.sum(-1) - 1).abs().max().item():.3e}")
    grads = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)
    dq, dk, dv = analytic_backward(q, k, v, lse_nat, dout, scale, group)
    worst = max(report("dq", dq, grads[0]), report("dk", dk, grads[1]), report("dv", dv, grads[2]))
    formula_ok = worst < 1e-9
    print(f"  formula {'matches autograd' if formula_ok else 'DOES NOT match autograd'} "
          f"(worst {worst:.3e})")

    print("2. the compatibility trap: consuming the kernel's LSE without converting it")
    # The kernel returns log2(SUM e^s) = lse_nat / ln2. A backward that treats that
    # as the natural LSE uses a value 1.4427x too large.
    lse_as_returned = lse_nat / LN2
    dq_bad, dk_bad, dv_bad = analytic_backward(q, k, v, lse_as_returned, dout, scale, group)
    wrong = max(report("dq (unconverted)", dq_bad, grads[0]),
                report("dk (unconverted)", dk_bad, grads[1]),
                report("dv (unconverted)", dv_bad, grads[2]))
    trap_detected = wrong > 1e-2
    print(f"  an unconverted LSE is off by {wrong:.3e} "
          f"({'detected: the convention has to be converted' if trap_detected else 'NOT detected'})")
    # ...and the correct consumption is P = exp2(s * log2(e) - lse_returned).
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(group, dim=1)) * scale
    masked = scores.masked_fill(causal_mask(T, dev).unsqueeze(0), float("-inf"))
    probs_conv = torch.exp2(masked * LOG2E - lse_as_returned.unsqueeze(-1))
    row_err = (probs_conv.masked_fill(causal_mask(T, dev).unsqueeze(0), 0.0).sum(-1) - 1).abs().max()
    print(f"  converted probabilities: worst |1 - row sum| = {row_err.item():.3e}")

    print("3. paired library: torch SDPA in bf16 on the same q/k/v")
    qb = q.detach().to(torch.bfloat16).to(torch.float32).requires_grad_(True)
    kb = k.detach().to(torch.bfloat16).to(torch.float32).requires_grad_(True)
    vb = v.detach().to(torch.bfloat16).to(torch.float32).requires_grad_(True)
    k_exp = kb.repeat_interleave(group, dim=1)
    v_exp = vb.repeat_interleave(group, dim=1)
    out_sdpa = F.scaled_dot_product_attention(
        qb.unsqueeze(0).transpose(1, 2).to(torch.bfloat16),
        k_exp.unsqueeze(0).transpose(1, 2).to(torch.bfloat16),
        v_exp.unsqueeze(0).transpose(1, 2).to(torch.bfloat16),
        is_causal=True).to(torch.float32).squeeze(0).transpose(0, 1)
    forward_gap = (out_sdpa - out.detach().to(torch.float32)).abs().max().item()
    print(f"  forward: SDPA(bf16) vs the float64 natural forward, max_abs = {forward_gap:.3e}")
    # The same upstream gradient for both sides, or the comparison compares two
    # different questions. repeat_interleave's own backward already sums the group
    # back into the kv heads, so kb.grad/vb.grad are the group-summed gradients.
    dout_b = torch.randn_like(out_sdpa)
    out_sdpa.backward(dout_b)
    scores_b = torch.einsum("thd,shd->hts", qb, k_exp) * scale
    scores_b = scores_b.masked_fill(causal_mask(T, dev).unsqueeze(0), float("-inf"))
    lse_b_nat = torch.logsumexp(scores_b, dim=-1)
    dqb, dkb, dvb = analytic_backward(qb, kb, vb, lse_b_nat, dout_b, scale, group)
    report("dq", dqb, qb.grad)
    report("dk", dkb, kb.grad)
    report("dv", dvb, vb.grad)
    print("  (this gap is between two different functions -- our forward rounds "
          "probabilities to bf16 and accumulates in fp32 -- so a library backward cannot "
          "be validated against a differently-rounded forward)")

    print("4. resource estimate for the deployment shape (T=4096, 16 attention layers)")
    Td, Hd, Hkd, Dd = 4096, 24, 4, 256
    per_token = (Hd * Dd * 2) * 3 + (Hkd * Dd * 2) * 4 + Hd * 4
    print(f"  per token per layer: q/o/dq bf16 {Hd*Dd*2} B each, k/v/dk/dv bf16 {Hkd*Dd*2} B each, "
          f"lse f32 {Hd*4} B -> {per_token} B")
    print(f"  activations for {Td} tokens x 16 attention layers: {per_token*Td*16/1e9:.2f} GB "
          f"(plus the MLP/GDN stacks)")
    print(f"  recomputing P from the LSE avoids a {Td}x{Td} bf16 probability tensor per head: "
          f"{Hd*Td*Td*2/1e9:.1f} GB per layer, which is why the LSE is the thing to save")
    print("  the forward runs partition_kv=false (null split-KV workspace), so a paired "
          "backward needs no split-KV reduction workspace either")

    print()
    if not (formula_ok and trap_detected):
        print("test_attention_backward_feasibility: FAIL")
        return 1
    print("test_attention_backward_feasibility: PASS (a backward from (q,k,v,LSE) is well "
          "defined and reproduces autograd; the LSE convention and the GQA grouping are the "
          "compatibility requirements a ported backward has to get right)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
