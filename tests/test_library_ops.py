import argparse
import ctypes
import sys

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule


def pointer(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def check(name, actual, expected, atol, rtol):
    assert torch.isfinite(actual).all(), name
    error = (actual.float() - expected.float()).abs().max().item()
    print(f"{name}: max_abs={error:.8g}", flush=True)
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def test_fla(library, device):
    state = torch.randn(1, 48, 128, 128, device=device) * 0.02
    reference_state = state.clone()
    alog = torch.randn(48, device=device, dtype=torch.bfloat16) * 0.5
    bias = torch.randn(48, device=device, dtype=torch.bfloat16) * 0.2
    for tokens in (1, 2, 5, 63, 64, 65, 128, 1, 1):
        qkv = torch.randn(tokens, 10240, device=device, dtype=torch.bfloat16) * 0.2
        a = torch.randn(tokens, 48, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        out = torch.full((1, tokens, 48, 128), float("nan"), device=device, dtype=torch.bfloat16)
        q, k, v = qkv.split((2048, 2048, 6144), dim=-1)
        q = q.reshape(1, tokens, 16, 128).repeat_interleave(3, dim=2)
        k = k.reshape(1, tokens, 16, 128).repeat_interleave(3, dim=2)
        v = v.reshape(1, tokens, 48, 128)
        g = -alog.float().exp() * F.softplus(a.float() + bias.float())
        expected, reference_state = torch_recurrent_gated_delta_rule(
            q, k, v, g.unsqueeze(0), b.sigmoid().unsqueeze(0),
            reference_state, True, use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize(device)
        status = library.test_fla(pointer(out), pointer(qkv), pointer(a), pointer(b),
                                  pointer(alog), pointer(bias), pointer(state), tokens)
        assert status == 0, ("native FLA status", status)
        check(f"{device}/FLA/T={tokens}/output", out, expected, 8e-4, 0.02)
        check(f"{device}/FLA/T={tokens}/state", state, reference_state, 0.0015, 0.025)


def test_norm_and_rope(library, device):
    for cols in (128, 256, 5120):
        x = torch.randn(5, cols, device=device, dtype=torch.bfloat16)
        x[0] = 0
        x[1] *= 1e-4
        weight = torch.randn(cols, device=device, dtype=torch.bfloat16) * 0.1
        out = torch.empty_like(x)
        xf = x.float()
        expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                    * (1 + weight.float())).bfloat16()
        torch.cuda.synchronize(device)
        assert library.test_norm(pointer(out), pointer(x), pointer(weight), 5, cols) == 0
        check(f"{device}/GemmaRMSNorm/{cols}", out, expected, 0.001, 0.008)
    positions = torch.tensor([0, 1, 127, 4095], device=device, dtype=torch.int64)
    q = torch.randn(4, 24, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, 4, 256, device=device, dtype=torch.bfloat16)
    references = []
    angles = positions.float()[:, None] / (1e7 ** (torch.arange(32, device=device).float() / 32))
    for x in (q, k):
        result = x.clone()
        first, second = x[..., :32].float(), x[..., 32:64].float()
        cosine, sine = angles.cos()[:, None, :], angles.sin()[:, None, :]
        result[..., :32] = first * cosine - second * sine
        result[..., 32:64] = second * cosine + first * sine
        references.append(result)
    torch.cuda.synchronize(device)
    assert library.test_rope(pointer(q), pointer(k), pointer(positions), 4) == 0
    for label, actual, expected in zip(("Q", "K"), (q, k), references):
        check(f"{device}/partial-RoPE/{label}", actual, expected, 0.002, 0.008)
        torch.testing.assert_close(actual[..., 64:], expected[..., 64:], atol=0, rtol=0)


def test_silu_mul(library, device):
    """The dense MLP's activation in both layouts (plan F1).

    `kernel_silu_mul` takes the contiguous [gate[n], up[n]] pair the two-GEMM path builds;
    `kernel_silu_mul_packed` takes the row-interleaved [tokens, 2*intermediate] buffer one
    N = 2*intermediate gate/up GEMM produces. The reference is computed here in float32/bf16.

    The third check is why this fixture exists: feeding the *packed* buffer to the *halves*
    kernel must not reproduce the reference once tokens > 1, because at tokens == 1 row 0's
    two halves are the whole buffer and the two contracts coincide. A fusion that reformats
    the GEMM output without changing the activation kernel's row/stride contract would pass
    a single-token test and fail this one.
    """
    for tokens, intermediate in ((1, 8), (2, 8), (4, 12), (3, 16)):
        scale = 0.5
        gate = (torch.randn(tokens, intermediate, device=device) * scale).bfloat16()
        up = (torch.randn(tokens, intermediate, device=device) * scale).bfloat16()
        expected = (F.silu(gate.float()) * up.float()).bfloat16()

        # The existing contract: [gate, up] halves of one flat buffer.
        halves = torch.cat([gate.reshape(-1), up.reshape(-1)]).contiguous()
        out_halves = torch.empty(tokens * intermediate, device=device, dtype=torch.bfloat16)
        assert library.test_silu_mul(pointer(out_halves), pointer(halves),
                                     tokens * intermediate) == 0
        check(f"{device}/SiLU-mul/halves/T={tokens}/I={intermediate}",
              out_halves.view(tokens, intermediate), expected, 0.002, 0.01)

        # The fused contract: one [T, 2I] row-major buffer with gate then up per row.
        packed = torch.cat([gate, up], dim=1).contiguous()
        out_packed = torch.empty(tokens, intermediate, device=device, dtype=torch.bfloat16)
        assert library.test_silu_mul_packed(pointer(out_packed), pointer(packed), tokens,
                                            intermediate) == 0
        check(f"{device}/SiLU-mul/packed/T={tokens}/I={intermediate}",
              out_packed, expected, 0.002, 0.01)

        if tokens > 1:
            naive = torch.empty_like(out_packed)
            assert library.test_silu_mul(pointer(naive), pointer(packed.reshape(-1)),
                                         tokens * intermediate) == 0
            differs = (naive.float() - expected.float()).abs().max().item()
            magnitude = expected.float().abs().max().item()
            print(f"{device}/SiLU-mul/packed-as-halves/T={tokens}: max_abs={differs:.6g} "
                  f"(reference magnitude {magnitude:.6g})", flush=True)
            assert differs > 0.1 * magnitude, (
                "the halves contract reproduced the row-interleaved reference, so the "
                "fixture is not exercising the layout it exists for")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    library = ctypes.CDLL(args.library)
    library.test_fla.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int]
    library.test_norm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    library.test_rope.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
    library.test_silu_mul.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int]
    library.test_silu_mul_packed.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
    torch.manual_seed(7)
    device_count = torch.cuda.device_count()
    if device_count < 1:
        # Without this the loop body never runs and the suite would report success
        # for tests it never executed.
        print("test_library_ops: SKIP (no CUDA device visible)", flush=True)
        return 0
    for index in range(min(2, device_count)):
        with torch.cuda.device(index):
            device = torch.device("cuda", index)
            test_norm_and_rope(library, device)
            test_fla(library, device)
            test_silu_mul(library, device)
    print(f"Library operator tests passed on {min(2, device_count)} device(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
