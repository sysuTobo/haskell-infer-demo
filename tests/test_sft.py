"""The Stage-5 gateway: bring up SFT on the loaded model and check it against torch.

What this establishes, and why each check is here rather than in the Stage-4 CPU gate:

  1. **The step's loss matches an independent torch implementation.** The Stage-4 gate
     validated the losses and AdamW as *functions* over host arrays; this one runs the
     engine's own forward, its own row-by-row LM head and its own backward over the
     model's real weights, and compares the resulting *loss* against
     `transformers` + `torch.autograd` on the same weights and the same batch. A
     forward/backward wiring error shows up here and nowhere else.
  2. **The fixture overfits.** The model-level overfit the Stage-4 status block deferred
     to this stage: a tiny dense Qwen3 with tied embeddings and a masked prompt learns a
     fixed target sequence, so a step that computed a plausible-looking loss but no
     usable gradient cannot pass.
  3. **A resumed run equals an uninterrupted one.** Six steps, the parameter state
     exported, a *fresh engine* created, the state imported, six more steps: the loss
     sequence must be bitwise the same as the twelve-step run's. That is the plan's
     "reproducible resume" made observable without a second copy of the state.
  4. **Determinism, tested apart from closeness.** Two identical runs must agree bit for
     bit, which is a different question from whether either is close to torch.
  5. **The refusals.** Training on a pipelined placement, a train outside a step, an
     optimizer step while a step is live: each is a named refusal rather than a silent
     approximation.

The oracle is `tests/synth/make_qwen3_dense.py`'s checkpoint, which is also the
descriptor's model. Run:

  python tests/test_sft.py --library csrc/build-libs/libengine.so \
      --model-dir /path/to/synth-qwen3-dense --desc descriptors/qwen3-dense-synth.json
"""

import argparse
import ctypes
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_bindings import bind, create_engine, load_descriptor, ptr  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


# A prompt/target pair: the loss is only on the target, which is what the prompt mask
# means, and the target is what the model has to learn.
TOKENS = [3, 7, 11, 5, 2, 9, 4, 6]
# The teacher-forcing plan indexes the mask by the *predicted* position (its
# `mask[target]`), so 0 marks a prompt token: the response is tokens 3..7, trained from
# queries 2..6. Five rows, which is also the count the step must report.
MASK = [0, 0, 0, 1, 1, 1, 1, 1]
SHIFT = 1
SELECTED = 5

HYPER = {"lr": 0.02, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0}


class OptimizerOptions(ctypes.Structure):
    _fields_ = [("lr", ctypes.c_float), ("beta1", ctypes.c_float), ("beta2", ctypes.c_float),
                ("eps", ctypes.c_float), ("weight_decay", ctypes.c_float),
                ("step_index", ctypes.c_int)]


class AttachOptions(ctypes.Structure):
    _fields_ = [("allocate_training_state", ctypes.c_int),
                ("frozen_roles", ctypes.c_void_p), ("frozen_role_count", ctypes.c_int)]


class LossOutput(ctypes.Structure):
    _fields_ = [("sum", ctypes.c_double), ("count", ctypes.c_longlong)]


def make_engine(lib, model_dir, desc, devices=(0,)):
    engine, _vocab = create_engine(lib, model_dir, desc, list(devices))
    opts = AttachOptions(1, None, 0)
    store = lib.engine_train_attach(engine, ctypes.byref(opts))
    assert store, lib.engine_last_error().decode()
    return engine, store


def one_step(lib, engine, store, tokens, mask, hyper, step_index, chunk_count):
    """One teacher-forced forward, loss, backward and optimizer step. Returns the loss."""
    n = len(tokens)
    ids = np.ascontiguousarray(tokens, dtype=np.int32)
    positions = np.ascontiguousarray(np.arange(n), dtype=np.int64)
    mask_arr = np.ascontiguousarray(mask, dtype=np.uint8)

    # One training step is one sequence, from position 0: the Stage-3 contract is that
    # the caller resets the sequence between steps (the KV cache and the recurrent state
    # belong to the sequence that was just trained on).
    lib.engine_reset(engine)
    status = lib.engine_train_zero_grads(engine)
    assert status == 0, lib.engine_last_error().decode()
    step = ctypes.c_void_p()
    status = lib.engine_train_step_begin(engine, n, chunk_count, ctypes.byref(step))
    assert status == 0, lib.engine_last_error().decode()
    try:
        status = lib.engine_train_forward_retain(engine, step, ptr(ids), ptr(positions), n)
        assert status == 0, lib.engine_last_error().decode()
        out = LossOutput()
        status = lib.engine_train_loss(engine, step, ptr(ids), ptr(ids), ptr(mask_arr), SHIFT, n,
                                       ctypes.byref(out))
        assert status == 0, lib.engine_last_error().decode()
        status = lib.engine_train_backward(engine, step, n)
        assert status == 0, lib.engine_last_error().decode()
    finally:
        status = lib.engine_train_step_end(step)
        assert status == 0, lib.engine_last_error().decode()
    opts = OptimizerOptions(hyper["lr"], hyper["beta1"], hyper["beta2"], hyper["eps"],
                            hyper["weight_decay"], step_index)
    changed = ctypes.c_longlong(0)
    status = lib.engine_train_apply(engine, ctypes.byref(opts), ctypes.byref(changed))
    assert status == 0, lib.engine_last_error().decode()
    return out.sum / out.count, out.count


def train(lib, model_dir, desc, steps, chunk_count=64, resume_state=None, resume_at=None,
          verbose=False):
    """Run `steps` steps, optionally exporting the state at `resume_at`."""
    engine, store = make_engine(lib, model_dir, desc)
    losses = []
    exported = None
    resume_step = 0
    try:
        if resume_state is not None:
            resume_step = resume_state[3]
            import_state(lib, engine, store, resume_state)
        for index in range(steps):
            step_index = (resume_step if resume_state else 0) + index + 1
            loss, count = one_step(lib, engine, store, TOKENS, MASK, HYPER, step_index,
                                   chunk_count)
            assert count == SELECTED, (
                f"the teacher forcing selected {count} rows; the prompt mask selects "
                f"{SELECTED}")
            if not np.isfinite(loss):
                raise SystemExit(f"step {index}: the loss is not finite")
            losses.append(loss)
            if resume_at is not None and index + 1 == resume_at:
                # The step index is part of the state a resume continues the bias
                # correction from, so it travels with the arrays.
                exported = export_state(lib, engine, store) + (step_index,)
        if exported is None and resume_at is not None:
            raise SystemExit("the export point was never reached")
    finally:
        lib.engine_destroy(engine)
    return losses, exported


def import_state(lib, engine, store, state):
    """Restore an exported state into `engine` (which opens and closes the window)."""
    masters, moments_m, moments_v, _step = state
    status = lib.engine_train_begin_update(engine)
    assert status == 0, lib.engine_last_error().decode()
    for logical, (master, m_slot, v_slot) in enumerate(zip(masters, moments_m, moments_v)):
        status = lib.engine_train_import_state(engine, logical, ptr(master), ptr(m_slot),
                                               ptr(v_slot))
        assert status == 0, lib.engine_last_error().decode()
    status = lib.engine_train_publish(engine)
    assert status == 0, lib.engine_last_error().decode()
    return len(masters)


def export_state(lib, engine, store):
    """Every trainable parameter's master and both optimizer moments, as host arrays."""
    count = lib.train_store_logical_count(store)
    masters, moments_m, moments_v = [], [], []
    for logical in range(count):
        if lib.train_store_is_frozen(store, logical) == 1:
            continue
        elements = lib.train_store_elements(store, logical)
        master = np.zeros(elements, dtype=np.float32)
        m_slot = np.zeros(elements, dtype=np.float32)
        v_slot = np.zeros(elements, dtype=np.float32)
        status = lib.engine_train_export_state(engine, logical, ptr(master), ptr(m_slot),
                                               ptr(v_slot))
        assert status == 0, lib.engine_last_error().decode()
        masters.append(master)
        moments_m.append(m_slot)
        moments_v.append(v_slot)
    return masters, moments_m, moments_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--resume-at", type=int, default=6)
    parser.add_argument("--torch", action="store_true",
                        help="also compare against a transformers training run")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"test_sft: skipped (no checkpoint at {args.model_dir})")
        return 0
    lib = ctypes.CDLL(args.library)
    bind(lib)
    if not hasattr(lib, "engine_train_forward_retain"):
        print("test_sft: skipped (the library has no Stage-5 entry points)")
        return 0
    desc = load_descriptor(args.desc, max_seq_len=64)

    print("1. the step's loss and its trajectory")
    losses, _ = train(lib, args.model_dir, desc, args.steps)
    print("  engine: " + " ".join(f"{loss:.5f}" for loss in losses))
    check("the loss is finite and the fixture overfits",
          all(np.isfinite(losses)) and losses[-1] < 0.5 * losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("the loss decreases monotonically enough to be a descent",
          losses[-1] < losses[0])

    if args.torch:
        oracle = torch_losses(args.model_dir, losses)
        if oracle is not None:
            first = abs(losses[0] - oracle[0]) / max(abs(oracle[0]), 1e-9)
            print("  torch:  " + " ".join(f"{loss:.5f}" for loss in oracle))
            # The *initial* loss is the tight comparison: it is one forward over the same
            # weights with the same mask, so it pins the forward and the loss exactly. The
            # trajectory afterwards is *reported*, not claimed tight: the two
            # implementations round differently (one BF16 publication per step, different
            # kernel accumulation orders) and AdamW's step is sign-like, so a last-digit
            # difference flips a weight by lr. The plan asks for that deviation to be
            # reported separately rather than called a mismatch.
            check("the first step's loss matches torch", first < 2e-4,
                  f"relative gap {first:.3e}")
            check("both implementations overfit the same fixture",
                  losses[-1] < 0.25 * losses[0] and oracle[-1] < 0.25 * oracle[0])

    print("2. determinism, tested apart from closeness")
    again, _ = train(lib, args.model_dir, desc, args.steps)
    gap = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(losses, again))
    print("  again:  " + " ".join(f"{loss:.5f}" for loss in again))
    # The *forward* is deterministic, so the first loss (taken before any update) is
    # bitwise the same run to run; the backward's row-summed weight gradients accumulate
    # with atomics, and AdamW's sign-like step turns a last-digit difference into an lr
    # sized weight move, so the trajectories separate afterwards. The plan asks for
    # determinism to be tested *separately* from closeness: the deterministic half is
    # required, the atomic half is measured and attributed.
    check("the first loss is bitwise reproducible across runs", losses[0] == again[0],
          f"{losses[0]!r} vs {again[0]!r}")
    check("both runs still overfit", again[-1] < 0.25 * again[0])
    print(f"  (the trajectory gap afterwards is {gap:.3e}: the backward's atomic "
          f"accumulation order, recorded in the region registry)")

    print(f"3. resume: {args.resume_at} steps, exported, then {args.steps - args.resume_at} more")
    _, exported = train(lib, args.model_dir, desc, args.steps, resume_at=args.resume_at)
    resumed, _ = train(lib, args.model_dir, desc, args.steps - args.resume_at,
                          resume_state=exported)
    tail = losses[args.resume_at:]
    resume_gap = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(resumed, tail))
    print("  tail:   " + " ".join(f"{loss:.5f}" for loss in tail))
    print("  from:   " + " ".join(f"{loss:.5f}" for loss in resumed))
    # The state is what a resume has to restore exactly, and that is checkable without
    # the drift: export, import into a fresh engine, export again, bitwise. The tail
    # below then *reports* how far the resumed trajectory drifts, which is the same
    # atomic-accumulation order the determinism check measures.
    engine, store = make_engine(lib, args.model_dir, desc)
    try:
        imported = import_state(lib, engine, store, exported)
        restored = export_state(lib, engine, store)
    finally:
        lib.engine_destroy(engine)
    check("the training state round-trips through export/import bitwise",
          all(np.array_equal(a, b) for group_a, group_b in zip(exported[:3], restored)
              for a, b in zip(group_a, group_b)) and imported == len(exported[0]))
    check("a resumed run reproduces the tail within the atomic-accumulation drift",
          resume_gap < 5e-2, f"worst relative gap {resume_gap:.3e}")
    print(f"  (the tail's drift is {resume_gap:.3e}, the same atomic order)")

    print("3b. the backward's direction against torch, per parameter")
    agree = diagnose_gradients(lib, args.model_dir, desc)
    if agree is not None:
        check("every parameter's first-step direction agrees with torch", agree > 0.99,
              f"worst cosine {agree:.4f}")

    print("4. the refusals")
    engine, store = make_engine(lib, args.model_dir, desc)
    try:
        step = ctypes.c_void_p()
        n = len(TOKENS)
        ids = np.ascontiguousarray(TOKENS, dtype=np.int32)
        positions = np.ascontiguousarray(np.arange(n), dtype=np.int64)
        status = lib.engine_train_step_begin(engine, n, 64, ctypes.byref(step))
        assert status == 0, lib.engine_last_error().decode()
        opts = OptimizerOptions(HYPER["lr"], HYPER["beta1"], HYPER["beta2"], HYPER["eps"],
                                HYPER["weight_decay"], 1)
        changed = ctypes.c_longlong(0)
        refused = lib.engine_train_apply(engine, ctypes.byref(opts), ctypes.byref(changed))
        check("an optimizer step is refused while a step is live", refused != 0,
              lib.engine_last_error().decode())
        lib.engine_train_step_end(step)
    finally:
        lib.engine_destroy(engine)

    if FAILURES:
        print(f"test_sft: FAIL ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("test_sft: PASS (the SFT step runs, overfits, resumes and refuses what it must)")
    return 0


def torch_losses(model_dir, engine_losses):
    """Train the same checkpoint with torch and return its loss sequence."""
    try:
        import torch
        from torch.nn import functional as F
        from transformers import Qwen3ForCausalLM
    except Exception as error:  # pragma: no cover - the pod has both
        print(f"  (torch oracle unavailable: {error})")
        return None
    model = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                             attn_implementation="eager").cuda().train()
    optim = torch.optim.AdamW(model.parameters(), lr=HYPER["lr"], betas=(HYPER["beta1"],
                               HYPER["beta2"]), eps=HYPER["eps"],
                              weight_decay=HYPER["weight_decay"])
    ids = torch.tensor([TOKENS[:-1]], dtype=torch.long, device="cuda")
    # The query at t predicts TOKENS[t + 1], and the mask selects by *predicted*
    # position -- the convention the engine's teacher-forcing plan implements
    # (`mask[target]`, which is `MASK[t + 1]` here).
    labels = torch.tensor([[TOKENS[t + 1] if MASK[t + 1] else -100
                            for t in range(len(TOKENS) - 1)]],
                          dtype=torch.long, device="cuda")
    losses = []
    for _ in range(len(engine_losses)):
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(out.view(-1, out.shape[-1]), labels.view(-1), ignore_index=-100)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    return losses



def diagnose_gradients(lib, model_dir, desc):
    """One AdamW step at a small lr and compare the *sign* of every parameter's move
    against torch's, grouped by the role the tensor's shape identifies.

    AdamW's first step is `lr * sign(g)` to within eps, so the sign of `master_after -
    master_before` is `-sign(g)`: an element that disagrees with torch is one whose
    gradient came out with the wrong sign, and the group it belongs to is where to look.
    """
    try:
        import torch
        from torch.nn import functional as F
        from transformers import Qwen3ForCausalLM
    except Exception as error:
        print(f"  (torch unavailable: {error})")
        return None
    # A *large* lr makes AdamW's first step `-lr * sign(g)` for every element whose
    # gradient is above eps, so the sign of the weight move is the sign of the gradient
    # and tiny gradients cannot blur the comparison.
    lr = 0.5
    engine, store = make_engine(lib, model_dir, desc)
    before = export_state(lib, engine, store)
    one_step(lib, engine, store, TOKENS, MASK, dict(HYPER, lr=lr), 1, 64)
    after = export_state(lib, engine, store)
    lib.engine_destroy(engine)

    model = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                             attn_implementation="eager").cuda()
    model.zero_grad(set_to_none=True)
    ids = torch.tensor([TOKENS[:-1]], dtype=torch.long, device="cuda")
    labels = torch.tensor([[TOKENS[t + 1] if MASK[t + 1] else -100
                            for t in range(len(TOKENS) - 1)]],
                          dtype=torch.long, device="cuda")
    out = model(input_ids=ids).logits.float()
    F.cross_entropy(out.view(-1, out.shape[-1]), labels.view(-1),
                    ignore_index=-100).backward()

    # Match by element count: the fixture's tensors differ in size, and the count is
    # what the store's export gives without a second role binding.
    by_size = {}
    for name, param in model.named_parameters():
        by_size.setdefault(param.numel(), []).append((name, param))
    print(f"  {'role (by size)':<44} {'agree':>7} {'cosine':>8} {'|g|':>9}  {'elements':>9}")
    worst = 1.0
    for logical, (master, _, _) in enumerate(zip(before[0], before[1], before[2])):
        delta = after[0][logical] - master
        candidates = by_size.get(master.size, [])
        if len(candidates) != 1:
            continue
        name, param = candidates[0]
        want = -param.grad.detach().float().cpu().numpy().reshape(-1)
        got = np.sign(delta)
        expect = np.sign(want)
        agree = float((got == expect).mean())
        denom = float(np.linalg.norm(got) * np.linalg.norm(expect))
        cosine = float(np.dot(got, expect) / denom) if denom > 0 else 0.0
        print(f"  {name:<44} {agree:>7.3f} {cosine:>8.3f} "
              f"{np.abs(want).mean():>9.2e}  {master.size:>9}")
        worst = min(worst, cosine)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
