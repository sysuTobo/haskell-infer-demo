#!/usr/bin/env python3
"""Stage 3's gate on a real engine (docs/plan-numeric-contract.md).

"Gate: tiny-model all-position forward agrees with an independent reference; tied
roles stay tied; a synthetic parameter update refreshes all readers; lifetime tests
reject update/free while a reader or backward context is active."

The fixture is the synthetic Qwen3-Next checkpoint: 4 layers (3 GDN + 1 attention),
hidden 256, vocab 1024. It is small enough that the *independent* reference is a real
transformers forward on the same device, computed here rather than read from a
golden file, so "agrees with an independent reference" is checked against the
implementation, not against a previous run of ours.

Four things are established:

  1. all-position forward: `engine_train_forward --all-logits` against
     `transformers(...).logits[0]`, every position, top-1 and an rms band;
  2. teacher forcing: the selected positions of a masked sequence, and their
     log-probabilities, against the reference's own log-softmax (independent) and
     against the engine's device-fused path (consistency);
  3. tied roles and a synthetic update: a descriptor variant whose lmHead role
     templates onto the embedding tensor makes the engine's parameter store collapse
     both roles into one logical parameter, and a published update then has to reach
     *both* readers - checked by comparing with a torch forward whose two tensors were
     edited with the same values. A no-op publication must be bitwise inert, which is
     what shows the write path itself is not lossy;
  4. the derived copy: publishing a new GDN norm weight must refresh the FP32 copy the
     forward actually reads (`gdn_norm_f32`), which is checked the same way.

Lifetime rejections at the engine level are checked too: a second update window, a
publication with no window, a master write for a frozen role, and an update while a
training step is live.

Run: python tests/test_train_forward.py --library csrc/build-libs/libengine.so \
         --model-dir /var/pony/cache/bohaotu-haskell/synth-qwen3next \
         --desc descriptors/qwen3-next-synth.json
"""
import argparse
import ctypes
import json
import os
import sys
import tempfile

import numpy as np

from engine_bindings import bind, create_engine, describe, load_descriptor, ptr

# Role ids are looked up by name rather than hardcoded: the enum ordinals are an
# implementation detail of model_desc.h and a wrong one silently tests another role.

TOKENS = 12
PROMPT = 3


def flat_rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=np.float64) -
                                  np.asarray(b, dtype=np.float64)) ** 2)))


def load_reference(model_dir, ids, tied=False):
    """The independent reference: a transformers forward over the same token ids."""
    import torch
    from transformers import AutoModelForCausalLM

    # Loaded plainly and moved, not with device_map: accelerate's device hooks keep
    # the weights in a separate map and a `.data` assignment on a module parameter can
    # then be invisible to the forward, which would make every "the edit took effect"
    # check vacuous.
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    if tied:
        # Mimic the tie the descriptor variant declares: the LM head *is* the
        # embedding tensor.
        with torch.no_grad():
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda")).logits[0].float().cpu().numpy()
    return model, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--desc", default="descriptors/qwen3-next-synth.json")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    # Paths arrive relative to the caller's cwd when this runs by hand and to the
    # build directory when ctest runs it, so a relative descriptor is resolved against
    # the repository root instead.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.desc):
        args.desc = os.path.join(repo, args.desc)
    # Like test_engine_resources skipping without a device: this gate needs the
    # synthetic checkpoint and torch, so an environment without them reports a skip
    # rather than a failure.
    if not os.path.isdir(args.model_dir):
        print(f"test_train_forward: skipped (no checkpoint at {args.model_dir})")
        return 0
    if not os.path.exists(args.desc):
        print(f"test_train_forward: skipped (no descriptor at {args.desc})")
        return 0

    lib = bind(ctypes.CDLL(args.library))
    devices = [int(x) for x in args.devices.split(",")]
    failures = []
    role_id = lambda name: lib.model_desc_role_from_name(name.encode())
    for needed in ("gdnNorm", "gdnDtBias", "gdnB", "embed", "lmHead"):
        if role_id(needed) < 0:
            print(f"cannot resolve role '{needed}'", flush=True)
            return 1
    ROLE_GDN_NORM = role_id("gdnNorm")
    # A role the trainer has no reason to train: the frozen-parameter case.
    ROLE_GDN_DT_BIAS = role_id("gdnDtBias")

    def check(name, condition, detail=""):
        print(f"  {'PASS' if condition else 'FAIL'} {name}{' — ' + detail if detail else ''}",
              flush=True)
        if not condition:
            failures.append(name)

    # A fixed token sequence, no tokenizer needed: the synthetic vocabulary is 1024
    # wide and the reference consumes the same ids.
    ids = [(17 * (t + 1)) % 1024 for t in range(TOKENS)]
    ids[0] = 5
    mask = np.zeros(TOKENS, dtype=np.uint8)
    mask[PROMPT:] = 1
    positions = np.arange(TOKENS, dtype=np.int64)

    print("=== 1. all-position forward vs an independent transformers reference ===")
    model, reference = load_reference(args.model_dir, ids)
    descriptor = load_descriptor(args.desc, max_seq_len=TOKENS)
    engine, vocab = create_engine(lib, args.model_dir, descriptor, devices)
    check("descriptor round-trip", describe(lib, engine) == descriptor)
    check("vocabulary width", vocab == 1024, f"vocab={vocab}")

    all_logits = np.empty((TOKENS, vocab), dtype=np.float32)
    selected = np.empty(TOKENS, dtype=np.int32)

    class TrainForwardOutput(ctypes.Structure):
        _fields_ = [("all_logits", ctypes.c_void_p), ("logprobs", ctypes.c_void_p),
                    ("selected", ctypes.c_void_p), ("selected_count", ctypes.c_int)]

    ids_arr = np.ascontiguousarray(ids, dtype=np.int32)
    out = TrainForwardOutput(ctypes.cast(ptr(all_logits), ctypes.c_void_p), None,
                             ctypes.cast(ptr(selected), ctypes.c_void_p), 0)
    status = lib.engine_train_forward(engine, ptr(ids_arr), TOKENS, ptr(positions), None,
                                      None, 1, ctypes.byref(out))
    check("all-position forward", status == 0, lib.engine_last_error().decode())
    if status == 0:
        per_position_top1 = sum(int(all_logits[t].argmax()) == int(reference[t].argmax())
                                for t in range(TOKENS))
        rms = flat_rms(all_logits, reference)
        print(f"  positions matching top-1: {per_position_top1}/{TOKENS}, rms={rms:.5f}", flush=True)
        check("every position's top-1 agrees", per_position_top1 == TOKENS)
        check("logit rms within band", rms <= args.rms_tolerance, f"rms={rms:.5f}")

    print("=== 2. teacher forcing: label shift, prompt mask, fused log-probabilities ===")
    lib.engine_reset(engine)
    logprobs = np.empty(TOKENS, dtype=np.float32)
    selected = np.empty(TOKENS, dtype=np.int32)
    out = TrainForwardOutput(None, ctypes.cast(ptr(logprobs), ctypes.c_void_p),
                             ctypes.cast(ptr(selected), ctypes.c_void_p), 0)
    status = lib.engine_train_forward(engine, ptr(ids_arr), TOKENS, ptr(positions), None,
                                      ptr(mask), 1, ctypes.byref(out))
    check("teacher-forced forward", status == 0, lib.engine_last_error().decode())
    # The mask marks targets PROMPT..TOKENS-1 as response tokens, so the queries whose
    # label is masked in are PROMPT-1 .. TOKENS-2: TOKENS - PROMPT of them.
    expected_selected = TOKENS - PROMPT
    if status == 0:
        check("selected positions", out.selected_count == expected_selected,
              f"{out.selected_count} vs {expected_selected}")
        # Independent: the reference's own log-softmax at those positions.
        ref_logprobs = []
        for j in range(out.selected_count):
            query = int(selected[j])
            label = ids[query + 1]
            row = reference[query].astype(np.float64)
            ref_logprobs.append(row[label] - (row.max() + np.log(np.exp(row - row.max()).sum())))
        fused = np.asarray(logprobs[:out.selected_count], dtype=np.float64)
        gap = float(np.abs(fused - np.asarray(ref_logprobs)).max())
        check("selected queries are the masked ones",
              [int(selected[j]) for j in range(out.selected_count)] ==
              list(range(PROMPT - 1, TOKENS - 1)))
        check("fused log-probabilities agree with the reference's log-softmax",
              gap < args.rms_tolerance, f"max gap={gap:.5f}")

        # Consistency: the same call with all logits must give the same numbers.
        lib.engine_reset(engine)
        all_logits2 = np.empty((TOKENS, vocab), dtype=np.float32)
        logprobs2 = np.empty(TOKENS, dtype=np.float32)
        selected2 = np.empty(TOKENS, dtype=np.int32)
        out2 = TrainForwardOutput(ctypes.cast(ptr(all_logits2), ctypes.c_void_p),
                                  ctypes.cast(ptr(logprobs2), ctypes.c_void_p),
                                  ctypes.cast(ptr(selected2), ctypes.c_void_p), 0)
        status = lib.engine_train_forward(engine, ptr(ids_arr), TOKENS, ptr(positions), None,
                                          ptr(mask), 1, ctypes.byref(out2))
        check("teacher-forced forward, host log-softmax path", status == 0,
              lib.engine_last_error().decode())
        if status == 0:
            host = np.asarray(logprobs2[:out2.selected_count], dtype=np.float64)
            gap = float(np.abs(host - fused).max())
            check("the fused and host paths agree", gap < 1e-4, f"max gap={gap:.2e}")

    print("=== 3. tied roles stay tied through a synthetic update ===")
    # A descriptor variant whose lmHead role templates onto the embedding tensor: the
    # tie is a semantic fact the descriptor implies, and the store has to resolve it.
    with open(args.desc) as handle:
        tied_descriptor = json.load(handle)
    names = tied_descriptor["role_names"]
    templates = tied_descriptor["role_templates"]
    templates[names.index("lmHead")] = templates[names.index("embed")]
    tied_descriptor["max_seq_len"] = TOKENS
    tied_descriptor["max_position_embeddings"] = max(TOKENS,
                                                     tied_descriptor["max_position_embeddings"])
    fd, tied_path = tempfile.mkstemp(suffix=".json", prefix="synth-tied-")
    with os.fdopen(fd, "w") as handle:
        json.dump(tied_descriptor, handle)
    tied_ids = [(17 * (t + 1)) % 1024 for t in range(TOKENS)]
    # The untied reference for the same ids, so the tie's own effect on the torch side
    # is visible in the log rather than assumed.
    _, reference_untied_for_tied_ids = load_reference(args.model_dir, tied_ids)
    model_tied, reference_tied = load_reference(args.model_dir, tied_ids, tied=True)
    print(f"  the torch tie moved its own logits by rms="
          f"{flat_rms(reference_tied, reference_untied_for_tied_ids):.5f}", flush=True)
    engine_tied, _ = create_engine(lib, args.model_dir, tied_descriptor, devices)
    check("tied descriptor round-trip", describe(lib, engine_tied) == tied_descriptor)

    def tied_forward(engine):
        lib.engine_reset(engine)
        rows = np.empty((TOKENS, vocab), dtype=np.float32)
        structure = TrainForwardOutput(ctypes.cast(ptr(rows), ctypes.c_void_p), None, None, 0)
        code = lib.engine_train_forward(engine, ptr(np.ascontiguousarray(tied_ids, dtype=np.int32)),
                                        TOKENS, ptr(positions), None, None, 1,
                                        ctypes.byref(structure))
        check("tied forward", code == 0, lib.engine_last_error().decode())
        return rows

    # Before any store or publication: the tied engine must already agree with the
    # tied torch model, or a later mismatch would be blamed on the update path.
    before_store = tied_forward(engine_tied)
    baseline_rms = flat_rms(before_store, reference_tied)
    print(f"  tied engine vs the tied torch reference (no update yet): rms={baseline_rms:.5f}",
          flush=True)
    check("the tied descriptor agrees with the tied reference", baseline_rms <= args.rms_tolerance,
          f"rms={baseline_rms:.5f}")
    lib.engine_reset(engine_tied)

    class AttachOptions(ctypes.Structure):
        _fields_ = [("allocate_training_state", ctypes.c_int),
                    ("frozen_roles", ctypes.c_void_p), ("frozen_role_count", ctypes.c_int)]

    frozen = np.ascontiguousarray([ROLE_GDN_DT_BIAS], dtype=np.int32)
    opts = AttachOptions(1, ctypes.cast(ptr(frozen), ctypes.c_void_p), 1)
    store = lib.engine_train_attach(engine_tied, ctypes.byref(opts))
    check("attach a store with training state", bool(store), lib.train_last_error().decode())
    if not store:
        print("cannot continue without a store", flush=True)
        return 1
    specs = lib.train_store_spec_count(store)
    logical = lib.train_store_logical_count(store)
    check("tying removes exactly one logical parameter", logical == specs - 1,
          f"{logical} logical vs {specs} specs")
    embed_logical = lib.train_store_logical_of(store, -1, role_id("embed"))
    lm_head_logical = lib.train_store_logical_of(store, -1, role_id("lmHead"))
    check("lmHead resolves onto the embedding", embed_logical == lm_head_logical and embed_logical >= 0,
          f"embed={embed_logical} lmHead={lm_head_logical}")
    check("the tied parameter has two readers",
          lib.train_store_alias_count(store, embed_logical) == 2)
    frozen_logical = lib.train_store_logical_of(store, 0, ROLE_GDN_DT_BIAS)
    check("the frozen role is frozen and has no master",
          lib.train_store_is_frozen(store, frozen_logical) == 1)
    opened = lib.engine_train_begin_update(engine_tied)
    refused = lib.engine_train_write_master(engine_tied, frozen_logical,
                                            ptr(np.zeros(4, dtype=np.float32)))
    print(f"  frozen write: begin_update={opened}, write={refused}, "
          f"error={lib.train_last_error().decode()!r}, engine_error="
          f"{lib.engine_last_error().decode()!r}", flush=True)
    check("a frozen parameter refuses a master write", opened == 0 and refused != 0)
    check("the refusal names the reason", b"master" in lib.engine_last_error())
    check("a second update window is refused", lib.engine_train_begin_update(engine_tied) != 0,
          lib.engine_last_error().decode())
    check("the refused update left the version alone", lib.train_store_version(store) == 0)

    # A no-op publication: the master is the current weight in FP32, so re-casting it
    # must reproduce the same BF16 bytes and the forward must not move at all.
    embed_f32 = model_tied.model.embed_tokens.weight.data.float().cpu().numpy().reshape(-1)
    check("master write (no-op)", lib.engine_train_write_master(
        engine_tied, embed_logical, ptr(embed_f32)) == 0, lib.train_last_error().decode())
    check("publish", lib.engine_train_publish(engine_tied) == 0, lib.train_last_error().decode())
    check("version advanced", lib.train_store_version(store) == 1)
    check("no stale derived copy remains", lib.train_store_stale_derived_count(store) == 0)
    lib.engine_reset(engine_tied)
    baseline_last_row = np.empty(vocab, dtype=np.float32)
    status = lib.engine_prefill(engine_tied, ptr(np.ascontiguousarray(tied_ids, dtype=np.int64)),
                                TOKENS, ptr(baseline_last_row))
    check("inference forward after a no-op publish", status == 0, lib.engine_last_error().decode())
    after = tied_forward(engine_tied)
    # A no-op publication (the master equal to the loaded weight) must leave the model
    # exactly where it was, on *every* reader. Comparing before/after is what catches a
    # publication that writes something the caller never asked for.
    moved_by_noop = float(np.abs(after - before_store).max())
    check("a no-op publication changes nothing", moved_by_noop == 0.0,
          f"max change={moved_by_noop:.3e}")
    # ...and the inference reader and the training reader are the same weights read two
    # ways, so their last-row logits agree bit for bit.
    gap = float(np.abs(baseline_last_row - after[-1]).max())
    check("both readers see the same weights", gap == 0.0, f"max gap={gap:.3e}")
    untied_rms = flat_rms(after, all_logits)
    print(f"  the tie moved the engine's own logits by rms={untied_rms:.5f}", flush=True)
    check("the tied descriptor differs from the untied one", untied_rms > 0.0)

    # Now a real edit: the tied master is perturbed, and both readers must move.
    edited = np.asarray(embed_f32, dtype=np.float32) * np.float32(1.05) + np.float32(0.01)
    lib.engine_train_begin_update(engine_tied)
    check("master write (perturbed)", lib.engine_train_write_master(
        engine_tied, embed_logical, ptr(np.ascontiguousarray(edited, dtype=np.float32))) == 0,
        lib.train_last_error().decode())
    check("publish the perturbed tied weight",
          lib.engine_train_publish(engine_tied) == 0, lib.train_last_error().decode())
    # The same edit on the torch side, applied to *both* tensors because they are tied.
    import torch
    with torch.no_grad():
        edited_tensor = torch.tensor(edited, dtype=torch.float32).reshape(
            model_tied.model.embed_tokens.weight.shape).to(torch.bfloat16).cuda()
        model_tied.model.embed_tokens.weight.data = edited_tensor.clone()
        model_tied.lm_head.weight.data = edited_tensor.clone()
        reference_edited = model_tied(
            torch.tensor([tied_ids], device="cuda")).logits[0].float().cpu().numpy()
    lib.engine_reset(engine_tied)
    edited_logits = np.empty((TOKENS, vocab), dtype=np.float32)
    out_edit = TrainForwardOutput(ctypes.cast(ptr(edited_logits), ctypes.c_void_p), None, None, 0)
    status = lib.engine_train_forward(engine_tied,
                                      ptr(np.ascontiguousarray(tied_ids, dtype=np.int32)), TOKENS,
                                      ptr(positions), None, None, 1, ctypes.byref(out_edit))
    check("training forward after the tied update", status == 0, lib.engine_last_error().decode())
    if status == 0:
        rms = flat_rms(edited_logits, reference_edited)
        moved = float(np.abs(edited_logits - after).max())
        top1 = sum(int(edited_logits[t].argmax()) == int(reference_edited[t].argmax())
                   for t in range(TOKENS))
        print(f"  tied update: moved={moved:.4f}, engine finite={np.isfinite(edited_logits).all()}, "
              f"reference finite={np.isfinite(reference_edited).all()}, "
              f"rms vs edited reference={rms:.5f}, top-1 {top1}/{TOKENS}", flush=True)
        check("the tied update moved the output", moved > 0.0)
        check("both readers saw the update (matches the edited reference)",
              rms <= args.rms_tolerance and top1 == TOKENS,
              f"rms={rms:.5f} top1={top1}/{TOKENS}")

    print("=== 4. a published GDN norm weight refreshes its FP32 derived copy ===")
    # The forward reads gdn_norm_f32, a copy cast from the BF16 source at load. If the
    # publication refreshed only the source, the engine would keep reading the old
    # weight and this comparison against torch would fail.
    # The descriptor describes layers by mixer kind, not by HF's layer_types name.
    gdn_layers = [i for i, kind in enumerate(tied_descriptor["layer_mixers"]) if kind == "gdn"]
    layer = gdn_layers[0]
    gdn_logical = lib.train_store_logical_of(store, layer, ROLE_GDN_NORM)
    check("the GDN norm parameter is present", gdn_logical >= 0, f"logical={gdn_logical}")
    norm_old = model_tied.model.layers[layer].linear_attn.norm.weight.data.float().cpu().numpy()
    norm_edit = np.asarray(norm_old, dtype=np.float32) * np.float32(1.10) + np.float32(0.005)
    lib.engine_train_begin_update(engine_tied)
    check("master write (gdn norm)", lib.engine_train_write_master(
        engine_tied, gdn_logical, ptr(np.ascontiguousarray(norm_edit, dtype=np.float32))) == 0,
        lib.train_last_error().decode())
    check("publish the gdn norm weight",
          lib.engine_train_publish(engine_tied) == 0, lib.train_last_error().decode())
    with torch.no_grad():
        model_tied.model.layers[layer].linear_attn.norm.weight.data = (
            torch.tensor(norm_edit, dtype=torch.float32)
            .reshape(model_tied.model.layers[layer].linear_attn.norm.weight.shape)
            .to(torch.bfloat16).cuda())
        reference_norm = model_tied(
            torch.tensor([tied_ids], device="cuda")).logits[0].float().cpu().numpy()
    lib.engine_reset(engine_tied)
    norm_logits = np.empty((TOKENS, vocab), dtype=np.float32)
    out_norm = TrainForwardOutput(ctypes.cast(ptr(norm_logits), ctypes.c_void_p), None, None, 0)
    status = lib.engine_train_forward(engine_tied,
                                      ptr(np.ascontiguousarray(tied_ids, dtype=np.int32)), TOKENS,
                                      ptr(positions), None, None, 1, ctypes.byref(out_norm))
    check("training forward after the norm update", status == 0, lib.engine_last_error().decode())
    if status == 0:
        rms = flat_rms(norm_logits, reference_norm)
        top1 = sum(int(norm_logits[t].argmax()) == int(reference_norm[t].argmax())
                   for t in range(TOKENS))
        print(f"  gdn norm update: engine finite={np.isfinite(norm_logits).all()}, "
              f"reference finite={np.isfinite(reference_norm).all()}, "
              f"rms vs reference={rms:.5f}, top-1 {top1}/{TOKENS}", flush=True)
        check("the derived FP32 copy was refreshed (matches the edited reference)",
              rms <= args.rms_tolerance and top1 == TOKENS, f"rms={rms:.5f} top1={top1}/{TOKENS}")

    print("=== 5. lifetime: a training step blocks an update ===")
    plan_count = ctypes.c_int(0)
    plan_elements = ctypes.c_int64(0)
    status = lib.engine_train_step_plan(engine_tied, TOKENS, 1, ctypes.byref(plan_count),
                                        ctypes.byref(plan_elements))
    check("step plan", status == 0, lib.train_last_error().decode())
    print(f"  the step retains {plan_count.value} values, {plan_elements.value} FP32 elements of "
          f"GDN chunk-boundary state", flush=True)
    check("the plan retains activations and chunk states",
          plan_count.value >= 3 * 4 and plan_elements.value > 0)
    step = ctypes.c_void_p()
    status = lib.engine_train_step_begin(engine_tied, TOKENS, 1, ctypes.byref(step))
    check("begin a training step", status == 0 and bool(step), lib.train_last_error().decode())
    check("an update is refused while a step is live",
          lib.engine_train_begin_update(engine_tied) != 0)
    check("the refusal names the reader", b"reader" in lib.train_last_error())
    check("end the step", lib.engine_train_step_end(step) == 0, lib.train_last_error().decode())
    check("an update opens once the step is gone", lib.engine_train_begin_update(engine_tied) == 0,
          lib.train_last_error().decode())

    lib.engine_destroy(engine_tied)
    lib.engine_destroy(engine)
    os.unlink(tied_path)

    print()
    if failures:
        print(f"test_train_forward: FAIL: {len(failures)} check(s): {failures}")
        return 1
    print("test_train_forward: PASS (all-position forward, teacher forcing, tied roles, a synthetic "
          "update refreshing both readers and the derived FP32 copy, and the step lifetime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
