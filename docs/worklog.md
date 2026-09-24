# Work Log

A running snapshot of what this project can do today, what has been verified and
where it is knowingly incomplete. [README.md](../README.md) describes the
component layout and how to build and test; [design.md](design.md) holds the
architecture rationale; [plan-numeric-contract.md](plan-numeric-contract.md) is
the proposal for the trainer, RL and inference-optimization work that is **not**
implemented here.

Last updated: 2026-09-24.

## Verified today

Everything below was run and passed on 2× A40 46 GB (sm_86) unless a line says
otherwise. No GPU result here is implied by a document edit alone.

| Gate | Result |
|---|---|
| `cargo test --locked --offline` | 11/11 |
| `ctest --test-dir csrc/build-libs` | 11/11 — `test_model_desc`, `test_safetensors`, `test_engine_resources`, `test_collective`, `test_attention`, `test_gdn`, `test_moe`, `test_mla`, `test_norm`, `test_rope`, `test_library_ops` |
| `cabal test all --enable-tests` | `infer-tests` 41/41, `infer-generation-tests` 14/14 |
| Qwen3.8-27B golden capture | bitwise identical to the pre-refactor baseline (`max_abs == 0`) |
| `tests/test_engine.py` (27B vs independent PyTorch logits) | 20/20 greedy tokens; logit RMS 0.02–0.04 |
| `tests/test_longseq.py` | 433-token chunk-split self-consistency (RMS 0.029) and 128-token generation coherence |
| `tests/test_tp.py --devices 0,1` (TP2) | identical greedy tokens, per-step logit RMS ≤ 0.05 |

## Supported model families

The descriptor + layer-kind design carries one committed snapshot per family in
`descriptors/`; the C engine stays family-agnostic.

| Family | Structure | Result (2× A40, oracle = independent PyTorch/Transformers) |
|---|---|---|
| Qwen3.8-27B | 16 full-attention (GQA, partial RoPE, output gate) + 48 GatedDeltaNet | golden bitwise after every refactor; vs reference RMS 0.02–0.04 |
| Qwen3-4B | dense attention | greedy tokens all hit |
| Qwen3-30B-A3B | MoE, 128 experts top-8 | 20/20 greedy; layer-split vs EP2 tokens identical |
| Qwen3-Next | GDN + MoE + shared expert (synthetic checkpoint) | 16/16 greedy |
| DeepSeek-V2-Lite | MLA + fine-grained MoE (64 experts top-6 + shared) | 16/16 greedy; per-step RMS 0.043–0.571 (gate 1.0), every token hit |

Cross-family note: the engine accumulates in FP32 while some references reduce
expert outputs in BF16, so the gate is per-family logit RMS plus *mandatory*
greedy-token agreement — never a relaxation of the top-1 check.

## Placement and hardware

- **Layer-wise split** (default): each device owns a contiguous block of layers.
- **Replicated tensor parallel** (`--tp N`): every rank runs the whole model with
  weight shards taken from the descriptor's `role_shards`, with a sub-layer
  all-reduce. Token-identical to the layer split, RMS ≤ 0.045.
- **Expert parallel** (`--ep N`): whole MoE experts split across ranks, routed
  partials merged once in FP32 (no all-to-all). Router and shared experts stay
  replicated.
- Combined TP+EP is deliberately rejected.
- Architectures: one `libengine.so` carries SASS for `86;89;90a` plus PTX.
  Runtime-verified on sm_86 (A40, full-model tests) and sm_89 (L20, operator
  suite including the FLA cubins). sm_90a is compile- and artifact-verified
  (`cuobjdump`) only — there is no H200 here.

## Recently completed

**Resource safety and regression gates** (finished and verified 2026-09-23;
commits `6d6e306`, `6417c72`, `fe2c2d4`). This closed the issues raised in the
prior code review while keeping legal models numerically unchanged:

- Generation: zero/negative budget never calls the engine; one consistent stop
  condition for the first and later tokens; engine errors propagate instead of
  becoming an empty "success".
- Tokenizer FFI: a lossless length-query capacity protocol plus an owned
  incremental decode handle, so a UTF-8 character split across tokens is emitted
  once complete rather than silently truncated.
- Safetensors: bounded schema parsing with checked shape/offset arithmetic;
  malformed or truncated shards are rejected rather than partially indexed.
- Allocation ownership: every layer-owned buffer is registered before anything
  that can fail, so a failed init releases everything it allocated.
- Cross-device copies: receivers record a completion event and the leader waits
  on it before reusing the source buffer.
- MLA: the shared-memory limit is computed up front and over-long sequences are
  rejected at `engine_create` and at the kernel entry.
- Expert parallel: the routed partial is merged in FP32 and rounded once.

The plan document for that work (`plan-resource-safety-regression.md`) has been
removed now that it is complete; the outcome lives here.

One regression was found and fixed during that build: Qwen3.8-27B ships a
rank-5 Conv3D vision tensor alongside its 1198 text tensors, and a strict rank
cap rejected the whole directory. The cap is now a parse boundary (8), with a
CPU case pinning the behaviour.

## Known gaps

- **Expert-parallel equivalence re-run is pending.** The EP-vs-layer-split check
  after the FP32 merge landed was stopped before it finished. The gate is
  unchanged (`test_tp.py --ep 2`: identical greedy tokens, logit RMS ≤ 0.05).
- **sm_90a has no runtime gate** — compile- and artifact-verified only.
- **Long context is not supported.** The MLA attention kernel's shared-memory
  budget caps the cached sequence length (a 16K context does not fit) and both
  `engine_create` and the kernel entry reject anything longer.
- **Greedy decoding only.** There is no temperature/seed sampling, no top-k/top-p
  and no batching; a CPU sampler migration is proposed in
  plan-numeric-contract.md (milestones T0–T4) but not implemented.
- **Performance is not optimized** (correctness-first): decode is a single-token
  full forward, and the logits are computed for the last position only.

## Documents

| File | Role |
|---|---|
| [README.md](../README.md) | Build, test entry points, usage, phase status, model weights |
| [design.md](design.md) | Architecture rationale, per-family layout differences, testing strategy |
| [plan-numeric-contract.md](plan-numeric-contract.md) | **Proposal, not implemented** — trainer (SFT/OPD/GRPO/DAPO/GSPO/PPO), bounded-staleness async RL, temperature-sampling migration, and an inference-optimization track (fusion, W4A16, speculative decoding) |
| [reference-output.json](reference-output.json) | Transformers reference tokens for the 27B debugging prompt |
