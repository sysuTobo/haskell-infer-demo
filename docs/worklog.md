# Work Log

A running snapshot of what this project can do today, what has been verified and
where it is knowingly incomplete. [README.md](../README.md) describes the
component layout and how to build and test; [design.md](design.md) holds the
architecture rationale; [plan-numeric-contract.md](plan-numeric-contract.md) is the
proposal for the trainer, RL and inference-optimization work — of which **Stages 0
through 3 are implemented here**: the execution manifest with capture provenance
(specified in [manifest-contract.md](manifest-contract.md)), the region inventory with
its cross-case harness (`csrc/regions.c`), the feasibility/invariance experiments that
measured what those regions actually guarantee, and the trainable runtime's parameter
lifecycle (`csrc/include/train.h`, `src/Infer/Trainer.hs`). All four are described
below.

Last updated: 2026-09-25.

## Verified

Everything below was run and passed on 2× A40 46 GB (sm_86) unless a line says
otherwise. No GPU result here is implied by a document edit alone.

Every row below was re-run on 2026-09-25 against the tree each stage left, and the
golden row was re-checked once more after Stage 3's engine changes (the training
runtime does not touch the inference path). The sm_89 and sm_90a lines in "Placement
and hardware" are the exceptions, and they say so.

| Gate | Result |
|---|---|
| `cargo test --locked --offline` | 11/11 |
| `ctest --test-dir csrc/build-libs` | 23/23 — `test_model_desc`, `test_safetensors`, `test_manifest`, `test_manifest_hashes`, `test_region_inventory`, `test_backward`, `test_engine_resources`, `test_collective`, `test_attention`, `test_gdn`, `test_moe`, `test_mla`, `test_norm`, `test_rope`, `test_region_cases`, `test_backward_kernels`, `test_gdn_invariance`, `test_attention_invariance`, `test_gemm_invariance`, `test_attention_lse`, `test_train`, `test_train_forward`, `test_library_ops` |
| `ctest test_backward` (CPU, Stage 4) | the Stage-4 region table against the Stage-1 inventory in both directions; masked CE, reverse KL, the token/sequence clipped objective and the group advantage reduction against an independent FP64 implementation with its gradient checked by central difference; AdamW against FP64 in PyTorch's own order (including the bias-correction/eps ordering) plus the BF16 publication; the checkpoint round-trip with every failure mode refused; a deterministic fixture overfitting 32/32 through this stage's own loss and optimizer, and a 12+8 resumed run landing bitwise on the uninterrupted 20-step parameters, moments and cursor |
| `ctest test_backward_kernels` (Stage 4, 2× A40) | every backward against a double-precision definition or a central difference of one: elementwise gates, residual branches, plain/Gemma RMSNorm, the GDN L2 norm, the gated norm, embedding (repeated ids summed), RoPE (the transposed rotation inverting the forward's), the Q/gate re-interleave, GEMM dX/dW, masked CE, AdamW, conv1d (d_x, d_weight, d_bias, d_state_in), GDN prepare (d_conv_out, d_a, d_b, d_A_log, d_dt_bias) — all 1e-8…1e-6 except the finite-difference rows at 1e-7…1e-3; attention forward+LSE vs the definition (2.1e-3 BF16 out, 1.5e-3 LSE) and its backward vs the FD of the definition (dQ 2.5e-4, dK 2.8e-4, dV 1.0e-3), with an analytic double reading of the device LSE reproducing the FD, so the base-2 convention is pinned; the GDN core backward with a nonzero initial state and a nonzero final-state gradient, one chunk and three chunks both matching the same reference (≤1.0e-7), d_state_start included; dQ and the whole GDN core backward bitwise reproducible, dK/dV reported (atomics) |
| `ctest test_train` (CPU, Stage 3) | tying (35 specs into 34 logical parameters on Qwen3-4B's descriptor), frozen parameters with no training state, the borrow/update/free lifetime rules, publication with derived-copy refresh, the accumulation schedule, replica sync, and the teacher-forcing plan |
| `ctest test_train_forward` (Stage 3, synthetic checkpoint) | all-position forward 12/12 top-1 vs a transformers forward (rms 0.004); teacher-forced selection and log-probabilities vs the reference's own log-softmax (gap 0.007); tied roles one logical parameter with two readers; a no-op publication bitwise inert; an updated tied weight and an updated GDN norm weight each matching a torch recomputation with the same edit (rms 0.004-0.01, 12/12); an update refused while a step is live |
| `cabal test infer-trainer-tests` (CPU, Stage 3) | the Haskell teacher-forcing plan and the C implementation of the same schedule agree across shifts, masks, forced labels and explicit positions |
| `ctest test_gdn_invariance` (claim B) | prepare **bitwise invariant 51/51** across lengths 2..128 and splits at L-1, L/2, 64; core output ≤ 9.2e-5 (7.2e-3 relative, ≈1 BF16 ULP), FP32 `ssm_state` ≤ 4.2e-4; a 64+64 split at L=128 is bitwise identical |
| `ctest test_attention_invariance` (claim C) | 264 tilings (head_dim 128/256 x GQA 24x4/24x8/8x4 x kv_len 1..269 x single-query/split), **202 bitwise**, worst 2.0e-3 relative (≈half a BF16 ULP); split-KV disabled is recorded from code, not assumed |
| `ctest test_gemm_invariance` (claim D) | 251 BF16-output measurements (33 bitwise), worst 5.4e-3 relative (**below one BF16 ULP**); 222 FP32-output measurements (22 bitwise), worst 3.0e-5 relative — invariance falsified and quantified |
| `ctest test_attention_lse` (claim E forward) | the LSE is available: base-2 (`log2 SUM e^s`, identified by a 69x margin over the two misreadings), layout `[qo_len, num_heads]` f32, fully written, and requesting it leaves the output **bitwise unchanged** |
| `tests/test_pp.py` (claim A) | Qwen3-4B on one device vs a two-device layer split: **324/324 taps byte-identical**, logits bitwise identical, identities a clean placement-only difference; the strict verdict is `rejected` because the device *set* differs in provenance |
| `tests/attention_backward_feasibility.py` (claim E backward) | the analytic backward from `(q,k,v,LSE)` reproduces torch.autograd in float64 to 1.7e-16; consuming the LSE without the log2 conversion moves `dv` by 3.04; a paired library (torch SDPA bf16) differs by 3.2e-3 forward / 2.8e-2 on gradients |
| `ctest -R test_region_inventory` (CPU) | the inventory covers the plan's 21 in-scope regions and nothing else, agrees with the manifest registry in both directions, and every `exact` pair is backed by that registry's `deterministic` |
| `ctest -R test_region_cases` (2× A40) | 18/18 registered `exact`/`unverified` pairs adjudicated; 8 `exact` pairs bitwise (output and persistent state); 7 unsupported shapes/cases rejected with a named reason; no trainer case offered by any of the 21 regions |
| `cabal test all --enable-tests` | `infer-tests` 66/66, `infer-generation-tests` 15/15, `infer-trainer-tests` 9/9 (the Haskell and C teacher-forcing plans must agree) |
| `manifest --model-dir <27B> --gpus 0,1 --check` | exit 0: a 12868-byte canonical document over 27 recorded regions carrying all three identities plus the parameter identity, build/runtime provenance fully established, two queries byte-identical, no unestablished provenance path, and every digest re-derived independently by `tests/manifest_check.py`. The byte count and the identities are the ones the *pre-Stage-4* registry produced; Stage 4 edits three registry rows (`attention_core`'s LSE, `masked_loss`'s stage, and `backward` from `not_implemented` to the backward inventory), which is a numerical-policy change: `ctest test_manifest`'s identity matrix is the gate that says only `numerical_policy_id` (and the `regions_sha256` inside it) moves, and the literals are re-derived by the same command |
| Two independent 27B captures, strict comparison | bitwise identical (`max_abs == 0` on every array) and verdict `admitted` (exit 0) — the identities, parameter identity and provenance all agree, over the two captures' own `numerical_policy_id` (`012c264c…314f0e`, taken before Stage 4's registry edit) |
| Pre-refactor golden vs a fresh 27B capture | bitwise identical (`max_abs == 0` on every array) with verdict `legacy/unverified` (exit 2) — the older capture's numeric arrays are compared, but nothing about its identity is invented. Re-run against the Stage-1 tree and again after Stage 3's engine changes, which is what shows the conv-activation export, the registry change and the training runtime are all numerically inert on the inference path |
| Qwen3.8-27B golden capture | bitwise identical to the pre-refactor baseline (`max_abs == 0`) |
| `tests/test_engine.py` (27B vs independent PyTorch logits) | 20/20 greedy tokens (one step is a BF16 tie the reference itself reports as equal-maximal); per-step logit RMS 0.015–0.038; descriptor round-trip and the invalid-input/capacity checks pass; chunk boundaries 129-token rms 1.06 and 64+64+1 rms 1.33, both top-1 stable |
| `tests/test_longseq.py` | 433-token chunk-split self-consistency (top-1 271/271, RMS 0.029) and 128-token generation coherence (4-gram repetition 0.008) |
| `tests/test_tp.py --devices 0,1` (TP2) | identical greedy tokens; per-step logit RMS 0.014–0.045 on the 10-token arm and 0.029–0.039 on its 129-token boundary arm (gate 0.05) |

The manifest rows ran on the real 27B with `semantic_id`
`890c5472…fabde`, `numerical_policy_id` `012c264c…314f0e`, `deployment_id`
`56f6e4aa…d6f4d` and `parameter_manifest_sha256` `4cb768d4…bbcc` over 1199
tensors (device ordinals 0,1; 32 layers each; layer-split placement), 27 recorded
regions and a fully established provenance. The region table moved twice as the plan
progressed — Stage 1 added `kv_write`, `conv_silu` and `masked_loss`
(`regions_sha256` `29b44b3a…506734`), and Stage 3 implemented the per-position FP32
log-softmax/gather that `masked_loss` had been waiting on, so it is no longer
`not_implemented` (`regions_sha256` `8456b213…e35357`) — and each time *only*
`numerical_policy_id` moved, which is exactly the projection the Stage-0 contract
states (a region-table change is a numerical change; the semantic, deployment and
parameter identities are untouched). The `git_commit` this build reports is `1f23880`,
which is *not* the tree that was built: see the build-revision gap below.

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

**Backward, losses and optimizer** (plan Stage 4, verified 2026-09-25). The engine can
now differentiate every region the dense/dense-hybrid path runs, and the three pieces a
trainer needs — the losses, the optimizer and a resumable checkpoint — are one tested
module rather than a plan:

- `csrc/include/backward.h` + `csrc/backward.c` are CUDA-free: the differentiation
  convention (a cast is identity for gradient propagation, so a pre-cast function's
  local derivative uses the pre-cast value while an operand derivative uses the rounded
  one — the output gate's gate-gradient is exactly that case), the losses (masked cross
  entropy, dense reverse KL, the token- and sequence-level clipped objective, the group
  advantage reduction), AdamW in PyTorch's own order, a CRC-32-checked checkpoint over
  parameters, optimizer moments, RNG and data cursor, and a counter-based RNG.
- `csrc/kernels/backward.cu` and `csrc/kernels/backward_paired.cu` are the kernels: the
  elementwise gates and branches, the four norms, embedding scatter-add, the RoPE
  transpose and the Q/gate re-interleave, GEMM dX/dW, the fused log-probability row,
  AdamW, GDN conv1d and prepare, and the two paired regions — attention from its saved
  base-2 LSE, GDN core from its retained chunk-boundary states.
- **the plan's Stage-4 table is code.** `backward_region_info` carries one row per table
  row, each naming the Stage-1 regions it differentiates; `ctest test_backward` walks it
  in both directions and `ctest test_backward_kernels` re-checks it against the
  inventory, so a row that is missing and a row that invents a region both fail. The
  Stage-1 GPU harness changed with it: its assertion that the backward region must be
  registered `not_applicable` (because no backward existed) is replaced by one that the
  backward registers no forward case pair *and* that its eleven Stage-4 rows are all
  implemented and all name inventoried regions.
- **the GDN rounding boundaries are documented before differentiating**, as this stage
  requires: `design.md` now carries the per-step table (the BF16 boundary after each L2
  norm and its 1e-6-inside-rsqrt epsilon, the BF16-rounded beta, the FP32 log-decay, the
  conv's single output rounding, and the gated norm's three). Two consequences are
  enforced rather than noted: the gated norm's weight gradient belongs to the BF16
  source, not to Stage 3's FP32 derived copy; and the duplicated key heads' contribution
  must be summed once, not once per group member — the bug the prepare gate caught.
- **an LSE is a statistic, not a tensor.** `kernel_attention_lse` is the engine's
  forward with a real LSE buffer (Stage 2 measured that asking for it leaves the output
  bitwise unchanged), and `kernel_attention_backward` recomputes the softmax from it, so
  no `[T,T]` probability tensor is retained. The base-2 convention is pinned by
  measurement rather than by reading the source: an analytic double reading of the
  device's own LSE reproduces the finite difference of the definition, and the backward
  needs *no* extra ln-2 (the ln-2 in Stage 2's Python harness belonged to a harness
  forward that defined P as 2^(s-L), which is not the softmax).
- **the gate's cases are all in the fixture.** Nonzero GDN initial state and nonzero
  final-state gradient; one chunk and three chunks crossing internal boundaries, both
  against the same reference; repeated embedding ids summed rather than overwritten; the
  GQA group's dK/dV summed across its queries; a tied weight getting one optimizer
  update; and a masked loss. One complete AdamW step matches an FP64 implementation of
  PyTorch's order, including the bias-correction/eps ordering that separates it from the
  textbook form.
- **a fixture overfits and resumes.** A deterministic separable fixture trains to 32/32
  in 20 steps through this stage's own loss and optimizer; a second run reaches the same
  bits; and a run that is saved at step 12, wiped, restored and continued for 8 more
  lands bitwise on the uninterrupted 20-step parameters, both moments and the data
  cursor. The model-level SFT overfit is Stage 5's, whose gate re-runs this against the
  transformer.
- Two limits are recorded, not papered over. The gradient pairing with the library
  forwards is not bitwise — attention's backward recomputes P in FP32 while the
  forward's PV product rounds it to BF16 (dV's ~1e-3 residual is exactly that gap,
  predicted by Stage 2's claim E), and the GDN core backward differentiates the
  recurrence rather than the cubin's `(I + A)^{-1}`/BF16-MMA decomposition; both are
  Stage 6 alignment work. And the GDN core backward's per-coordinate reduction is
  O(tokens × head_dim) rather than blocked, which is correct and reproducible but is
  the first thing to fix when training throughput matters.

Three bugs are worth remembering because each was a *silent* wrong answer rather than a
crash, and each is now a shape the gate rejects. A host `for` loop over a device pointer
does not fail to compile — it segfaults, and it appeared three times (the conv1d and
prepare wrappers zeroing a gradient, and the GDN core seeding its workspace); the fix is
a kernel-side zero or a `cudaMemcpyAsync`. A wrapper that offsets the V plane of the KV
cache *and* a kernel that offsets it again leaves dV correct (it needs no V) while
silently corrupting dP, hence dQ and dK — which is why the gate compares all three
gradients and why the offset is now explained where it is done. And a test fixture whose
operands are "a BF16 value times a constant" is a different operand after the upload
rounds it, so the finite difference is of a function the kernel never evaluated; every
fixture now rounds to the storage type explicitly.

**Trainable runtime and parameter lifecycle** (plan Stage 3, verified 2026-09-25).
The inference path is untouched; what is new is the ownership a trainer needs.

- **The parameter store** (`csrc/train.c`, `csrc/include/train.h`) is the plan's
  ownership object: logical parameters with tying already resolved, each with a
  version, an FP32 master, a BF16 compute weight whose buffer *is* the engine's own
  weight, gradient and optimizer-slot presence, the devices that hold a copy, and the
  derived copies that depend on it. It is deliberately CUDA-free — the buffers are
  opaque slots — so every rule in it is exercised by a CPU test and the same rules
  govern the engine.
- **Three rules are rejections, not comments.** An update fails while any context or
  step borrows the current version; a store or step cannot be destroyed while a step
  is live or its values are retained; and an update cannot end while a derived copy of
  a published parameter is stale, which is what makes "refresh `gdn_norm_f32`, not
  just its BF16 source" enforceable rather than aspirational.
- **The teacher-forced forward** evaluates the LM head one row at a time on the
  device, so a loss never materialises a [tokens, vocab] tensor, and returns
  natural-log log-probabilities for the selected positions (next-token label shift, a
  prompt/padding mask, explicit positions). One sequence at a time: independent
  sequences are never flattened into one causal sequence.
- **The training step** retains what a backward will consume — a mixer output, an ffn
  output and a residual per layer (the residual stream is overwritten in place, which
  is exactly why it has to be retained) plus the GDN chunk-boundary state per GDN
  layer under a full-sequence schedule — and records alias rules and free points.
  Stage 4 adds the backward that consumes them.
- **The Haskell side** (`src/Infer/Trainer.hs`, `Infer.Trainer.Types`,
  `Infer.Trainer.Plan`) owns the schedule and the typed handles: the store and step
  are opaque newtypes, and the long-running calls are imported @safe@ so another
  Haskell thread can run while they work. The schedule is implemented twice on
  purpose, in Haskell and in C, and `cabal test infer-trainer-tests` requires the two
  to agree — a split that is only asserted drifts.

**The bug this gate caught** is worth recording: the first publication cast *every*
master into its compute weight, and the store's masters were allocated zeroed, so the
first update silently wiped every parameter the caller had not written. The engine
now seeds each master from the loaded weight in FP32, and the gate compares the model
*before and after* a no-op publication instead of only comparing two
post-publication readers with each other.

**Feasibility and invariance experiments** (plan Stage 2, verified 2026-09-25).
Stage 1 measured region case pairs; Stage 2 asked what those measurements *mean*
and where the guarantees stop. Six experiments, one per claim:

- **A — pipeline-parallel inertness, scoped.** Qwen3-4B (a model that fits one A40;
  the 27B does not) with all 36 layers on device 0 versus a two-device layer split:
  bitwise identical logits and 324/324 byte-identical tap dumps covering every
  layer's residual stream, mixer output and ffn output. The strict manifest verdict
  is `rejected`, and that is the contract working: a one-device and a two-device run
  differ in *runtime provenance* (the device list itself), and strict admission
  requires provenance to agree. The identities confirm the difference is placement
  only — semantic, numerical-policy and parameter identities equal, `deployment_id`
  different. The pass is scoped to the tested configuration (same build, two A40).
- **B — GDN decomposition, with attribution.** Raw inputs and a nonzero initial
  state held fixed, lengths 2..128, arms of whole/recurrent/half/chunk64/tail1. The
  **prepare stage is bitwise invariant across every arm** (51/51 measurements), so
  the residual difference is the core's: output ≤ 9.2e-5 (7.2e-3 relative, about one
  BF16 ULP) and FP32 `ssm_state` ≤ 4.2e-4. A split aligned to the FLA chunk size
  (64+64 at L=128) is bitwise identical; the recurrent path and a one-token tail are
  where it moves. This is the region-level attribution the plan asked for, and it
  also says the projections are *not* implicated — they are claim D's.
- **C — attention tiling.** 264 tilings over head_dim 128/256, GQA 24x4/24x8/8x4 and
  kv_len 1..269, comparing one full call against a prefix/suffix split and against
  one call per query over a cache built on the host. 202 are bitwise identical; the
  worst is 2.0e-3 relative (about half a BF16 ULP) and the deviations cluster where
  a split boundary does not align with the query tile. Split-KV is *recorded as
  disabled from code*, not assumed: the engine passes a null workspace and
  FlashInfer's dispatcher clears `partition_kv` when the workspace is null.
- **D — GEMM shape invariance, falsified and quantified.** One M-row call against
  one call per row (decode's shape) and against prefix/suffix splits, over the real
  projection shapes (N x K of 1024..248320 x 5120, 5120 x 6144, 5120 x 17408) and
  M 1..434. Only 33 of 251 BF16-output measurements are bitwise, worst 5.4e-3
  relative (below one BF16 ULP); only 22 of 222 FP32-output measurements are
  bitwise, worst 3.0e-5 relative. cuBLAS selects by shape, so the accumulation order
  changes with M — this is the measurement that turns six Stage-1 `unverified` pairs
  into measured exceptions.
- **E — the attention forward/backward pair.** The forward *can* produce what a
  backward needs: FlashInfer's prefill writes a base-2 LSE (`log2 SUM e^s`, layout
  `[qo_len, num_heads]` f32, every element written) when asked, and asking leaves the
  attention output bitwise unchanged. The convention is identified empirically (the
  two plausible misreadings are 69x and 1000x further off) because the kernel folds
  log2(e) into the scores so it can use exp2. On the backward side, the analytic form
  from `(q, k, v, LSE)` reproduces `torch.autograd` in float64 to 1.7e-16 including
  the GQA grouping, and the two compatibility requirements are *demonstrated*: an
  unconverted LSE moves `dv` by 3.04, and a paired library (torch SDPA in bf16)
  differs from this forward by 3.2e-3 forward and up to 2.8e-2 on gradients, so a
  library backward cannot be bolted on without a paired gradient check. Resource
  estimate: 45 KB of saved state per token per layer, 2.96 GB for 4096 tokens x 16
  attention layers, and recomputing P from the LSE avoids a 4096x4096 bf16 tensor
  (0.8 GB per layer).

The experiments also produced the stage's registry change: six case pairs that Stage
1 could only call `unverified` are now measured `exception`s carrying the tested
architecture, the tested shapes, and bounds rounded up to two significant digits
(`attention_core` x2, `gemm_bf16`, `gemm_fp32_lmhead`, `gdn_core` x2). Nothing was
promoted to `exact`: measuring a bound is not establishing a reduction order, and
the manifest registry's `deterministic` verdict is still the only thing that
authorises an `exact` pair.

The dW guarantee is defined in
[plan-numeric-contract.md](plan-numeric-contract.md#stage-2--feasibility-and-invariance-experiments):
parameter gradients are independent of the microbatch grouping only under a *fixed
accumulation schedule*, bitwise equality is claimed only under that schedule,
adding tokens is explicitly out of scope, and the forward per-row invariance
measured here is a necessary input to that guarantee rather than the guarantee
itself.

**Region inventory and cross-case harness** (plan Stage 1, verified 2026-09-25).
The engine can now say which forward *cases* a region is reachable under and what
has been established about each pair of them, and a device harness re-derives the
claims rather than asserting them:

- `csrc/regions.c` inventories the 21 in-scope dense/dense-hybrid regions — every
  one of the plan's Stage-1 bullets — with its inputs, outputs, persistent state,
  saved-for-backward values and case availability, and registers each reachable
  case pair as `exact`, a quantified `exception`, `unverified` or
  `not_applicable`. MLA, MoE and the TP/EP collectives are listed as *excluded*
  with the reason, so "not inventoried" cannot be mistaken for "not applicable".
- Eight cells of the Stage-0 `cases` column claimed something unreachable and are
  removed: `train_forward` in six rows, `recompute` in one, and `backward` in
  `gemm_bf16`. That column is hashed into `numerical_policy_id`, so advertising a
  traversal that does not exist is a policy claim, not a note;
  `test_region_inventory` now fails if any manifest row names a traversal case.
- `ctest test_region_inventory` (CPU, no GPU and no weights) checks the two things
  a reader cannot: that the plan's bullet list maps onto the inventory region by
  region and that nothing else is invented, and that the inventory cannot drift
  from the Stage-0 registry — every inventory region must be a manifest region,
  every manifest region must be inventoried or excluded, and an `exact` pair is
  accepted only where that registry already says `deterministic`.
- `ctest test_region_cases` runs the fixtures: identical inputs and identical
  *nonzero* persistent state under each applicable case, adjudicated against the
  registered verdict. It fails if a registered pair was skipped, and re-runs the
  unsupported-shape/case rejections.

Measured on 2× A40 (sm_86), one line per registered pair:

| Region | Case pair | Verdict | Output max_abs | State max_abs |
|---|---|---|---|---|
| embedding | chunked_prefill / decode | exact | 0 | — |
| rope | chunked_prefill / recurrent_prefill | exact | 0 | — |
| q_gate_split | chunked_prefill / decode | exact | 0 | — |
| kv_write | chunked_prefill / decode | exact | 0 | 0 (whole cache) |
| attention_output_gate | chunked_prefill / decode | exact | 0 | — |
| residual_add | chunked_prefill / decode | exact | 0 | — |
| silu_mul | chunked_prefill / decode | exact | 0 | — |
| conv_silu | chunked_prefill / recurrent_prefill | exact | 0 | — |
| rmsnorm | chunked_prefill / decode | unverified | 0 | — |
| per_head_norm | chunked_prefill / decode | unverified | 0 | — |
| attention_core | chunked_prefill / decode | unverified | 0 | 0 (cache) |
| attention_core | chunked_prefill / tail1 | unverified | 0 | — |
| gemm_bf16 | chunked_prefill / decode | unverified | 0 | — |
| gemm_fp32_lmhead | chunked_prefill / decode | unverified | 4.77e-07 | — |
| gdn_conv1d | chunked_prefill / recurrent_prefill | unverified | 0 | 0 (shift register) |
| gdn_core | chunked_prefill / recurrent_prefill | unverified | 6.10e-05 | 3.69e-04 (FP32 ssm_state) |
| gdn_core | chunked_prefill / tail1 | unverified | 3.05e-05 | 3.06e-04 |
| gdn_gated_norm | chunked_prefill / recurrent_prefill | unverified | 0 | — |

On the real 27B the Stage-1 tree also re-established the Stage-0 gates rather than
assuming them: `manifest --check` exits 0 on a 12868-byte canonical document whose
provenance is fully established, with two byte-identical queries and every digest
re-derived independently by `tests/manifest_check.py`; the pre-refactor golden is
still bitwise identical (`max_abs == 0` on every array, verdict
`legacy/unverified`, exit 2), which is what shows exporting the conv activation and
changing the registry are numerically inert; and the model-only harnesses re-run
green — `tests/test_engine.py` against the independent PyTorch reference (20/20
greedy tokens, per-step logit RMS 0.015–0.038), `tests/test_longseq.py` (433-token
chunk-split top-1 271/271, RMS 0.029) and `tests/test_tp.py --devices 0,1`
(identical greedy tokens, RMS 0.014–0.045). The last two matter here because they
are the model-level counterparts of the chunked-prefill and placement case pairs
this stage inventories at region level.

Two things about that table. A measured zero under `unverified` is *not* promoted
to `exact`: `exact` requires the Stage-0 registry to say `deterministic` (no
cross-thread reduction, no library tiling decision), and FlashInfer, cuBLAS and
the FLA cubins are not. Promotion is a Stage-2 decision, taken by the experiment
that establishes the reduction order, not by one measurement. And the GDN core's
chunk-versus-recurrent difference is small *at the region boundary* (6e-05 output,
3.7e-04 state) while whole-model logit RMS between prefill schedules has been
observed around 1.0 — the decomposition difference accumulates across 48 GDN
layers, which is exactly why the region boundary is where attribution has to start
(Stage 2 claim B).

`region_ffi` also measures what a region boundary costs: host-side enqueue
(argument validation plus the launch) against device time, on real enqueues with
no host synchronization between them. For the tiny fixtures here the host side
dominates — 1.9–5.5 µs per enqueue against 3.7–10.0 µs of device time, 68–99%
host share, and 7.0/7.1 µs per thread with two host threads enqueuing through their
own handles and non-blocking streams — which is a statement about these fixtures
and about the engine's
Haskell FFI being *model-level* today: the number is the C region entry point, not
a `ccall`, and Stage 3's region handles are what would make the two comparable.

**Execution manifest and capture provenance** (plan Stage 0, verified 2026-09-25).
The engine can now answer *which numerical execution did this run observe*, as
three content identities over canonical JSON blocks plus the provenance a bitwise
comparison has to agree on — and a capture records the document it was taken
under:

- `engine_manifest` / `haskell-infer-demo manifest` report `semantic_id` (dimensions,
  layer/role semantics, tied-role relations, the mathematical conventions),
  `numerical_policy_id` (the region/case → implementation binding table with its own
  digest, the effective constants, dtype/rounding and reduction choices, and the
  sampler's transform — the GEMM algorithm policy is recorded as *unpinned*, not as
  a default), `deployment_id` (placement, devices, the per-layer owner, the shard
  plan) and the immutable parameter identity (a canonical tensor index over 1199
  tensors; the raw content hash is `null` unless a caller pays for it).
- Build provenance comes from a header the build step generates over its own
  artefacts (git revision, CUDA toolkit and target lists, the Triton/FLA versions the
  AOT generator asserted, hashes of the generated kernels and of the FlashInfer
  header compiled against, the toolchain flag digest). Runtime provenance comes from
  CUDA queries. An unestablished fact is reported as
  `unknown`/`unavailable`/`unsupported`/`unspecified` and *refuses* strict admission
  rather than being defaulted into looking comparable.
- Comparisons have modes, not tolerances: `strict` (identities, parameter identity,
  placement unless a scoped claim is declared, and provenance must all agree; the
  numeric arrays must then still be bitwise identical), `--deployment-scoped`
  (reported as a scoped exception, never an identity-level pass), `diagnostic`
  (reported for attribution, never a pass, exit 3) and `legacy` (a document without a
  manifest version stays numerically comparable with an explicit `legacy/unverified`
  result and no invented identity, exit 2).
- The committed region registry records the implementation each region ran and, per
  region, determinism/mechanism/RNG-dependency separately (27 regions: 11
  deterministic by construction, 13 unverified because a library or cross-device
  reduction order is not established, 3 `not_implemented` — the backward region,
  the proposed loss region and the proposed sampler).
- The canonical encoding is what makes the digests checkable elsewhere: keys sorted
  by byte value, no whitespace, integers bare, non-integer constants as decimal
  strings, printable ASCII only. `test_manifest` pins the identity matrix without a
  GPU, `test_manifest_hashes` re-derives every digest with `hashlib`, and the Haskell
  side parses the same documents.

Two contract violations and one classification gap were caught by the engine's own
document and by re-reading the plan, rather than by the unit tests: a missing
top-level `manifest_version` (which made the manifest unparseable), a region flag
emitted as an integer where the contract fixes a boolean, and the sampler's
transform/arithmetic not being part of the numerical policy. All three now have
gates, including a type check in the Python verifier and a variant in the identity
matrix that requires a sampler change to move `numerical_policy_id`.

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

- **The manifest's build revision can lag the tree that was built.**
  `csrc/gen_build_info.cmake` reads `git rev-parse HEAD`, but the custom command
  that runs it is declared with the generated AOT kernels and the script itself as
  its dependencies — not the repository revision — so a commit that changes no
  kernel does not regenerate `build_info.h`. Observed on 2026-09-25: a build from a
  tree at `0df2bd8` plus uncommitted Stage-1 work still reported `git_commit`
  `1f23880`, the revision current when the header was last regenerated. The
  identities are unaffected (adding three regions to the table moved
  `numerical_policy_id` and nothing else, as the Stage-0 design predicts), and the
  generated-kernel and flag digests still move when the arithmetic does, but
  `provenance.build.git_commit` is not by itself evidence of which tree ran. The fix
  is to regenerate the header on every build while touching its mtime only when the
  content changes (an always-run custom target around a content-compare write),
  which is a build-system change needing its own re-verification.
- **The manifest's per-device `kernel_path` and `triton_cubin_arch` are a
  selection rule, not an observation.** Which binary the driver actually launched
  is not queryable per kernel, so the manifest reports what the build's target
  lists plus the device's compute capability select for it, and the field names say
  `selected`.
- **The forward regions are not yet registered under the `train_forward` case.**
  Stage 3 added `engine_train_forward` (one sequence, all positions, teacher-forced),
  so that case is now reachable in principle, while the Stage-1 inventory still lists
  the trainer traversal as unavailable everywhere. Registering the case means fixtures
  that compare the training forward against the inference forward at region level, and
  half-doing it would put a coverage claim in the registry that no fixture backs, so it
  is an open item rather than a partial edit.
- **The training path has no backward, losses or optimizer yet.** Stage 3 delivers the
  ownership objects, the teacher-forced forward and the retention a backward needs;
  the masked loss reduction, the gradient kernels and the optimizer are Stage 4, which
  is why the `masked_loss` region is still recorded as unfinished in the registry.
- **The FP32 masters are per-parameter, not sharded.** A store with training state for
  a 27B checkpoint would need ~54 GB of masters plus gradients and two optimizer slots
  on a 46 GB device, so the gate attaches training state for the synthetic model and
  the 27B path is bookkeeping-only. Sharding or offloading the optimizer state is
  Stage 4's problem and is stated here rather than discovered later.
- **No region has an established reduction order beyond the elementwise ones.** The
  registry marks 13 of 27 regions `unverified`, and the GEMM algorithm policy is
  recorded as `cublas_default_heuristic_unpinned`: an exact claim about that region
  is unsupported until the plan's Stage 2 pins or replaces it. Stage 1 measures the
  case-pair deltas (above) but promotes none of them to `exact`.
- **Six case pairs are registered `exception`, the rest are `unverified` or
  `exact`.** The exceptions carry bounds measured over the Stage-2 matrix and
  rounded up; they are regression ceilings, not proofs, and the underlying library
  reduction orders are still not established (that is what promotion to `exact`
  would require).
- **`capture_logits.py --compare` exits 0 on a rejected comparison.** Its exit code
  reports that the comparison *ran*, not what it found; the strict exit-code
  contract (0 admitted / 1 rejected / 2 legacy / 3 diagnostic) belongs to the
  `manifest-compare` CLI. Scripted gating must read the printed verdict (or use the
  CLI). Stage 2's claim-A fixture was initially fooled by this before checking the
  identities directly.
- **The LSE's last bits are the kernel's, not an identity.** FlashInfer's LSE agrees
  with a float64 `log2 SUM e^s` to ~2e-3 absolute (the same reduced-precision softmax
  denominator the PV product uses), which is why the Stage-2 probe identifies the
  convention by its margin over the alternatives rather than gating on an exact
  match. A backward that must match the forward to the bit has to recompute the
  scores the same way, which is a Stage-4 concern.
- **The trainer traversal is registered as unavailable, not as implemented.**
  `train_forward`, `eval_no_autograd`, `recompute` and `backward` are reachable from
  no region, and both new gates refuse a region that advertises them: registering a
  case with no API would claim a trainer Stages 3-4 have not built.
- **`weights.content_sha256` is null** unless a caller chooses to hash 50 GiB of
  tensor data; the parameter-manifest digest over the tensor index is what strict
  admission compares.
- **Backward regions do not exist**, so the registry records them as
  `not_implemented` rather than assuming a determinism verdict for them.
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
| [manifest-contract.md](manifest-contract.md) | The execution manifest: canonical encoding, field ownership and projections, the region determinism registry, and the comparison modes |
| [plan-numeric-contract.md](plan-numeric-contract.md) | **Proposal, not implemented** (Stages 0-3 are implemented; see above) — trainer (SFT/OPD/GRPO/DAPO/GSPO/PPO), bounded-staleness async RL, temperature-sampling migration, and an inference-optimization track (fusion, W4A16, speculative decoding) |
| [reference-output.json](reference-output.json) | Transformers reference tokens for the 27B debugging prompt |
