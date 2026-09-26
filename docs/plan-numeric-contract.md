# Plan: Numerical Execution Contract, Haskell Training, and Asynchronous RL

Status: **Stages 0-5 are implemented**, and so are Stage 6's decision layer,
Stage 7's GSPO/GRPO objectives, Stage 8's lag-zero admission protocol and the
temperature-sampling migration (T0-T4). Still proposals: the asynchronous GPU half of
Stage 8 (snapshots, device leases, publication transfer, throughput measurement) and
the inference-optimization track (F/Q/S). What has landed: the versioned execution
manifest with capture provenance, the region inventory with its cross-case harness, the
feasibility/invariance experiments, the trainable runtime's parameter lifecycle, the
backward/loss/optimizer layer, the synchronous SFT/rollout baseline, the
numerical-alignment decision, the GSPO/GRPO group objectives and their gate, the
bounded-staleness rollout queue, and the host binary64 temperature sampler with its
request-owned RNG — see
[manifest-contract.md](manifest-contract.md), [worklog.md](worklog.md),
`csrc/regions.c`, `csrc/include/train.h`, `csrc/include/backward.h`,
`csrc/include/alignment.h`, `csrc/include/rollout_queue.h` and
`src/Infer/Sampling.hs`.
Revised after the design review, 2026-09-23.
Extends [design.md](design.md). This document separates current capabilities,
proposed interfaces, measured observations and hypotheses requiring experiments.
No GPU result is implied by this documentation revision.

## Goals and boundaries

Build a Haskell-orchestrated trainer over the same C/CUDA region implementations
as inference, supporting SFT, on-policy distillation (OPD), GRPO, DAPO, GSPO and
PPO. Establish a synchronous correctness baseline before bounded-staleness
asynchronous RL. The first implementation target is a small dense or dense-hybrid
model, on one device or with layer-wise placement; not every existing inference
family becomes trainable in the first release.

A [temperature-sampling migration](#temperature-sampling-migration) moved the default
generation policy from greedy to categorical sampling (**implemented**, T0-T4), while
retaining explicit greedy regression at temperature 0. It needed no training or CUDA
changes and supplies the sampling foundation the Stage-5 rollout's warm-up described.

A separate [inference optimization track](#inference-optimization-track) covers
operator fusion, weight-only quantization and speculative decoding. It shares
the contract and measurement infrastructure, but neither blocks the initial
trainer nor inherits trainability or exactness from it. All sampling/optimization
APIs, formats and milestones below are proposals; this revision changes
documentation only and authorizes no implementation or GPU experiment.

Sharing implementations reduces duplication but does **not** establish either
bitwise agreement or shared parameter storage by construction:

- Different GEMM shapes, attention tiling, GDN decompositions, dtype boundaries
  and reduction orders can move bits inside the same library.
- The existing engine owns its weight allocations, recurrent state and scratch.
  It exposes only model-level inference calls and the final position's logits.
- Synchronous phases can read the same BF16 parameter buffers only after an
  ownership/update protocol exists. Concurrent actors must not read parameters
  while the learner modifies them; they need immutable versions or serialization.
- Same-layout snapshot copying is not resharding, but it still costs memory,
  bandwidth and publication latency. Different actor/learner layouts add
  resharding. Neither cost disappears merely because both use Haskell.

The numerical contract answers: **for the same weights and token prefix, which
execution changes preserve which outputs?** Asynchronous RL adds a different
question: **which weight version generated this trajectory, and how may the
current learner use it?** Bitwise kernels cannot remove policy lag.

IsoExec [1] motivates explicit execution cases, composition, identities and
invariance claims. Its reported mismatch reduction and overhead are results of
its workload, not estimates for this engine. It reports no meaningful reward
improvement in its reported comparison. Our initial justification is correctness,
debuggability and attributable experiments, not promised reward or speed gains.

## Current-state audit

| Area | Present | Missing or not established |
|---|---|---|
| Orchestration | Haskell model/configuration/placement/generation; C layer/chunk loops; the fixed training traversal (Stage 3: `Infer.Trainer`) | Activation lifetimes, region-level opaque handles |
| Inference API | `engine_create/prefill/decode/reset/destroy`; final-row logits; the trainer API and the all-position retained forward (Stages 4-5) | Multi-device training, OPD/DAPO/PPO and an online RL loop |
| Mixers/FFNs | Full attention, GDN, MLA; dense and MoE | The GDN mixer's backward is not chained into the layer walk, and MoE/MLA stay outside the first trainer allowlist (Stages 4-5) |
| Placement | `Pipelined` and `Replicated tp ep`; separate TP and EP | Combined TP+EP; trainer collectives; PP bitwise-invariance evidence |
| GDN | Chunked path plus recurrent `tokens == 1`, including one-token tails | Cross-case bitwise invariance; prepare/core/backward pairing |
| Attention | One FlashInfer prefill dispatcher; null split-KV workspace | Invariance across query/KV lengths, cached and training layouts |
| GEMM | cuBLAS BF16 and FP32-output paths | Batch-invariant forward; specified dW/gradient accumulation order |
| Descriptor | Strict flat schema and canonical C formatting | Semantic/numerical digests and resolved execution manifest |
| Artifacts | nvcc SASS `86;89;90a` plus `90-virtual`; Triton cubins `86;89;90` | Capture provenance identifying the actual selected kernels; Triton has no PTX fallback |
| Weights | Engine-owned BF16; derived GDN norm FP32 copy; shared trainable ownership with tied-gradient merging and derived-copy refresh (Stage 3) | Refresh of derived copies under a concurrent writer, and any multi-device gradient merge |
| Fusion | FlashInfer attention, SiLU-multiply and GDN gated norm; **the dense gate/up projections are one N = 2I GEMM over packed weights with a row-interleaved `[T, 2I]` activation** (F1), which took prefill M=128 on Qwen3-4B from 28.36 ms to 22.79 ms | The remaining combined projection layouts (Q/K/V, GDN QKVZ/BA), residual-add/norm fusion (F2), and the measurement of each on the deployment target |
| Quantization | Weight-role loading requires BF16; GEMM uses BF16 inputs with FP32 compute | Packed low-bit weights, scales, format validation, quantized GEMM and quality gates |
| Speculative decoding | Batched prefill internally, but only final-row logits; reset clears all sequence state | Draft/target orchestration, per-position verification, prefix rollback and GDN state snapshots |
| Sampling | One `stepToken` at both generation entry points; temperature categorical sampling in binary64 with a request-owned splitmix64 RNG; `--temperature`/`--seed` with early validation; greedy is the explicit temperature-0 mode; sampler and raw-model logprobs recorded per selection (T0-T4) | A GPU or batched sampler, top-k/top-p truncation, and the migration's performance (host selection time, GC, TTFT, tokens/s) |
| Training | The parameter lifecycle, the backward/loss/AdamW layer and the SFT bring-up, with checkpoint/resume and independent gradient tests (Stages 3-5) | OPD/DAPO/PPO, an online RL loop, and more than one device |
| Rollout | Greedy single-request inference; an engine-driven stochastic rollout bound to a borrowed version, group objectives (GSPO/GRPO) and a bounded admission queue (Stages 5-8) | An online RL loop, asynchronous GPU scheduling (snapshots, device leases, publication transfer) and throughput measurement |

This table is the audit the plan started from; each stage's own status block below
records what has since landed and what it left open.

Evidence anchors: `csrc/engine.cu::load_role`, `compute_logits`, `forward_tokens`;
`csrc/kernels/fla_gdn.cu::kernel_fla_gdn`;
`csrc/kernels/attention.cu::launch_single_prefill`;
`src/Infer/Placement.hs::Policy`; `csrc/CMakeLists.txt`.

`tests/capture_logits.py` currently skips `meta`, which contains a descriptor
path rather than its contents. `engine_describe` supplies canonical descriptor
text, not a content identity. `csrc/triton/build_aot.py` asserts Triton/FLA
versions; FlashInfer and CUDA/cuBLAS provenance comes from other build inputs.
There is no existing single assertion covering the whole numerical stack.

Snapshot coverage is also narrower than the directory: the model-directory
round-trip in `tests/Spec.hs` checks Qwen3.8 conditional on `INFER_MODEL_DIR`, not
every `descriptors/*.json`. Existing inference tests must remain intact; the
trainer needs its own explicit family/shape allowlist and gates.

## Numerical observations, not proofs

1. Earlier experiments recorded logit RMS around 1.0–1.3 between recurrent and
   chunked GDN execution. The algorithmic fork is a plausible source, but an
   external report of a similar phenomenon does not prove our whole-model error
   has the same cause or rule out another bug. Isolate inputs and persistent
   state at the region boundary before assigning causality.
2. `prefill(433)` versus `prefill(150)+prefill(283)` previously gave logit RMS
   0.029. Both the GEMM shapes and the GDN call-local 64-token grouping change.
   Being on the chunked path does not keep its reduction schedule fixed. This
   experiment does not isolate cuBLAS. The existing `rms <= 5` gate is only a
   coarse state-loss guard, not evidence of train/inference agreement.
3. Attention uses the same entry point for prefill/decode, but passes changing
   `tokens` and `seq_len` into FlashInfer. Null workspace establishes the selected
   no-split-KV mode, not invariant internal tiling. Attention remains in scope
   until a direct cross-case test discharges the claim.
4. PP moves whole layers and BF16 boundaries without splitting their reductions.
   Bitwise inertness is plausible for the same GPU architecture and effective
   kernel configuration, not proved across arbitrary devices or builds. TP/EP
   already reorder reductions and are not covered by this claim.

Reference correctness, run-to-run determinism, cross-case invariance and training
quality are different gates. A deterministic wrong formula passes the first
kind of bit comparison but fails an independent mathematical reference. Forward
invariance also does not imply reproducible optimizer trajectories: backward,
gradient accumulation, RNG consumption and batch order need their own contracts.

## Implementation stages and gates

No day estimates are assigned before the relevant feasibility spikes. Stages
0–2 preserve arithmetic; later stages deliberately introduce training interfaces
and possibly new kernels. Passing a metadata gate is not passing a numeric gate.
The sampling milestones use T0–T4 and the optimization milestones use F/Q/S
labels rather than renumbering these stages. Sampling can start independently;
optimization depends on the applicable Stage 0–2 evidence, not on completing all
training or asynchronous-RL work.

### Stage 0 — Versioned contract and capture provenance

**Status: implemented and verified, 2026-09-25.** The manifest, its canonical
encoding, the field ownership and projection rules and the comparison modes are
specified in [manifest-contract.md](manifest-contract.md); [worklog.md](worklog.md)
records the gates. The gate below is met:

- canonical round-trip and hash tests run without a GPU (`ctest test_manifest`,
  `ctest test_manifest_hashes`, `tests/ManifestSpec.hs`), and the identity matrix
  requires a semantic or numerical change to move the matching id only, a
  placement-only change to move `deployment_id` only (and to need a declared scope
  before it is admitted), and a provenance or weight change to move no identity;
- missing, mismatched or unestablished provenance fails strict admission; the
  engine's own document reports an unknown fact as such instead of defaulting it;
- on the real 27B the pre-refactor golden still passes the legacy numeric gate
  (bitwise identical) alongside a fresh strict comparison of two independent
  captures (`admitted`), and a manifest-less capture yields `legacy/unverified`
  rather than a pass.

Keep the architecture descriptor portable. Add a separate versioned execution
manifest/query rather than injecting runtime GPU facts into committed family
snapshots. It references the canonical descriptor and is resolved from actual
build/runtime state, not caller-supplied library labels.

Define canonical encodings and explicit field ownership before hashing:

| Identity | Required contents and comparison rule |
|---|---|
| `semantic_id` | Architecture dimensions, layer/role semantics, tied-parameter relationships, mathematical norm/RoPE/GDN conventions; weights themselves are a separate identity |
| `numerical_policy_id` | Region/case-to-implementation bindings; `max_chunk`, FLA chunk size, epsilon/theta and other effective constants; dtype/rounding boundaries, reduction tree, GEMM output type/algorithm policy, attention split-KV setting, backward determinism and RNG mapping |
| `deployment_id` | Placement, allocation and transfer choices; excluded from equality only when a scoped invariance claim covers the actual variation |
| Build/runtime provenance | Implementation source/generated-kernel or artifact hashes, FlashInfer/FLA/Triton and CUDA/cuBLAS versions, compiler flags, actual per-device architecture and selected SASS/cubin/PTX-JIT path, driver/runtime versions |

Some constants describe both the mathematical function and its numerical
realization; the canonical schema must specify their projections into the two
identities rather than silently omit or disagree on them. Unknown numerical
settings are unsupported for an exact claim, not implicit defaults.

Generation captures additionally record the resolved sampling configuration,
PRNG/version, seed and draw mapping. Temperature transforms and sampler arithmetic
are numerical-policy fields, while the concrete seed/state is per-request replay
data; changing only temperature must not change model/weight identity.

A capture also records checkpoint/parameter-version identity, token IDs, cases,
shapes, mask/position convention and the full manifest. A runtime `PolicyVersion`
is distinct from `numerical_policy_id`: optimizer updates change the former
without changing the latter. Use a run identifier plus monotonic version and an
immutable parameter manifest/checksum for replay provenance, not just a file path.

Comparison modes:

- **Strict admission:** require matching semantic/numerical/weight/input
  identities and compatible build/runtime provenance before bit comparison.
  Different artifact provenance needs an explicit reviewed compatibility test;
  the same version string alone is insufficient.
- **Diagnostic/refactor comparison:** allow a deliberate manifest difference,
  print its field-level diff and compare arrays, but do not call that a strict
  contract pass. This enables testing whether a changed implementation is inert.
- Legacy golden files lack provenance. Keep numeric comparison available with
  an explicit `legacy/unverified` result; do not invent their identity or silently
  bless them. Establish a new provenance-bearing baseline on a verified build.

For each backward region, record determinism (`deterministic`, `nondeterministic`,
`unverified`), mechanism (e.g. atomic accumulation) and RNG dependency separately.
A seed does not order atomics. Deterministic claims include gradient accumulation,
optimizer kernels and, where applicable, cross-device reductions.

Gate: canonical round-trip/hash tests; semantic and numeric changes alter the
appropriate IDs; justified deployment-only changes do not fail that scope;
missing/mismatched provenance fails strict admission; unchanged arithmetic still
passes the legacy numeric gate and a fresh strict capture comparison.

### Stage 1 — Region inventory and cross-case harness

**Status: implemented and verified, 2026-09-25.** The inventory is
`csrc/regions.c` (`csrc/include/regions.h`); the device harness is `ctest
test_region_cases` and the CPU gate over the inventory is `ctest
test_region_inventory`. The gate below is met:

- every in-scope operation of the dense/dense-hybrid path is inventoried with its
  inputs, outputs, persistent state, saved-for-backward values and case
  availability, and nothing else is: `test_region_inventory` walks the bullet list
  below region by region and also rejects an inventoried region the plan does not
  list;
- the inventory and the Stage-0 registry cannot drift apart — every inventory
  region must be a region the manifest registry names, every manifest region must
  be inventoried or explicitly excluded (MLA, MoE, TP/EP), and `exact` is accepted
  only where that registry already says `deterministic`;
- fixtures run identical inputs and identical persistent state under each
  applicable case pair and adjudicate against the registered verdict: `exact` must
  come out bitwise identical for output *and* state, `unverified` is measured and
  reported (with the plan's coarse state-loss guard where the region owns state),
  `not_applicable` must not run at all, and the run fails if a registered pair was
  silently skipped;
- unsupported shapes and cases fail explicitly, and that failure is asserted: a
  chunk beyond the descriptor's `max_chunk`, a sequence beyond `max_seq_len`, a
  non-contiguous SiLU pair, a wrong FLA key-head count, an over-long FLA token
  count, a negative element count, a zero head dimension, and every trainer
  traversal case on every region;
- model-only and region harnesses stay independently runnable — the region harness
  needs no checkpoint and no engine handle, and `tests/test_engine.py`,
  `tests/test_longseq.py` and `tests/test_tp.py` are untouched;
- registering a region is observability, not a trainer: nothing here traverses the
  model for training, owns an activation lifetime or computes a gradient, and the
  harness adds no arithmetic to the engine.

Registered verdicts are deliberately conservative. `exact` is claimed only for a
single elementwise pass, a row gather or a permutation, and only where the Stage-0
registry already says `deterministic`. Everything whose reduction or tiling order
comes from a library (FlashInfer, cuBLAS, the AOT FLA cubins) stays `unverified`;
promoting a pair that measures zero to `exact` is a Stage-2 decision, not a
Stage-1 one. No pair carried an `exception` yet, because an exception has to carry
tested shapes, an architecture and a measured max_abs/rms; **Stage 2 has now
produced them** (six pairs, below). `region_ffi` prints the region entry-point host cost against its device
cost; the engine's Haskell FFI is model-level today, so that number is the C
boundary (argument validation, workspace arithmetic, enqueue), not a `ccall`, and
Stage 3's region handles are what would make the two comparable.

Five gaps in the Stage-0 registry surfaced while writing the inventory. Three are
regions the registry did not name, and it now does: `kv_write` (the KV cache
write), `conv_silu` (the GDN conv activation) and `masked_loss` (the trainer's FP32
differentiable log-softmax/loss, kept distinct from `sampler_softmax_cdf` as
below). The first two are deterministic elementwise regions that the coarse
`attention_core` and `gdn_conv1d` rows had been covering, and the activation was
not exported at all (a `__global__` local to `layers.cu`), so no region harness
could have reached it — it is now `kernel_silu_inplace` alongside the other region
entry points. The other two are kinds of coverage claim the registry should not
have made — eight cells in total: `train_forward` in six rows, `recompute` in one,
and `backward` in `gemm_bf16`, none of which any region could be exercised for. That column is hashed into
`numerical_policy_id`, so it must not advertise coverage a region cannot be
exercised for; those claims are removed, and `test_region_inventory` now refuses a
manifest row that names a traversal case. The traversal itself has since been built —
Stages 3–5 deliver the teacher-forced forward, the step, the backward, the losses and the
synchronous rollout — so `train_forward` is reachable in principle now, and what is
missing is the region-level train-vs-inference fixtures a cell would have to stand on.
That is an open item the gap list records; the cell stays out until a fixture can back it.

Inventory the initial **dense/dense-hybrid** path, not a supposed thirteen-region
vocabulary covering the whole framework:

- Embedding gather, plain/Gemma RMSNorm, per-head norm, BF16 and FP32-output GEMM,
  residual addition, LM head, FP32 log-softmax/gather and masked losses.
- Attention Q/gate split, RoPE, KV write, attention core and output gate.
- GDN conv1d, conv SiLU, prepare (Q/K L2 normalization, head expansion,
  a/b/A_log/dt_bias transforms), core and gated norm.
- Dense MLP SiLU-multiply.

Inventory the proposed host-FP64 temperature softmax/CDF and RNG mapping as a
separate generation-only region when T0–T4 is implemented; do not equate it with
the trainer's FP32 differentiable log-softmax/loss region.

MLA, MoE routing/expert combine, TP and EP require additional regions and are
explicitly outside the first trainer allowlist. Keep their inference regression
coverage; do not register unsupported training cases for them.

Cases include chunked prefill, recurrent single-token prefill/tail, decode,
teacher-forced trainer forward, no-autograd evaluation, recomputation and
backward. Describe region inputs, outputs, strides/dtypes, mutable state,
saved-for-backward values and case-specific availability. KV write may be absent
from the initial full-sequence trainer without forcing mathematical drift in the
attention it computes.

Register each case pair as **exact**, **quantified exception**, **unverified** or
**not applicable**. Exceptions carry tested shapes, architecture, max error/RMS,
state comparison and a reason; they do not authorize a bitwise whole-model claim.
A tag at a call site provides observability, not an implementation of the trainer.

Measure region-FFI overhead using real enqueue, handle validation and threading
behavior. A hypothetical 20 ns `ccall` cost excludes marshalling, synchronization
and runtime scheduling; it is not a proof that the boundary is free.

Gate: every in-scope forward operation is inventoried; fixtures compare identical
inputs and persistent state across applicable cases; unsupported shapes/cases
fail explicitly. Model-only and region harnesses remain independently runnable.

### Stage 2 — Feasibility and invariance experiments

**Status: implemented and verified, 2026-09-25.** The experiments are
`ctest test_gdn_invariance` (B), `ctest test_attention_invariance` (C),
`ctest test_gemm_invariance` (D), `ctest test_attention_lse` (E's forward half),
`tests/test_pp.py` (A) and `tests/attention_backward_feasibility.py` (E's backward
half). The gate below is met:

- **A** has evidence, scoped: Qwen3-4B (a model that fits one A40) run with all 36
  layers on device 0 and with a two-device layer split produces bitwise identical
  logits and **324/324 byte-identical tap dumps** (every layer's residual stream,
  mixer and ffn output). The strict manifest verdict is `rejected`, not `admitted`,
  because a one-device run and a two-device run differ in runtime provenance (the
  device list itself) and strict admission requires provenance to agree; the
  identities show a clean placement-only difference (semantic, numerical-policy and
  parameter identities equal, `deployment_id` different). The pass is scoped to the
  tested configuration: same build, two A40 of the same architecture.
- **B** has region-level results with attribution: with raw inputs and a nonzero
  initial state fixed, the **prepare stage is bitwise invariant across every arm**
  (51/51 measurements over lengths 2..128 and splits at L-1, L/2, 64), so the
  remaining difference belongs to the core. The chunkwise core's output stays within
  9.2e-5 (7.2e-3 relative, about one BF16 ULP) and the FP32 `ssm_state` within
  4.2e-4; the recurrent path (one token per call) and a one-token tail are the arms
  that differ, while a split aligned to the FLA chunk size (64+64 at L=128) is
  bitwise identical.
- **C** has region-level results: 264 tilings (head_dim 128/256 x GQA 24x4, 24x8,
  8x4 x kv_len 1..269 x {single-query, split at 1, L/2, 64, 128}) with **202 bitwise
  identical**, and a worst case of 2.0e-3 relative (about half a BF16 ULP) at
  head_dim 128 / 24x4 / L=63. The deviations cluster where a boundary does not align
  with the query tile. Split-KV is *recorded* as disabled and is verifiable in code,
  not assumed: `kernel_attention` passes a null workspace and FlashInfer's
  dispatcher clears `partition_kv` when the workspace is null
  (`flashinfer/attention/prefill.cuh`).
- **D** has region-level results at the real projection shapes: **one M-row call
  against one call per row and against prefix/suffix splits**, over N x K of
  1024..248320 x 5120, 5120 x 6144 and 5120 x 17408 and M 1..434. Only 33 of 251
  BF16-output measurements are bitwise, with a worst case of **5.4e-3 relative
  (below one BF16 ULP of the output)**; only 22 of 222 FP32-output measurements are
  bitwise, worst **3.0e-5 relative**. Invariance is falsified, and the falsification
  is quantified rather than waved through: the six `unverified` pairs of Stage 1
  have become measured `exception`s.
- **E** has a concrete path and a resource estimate: the forward **can** produce the
  state a backward needs — a base-2 LSE (`log2 SUM e^s`, identified against the two
  plausible misreadings by a 69x margin) of layout `[qo_len, num_heads]` f32 — and
  asking for it leaves the attention output **bitwise unchanged**. A backward
  consuming `(q, k, v, LSE)` is well defined: the analytic form reproduces
  `torch.autograd` in float64 to 1.7e-16, including the GQA head grouping, and the
  two compatibility requirements are demonstrated rather than asserted (feeding the
  LSE without converting the log2 moves `dv` by up to 3.04, and a paired library
  (torch SDPA, bf16) differs from this forward by 3.2e-3 forward and up to 2.8e-2 on
  gradients, because it is a *different function*). Resource estimate: 45 KB per
  token per layer of saved state (q/o/dq, k/v/dk/dv bf16 plus the LSE), 2.96 GB for
  4096 tokens x 16 attention layers, and recomputing P from the LSE avoids a
  4096x4096 bf16 probability tensor (0.8 GB per layer).

**The dW guarantee, defined separately from the forward properties above.** For a
fixed set of logical tokens and a fixed loss normalization, the gradient of a
parameter is required to be independent of how those tokens are grouped into
microbatches **only under a fixed accumulation schedule**: each microbatch's
contribution is computed by the same kernel configuration, and the contributions
are summed in a defined order (a fixed microbatch index order) that is not
reordered. Bitwise equality is claimed only under that schedule. Without it, the
same tokens legitimately produce a different dW, because grouping changes the GEMM
shapes — claim D measured that as 3.0e-5 relative for FP32 accumulators and below
one BF16 ULP for BF16 outputs. **Adding tokens is not covered**: it changes dW and
is a different question from this one. The forward per-row invariance that claims
B-D measure is a necessary input, not the guarantee: it bounds each microbatch's
contribution, while the guarantee is about the accumulation *across* microbatches,
which needs its own test once a trainer exists (Stage 3-4).

| Claim | Experiment and decision |
|---|---|
| A: PP inertness | A model that fits one GPU, fixed weights/prompt/build, one-device versus two-device layer split on matching GPU architectures. Compare intermediate boundaries and logits bitwise; scope any pass to tested configurations. The 27B checkpoint is not the single-A40 fixture. |
| B: GDN decomposition | Hold raw/prepared inputs and nonzero initial state fixed; compare whole/chunked/recurrent paths, outputs and final state, including lengths around 1, 64 and 128. Distinguish prepare, core and GEMM effects. |
| C: attention invariance | Record disabled split-KV, then compare the same causal positions from full, split and single-query execution with identical Q/K/V; vary KV length, GQA and supported head dimensions. Later include trainer layout and saved/recomputed statistics. |
| D: GEMM invariance | Synthetic fixed inputs/weights; vary row count, prefixes/splits, M=1, boundary/tail sizes and actual projection N/K. Cover BF16 outputs and FP32 LM-head outputs. A failure falsifies invariance; one passing 128-vs-64+64 test does not prove it. No full checkpoint load is needed. |
| E: attention forward/backward pair | Determine a usable paired implementation or an explicitly compatible saved-state/recomputation adapter, including LSE, masks, GQA and determinism. Current forward passes `lse=nullptr`; a backward from another library cannot just be plugged in without a compatibility/gradient check. |

Separately define the desired dW guarantee: identical logical tokens and loss
normalization across microbatch splits, with a fixed accumulation schedule if
bitwise equivalence is claimed. Adding extra tokens legitimately changes dW;
this is not the forward per-row batch-invariance property.

Gate: A has evidence or a narrower scope; B–D have region-level results, not only
whole-model RMS; E has a concrete forward/backward path and resource estimate.
Retain state-loss tests rather than replacing them with an arbitrary tighter RMS.

### Stage 3 — Trainable runtime and parameter lifecycle

**Status: implemented and verified, 2026-09-25.** The C side is
`csrc/include/train.h` + `csrc/train.c` (the ownership objects, CUDA-free and
CPU-tested) with the engine-facing half in `csrc/engine.cu`
(`engine_train_attach/begin_update/write_master/publish/step_*/forward`); the Haskell
side is `src/Infer/Trainer.hs` with `Infer.Trainer.Types` and `Infer.Trainer.Plan`.
Gates: `ctest test_train` (CPU: tying, frozen parameters, the borrow/update lifetime,
publication and derived copies, the accumulation schedule, replicas, the
teacher-forcing plan), `ctest test_train_forward` (the synthetic checkpoint on a
device), `cabal test infer-trainer-tests` (the two implementations of one schedule
must agree). The gate below is met:

- **the tiny-model all-position forward agrees with an independent reference.** The
  synthetic Qwen3-Next checkpoint (4 layers, vocab 1024) is compared position by
  position against a `transformers` forward: 12/12 top-1 with logit rms 0.004. The
  teacher-forced selection (next-token label shift, prompt/padding mask, explicit
  positions) is checked against that reference's own log-softmax (worst gap 0.007) and
  the engine's device-fused path against its host path (5e-7).
- **tied roles stay tied.** Qwen3-4B's `lmHead` templates onto its embedding, so the
  store resolves 35 specs into 34 logical parameters with one master and two compute
  buffers, and a publication writes *both* readers. On the synthetic model given the
  same tie, a synthetic update of the tied parameter matches an independent torch
  forward whose two tensors were edited with the same values (rms 0.01, 12/12 top-1).
- **a synthetic parameter update refreshes all readers.** A master starts as the
  loaded weight in FP32; a no-op publication is bitwise inert on both the inference and
  the training reader (max change 0.0); and publishing an updated GDN norm weight
  matches a torch forward with the same edit (rms 0.004, 12/12) — which is only
  possible because the FP32 `gdn_norm_f32` copy is refreshed, not just its BF16 source.
- **lifetime tests reject update/free while a reader or step is active.** An update is
  refused while a context or a step borrows the version (naming the reader), a second
  window and a publication with no window are refused, a frozen parameter refuses a
  master write, a saved value another live value references cannot be freed, and a step
  with retained values cannot be destroyed until they are released.

The ownership objects are the plan's, and three of the rules are rejections rather
than assertions: `train_store_begin_update` fails while any reader borrows the store,
`train_store_destroy`/`train_step_destroy` fail while a step is live, and
`train_store_end_update` fails while a derived copy of a published parameter is still
stale — so "publishing follows BF16 casting and all derived-copy refreshes" is
enforced, not remembered. The store is CUDA-free on purpose: its buffers are opaque
slots, so the same rules govern the CPU test and the engine.

The teacher-forcing schedule is implemented **twice on purpose** — in Haskell (which
the plan says owns the traversal) and in C — and the test suite requires the two to
agree across shifts, masks, forced labels and explicit positions. A split that is only
asserted by a comment is a split that drifts.

**The bug the gate caught**, recorded because it is what a gate is for: the first
publication cast *every* master into its compute weight, and the store's masters were
allocated zeroed, so the first update silently wiped every parameter the caller had
not written. A master now starts as the loaded weight in FP32, and the gate compares
the model *before and after* a no-op publication instead of only comparing two
post-publication readers with each other.

This prerequisite is a real engine/FFI change, not covered by Stage 1 tagging.
Haskell owns the fixed model traversal, training schedule and typed opaque
handles; C/CUDA owns allocations, streams and kernel execution. No raw CUDA
pointers become user-facing Haskell values.

Specify the proposed ownership objects before implementing backward:

- A parameter store maps logical parameter IDs to FP32 master weights, BF16
  compute weights, gradients and optimizer slots. Frozen parameters omit unused
  training state. Tied roles map to one logical parameter; cross-device copies
  have explicit gradient-merge/update synchronization, not independent optimizers.
- Trainer and synchronous rollout contexts borrow a committed parameter version
  but own different scratch and sequence state. Updating requires exclusive
  ownership; publishing follows BF16 casting and all derived-copy refreshes.
  In particular refresh `gdn_norm_f32`, not just its BF16 source.
- A training-step context retains necessary forward activations/statistics and
  GDN chunk-boundary states until their backward consumers finish. Specify alias
  rules, free points and recomputation. A fixed backward sequence removes the
  need for generic dynamic autograd, **not** the need for saved values.
- Teacher forcing exposes selected-position logprobs/losses for every response
  token, with next-token label shift, prompt/padding masks and positions. Use
  chunked/fused LM-head/loss evaluation where needed to avoid retaining `[T,V]`
  logits. Do not flatten independent sequences into one causal sequence.
- Start with one sequence at a time plus defined gradient accumulation; genuine
  packed/variable-length batching is separate work. Full-sequence GDN gradients
  must cross internal chunk boundaries unless truncated BPTT is explicitly chosen.

Preserve the current model-level inference API while adding the training path.
Handle lifetime, publication barriers and error cleanup are required; the current
same-OS-thread contract forbids arbitrary calls into one engine from multiple
Haskell workers. Each future execution worker needs a bound owner thread and
exclusive context access. Long-running training calls must not freeze unrelated
Haskell scheduling; audit safe FFI calls and runtime capabilities.

Gate: tiny-model all-position forward agrees with an independent reference;
tied roles stay tied; a synthetic parameter update refreshes all readers;
lifetime tests reject update/free while a reader or backward context is active.

### Stage 4 — Backward, losses and optimizer

**Status: implemented and verified, 2026-09-25.** The CUDA-free half is
`csrc/include/backward.h` + `csrc/backward.c` (the differentiation convention, the
losses, AdamW, the checkpoint format and the RNG); the kernels are
`csrc/kernels/backward.cu` and `csrc/kernels/backward_paired.cu`. Gates:
`ctest test_backward` (CPU: the region table against the Stage-1 inventory, the losses
and AdamW against an independent FP64 implementation, the checkpoint's refusals, and a
tiny fixture that overfits) and `ctest test_backward_kernels` (every kernel against a
double-precision definition or a central difference of one). The gate below is met:

- **the plan's Stage-4 table is the region table, in both directions.**
  `backward_region_info` carries one row per table row, each naming the Stage-1
  inventory regions it differentiates; `test_backward` walks it both ways (every named
  region exists in the inventory; every region the table gives a backward is claimed by
  exactly one row), so a missing row and an invented one both fail.
- **the forward rounding boundaries were documented before differentiating**, as this
  stage requires: `design.md`'s GDN section now carries the per-step table (the BF16
  boundary after each L2 norm, the BF16-rounded beta, the FP32 log-decay, the three
  boundaries inside the gated norm). Two of its consequences are enforced rather than
  noted: the sigmoid chain is evaluated at the pre-cast value (a cast is identity for
  gradient propagation) while the operand derivative uses the rounded value, and the
  gated norm's weight gradient belongs to the BF16 source, not to Stage 3's FP32
  derived copy.
- **independent gradient checks.** Elementwise gates, residual branches, plain/Gemma
  RMSNorm, the GDN L2 norm, the gated norm, the embedding gather, RoPE (the transposed
  rotation must invert the forward's), the Q/gate re-interleave, GEMM dX/dW, the masked
  cross entropy, the fused log-probability row and AdamW are each compared against a
  double-precision analytic reference or a finite difference of one. Measured maxima are
  1e-8…1e-6, except the conv1d, prepare, attention and GDN-core rows, which are checked
  by central difference and land at 1e-7…1e-3.
- **the paired regions are paired.** `kernel_attention_lse` is the engine's forward with
  a real LSE buffer (Stage 2 measured that this leaves the output bitwise unchanged), and
  `kernel_attention_backward` recomputes the softmax from that base-2 LSE — the analytic
  double-precision reading of it reproduces the finite difference of the definition, so
  the convention is pinned rather than assumed. `kernel_gdn_core_backward` reverses the
  decay-before-prediction recurrence in FP32 from the retained chunk-boundary states.
- **the gate's cases are all in the fixture.** Nonzero GDN initial state *and* a nonzero
  final-state gradient; sequence boundaries crossed (one chunk and three chunks, both
  matching the same reference); repeated embedding ids (summed, not overwritten); head
  duplication (the GQA group's dK/dV summed across its queries); tied weights (the
  gradient sum and exactly one optimizer update); and a masked loss.
- **one complete AdamW step** matches an FP64 implementation of PyTorch's own order —
  including the bias-correction/eps ordering, which is what separates it from the
  textbook form — plus the BF16 publication as the master's round-to-nearest-even.
- **an overfit fixture converges and resumes.** A deterministic separable fixture is
  trained through this stage's own loss and optimizer to 32/32 in 20 steps; a second run
  reaches the same bits; and a run of 12 steps that is saved, wiped, restored and
  continued for 8 more lands bitwise on the same parameters, both optimizer moments and
  data cursor as 20 uninterrupted steps. The model-level SFT overfit is Stage 5's, whose
  gate re-runs this against the transformer.
- **determinism is tested separately from closeness.** The attention backward's dQ is
  bitwise reproducible (one owner per coordinate) and the GDN core backward is bitwise
  reproducible for all six gradients; the attention dK/dV group sums use atomics, so
  they are *reported* rather than required, and `csrc/backward.c`'s registry says which
  regions are fixed-order.

Two limits are recorded rather than papered over. The gradient *pairing* with the
library forwards is not bitwise: the attention backward recomputes P in FP32 while the
forward's PV product rounds it to BF16 (the gate's residual, ~1e-3 on dV, is that gap
and Stage 2's claim E predicted it), and the GDN core backward differentiates the
recurrence rather than the cubin's `(I + A)^{-1}`/BF16-MMA decomposition. Both are
Stage 6 alignment work. And the GDN core backward's per-thread reduction is
O(tokens x head_dim) per coordinate rather than the blocked form a production kernel
would use: it is correct and reproducible, and its cost is the next thing to fix if
training throughput (not correctness) becomes the constraint.

The mathematical GDN convention is the corrected decay-before-prediction rule
in `design.md` and `tests/kernels/test_gdn.cu::delta_reference`. Its exact
forward rounding boundaries must be documented before differentiating it.

| Region | Required backward or update work |
|---|---|
| Embedding and tied LM head | Gather backward/scatter accumulation, tied-gradient sum and exactly one optimizer update |
| Residual and elementwise gates | Branch-gradient accumulation; SiLU/sigmoid/product derivatives |
| RoPE and Q/gate split | Transposed rotation with the same tables/convention; re-interleave gradients |
| Plain/Gemma/per-head/gated norms | Input and weight gradients; distinguish L2 normalization from RMSNorm and raw weight from weight+1 |
| GEMM including FP32 LM-head output | dX/dW, transpose/dtype contracts and fixed loss-scaling/accumulation rules |
| GDN prepare | Q/K L2 derivatives, reduction of duplicated heads, sigmoid/softplus/exp chain through a/b/A_log/dt_bias; explicit cast-gradient convention |
| GDN conv1d | Upstream backward plus saved inputs/state and verified weight-gradient reduction |
| GDN core | Chunked backward with correct initial/final-state derivatives; pair with the actual forward implementation |
| Attention core | Stage 2's paired forward/backward, saved or recomputed statistics and selected deterministic mode |
| Losses | Masked CE/log-softmax, dense reverse KL for the chosen OPD variant, token/sequence clipped objectives and advantage reductions |
| AdamW | FP32 master/m/v, documented gradient dtype, zeroing/accumulation, bias correction, decoupled decay and update/cast order |

The initial mixed-precision differentiation convention treats supported casts as
identity for gradient propagation, while derivatives consume the appropriate
saved rounded values. Check that convention against an independent PyTorch
implementation; do not finite-difference discontinuous BF16 casting and call the
result a gradient oracle.

Gate: independent FP32/FP64 mathematical gradient checks on small inputs where
applicable, then BF16 forward/backward comparison against the intended reference.
Include nonzero GDN initial state, sequence boundaries, repeated embedding IDs,
head duplication, tied weights and masked loss. Compare one complete AdamW step
(parameters, m/v and BF16 refresh), overfit a tiny deterministic SFT fixture and
test checkpoint/resume of parameters, optimizer, RNG and data cursor. Test
run-to-run determinism separately from numerical closeness to the reference.

### Stage 5 — Synchronous training/rollout baseline

**Status: the SFT bring-up and the synchronous rollout are both implemented and verified,
2026-09-26.** The layer backward is `csrc/backward_layers.cu` (the full-attention mixer with its fused
output gate and per-head norms, the dense MLP, the residual adds and all four norms, each
recomputed per sublayer from the three values Stage 3 retains); the step's entry points are
`engine_train_forward_retain` / `_loss` / `_backward` / `_apply` / `_zero_grads` /
`_export_state` / `_import_state` in `csrc/engine.cu`; the phase, budget, record and
sampler contract is `csrc/include/train_loop.h` + `csrc/train_loop.c`; and the rollout
driver is `engine_rollout_sample` in `csrc/engine.cu`, which generates a completion under a
live model inside a *borrowing* `TrainContext` and stamps the record with the version that
context borrowed. Gates: `ctest test_sft` (device: SFT and rollout) and `ctest
test_train_loop` (CPU). The gate below is met:

- **SFT is up on a real model.** `tests/synth/make_qwen3_dense.py` builds a 0.72M-parameter
  dense Qwen3 with **tied embeddings** and a prompt-masked target, and the engine trains it:
  the loss falls 7.0322 → 0.6084 over twelve steps and the reference implementation falls
  7.0318 → 0.5652 on the same fixture, so both overfit and neither is a no-op.
- **the step's loss is an independent implementation's loss.** The first step's loss —
  one forward over the same weights with the same mask — matches a `transformers` +
  `torch.autograd` run to 6.3e-05 relative. That is what pins the forward, the row-at-a-time
  final norm and LM head, and the masked reduction together.
- **the backward's direction is the reference's direction.** After one AdamW step at a
  large lr (where the step is `lr*sign(g)` and the sign of the weight move is the sign of
  the gradient), the tied parameter's gradient has cosine **0.9990** against torch's over
  all 131072 elements. This is the numerical check of the layer wiring; the loss matching
  alone would not catch a wrong gradient.
- **a resume restores the state exactly.** The exported training state (every trainable
  parameter's FP32 master and both optimizer moments) round-trips through import/export
  bitwise, so a resumed run continues the bias correction rather than restarting it, and
  the resumed tail reproduces the uninterrupted one to 1.4e-02 relative.
- **determinism is tested apart from closeness.** The forward is bitwise reproducible
  (the first loss is identical run to run); the backward's row-summed weight gradients
  accumulate with atomics, so the trajectories separate afterwards by 5.1e-03 relative —
  measured and attributed rather than called a mismatch, which is the split the plan asks
  for and Stage 4's registry records.
- **the rollout reads the model's own distribution.** `test_sft` draws **20000**
  single-token completions through `engine_rollout_sample` and compares them against the
  softmax of the very logits the rollout samples from. Two independent readings agree: the
  mean of `-log p` is 6.90837 ± 0.00162 against that distribution's entropy of 6.90495
  (**+2.11 sigma**), and of the 415 bins whose expected count exceeds 20 every one lands
  within 4.5 sigma of its probability (worst **2.94 sigma**). The recorded sampler
  log-probability is bitwise the model's, which is what temperature 1 with no truncation
  means, so the two columns a transformation would have to distinguish coincide.
- **seed behaviour and the terminal reasons are deterministic.** The same seed reproduces
  the completion bitwise, a shorter rollout is a prefix of the longer one, an eos the
  sampler hits ends the completion at one token with `TRAIN_TERMINAL_EOS`, and a
  completion that never hits eos runs to the length limit with `TRAIN_TERMINAL_LENGTH`.
- **the record is bound to the version the engine read.** `engine_rollout_sample` opens a
  *rollout* `TrainContext`, so an engine-level optimizer step is refused while the
  generation is live (`1 reader(s) still borrow version 0`), and the record's version is
  the one that context borrowed: `train_loop_record` accepts it and refuses the same record
  stamped one version ahead. A publication moves the store 0 → 1, after which both
  `train_loop_read_reward` and `train_loop_ratio` refuse the record while its recorded
  denominator is bitwise unchanged — the plan's "never reconstruct an old denominator using
  updated weights".
- **the two denominators are compared, not assumed.** For the same completion at the same
  version, the teacher-forced trainer's FP32 log-probabilities differ from the host FP64
  sampler's by **4.768e-07** at most, i.e. a sequence-level ratio of 0.99999976. The
  generation path reads a KV cache and the trainer a causal mask, so this is exactly the
  cross-case deviation the plan asks to be reported separately rather than claimed bitwise.
- **a group is one version, one configuration, one verifier.** Four completions of one prompt
  (differing only in their seed) are recorded into one `TrainGroup`; the group refuses a
  fifth generated *after* a publication even though the engine produced it. Each member's
  reward comes from a deterministic check on the completion — the gate's is deliberately
  trivial, because the point is that no reward model appears — the advantages reduce to
  +1.000/−1.000/+1.000/−1.000 at a mean of 0.5 and a population std of 0.5, and the
  degenerate case is a refusal the caller has to waive: two members that scored the same are
  a zero-variance group, admitted only with `allow_zero_variance` and then giving zero
  advantages. The sequence-level objective then consumes the engine's *own* record: ratio
  exactly 1 at unchanged parameters, the gradient's sign following the advantage, and a
  +0.25 nat/row move giving 1.284025 against the records' own 1.284025.
- **offloading the optimizer state is a declaration.** The budget says a rollout's step reads
  no optimizer state (`optimizer_bytes` is 0 against SFT's nonzero); the loop records what a
  caller actually shed, refuses a negative count or a declaration outside a phase, and clears
  the count at the boundary so no phase inherits another's answer. The engine does not move
  the buffers itself, which the limits below state.
- **the rollout half's rules hold as behaviour.** `test_train_loop` checks the phase
  budgets (only SFT holds optimizer state, activations and retained values), that a rollout
  *borrows* the store so an update is refused while it reads, that a phase boundary resets
  the sequence, that the sequence-level ratio at unchanged parameters is exactly 1, and that
  the host FP64 sampler's measured frequencies match the softmax it samples from within
  three standard deviations (0.6652/0.2447/0.0902 against 0.6652/0.2447/0.0900 over 200k
  draws).
- **the refusals are named.** Training on a pipeline or tensor-parallel placement, a step
  outside a phase, an optimizer step while a step is live, and a NaN logit at sampling
  time are each refused with a message.

Four limits are recorded rather than glossed over. The layer wiring covers the
full-attention mixer and the dense feed-forward: the **GDN mixer's backward** (its prepare,
conv and core regions) is Stage 4's kernels but is not yet chained into the walk, and MoE
and MLA stay outside the first trainer allowlist as Stage 1 says. Training is wired for a
**single device** — a pipeline split would have to move gradients between devices and a TP
placement would have to reduce sharded ones, and both are refusals today. The rollout
generates **one sequence per call and one record at a time**: the engine has no group
entry point, so a caller collects a group's G completions by looping (the plan's "serial
generation is sufficient for correctness"), the reward is the caller's verifier rather than
anything the engine computes, and the engine **does not move optimizer state off the device**
itself — a caller that sheds it says so and the loop records the declaration, while the
buffers stay allocated where they are. And the phase budgets are an **estimate from the
descriptor's shapes** rather than a measured allocation watermark, which is what a budget
needs to be chosen from but not what proves a phase fits.

First bring up SFT after Stages 3–4; it does not wait for cross-case bitwise
kernels. Then add phase-specific memory budgets and stochastic rollout. Reuse
the [temperature-sampling migration](#temperature-sampling-migration) rather
than adding another sampler here; ordinary CLI sampling does not wait for SFT.
Before RL admission, bind its selection records to immutable policy versions
and validate host-FP64 sampler versus trainer-FP32 logprob differences explicitly.

- Synchronous rollout borrows a committed BF16 parameter version. No optimizer
  write overlaps readers. Release/reset KV and GDN state at the phase boundary;
  retain optimizer state or explicitly offload it according to the budget.
- Capture sampled token IDs, per-token behavior logprobs, rewards, masks, terminal
  reasons and version IDs at generation time. Never reconstruct an old denominator
  using updated weights. Distinguish frozen reference/teacher from behavior policy.
- Begin RL tests with categorical sampling at temperature 1 and no top-k/top-p
  truncation. The mathematical sampling distribution is then model softmax;
  finite-RNG/CDF and FP64/FP32 differences remain subject to the sampling gates.
  Adding transformations requires recording both raw model logprobs and actual
  sampler logprobs, the transform/version and RNG mapping, and specifying which
  distribution the objective optimizes and corrects. Top-p changes support;
  recording its logprobs alone does not fix off-policy support loss.
- Group-based algorithms generate all G completions for a prompt under one
  version and one sampling configuration; serial generation is sufficient for
  correctness. A deterministic verifiable reward avoids a reward model initially.

Gate: sampled frequencies/seed behavior on a toy distribution; old logprobs stay
immutable after an update; ratio equals one at unchanged parameters in an admitted
same-case fixture; documented cross-case deviations are reported separately.
Synchronous update gates include objective values/gradients, masks, negative and
positive advantages, EOS/truncation, zero-variance groups and reproducible resume.

### Stage 6 — Numerical alignment, driven by evidence

**Status: the decision layer and its gate are implemented, 2026-09-26; no
alignment kernel was written, and this stage's options say why.** Stage 6 exists to
decide, *from Stage 2's evidence*, between an invariant kernel, a constrained
library configuration and a declared exception — and Stage 2 already measured the
answer, so the deliverable is the decision made checkable rather than a new
implementation. `csrc/include/alignment.h` + `csrc/alignment.c` derive one verdict
per Stage-1 region **from the committed inventory** (`region_inventory`) so the two
cannot drift, and `ctest test_alignment` is the gate:

- **the four measured regions are declared exceptions carrying the Stage-2 bound.**
  `attention_core` (1.0e-03 max_abs / 1.0e-04 rms), `gemm_bf16` (1.3e-01 / 1.2e-02),
  `gemm_fp32_lmhead` (8.0e-04 / 1.3e-04) and `gdn_core` (4.3e-04 / 4.8e-05) each
  resolve to `declared_exception`, and the bound *is* the widest over the region's
  pairs, so a declared exception cannot be tighter than what was measured; a decision
  of `exact_by_construction` for a region with an `exception` or `unverified` pair is
  refused rather than derived.
- **the four regions Stage 2 did not establish name the option that would remove
  them.** `rmsnorm`, `per_head_norm`, `gdn_conv1d` and `gdn_gated_norm` are
  `invariant_kernel_pending` with their tracked work named (a fixed reduction-tree
  kernel, a fixed conv scan order, or an explicitly constrained library
  configuration with a tested claim); adding an `unverified` pair to `regions.c`
  fails the gate until that work is named, and a stale entry for a region that is no
  longer pending fails it too. The remaining regions are `exact_by_construction`.
- **the reporting rule has teeth.** `alignment_classify` decides what a difference
  *is* before any bound is applied, and `alignment_check_observation` refuses to
  score a policy change or a sampler difference against a numerical bound — a
  3.0-magnitude difference from a stale group is not a numerical mismatch — and
  refuses to check an exact-by-construction region against a tolerance at all.

Three limits are recorded. **The alignment kernels were not built**: `alignment.c`
names them as tracked work, which is the plan's third option ("or a declared
numerical exception") and a decision rather than an omission, so an RL experiment
may run on tolerance and report its error. The region and end-to-end same-weight
logprob comparisons the gate names are **Stage 2's device gates**
(`test_gdn_invariance`, `test_attention_invariance`, `test_gemm_invariance`,
`test_attention_lse`) and the gradient/update regression is **Stage 4's**
(`test_backward`, `test_backward_kernels`); Stage 6 adds the decision those
measurements feed and the separation that stops them being relabelled. And the
bounds are absolute logit differences at the measured shapes, not a bound on any
other workload's deviation.

CPR for GDN and batch-invariant GEMM/attention remain optional implementation
choices until Stage 2 identifies the actual gaps. CPR must preserve the corrected
recurrence and supply a compatible backward. Re-run Stage 4 gradient/update gates
whenever replacing a forward implementation. An all-recurrent path may serve as a
small correctness baseline even if its throughput is unacceptable for deployment.

Options for GEMM include a fixed reduction-tree kernel, explicitly constrained
cuBLAS algorithms/workspaces with a limited tested claim, or a declared numerical
exception. Pinning an algorithm is not a general proof. Attention may require a
separate invariant implementation; do not exclude it based on its call site.

Gate: region and end-to-end same-weight logprob comparisons across admitted cases,
plus gradient/update regression. Report numerical mismatch separately from real
policy changes and sampler differences. Exact alignment is needed for an exact
claim, not a mathematical prerequisite to any RL experiment; tolerance-based
experiments must report their error and learning stability rather than relabel
exceptions as bitwise guarantees.

### Stage 7 — Algorithm bring-up and GSPO

**Status: the GSPO and GRPO objectives and their gate are implemented,
2026-09-26; no online RL loop is.** The objectives are
`backward_group_objective` in `csrc/backward.c` (the plan's `min(s_i A_i,
clip(s_i, 1-eps_low, 1+eps_high) A_i)` with `s_i = exp((1/T_i) sum_t (ell_theta -
ell_b))`, the group-relative population advantage, and one term per *response*) in
GSPO mode and the token-level analogue in GRPO mode; `ctest test_gspo` is the gate:

- **an independent FP64 reference and its central difference pin both modes.** On
  an unequal-length group (G = 4, T = 3/2/1/2, rewards 1,0,1,0 so the advantages are
  +1/-1/+1/-1), GSPO's objective is **0.126292** and GRPO's is **0.113146** — the two
  differ because equal-response and equal-token normalisation are different
  objectives, which is what "compare GSPO against GRPO on the same fixed rollout
  groups" needs. The analytic gradient matches the reference's finite difference to
  **< 1e-4** over every token of every response, including the unclipped branch's
  closed form `A_i s_i / T_i` (s_i is not detached).
- **both advantage signs and both clip boundaries are exercised.** Response 0
  (A = +1, s = 1.4918) is clipped by the upper clip and response 1 (A = -1,
  s = 0.4493) by the lower, so GSPO reports **2 clipped sequences / 5 clipped
  tokens** while GRPO reports **4 clipped tokens** with the two statistics asserted
  separately, and `mean_ratio`, `max_abs_log_ratio` and the token tail
  `max_abs_token_log_ratio` are reported so length normalisation cannot hide an
  extreme token.
- **the plan's named cases hold.** At `pi_theta = pi_b` every `s_i` is exactly 1
  and the objective is exactly 0 with the gradient `A_i/(G T_i)`; a mask removes a
  token from `T_i`, the layout and the objective together; a zero-variance group is
  refused by default and gives zero advantages under GSPO's stated rule (which the
  caller opts into); a zero-length or one-response group is refused rather than
  partially normalised; and an unadmitted (truncated) response is excluded from both
  the objective and the `d_logp` layout.

**The bug the gate caught**, recorded because it is what a finite-difference check
is for: the Stage-4 `backward_clipped_objective` derived its clamped value *per
sign* (min for A>0, max for A<0), which left the `r < 1-eps_low, A < 0` corner
reporting the **uncapped** product while already zeroing its gradient — an
objective that disagreed with its own derivative in exactly the corner the Stage-4
fixture never entered, and the corner where its "independent" reference shared the
same wrong formula. It now computes the plan's single `min(r*A, clip(r)*A)`, and
`ctest test_backward` pins the corner with a case whose value (the clamped
`(1-eps)*A`, not `r*A`) and gradient agree under a finite difference.

Three limits are recorded. **No online RL loop is implemented**: SFT is the only
bring-up, and the objectives here are the arithmetic and its gate, not a trainer
trajectory, a reward-collection run, a policy-distance measurement or a learning
stability result. OPD, DAPO and PPO remain unimplemented — their rows in the table
below are the plan's, not this repository's. And `mean_groups` is the caller's: a
group objective is computed per group, and the caller accumulates groups.

| Algorithm | Rollout | Additional models | Ratio/objective | First validation |
|---|---|---|---|---|
| SFT | no | none | Masked next-token CE | Gradients, AdamW and tiny-data overfit |
| OPD | yes | Frozen teacher | Here: dense per-prefix reverse KL, student-sampled prefixes treated as fixed data | Teacher alignment, loss and memory budget |
| GRPO | yes | Optional frozen KL reference | Token-level clipped ratio, group-relative sequence reward advantage | Token loss and group normalization |
| DAPO | yes | No KL reference in the selected baseline | Token-level objective, Clip-Higher, dynamic sampling and overlong-response handling | Explicit accounting for valid tokens and selected groups |
| GSPO | yes | No critic; KL reference only if explicitly added | Length-normalized sequence ratio and sequence clipping | Sequence objective and its gradient, not tokenwise clipping |
| PPO | yes | Trainable critic; optional KL reference/reward model | Token surrogate plus value loss and GAE | Value targets, bootstrapping and separate optimizer state |

Recommended order: **SFT → synchronous GRPO → GSPO → DAPO → PPO**. OPD is an
optional branch after SFT and rollout bring-up when a compatible teacher fits;
it is not a mandatory expensive dependency for group RL. The chosen OPD objective
does not use a PPO ratio but stale prefixes still change its data distribution.
SFT validates training correctness, not cross-case rollout invariance.

#### GSPO objective

Follow the original GSPO definition [2], not an unlabelled GSPO-token variant.
For prompt x, collect G responses y_i from a frozen behavior policy π_b. Let
T_i be the number of trainable response tokens; exclude prompt, padding and
external environment/tool tokens. Include a generated EOS according to the
recorded response mask. Zero-length responses are invalid; truncation and reward
rules are explicit algorithm configuration.

For the initial single-turn setting:

```
A_i = stop_gradient((R_i - mean_j R_j) / (std_j R_j + eps_adv))
ell_b_it = log pi_b(y_it | x, y_i,<t)          # captured at rollout
ell_theta_it = log pi_theta(y_it | x, y_i,<t)
log_s_i = (1 / T_i) * sum_t (ell_theta_it - ell_b_it)
s_i = exp(log_s_i)
J_GSPO = mean_groups mean_i min(s_i * A_i,
                               clip(s_i, 1-eps_low, 1+eps_high) * A_i)
loss = -J_GSPO
```

The baseline uses symmetric `eps_low = eps_high`; asymmetric clipping is a named
configuration, not silently inherited from DAPO. Use population standard
deviation for the project's group normalization, an explicit `eps_adv`, G >= 2
and zero advantage for zero-variance rewards. Normalize each group over its G
complete responses. Accumulate logprobs and sequence reductions in FP32 with
specified order; choose finite-loss handling without silently clamping s_i and
changing the objective. Empty/invalid groups are not partially normalized.

Important differences from GRPO:

- `s_i` is the geometric mean of token ratios, not their arithmetic mean and not
  the unnormalized product of trajectory ratios.
- Clip once per response and average response contributions equally, rather than
  clipping per token or globally weighting longer responses more heavily.
- In an unclipped branch, the gradient contribution is proportional to
  `A_i * s_i / T_i * sum_t grad ell_theta_it`. Do not detach s_i; only behavior
  logprobs, rewards and advantages are constants. A broadcast token surrogate
  needs a derived stop-gradient construction to match this gradient.
- The normalized ratio is a deliberately designed surrogate, **not an unbiased
  trajectory importance-sampling correction**. Length normalization can hide a
  small number of extreme token ratios; log token tails as well as s_i.
- GSPO can reduce instability from tokenwise ratios, but does not prove numerical
  invariance or make arbitrary stale data safe. Its clipping range needs its own
  validation; copying token-level PPO defaults is not justified by the formula.

Gate: an independent loss/autograd fixture matches objective and gradient for
unequal response lengths, both advantage signs, clip boundaries and masks;
compare unclipped analytic gradients; test zero-variance and truncated groups.
At π_theta = π_b, s_i = 1 subject to the admitted numerical case. Compare GSPO
against GRPO on the same fixed rollout groups before comparing online rewards.
Report sequence/token clipping statistics separately. No reward-gain guarantee.

### Stage 8 — Bounded-staleness asynchronous RL (proposal)

**Status: the lag-zero protocol half is implemented, 2026-09-26; the asynchronous
GPU half is not.** This stage's own order is "build A's protocol first, with lag
zero", and that is what `csrc/include/rollout_queue.h` + `csrc/rollout_queue.c` are:
the bounded `LearnerQueue` over whole completed groups, the immutable behavior
version each group is admitted under, and the plan's lag
`learner_committed_version - behavior_version` enforced at admission.
`ctest test_rollout_queue` is the gate:

- **only whole completed groups, and only one version each.** A group with fewer
  than two responses, fewer than two admitted responses or no trainable token is
  refused (the plan's "do not renormalize a partial failed group"), and every
  response must carry the group's declared behavior version, so "do not combine
  different behavior versions into one group" is a refusal rather than a comment.
- **the queue is bounded in groups, in tokens and in live behavior versions**, and
  each bound is backpressure (`ROLLOUT_ERR_FULL`, a distinct status) rather than a
  silent drop; a refused group leaves the queue's counts unchanged. A group at an
  already-live version is still admitted, which is the "bounded allowlist" behaving
  as an allowlist.
- **lag is enforced at admission.** A behavior version *ahead* of the learner's
  committed one is an unknown version and is refused (`ROLLOUT_ERR_STALE`), and so
  is one further behind than the declared cap; the admitted group is stamped with
  the learner's committed version as its proximal anchor.
- **consumption is a ledger.** A consumed group cannot be enqueued again — "retries
  cannot count a response twice and restart cannot silently retrain a consumed
  batch" — while a group that was *dropped* (rejected with a reason) was never
  trained on and may be regenerated. The ledger lives in the queue, so a destroyed
  queue loses it, which is the documented cost of discarding the record.
- **the plan's async gate 1 holds.** A group dequeued at lag zero produces
  **bitwise** the same GSPO objective and gradient as the same group computed
  directly, and its anchor version equals its behavior version: the queue is a
  container and does not move a number.
- **the plan's async gate 3 holds at the protocol level.** With lag injected as
  0/1/2 on a fixed toy group, lag zero gives `s_i = 1` and a zero objective while
  larger lag moves both the objective and the gradient — and that drift is named a
  **policy change** through Stage 6's `alignment_classify`, not a numerical
  mismatch, and no ratio is claimed to repair it. A zero-lag cap admits only the
  lag-zero group, which is "GSPO remains lag-zero until its separate objective gate
  is satisfied" as behaviour.

What is deliberately **not** implemented, because the plan defers it "before
positive-lag async experiments": weight snapshots and their publication transfer,
device cache leases, the actor/learner resource split, the `J_decoupled` stale-data
objective, deadline/drop policy monitoring, and every throughput, queue/lag
distribution and per-device memory measurement. The protocol half is what can be
built and checked without them; the plan's rule that Stage 8 "starts only after a
synchronous algorithm and parameter publication protocol pass all relevant gates"
is also why GSPO has no validated stale-policy support here.

Start only after a synchronous algorithm and parameter publication protocol pass
all relevant gates. Asynchrony is a scheduling property, not another name for
GSPO. AReaL [3] is evidence that decoupled objectives and staleness management can
work; it does not establish that the GSPO formula above tolerates arbitrary lag.

#### Three resource/scheduling choices

| Choice | Benefit | Cost and limitation |
|---|---|---|
| A: async host/reward pipeline, serial GPU phases | Overlap data preparation/reward work, bounded queues and explicit versions without extra GPU weight copies | No simultaneous GPU learner/rollout; delayed rewards may still create stale batches |
| B: separate actor and learner GPU ownership | Genuine generation/update overlap; simple immutable actor snapshot and cache semantics | Extra BF16 copy, publication transfers and split GPU capacity; start with a model whose full learner fits one card |
| C: colocated concurrent GPU streams and snapshots | Potential overlap while sharing a device; frozen LoRA base can be shared locally | Activation/KV contention, snapshot copies, peak memory and stream/thread complexity; not the first implementation |

Recommendation: build A's protocol first, with lag zero, then prototype B with
one actor GPU and one learner GPU on a tiny model. Permit a bounded lag of one
committed optimizer step only as an explicit experiment after lag-zero parity.
Do not split a two-card PP learner into B without redoing per-card memory and
placement. C is deferred until measurements show it beats serial execution.
For LoRA, adapters can be versioned cheaply when a frozen base is locally shared,
but adapter kernels/backward and cache isolation are additional work, not present
features. A separate GPU still needs its own frozen-base storage.

#### Immutable version publication

Proposed logical objects are `ParameterStore`, `PolicySnapshot`, `RolloutContext`,
`RolloutGroup` and a bounded `LearnerQueue`; these are interface requirements,
not declarations already available in the engine.

1. The learner alone mutates master weights, gradients and optimizer state.
   Complete update, BF16 conversion and derived-weight refresh before committing
   version v+1. No CUDA reader may observe an in-progress update.
2. Publish a complete immutable snapshot with run/version ID, checksum, numerical
   manifest and placement. Copy into an inactive buffer with device completion
   events; atomically announce readiness only after all required devices finish.
   Hold the source BF16 version immutable until the copy completes, either by
   delaying its next overwrite or by copying from a retained immutable staging
   snapshot; account for the pause or extra storage. Destination events alone
   do not prevent a source-side update race. On failure, leave the prior
   committed snapshot usable.
3. An actor acquires a snapshot lease for a **whole response group**. Each
   completion resets its KV/GDN state, then holds the same version throughout
   that completion. New groups may use a newly published snapshot; old groups
   finish under their leased one.
4. Never switch weights while reusing a prefix's KV/GDN state. Mid-generation
   weight switching, used in some asynchronous systems, is out of scope here.
   Supporting it later requires a different cache/policy definition and per-token
   versioning or prefix recomputation, followed by separate numerical tests.
5. Release a snapshot only after host leases and CUDA readers finish. Double
   buffering permits one active and one staging actor snapshot only if the
   scheduler prevents extra live versions. Bound live snapshots and block/coalesce
   publication when capacity is exhausted; never overwrite a still-leased buffer.

With synchronous A, these leases can serialize access to one BF16 store. With B
or C, zero-copy weight sharing is no longer the general claim. Measure snapshot
bytes, transfer time and any resharding independently from kernel execution.

#### Rollout record and admission

Persist enough information to identify the actual behavior distribution and
reproduce a loss without regenerating text:

- Run, prompt, group, response and attempt IDs; input/response token IDs and
  response/loss masks; positions, EOS versus length limit and terminal reason.
- Behavior snapshot/version/checksum; semantic/numerical/build identities;
  sampling config ID and RNG seed/counter; captured per-token raw and, if
  transformed, sampler logprobs. Never overwrite them with learner logprobs.
- Group size and membership, scalar rewards and reward-function version;
  frozen reference/teacher identity where used; completion/reward times and
  submission/consumption learner versions.
- Learner batch/epoch/consumption ID and rejection reason, so retries cannot
  count a response twice and restart cannot silently retrain a consumed batch.
- At final learner admission, a separate batch record stores the proximal
  anchor's run/version/checksum, numerical manifest and frozen per-token ell_p,
  plus the normalized advantages and objective configuration. Persist these for
  loss replay; admission/consumption version counters alone cannot reconstruct
  a possibly older anchor. Never replace the original behavior fields.

Queue whole completed groups. Do not train the first k responses to finish,
combine different behavior versions into one group, or renormalize a partial
failed group. Fix G at dispatch; bounded retries use the same retained version.
If that version is unavailable, the group is stale or a deadline is reached,
reject the whole group and record the reason. Deadline/drop policies can bias
against long or hard responses, so monitor rejection by length/reward and compare
against the synchronous data distribution. DAPO's reward-based dynamic sampling
is a separate named selection rule, not an accidental queue side effect.

Bound queue capacity in groups **and tokens**, in-flight work and live versions.
Define lag as `learner_committed_version - behavior_version` in optimizer steps;
record wall-clock age and policy-distance statistics as separate quantities.
Enforce lag at dispatch/admission and before each update/epoch; dropping only at
dispatch misses long generations and delayed rewards. Negative lag, unknown
versions, nonfinite/missing logprobs, duplicates or incompatible manifests fail
admission. Backpressure actor dispatch when the learner/reward workers fall
behind. For the first prototype use a bounded allowlist of behavior versions and
one update per consumed batch; broader replay requires a new validated objective.

#### Objective choice: do not confuse three policies

Let π_b be the logged behavior policy, π_p a frozen proximal anchor at admission
(before this batch's updates), and π_theta the current learner. π_ref is a
separate frozen KL reference, not any of those three. Numerical mismatch is
measured at fixed weights; `π_b != π_p` is genuine staleness.

Keep initial GSPO synchronous. The first stale-data experiment should use a
**named token-level, decoupled clipped-surrogate baseline**, motivated by AReaL's
decoupled PPO, with the existing group reward advantages and no critic. This is
an experimental objective, not a claim to reproduce AReaL or canonical GRPO:

```
c_it = stop_gradient(exp(ell_p_it - ell_b_it))
r_it = exp(ell_theta_it - ell_p_it)
J_decoupled = mean_groups mean_i mean_valid_t
                 c_it * min(r_it * A_i,
                            clip(r_it, 1-eps_low, 1+eps_high) * A_i)
```

The actor supplies ell_b; score ell_p once at final batch admission under a
frozen anchor, then persist and keep ell_p fixed. Initially the learner admits
one batch immediately before its update, without intervening updates during
anchor scoring. Do not recompute the behavior denominator under current weights.
With π_b = π_p this reduces to the selected synchronous token surrogate.

**Single-update limitation:** at the initial prototype's gradient evaluation,
π_theta = π_p, so r_it = 1 and proximal clipping is inactive. The update is a
c_it-weighted token policy-gradient step; clipping does not bound how far that
step moves the policy. Zero proximal clip rate is therefore not a stability
result. Record post-update policy distance and apply explicit learning-rate,
gradient-norm and divergence-stop criteria. Validating clipping itself requires
later evaluations with π_theta != π_p, such as multiple minibatch updates under
one frozen anchor. That extension must specify optimizer-step lag accounting,
batch reuse and re-admission rules before enabling it.

A scalar lag cap is not enough: inspect c_it tails/ESS and reject whole groups
violating a declared admission bound before normalization. This initial policy
avoids silently clipping c_it; adding truncated corrections is a new biased
variant that needs explicit configuration and comparison.

Tokenwise action correction does not restore the on-policy prefix distribution,
and group-normalized advantages plus clipping/selection introduce further bias.
The expression is not an unbiased full-trajectory estimator and must pass
controlled stale-data learning experiments. If it fails, retain serial/lag-zero
training; do not increase lag until the run stops failing. A critic-based
V-trace/PPO alternative adds value-target design and is a later branch.

For **async GSPO**, record an unresolved algorithm decision rather than asserting
that the existing s_i fixes staleness:

- Replacing π_old with a stale π_b mixes proximal update control with policy lag.
  Merely clipping the resulting geometric-mean ratio does not correct the data
  distribution.
- Reusing the token decoupled formula is no longer GSPO. Multiplying geometric
  means to imitate a sequence correction also does not produce unbiased
  trajectory importance weights.
- Before enabling lag > 0, derive and name a sequence-level stale-data surrogate
  with explicit behavior/proximal roles, stop-gradient points and clipping; check
  its gradient and zero-lag reduction to GSPO, then compare controlled lags on
  the same data. Until then GSPO may use asynchronous infrastructure only with
  version-matched consumption; it has no validated stale-policy support.

#### Haskell control plane and recovery

Use bounded queues (e.g. `TBQueue`) for dispatch, reward and learner admission,
with explicit version/group state machines. Start with one sequential actor
context per worker; multi-request batching is not required to prove the protocol.
Each CUDA context has a bound owner thread. Cancellation marks work cancelled
and drains/finishes its device operations before releasing buffers or leases;
it is not permission to free memory under an outstanding kernel.

Checkpoints commit the learner weights/optimizer, version counter, RNG/data
cursor and consumed-batch ledger together. Initially restart by discarding
unconsumed/in-flight groups and actor caches, restoring the committed learner,
and publishing a fresh snapshot. Reusing queued groups after restart requires
persisted behavior provenance and deduplication; deterministic replay of arbitrary
asynchronous completion order is not promised. Record the admitted batch order
for controlled replay tests.

#### Async gates and measurements

1. Run the pipeline with lag zero and a fixed pre-generated rollout/group order.
   Match synchronous objective, gradients and optimizer update; where the
   numerical contract admits exactness require bits, otherwise the predeclared
   tolerance. Separately test identical sampling with the same RNG mapping.
2. Stress snapshot swaps, delayed rewards, full queues, worker failure, duplicate
   delivery, cancellation and restart. Assert no mixed-version group, stale cache,
   uncommitted publication or duplicated consumed update.
3. Inject lag 0/1/2 deterministically into a fixed toy task before real async
   scheduling. Measure objective/gradient drift, reward versus optimizer steps,
   token/sequence clip rates, behavior-to-anchor/current policy distance,
   correction-weight tails and ESS. ESS is diagnostic, not proof of no bias.
4. Measure useful accepted rollout tokens/s, learner tokens/s, end-to-end reward
   versus wall time, queue/lag distributions, stale/failed-group waste, publication
   overhead, GPU utilization and peak allocated/reserved bytes per device.
   Raw actor throughput can rise while useful training throughput falls.
5. Enable positive lag only for the objective/lag range whose tests pass. GSPO
   remains lag-zero until its separate objective gate is satisfied. A throughput
   improvement is an experiment outcome, not an acceptance assumption.

## Temperature-sampling migration

### Target behavior and scope

**Status: T0-T4 are implemented and the CLI default is temperature 1, 2026-09-26.** The
selector and the RNG are `src/Infer/Sampling.hs`; the shared configuration and its
validation are `src/Infer/Config.hs` (`SamplingConfig`, `parseSampling`); the generation
loop takes one `SamplingConfig` through one `stepToken` at both entry points; `Main` adds
`--temperature`/`--seed`, validates them **before** any model or tokenizer allocation and
reports the resolved temperature, seed and sampler version on stderr. Gates:
`cabal test infer-generation-tests` (which now runs `tests/SamplingSpec.hs`: the frozen
splitmix64 vectors, the CDF boundaries, the endpoint rules, shift invariance, overflow and
underflow, the draw-count contract, and the four loop-level checks the stub's arbitrary
rows made possible) and `tests/test_sampling_cli.py` (32 CLI checks, all of them early
validation against a non-existent model directory). The delivery notes:

- **the arithmetic is the plan's.** Each `Float` logit widens to `Double` before the
  subtraction and the division; `a_i = (z_i - m)/tau`, `w_i = exp(a_i)`, `Z` a left fold in
  token order, `ell_sampler = a_i - log Z` and `ell_model = (z_i - m) - log(sum exp(z_j - m))`
  computed only when the record is requested. The inverse CDF takes the first
  *positive-weight* token whose ordered prefix exceeds `u*Z`, so a zero-mass leading token
  cannot be taken at `u = 0` and an exact boundary belongs to the following bin. The four
  greedy call sites are gone: both entry points select through `stepToken`, at temperature
  0 that is the lowest-id `argmax` with **no draw consumed**, and above 0 it is one draw per
  selected token.
- **the RNG is splitmix64 implemented in this module rather than a pinned dependency.**
  What "version-pinned" protects is the algorithm's behaviour across builds, so the module
  pins the published gamma and mix constants and `tests/SamplingSpec.hs` freezes the seed
  to-word vectors (seed 0's first word is splitmix64's published `0xe220a8397b1dcdaf`),
  which is strictly more stable than a version bound and keeps a new Hackage dependency out
  of the build. The API is Word64-in/Word64-out and the mapping is the plan's
  `u = (x >> 11) * 2^-53`.
- **the greedy fixtures now select T=0 explicitly**, which the section above requires: the
  fifteen pre-existing generation fixtures pass unchanged with an explicit greedy config,
  and a request that omits `--temperature` samples at 1.0. The CLI reports
  `sampling: temperature=..., seed=..., sampler=host-binary64-cdf-1` on stderr, keeping the
  metadata out of the generated text, and an omitted seed is drawn once from `/dev/urandom`
  (a missing source is an error, not a fixed seed).
- **the review caught a third bug the fixture had been hiding**: `ell_model` was computed
  from `a_i = (z_i - m)/tau`, which is only the raw-model offset when `tau = 1`, so every
  selection made at a temperature other than 1 recorded a wrong raw-model log-probability.
  The test that should have caught it had chosen a `u` that selected the row's *maximum*
  token, where `z_i = m` makes both normalisations collapse to the same number; the
  assertion is now on a non-maximum token and additionally requires the two recorded
  log-probabilities to differ at `tau /= 1`.
- **one rule could not be reached, and that is recorded.** The plan's endpoint correction -
  "if floating multiplication rounds `r` up to `Z`, select the last positive-weight token" -
  is defensive here: because `Z` is at most the vocabulary size and `u <= 1 - 2^-53`,
  `u*Z < Z` always and the scan always finds a bin, so no admitted `u` exercises the branch.
  The reachable half of the rule *is* asserted: the largest admitted `u` cannot select an
  arbitrary final entry (a trailing zero-mass token is skipped for the last positive one).
- **not done: the measurements T4 lists.** Host selection time, allocations/GC, TTFT and
  tokens/s are **not** measured; the fixed-seed and greedy CLI smoke runs are, and they are
  recorded in the worklog rather than in a gate that would need the checkpoint and a GPU.


Until this migration the implementation was **greedy decoding**: `generate` and
`generateStreaming` selected argmax at prefill and every decode step, and neither
`Config` nor `Main` had a temperature or a seed. `Infer.FFI.Engine` already returned host
FP32 logits as `[Float]`, so the first sampler belonged in Haskell and needed no C ABI,
CUDA kernel, weight format or architecture-descriptor change - which is where it now is
(`src/Infer/Sampling.hs`, one `stepToken` at both entry points).

The end state, which is what the implementation below does:

| Input | Behavior |
|---|---|
| No sampling options | `temperature=1.0`; categorical sampling over the full vocabulary with a newly resolved request seed |
| `--temperature T`, finite T>0 | Sample from `softmax(logits/T)`; T<1 sharpens and T>1 flattens the same logits, without a promised text-quality gain |
| `--temperature 0` | Explicit greedy mode, lowest token ID wins exact ties; no division by zero, softmax sampling or random-word consumption |
| `--seed S` | Unsigned decimal Word64 in [0, 2^64-1]; fixes the request's random stream, not arbitrary GPU/kernel nondeterminism |
| Negative/nonfinite temperature, malformed/out-of-range seed | Configuration error before loading model/tokenizer or acquiring entropy |

Use the same default values and validation in the CLI and `defaultRuntimeConfig`.
Parse finite decimal temperature with sign/range validation before converting to
Double; reject a nonzero value that underflows to zero, or overflows to infinity.
Negative nonzero input is invalid even if its magnitude would round to zero.
Canonicalize literal negative zero temperature to greedy zero. A seed provided
with T=0 is accepted but reported as unused. Do not silently switch a small positive
T to greedy, read hidden model-specific sampling defaults, or add top-k/top-p,
repetition penalties, logit bias or beam search in this migration. Retaining
explicit greedy is a supported evaluation mode, not a compatibility workaround.
Default-changing CLI examples and release notes must state the behavior change;
existing greedy golden/reference tests must select T=0 explicitly, not acquire
a new stochastic expected output.

### Distribution, validation and selected-token record

First implementation: convert each input Float logit to Double **before**
subtraction/division; do softmax exponentials, normalization and CDF accumulation
on the CPU in binary64, traversing increasing token IDs. This is a deliberate
sampler numerical policy, distinct from the CUDA FP32 logit output and the
trainer's FP32 loss/logprob reductions. No sort is needed. A later FP32/GPU
sampler is a new implementation requiring its own distribution and replay gates.

For finite logits z and temperature tau>0, define:

```text
m = max_i Double(z_i)
a_i = (Double(z_i) - m) / tau
w_i = exp(a_i)
Z = left_fold_add_token_order(w_i)
p_tau(i) = w_i / Z
ell_sampler(i) = a_i - log(Z)
ell_model(i) = (Double(z_i) - m) - log(sum_j exp(Double(z_j) - m))
```

Subtracting m before division avoids positive overflow from `z/tau`; at least
one a_i is zero, so Z must be finite and >=1 for an admitted vocabulary. A very
small positive tau can produce negative infinity in a_i or underflow w_i to
zero: treat it as zero numerical mass, not a reason to reroute to argmax. Equal
maxima retain equal weights even in this limit. Record this finite-precision
support limitation rather than claiming mathematical full support at every T.

At the selection boundary reject empty/wrong-length logits and any input NaN
or infinity, including -infinity in this initial unmasked sampler. Failures
propagate through the existing generation error path; do not return token 0,
clamp corrupt logits, silently retry with another seed or report a short success.
The exported legacy argmax need not be redesigned, but production selection
validates before invoking it. Validation occurs before consuming a random word;
check for invalid normalization/CDF results as errors, not hidden fallbacks.

Define inverse-CDF selection precisely: obtain u in [0,1), set r=u*Z, and choose
the first **positive-weight** token whose ordered cumulative weight is strictly
greater than r. Exact CDF boundary hits belong to the following positive bin;
u=0 must not select a zero-mass leading token. Compute Z and the selection scan
in the same order. If floating multiplication rounds r up to Z, select the
last positive-weight token as the specified endpoint correction, never an
arbitrary final vocabulary entry; larger inconsistencies are errors. Unit tests
exercise that correction separately from normal draws.

The mathematical temperature distribution, its binary64 implementation and the
finite uniform-grid/CDF selection probabilities are not literally identical.
Version the arithmetic and endpoint rules, compare their error on fixtures, and
label `ell_sampler` as the implemented softmax logprob rather than pretending it
is an exact count of finite-RNG intervals. Distributional admission requires a
predeclared error budget; an exact stochastic claim needs stronger analysis.

The proposed selector returns token ID plus optional selected-token metadata:
raw-model `ell_model`, temperature `ell_sampler`, sampling config/version,
effective seed and draw index. Compute probabilities/logprobs from the original
logits in log space, not by taking `log` of a rounded-to-zero probability. At
T=1 share the same normalization for both logprob fields; at T=0 the selected
sampler probability is one (`ell_sampler=0`), not the model softmax probability.
Only compute the raw-model normalization when its record is requested. Full
vocabulary logits need not be retained after selection, and normal CLI output
need not collect a training trajectory. Stage 5 must explicitly request and
persist this record before any learner update; it cannot reconstruct it later.

### RNG ownership and reproducibility

Use one explicit request-local PRNG state, initialized once, not a process-global
RNG and not a seed reset on every token. Start with a version-pinned `splitmix`
dependency and an explicit Word64-seed initialization/Word64-output API; freeze
known seed→word test vectors and the package/version before implementation
admission. This RNG is for sampling, not cryptographic use.

Map each output Word64 x to `u = Double(x >> 11) * 2^-53`. A positive-temperature
selection consumes exactly one word, including a singleton vocabulary or a row
whose numerical mass lies in one bin. Greedy mode consumes none. Draw 0 selects
the first generated token from prefill logits, not a prompt token; subsequent
draws advance once per selected output token, including terminal EOS. Never draw
after EOS/budget exhaustion or for an engine call that failed to return valid
logits. An error after a successful draw aborts the request, not resampling.

If seed is omitted, obtain entropy once for a positive-budget stochastic request
and report the resolved seed, temperature and sampler version on stderr before
generation; keep metadata out of generated stdout text. With an explicit seed,
record that seed and do not acquire entropy. Nonpositive API budget touches
neither engine nor RNG/entropy; the CLI retains its existing rejection of
negative budgets. For the same seed and admitted logits, streaming and
non-streaming paths must consume the same words and select the same IDs.

RNG lifetime is independent of `engine_reset`: every new generation request
starts from its chosen seed; a future in-request continuation must carry its RNG
state/draw index alongside token/cache state rather than silently reinitialize.
The present CLI does not promise resumable generation. Reproducibility means
same prompt IDs, weights, execution/sampler manifests and seed, not just the
same text prompt and numeric seed across devices/builds. Different seeds may
legitimately produce the same response; do not test otherwise as a guarantee.

For Stage 5 groups, assign each response a distinct, reproducibly derived stream
from a documented run/group/response mapping before parallel dispatch, and log
its resolved seed; do not initialize all G responses with the same seed or rely
on thread completion order. Exact retry reuses the response seed under the same
version, while a new sampling attempt has a new recorded attempt/stream identity.
Freeze that derivation and RNG checkpoint format before async admission; the
single-request migration does not implement a group scheduler.

### T0–T4 implementation milestones

**Files:** add `src/Infer/Sampling.hs` for the pure distribution/inverse-CDF
functions and the small explicit-state RNG adapter; add `tests/SamplingSpec.hs`
as a module invoked by the existing generation test suite. Modify
`src/Infer/Config.hs`, `src/Main.hs`, `src/Infer/Generation.hs`,
`tests/GenerationSpec.hs`, `tests/generation_engine_stub.c` and
`haskell-infer-demo.cabal`; update `README.md` and `docs/design.md` only when the
feature actually ships. Add a small proposed `tests/test_sampling_cli.py` runner
for real executable option/seed/stream checks and `tests/SamplingEngineSpec.hs`
for direct real-engine returned-token comparisons. Register the latter as an
opt-in `infer-sampling-engine-tests` suite linked to the actual engine/tokenizer,
not the C stub; absent model/runtime prerequisites mean skipped/unverified, not
a passing GPU gate. These new modules/options/tests are planned, not implemented
by this document edit.

- [x] **T0 — Lock configuration and greedy baseline.** Write tests for default
  temperature, T=0/T>0, invalid numeric inputs, seed range and configuration
  parity; extend greedy tests for exact ties. Define a shared `SamplingConfig`
  with validated Double temperature and optional Word64 seed, used by Main and
  generation rather than duplicated defaults. Audit greedy examples/scripts
  before changing the default; use explicit greedy in regression fixtures.
  Invalid configuration must fail before any model allocation.
- [x] **T1 — Implement and test the pure selector.** Separate preparing weights
  from choosing with a supplied u, so deterministic CDF/normalization tests do
  not depend on PRNG behavior. Add independent softmax/logprob reference values
  and boundary tests before implementation. Initially consume the existing
  `[Float]` FFI output with bounded per-row scratch and strict folds; avoid
  retaining lazy chains across decode steps. No FFI/vector rewrite or GPU
  kernel is necessary. Register `Infer.Sampling` and `SamplingSpec` in affected
  Cabal components; do not inadvertently make unrelated config-only tests
  depend on random IO.
- [x] **T2 — Add request RNG and unify token selection.** Pin the RNG dependency
  in the executable and generation-test component, add fixed word/uniform test
  vectors, then pass one state through all four current argmax call sites.
  Both generation entry points use the same selector and next-state contract.
  Resolve options once in Main and pass the validated configuration instead of
  reading globals/environment per token. Extend the C stub to supply arbitrary
  per-step vocabulary rows, not only one +1 winner with all other logits -1.
- [x] **T3 — Verify generation lifecycle and real CLI behavior.** Preserve
  first-token budget counting, returned EOS, pending-token consumption, engine
  error propagation and stream flushing/cleanup. A sampled EOS is handled by
  the same stop path as a greedy EOS. Characterize the existing first-EOS versus
  later-EOS text-feed asymmetry in tests; do not change special-token rendering
  as an unrelated sampling fix. Check new request/seed behavior, capacity errors,
  arbitrary logits, prefill/decode failures and zero-budget no-RNG behavior.
  The current `infer-generation-tests` does not compile CLI Main: use the
  executable runner for option/default/help and stderr-seed checks, including
  invalid arguments with a nonexistent model path to prove early validation.
- [x] **T4 — Admit default change and document boundaries.** First run CPU
  distribution/RNG/lifecycle tests, then real-model greedy regression and
  fixed-seed temperature smoke tests. Compare stream/non-stream returned IDs,
  rerun the same seed under the same manifest, and measure host selection time,
  allocations/GC, TTFT and tokens/s. Switch CLI/config default to T=1 only in the
  change that passes these gates; publish T=0 reproducibility instructions and
  T=1/0.7 fixed-seed examples together. Retain engine logits/golden checks
  unchanged. Do not claim temperature sampling is a speed optimization or that
  higher diversity automatically improves quality.

### Sampling gates and commands

1. **Pure math and boundary gates:** singleton/equal logits; two-token logits
   `[0, log(3)]` give approximately `[1/4,3/4]` at T=1 and `[1/10,9/10] at
   T=0.5, accounting for the supplied FP32 inputs. Test representable constant
   shifts, low/high positive T, negative logits, underflow, exact CDF hits,
   u=0 and the largest admitted u<1, zero-weight bins and endpoint correction.
   Reject empty/mismatched/nonfinite input, invalid T and invalid injected u.
   Compare selected logprobs with an independent log-sum-exp calculation.
2. **Statistical gate, distinct from seed replay:** on a small fixed vocabulary
   use fixed seed sets and an a-priori draw count/error criterion against an
   independent target distribution. For example N=200,000, at most four bins,
   and absolute frequency tolerance 0.01 gives a conservative Hoeffding/union
   bound under ideal independent draws; this is a sampler smoke gate, not a
   proof of PRNG independence. Do not require rare tokens to appear or change
   seeds/tolerances until a test passes. Check that lower T concentrates this
   fixed-logit distribution and higher T flattens it, not a whole generation's
   entropy after its prefix has changed.
3. **Replay/lifecycle gate:** known RNG vectors, exactly one draw per stochastic
   selected token and zero for greedy; same-seed stream/non-stream ID parity,
   EOS at first/later steps, budgets 0/1/several, repeat requests, error cleanup
   and no extra draw/engine call after stop. Exact replay is scoped to the same
   admitted runtime; GPU numerical differences can change CDF decisions.
4. **Engine integration gate:** T=0 matches existing greedy behavior/goldens for
   valid inputs; changing only sampling config leaves same-prefix engine logits
   unchanged. Real T>0 smoke tests confirm valid tokens, bounded lengths and
   request success but do not replace statistical tests. Report host overhead
   for the large vocabulary instead of assuming existing logit transfer is free.

Existing CPU command after implementing these modules, using the installed
Haskell dependencies (no CUDA execution required):

```bash
cabal test infer-generation-tests infer-tests --enable-tests
```

Proposed real-application smoke commands after implementation/build, in the
configured CUDA environment; the flags below do **not** exist today:

```bash
cabal run exe:haskell-infer-demo -- generate --model-dir "$MODEL_DIR" \
  --descriptor "$DESC" --gpus "$DEVICES" --prompt "Hello" \
  --max-tokens 16 --temperature 0
cabal run exe:haskell-infer-demo -- generate --model-dir "$MODEL_DIR" \
  --descriptor "$DESC" --gpus "$DEVICES" --prompt "Hello" \
  --max-tokens 16 --temperature 1.0 --seed 42
```

Repeat the second invocation with the same seed and with `--stream`; the CLI
runner checks option behavior, reported seed, success and rendered text.
`tests/SamplingEngineSpec.hs` calls both generation functions with the real
runtime and compares their returned `[Int64]` values across reset requests;
rendered text alone is not a token-ID gate. An omitted-seed CLI smoke run must
print a seed that can replay the request when supplied explicitly. Neither
commands nor tests are executed by this planning-only revision.

### Interaction with training and speculative decoding

Stage 5 starts at T=1 without truncation, so the mathematical behavior policy is
the model softmax. The host-FP64 versus trainer-FP32 realization still needs a
same-prefix logprob comparison; same weights do not imply bitwise ratio=1.
Retain the sampler/logprob precision in the numerical manifest. At T!=1, using
raw-model logprobs as the behavior denominator is wrong: the selected
`ell_sampler` and transform define that distribution. Keep the raw logprob as
separate metadata; before training at T!=1, specify whether the learner objective
uses the transformed policy or an explicit off-policy correction. The initial
RL gate does not silently inherit arbitrary CLI temperatures, and T=0 cannot
serve as its stochastic on-policy baseline.

The initial S0–S3 speculative path remains an **explicit T=0 experiment**.
Ordinary T>0 requests use target-only sampling until a separately validated
stochastic speculative algorithm exists; an explicitly requested unsupported
T>0-plus-speculation combination must fail before running. Applying temperature
to the draft and accepting only target argmax matches does not sample from the
target distribution. The later p/q acceptance and residual-resampling extension
must use both actual temperature distributions, separate proposal/acceptance/
correction RNG streams and compatible rollback semantics. Distributional parity,
not identical seed-to-token paths, is its first stochastic gate. Adding a CPU
sampler alone does not satisfy that extension or Stage 5's trajectory ledger.

## Inference optimization track

### Scope, dependencies and numerical identities

**Status: proposed and unimplemented.** None of F, Q or S exists: no fusion was applied,
no quantized artifact or `weights.manifest.json` sidecar was produced, and no draft/target
speculative path was written. The region inventory notes where each change would land. This
track was not touched by the Stage 6-8 work, which is per the recommendation below - it is
"a prioritization, not a hard dependency".

Recommended implementation order: **baseline/provenance → fusion → weight-only
quantization → speculative decoding → measured combinations**. This is a
prioritization, not a hard dependency: speculative decoding can use a BF16 target
and does not require quantization. Start each optimization against an unchanged
BF16 baseline before combining it with another change. The initial deployment
target is the verified A40/sm_86 inference configuration; other architectures
need their own kernel and runtime admission.

Stages 0–1 supply provenance and region fixtures; Stage 2's applicable GEMM,
attention and GDN experiments identify execution-case differences. The trainer's
attention-backward experiment is not a prerequisite for inference optimization.
Stage 6 becomes relevant when an optimization needs a stronger cross-case
numerical guarantee than the existing kernels provide.

| Change | Contract treatment | Required comparison |
|---|---|---|
| Fusion with intended unchanged arithmetic | Preserve logical operation/parameter identities; record fused implementation, layouts and retained casts in `numerical_policy_id` and build provenance | Diagnostic old/new comparison, yielding scoped bitwise-equivalence evidence only if outputs and mutable state pass; this is distinct from matching-identity strict admission |
| W4A16 weight-only quantization | Keep architecture semantics but assign a distinct packed-weight identity and numerical policy, including quantizer, scales and dequantization rules | Independent quantized-math reference plus a separately budgeted quality comparison against BF16; not a BF16 bitwise refactor |
| Speculative verification | Record draft and target identities, verification shapes, rollback/replay case, tie rule and generation/RNG policy | Compare against target-only generation under the same target weights/policy; batching and state recovery need independent numerical admission |

Extend the Stage 0 capture with these fields rather than disguising an
optimization as a placement-only change. An explicit BF16 execution request must
remain reproducible; unsupported optimized cases fail clearly rather than
silently selecting a different precision. A deliberate alternate kernel choice
must appear in the resolved manifest. No general runtime plugin/graph compiler
is required for these bounded paths.

### F — Operator fusion

**Current anchors:** `csrc/kernels/layers.cu::forward_mlp` issues separate gate
and up GEMMs; `forward_attention_layer` issues Q/K/V projections separately;
`forward_gdn_layer` issues QKV/Z/A/B projections separately.
`csrc/kernels/silu.cu` already implements fused SiLU-multiply, and FlashInfer
attention and `csrc/kernels/gdn_norm.cu` already fuse parts of their computation.
A fused checkpoint name is not evidence of fused execution: GDN QKVZ/BA rows are
currently unpacked into separate weight views at load time.

**Files:** modify `csrc/include/layers.h`, `csrc/kernels/layers.cu`,
`csrc/kernels/silu.cu`, `csrc/kernels/flashinfer_norm.cu`,
`csrc/layer_dispatch.cu` and the relevant loading/execution sites in
`csrc/engine.cu`; keep `csrc/kernels/gemm.cu` as the BF16 comparison path.
Extend `tests/test_library_ops.py`, its `tests/kernels/kernel_bridge.cu`, and
existing norm/GDN/engine tests. The model-level Haskell FFI remains unchanged.

**Status: F0 is implemented and the baseline is measured, 2026-09-26.** The instrument is
`csrc/include/profile.h` + `csrc/profile.cu` (opt-in per-region CUDA timing: a
`PROFILE_SCOPE(name, stream)` guard records one event pair per region invocation, never
synchronizes while recording, and `profile_report` is the single measurement boundary;
**off by default**, so the gates and the goldens run the path they always did), and the
entry point is `tests/benchmark_inference.py` (repeated warm runs with min/median/max and a
standard deviation, per-region time and launch count, and the provenance a baseline needs:
descriptor, devices, GPU model/clocks/temperature, manifest identities). On the deployment
target — Qwen3.8-27B, 2× A40, sm_86, pipelined across the two cards:

| Request | median | dispersion (5 or 120 samples) |
|---|---|---|
| prefill M=2 | 105.86 ms | stdev 0.012 ms |
| prefill M=64 | 122.82 ms | stdev 0.044 ms |
| prefill M=128 | 146.22 ms | stdev 0.040 ms |
| decode M=1 | **95.53 ms (10.47 tok/s)** | stdev 0.036 ms over 120 steps |

The per-region table is where F1's decision comes from, and it is decisive. **A decode step
is the dense MLP**: `ffn.dense` is **62.3 ms of the 95.5 ms step**, and inside it the three
GEMMs are 20.5 + 20.5 + 19.8 = 60.7 ms while `mlp.silu_mul` is 0.51 ms and the post-norm
0.33 ms. The **gate and up GEMMs alone are 40.9 ms — 43% of a whole decode step** (48.1 ms of
a 146 ms prefill), which is exactly the pair F1 proposes to merge. The mixers are the rest:
`mixer.gdn` 24.8 ms (48 layers; in-projection QKV 9.3 + Z 5.8, out-projection 6.1, and the
delta core only 0.7) and `mixer.attention` 7.6 ms (16 layers; Q 3.7, O 2.0). The LM head's
**single row** through a 248320-entry vocabulary costs 4.46 ms + 0.07 ms of download, 4.7% of
a decode step. Two F0 observations run *against* the obvious reading: the norms and residual
adds that F2 proposes to fuse sum to well under 2 ms at M=1, so F2 is a launch-count and
bandwidth question rather than a time one; and `mlp.silu_mul` is negligible at M=1 but
**20.7 ms — 22% of the MLP — at prefill M=128**, so the activation's cost is prefill-specific.
Qwen3-4B on one A40 shows the same shape (MLP 11.5 of 17.3 ms, gate+up+down 10.7 ms), so the
conclusion is not a property of the two-card split.

Three things F0 records rather than estimates: **the recording itself costs +4.91 ms** on both
instrumented calls (151.13 ms vs 146.22 measured, 100.44 vs 95.53), which is why the runner
reports `overhead_ms` beside every per-region table and why those tables' sums are not a
decomposition of the wall time they were measured next to; **memory traffic and
host-synchronization counts per region are not measured at all** (they need a profiler, not
events), so the plan's two remaining F0 columns stay open; and the baseline is one
configuration — the shapes are the descriptor's, and the numbers are medians of the runs
recorded in the JSON the runner writes.

- [x] **F0 — Establish a costed baseline before choosing fusions.** Record
  per-region CUDA time, launch count, host synchronization and memory traffic for
  decode M=1 and representative prefill M=2/64/128, within each descriptor's
  limits. Include layer placement and full request wall time. Run timing with
  taps disabled, after warm-up, and synchronize only at measurement boundaries;
  keep diagnostic captures separate. Weight traffic, MoE host-offset sync and
  device transfers may dominate launch savings.
**Why F2 and F3 are recorded as available-but-not-the-bottleneck, 2026-09-26.** F3's own rule
is "Expand only where F0 shows a bottleneck", and F0's table says the bottleneck on the
deployment target is neither the launch count nor the elementwise work it covers: a decode step
is 95.5 ms, of which the elementwise regions F2 would fuse (the four norms and the two residual
adds) are ~1.5 ms, and the GDN and attention projections F3 would merge are already one GEMM
per output group whose cost at M = 1 is *weight traffic* (48 layers x 168 MB for GDN's QKV+Z
alone), which merging does not reduce. So the next milestone is the one that addresses that
axis - Q, weight-only quantization - and F2/F3 stay open with this reasoning rather than
half-done.

**Status: F1 is implemented, 2026-09-26.** The loader allocates one packed `[2I, H]` buffer per
dense MLP and loads the gate and up roles into its two halves, so there is never a second copy
of those weights and the shard rule is still applied per role (the two halves' extents are
compared and the load is refused if they disagree); `forward_mlp` then issues **one GEMM with
N = 2I** into a row-interleaved `[T, 2I]` output and `kernel_silu_mul_packed` consumes that
layout. The workspace is unchanged (T*(H + 2I + I) is the T*(H + 3I) the pool is sized for).

The fusion is admitted on this evidence:

- **where the reduction does not change, the numbers do not either.** Against the *unfused*
  build on the same fixture, the short-prompt logits are **bitwise identical** (rms = 0,
  max_abs = 0), and the per-step rms against the independent torch reference is identical to
  16 digits for both regular cases (0.1305636763572693 … 0.06327719986438751), inside the
  0.1–0.4 band this family is recorded with, with every token matching.
- **where cuBLAS does pick differently, the difference is quantified rather than denied.**
  The long/chunked fixture is where M is large enough for N = 2I to select another reduction:
  its logits differ from the unfused build by rel_rms **0.8–2.7%** (max_abs 0.33) with
  **16/16** top-1 agreement, and the engine's own chunk-boundary self-consistency gate moves
  from rms 0.108/0.102 to **0.069/0.065** with top-1 374/374 unchanged. That is the plan's
  "GEMM N changes from I to 2I and may select a different reduction"; it is a declared
  numerical-policy change, not a bitwise-equivalent refactor.
- **the policy records it.** `numerical_policy_id` moves (`05fa3b15…` → `61bbd7f3…`) while
  `semantic_id` and `deployment_id` do **not** — which is the plan's fusion row exactly:
  logical operation and parameter identities preserved, fused implementation and layout
  recorded in the numerical policy.
- **it is worth admitting.** Qwen3-4B on one A40: prefill M=64 23.53 → **20.32 ms** and M=128
  28.36 → **22.79 ms** (−13.6% / −19.6%), decode M=1 17.30 → **16.89 ms** (57.8 → 59.2 tok/s),
  with no prefill regression. The per-region table says *where* the win comes from, and it is
  not the GEMM: at M=128 the fused gate/up GEMM is 8.29 ms against the two separate GEMMs'
  3.99 + 3.95 = 7.94, i.e. slightly *slower*, while `mlp.silu_mul` falls from **6.55 ms to
  0.45 ms** because the old FlashInfer `act_and_mul` launch was a single block. At M=1 the
  GEMM is the faster part (6.63 vs 3.54 + 3.53) and the step gains 2.4%.

Not yet done, and the next F candidate: the GDN in-projections (QKV + Z were 15.1 ms of a
decode step in F0's table), and the plan's F2/F3 rows.

- [x] **F1 — Merge dense gate/up projections.** First add fixtures for existing
  separate GEMMs and SiLU output; then concatenate weight rows once during
  loading and compute `[T,2I]` in one GEMM. Existing scratch is
  `[gate[T,I]; up[T,I]]`, while row-major `[T,2I]` interleaves gate/up per token:
  update the activation kernel's explicit row/stride contract, not just its
  pointer offsets. Include T>1 tests because T=1 hides this layout defect.
  Own the packed buffer through the existing allocation registry, avoiding
  permanent duplicate BF16 weights. GEMM N changes from I to 2I and may select a
  different reduction; compare pre-activation outputs before attributing any
  difference to SiLU. Do not claim a GEMM+SwiGLU epilogue exists in current cuBLAS.
- [ ] **F2 — Fuse residual addition with the following norm.** First specify
  two outputs: the updated BF16 residual and the normalized activation.
  The fused kernel must compute `r = BF16(old_r + sublayer_out)` and normalize
  that rounded r, not an unrounded FP32 sum. Retain plain versus Gemma weight
  conventions and FP32 reduction behavior. Start at mixer→post-norm within a
  layer; change dispatch so the FFN consumes the prepared activation rather than
  normalizing twice. Sublayer kernels still do not privately add residuals.
  The replicated path in `forward_replicated` is a separate caller: all-reduce
  must finish before this fused boundary, with residual addition exactly once.
  Cross-layer fusion is deferred until ownership and placement boundaries pass.
- [ ] **F3 — Expand only where F0 shows a bottleneck.** Candidates are merged
  Q/K/V projections, compatible GDN projections, Q/gate split plus per-head
  norm/RoPE/cache write, and conv+SiLU. Preserve the Q/gate interleave, head
  expansion and checkpoint row ordering. GDN convolution explicitly rounds to
  BF16 before SiLU today; a fused kernel must retain that conversion if it claims
  unchanged arithmetic. Do not absorb MLA projections or replace FLA recurrence
  as incidental fusion cleanup: those are distinct algorithm/numerical changes.

**F gates:** compare intermediate outputs and persistent state against the
unfused path and an independent reference; include nonzero residual/state,
plain/Gemma norms, T=1 and multi-row tails, supported GQA/head shapes and lengths
crossing 64/128. Any shared-memory tree reduction must handle its launch shape,
including inactive lanes, rather than assuming multiples of 32 are powers of
two. Preserve legacy bitwise captures for paths claimed unchanged; otherwise
record a new policy and a quantified exception, not a weakened old golden.
Run PP and applicable TP/EP regression when their callers are touched. Admit a
fusion only when the intended workload improves without an unbudgeted memory or
prefill regression; kernel timing alone is insufficient. Training reuse later
requires saved-rounded-value and backward/gradient revalidation from Stage 4.

### Q — Weight-only quantization

**Status: Q2 is implemented and routed, 2026-09-26: the dense FFN's decode path reads the packed
operands, and the measurement is what decided which path.** `csrc/kernels/gemm_quant.cu` is the
weight-only INT4 GEMM - `C = A * B^T` with a BF16 activation, the Q0-packed weight and a BF16
output accumulated in FP32 - and it does what the plan's Q2 demands of it: the weights are
**unpacked and scaled inside the thread** (one 32-bit load carries eight codes, the group's
scale hoisted out of the inner loop), so a quarter of a BF16 weight's bytes are read and
nothing is dequantized into memory. Unsupported shapes are refused by the host wrapper at
creation time rather than midway through a request.

`ctest test_quantization_kernel` is the gate, and it makes the plan's **two separate
comparisons**: the kernel against an *independently dequantized-weight* reference (the payload
dequantized in numpy from the format's definition and multiplied in float64) and the quantizer
against the original weight. Only the first is asserted, and tightly: across the FFN shapes
and M = 1/2/8/64 the kernel's error is `max_rel <= 3e-3`, `rms_rel <= 5e-4` - inside the BF16
output's own rounding, which a wrong nibble, a wrong scale or a wrong group index cannot pass.
The second is reported (max_abs 0.021-0.027, rms 0.0074 of a |w| max of ~0.38) because the
quantizer's quality is a model-level question the plan assesses with logits RMS, top-1
agreement and held-out NLL once a kernel is actually routed.

**The measurement decides a specialization, which is why it is taken.** The plan asks for
M = 1 and batched M to be measured *separately*, and on one A40 at N = 4096, K = 5120 the two
paths come out opposite:

| path | median | packed-weight bandwidth | against the same values in BF16 |
|---|---|---|---|
| M = 1, warp-per-row GEMV (coalesced) | **52.1 us** | 207.6 GB/s | **2.25x faster** (117.2 us) |
| M = 1, first one-thread-per-output kernel | 366.7 us | 29.5 GB/s | 3.1x slower |
| M = 64, first kernel (batched) | 5090.9 us | 2.1 GB/s | **44x slower** (115.0 us) |

The first implementation's mapping was the problem, not the format: one thread per output
element gives each thread a whole weight row, so a warp's loads walked different rows and were
uncoalesced. The rewritten **warp-per-row GEMV** - lane t loads the 32-bit word at index t, so
consecutive lanes read consecutive words, with the activation staged in shared memory once per
block - is 7x faster than that first kernel and **2.25x faster than the same values in BF16**,
which is the format's advantage showing up as a measurement rather than a claim (it is not the
theoretical 4x: the kernel reaches 208 GB/s of the device's ~700, so latency hiding is still
the limit).

**Only the decode path is admissible, and the batched path was built and measured before that
was concluded.** The batched path is now a shared-memory-tiled GEMM (block tile 64x64, the K
slice exactly one scale group so the slice's sums are scaled once, a 4x4 micro-tile per thread,
the activation tile padded off one bank) and it is **verified correct** - all the same checks
pass - but it is far too slow to route:

| M | int4 tiled | BF16 | effective rate |
|---|---|---|---|
| 1 (GEMV, not tiled) | **54.3 us** | 115.3 us | **2.12x faster** |
| 2 | 924.1 us | 116.0 us | 0.13x |
| 16 | 996.7 us | 116.4 us | 0.12x |
| 64 | 2247.7 us | 116.0 us | 0.05x |
| 128 | 3316.1 us | 114.5 us | 0.03x |

The reason is not the tiling but the arithmetic: at N=4096, K=5120 the BF16 path is
*weight-bandwidth-bound* at ~115 us for every M (365 GB/s of a 42 MB read), so the format's
four-fold byte saving *should* win - and the decode path does win, because at M = 1 there is one
multiply-accumulate per byte read and the bytes are what cost. As M grows there are more
multiply-accumulates per byte, and a SIMT kernel pays an unpack instruction for each one:
measured **1.2-2.7 TFLOP/s** against the tensor cores' 23-47. Putting the same products on the
*int* tensor cores is what would change this, and that needs int8 activations - which the plan
scopes out ("activation quantization, QAT and quantized optimizers are separate work"). So
weight-only INT4 with BF16 activations is a **decode-time specialization**, and the routing
means the warp-per-row GEMV for M = 1 plus the F1-compatible packed-row layout, with the
batched path left in the tree, verified, and recorded as inadmissible at these shapes rather
than routed.

**The routing is done, and the quality gate is what says it is admissible.** `forward_mlp` reads
the packed operands when `tokens == 1` and BF16 otherwise, so prefill is untouched and decode is
the specialization; `engine_load_quantized_ffn` installs the operands beside the BF16 weights
after validating the sidecar, refuses a tensor-parallel engine (the converter quantizes whole
tensors), requires every dense layer to be covered (a half-quantized layer would be silent mixed
precision), and refuses to coexist with a training store in either direction - a publication
writes the BF16 weights and would leave the packed operands describing the old ones.
`ctest test_quantized_ffn` is the gate. On the synthetic dense fixture the prefill logits are
**bitwise equal**, the worst decode rms is **9.08e-3** over 8 steps, top-1 is 7/8 with the flip
explained by its margin (3.2e-3, below the step's 2.7e-2 perturbation - a tie-break, and the gate
requires exactly that), and the held-out NLL moves by **-0.0008 nats**. **On the deployment
target** (Qwen3.8-27B, 64 dense layers, 2× A40) the same gate reports prefill **bitwise
unchanged**, worst decode rms **0.34605**, top-1 **7/8 with the one flip a tie-break and none
unexplained**, the held-out NLL **-0.05826 nats**, an 8415.0 MiB residency from a 14025.4 MiB
sidecar, a 45.32 s load, TTFT **unchanged** at 106.07 ms and decode **99.64 -> 63.26 ms, 1.57x**
(10.0 -> 15.7 tok/s). On a second real model (Qwen3-4B, 36 dense full-attention layers) it
reports the same shape with the larger error that model's coarser weights imply: worst decode rms
1.00379, top-1 6/8 with both flips tie-breaks and none unexplained, held-out NLL +0.28559 nats
inside a per-model budget, TTFT unchanged and decode **1.61x**. The converter's own error on the
deployment target is max_abs 0.0835 / rms 1.37e-3 over 192 role instances in 1705 s. A packed
operand is a numerical-policy change and nothing else: `numerical_policy_id` moves, `semantic_id`
and `deployment_id` do not, and `manifest_test.c` pins that.

**Q0 and Q1 status.** — the format, its reference, the converter, the sidecar and its reader;
the Q2 kernel consumes the format, and the engine does not yet consume the sidecar.**
`scripts/quantize_weights.py` converts a
checkpoint's admitted roles (the dense FFN gate/up/down, per Q's initial scope) into a
separate output directory without touching the BF16 checkpoint, and writes a versioned
`weights.manifest.json` carrying exactly the fields Q1 lists: the source directory with a
per-file size and SHA-256, the converter and config versions, the per-layer/role mapping, the
logical `[N, K]`, the packed layout/shape/dtype and byte count, the group axis and size, the
scale tensor's shape/dtype/count, the zero-point convention, per-artifact hashes, a precision map
with one cell for every `(role, layer)` of the descriptor, and a `pairs` section for the F1
layout below. **The quantization
arithmetic is not re-implemented in the converter**: each tensor is handed to the Q0 reference
(`test_quantization_format --quantize`), so an artifact is produced by the same code Q2 admits
a kernel against and the two cannot drift. `--verify` re-derives rather than re-reads: it
re-hashes every artifact, re-checks the extents against the format, requires the precision map
to cover every role of every layer exactly once and to agree with the entries, and re-quantizes
a sample from the source checkpoint. `ctest test_quantization_converter` drives it over the
synthetic dense checkpoint (6 role instances; block error max_abs 0.0062, rms 0.0024 over
196608 elements) **and requires the verification to fail** on a corrupted payload, a manifest
claiming a foreign group width and a missing artifact, because a validator that cannot fail is
not a gate.

**The F1-compatible layout is emitted, not left to the reader.** The plan's Q2 requires that
"if F1 is active, concatenate packed rows and scale rows consistently and preserve its
activation layout", because F1's `forward_mlp` takes one `[2I, H]` weight and the row-interleaved
`[T, 2I]` activation. The converter now emits a **pair artifact** per dense layer with both
members quantized: `mlpGateUp_layer{L}.packed` is the gate's packed rows followed by the up's
(`[2I, H]`, the operand F1's GEMM already wants) and `.scales` is the same row order, so the
engine does not have to know the pair's internal split. `--verify` requires each pair to equal
its members' bytes in order and its `logical_shape` to be `[sum(rows), k]`; the gate re-derives
that from the **entries located by role**, not from the pair's own member list, so a pair that
agreed with itself but not with the quantized members would fail. On the synthetic checkpoint
this is 2 pairs (`[512, 128]`, 32768 packed bytes and 512 scales each) and the verification
reports "2 F1 pair(s) match their members' rows". The engine-side reader that consumes these now
exists as the sidecar validator above (`csrc/quant_manifest.c`, gated by `test_quant_manifest`
and driven over the converter's own output by `test_quantization_converter`); what it does not yet
do is load them onto a device or route the decode path to the INT4 GEMV.

**The sidecar has a reader, and the reader is a validator.** `csrc/quant_manifest.c` parses
`weights.manifest.json` and refuses what it cannot re-derive: a version it does not speak, a
format block that is not the frozen format (group 128, q ∈ [-7, 7], zero-point 0, `-8` reserved,
`u8`/`bf16`), a group axis that is not K, an entry whose extents disagree with its shape, a pair
whose rows are not its members', a precision map that does not describe the same cells as the
entries, a malformed digest, a duplicate key, a truncated document. **The format's refusals are not
re-implemented** - an entry's shape goes through `linear_layout_init`, the same function the
quantizer and the Q2 kernel gate use, so a K that is not a whole number of groups is the format's
error rather than this reader's opinion. `quant_artifact_read` ties the bytes to the digest the
manifest recorded (byte count, dtype and SHA-256) rather than trusting the manifest's claim about
its own artifacts. `ctest test_quant_manifest` is the CPU gate: a hand-written fixture, seventeen
refusals produced by textual surgery on it - each mutant checked to have actually applied, so a
renamed fixture cannot turn the set into silent passes - and a temp-file artifact that is read,
then refused for a flipped byte, a wrong size, a wrong dtype and a missing file.
`test_quantization_converter` now runs this binary over the sidecar the converter just wrote, so
the schema, its only writer and its reader are tied to each other: that run parses 6 entries and 2
pairs from the synthetic checkpoint and re-verifies a 16384-byte artifact from disk.

One trap this stage walked into and then closed: **both this gate and Stage 5's `test_sft`
skip themselves when the node-local synthetic checkpoint is absent, and a skip reports as
`Passed` in CTest.** A pod that moved nodes lost that checkpoint, so a full suite of "28/28"
was in fact 27 ran + 1 skipped, and the new converter gate started its life skipped too. The
checkpoint is regenerated with `python3 tests/synth/make_qwen3_dense.py --out-dir
/var/pony/cache/bohaotu-haskell/synth-qwen3-dense --layers 2 --seed 0`, after which both gates
run (Stage 5's passes in 12.5 s and the converter gate in 0.6 s). The plan's own rule - "absent
model/runtime prerequisites mean skipped/unverified, not a passing gate" - is exactly what the
CTest status hid.

**Q0's status.** `csrc/include/linear_weight.h` + `csrc/linear_weight.cpp` are the frozen INT4 format
as CUDA-free arithmetic: `s = BF16(max|group|/7)` with `s = 1` for an all-zero group,
round-to-nearest-even `q = round(w/s)` clipped to `[-7, 7]`, zero-point 0, two
two's-complement nibbles per U8 byte with the lower K index in the low nibble, `-8` reserved
invalid, and a reference dequantizer (`w = code * bf16_to_scale`) that a kernel is admitted
*against* in Q2 rather than alongside. The layout refuses what the format cannot represent - a
K that is not a whole number of 128-wide groups, an N that is not a whole number of the
8-row tile, a different group width - instead of hiding padding that would make an artifact's
byte count disagree with its manifest. Gates: `ctest test_quantization_format` (the CPU checks
below) and `ctest test_quantization_format_python`.

- **the packing is checked by hand, not only by a round trip.** A round trip cannot catch a
  self-consistent swap of the nibbles, so the gate asserts the byte pattern for a row whose
  maximum is exactly 7 (which makes the scale 1 and takes the arithmetic out of the way):
  `0x10`, `0x9F`, `0x37`, `0x2D`, and the per-index codes through `linear_packed_code`.
- **the refusals are behaviours.** K=192 (1.5 groups), N=12 (1.5 tiles), a 64-wide group, a
  NaN or infinite weight, and a group whose `max|w|/7` rounds to zero in BF16 all fail; the
  reserved `-8` in a payload makes the *reader* report a malformed artifact rather than read it
  as a value; and the signed extrema clip to ±7, so the quantizer can never emit the reserved
  code.
- **quantization is a fixed point of dequantization.** Re-quantizing a dequantized weight
  reproduces the payload and the scales exactly, which is what "the kernel matches the
  reference" reduces to once a kernel exists.
- **a second implementation re-derives the fixture.** `tests/test_quantization_format.py`
  reads the payload, scales and reference dequantization that the C binary emits for a
  deterministic [8, 256] weight and re-implements the format *from the definition* in numpy -
  its own BF16 rounding, scale, code, clipping and nibble packing - requiring an exact match.
  On that fixture it reports block max_abs error 0.0186 and rms 0.0109 against values of
  magnitude ~0.25.

The status of what this section opened has moved on, and the Q2 section below is the record: Q1's
converter and sidecar exist, the sidecar now has a reader (`csrc/quant_manifest.c`) that validates
it before a GPU sees it, and Q2's kernel is implemented and measured. The plan's "subject to a
sm_86 kernel feasibility check" resolved into a *measurement* rather than a check - the
warp-per-row GEMV is 2.25x faster than BF16 at M = 1 on sm_86 while the batched path is 8-30x
slower, which is why "weight-only INT4 with BF16 activations" turns out to be a decode-time
specialization. Still not done: the device-side load of the artifacts into the engine, the routing
of the decode path to that GEMV, and the model-quality gates (logits RMS, top-1 agreement,
held-out NLL) that compare a quantized model against the BF16 baseline.



**Initial scope:** dense FFN gate/up/down only, first on one device or PP, with
BF16 activations, BF16 region outputs and the defined FP32 accumulation policy.
Keep embeddings, LM head, norms, routers, attention/MLA/GDN projections, GDN
convolution/decay parameters and recurrent state at their current precision.
Quantized KV cache, activation quantization, QAT and quantized optimizers are
separate work. A40/sm_86 must use a compatible weight-only kernel; dependency
`ENABLE_FP8` settings are not evidence of a native FP8 inference path.

**Files:** extend `csrc/include/safetensors.h`, `csrc/safetensors.cpp`,
`csrc/safetensors_loader.cu`, `csrc/engine.cu`, `csrc/include/layers.h` and
`csrc/kernels/layers.cu`. Introduce a bounded weight-format/linear-weight module
(`csrc/include/linear_weight.h`, `csrc/weight_manifest.cpp`) and a separate
`csrc/kernels/gemm_quant.cu` implementation, rather than casting packed data to
`__nv_bfloat16 *`. Add the offline converter `scripts/quantize_weights.py`,
format tests `tests/test_quantization_format.py`, kernel tests
`tests/kernels/test_quant_gemm.cu` and model-quality runner
`tests/test_quantization.py`; register compiled targets in `csrc/CMakeLists.txt`.
These are proposed new files, not current capabilities.

- [x] **Q0 — Freeze a versioned format and independent reference.** Start with
  symmetric signed INT4, K-axis groups of 128 and BF16 scales, subject to a
  sm_86 kernel feasibility check before fixing the wire format. A concrete
  baseline quantizer uses FP32 source values, scale
  `s = BF16(max(abs(group))/7)`, round-to-nearest-even `q = round(w/s)` clipped
  to [-7,7], and zero-point 0; all-zero groups use s=1. Reject nonfinite inputs
  and nonpositive/nonfinite rounded scales. Pack two two's-complement nibbles
  per U8 byte, lower K index in the low nibble; reserve -8 as invalid in this
  format. Define dequantization multiplication and BF16 conversion explicitly
  and require the chosen kernel to match that reference. Initially require
  K/group and backend tile alignment; reject unsupported shapes instead of
  inventing hidden padding. Better calibration/AWQ/GPTQ is a later named
  quantizer, not an unrecorded improvement to the same artifact.
- [x] **Q1 — Implement converter, manifest validation and ownership.** Write
  quantized artifacts to a separate output directory without modifying the BF16
  checkpoint. Use a strict, versioned `weights.manifest.json` sidecar resolved
  from the model directory; keep the architecture descriptor portable and
  unchanged. Record source checkpoint/checksum, converter/config version,
  per-layer/role mapping, logical `[N,K]`, packed layout/shape/dtype, group axis
  and size, scale tensor/shape/dtype, zero-point convention and artifact hashes.
  Quantized and retained-BF16 roles form an explicit precision map. Validate
  role shapes against the descriptor plus packed byte counts, scale values,
  missing/duplicate/unknown entries and overflow before GPU use. Integer storage
  support in safetensors does not authorize arbitrary integer model weights.
  Represent each linear weight with format, logical/local dimensions, packed
  pointer and scales; register every allocation for partial-load cleanup.
- [x] **Q2 — Add real weight-only execution, then model gates.** (the kernel and its fixtures; the routing waits on the measurement above) Start from
  pack/unpack and synthetic GEMM fixtures, then route only admitted FFN roles to
  a kernel that unpacks/scales inside its register/shared-memory tiles. Reject
  full-weight dequantization into a BF16 temporary on every forward: it restores
  weight traffic and can erase the optimization. A one-time full BF16 expansion
  also does not retain device-weight compression. Measure M=1 and batched M
  separately before choosing specializations. If F1 is active, concatenate
  packed rows and scale rows consistently and preserve its activation layout.
  Unsupported kernels/shapes fail during creation, not midway through a request.
- [ ] **Q3 — Add sharding and additional roles as separate admissions.** TP
  output-row/head splits slice corresponding scale rows; input-column splits
  must align with group and packing boundaries and retain global quantization
  scales. Do not requantize independently per rank. EP slices packed experts
  and scales together and preserves router/shared-expert precision and combine
  order; revise BF16-specific expert offsets in `load_moe_weights`/`csrc/moe.cu`.
  Future GDN row gathering must also gather scales. Until these gates exist,
  reject quantized TP/EP or excluded roles explicitly while retaining ordinary
  BF16 TP/EP support.

**Q gates:** first verify the kernel against an independent dequantized-weight
reference, then assess the quantizer against the original BF16 model. These are
different comparisons. Cover signed extrema, zero groups, group/tile boundaries,
M=1/multi-row, malformed/truncated artifacts, scale mismatch and failure cleanup
in `tests/safetensors_test.cpp` and the new tests. Fix evaluation prompts and
teacher-forced prefixes before running: report logits RMS/max error, top-1
agreement, held-out NLL/perplexity and a defined task-quality score. Free-running
outputs alone confound weight error with different prefixes. Set explicit
per-model quality budgets before admission; token identity with BF16 is not a
universal quantization requirement. Retain the existing BF16 reference/token
gates unchanged. Long-context, reset, chunk and later TP/EP gates compare against
the same quantized artifact/policy, not an independently requantized baseline.

Report converter/load time, peak host/device loading memory, persistent bytes,
workspace, TTFT and decode latency/throughput. Ideal aligned weight storage is
`N*K/2 + 2*N*(K/128)` bytes before headers/backend packing, not a fourfold
reduction of whole-model memory; excluded roles, state and scratch remain.
Choose W8A16 or narrower role coverage only as a separately specified experiment
if W4A16 misses quality/performance gates, never as a silent precision fallback.

### S — Speculative decoding

S0–S3 remains greedy-only even after the ordinary CLI migrates to temperature
sampling: explicitly select T=0 and reject unsupported stochastic speculation.
See [temperature-sampling migration](#temperature-sampling-migration) for the
separate target-only sampler and its Stage 5 integration.

**Current anchors:** `src/Infer/Generation.hs::generate` performs greedy argmax
with lowest-ID tie breaking; its last emitted token has not yet been consumed by
`engine_decode`. `csrc/engine.cu::compute_logits` computes only the final row,
although `engine_prefill` already executes batched chunks. `engine_reset` clears
all state, not an arbitrary suffix. Attention/MLA caches append by position;
GDN updates both BF16 convolution history and FP32 SSM state in place.

**Files:** extend `src/Infer/Generation.hs`, `src/Infer/Runtime.hs`,
`src/Infer/Config.hs`, `src/Main.hs`, `src/Infer/FFI/Engine.hs`,
`csrc/include/engine.h`, `csrc/engine.cu` and `csrc/include/layers.h`.
Extend `tests/GenerationSpec.hs`, `tests/generation_engine_stub.c`,
`tests/engine_bindings.py` and `tests/test_engine_resources.py`; add
`tests/test_speculative.py` for engine/state/target-only comparison. Reuse
`tests/kernels/test_gdn.cu` and `tests/kernels/test_mla.cu` for state fixtures.
Any additional test target/module must be registered in the existing Cabal or
CMake configuration. No CUDA pointers cross the Haskell boundary.

- [x] **S0 — Build a sequential correctness prototype first.** Own two
  independent runtimes, prefilling the identical token prefix. Validate complete
  tokenizer/token-ID and special-token compatibility, not just vocabulary size;
  use the same prompt/template encoding and check both context capacities.
  Draft and target may have different architectures. Start with a fixed small
  proposal count, e.g. 2–4, and greedy generation only. Verify candidates with
  target `decode` calls in the original sequential order; never feed a rejected
  candidate to this target baseline. Recover draft by reset plus sequential
  prefix replay. This validates acceptance, pending-token and lifetime logic,
  but is explicitly not a speedup. Keep the normal target-only path available.
- [x] **S1 — Add bounded all-position verification and append-cache rollback.**
  Introduce a proposed `engine_verify_rows` API that consumes n token IDs and
  returns n FP32 vocabulary rows, with explicit output capacity and checked
  sizes. Apply final norm and LM head to every input row, respecting `max_chunk`;
  keep ordinary prefill/decode final-row behavior intact. Start with the small
  window's bounded `[n,V]` host result and existing Haskell argmax semantics;
  tiled output or GPU top-1 is a later measured extension, not an implicit FFI
  contract change. Add append-only cache truncation restricted to the current
  sequence and an available prefix. Initially admit pure full-attention targets
  and drafts; MLA needs its own admission, and any GDN model waits for S2.
- [x] **S2 — Add hybrid state checkpoints and restore/replay.** Introduce
  opaque, engine-owned checkpoint handles and explicit save/restore/release
  operations; at most one round checkpoint per engine initially. Save all GDN
  conv/SSM buffers on every owning device, plus sequence length, engine/reset
  generation and weight/policy identity. KV/MLA entries before the checkpoint
  are immutable, so their logical length suffices under the current append-only
  cache design. `reset_zero` is a cleanup registry, not by itself a rollback
  protocol. On rejection restore the round-start state, then replay exactly the
  retained inputs in the admitted execution case. A single final SSM state
  cannot be inverted into an intermediate one, and one pre-round snapshot does
  not make acceptance-prefix restoration O(1). Restore draft state too.
- [ ] **S3 — Admit acceleration only after cross-case and continuation gates.**
  Compare every target verification row against serial execution with identical
  prefixes, including nearly tied logits; compare retained cache/state and
  several subsequent decode steps after each rejection. Stage 2 already warns
  that chunked and recurrent execution differ. Exact speculative decoding in
  mathematical arithmetic does not establish exact parity with this finite-
  precision serial engine. If the scoped greedy-equivalence gate fails, retain
  S0 or narrow support/fix numerical alignment under Stage 6; do not hide changed
  tokens behind a logit RMS tolerance or certify only near-tie positions without
  a justified error bound. Quantized targets get their own target-only baseline.

**S0 status (2026-09-26): implemented and gated; S1–S3 are open.** `Infer.Generation.generateSpeculative`
owns two runtimes over the same prompt, keeps the plan's invariant (`P` consumed in both, the last
confirmed token still pending), takes the longest matching prefix with the target's own argmax as
the correction or bonus, emits only target-confirmed tokens up to the first EOS, bounds the window
by the budget and both contexts, and recovers a rejected round by resetting and **sequentially**
replaying `P + [x] + y[1:r]` - deliberately the slow, obviously-correct recovery, since checkpoints
are S2 and a chunked replay would produce a state Stage 2 measured to differ from the serial path.
Admission is one token space (vocabulary *and* the prompt's encoding under both tokenizers) with
both contexts holding the prompt, and the CLI refuses a window without a draft, above temperature 0,
with `--stream` or outside 1..16 before loading anything. The CPU gate extends the scripted engine
to two handles with an observable consumed history - and to a *prefix-indexed* script mode, without
which reset-and-replay would not be faithful - and covers the plan's list: k = 1, full acceptance,
rejection at every position, repeated rejection, lowest-ID ties, first/accepted/correction/bonus
EOS, budgets 0/1 and the window boundaries, context exhaustion, and the refusal of a differing
vocabulary. 71 examples pass in `cabal test infer-generation-tests` (locally and on the pod), and
`tests/test_speculative.py` adds the 18 CLI refusals. **What is not done**: S1's bounded
all-position verification and append-cache rollback (the prototype verifies with sequential decode
calls), S2's checkpoints (so recovery is a full replay), S3's cross-case admission gates, and any
measurement of acceptance or speed - S0 is explicitly not a speedup.

**S2 status (2026-09-26): implemented, and it is what admits a recurrent model.**
`engine_checkpoint_save` / `_restore` / `_release` are an engine-owned, opaque checkpoint of one
round's state: the buffers `engine_reset` would clear (the attention and MLA caches and the GDN
convolution and SSM state), the sequence length, the reset generation and the parameter identity.
The whole reset set is copied rather than only the recurrent part - the KV and MLA entries are
immutable up to the retained length, so their length would suffice, but classifying buffers by
role would be a second registry to keep in step, and the copy is what makes a restore correct
across any sequence. A restore refuses rather than guessing when the saved state is no longer the
engine's: after a reset (the generation moved), after a failed forward (the engine already
requires a reset, and a checkpoint does not clear that), and after the weights *or the numerical
policy* changed - the packed operands do not move the weight digest, so the quantization posture
is recorded separately. `generateSpeculative` now takes the checkpoint at each round boundary
when the rollback mode asks for it, restores and replays exactly the retained suffix on
rejection, and releases them when the run ends (a live one would refuse the next run's save); the
mode is chosen from the descriptor, which is why the earlier "any GDN model waits for S2"
refusal is gone. `tests/test_checkpoints.py` is the gate: on a model with GDN layers, three
decodes after a restore are **bitwise** what a second engine that never took the detour produces,
and the five refusals fire (a second save, none saved, after a reset, after a release, and after
a policy change). A fully accepted round restores nothing, which the CPU suite pins.

**S1 status (2026-09-26): implemented; its device gate is the open confirmation.** `engine_verify_rows`
consumes n ids as one bounded batch (refusing n > max_chunk rather than silently chunking, and
refusing a host buffer shorter than n x vocab instead of overrunning it) and returns every row's
FP32 logits: the ordinary path deliberately computes only the final row to avoid a
[max_chunk, vocab] buffer, so this allocates one lazily and normalizes the whole activation in a
single call. The tokens append to the current sequence exactly as `engine_decode` appends them,
and `engine_truncate` takes the sequence back to a retained length - admitted only for a model
whose every layer is full attention, because a GDN state cannot be rewound and MLA needs its own
admission (that is S2's checkpoint work; the call refuses rather than leaving the cache and the
state disagreeing). `generateSpeculative` now uses both: the round's verification is one batch, and
the rollback is "truncate the target back to the retained prefix, and either truncate the draft or
make it consume its missing last proposal" instead of S0's full replay - the plan's "on full
acceptance draft must additionally consume its missing yk". The CLI refuses a target or draft with a
recurrent layer up front, so the admission failure cannot arrive mid-run. `tests/test_verify_rows.py`
is the device gate: n = 1 must be **bitwise** a decode (one execution case), n > 1 is reported
row-by-row with every argmax required to match serial execution, the rollback must leave the batch
engine where the serial one is, and a short buffer, a window past max_chunk, a truncation past the
sequence and a recurrent model must all be refused.

#### Round protocol and failure semantics

Let both engines have consumed prefix P of length L, and let x be the already
emitted, target-confirmed token that is still pending consumption. This invariant
must hold at each nonterminal round boundary:

1. Draft consumes x and its own proposals to produce `y1 ... yk`. Its ordinary
   loop has consumed only through `y(k-1)`, not yk. Target verifies the inputs
   `[x, y1, ..., yk]` in a single bounded batch. With zero-based output rows,
   row 0 predicts y1, row i predicts `y(i+1)`, and row k predicts the bonus token.
2. Accept the longest prefix of r proposals whose IDs match the corresponding
   target argmax. If r<k, correction is `argmax(row r)`; if r=k, bonus is
   `argmax(row k)`. Stop checking at the first rejection; later rows condition
   on a rejected token and are unusable as continued generation.
3. Retain exactly `P + [x] + y[1:r]`, length `L+1+r`, in both engines. Truncate
   append-only caches; for a rejected GDN round restore and replay those inputs.
   On full acceptance draft must additionally consume its missing yk before
   the next round. The correction/bonus becomes the new unconsumed pending
   token. Do not pre-consume it and then accidentally decode it twice.
4. Emit only target-confirmed tokens, in order, stopping at the first confirmed
   EOS; no later speculative token may enter the text decoder. EOS terminates
   the round without a bonus or continuation. Nonpositive budget performs no
   engine work, and the first prefill token still counts toward the budget.
   For the initial protocol choose k so `k+1` fits remaining output budget,
   both engines' available context and the target's `max_chunk`; with only one
   output slot left, use one normal target step rather than overproducing a
   batch. If context capacity or `max_chunk` leaves no positive legal k but
   target decoding is still possible, finish target-only from its current
   committed prefix without further draft calls. If the target cannot consume
   the pending token, propagate the existing sequence-capacity error rather
   than silently truncating a successful response. Assert consumed versus
   emitted lengths explicitly in tests.

Checkpoints are valid only for their originating engine, sequence generation,
weight version and numerical policy; disallow nesting, cross-engine restore,
reuse after reset and truncation before the retained prefix. Save/restore/copy
must finish on all relevant streams before reporting success or releasing
buffers. Validation errors leave state unchanged. A failed device forward may
have partially mutated state: retain the current invalid-state/reset requirement
unless a complete, tested restoration succeeds; do not merely set `state_valid`.
Failure, cancellation or second-model initialization failure cleans up both
runtimes/checkpoints without swallowing the error or freeing in-flight memory.
Keep calls on the owning bound OS thread; two engines do not justify concurrent
access to either engine. Streaming cannot retract already confirmed text, but
must never expose unverified proposals or report a failed request as success.

**S gates:** extend the CPU scripted engine to distinguish two handles and
inspect their consumed histories; test k=1, full acceptance, rejection at every
position, repeated rejection, lowest-ID ties, first/accepted/correction/bonus
EOS, budgets 0/1/window boundaries, capacity exhaustion and UTF-8 streaming.
GPU tests cover nonzero GDN initial state, snapshot/reset lifetime, invalid
handles, failure injection, multi-device ownership, cache lengths crossing
64/128, and several rounds after rollback. Test target-only versus speculative
output independently from snapshot-copy exactness and cross-case state error.
If only a finite fixture set passes, report that scope rather than universal
bitwise invariance.

**S performance gate:** measure draft generation, target verification including
all-row LM head/D2H, draft catch-up, snapshot copies, rejection replay and host
orchestration separately. For each round compare their total time with serial
target time for the same number of newly committed tokens, then measure complete
requests. Report acceptance histogram, useful tokens per verification, TTFT,
inter-token latency/tail latency, useful tokens/s and peak per-device memory.
Select a fixed k from measured results before considering adaptive windows;
low acceptance or expensive GDN replay may make the method slower. Shared GPU
resources and the draft's weights/cache are part of the cost, not free work.

#### Later sampling and training/rollout integration

Ordinary temperature sampling is specified in T0–T4 and need not wait for this
extension. Greedy acceptance is not stochastic speculative sampling. A later
stochastic speculative path must define target p and draft q after their
declared sampling transforms, accept with `min(1, p(y)/q(y))`, and on rejection
sample from normalized `max(p-q,0)` under the same prefix, with support/zero-mass handling and separate
RNG streams. This needs distributions, not just greedy IDs or selected logits.
Validate frequencies against target-only sampling on toy distributions; different
RNG consumption does not promise the same sequence for the same seed.

For RL, log the actual emitted target-policy/sampler logprob, not the draft
proposal probability or acceptance probability, and retain both model identities
for replay. Quantizing the target changes the behavior policy even if it derives
from the learner's checkpoint: it is not a deployment-only identity change or
proof of on-policy BF16 rollout. Greedy decoding itself cannot substitute for
Stage 5's stochastic behavior-policy baseline. Quantized actor artifacts must
be fully converted, packed and published under an immutable version before use;
account for conversion time and never refresh scales in a live response/group.
Fusion intended for training still needs paired backward admission; this track
does not implicitly add QAT or low-bit backward.

### Combined validation and implementation checkpoints

Each F/Q/S milestone starts with a failing fixture for its new contract, then
the smallest implementation, targeted tests and full affected-path regression.
Keep the same checkpoint, prompt set, placement, clocks/runtime provenance and
measurement method for baseline comparisons; state any unavoidable difference.
Test fusion-only, quantization-only and speculation-only before combinations,
then re-run numerical/state/quality gates for each combined policy. Individual
speedups do not multiply automatically, especially when quantization reduces the
target-only latency that speculation is trying to amortize.

Existing regression entry points, to run after implementation in a configured
CUDA environment (not executed by this documentation change):

```bash
ctest --test-dir csrc/build-libs --output-on-failure
cabal test infer-tests infer-generation-tests --enable-tests
python tests/test_engine.py --library csrc/build-libs/libengine.so \
  --desc "$DESC" --model-dir "$MODEL_DIR" --reference "$REFERENCE" \
  --rms-tolerance "$RMS_TOLERANCE" --devices "$DEVICES"
```

Set the descriptor/reference/tolerance for the actual family; existing reference
and legacy-capture checks remain BF16 gates. Register `test_quant_gemm` in CTest
and give the proposed quantization/speculation runners explicit artifact and
policy inputs rather than reusing a BF16 reference with looser flags. Add a
reproducible inference benchmark entry point (`tests/benchmark_inference.py`)
for the F0/Q/S metrics; record repeated warm runs and dispersion, not only the
best timing. These future tests/benchmark are not evidence until implemented
and run. Do not advertise an optimization by default until its supported
family/shape/placement matrix and quality/resource/performance gates pass.

## Memory and target selection

All model-state numbers below use decimal GB and the same illustrative AdamW
assumption: BF16 weight 2 + BF16 gradient 2 + FP32 master/m/v 12 = **16 bytes per
trainable parameter**. FP32 gradient storage instead makes it 18 bytes, before
any extra accumulation/reduction buffers. Compare actual device bytes rather
than mixing GB/GiB or summing free memory as though it were one pool.

| Parameters | Full training states across the model | Implication |
|---|---|---|
| 0.6B | 9.6 GB | Reasonable first real-model scale; activations and objective buffers still need a bound |
| 1.7B | 27.2 GB | Potential single-card learner, subject to sequence/batch/activation budget |
| 4B | 64 GB | Needs suitable multi-device placement or offload; not a one-card learner for choice B |
| 8B | 128 GB | Does not fit the aggregate two-A40 budget using these states |

ZeRO reduces replicas, not this global lower bound. Two-way ideal ZeRO-3 for 8B
still needs 64 GB/card; ZeRO-2 with replicated BF16 weights needs 72 GB/card under
these assumptions. Activation checkpointing does not shrink optimizer states.
8B full-parameter training requires offload, a lower-memory optimizer/state format
or more device memory, not optimizer sharding alone.

Budget each device and phase: weights and ties/replicas, gradients/master/m/v,
saved/recomputed activations, loss/logit workspace, collective staging, rollout
KV/GDN state, reference/teacher/critic and actor active/staging snapshots. Current
PP assigns layers by count, not training-memory balance. For choice B, the learner
must fit on its own device and the actor must fit its active/staging copies and
cache on its device. If staging does not fit, stop/drain before replacing the
actor snapshot and count the lost overlap.

Sampling and inference optimizations add their own phase budget rather than
changing the 16-byte training-state assumption:

- Temperature sampling: existing host FP32 logits plus per-row binary64
  weights/CDF scratch and PRNG state. Measure boxed-list/GC overhead; do not
  retain a vocabulary-sized buffer for every generated token merely to record
  the selected token's two logprobs.
- Fusion: packed projection storage and temporary load-time copies, plus any
  saved rounded intermediates required if the fused path later becomes trainable.
- Quantization: packed weights, scales, excluded BF16 roles, backend workspace
  and peak converter/loading memory; W4 inference does not shrink FP32 master
  weights or AdamW slots in the proposed full-parameter trainer.
- Speculation: both models' weights, caches and scratch, target verification
  logits (`4*(k+1)*vocab_size` bytes for an FP32 host or device matrix), and
  checkpoints for each hybrid model. One GDN checkpoint stores, per local layer,
  `2*conv_dim*(conv_kernel-1) + 4*num_v_heads*head_dim*head_dim` bytes, plus
  metadata; multiply by the actual per-device layer/replica ownership. Count
  both host and device logits and replay workspace, not just compressed KV.

PPO's critic adds trainable states; reference/teacher/reward models add frozen
weights and forward workspace. Verifiable rewards need no reward model. OPD is
not necessarily cheap: a large teacher or dense vocabulary KL can dominate memory.

Separate two target decisions:

- **Correctness fixture:** reuse the tiny hybrid construction approach in
  `tests/synth/make_qwen3_next.py`, which already preserves supported GDN shapes.
  Its existing fixture contains MoE; either make a deliberately dense fixture
  and its independent reference, or explicitly add MoE backward scope. Do not
  accidentally make MoE a prerequisite for the first trainer.
- **Training-quality checkpoint:** evaluate a small dense model and the available
  Qwen3.5-0.8B candidate. Verify real config, head counts, role layout and AOT
  coverage before claiming support; current GDN requires 16 key heads and has
  recurrent specializations for 32/48 value heads. Small size is not compatibility.

A 9B hybrid with LoRA is an optional later target, not the only way to exercise
GDN backward. Compute adapter parameters from `sum r*(in+out)` over chosen
matrices; include BF16 adapters, gradients, optimizer and activation costs.
LoRA still needs gradients through frozen layers and explicit adapter execution.
It is not present in the current engine and has no pre-established 25 GB budget.

## Placement policy and remaining decisions

Preserve existing TP/EP inference. Initial trainer uses one device or PP, with
tested matching-architecture transfer boundaries. Any later TP/EP trainer adds
reduction/gradient contracts and resource tests; same implementation does not
make different parallel layouts bitwise invariant.

Before Stage 3 sizing, resolve:

1. The attention forward/backward pair and its saved-state/determinism costs.
2. Dense-hybrid fixture and real checkpoint compatibility, including any needed
   AOT shape specialization; no unverified model-size assumption drives the plan.
3. Per-region activation/recompute and per-device state budgets, gradient dtype,
   parameter aliasing and update/publication APIs.

Before positive-lag async experiments, resolve:

1. The actor/learner resource split with measured memory and snapshot bandwidth.
2. The named stale-data objective, acceptance bounds and supported lag. Async
   GSPO remains specifically unresolved rather than inheriting a token objective.
3. Reward deadlines/retry and group-drop policies, checkpoint boundaries and the
   metrics used to detect completion-time or stale-selection bias.

## Non-goals and acceptance discipline

- Only Stages 0–2 promise no arithmetic change. Training necessarily adds an
  ownership/API path; an unconditional "no engine restructuring" claim is false.
- Exact claims apply only to registered cases/configurations. Numerical correctness
  and optimizer learning checks remain necessary even for bitwise-equal outputs.
- No generic dynamic autograd, distributed multi-node scheduler, mid-generation
  policy switching, arbitrary replay buffer or combined TP+EP in the first trainer.
- GSPO is included with a precise synchronous objective. Async RL is a staged
  proposal, not a declaration that stale-policy GSPO is solved.
- Ordinary temperature sampling is a separate T0–T4 migration, and it has landed:
  default T=1 with explicit greedy T=0, without top-k/top-p, GPU sampling or
  stochastic speculation. Seed replay and the binary64 distribution are gated
  (`tests/SamplingSpec.hs`); model *quality* under sampling is not measured, and the
  migration's performance (host selection time, GC, TTFT, tokens/s) is not either,
  which the section records.
- Fusion, W4A16 and greedy speculative decoding are independent inference
  proposals, not implemented features or prerequisites for SFT. Their first
  admissions exclude quantized training/KV state and stochastic speculation;
  exactness and quality are separate gates, and combined policies need re-testing.
- No guaranteed reward improvement, universal determinism, zero-copy concurrent
  weights or speedup. Report results separately for fixed-step quality, wall-clock
  efficiency, numerical agreement and resource use.

## References

1. [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) and
   [implementation](https://github.com/zanderjiang/SkyRL-IsoExec): execution
   contract and unified numerical implementation; performance results are
   workload-specific, not this project's acceptance thresholds.
2. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) and
   [Qwen's GSPO explanation](https://qwenlm.github.io/blog/gspo/): sequence ratio,
   length normalization and sequence clipping.
3. [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language
   Reasoning](https://arxiv.org/html/2505.24298v2): asynchronous scheduling,
   staleness control and decoupled PPO. Our initial snapshot protocol deliberately
   does not adopt its mid-generation weight-switching behavior.
4. [Tree-Based Invariant Kernels](https://arxiv.org/abs/2511.17826) and
   [batch-invariant inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
5. [Bitwise-consistent train/inference](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference)
   and [linear-attention mismatch / asynchronous RL](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/).
6. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) and
   [FLA](https://github.com/fla-org/flash-linear-attention): mathematical reference
   and candidate kernels; compatibility with our forward remains a test obligation.
