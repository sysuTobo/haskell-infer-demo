# Plan: Numerical Execution Contract, Haskell Training, and Asynchronous RL

Status: Stages 2–8 proposed, not implemented; **Stages 0 and 1 are implemented**
(the versioned execution manifest with capture provenance, and the region
inventory with its cross-case harness — see [manifest-contract.md](manifest-contract.md),
[worklog.md](worklog.md) and `csrc/regions.c`).
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

A [temperature-sampling migration](#temperature-sampling-migration) changes the
planned default generation policy from greedy to categorical sampling, while
retaining explicit greedy regression. It can be implemented without training or
CUDA changes and supplies the sampling foundation for Stage 5.

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
| Orchestration | Haskell model/configuration/placement/generation; C layer/chunk loops | Training traversal, activation lifetimes, region-level opaque handles |
| Inference API | `engine_create/prefill/decode/reset/destroy`; final-row logits | All-token loss/logprob evaluation, parameter updates, trainer API |
| Mixers/FFNs | Full attention, GDN, MLA; dense and MoE | All backward paths; training support matrix |
| Placement | `Pipelined` and `Replicated tp ep`; separate TP and EP | Combined TP+EP; trainer collectives; PP bitwise-invariance evidence |
| GDN | Chunked path plus recurrent `tokens == 1`, including one-token tails | Cross-case bitwise invariance; prepare/core/backward pairing |
| Attention | One FlashInfer prefill dispatcher; null split-KV workspace | Invariance across query/KV lengths, cached and training layouts |
| GEMM | cuBLAS BF16 and FP32-output paths | Batch-invariant forward; specified dW/gradient accumulation order |
| Descriptor | Strict flat schema and canonical C formatting | Semantic/numerical digests and resolved execution manifest |
| Artifacts | nvcc SASS `86;89;90a` plus `90-virtual`; Triton cubins `86;89;90` | Capture provenance identifying the actual selected kernels; Triton has no PTX fallback |
| Weights | Engine-owned BF16; derived GDN norm FP32 copy | Shared trainable ownership, tied-gradient merging, derived-copy refresh |
| Fusion | FlashInfer attention, SiLU-multiply and GDN gated norm; dense gate/up still separate GEMMs | Combined projection layouts, residual-add/norm fusion and measured end-to-end benefit |
| Quantization | Weight-role loading requires BF16; GEMM uses BF16 inputs with FP32 compute | Packed low-bit weights, scales, format validation, quantized GEMM and quality gates |
| Speculative decoding | Batched prefill internally, but only final-row logits; reset clears all sequence state | Draft/target orchestration, per-position verification, prefix rollback and GDN state snapshots |
| Sampling | Four argmax call sites in streaming/non-streaming generation; host `[Float]` logits; no temperature/seed options or RNG dependency | Shared token selector, temperature categorical distribution, request-owned RNG, sampler logprobs and distribution/replay gates |
| Training | No optimizer, objectives or saved-activation runtime | Full training path, checkpoint/resume and independent gradient tests |
| Rollout | Greedy single-request inference | Stochastic sampling, behavior logprobs, groups, rewards, versioned queues |

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
Stage-1 one. No pair is `exception` yet, because an exception has to carry tested
shapes, an architecture and a measured max_abs/rms, which is what Stage 2
produces. `region_ffi` prints the region entry-point host cost against its device
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
and `backward` in `gemm_bf16`, all for a traversal that does not exist. That column is hashed into
`numerical_policy_id`, so it must not advertise coverage a region cannot be
exercised for; those claims are removed, and `test_region_inventory` now refuses a
manifest row that names a traversal case.

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

The current implementation is **greedy decoding**, not stochastic sampling:
`src/Infer/Generation.hs::generate` and `generateStreaming` select argmax at
prefill and every decode step. `src/Infer/Config.hs` and `src/Main.hs` have no
temperature or seed fields. `src/Infer/FFI/Engine.hs` already returns host FP32
logits as `[Float]`; the first sampler therefore belongs in Haskell and needs no
C ABI, CUDA kernel, weight format or architecture-descriptor change.

Proposed end state, after T0–T4 gates pass:

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

- [ ] **T0 — Lock configuration and greedy baseline.** Write tests for default
  temperature, T=0/T>0, invalid numeric inputs, seed range and configuration
  parity; extend greedy tests for exact ties. Define a shared `SamplingConfig`
  with validated Double temperature and optional Word64 seed, used by Main and
  generation rather than duplicated defaults. Audit greedy examples/scripts
  before changing the default; use explicit greedy in regression fixtures.
  Invalid configuration must fail before any model allocation.
- [ ] **T1 — Implement and test the pure selector.** Separate preparing weights
  from choosing with a supplied u, so deterministic CDF/normalization tests do
  not depend on PRNG behavior. Add independent softmax/logprob reference values
  and boundary tests before implementation. Initially consume the existing
  `[Float]` FFI output with bounded per-row scratch and strict folds; avoid
  retaining lazy chains across decode steps. No FFI/vector rewrite or GPU
  kernel is necessary. Register `Infer.Sampling` and `SamplingSpec` in affected
  Cabal components; do not inadvertently make unrelated config-only tests
  depend on random IO.
- [ ] **T2 — Add request RNG and unify token selection.** Pin the RNG dependency
  in the executable and generation-test component, add fixed word/uniform test
  vectors, then pass one state through all four current argmax call sites.
  Both generation entry points use the same selector and next-state contract.
  Resolve options once in Main and pass the validated configuration instead of
  reading globals/environment per token. Extend the C stub to supply arbitrary
  per-step vocabulary rows, not only one +1 winner with all other logits -1.
- [ ] **T3 — Verify generation lifecycle and real CLI behavior.** Preserve
  first-token budget counting, returned EOS, pending-token consumption, engine
  error propagation and stream flushing/cleanup. A sampled EOS is handled by
  the same stop path as a greedy EOS. Characterize the existing first-EOS versus
  later-EOS text-feed asymmetry in tests; do not change special-token rendering
  as an unrelated sampling fix. Check new request/seed behavior, capacity errors,
  arbitrary logits, prefill/decode failures and zero-budget no-RNG behavior.
  The current `infer-generation-tests` does not compile CLI Main: use the
  executable runner for option/default/help and stderr-seed checks, including
  invalid arguments with a nonexistent model path to prove early validation.
- [ ] **T4 — Admit default change and document boundaries.** First run CPU
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

- [ ] **F0 — Establish a costed baseline before choosing fusions.** Record
  per-region CUDA time, launch count, host synchronization and memory traffic for
  decode M=1 and representative prefill M=2/64/128, within each descriptor's
  limits. Include layer placement and full request wall time. Run timing with
  taps disabled, after warm-up, and synchronize only at measurement boundaries;
  keep diagnostic captures separate. Weight traffic, MoE host-offset sync and
  device transfers may dominate launch savings.
- [ ] **F1 — Merge dense gate/up projections.** First add fixtures for existing
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

- [ ] **Q0 — Freeze a versioned format and independent reference.** Start with
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
- [ ] **Q1 — Implement converter, manifest validation and ownership.** Write
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
- [ ] **Q2 — Add real weight-only execution, then model gates.** Start from
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

- [ ] **S0 — Build a sequential correctness prototype first.** Own two
  independent runtimes, prefilling the identical token prefix. Validate complete
  tokenizer/token-ID and special-token compatibility, not just vocabulary size;
  use the same prompt/template encoding and check both context capacities.
  Draft and target may have different architectures. Start with a fixed small
  proposal count, e.g. 2–4, and greedy generation only. Verify candidates with
  target `decode` calls in the original sequential order; never feed a rejected
  candidate to this target baseline. Recover draft by reset plus sequential
  prefix replay. This validates acceptance, pending-token and lifetime logic,
  but is explicitly not a speedup. Keep the normal target-only path available.
- [ ] **S1 — Add bounded all-position verification and append-cache rollback.**
  Introduce a proposed `engine_verify_rows` API that consumes n token IDs and
  returns n FP32 vocabulary rows, with explicit output capacity and checked
  sizes. Apply final norm and LM head to every input row, respecting `max_chunk`;
  keep ordinary prefill/decode final-row behavior intact. Start with the small
  window's bounded `[n,V]` host result and existing Haskell argmax semantics;
  tiled output or GPU top-1 is a later measured extension, not an implicit FFI
  contract change. Add append-only cache truncation restricted to the current
  sequence and an available prefix. Initially admit pure full-attention targets
  and drafts; MLA needs its own admission, and any GDN model waits for S2.
- [ ] **S2 — Add hybrid state checkpoints and restore/replay.** Introduce
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
- Ordinary temperature sampling is a separate T0–T4 migration: planned default
  T=1 with explicit greedy T=0, without top-k/top-p, GPU sampling or stochastic
  speculation. Seed replay, distributional correctness and model quality are
  separate claims. This plan does not change today's greedy implementation.
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
