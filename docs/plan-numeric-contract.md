# Plan: Numerical Execution Contract, Haskell Training, and Asynchronous RL

Status: proposed, not implemented. Revised after the design review, 2026-09-23.
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

### Stage 0 — Versioned contract and capture provenance

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

Inventory the initial **dense/dense-hybrid** path, not a supposed thirteen-region
vocabulary covering the whole framework:

- Embedding gather, plain/Gemma RMSNorm, per-head norm, BF16 and FP32-output GEMM,
  residual addition, LM head, FP32 log-softmax/gather and masked losses.
- Attention Q/gate split, RoPE, KV write, attention core and output gate.
- GDN conv1d, conv SiLU, prepare (Q/K L2 normalization, head expansion,
  a/b/A_log/dt_bias transforms), core and gated norm.
- Dense MLP SiLU-multiply.

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
kernels. Then add phase-specific memory budgets and stochastic rollout:

- Synchronous rollout borrows a committed BF16 parameter version. No optimizer
  write overlaps readers. Release/reset KV and GDN state at the phase boundary;
  retain optimizer state or explicitly offload it according to the budget.
- Capture sampled token IDs, per-token behavior logprobs, rewards, masks, terminal
  reasons and version IDs at generation time. Never reconstruct an old denominator
  using updated weights. Distinguish frozen reference/teacher from behavior policy.
- Begin RL tests with categorical sampling at temperature 1 and no top-k/top-p
  truncation. This makes the sampling distribution equal to model softmax.
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
