# Plan: A Numerical Execution Contract, and a Haskell Training Framework

Status: proposed, not started. Extends `docs/design.md` §Testing strategy and
§Future work.

## Why this document exists

This project will grow a training framework, **also orchestrated in Haskell over
the same C/CUDA region library**, targeting small models with SFT, PPO, GRPO,
DAPO and on-policy distillation. That changes several conclusions which were
reasonable while the codebase was inference-only, because it introduces a second
consumer of the same arithmetic that must agree with the first one bit for bit —
and because four of the five algorithms need a rollout engine, which is this
engine.

The reference point is IsoExec (SkyRL team, vLLM blog 2026-08-21). IsoExec
eliminates the train/inference mismatch in RL workloads with two parts: an
**execution contract** declaring and enforcing every execution detail that
affects floating-point rounding, and a **unified model** whose kernels are
batch-invariant and bitwise identical across `trainer_fwd`, `engine_prefill` and
`engine_decode`. Measured on 8×H100, synchronous DAPO on Qwen3.5-35B-A3B: mean
rollout-vs-train logprob absolute difference 1.648e-2 → 6.744e-7, at 25.3%
end-to-end overhead.

Three things about that result shape this plan:

- IsoExec reports **no meaningful reward improvement** over 50 steps. What
  bit-exactness buys is debuggability and a well-defined contract, not model
  quality. Every justification below is argued on those terms.
- IsoExec's most expensive property is the one this project can get
  structurally. Their starting point is two engines (vLLM + Megatron) with two
  model definitions kept aligned by discipline and by a contract adapter on each
  side. With one Haskell orchestration layer composing one region library for
  both training and inference, "unified model" is true **by construction**. That
  is the strongest available version of the thesis in `docs/design.md:5-7`.
- A unified model has a second, less-discussed payoff: **no weight
  resharding between phases**. vLLM+Megatron setups must broadcast actor weights
  from the trainer's sharded layout into the rollout engine's layout every RL
  step — a first-order overhead. If the rollout engine and the trainer read the
  *same* weight buffers, that transfer disappears entirely. Corollary: this
  property holds only while both phases use the same parallel layout. Layer-wise
  (PP) placement gives that for free; introducing tensor parallel with different
  degrees for train vs. rollout would give it back. See §Placement.

## What a trainer changes

| Topic | Inference-only verdict | Verdict with a Haskell trainer |
|---|---|---|
| Execution contract | Nice-to-have; payoff is faster debugging | Architectural core; must carry `trainer_fwd` / `trainer_bwd` cases |
| Backward region library | Did not exist as a topic | **The critical path.** Nothing trains until it exists |
| CPR GatedDeltaNet | Skip — no third party to align with | Required for PPO/GRPO/DAPO; deferrable for SFT/OPD |
| Memory ownership | `engine_create` allocates its full budget up front | Rollout and trainer must co-reside; the budget becomes negotiable |
| Moving the layer/chunk loop to Haskell | Cleanup with modest benefit | Prerequisite: a trainer needs traversal modes the current FFI cannot express |
| Raw-pointer operator orchestration | Rejected | Rejected harder — IsoExec exposes *regions + contract*, not raw kernels |
| Typed tensor handles | Third choice | Rises: region-granularity FFI must pass tensors across the boundary, and gradients are just more tensors |
| "Layer-wise placement is bitwise inert" claim | Worth a comment | Must be a tested claim, and is now also what preserves zero weight resharding |

## Audit: contract fields vs. current state

IsoExec's contract has four fields. Mapping them onto this codebase:

| Contract field | Already present | Gap |
|---|---|---|
| `cases` | `engine_prefill` / `engine_decode` exist as C entry points | Not *declared* as cases. Prefill additionally contains two undeclared cases (chunked pipeline vs. 1-token recurrent), selected by a hardcoded `if (tokens == 1)` at `csrc/kernels/fla_gdn.cu:68` — invisible to any contract. `trainer_fwd` / `trainer_fwd_no_autograd` / `trainer_bwd` do not exist |
| `composition` constants | The descriptor already pins the reduction-decomposition parameters: `fla_chunk_size: 64` and `max_chunk: 128` (`src/Infer/Descriptor.hs:189,244`, range-checked at `:373`), plus `rms_eps`, `rotary_dim`, `norm_style`, `q_gate_interleave` | No `impl` identity. FlashInfer 0.5.3 / fla-core 0.5.2 / Triton 3.4.0 exist only as a build-time assertion (`csrc/triton/build_aot.py:145`) and a download list |
| `composition` arch | AOT cubins are built per architecture `86;89;90` with **no PTX fallback** (`csrc/CMakeLists.txt:4-7`); the launcher picks the image matching the device | **Architecture appears in no artifact.** The same source produces different SASS on sm_86 vs sm_90, so a golden capture is silently incomparable across GPU models, and nothing records this |
| `claims` | We rely on one claim constantly: layer-wise placement across devices is bitwise inert | Never written down, never tested. IsoExec independently justifies it ("PP moves whole layers between devices; as long as boundary dtype is fixed it does not split intra-layer reductions") — our boundary is fixed BF16 and `copy_across_devices` moves activations only, so the claim should hold, but it must be discharged by a test |
| `identities` (SHA-256) | `engine_describe` emits the canonical descriptor; `desc_version` exists | `desc_version: 1` is a *wire-format* version, not a content digest. There is no identity at all. Worse, `tests/capture_logits.py:91-92` explicitly `continue`s past `meta` during comparison, so the bitwise gate **discards all provenance** — and `meta` (`:70-72`) records only the descriptor's *path*, not its content, kernel versions, or arch |

Two contract entries already exist as prose instead of data:
`csrc/kernels/layers.cu:206` — *"Convolution returns unfused BF16; preserve its
rounding before SiLU"* — is a rounding-schedule declaration living in a comment;
`docs/design.md:258-260` attributes ~1e-2 logit movement to "BF16 operator
reassociation and cuBLAS algorithm selection". Both belong in the contract.

### A fifth field the inference-only analysis did not need: backward determinism

IsoExec's `cases` include `trainer_fwd_no_autograd`, which hints at the issue.
Backward passes are nondeterministic in a *worse* way than forward passes: not
merely shape-sensitive but **run-to-run nondeterministic**, because gradient
accumulation commonly uses atomics. Flash-attention backward accumulates dQ/dK
with atomics; weight-gradient reductions over the token axis (e.g. the
`dweight_workspace_ptr` / `dbias_workspace_ptr` fields already declared in
`csrc/third_party/causal_conv1d/causal_conv1d.h:56`) are cross-token reductions
with the same exposure.

Consequence: a contract that covers only the forward cannot deliver
reproducible training. The contract needs an explicit determinism tier per
backward region — `{deterministic, atomic, seeded}` — and any region marked
`atomic` is excluded from bitwise reproducibility claims by declaration rather
than by surprise. This is cheap to state now and expensive to retrofit after a
training run produces unreproducible loss curves.

## Known invariance gaps (measured)

1. **Chunked vs. recurrent GDN.** 1-token recurrent prefill vs. the chunk
   pipeline on the same prompt: logit RMS ≈ 1.0–1.3 (logit std ≈ 1.84). Root
   cause is the algorithmic fork at `csrc/kernels/fla_gdn.cu:68`, not a state
   bug. IsoExec measured the same phenomenon independently between FLA's chunked
   kernel and vLLM's fused recurrent kernel: mean elementwise absolute
   difference ~1e-4, max 0.25. A second team confirms our number is expected
   behaviour.
2. **Chunk decomposition is not invariant even within the chunked path.**
   `prefill(433)` vs. `prefill(150)+prefill(283)`: logit RMS 0.029. Both sides
   take the chunked path, so this cannot come from gap 1. The chunk shapes differ
   ({128,128,128,49} vs. {128,22} and {128,128,27}), which points at cuBLAS
   choosing a different algorithm per shape — exactly what `docs/design.md:258-260`
   already observes. The current tolerance (`docs/design.md:254`: "top-1 equal,
   rms ≤ 5") is a state-loss guard, not an invariance guard, and must tighten by
   three orders of magnitude for a trainer.
3. **Attention is already case-uniform** — favourable, and stronger than it first
   looks. `kernel_attention` builds one `flashinfer::SinglePrefillParams` and
   dispatches only on a *compile-time* template parameter
   (`head_dim == 256 ? launch_single_prefill<256,256> : launch_single_prefill<128,128>`,
   `csrc/kernels/attention.cu:135-137`). Decode is the `tokens == 1` case of the
   same call, and split-KV is disabled (`nullptr` workspace). So the reduction
   structure varies with neither token count nor `seq_len` at the call site; only
   FlashInfer's internal tiling over `seq_len` can vary. This narrows forward
   alignment to `gdn.core` alone.

Gap 2 matters most. A trainer sees `[B*S, H]` GEMM shapes; inference sees
`[chunk, H]`. If cuBLAS algorithm selection is shape-dependent, **identical code
on both sides still produces different bits**, and no contract discipline fixes
it. IsoExec's answer is a batch-invariant GEMM (their `pik`: split K into
contiguous leaves, deterministic Tensor Core MMA with FP32 accumulation,
contract-pinned leaf mapping and binary operation schedule, NCCL moving partial
sums only). Our entire GEMM path is cuBLAS (`checked_gemm` → `gemm_bf16` in
`csrc/kernels/layers.cu`), and the backward of a GEMM adds `dW = Xᵀ·dY` — a
reduction over the token axis, i.e. precisely where split-K bites hardest.

## What does not exist yet

Audited against the training goal; these are absences, not defects.

| Need | Current state |
|---|---|
| GDN backward | **Explicitly excluded.** `csrc/triton/gated_delta_rule_chunkwise.py:17` states "no backward / varlen / generic autotune surface"; all 7 AOT kernels are forward specializations, and `csrc/triton/build_aot.py:10` imports only `fused_recurrent_gated_delta_rule_fwd_kernel`. Training GDN layers requires adding FLA's chunked backward kernels to the AOT surface |
| Causal conv1d backward | Interface declared (`ConvParamsBwd` at `csrc/third_party/causal_conv1d/causal_conv1d.h:56`) but **implementation not vendored** — only `causal_conv1d_update.cu` is present. Needs the corresponding backward translation unit from upstream |
| Attention backward | **Absent entirely.** No backward path for `attn.core` anywhere in the tree. FlashInfer is inference-oriented; whether it offers a usable, deterministic prefill backward is an unverified risk and the single largest unknown in this plan |
| Optimizer | Absent. AdamW with FP32 master weights must be written (Haskell orchestration, CUDA or plain elementwise kernels) |
| Loss/objective regions | Absent. Cross-entropy + logprob gather for SFT; reverse-KL for OPD; clipped surrogate for PPO/GRPO/DAPO; group advantage normalization for GRPO; Clip-Higher + dynamic sampling + token-level loss for DAPO |
| Memory phase management | Absent. `engine_create` allocates its whole budget up front (`docs/design.md` §Memory budget). Rollout and trainer cannot currently co-reside |
| Sampling | Absent by design (`docs/design.md:12`, greedy only). PPO/GRPO/DAPO/OPD all require temperature/top-p sampling in the rollout |

## Plan

### Stage 0 — Contract skeleton and identity digest

~1 day. Purely additive; touches no kernel.

- Add a `numerical_policy` block to the descriptor:
  `{flashinfer, fla_core, triton, cubin_arch, accumulate_dtype}`. `cubin_arch` is
  filled by C from `cudaDeviceGetAttribute` at runtime, not hardcoded. Library
  versions are exported from the existing assertion at
  `csrc/triton/build_aot.py:145`, already the single source of truth.
- Add `engine_desc_identity()` returning the SHA-256 of that block.
  `engine_describe` already produces a canonical serialization.
- Split identities into IsoExec's three tiers: `semantic` (architecture —
  already covered), `numerical_policy` (implementations and constants that can
  move bits), `deployment` (proven not to move bits — memory planning, transfer
  config; **not required to match**). The tiering is what stops the contract
  flagging irrelevant changes.
- Reserve the backward determinism tier (`{deterministic, atomic, seeded}`) in
  the schema now, even though no backward region exists yet.
- In `tests/capture_logits.py`, write the identity into the npz and turn the
  `meta` skip at `:91-92` into a **precondition gate**: differing identities
  report "contract changed: X → Y" and exit before elementwise comparison.

Gate: the existing golden comparison still passes on an unchanged build.

### Stage 1 — Name the regions; build the cross-case bitwise harness

~3–5 days.

Distinct region *kinds* across the whole model:

- shared: `norms.gemma_l2`, `gemm.bf16`, `residual.add`
- attention: `attn.qg_deinterleave`, `attn.rope`, `attn.kv_write`, `attn.core`,
  `attn.output_gate`
- GDN: `gdn.conv1d`, `gdn.conv_silu`, `gdn.core` (two impls: chunked,
  recurrent), `gdn.gated_norm`
- MLP: `mlp.silu_mul`

Thirteen kinds. Per token forward they are invoked ~1168 times (16 attention
layers × 19 + 48 GDN layers × 18). At region granularity the orchestrator issues
~1168 calls per token; at ~20 ns per `ccall` that is ~23 µs against a 10–30 ms
decode step — **below 0.2%, so performance is not an argument at this
granularity.** (An earlier estimate treating "thousands of calls" as a
performance objection was wrong and is retracted here.)

Work: introduce a `RegionId` table in C and tag each kernel call site in
`csrc/kernels/layers.cu` and `csrc/layer_dispatch.cu` (naming and registration,
not a rewrite of the arithmetic); promote the `if (tokens == 1)` fork at
`csrc/kernels/fla_gdn.cu:68` from a hardcoded branch to a contract-selected
`(region, case) → impl` binding; declare today's cases
(`engine_prefill_chunked`, `engine_decode`) and reserve `trainer_fwd`,
`trainer_fwd_no_autograd`, `trainer_bwd`; build IsoExec's admission test — every
region must pass a cross-case bitwise comparison before its implementation may
be registered in `composition`. That test is the only mechanism turning
"RMS ≈ 1.0 here is expected" from institutional memory into a recorded fact.

Gate: `composition` lists every region with either a passing bitwise result or an
explicit, quantified exemption (e.g. `gdn.core`: chunked vs. recurrent max_abs
recorded, not zero).

### Stage 2 — Discharge the claims we already rely on

~2 days on the existing 2-GPU target.

- **Claim A (PP inertness):** same prompt, all layers on one device vs. the
  layer-wise split across two; require bitwise-identical logits. Load-bearing
  today, untested, and now also the property that keeps weight resharding at
  zero.
- **Claim B (chunk decomposition):** quantify gap 2 and record it with its number
  attached, replacing the loose `rms ≤ 5` tolerance.
- **Claim C (attention split-KV):** record "split-KV disabled" as a contract
  constant, citing `csrc/kernels/attention.cu:135`.
- **Claim D (GEMM batch invariance) — highest information per unit cost.** Fix
  weights and input; vary only token count (128 vs. 64+64); compare one GEMM
  region's output bitwise. One iteration (~7–12 min including a cold weight load)
  decides whether Stage 5 is thousands of lines or tens of thousands.
- **Claim E (attention backward feasibility) — new, and blocking.** A spike, not
  a test: determine whether a deterministic prefill backward for `attn.core` can
  be obtained from FlashInfer, must be written, or must be borrowed from another
  library. Stage 3 cannot be sized until this is answered.

Gate: A and C pass; B and D have recorded numbers and a verdict; E has a decision.

### Stage 3 — Backward region library (critical path)

Not sizeable until Claim E resolves. Rough difficulty by region:

| Region | Backward | Difficulty |
|---|---|---|
| `residual.add` | passthrough | trivial |
| `attn.qg_deinterleave` | re-interleave | trivial |
| `mlp.silu_mul`, `attn.output_gate`, `gdn.conv_silu` | elementwise | easy |
| `attn.rope` | inverse rotation | easy |
| `norms.gemma_l2`, `gdn.gated_norm` | RMSNorm / RMSNormGated backward | moderate (FLA provides the gated variant) |
| `gemm.bf16` | two GEMMs (dX, dW) | easy to write, **hard to make batch-invariant** — dW reduces over tokens |
| `gdn.conv1d` | vendor upstream bwd | moderate; workspace-based weight grad is an atomicity risk |
| `gdn.core` | FLA chunked backward, added to the AOT surface | hard, but the kernels exist upstream |
| `attn.core` | flash-attention backward | **hard and unverified** — the plan's largest risk |
| `attn.kv_write` | none | inference-only region; must be declared absent from `trainer_*` cases |

Two structural notes. First, `attn.kv_write` and the KV cache have no place in a
trainer forward: teacher forcing computes causal attention over a fresh
`[T, T]` mask with no cache. That is a genuine case difference and must be
declared in the contract, because cached attention over `seq_len` and uncached
attention over `T` can tile differently even when mathematically identical.
Second, backward composition is a *fixed region sequence* for a fixed
architecture — Haskell expresses it as a list, with no general autograd tape.
That is simpler and more auditable than a generic engine, and it is what makes
Haskell-orchestrated training realistic.

### Stage 4 — Memory phase co-residency

Required before any rollout-based algorithm (PPO/GRPO/DAPO/OPD) can run at all.

`engine_create` currently allocates its full budget up front. An RL step needs
rollout (KV cache, no gradients) and training (activations, gradients, optimizer
states) on the same devices. Options, in increasing invasiveness: a negotiated
budget parameter; freeing/reallocating the KV cache between phases; gradient and
optimizer offload to host between phases. Because the weights are shared by
construction (§Why this document exists), weight resharding is *not* among the
things to build — that is the payoff of the unified design and should be
measured and reported as such.

### Stage 5 — CPR and batch-invariant GEMM

Expensive; gated on Stages 2–3.

- **CPR** (chunkwise-parallel recurrent) needs to cover `gdn.core` only, since
  attention is already case-uniform — far less work than IsoExec, which also
  handled TP/EP/SP invariance and MoE routing. CPR keeps the recurrence as the
  primary function but solves it chunk-parallel: prefill computes chunk-boundary
  states then runs a parallel recurrent scan within chunks; decode stays
  recurrent and re-synchronises hidden state every C tokens. IsoExec's per-layer
  costs: prefill 1.67×, decode 1.38×, trainer fwd+bwd 1.43×.
- **Do not** take TorchTitan's all-recurrent route (every forward recurrent, only
  backward chunked): measured 4.42× trainer fwd+bwd, 4.31× prefill, 2–3× slower
  end-to-end on math and ~5× on agent workloads; IsoExec judges it impractical.
- **Batch-invariant GEMM** is gated on Claim D. If cuBLAS is shape-sensitive the
  options are (a) a fixed-reduction-order GEMM along `pik` lines with FP32
  accumulation, (b) pinning cuBLAS algorithm and workspace explicitly with split-K
  disabled — partial, not guaranteed, or (c) declaring GEMM regions out of
  contract scope and saying so loudly.

### Stage 6 — Algorithm bring-up, in risk order

Not in order of importance — in order of how much contract machinery each one
exercises before it can produce a correct gradient.

| Algorithm | Rollout | Reference | Critic | Teacher | Importance ratio | Mismatch sensitivity |
|---|---|---|---|---|---|---|
| SFT | no | no | no | no | no | **none** — validates Stage 3 alone |
| OPD | yes | no | no | yes | no (dense per-token reverse KL) | low — rollout affects only *which* tokens appear |
| DAPO | yes | no (KL term dropped) | no | no | **yes** | high |
| GRPO | yes | usually | no | no | **yes** | high |
| PPO | yes | yes | **yes** | no | **yes** | high, plus a second trainable model |

Recommended order: **SFT → OPD → DAPO/GRPO → PPO.** SFT needs no rollout, so it
validates the backward region library with zero contract exposure. OPD adds
rollout co-residency and sampling but has no importance ratio, so a residual
mismatch degrades it gracefully rather than collapsing it. DAPO/GRPO are where
the contract earns its keep — the Fireworks GLM-5.2 report (train-infer KL ≈
0.013 → ~45% of tokens clipped → reward collapse around step 20, versus a
bitwise-aligned run with zero clipped tokens) is the failure mode being defended
against. PPO comes last because the critic is a second trainable model: most
memory, most complexity, and it multiplies every gap above.

## Trainer scope constraints

Mixed-precision AdamW costs 16 bytes/parameter (BF16 weight 2 + BF16 gradient 2
+ FP32 master 4 + FP32 m 4 + FP32 v 4). Against the 2× A40 46 GB target (92 GB):

| Model size | Full-param states | Verdict on 92 GB |
|---|---|---|
| 0.6 B | 9.6 GB | comfortable |
| 1.7 B | 27 GB | comfortable |
| 4 B | 64 GB | fits, tight once activations and KV cache are counted |
| 8 B | 128 GB | not without ZeRO-2/3 + activation checkpointing |

So "small models" means roughly ≤4 B full-parameter, and 7–8 B only with optimizer
sharding. Rollout-based algorithms multiply this: PPO at 1.7 B needs actor 27 GB
+ critic 27 GB + frozen reference 3.4 GB + reward model 3.4 GB ≈ 61 GB before
activations or KV cache. GRPO/DAPO/OPD need roughly half that (no critic; DAPO no
reference). At 0.6 B everything is comfortable.

**The tension worth naming:** the region that most needs the contract
(`gdn.core`, the only algorithmic fork) belongs to the hybrid-GDN model class,
whose smallest members are around 9 B — too large for full-parameter training
here. The resolution is LoRA on a hybrid model: a frozen 9 B BF16 base is 18 GB,
LoRA at r≈16 on the projection matrices adds ~10⁸ trainable parameters and
~1.4 GB of optimizer state, so the whole thing lands near 25 GB and leaves room
for a frozen teacher (OPD) or reference (GRPO). That combination — **9 B hybrid +
LoRA — is the only configuration that exercises GDN backward and fits**, and it
should be the primary training target. The other descriptor families
(`qwen3-30b-a3b`, `mixtral-8x7b`) do not help; both exceed the budget.

An open question this creates: whether a *smaller* hybrid-GDN checkpoint exists to
train full-parameter. If one does, it is a better Stage 3/6 vehicle than LoRA,
because full-parameter training exercises `gemm.bf16` backward on every weight
rather than on adapters only.

## Placement

`src/Infer/Placement.hs:20-21` currently offers only `Pipelined`, with
tensor/expert parallel deferred (also `docs/design.md:275`). Two independent
reasons to keep it that way until the contract exists:

- **Numerics.** Layer-wise placement does not split intra-layer reductions.
  Tensor parallel splits the contraction dimension across ranks; expert parallel
  changes how expert outputs are combined. Once either lands, logits become a
  function of device count unless a fixed reduction tree is pinned in the
  contract (IsoExec's `pik`, or TBIK's global tree for the row-parallel case).
- **Weight residency.** PP keeps one copy of the weights readable by both phases.
  TP with different degrees for train and rollout reintroduces per-step
  resharding — the overhead the unified design was supposed to eliminate.

This belongs in the `Policy` comment now, while it is free.

## Open decisions

Resolved: the trainer is Haskell-orchestrated over the same region library; the
targets are small models with SFT, PPO, GRPO, DAPO and OPD.

Still open, and blocking Stage 3 sizing:

1. **Claim E** — can a deterministic `attn.core` backward be obtained, or must it
   be written? This is the largest unknown in the plan.
2. **Primary training target** — 9 B hybrid + LoRA (fits, exercises GDN), or a
   smaller hybrid checkpoint full-parameter if one can be found (better GEMM
   backward coverage)?
3. **Sampling** — `docs/design.md:12` records greedy-only as a deliberate constraint.
   Every rollout-based algorithm needs temperature/top-p. This is a small piece
   of work but it is on the critical path for Stage 6 and is not yet scheduled.

Stages 0–2 depend on none of these. They are additive, protected by the existing
golden comparison, and reversible.

## What this plan deliberately does not do

- Does not restructure the engine. IsoExec kept vLLM's scheduler, KV cache manager
  and CUDA graph capture, and Megatron's training stack, injecting the contract
  through adapters at existing extension points. The same minimal-touch approach
  applies here.
- Does not change any numerics. Stages 0–2 must leave the golden comparison
  bitwise identical; that is the gate, not a hope.
- Does not promise better model quality. IsoExec observed no reward gain in 50
  steps. The payoff is that a mismatch becomes a named, located, recorded fact
  instead of a multi-iteration GPU debugging session — plus, uniquely to the
  unified design, the elimination of per-step weight resharding.

## References

- IsoExec (vLLM blog, 2026-08-21): `vllm.ai/blog/2026-08-21-isoexec`
- Implementation: `github.com/zanderjiang/SkyRL-IsoExec`
- TBIK, Tree-Based Invariant Kernels: `arxiv.org/abs/2511.17826`
- Bitwise-consistent train/inference (vLLM × TorchTitan):
  `vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference`
- Zero train-inference mismatch for linear attention / async RL (TorchTitan,
  all-recurrent GDN): `yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/`
- Gated DeltaNet: `arxiv.org/pdf/2412.06464`
- FLA (flash-linear-attention): `github.com/fla-org/flash-linear-attention`
- Defeating Nondeterminism in LLM Inference (Thinking Machines, batch
  invariance): `thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/`
