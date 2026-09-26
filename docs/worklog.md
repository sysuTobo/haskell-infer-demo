# Work Log

A running snapshot of what this project can do today, what has been verified and
where it is knowingly incomplete. [README.md](../README.md) describes the
component layout and how to build and test; [design.md](design.md) holds the
architecture rationale; [plan-numeric-contract.md](plan-numeric-contract.md) is the
proposal for the trainer, RL and inference-optimization work — of which **Stages 0
through 5, Stage 6's decision layer, Stage 7's group objectives and Stage 8's lag-zero
admission protocol are implemented here**: the execution manifest with capture
provenance (specified in [manifest-contract.md](manifest-contract.md)), the region
inventory with its cross-case harness (`csrc/regions.c`), the feasibility/invariance
experiments that measured what those regions actually guarantee, the trainable
runtime's parameter lifecycle (`csrc/include/train.h`, `src/Infer/Trainer.hs`), the
backward/loss/optimizer layer and the synchronous SFT/rollout baseline
(`csrc/include/backward.h`, `csrc/include/train_loop.h`), the numerical-alignment
decision (`csrc/include/alignment.h`), the GSPO/GRPO group objectives and the bounded
rollout queue (`csrc/include/rollout_queue.h`), and the temperature-sampling migration
(`src/Infer/Sampling.hs`, T0-T4). All are described below. What remains a proposal is the
asynchronous GPU half of Stage 8 (snapshots, device leases, publication transfer, throughput
measurement) and the inference-optimization track (F/Q/S).

Last updated: 2026-09-26.

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
| `ctest --test-dir csrc/build-libs` | 33/33 — `test_model_desc`, `test_safetensors`, `test_manifest`, `test_manifest_hashes`, `test_region_inventory`, `test_backward`, `test_train_loop`, `test_alignment`, `test_gspo`, `test_rollout_queue`, `test_engine_resources`, `test_collective`, `test_attention`, `test_gdn`, `test_moe`, `test_mla`, `test_norm`, `test_rope`, `test_region_cases`, `test_backward_kernels`, `test_sft`, `test_gdn_invariance`, `test_attention_invariance`, `test_gemm_invariance`, `test_attention_lse`, `test_train`, `test_train_forward`, `test_library_ops`, `test_quantization_format`, `test_quantization_format_python`, `test_quant_manifest`, `test_quantization_converter`, `test_quantization_kernel` (on a **shared** GPU `test_engine_resources` reads the device's free memory before and after four failing `engine_create` rounds, so it needs a quiescent device: 2026-09-26 a co-tenant holding 28 GB on device 0 moved the reading by 736 MiB and failed a 16 MiB slack, the same test passed on the idle device 1, and the suite's second run of the day — with the device free — was 25/25) |
| `ctest test_backward` (CPU, Stage 4) | the Stage-4 region table against the Stage-1 inventory in both directions; masked CE, reverse KL, the token/sequence clipped objective and the group advantage reduction against an independent FP64 implementation with its gradient checked by central difference; AdamW against FP64 in PyTorch's own order (including the bias-correction/eps ordering) plus the BF16 publication; the checkpoint round-trip with every failure mode refused; a deterministic fixture overfitting 32/32 through this stage's own loss and optimizer, and a 12+8 resumed run landing bitwise on the uninterrupted 20-step parameters, moments and cursor |
| `ctest test_backward_kernels` (Stage 4, 2× A40) | every backward against a double-precision definition or a central difference of one: elementwise gates, residual branches, plain/Gemma RMSNorm, the GDN L2 norm, the gated norm, embedding (repeated ids summed), RoPE (the transposed rotation inverting the forward's), the Q/gate re-interleave, GEMM dX/dW, masked CE, AdamW, conv1d (d_x, d_weight, d_bias, d_state_in), GDN prepare (d_conv_out, d_a, d_b, d_A_log, d_dt_bias) — all 1e-8…1e-6 except the finite-difference rows at 1e-7…1e-3; attention forward+LSE vs the definition (2.1e-3 BF16 out, 1.5e-3 LSE) and its backward vs the FD of the definition (dQ 2.5e-4, dK 2.8e-4, dV 1.0e-3), with an analytic double reading of the device LSE reproducing the FD, so the base-2 convention is pinned; the GDN core backward with a nonzero initial state and a nonzero final-state gradient, one chunk and three chunks both matching the same reference (≤1.0e-7), d_state_start included; dQ and the whole GDN core backward bitwise reproducible, dK/dV reported (atomics) |
| `ctest test_sft` (Stage 5, A40/L20) | SFT on the tiny dense checkpoint (0.72M params, tied embeddings, prompt-masked target) against a `transformers` training run: the first step's loss matches to 6.3e-05 relative; the tied parameter's gradient (after one large-lr AdamW step, where the move's sign *is* the gradient's) has cosine **0.9990** over all 131072 elements; the engine overfits 7.0322 → 0.6084 and the reference 7.0318 → 0.5652; the forward is bitwise reproducible run to run; the training state (FP32 masters and both optimizer moments) round-trips through export/import bitwise; and a pipeline/TP placement, an optimizer step during a live step and a NaN logit are each refused. The rollout section drives `engine_rollout_sample`: the record's fields, the seed reproducing a completion bitwise and a shorter rollout being its prefix, EOS vs the length limit, **20000 draws** whose mean `-log p` is 6.90837 ± 0.00162 against the distribution's entropy of 6.90495 (+2.11 sigma) with all 415 counted bins inside 4.5 sigma (worst 2.94), the record's version being the one the engine *read* (a +1 stamp refused) and an optimizer step refused while the borrow is live, and — for the same completion — the trainer's FP32 denominator differing from the sampler's FP64 one by 4.768e-07 (ratio 0.99999976), with the recorded denominator bitwise unchanged after a publish. Its **group** section generates four completions of one prompt (seeds as part of the fixture), gives each a reward from a deterministic check on the completion, records them into one `TrainGroup` whose version is pinned (a fifth completion generated *after* a publication is refused), reads the rewards back, reduces the advantages (+1.000/−1.000/+1.000/−1.000 at mean 0.5 and population std 0.5), refuses a zero-variance subset unless the caller waives it, runs the **sequence-level objective over the engine's own record** (ratio exactly 1 at unchanged parameters, the gradient's sign following the advantage, +0.25 nat/row giving 1.284025 against the records' 1.284025), and checks the offload declaration round-trips and is cleared at the boundary |
| `ctest test_train_loop` (CPU, Stage 5) | the phase budgets (only SFT holds optimizer state, activations and retained values; a rollout's total is strictly smaller), the phase machine (a rollout creates a real borrowing context, so the store refuses an update while it reads; a phase boundary resets the sequence), the **optimizer-offload declaration** (a phase starts at nothing, a negative count or a declaration outside a phase is refused, the count round-trips, and the boundary clears it so the next phase cannot inherit it), the selection records (a record stamped with a superseded version cannot be read, and a group cannot mix versions), the sequence-level ratio at unchanged parameters being exactly 1, the objective/advantage items (masks, negative and positive advantages, a zero-variance group refused), the host FP64 sampler (frequencies 0.6652/0.2447/0.0902 against the softmax 0.6652/0.2447/0.0900 over 200k draws, inside three standard deviations; the same seed reproduces the draw sequence), the **explicit host-FP64 vs trainer-FP32 log-probability difference** the plan asks for (2.98e-08 on the toy distribution, measured rather than assumed), and that EOS and truncation are distinct terminal reasons with the mask distinguishing an EOS completion from a truncated one |
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
| `cabal test all --enable-tests` | `infer-tests` 66/66, `infer-generation-tests` 54/54 (15 generation-loop plus 35 temperature-sampling plus 4 loop-level sampling), `infer-trainer-tests` 9/9 (the Haskell and C teacher-forcing plans must agree) |
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

**S2: round checkpoints, which admit a model with a recurrent layer** (plan S2; verified
2026-09-26).

- `engine_checkpoint_save` / `_restore` / `_release` are an **engine-owned, opaque** checkpoint of
  one round's state, at most one per engine (the plan's "at most one round checkpoint per engine
  initially", so `save` reports success rather than handing back a handle). What is saved is every
  buffer `engine_reset` would clear - the attention and MLA caches and the GDN convolution and SSM
  state - plus the sequence length, the reset generation and the parameter identity. The whole
  reset set is copied rather than only the recurrent part: the KV and MLA entries are immutable up
  to the retained length, so their length *would* suffice, but classifying buffers by role would
  be a second registry to keep in step, and the copy is what makes a restore correct across any
  sequence.
- **A restore refuses rather than guessing** when the saved state is no longer the engine's: after
  `engine_reset` (the reset generation moved), after a failed forward (the engine already requires
  a reset, and a checkpoint does not clear that - the plan's "do not merely set `state_valid`"),
  and after the weights **or the numerical policy** changed. The packed operands do not move the
  weight digest, which is why the quantization posture is recorded separately - and that is what
  makes loading INT4 after a save observably invalidate the checkpoint.
- `generateSpeculative` takes the checkpoint at each round boundary when the rollback mode asks
  for it, and on rejection restores it and replays exactly the retained suffix - the plan's
  "restore the round-start state, then replay exactly the retained inputs". A fully accepted round
  restores nothing (the target is already at the retained prefix and only the draft's
  never-consumed last proposal is missing), which the CPU suite pins. The checkpoints are released
  when the run ends, because a live one would refuse the next run's save; the mode comes from the
  descriptor, so **the earlier "any GDN model waits for S2" refusal is gone** - S2 is that wait
  arriving.
- Gate `tests/test_checkpoints.py` on the real engine: on a model with GDN layers, three decodes
  after a restore are **bitwise** what a second engine that never took the detour produces (a
  6.46 MB checkpoint on the synthetic hybrid fixture), and the refusals all fire - a second save,
  none saved, after a reset, after a release, and after a policy change. `cabal test
  infer-generation-tests` is **73 examples, 0 failures** (locally and on the pod) with the two new
  checkpoint cases, and the full suite is **34/34**.
- One compile error of mine the pod caught: `free_checkpoint` was defined after `engine_destroy`,
  which uses it (the local CPU tests do not build that file).

**S1: bounded all-position verification and append-cache rollback** (plan S1; verified
2026-09-26).

- `engine_verify_rows` consumes n ids as **one bounded batch** and returns every row's FP32
  logits. The ordinary path computes only the final row on purpose (to avoid a
  `[max_chunk, vocab]` buffer), so this allocates one lazily and normalizes the whole activation
  in a single call. It refuses n > `max_chunk` rather than silently chunking - a chunked
  verification would answer rows from a different execution case - and refuses a host buffer
  shorter than `n x vocab` rather than overrunning it.
- `engine_truncate` takes the sequence back to a retained length, so the next append overwrites
  what was dropped. It is admitted only for a model whose every layer is full attention: a GDN
  state cannot be rewound and MLA needs its own admission, so the call *refuses* rather than
  leaving the cache and the recurrent state disagreeing. That admission is S2's checkpoints.
- `generateSpeculative` now uses both: the round's verification is one batch, and the rollback is
  "truncate the target back to the retained prefix, and either truncate the draft or make it
  consume its missing last proposal" instead of S0's full replay. That is the plan's "on full
  acceptance draft must additionally consume its missing yk", and it is what stops a rejected
  round from costing a replay of the whole prefix. The ordinary-step branch (no legal window)
  also steps the draft, so the two runtimes still hold the same prefix.
- Admission is checked before anything is generated: the CLI refuses a target or draft with a
  recurrent layer, so the rollback's refusal cannot arrive mid-run.
- Gates: `tests/test_verify_rows.py` on the real engine - n = 1 is **bitwise** a decode, and for
  n = 4 the batch rows match serial execution to **2.4e-7 max_abs with all four argmaxes
  agreeing**, the rollback leaves the batch engine exactly where a clean prompt is, and a short
  buffer, a window past `max_chunk`, a truncation past the sequence and a GDN model are all
  refused. The scripted-engine half is in `cabal test infer-generation-tests` (**71 examples, 0
  failures**), where two new counters pin that a rejecting round truncates the target and a fully
  accepted one does not, and that the draft advances rather than replays.
- One bug of mine the wiring exposed: with truncation-based rollback the ordinary-step branch had
  left the draft one token behind, which the sequence-length assertion caught.
- **Not done**: S2 (checkpoints, so a rejected round is O(1) instead of truncate-or-catch-up and
  a GDN model can be admitted) and S3 (the cross-case and continuation gates, and any acceptance
  or throughput measurement).

**S0: the speculative-decoding prototype, two runtimes and the round protocol** (plan S0;
verified 2026-09-26).

- `Infer.Generation.generateSpeculative` is the plan's S0: two independent runtimes over the same
  prompt, greedy only, a fixed window, and the target verifying candidates with **sequential**
  decode calls. It keeps the plan's invariant - both engines have consumed `P` and the last
  confirmed token is still pending - takes the longest matching prefix with the target's own
  argmax as the correction (or the bonus row on full acceptance), emits only target-confirmed
  tokens up to and including the first EOS, and bounds the window by the remaining budget (a round
  commits at most `k+1`) and by the context both engines still have.
- **Recovery is reset plus sequential replay, on purpose.** Checkpoints are S2, and a chunked
  replay would produce a state Stage 2 measured to differ from the serial path - so S0 is
  explicitly not a speedup, and the recovery is the slow one that cannot be wrong for a reason
  unrelated to the protocol.
- Admission is **one token space**, which the runtime checks as more than a vocabulary size: the
  same text must encode to the same ids under both tokenizers, and both contexts must hold the
  prompt. The CLI refuses a window without `--draft-model-dir`, above temperature 0, with
  `--stream`, or outside 1..16, all before any model is loaded.
- **The scripted engine now owns two handles with observable consumed histories**, and its script
  gained a prefix-indexed mode: the argmax after consuming a prefix is `script[len(prefix)]`.
  Without that mode reset-and-replay is not faithful (the answer would depend on how many calls
  had happened rather than on what was consumed), and the consumed history is what the retention
  claim is asserted against. The pre-existing fixtures' call-indexed semantics is untouched - a
  `NULL` handle still means the default one - and all of them still pass.
- Gates: `cabal test infer-generation-tests` is **71 examples, 0 failures** (63 pre-existing, 9 new
  round-arithmetic cases, 8 two-handle engine cases), run locally and on the pod; and
  `tests/test_speculative.py --exe "$(cabal list-bin ...)"` is **18 checks** over the CLI refusals
  on the pod. The plan's list is covered: k = 1, full acceptance, rejection at every position,
  repeated rejection, lowest-ID ties, first/accepted/correction/bonus EOS, budgets 0/1 and the
  window boundaries, context exhaustion, a differing vocabulary, and target-only versus
  speculative output.
- **Not done (S1-S3)**: bounded all-position verification with append-cache rollback (the
  prototype verifies sequentially, so the target consumes rejected candidates and the round pays a
  full replay), hybrid state checkpoints and restore, the cross-case and continuation admission
  gates, and every acceptance/throughput measurement. The device half of
  `tests/test_speculative.py` (`--target-dir`, comparing a real run's speculative output against
  the target-only output) is written and needs a free GPU.

**The dense FFN's decode path now reads the packed INT4 operands** (plan Q2's routing and its
model-quality gate; verified 2026-09-26).

- `engine_load_quantized_ffn(engine, manifest_dir)` reads a converted sidecar through the
  validator above and installs the operands **beside** the BF16 weights (both forms resident),
  because the measurement put the two paths in different regimes: `forward_mlp` reads INT4 when
  `tokens == 1` and BF16 otherwise, so **prefill is bitwise unchanged** and decode is where the
  format pays. Dequantizing for batched M is exactly what the Q2 kernel was written to avoid, and
  the tiled kernel that would avoid both is inadmissible at these shapes.
- The load is **all-or-nothing across dense layers** (a layer with a pair but no `mlpDown` would
  silently run mixed precision), refuses tensor parallelism (the converter quantizes whole
  tensors; a rank's shard is not what the sidecar describes), and checks each entry against *this*
  layer's own extents - `role_extent` for the gate/up/down roles - and each artifact's SHA-256
  against the bytes on disk before anything is uploaded.
- **A training store and the INT4 operands exclude each other**, in both directions and with the
  reason named: a publication rewrites the BF16 compute weights and would leave the packed
  operands describing the old ones, so a training step on such an engine would compute gradients
  against weights the optimizer never updates. Both refusals are gated.
- **The identity discipline holds end to end**: a packed operand is a numerical-policy change and
  nothing else, so `numerical_policy_id` moves while `semantic_id` and `deployment_id` do not.
  The manifest gained a `weight_quantization` field (`none` |
  `int4_symmetric_group_bf16_scale_decode_only`), and `manifest_test.c` now pins that only the
  numerical identity moves.
- `ctest test_quantized_ffn` is the gate. On the synthetic dense fixture: prefill logits
  **bitwise equal**, worst decode rms **9.08e-3** over 8 steps, top-1 7/8 with the single flip
  explained (bf16 top-1 margin 3.2e-3 < the step's 2.7e-2 max_abs difference - a tie-break, not a
  regression), held-out NLL delta **-0.0008 nats**. The gate prints the margin and the
  perturbation side by side, so a flip that the perturbation cannot explain fails.
- Refusals through the same entry point: a corrupted **pair payload** (the artifact the engine
  actually reads - the first version of the test corrupted a *member's* file, which nothing
  loads, and the gate caught that by failing) and a tampered format block are both refused with
  the reason.
- **Measured on the deployment target** (Qwen3.8-27B, 64 dense layers, hidden 5120, intermediate
  17408, all four multiples of the group; 2× A40): the converter produced 192 role instances and
  64 F1 pairs in **1705 s** at a per-element block error of max_abs 0.0835 / rms 1.37e-3, and the
  gate then reports prefill logits **bitwise unchanged**, worst decode rms **0.34605** over 8
  steps, top-1 **7/8 with the one flip a tie-break and none unexplained**, and the held-out NLL
  **-0.05826 nats**. The cost side the plan's Q gates ask for comes from the same run: the sidecar
  is 14025.4 MiB on disk and **8415.0 MiB resident** (the F1 pair duplicates its members on disk;
  only the pair plus every `mlpDown` is loaded), the load takes **45.32 s**, TTFT is **unchanged**
  (106.068 ms BF16 vs 106.042 ms INT4 - prefill still reads BF16), and decode goes **99.637 ms ->
  63.264 ms, 1.57x faster** (10.0 -> 15.7 tok/s). That is the format's advantage as a measurement
  on the real model, and it is the decode-only specialization the kernel measurement predicted.
- **The same gate on a second real model** (Qwen3-4B, 36 dense full-attention layers): prefill
  bitwise unchanged, worst decode rms **1.00379** with top-1 **6/8 and both flips tie-breaks, none
  unexplained**, held-out NLL **+0.28559 nats** inside the same documented budget, 1322.6 MiB
  resident from a 2204.5 MiB sidecar, 6.04 s load, TTFT unchanged at 17.27 ms and decode **1.61x**
  (17.09 -> 10.60 ms). The larger per-step error is the smaller model's coarser weights
  (block rms 2.86e-3 against the 27B's 1.37e-3) accumulating over its 36 layers - which is why the
  budget is per model, as the plan requires, and why the gate reports the margin beside every flip
  instead of asking for token identity with BF16.

**The `weights.manifest.json` reader, as a validator** (plan Q1's sidecar, consumed by Q2;
verified 2026-09-26).

- `csrc/include/quant_manifest.h` + `csrc/quant_manifest.c`: a CUDA-free parser for the sidecar
  the converter writes. It is a *validator*, not a decoder: `quant_manifest_parse` refuses a
  version it does not speak, a format block that is not the frozen format (group 128, q ∈ [-7, 7],
  zero-point 0, `-8` reserved, `u8`/`bf16`), a group axis that is not K, an entry whose recorded
  packed bytes / scale count disagree with its shape, a pair whose rows are not its members', a
  precision map that does not describe the same cells as the entries, a malformed digest, a
  duplicate key and a truncated document.
- **The format's refusals are not re-implemented.** An entry's shape goes through
  `linear_layout_init` - the same function the quantizer and the Q2 kernel gate use - so "K is not
  a whole number of groups" is the format's error rather than the reader's opinion, and a foreign
  group cannot be described two ways.
- `quant_artifact_read` ties the bytes to the digest the manifest recorded: it requires the file's
  size, the recorded dtype and the recorded SHA-256, so the payload the manifest talks about is the
  payload on disk. It returns a fresh buffer rather than a view, because the caller owns it.
- `ctest test_quant_manifest` (CPU, and locally buildable with gcc/g++) holds a hand-written
  fixture and **seventeen refusals** produced by textual surgery on it - each mutant checked to have
  actually applied, so a renamed fixture cannot turn the set into silent passes - plus a temp-file
  artifact that is read and then refused for a flipped byte, a wrong size, a wrong dtype and a
  missing file.
- `test_quantization_converter` now runs the binary over the sidecar it just wrote
  (`--reader $<TARGET_FILE:test_quant_manifest>`), which ties the schema to its only writer and its
  reader: that run parses 6 entries and 2 pairs from the synthetic checkpoint and re-verifies a
  16384-byte artifact against its digest. The suite is 33 tests, all passing.
- **not done**: the engine does not load these artifacts yet. Loading them into owned device
  buffers and routing the decode (M = 1) path to the INT4 GEMV is the next piece, after which the
  model-quality gates (logits RMS, top-1 agreement, held-out NLL against the BF16 baseline) have
  something to measure.

**The flaky tied publication was a real ordering bug, not a tolerance** (found and fixed
2026-09-26, while checking the Q1/Q2 increment). Stage 3's `test_train_forward` had been
failing roughly half of its runs at the post-update comparison, and it was recorded as "the
engine's training forward is not bitwise reproducible". That was wrong in an important way,
and the fix is one call.

- **The evidence.** Repeated runs were fingerprinted: the torch reference and the host inputs
  were bitwise identical across processes while the engine's post-update logits took several
  different values (`5ba7e0bb...`, `3701bbaa...`, ... against the correct `ce660ab5...` as an
  independently computed `bf16(edited)` hash). Two forwards with the same weights agreed
  bitwise *within* a process, so the forward was not the variable. `engine_train_export_state`
  showed the master already equal to the perturbation, and a dump at the end of
  `engine_train_publish` showed the BF16 readers holding the right bytes - yet a dump inside
  the cast loop, immediately after the copy, caught one reader holding **neither** the old nor
  the new weight. An identical re-publish always landed correct.
- **The cause.** `engine_train_write_master` used a blocking `cudaMemcpy` from a pageable host
  array. That call only guarantees the bytes reached the driver's staging buffer; the final DMA
  is ordered in the *calling thread's* stream, while every kernel that reads the master runs on
  a context stream. The publication could therefore cast a half-written master. A no-op write
  hid it completely, because its master already equalled the loaded weight, so even a partial
  read produced the right bytes - which is why only the perturbed path flaked.
  `engine_train_import_state` had always used the stream-ordered form (`cudaMemcpyAsync` on the
  engine's stream plus `cudaStreamSynchronize`); this writer was the one that did not.
- **The fix** makes the upload run on the master's own context stream and synchronize it. Eight
  consecutive runs now produce one value (`5ba7e0bb...`, rms 0.00399, top-1 12/12) where before
  there were five outcomes and about half failed. No tolerance was touched: the gate had been
  working, and the bug was what it was catching. The same class of mistake is invisible in the
  successful path, so it is worth stating plainly - **a blocking H2D copy does not order the
  destination DMA against kernels on another stream**.

**F0: the costed baseline for the inference-optimization track** (plan F0, verified
2026-09-26).
The plan's fusion milestones all start by asking where the time goes, so F0 delivers the
instrument and the numbers rather than a guess:

- `csrc/include/profile.h` + `csrc/profile.cu` are **opt-in per-region CUDA timing**: a
  `PROFILE_SCOPE(name, stream)` guard records one event pair per region invocation and never
  synchronizes while recording; `profile_report` is the single measurement boundary. It is
  **off by default**, so 28/28 ctest and the goldens still run the untouched path, and the
  switch is a library call rather than a build flag so one binary serves both a benchmark and
  a correctness run. 31 scopes sit at the region call sites (the dispatcher's mixer/ffn
  choices, the dense MLP's five steps, the attention layer's ten, the GDN layer's ten, the
  final norm and LM head, and the residual adds).
- `tests/benchmark_inference.py` is the entry point: repeated warm runs with min/median/max
  and a standard deviation for prefill M=2/64/128 and single-token decode, the per-region
  table, and the provenance a baseline needs (descriptor, device list, GPU model/clocks/
  temperature, manifest identities). It is run by hand, not registered in CTest - a timing
  threshold in the suite would be a flaky gate.
- **the deployment target's baseline** (Qwen3.8-27B, 2× A40, sm_86, pipelined): prefill
  M=2 105.86 ms, M=64 122.82 ms, M=128 146.22 ms, decode M=1 **95.53 ms = 10.47 tok/s**,
  dispersions ≤ 0.044 ms over 5 (or 120) samples.
- **the per-region table decides F1's question**: a decode step is the dense MLP
  (`ffn.dense` 62.3 of 95.5 ms), and inside it the three GEMMs are 20.5 + 20.5 + 19.8 ms
  while the SiLU-multiply is 0.51 ms. **Gate + up alone are 40.9 ms, 43% of a decode step**,
  which is the pair F1 proposes to merge. The GDN in-projections are the next lever (QKV 9.3
  + Z 5.8 ms of 48 layers) and the LM head's single row costs 4.46 ms through a 248320-entry
  vocabulary. Conversely F2's norms and residual adds sum to under 2 ms at M=1, and
  `mlp.silu_mul` is negligible there but 20.7 ms (22% of the MLP) at prefill M=128.
- **the measurement's own cost is reported**: recording adds +4.91 ms to the instrumented
  calls (151.13 vs 146.22 ms), so `overhead_ms` sits beside every per-region table and those
  sums are not a decomposition of the wall clock. A two-device bug was found and fixed here
  too: events belong to a device and `cudaEventElapsedTime` only reads events of the current
  one, so the pool is now created per device - and an earlier zero-initialised slot map made
  a NULL handle look valid on device 0, which poisoned the stream and aborted the 2-device
  forward at the third scope.

**Q1+Q2: the F1-compatible pair artifact** (plan Q2's "concatenate packed rows and scale rows
consistently", verified 2026-09-26).

- The Q2 routing names the F1 layout, and F1's `forward_mlp` takes **one** `[2I, H]` weight with
  the row-interleaved `[T, 2I]` activation, so the converter now emits a **pair artifact** per
  dense layer with both members quantized: `mlpGateUp_layer{L}.packed` is the gate's packed rows
  followed by the up's, and `.scales` is the same row order - the engine reads one operand and
  does not need to know the pair's internal split.
- The manifest gains a `pairs` section (`name`, `layer`, `members`, `rule`, `logical_shape`, the
  artifact hashes and the group), and `--verify` requires the pair to equal its members' bytes in
  order with `logical_shape == [sum(rows), k]`. The gate re-derives that from the **entries
  located by role**, not from the pair's own member list, so a pair consistent with itself but
  not with the quantized members fails.
- On the synthetic checkpoint this is 2 pairs (`[512, 128]`, 32768 packed bytes and 512 scales
  each); the verification prints "2 F1 pair(s) match their members' rows", and the three negative
  cases (corrupted payload, foreign group width, missing artifact) still fail as required.
- **not done**: the engine-side manifest reader that loads these artifacts, the smoke dispatch
  that selects the INT4 GEMV for the decode (M=1) path, and the model-quality gates (logits RMS,
  top-1 agreement, held-out NLL against the BF16 baseline).

**Q2: the weight-only INT4 GEMM, verified numerically and measured (plan Q2, first half,
2026-09-26).**

- `csrc/kernels/gemm_quant.cu`: `C = A * B^T` with a BF16 activation, the Q0-packed weight and
  a BF16 output accumulated in FP32. The weights are **unpacked and scaled inside the thread**
  (one 32-bit load carries eight codes, the group's scale hoisted out of the inner loop), which
  is the plan's alternative to "full-weight dequantization into a BF16 temporary on every
  forward". Shapes the format cannot hold are refused by the host wrapper at creation time.
- `ctest test_quantization_kernel` makes the plan's **two separate comparisons**: the kernel
  against an independently dequantized-weight reference (the payload dequantized in numpy from
  the format's definition, multiplied in float64) and the quantizer against the original
  weight. Only the first is asserted, tightly: across the FFN shapes and M = 1/2/8/64,
  `max_rel <= 3e-3` and `rms_rel <= 5e-4` - inside the BF16 output's own rounding. The second is
  reported (max_abs 0.021-0.027, rms 0.0074 of |w| max ~0.38).
- **the measurement then decided a specialization, and the specialisation is the one the
  first kernel lacked.** On one A40 at N=4096, K=5120, against *the same values in BF16*
  measured through torch:

  | path | median | packed-weight bandwidth | vs BF16 |
  |---|---|---|---|
  | M=1, warp-per-row GEMV (coalesced) | **52.1 us** | 207.6 GB/s | **2.25x faster** (117.2 us) |
  | M=1, first one-thread-per-output kernel | 366.7 us | 29.5 GB/s | 3.1x slower |
  | M=64, first kernel (batched) | 5090.9 us | 2.1 GB/s | **44x slower** (115.0 us) |

  The first kernel's mapping was the fault: one thread per output element walks a whole weight
  row per thread, so a warp's loads land on different rows and are uncoalesced. The rewritten
  **warp-per-row GEMV** - lane t loads the 32-bit word at index t (consecutive lanes, consecutive
  words) with the activation staged in shared memory once per block - is 7x faster than that
  and **2.25x faster than BF16 on the same values**, which turns the format's advantage into a
  measurement instead of a claim. It is not the theoretical 4x: the kernel reaches 208 GB/s of
  the device's ~700, so latency hiding is still the limit.
- **the batched path was then built and measured, and the verdict is decode-only.** The batched
  kernel is now a shared-memory-tiled GEMM (64x64 block tile, the K slice exactly one scale
  group so each slice's sums are scaled once, a 4x4 micro-tile per thread, the activation tile
  padded off one bank). Two bugs the gate caught on the way: applying only the *last* slice's
  scale (every group has its own), and an early `return` for threads with no output rows, which
  left the block's `__syncthreads()` with a different set of arrivals and silently corrupted the
  M=2 and M=8 cases. Both fixed (the second became a work guard that still reaches the
  barriers); all 13 checks pass on both paths. But it is 8-30x slower than BF16 (M=2 924 us, M=16
  997 us, M=128 3316 us against BF16's flat ~115 us), because at N=4096, K=5120 BF16 is
  *weight-bandwidth-bound* at every M while the SIMT int4 kernel is *instruction-bound* on the
  nibble unpack (measured 1.2-2.7 TFLOP/s against the tensor cores' 23-47). Only M=1 wins
  (**54.3 vs 115.3 us, 2.12x**). Putting those products on the int tensor cores needs int8
  activations, which the plan scopes out, so **weight-only INT4 with BF16 activations is a
  decode-time specialization**.
- two fixture bugs the gate caught before the kernel was trusted: the test passed a float32
  activation where the kernel reads BF16, and wrote the BF16 output into a float32 buffer
  (which produced denormals whose signs tracked the reference - the giveaway). Both were in the
  test, not the kernel, and both are why the "independent reference" check exists.

**Q1: the converter and the weights.manifest.json sidecar** (plan Q1, verified 2026-09-26).

- `scripts/quantize_weights.py` converts the admitted roles - the dense FFN gate/up/down, Q's
  initial scope - into a **separate output directory** and leaves the BF16 checkpoint
  byte-identical (it hashes every source file before and after and refuses to continue if they
  changed). It writes a versioned `weights.manifest.json` with exactly the fields Q1 lists:
  source directory with per-file size and SHA-256, converter and config version, the
  per-layer/role mapping, the logical `[N, K]`, the packed layout/shape/dtype and byte count,
  the group axis and size, the scale tensor's shape/dtype/count, the zero-point convention,
  per-artifact hashes, and a precision map with one cell for every `(role, layer)` cell of the
  descriptor (28 cells for the 2-layer fixture, 6 of them int4). It also records the measured
  elementwise error so a quality gate has the number rather than a claim.
- **the converter does not re-implement the quantizer**: each tensor goes to the Q0 reference
  (`test_quantization_format --quantize`), so an artifact is produced by the same code Q2 will
  admit a kernel against and the two cannot drift.
- `--verify` **re-derives rather than re-reads**: it re-hashes every artifact, re-checks the
  extents against the format, requires the precision map to cover every role of every layer
  exactly once and to agree with the entries, and re-quantizes a sample from the source.
- `ctest test_quantization_converter` drives it over the synthetic dense checkpoint (6 role
  instances, block error max_abs 0.0062 / rms 0.0024 over 196608 elements) and **requires the
  verification to fail** on a corrupted payload, a manifest claiming a foreign group width, and
  a missing artifact.
- **a trap this stage found and closed**: both this gate and Stage 5's `test_sft` skip
  themselves when the node-local synthetic checkpoint is absent, and **CTest reports a skip as
  `Passed`**. A pod that moved nodes had lost
  `/var/pony/cache/bohaotu-haskell/synth-qwen3-dense`, so a "28/28" suite was 27 ran + 1
  skipped, and this new gate began life skipped too. Regenerating the checkpoint
  (`python3 tests/synth/make_qwen3_dense.py --out-dir ... --layers 2 --seed 0`) makes both run:
  Stage 5's passes in 12.5 s, this one in 0.6 s. A green count that includes skips is the
  failure mode the plan names when it says an absent prerequisite is "skipped/unverified, not a
  passing gate".
- **not done**: Q2 (no quantized GEMM consumes the artifact, so the reference has no kernel to
  admit yet), the plan's sm_86 kernel feasibility check, and the model-quality gates (logits
  RMS, top-1 agreement, held-out NLL) that compare a quantized model against the BF16 baseline.

**Q0: the weight-only INT4 format and its independent reference** (plan Q0, verified
2026-09-26).
F0's baseline decided the order: a decode step on the deployment target is 95.5 ms and the
elementwise regions F2 would fuse are ~1.5 ms of it, while the projections F3 would merge are
weight-traffic-bound at M = 1 - so the milestone that addresses the bottleneck is quantization,
not more fusion.

- `csrc/include/linear_weight.h` + `csrc/linear_weight.cpp` are the frozen INT4 format as
  CUDA-free arithmetic: `s = BF16(max|group|/7)` (with `s = 1` for an all-zero group),
  round-to-nearest-even `q = round(w/s)` clipped to [-7, 7], zero-point 0, two
  two's-complement nibbles per U8 byte with the lower K index in the low nibble, `-8` reserved
  invalid, and a **reference dequantizer a kernel is admitted against** (Q2) rather than
  alongside. The layout refuses K that is not a whole number of 128-wide groups, N that is not
  a whole number of the 8-row tile, and any other group width, instead of padding silently.
- **the packing is asserted by hand**: a round trip cannot catch a self-consistent nibble swap,
  so the gate checks the byte pattern for a row whose maximum is exactly 7 (scale 1, arithmetic
  out of the way): 0x10, 0x9F, 0x37, 0x2D, plus every index through `linear_packed_code`.
- **the refusals are behaviours**: half-built extents, NaN/Inf weights, a group whose
  `max|w|/7` rounds to zero in BF16, the reserved code in a payload (the *reader* reports a
  malformed artifact), and clipping at the signed extrema so the quantizer can never emit -8.
- **quantization is a fixed point of dequantization**: requantizing a dequantized weight
  reproduces the payload and the scales exactly.
- **a second implementation re-derives the fixture**: `tests/test_quantization_format.py`
  re-implements the format from the definition in numpy (its own BF16 rounding, scale, code,
  clipping, packing) against what the C binary emits for a deterministic [8, 256] weight and
  requires an exact match on payload, scales and dequantization; the fixture's block error is
  max_abs 0.0186 / rms 0.0109 against values of magnitude ~0.25.
- gates: `ctest test_quantization_format` and `ctest test_quantization_format_python` (both
  CPU; they build and pass locally with gcc/g++ as well). **Not done**: the plan's sm_86 kernel
  feasibility check, which Q0 says the wire format waits on - so the format is implemented and
  gated but not yet exercised by a kernel - and Q1/Q2 in full (no converter, no
  `weights.manifest.json` reader, no quantized GEMM).

**F1: the dense gate/up projections are one GEMM** (plan F1, verified 2026-09-26).
F0's table put the gate/up pair at 40.9 ms of the 27B's 95.5 ms decode step; F1 fuses it:

- the loader allocates **one packed [2I, H] buffer** per dense MLP and loads the gate and up
  roles into its two halves (`role_extent` + `load_role_into` in `csrc/engine.cu`), so there
  is never a second copy of these weights. Both roles must shard on the same axis and that is
  compared, with the load refused if the extents disagree.
- `forward_mlp` issues **one GEMM with N = 2I** into a row-interleaved `[T, 2I]` output and
  `kernel_silu_mul_packed` consumes that layout; the workspace is unchanged because
  T*(H + 2I + I) is the T*(H + 3I) the pool is sized for.
- **the fusion is admitted on measured evidence, and one number runs against the
  expectation.** Against the unfused build *on the same fixture*, the short-prompt logits are
  **bitwise identical** and the per-step rms against the independent torch reference is
  identical to 16 digits (0.1305636763572693 … 0.06327719986438751), with every token
  matching - i.e. where cuBLAS keeps the same reduction, the fusion changes nothing. Where
  N = 2I makes it choose differently (the long/chunked fixture) the logits differ by
  rel_rms 0.8-2.7% (max_abs 0.33) with 16/16 top-1, and the engine's chunk-boundary
  self-consistency gate moves 0.108/0.102 → 0.069/0.065 with top-1 374/374. That is recorded
  as a **declared numerical-policy change**, not as a bitwise refactor.
- **the policy records it**: `numerical_policy_id` moves (05fa3b15… → 61bbd7f3…) while
  `semantic_id` and `deployment_id` do not, which is the plan's fusion row exactly.
- **the win is not where F0's table suggested.** Qwen3-4B on one A40: prefill M=64
  23.53 → 20.32 ms, M=128 28.36 → **22.79 ms** (−19.6%), decode M=1 17.30 → 16.89 ms
  (59.2 tok/s), no prefill regression. Per region at M=128 the fused gate/up GEMM is 8.29 ms
  against the separate GEMMs' 3.99 + 3.95 = 7.94 — slightly *slower* — while
  `mlp.silu_mul` falls from **6.55 ms to 0.45 ms** because the old FlashInfer
  `act_and_mul` dispatch was a single block. At M=1 the GEMM is the faster half (6.63 vs
  7.07) and the step gains 2.4%.
- ctest is 28/28 with the fused path, and the fixtures from the previous commit pin both
  activation contracts (bitwise against an independent reference at T = 1/2/3/4).

**The temperature-sampling migration** (plan T0-T4, verified 2026-09-26).
The plan's generation track is independent of the trainer and was the last unimplemented
piece of the document; the CLI now samples from `softmax(logits/T)` by default:

- `src/Infer/Sampling.hs` is the pure selector and the request RNG. The arithmetic is the
  plan's: logits widen to `Double` before the subtraction and the division, `a_i =
  (z_i - m)/tau`, `w_i = exp(a_i)`, `Z` a left fold in token order, `ell_sampler = a_i -
  log Z` and `ell_model` only when the record is requested; the inverse CDF takes the first
  *positive-weight* token whose ordered prefix exceeds `u*Z`, so a zero-mass leading token
  cannot be taken at `u = 0`. The four greedy `argmax` call sites are gone: one
  `stepToken` serves both generation entry points, at temperature 0 it is the lowest-id
  argmax with **no draw consumed**, and above 0 it is one draw per selected token.
- **the RNG is splitmix64 in-module rather than a pinned dependency.** What the plan's
  "version-pinned dependency" protects is the algorithm's behaviour across builds, so the
  module pins the published gamma and mix constants and `tests/SamplingSpec.hs` freezes the
  seed-to-word vectors (seed 0's first word is the published `0xe220a8397b1dcdaf`) - which
  is strictly more stable than a version bound and keeps a new Hackage package out of the
  build. The API is Word64-in/Word64-out and the mapping is the plan's `(x >> 11) * 2^-53`.
- `src/Infer/Config.hs` owns the configuration and its refusals: a negative nonzero
  temperature even when its magnitude would underflow, a nonzero literal that underflows to
  zero, an overflow to infinity, literal `-0` canonicalised to greedy, and a seed outside
  `[0, 2^64-1]`. `Main` validates **before** any model or tokenizer allocation, resolves an
  omitted seed once from `/dev/urandom` (a missing source is an error, not a fixed seed) and
  reports `temperature`, `seed` and the sampler version on **stderr**, so stdout stays text.
- **the greedy regression fixtures now select T=0 explicitly**, as the plan requires: the
  fifteen pre-existing generation fixtures pass unchanged with an explicit greedy config,
  and a request that omits `--temperature` samples at 1.0.
- gates: `tests/SamplingSpec.hs` (frozen RNG vectors, the CDF boundaries, exact hits and
  zero-mass bins, shift invariance, overflow/underflow, the draw-count contract, same-seed
  stream/non-stream parity, a shorter request as a prefix) and `tests/test_sampling_cli.py`
  (32 CLI checks, every one of them early validation against a non-existent model
  directory). Plus the fixed-seed and greedy real-model smoke runs recorded below.
- **three bugs the gate caught**: the rewritten `argmax` lost the infinite-list `..` in
  `zip xs ([0 ..] :: [Int])`, so every greedy token came out 0 - the pre-existing greedy
  fixtures failed loudly, which is what they are for; a spec expectation of
  `uniformFromWord 0x0008000000000000` was written as `2^-13`'s reciprocal; and
  `ell_model` was computed from `a_i = (z_i - m)/tau`, which is the raw-model offset only at
  `tau = 1`, so any selection at another temperature recorded a wrong raw-model
  log-probability - and the test that should have caught it had picked a `u` that selected
  the row's maximum token, where `z_i = m` collapses both normalisations to the same value.
  The assertion now uses a non-maximum token and requires the two log-probabilities to
  differ at `tau /= 1`.
- **one rule could not be reached, and that is recorded rather than hidden.** The plan's
  endpoint correction ("if floating multiplication rounds `r` up to `Z`, select the last
  positive-weight token") is defensive here: `Z` is at most the vocabulary size and
  `u <= 1-2^-53`, so `u*Z < Z` always and the scan always finds a bin. The reachable half
  of the rule is asserted instead - the largest admitted `u` cannot pick an arbitrary final
  entry, because a trailing zero-mass token is skipped for the last positive one.
- **not done: the performance half of T4.** Host selection time, allocations/GC, TTFT and
  tokens/s are not measured; there is no benchmark runner in this repository yet.

**Stage 6's alignment decision, Stage 7's group objectives and Stage 8's lag-zero
protocol** (plan Stages 6-8, verified 2026-09-26).
These three stages are where the plan stops specifying a trainer and starts specifying
what to do about what it measured:

- `csrc/include/alignment.h` + `csrc/alignment.c` are Stage 6's **decision layer**: one
  verdict per Stage-1 region, *derived* from the committed inventory rather than copied
  from it, so a region cannot carry two different stories. The four regions Stage 2
  measured (attention core, both GEMM outputs, GDN core) are declared exceptions carrying
  the measured bound; the four it did not establish (both norms, the conv scan and the
  gated norm) are invariant-kernel-pending with the option that would remove them named;
  the rest are exact by construction. The gate also enforces the plan's reporting rule:
  `alignment_classify` decides what a difference *is*, and scoring a policy change or a
  sampler difference against a numerical bound is refused, as is checking an
  exact-by-construction region against a tolerance. **No alignment kernel was written** —
  a declared exception is the plan's own third option, and the tracked work is named.
- `backward_group_objective` in `csrc/backward.c` is Stage 7's **GSPO and GRPO**: the
  plan's `min(s_i A_i, clip(s_i) A_i)` with the length-normalized sequence ratio and the
  population group advantage, in the plan's `mean_i` form (one term per *response*), with
  a token-level mode alongside it so the two can be compared on the same fixed groups.
  `ctest test_gspo` checks both against an independent FP64 reference and, decisively,
  against that reference's central difference: the analytic gradient matches to < 1e-4
  everywhere, including the unclipped branch's `A_i s_i / T_i` (s_i not detached).
  Sequence and token clipping statistics are reported separately, and the named cases
  (both advantage signs, both clip boundaries, masks, unequal lengths, π_θ = π_b,
  zero-variance groups, truncated groups) each have their own check.
- **the gate caught a latent bug in Stage 4's objective**: `backward_clipped_objective`
  derived its clamped value per sign (min for A>0, max for A<0), so the `r < 1-eps_low,
  A < 0` corner reported the *uncapped* product while already zeroing its gradient — an
  objective that disagreed with its own derivative, in the one corner the Stage-4 fixture
  never entered and where the implementation and its "independent" reference shared the
  same wrong formula. It now computes the plan's single `min(r*A, clip(r)*A)`, and
  `test_backward` pins the corner with a case where the value (the clamped `(1-eps)*A`,
  not `r*A`) and the gradient agree under a finite difference.
- `csrc/include/rollout_queue.h` + `csrc/rollout_queue.c` are Stage 8's **lag-zero
  protocol**, which is the plan's own first step ("build A's protocol first, with lag
  zero"): a bounded `LearnerQueue` over whole completed groups, an admission lag
  `learner_committed_version - behavior_version` enforced at dequeue, a bounded allowlist
  of live behavior versions, and a consumed-group ledger so a retry cannot count a
  response twice. The gate checks the plan's async gate 1 (a group dequeued at lag zero
  reproduces the synchronous objective and gradient **bitwise**) and gate 3 (lag 0/1/2
  injected on a fixed group: lag zero gives s_i = 1 and a zero objective, larger lag
  drifts the objective and gradient, and Stage 6's classifier names that drift a **policy
  change** rather than a numerical mismatch). **The asynchronous GPU half is not
  implemented** — snapshots, device leases, publication transfer and every throughput
  measurement are recorded as open, because they need the actor/learner resource split
  the plan defers.

**The SFT step and the synchronous training/rollout baseline** (plan Stage 5, verified
2026-09-26).
Stages 3 and 4 built the ownership objects and the region backwards; this stage chains them
into an actual step and adds the baseline the plan's RL work reads from:

- `csrc/backward_layers.cu` is the layer walk: the full-attention mixer (fused QKV with the
  output gate, per-head Q/K norms, partial RoPE, the KV write, the attention core from its
  recomputed base-2 LSE, the output gate, the output projection) and the dense MLP, with the
  residual adds and the four norms. Each sublayer **recomputes** its own fine-grained
  activations from the three boundaries the step retains, so the retained set stays at three
  values per layer and the trade is arithmetic for memory.
- the engine's step is five entry points (`engine_train_forward_retain`, `_loss`,
  `_backward`, `_apply`, `_zero_grads`) plus a state export/import pair, so a caller can
  inspect the loss, run the backward, step AdamW and publish through Stage 3's store.
- `csrc/include/train_loop.h` + `csrc/train_loop.c` are the synchronous baseline's
  contract: the phase budgets, the phase machine (a rollout *borrows* the store, so the
  store refuses an update while it reads), the version-bound selection records, the
  sequence-level ratio, and a host FP64 temperature-1 sampler. All CUDA-free and gated on
  the CPU.
- **the step is checked against an independent training run**, not just against itself: the
  first step's loss matches `transformers` + `torch.autograd` to 6.3e-05 relative, the tied
  parameter's gradient direction (read from one large-lr AdamW step, where the move's sign
  is the gradient's) has cosine 0.9990 against torch's, and the fixture overfits on both
  sides (7.03 → 0.61 and 7.03 → 0.57). The forward is bitwise reproducible; the backward's
  atomic weight-gradient accumulation is measured (5.1e-03 after twelve steps, 1.4e-02 on
  the resumed tail) and reported, which is the plan's "determinism tested separately from
  closeness".
- **four bugs came out of that comparison**, and each was silent: the log-probability and
  the loss's slope shared one pool slot, so the backward consumed the log-probability
  (~-7) as the loss's upstream slope and every gradient came out inverted and 140× too
  large (the diagnostic that found it: cosine -0.998 on the tied parameter); the layer walk
  never seeded the residual gradient, so it accumulated the sublayer terms into whatever the
  buffer held (a *growing* loss); the publication closes the update window itself, so a
  second close was an error; and the pool-sizing formula under-counted the pre-norm
  snapshots (a named refusal, which is what the pools are for).
- **the rollout half's rules are behaviours, and the engine drives them live**:
  `engine_rollout_sample` generates a completion under the loaded model inside a *borrowing*
  `TrainContext`, so an engine-level optimizer step is refused while it reads and the record
  is stamped with the version that context borrowed (the loop then accepts it and refuses a
  copy stamped one version ahead). The distribution is checked where it matters — 20000
  draws agree with the softmax of the very logits they are drawn from, by entropy
  (6.90837 ± 0.00162 against 6.90495, +2.11 sigma) and by histogram (all 415 counted bins
  within 4.5 sigma, worst 2.94) — and the record's denominator is bitwise unchanged by a
  publication that makes it unreadable.
- **the two denominators are compared rather than unified**: for the same completion at the
  same version, the teacher-forced trainer's FP32 log-probability differs from the host FP64
  sampler's by 4.768e-07 (a sequence ratio of 0.99999976). That is the plan's "validate
  host-FP64 sampler versus trainer-FP32 logprob differences explicitly", and it is reported
  as the cross-path deviation it is.
- **a group is one version, one configuration and one verifier**: four completions of one
  prompt differ only in their seed, land in one `TrainGroup` whose version its first member
  pins (a fifth completion generated after a publication is refused), and carry rewards from
  a *deterministic check on the completion* — no reward model. The advantages reduce over
  those rewards (+1.000/−1.000/+1.000/−1.000), the degenerate zero-variance subset is a
  refusal the caller has to waive, and the sequence-level objective runs over the engine's
  own record: ratio exactly 1 unchanged, the gradient's sign following the advantage, and a
  +0.25 nat/row move giving 1.284025 against the records' own 1.284025.
- **offloading the optimizer state is a declaration rather than a guess**:
  `train_loop_declare_optimizer_offload` records what a caller actually shed, the phase
  boundary clears it so no phase inherits another's answer, and a negative count or a
  declaration outside a phase is refused. The engine does not move the buffers yet, which the
  plan's Stage-5 limits say.
- What is **not** wired is named in the plan's Stage-5 status and the gaps list: the GDN
  mixer's backward in the walk, the placements, a batched group driver (a group's G
  completions are collected by looping), and an optimizer-state move the engine performs
  itself.

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
  case no region could be exercised for is a policy claim, not a note;
  `test_region_inventory` now fails if any manifest row names a traversal case. (The
  traversal has since been built — Stages 3–5 — so `train_forward` is reachable in
  principle; the cell stays out until region-level train-vs-inference fixtures exist, which
  the gaps list records.)
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
  deterministic by construction, 15 unverified because a library, a cross-device
  reduction order or (Stage 4) an atomically-accumulated group sum is not established,
  and 1 `not_implemented` — the proposed sampler). Stage 4 moved two rows out of
  `not_implemented`: `masked_loss` (its masked reduction and backward landed) and
  `backward`, whose determinism is now recorded with the reason it is only partly
  pinned.
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
- **The retained-value contract is exercised by the gate, not by the trainer's step.**
  `backward_required_values` and `backward_check_retained` are the by-name contract a region
  harness runs under, and `ctest test_backward` drives the refusals directly. The SFT step
  keeps Stage 3's retention plan instead (`layerN.mixerOut` / `ffnOut` / `residual` plus the
  GDN chunk states, a different vocabulary from the table's), so nothing refuses a step whose
  plan went stale — the SFT gate would show it as a wrong loss or gradient rather than as a
  named refusal. Routing the step's plan through the table (one vocabulary for both, or an
  explicit mapping) is an open item.
- **The forward regions are not yet registered under the `train_forward` case.**
  The traversal now exists in full — Stage 3 added `engine_train_forward`, Stage 4 the
  backward, the losses and the optimizer, Stage 5 the step and the synchronous rollout — so
  the case is reachable in principle, while the Stage-1 inventory still lists the trainer
  traversal as unavailable everywhere. Registering the case means fixtures that compare the
  training forward against the inference forward at region level, and half-doing it would
  put a coverage claim in the registry that no fixture backs, so it is an open item rather
  than a partial edit. The registry's `masked_loss` row is what the trainer's arrival *did*
  change: it moved out of `not_implemented` into `unverified`.
- **The training path's gradient pairings are not bitwise, and Stage 6 declared them.**
  Stage 4 delivers the backward, the losses and the optimizer, but three pairings stop
  short of bitwise and are recorded rather than glossed: attention's backward
  recomputes P in FP32 while the forward's PV product rounds it to BF16 (dV carries
  ~1e-3 for that reason, which Stage 2's claim E predicted); the GDN core backward
  differentiates the recurrence rather than the cubin's `(I+A)^{-1}`/BF16-MMA
  decomposition; and the attention dK/dV group sums accumulate with atomics, so they
  are reported by the gate rather than required to be reproducible (dQ, which finishes
  inside its own block, and the whole GDN core backward are bitwise). Stage 6's
  `alignment.c` records these as **declared exceptions for the regions that were measured**
  (the attention core and both GEMM outputs) and as **invariant-kernel-pending for the four
  it did not establish** (both norms, the conv scan and the gated norm), so "the pairings
  are not bitwise" is a decision with a named bound rather than a silent remainder. The
  kernels that would remove a pending region were **not written**.
- **No online RL loop, and no algorithm beyond SFT.** Stage 7 delivers the GSPO and GRPO
  *objectives* and their gradient gate (`csrc/backward.c`, `ctest test_gspo`), not a
  trainer that runs them: there is no rollout→reward→update trajectory, no policy-distance
  or stability measurement, no `mean_groups` accumulation (a group objective is computed
  per group and the caller accumulates), and OPD, DAPO and PPO have no implementation. The
  plan's Stage-7 table is a plan, not this repository's state.
- **Stage 8's asynchronous half does not exist.** `csrc/rollout_queue.c` is the lag-zero
  protocol: a bounded queue, version admission and a consumed ledger. Weight snapshots,
  device cache leases, publication transfer, the actor/learner resource split, the named
  stale-data objective (`J_decoupled`) and every throughput/lag-distribution/per-device
  memory measurement are **not implemented**, which is what the plan's "start only after a
  synchronous algorithm and parameter publication protocol pass all relevant gates" defers.
- **The node-local synthetic checkpoints are a hidden prerequisite for two gates.** `test_sft`
  and `test_quantization_converter` skip themselves when
  `/var/pony/cache/bohaotu-haskell/synth-qwen3-dense` is absent, and CTest reports a skip as
  `Passed`, so a suite can be green while a gate never ran - which is what happened after the
  pod moved nodes. The fixture is regenerated with `python3 tests/synth/make_qwen3_dense.py
  --out-dir /var/pony/cache/bohaotu-haskell/synth-qwen3-dense --layers 2 --seed 0` (~1 minute),
  and both gates then run. A stronger fix would store the fixture on the PVC next to the other
  models, or make a skip a distinct CTest status (`set_tests_properties(... SKIP_RETURN_CODE)`)
  so the count cannot hide it.
- **The quantized GEMM's batched path is not admissible, and only decode is routed.** The
  warp-per-row GEMV is verified *and* measured 2.25x faster than BF16 at M = 1 (52.1 vs
  117.2 us), so that is what the engine now uses for a single-token forward; the batched path
  was built as a shared-memory-tiled GEMM and is verified correct, but measured 8-30x *slower*
  than BF16 (M=2 924 us, M=128 3316 us against BF16's flat ~115 us) because at these shapes BF16
  is weight-bandwidth-bound while a SIMT int4 kernel is instruction-bound on the nibble unpack.
  Reaching that regime needs the int tensor cores, i.e. int8 activations, which the plan scopes
  out. Prefill therefore still runs cuBLAS BF16, which is also why both weight forms are
  resident when INT4 is loaded.
- **Quantization stops at dense FFN decode.** Q0's format and reference, Q1's converter and
  sidecar and its reader, Q2's kernels and the engine-side load and dispatch are implemented,
  gated and measured. What is not done: any role other than the dense FFN's gate/up/down (MoE,
  attention and GDN roles are outside Q's scope), activation quantization or QAT (the plan scopes
  both out), and a region-inventory entry for the INT4 operand itself - the numerical-policy
  field and the kernel gate carry that identity today, while a Stage-6 alignment verdict for the
  packed path would be the inventory's job.
- **F0's memory-traffic and host-synchronization columns are not measured.** The plan's F0
  asks for four quantities per region; this repository now measures two of them (CUDA time
  and launch count). Per-region **memory traffic** and **host-synchronization counts** need a
  profiler (ncu/nsys) rather than CUDA events, so the per-region table's bandwidth story - and
  the plan's warning that "weight traffic, MoE host-offset sync and device transfers may
  dominate launch savings" - is still unwatched, which matters most for F1's claim once the
  GEMM count drops.
- **The sampling migration's performance is unmeasured.** The plan's T4 asks for host
  selection time, allocations/GC, TTFT and tokens/s; none is measured, and the vocabulary
  sized scratch the selector allocates per row (the `a_i` and `w_i` lists) is transient but
  unquantified. A benchmark runner (`benchmark_inference.py`, proposed in the plan's
  optimization track) is the natural home for it.
- **The sampler's per-row scratch is an unquantified allocation.** `prepare` builds two
  `Double` lists per selected token. The plan's memory note says not to retain a
  vocabulary-sized buffer per generated token *merely to record the logprobs*, which this
  does not do, but it does allocate them transiently and the cost is not measured.
- **The trainer forward's flakiness was found and fixed (2026-09-26).** It was recorded here as
  "the engine's training forward is not bitwise reproducible on the Qwen3-Next synth fixture",
  with the measured draw (rms 0.00399 to 0.05386 against a 0.05 tolerance, failing about a
  quarter of runs), the observation that the reference never moved, and the guess that the MoE
  path was the home. The guess was wrong: the forward *is* reproducible (two forwards with the
  same weights agree bitwise), and the variable was the **published weight**, because
  `engine_train_write_master`'s blocking `cudaMemcpy` from pageable memory does not order the
  destination DMA against kernels launched on the engine's context streams - so the publication
  could cast a half-written master. A no-op write hid it (its master already equalled the loaded
  weight), which is why only the perturbed comparison drew. Fixed by making the upload run on
  the master's context stream and synchronizing it, the form `engine_train_import_state` already
  used; eight consecutive runs now give one value. The tolerance was never retuned, and the
  bullet's own conclusion - "the nondeterminism is the thing to fix" - was the right one.
- **The SFT step's layer wiring covers the attention and dense-MLP path.** The GDN
  mixer's backward (prepare, conv and core - Stage 4's kernels) is not yet chained into
  the walk, and MoE and MLA remain outside the first trainer allowlist as Stage 1 says, so
  a hybrid or sparse model trains only its dense path today.
- **Training is wired for a single device.** A pipeline split would have to move gradients
  between devices and a tensor-parallel placement would have to reduce sharded ones; both
  are refusals rather than silent approximations, and the placement work is what Stage 6+
  would have to carry.
- **the rollout's contract is gated on the CPU, and the engine drives it live.**
  `train_loop`'s phases, budgets, records, version binding, offload declaration and sampler
  are exercised by `test_train_loop`; `engine_rollout_sample` generates a completion under a
  live model inside a borrowing `TrainContext`, so the version a record is stamped with is the
  one the engine read; and `test_sft`'s group section collects a real group, scores it with a
  deterministic verifier and reduces the advantages and the objective over the engine's own
  records. What is still the caller's: a group's G completions are collected by looping (there
  is no group entry point), the reward is the caller's verifier, and the optimizer state is
  *declared* shed rather than moved by the engine. The sampler itself is complete and
  reusable, which is what "reuse the temperature-sampling migration" asks for.
- **The rollout's two denominators are compared, not unified.** The host FP64 sampler and
  the trainer's FP32 log-probability differ by 4.8e-07 on the tiny fixture (a sequence ratio
  of 0.99999976), which is the cross-case deviation the plan asks be reported separately.
  On a longer sequence the generation path and the teacher-forced path would diverge more
  (the first reads a KV cache, the second a causal mask over the chunk), so a 27B-scale
  measurement of the same gap is owed before an RL objective mixes the two.
- **The optimizer state is declared offloaded, not moved.** `train_loop` owns no buffers, so
  the declaration is the caller's act and the engine keeps the master and moment slots
  allocated where they are: a rollout that needs the memory back has to move the state
  itself, and what the stage guarantees is that the phase boundary reports the answer instead
  of assuming it.
- **The rollout generates serially, one sequence per engine call.** The plan allows serial
  generation for a group's correctness, and the engine has no batched group driver.
- **The phase budgets are an estimate from the descriptor's shapes**, not a measured
  allocation watermark: they are what a phase choice is made from, and proving a phase fits
  is a measurement the engine does not take yet.
- **The GDN core backward's reduction is correct but not blocked.** Each coordinate
  sums over the key range rather than tiling the way the forward's kernels do, so its
  constant factor is the next thing to fix when training throughput (not correctness)
  becomes the constraint. It is bitwise reproducible, which the gate checks.
- **The FP32 masters are per-parameter, not sharded.** A store with training state for
  a 27B checkpoint would need ~54 GB of masters plus gradients and two optimizer slots
  on a 46 GB device, so the gate attaches training state for the synthetic model and
  the 27B path is bookkeeping-only. Stage 4 added the optimizer's FP32 master/m/v
  triple, which does not change that shape, so sharding or offloading the optimizer
  state is a Stage-5+ problem and is stated here rather than discovered later.
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
  no region, and both gates refuse a region that advertises them. Stage 4 makes that a
  deliberate distinction rather than a stopgap: the backward *exists* now, but it is a
  traversal of its own, so its coverage lives in the Stage-4 backward inventory
  (`backward_region_info`, which the same gates check against the Stage-1 inventory)
  instead of a per-region forward case. `train_forward` and `eval_no_autograd` still
  have no API, and `recompute` is an attention-backward option rather than a case.
- **`weights.content_sha256` is null** unless a caller chooses to hash 50 GiB of
  tensor data; the parameter-manifest digest over the tensor index is what strict
  admission compares.
- **The backward's registry row is `unverified`, not `deterministic`.** Now that the
  backward exists (Stage 4) the registry records *why* it is only partly pinned: the
  elementwise, norm, RoPE and loss backwards reduce in a fixed order, while the scatter
  and group-sum ones accumulate with atomics. The gate measures the split rather than
  promoting the whole row.
- **Expert-parallel equivalence re-run is pending.** The EP-vs-layer-split check
  after the FP32 merge landed was stopped before it finished. The gate is
  unchanged (`test_tp.py --ep 2`: identical greedy tokens, logit RMS ≤ 0.05).
- **sm_90a has no runtime gate** — compile- and artifact-verified only.
- **Long context is not supported.** The MLA attention kernel's shared-memory
  budget caps the cached sequence length (a 16K context does not fit) and both
  `engine_create` and the kernel entry reject anything longer.
- **No top-k/top-p, batching or a GPU sampler.** Temperature sampling landed
  (`src/Infer/Sampling.hs`, milestones T0-T4: binary64 softmax/CDF with a request-owned
  splitmix64 RNG, the default is T=1 and greedy is the explicit T=0 mode), but there is no
  truncation, no batched or GPU-side sampler, and the migration's performance is
  unmeasured - see the temperature-sampling entry under Recently completed.
- **Performance is not optimized** (correctness-first): decode is a single-token
  full forward, and the logits are computed for the last position only.

## Documents

| File | Role |
|---|---|
| [README.md](../README.md) | Build, test entry points, usage, phase status, model weights |
| [design.md](design.md) | Architecture rationale, per-family layout differences, testing strategy |
| [manifest-contract.md](manifest-contract.md) | The execution manifest: canonical encoding, field ownership and projections, the region determinism registry, and the comparison modes |
| [plan-numeric-contract.md](plan-numeric-contract.md) | **Partly implemented** — Stages 0-8 as far as they reach without the asynchronous GPU half, plus the temperature-sampling migration (T0-T4); the bounded-staleness async GPU scheduler, OPD/DAPO/PPO and the inference-optimization track (fusion, W4A16, speculative decoding) are proposals |
| [reference-output.json](reference-output.json) | Transformers reference tokens for the 27B debugging prompt |
