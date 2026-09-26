# haskell-infer-demo

A demo LLM inference framework written in **Haskell** with a C/CUDA backend,
targeting **Qwen3.8-27B** (hybrid Full-Attention + GatedDeltaNet architecture).

## Highlights

- **Haskell orchestration**: model definition, GPU partitioning and generation loop;
  a C ABI connects to the native weight loader and GPU operators.
- **Multi-GPU placement, three policies**: layer-wise partitioning (each device owns a
  contiguous block of layers), replicated tensor parallel (`--tp N`: every rank runs
  the whole model with its weight shards) and expert parallel (`--ep N`: whole MoE
  experts split across ranks, partials merged in FP32). Tested on 2× A40 46 GB.
- **Hybrid architecture support**: 16 full-attention layers (GQA, partial RoPE,
  output gate) + 48 GatedDeltaNet layers (causal conv1d, gated delta rule,
  gated RMSNorm).
- **Model families**: the descriptor + layer-kind design covers dense hybrid
  models (Qwen3.8-27B, verified), dense attention with or without sparse MoE
  (Qwen3-4B and Qwen3-30B-A3B, both verified against independent PyTorch
  references; Mixtral shares their adapter and snapshot layout) and multi-head
  latent attention (DeepSeek-V2-Lite, verified). `descriptors/` carries a
  snapshot per family.
- **Single-request temperature sampling**: categorical draws from `softmax(logits/T)`
  with a request-owned splitmix64 RNG and a reported, replayable seed (default `T=1`;
  `--temperature 0` is the explicit greedy mode), plus streaming output decoded
  incrementally, so a character whose bytes span several tokens is emitted once
  complete.
- **Three-language build**: Haskell (Cabal) + C/CUDA (CMake) + Rust (Cargo), with
  one command running every layer's test suite.

## Architecture

```
Haskell (GHC 9.6)
├── Descriptor.hs        Model descriptor: typed record + flat JSON codec
├── Descriptor/Adapter/  Family adapters (HF config.json → descriptor)
├── Placement.hs         Placement policies (layer-wise, replicated TP, EP)
├── Model.hs             Per-layer plan: (mixer, ffn, device)
├── Runtime.hs           Engine lifecycle, weight loading
├── Generation.hs        Greedy decode loop, streaming output
├── Tokenizer.hs         FFI → Rust tokenizer (capacity protocol + stream)
└── FFI/Engine.hs        FFI → C engine API
         │
         │ foreign import ccall
         ▼
C/CUDA (sm_86, CUDA 12.9)
├── model_desc.c            Strict parser for the descriptor (no family knowledge)
├── safetensors.cpp         Checkpoint parsing and bounds validation (no CUDA)
├── safetensors_loader.cu   Capacity-checked uploads to the owning device
├── layer_dispatch.cu       Per-layer norm -> mixer -> ffn sequence, kind dispatch
├── engine.cu               Multi-GPU forward, weight ownership, chunked prefill
├── collective.cu           Cross-device copies and the leader all-reduce
├── moe.cu                  Router, expert permutation, GEMMs, combine (BF16/FP32)
├── tap.cu                  Layer/sub-layer activation taps for debugging
├── triton/                 FLA-derived chunk kernels and upstream FLA decode AOT
└── kernels/
    ├── gemm.cu             cuBLAS BF16 matrix multiplication
    ├── layers.cu           Per-layer norm/mixer/ffn orchestration on-device
    ├── flashinfer_norm.cu  FlashInfer GemmaRMSNorm and partial RoPE
    ├── attention.cu        FlashInfer causal attention, KV cache and output gate
    ├── gdn_conv.cu         Native causal-conv1d adapter (width 4)
    ├── fla_gdn.cu          FLA chunked prefill and recurrent decode
    ├── gdn_norm.cu         Model-specific gated norm, casts and residual operations
    ├── mla.cu              MLA decoder and its shared-memory bound
    ├── silu.cu             FlashInfer SiLU × mul
    └── embedding.cu        Token embedding lookup
         │
         │ C ABI
         ▼
Rust (tokenizer-ffi)
└── lib.rs           HuggingFace tokenizers wrapper: length-query protocol and an
                     owned incremental decode handle
```

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| GHC | 9.6+ | via [ghcup](https://www.haskell.org/ghcup/) |
| Cabal | 3.10+ | installed with ghcup |
| CUDA Toolkit | 12.9.1 | nvcc targeting sm_86; sm_89/sm_90a SASS + compute_90 PTX built alongside |
| FlashInfer | 0.5.3 | C++ headers only; no TVM/Python runtime |
| Triton / fla-core | 3.4.0 / 0.5.2 | Build-time AOT; cubins embedded in libengine.so |
| Python / PyTorch | 3.10+ / 2.8 | Build and independent numerical tests only |
| CMake | 3.18+ | for CUDA build |
| Rust | 1.75+ | for tokenizer-ffi |
| GPU | 2× A40 46GB | BF16 model weights occupy about 50 GiB total |

## Build

The setup script installs CUDA and kernel dependencies into an isolated directory;
it does not update the driver or system CUDA. It reuses an existing Python environment
with PyTorch, Triton 3.4.0 and einops. Use local SSD storage, not a small-file-limited PVC.

```bash
python3.10 scripts/setup-kernels.py --root /path/on/local-ssd/kernel-deps
source /path/on/local-ssd/kernel-deps/env.sh
./scripts/build.sh --tests
```

The native engine is `csrc/build-libs/libengine.so`; Haskell links to this shared
library so rebuilding CUDA does not leave a stale statically linked engine.
Runtime requires neither Python nor PyTorch.

`--tests` adds every layer's suite, in order, stopping at the first failure:
`cargo test --locked --offline`, the CTest suite (CPU and GPU), then
`cabal test all --enable-tests`.

Architectures: one `libengine.so` carries SASS for `86;89;90a` plus PTX, so the
same build runs on A40 (sm_86), L20 (sm_89) and H200 (sm_90a); the PTX is the
forward-compatibility path for newer devices. Verified at runtime on sm_86
(A40, full-model tests) and sm_89 (L20, operator suite incl. the FLA cubins);
sm_90a is compile- and artifact-verified (`cuobjdump`) pending H200 hardware. The Triton AOT cubins of
the FLA kernels have no PTX equivalent, so they are compiled per architecture
(`ENGINE_TRITON_ARCHS`, default `86;89;90`) and picked at runtime from the
device's compute capability -- a device with no matching cubin fails with an
explicit error instead of a driver error. Both lists are CMake cache variables
(`CMAKE_CUDA_ARCHITECTURES`, `ENGINE_TRITON_ARCHS`), and `CUDA_ARCH` still
overrides the nvcc list in `scripts/build.sh`.

### Model descriptor

What the engine knows about an architecture travels in one flat JSON document,
derived by a family adapter from the checkpoint's `config.json`:

```bash
# Dump (and commit) the canonical descriptor for a model directory
cabal run haskell-infer-demo -- descriptor --model-dir "$MODEL_DIR" --write descriptors/qwen38-27b.json
# Inspect a descriptor and the placement it implies, without loading weights
cabal run haskell-infer-demo -- show-config --descriptor descriptors/qwen38-27b.json --gpus 0,1
```

Adding a family means adding an adapter (`src/Infer/Descriptor/Adapter/`) plus a
committed snapshot; the C engine stays family-agnostic. `--descriptor FILE` works
for `generate` too, and `--check-descriptor` makes the engine echo back the
descriptor it parsed and fails the run if the two sides disagree.

### Execution manifest

The descriptor is portable architecture; it carries no runtime facts and it mixes
placement into its text. The **execution manifest** is the separate, versioned
answer to *which numerical execution did this run observe*, as three content
identities (`semantic_id`, `numerical_policy_id`, `deployment_id`) plus the
build/runtime provenance and the immutable parameter identity a bitwise capture
comparison has to agree on. Field ownership, the canonical encoding and the
admission rules are specified in [docs/manifest-contract.md](docs/manifest-contract.md).

```bash
# Report the manifest for a model, device set and placement
cabal run haskell-infer-demo -- manifest --model-dir "$MODEL_DIR" --gpus 0,1 --check --write /tmp/m.json
# Admit or reject a comparison of two captures' manifests
cabal run haskell-infer-demo -- manifest-compare old.json new.json [--mode strict|diagnostic] [--deployment-scoped]
```

An unestablished fact (an unknown toolkit version, an unavailable cuBLAS query, an
unspecified sampling policy) is reported as such and makes a strict comparison
refuse — it is never defaulted into looking comparable.

Placement is chosen on the command line: the default is the layer-wise split over
`--gpus`, `--tp N` switches to replicated tensor parallel (every device holds the
whole model with its weight shards) and `--ep N` splits whole MoE experts across
the devices (router and shared experts stay replicated):

```bash
cabal run haskell-infer-demo -- generate --model-dir "$MODEL_DIR" --gpus 0,1 --tp 2 -p "Hello"
cabal run haskell-infer-demo -- generate --model-dir "$MODEL_DIR" --gpus 0,1 --ep 2 -p "Hello"
python tests/test_tp.py --library csrc/build-libs/libengine.so --model-dir "$MODEL_DIR" --devices 0,1
python tests/test_tp.py --library csrc/build-libs/libengine.so --model-dir "$MODEL_DIR" --devices 0,1 --ep 2
```

Full-model validation uses independently generated reference logits:

```bash
python tests/reference_model.py --model-dir "$MODEL_DIR" --output /tmp/reference.npz
python tests/test_engine.py --library csrc/build-libs/libengine.so \
  --model-dir "$MODEL_DIR" --reference /tmp/reference.npz
```

The reference uses eager PyTorch attention and the Transformers mathematical GDN
implementation; native BF16 operators need not produce bit-identical logits, and
equal highest BF16 reference logits are treated as ties.

## Tests

- `ctest --test-dir csrc/build-libs` runs eleven suites that need no GPU at all:
  `test_model_desc` (descriptor parsing, structural validation, the
  engine-capability gate for the AOT GDN layout, canonical echo and the tp-role
  coverage rule), `test_safetensors` (malformed headers, offsets, shapes and
  dtypes, higher-rank tensors, row/slice capacity arithmetic), `test_manifest`
  (SHA-256 against the FIPS vectors, canonical format determinism, and the
  identity matrix: a semantic or numerical change moves the matching id only, a
  deployment-only change moves `deployment_id` only, a provenance or weight change
  moves no identity), `test_manifest_hashes`, which re-derives every digest from
  the emitted document with `hashlib` so the emitter cannot certify itself, and
  `test_region_inventory`, which checks the plan's in-scope forward path is
  inventoried region by region, that the inventory and the manifest registry cannot
  drift apart, and that every registered case-pair verdict carries the evidence it
  needs, `test_train_loop`, which checks Stage 5's training/rollout baseline: the phase
  budgets, the phase machine (a rollout's borrow is what makes the store refuse an
  update), the version-bound selection records, the exactly-one ratio at unchanged
  parameters and the host FP64 sampler's measured frequencies; `test_backward`, which
  checks the Stage-4 differentiation contract: the
  Stage-4 region table against that same inventory in both directions, the losses and
  AdamW against an independent FP64 implementation, the checkpoint's refusals, a
  deterministic fixture that overfits to 32/32 and a resume that has to land on the
  uninterrupted run's bits; `test_alignment`, which checks Stage 6's numerical-alignment
  decision (one verdict per Stage-1 region derived from the inventory rather than restated
  beside it, a declared exception carrying a measured bound, a pending region naming its
  tracked work, and the refusals that keep a policy change or a sampler difference out of
  the numerical column); `test_gspo`, which checks Stage 7's GSPO and GRPO group objectives
  against an independent FP64 reference *and its central difference* on an unequal-length
  group, with both advantage signs, both clip boundaries, masks, zero-variance and
  truncated groups; `test_rollout_queue`, which checks Stage 8's lag-zero admission
  protocol (whole completed groups only, bounds in groups/tokens/live versions, lag
  enforced at admission, a consumed-group ledger, and lag 0/1/2 injection reported as a
  policy change); and `test_train`, which checks the trainable runtime's
  ownership objects:
  tying, frozen parameters with no training state, the borrow/update/free lifetime
  rules, publication with derived-copy refresh, the accumulation schedule, replicas and
  the teacher-forcing plan. The rest of ctest
  needs a GPU: `test_engine_resources` (repeated failing initializations leave no
  handle and no device memory; skips itself when no device is visible),
  `test_collective` (event-ordered copies and the cross-device all-reduce on 2
  GPUs — once per element type — skipped with fewer devices), `test_region_cases`
  (the Stage-1 cross-case harness: identical inputs and persistent state under each
  applicable case, adjudicated against the committed inventory, plus the
  unsupported-shape/case rejections and the region entry-point cost), the Stage-2
  invariance matrices `test_gdn_invariance` (prepare-vs-core attribution over
  lengths and splits), `test_attention_invariance` (full vs split vs single-query
  over head dims, GQA and KV lengths), `test_gemm_invariance` (one M-row call vs
  per-row and prefix splits at the real projection shapes) and `test_attention_lse`
  (the query that asks FlashInfer for the LSE a backward needs, and checks the
  output is unchanged), `test_backward_kernels` (the Stage-4 gradient gate: every
  backward against a double-precision definition or a central difference of one, the
  attention backward against the analytic reading of the device's own base-2 LSE, the
  GDN core backward with a nonzero initial state across one and three chunks, and the
  run-to-run reproducibility of both paired kernels), `test_train_forward` (the Stage-3 gate on the synthetic
  checkpoint: all-position forward against a transformers forward, teacher-forced
  log-probabilities, tied roles, a synthetic update that must refresh both readers and
  the derived FP32 copy, and the training-step lifetime; it skips itself when the
  checkpoint is absent), and the
  `test_sft` (the Stage-5 gate: the SFT step against a `transformers` training run on the
  tiny dense checkpoint - the loss, the per-parameter gradient direction and the overfit -
  plus determinism, a bitwise state round-trip and the refusals; its rollout section drives
  `engine_rollout_sample` and checks the record's fields, the seed behaviour, EOS vs the
  length limit, that 20000 draws reproduce the model's own entropy and histogram, that the
  record's version is the one the engine *read*, and the measured gap between the host FP64
  sampler's denominator and the trainer's FP32 one; its group section collects four
  completions under one version into one `TrainGroup`, scores them with a deterministic
  verifier (no reward model), reduces the advantages, refuses a zero-variance subset unless
  the caller waives it, and runs the sequence-level objective over the engine's own record.
  It skips itself when the
  checkpoint is absent, and the
  operator-level regressions `test_attention` (causal GQA, KV write, output
  gate), `test_gdn` (FLA recurrent decode, causal-conv1d, gated norm), `test_moe`
  (router, permute, expert GEMMs, combine, EP shards), `test_mla` (MLA chunking
  self-consistency and the attention block-size boundary) and `test_library_ops`
  (FLA chunk pipeline T=1..128 vs PyTorch recurrent on both GPUs, GemmaRMSNorm,
  partial RoPE; skips itself when no device is visible).
- `tests/test_engine.py` — full 27B vs independently generated reference logits
  (20/20 argmax), state reset, invalid-input/capacity checks, chunk-split
  self-consistency.
- `tests/test_longseq.py` — 433-token prompt chunk-split consistency plus
  128-token generation coherence (repetition-rate and tail-degradation checks).
- `tests/test_pp.py` — pipeline-parallel inertness (plan Stage 2 claim A): the same
  checkpoint with every layer on one device and with a two-device layer split,
  comparing logits through the capture machinery and every per-layer/mixer/ffn tap
  dump byte for byte.
- `tests/attention_backward_feasibility.py` — the attention forward/backward pair
  (claim E): establishes the LSE convention a backward has to consume, checks the
  analytic backward against `torch.autograd`, and reports what a paired library
  would cost in tolerance. Pure torch, no engine or checkpoint; the kernel half of the
  pair (`kernel_attention_lse` + `kernel_attention_backward`, with the convention
  re-established against the device's own LSE) is `ctest test_backward_kernels`.
- `tests/test_tp.py` — placement equivalence on 2 GPUs: the layer-wise split
  against replicated tensor parallel (`--tp 2`) or expert parallel (`--ep 2`,
  with `--desc` and the MoE model), requiring identical greedy tokens and a
  per-step logit RMS within `--rms-gate` (default 0.05).
- `tests/capture_logits.py` — records greedy logits for fixed prompts together
  with the execution manifest the capture was taken under, and compares two
  captures bitwise. The manifest decides the admission: identities and provenance
  must agree (`strict`), a placement-only difference needs `--deployment-scoped`
  and is reported as scoped, `--compare-mode diagnostic` reports a deliberate
  difference without calling it a pass (exit 3), and a capture without a manifest
  is legacy/unverified (exit 2) with its numeric arrays still comparable. The
  numeric gate is unchanged: every array has to be bitwise identical, non-finite
  values fail, and a manifest verdict never loosens it.
- `tests/test_manifest_compare_cli.py` — drives the real executable's
  `manifest-compare` over built manifests and checks the exit codes the contract
  fixes (0 admitted, 1 rejected, 2 legacy/unverified, 3 diagnostic-only).
- `cabal test all --enable-tests` — three suites, none needing a GPU:
  `infer-tests` (descriptor round-trip, layer plan and placement; with
  `INFER_MODEL_DIR` set it also checks the adapter still reproduces
  `descriptors/*.json`) and `infer-generation-tests`, which drives the real
  generation loop and tokenizer wrapper against a scriptable C stub of the engine
  — token budgets, first/later EOS, prefill/decode failures and the cleanup that
  follows them — plus `SamplingSpec` for the temperature selector: the frozen splitmix64
  vectors, the CDF boundaries and endpoint rules, the draw-count contract, and same-seed
  stream/non-stream parity. `tests/test_sampling_cli.py` checks the sampling options and
  their early validation against the built executable (it needs the binary, not a gate). `infer-trainer-tests` links `csrc/train.c` directly and requires the
  Haskell teacher-forcing plan and the C implementation of the same schedule to agree
  across shifts, masks, forced labels and explicit positions.
- `ctest` also runs `test_norm` (both RMSNorm variants) and `test_rope` (partial
  and full rotation), each against a CPU reference.
- `tests/benchmark_inference.py` is the costed baseline the plan's fusion work starts from:
  repeated warm runs of prefill (M=2/64/128) and single-token decode with min/median/max and
  a standard deviation, per-region CUDA time and launch count from the opt-in
  `profile_scope_*` hooks (off by default, so the gates run the untouched path), and the
  provenance a baseline needs (descriptor, devices, GPU clocks, manifest identities). It is
  run by hand — a timing threshold in the suite would be a flaky gate. Example:
  `python3 tests/benchmark_inference.py --library csrc/build-libs/libengine.so \
  --model-dir "$MODEL_DIR" --desc "$DESC" --devices 0,1 --json /tmp/f0.json`.
- `tests/test_engine.py --rms-tolerance` defaults to 0.1 and is raised per family
  where the reference's arithmetic differs (plain-norm families and MoE expert
  reduction land around 0.1-0.4); every greedy token still has to match.

## Usage

```bash
# Phase 1: FFI verification (no model weights needed)
cabal run haskell-infer-demo -- hello-gpu --device 0 --value 42

# Show model configuration (descriptor + placement; no weights needed)
cabal run haskell-infer-demo -- show-config --descriptor descriptors/qwen38-27b.json

# Report the execution manifest (identities + build/runtime provenance)
cabal run haskell-infer-demo -- manifest --model-dir /path/to/model --gpus 0,1 --check

# Generate text (requires model weights)
cabal run haskell-infer-demo -- generate \
  --model-dir /path/to/Qwen3.8-27B \
  --gpus 0,1 \
  --max-tokens 256 \
  --stream \
  -p "Explain why the sky is blue."
```

### Sampling

`generate` samples from `softmax(logits/T)` at temperature `T > 0` and decodes greedily at
`--temperature 0`. **The default is `--temperature 1.0`, not greedy** (plan
`docs/plan-numeric-contract.md`, "Temperature-sampling migration"); the resolved
temperature, seed and sampler version are reported on **stderr** before generation, so the
generated text on stdout stays clean. A seed omitted with `--temperature 1.0` is drawn once
from the OS and reported, which is what makes a run replayable after the fact:

```bash
# Greedy: reproducible by construction, no random word consumed
cabal run haskell-infer-demo -- generate --model-dir "$MODEL_DIR" --gpus 0,1 \
  -p "Hello" --max-tokens 16 --temperature 0

# Categorical sampling at T=1 with a fixed seed: the same prompt and seed replay the same tokens
cabal run haskell-infer-demo -- generate --model-dir "$MODEL_DIR" --gpus 0,1 \
  -p "Hello" --max-tokens 16 --temperature 1.0 --seed 42

# The same request with --stream selects the same tokens; an omitted seed prints one that
# replays the request when supplied explicitly
cabal run haskell-infer-demo -- generate --model-dir "$MODEL_DIR" --gpus 0,1 \
  -p "Hello" --max-tokens 16 --temperature 0.7 --stream
```

A negative, non-finite or unrepresentable temperature and a seed outside `[0, 2^64-1]` are
configuration errors reported **before** any model or tokenizer is loaded. A seed supplied
with `--temperature 0` is accepted and reported as unused, because greedy consumes no draw.
There is deliberately no top-k, top-p, repetition penalty or beam search in this migration;
`tests/SamplingSpec.hs` (run by `cabal test infer-generation-tests`) and
`tests/test_sampling_cli.py` are its gates.

## Model Weights

Download Qwen3.8-27B from HuggingFace:

```bash
huggingface-cli download Qwen/Qwen3.8-27B --local-dir weights/Qwen3.8-27B
```

Expected: ~50 GiB in 18 safetensors shards + tokenizer.json.

## Project Status

All phases complete and verified on 2× A40 (sm86). The operator layer was
migrated from handwritten CUDA to FlashInfer + FLA + causal-conv1d.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding, build system, FFI hello-world | ✅ |
| 2 | Library operators (FlashInfer norm/RoPE/SiLU, cuBLAS GEMM, embedding) | ✅ |
| 3 | Full attention (FlashInfer causal prefill, KV cache, GQA, output gate) | ✅ numerically tested |
| 4 | GatedDeltaNet (causal-conv1d, FLA chunked/recurrent, gated norm) | ✅ vs PyTorch |
| 5 | Haskell model definition, safetensors loading, GPU partition | ✅ 1199 tensors |
| 6 | Multi-GPU engine, batched chunked prefill (≤128 tokens/chunk) | ✅ 2× A40 |
| 7 | Tokenizer + CLI + greedy generation + streaming | ✅ coherent text |
| 8 | End-to-end validation (27B logits, 433-token long sequence) | ✅ 20/20 argmax |
| 9 | Descriptor-driven families (dense + MoE + Qwen3-Next + DeepSeek-V2 MLA), multi-arch SASS/PTX, placement policies (layer split, TP, EP) | ✅ verified on sm_86 (A40); sm_89 operator suite on L20 |
| 10 | Resource safety and regression gates: buffer ownership at allocation, cross-device read-completion ordering, bounded safetensors parsing, tokenizer capacity/streaming protocol, generation budget/EOS/error semantics, MLA shared-memory bound, TP shard-coverage rule, FP32 expert-parallel merge | ✅ verified on sm_86 (A40): ctest 11/11, cargo 11/11, hspec 41 + 14, Qwen3.8 golden bitwise identical, TP2 rms ≤ 0.05 with identical tokens |
| 11 | Execution manifest and capture provenance (plan Stage 0): content identities `semantic_id`/`numerical_policy_id`/`deployment_id` over canonical blocks, build-time provenance generation, parameter identity, region/case determinism registry, and strict/diagnostic/legacy capture comparison | ✅ CPU gates (ctest `test_manifest` + `test_manifest_hashes`, hspec manifest specs, CLI runner); engine query verified on sm_86 |
| 12 | Region inventory and cross-case harness (plan Stage 1): a committed inventory of every in-scope dense/dense-hybrid forward region (inputs, outputs, persistent state, saved-for-backward values, case availability) with a per-case-pair verdict (`exact`/`exception`/`unverified`/`not_applicable`), the device fixtures that compare identical inputs and state across cases, explicit unsupported-shape/case rejection, and the region entry-point cost | ✅ CPU gate `test_region_inventory` + device harness `test_region_cases` on sm_86 |
| 14 | Trainable runtime and parameter lifecycle (plan Stage 3): a parameter store with logical ids, tying, frozen parameters, versions and derived copies; exclusive update windows whose publication casts masters and refreshes every derived copy; teacher-forced all-position forward with a fused row-by-row log-softmax; a training step that retains activations and GDN chunk-boundary states; and the Haskell schedule with its typed handles | ✅ CPU store/lifetime gate + device gate on the synthetic checkpoint + the Haskell schedule cross-check |
| 16 | Synchronous training/rollout baseline (plan Stage 5): the layer-level backward that chains the Stage-4 regions into an SFT step (attention with its fused output gate and per-head norms, the dense MLP, the residual adds and the four norms, recomputed per sublayer from the three retained boundaries), AdamW and publication over the parameter store, resumable training state, the CUDA-free phase/budget/selection-record/host-FP64-sampler contract for the rollout half, and the rollout driver (`engine_rollout_sample`, which generates under a *borrowing* `TrainContext` and stamps the record with the version that context read) | ✅ CPU gate `test_train_loop`; device gate `test_sft` against a `transformers` training run on the tiny dense checkpoint (first-step loss 6.3e-05 relative, the tied parameter's gradient cosine 0.9990, the fixture overfits 7.03 -> 0.61) plus its rollout section (20000 draws matching the model's entropy to +2.11 sigma and every counted bin to 2.94 sigma, the version binding with a +1 stamp refused, and the sampler-vs-trainer log-probability gap 4.8e-07) and its group section (four completions under one version, a deterministic verifier, a zero-variance subset refused, and the sequence-level objective ratio exactly 1 on the engine's own record) |
| 19 | Bounded-staleness admission protocol (plan Stage 8): a bounded `LearnerQueue` over whole completed groups, immutable behavior versions, the admission lag `learner_committed_version - behavior_version`, a bounded allowlist of live versions, and a consumed-group ledger so a retry cannot count a response twice | ✅ CPU gate `test_rollout_queue`: whole groups only, both capacity bounds, a version ahead of the learner and one past the cap both refused, lag zero reproducing the synchronous objective and gradient bitwise, and deterministic lag 0/1/2 injection reported as a **policy change** through Stage 6's classifier. **The asynchronous GPU half (snapshots, device leases, publication transfer, throughput) is not implemented** and is recorded as such |
| 18 | GSPO and GRPO group objectives (plan Stage 7): the plan's `min(s_i A_i, clip(s_i) A_i)` with the length-normalized sequence ratio and the population group advantage, in the plan's `mean_i` form (one term per *response*) plus a token-level mode, with separate sequence and token clipping statistics | ✅ CPU gate `test_gspo`: both objectives against an independent FP64 reference and its central difference on an unequal-length group (GSPO 0.126292 vs GRPO 0.113146, which is why equal-response and equal-token weighting must be compared on the same groups), the unclipped analytic `A_i s_i / T_i` (s_i not detached), both advantage signs, both clip boundaries, masks, π_θ = π_b, zero-variance and truncated groups. No online RL loop is implemented |
| 17 | Numerical-alignment decision (plan Stage 6): one verdict per Stage-1 region, derived from the committed inventory — a measured `exception` is a declared exception carrying its widest bound, an `unverified` pair is invariant-kernel-pending with the option that would remove it named, the rest are exact by construction — plus the reporting rule that keeps a numerical mismatch separate from a policy change and a sampler difference | ✅ CPU gate `test_alignment`: every region resolves, an exception without a measured bound and a pending region without named work are both refused, a policy change or sampler difference is refused as a numerical comparison, and an exact-by-construction region is not checked by a tolerance. **No alignment kernel was written**: the tracked work (a fixed reduction tree / constrained library configuration, CPR for GDN) is named rather than silently assumed |
| 15 | Backward, losses and optimizer (plan Stage 4): a CUDA-free differentiation contract (cast-is-identity, the saved-statistic rule, the losses, AdamW, a CRC-checked checkpoint over parameters/moments/RNG/cursor) and one backward per Stage-4 table row — elementwise, the four norms, embedding scatter-add, RoPE, Q/gate split, GEMM dX/dW, GDN conv1d/prepare, the paired attention-from-LSE and GDN-core-from-chunk-states, plus the losses' gradient | ✅ CPU gate `test_backward` (losses/AdamW vs FP64, checkpoint refusals, an overfit that resumes bitwise) + device gate `test_backward_kernels` (every kernel vs a double-precision definition or its central difference) on sm_86 |
| 13 | Feasibility and invariance experiments (plan Stage 2): PP inertness on a one-GPU model (logits + 324 tap dumps), GDN decomposition with prepare/core attribution, attention tiling across head dims/GQA/KV lengths, GEMM shape invariance at the real projection shapes, and the attention forward/backward pair (LSE availability, convention, gradient check, resource estimate) | ✅ six experiments on sm_86; six case pairs became measured `exception`s; no pair promoted to `exact` |

Known gaps, stated rather than implied:

- The expert-parallel equivalence re-run after the FP32 merge landed was stopped
  before it finished, so that verdict is pending. The gate itself is unchanged
  (`test_tp.py --ep 2`: identical greedy tokens, logit RMS ≤ 0.05).
- sm_90a is compile- and artifact-verified only; there is no H200 hardware here.
- Long context is not supported: the MLA attention kernel's shared-memory budget
  caps the cached sequence length (a 16K context does not fit) and both
  `engine_create` and the kernel entry reject anything longer.

## Design Decisions

See [docs/design.md](docs/design.md) for the full architecture rationale.

Key choices:
- **Three placement policies** from one descriptor: layer-wise partitioning,
  replicated tensor parallel and expert parallel, whose weight splits come from
  the descriptor's per-role shard rules (no family knowledge in the engine)
- **Library-backed operators** with a small native C ABI, not a wrapper around a serving engine
- **Chunked prefill and recurrent decode** with reusable per-device scratch and state
- **BF16** storage throughout (no quantization), with FP32 accumulation where it
  matters: norms, the MoE expert sum, and now the cross-rank merge, which keeps a
  partial in FP32 until one rounding turns it into the activation
- **The element type belongs to the data**: the cross-device primitives take the
  buffers' type (F32, F16, BF16, either FP8 flavour) and size every copy from it
  rather than assuming one
- **Ownership from the moment of allocation**: a layer buffer is registered with
  its layer before anything that can fail, so a failed initialization releases
  everything it had allocated
- **Greedy decoding** only (no sampling)

## License

Project code: MIT. The vendored FLA-derived chunk kernels retain the Apache-2.0
license in `csrc/triton/LICENSE`; causal-conv1d retains its license in
`csrc/third_party/causal_conv1d/LICENSE`. FlashInfer and upstream FLA retain their
respective dependency licenses.
