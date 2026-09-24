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
- **Single-request greedy decoding**: CLI with streaming token output decoded
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

- `ctest --test-dir csrc/build-libs` runs four suites that need no GPU at all:
  `test_model_desc` (descriptor parsing, structural validation, the
  engine-capability gate for the AOT GDN layout, canonical echo and the tp-role
  coverage rule), `test_safetensors` (malformed headers, offsets, shapes and
  dtypes, higher-rank tensors, row/slice capacity arithmetic), `test_manifest`
  (SHA-256 against the FIPS vectors, canonical format determinism, and the
  identity matrix: a semantic or numerical change moves the matching id only, a
  deployment-only change moves `deployment_id` only, a provenance or weight change
  moves no identity) and `test_manifest_hashes`, which re-derives every digest from
  the emitted document with `hashlib` so the emitter cannot certify itself. The
  rest of ctest
  needs a GPU: `test_engine_resources` (repeated failing initializations leave no
  handle and no device memory; skips itself when no device is visible),
  `test_collective` (event-ordered copies and the cross-device all-reduce on 2
  GPUs — once per element type — skipped with fewer devices) and the
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
- `cabal test all --enable-tests` — two suites, neither needing a GPU:
  `infer-tests` (descriptor round-trip, layer plan and placement; with
  `INFER_MODEL_DIR` set it also checks the adapter still reproduces
  `descriptors/*.json`) and `infer-generation-tests`, which drives the real
  generation loop and tokenizer wrapper against a scriptable C stub of the engine
  — token budgets, first/later EOS, prefill/decode failures and the cleanup that
  follows them.
- `ctest` also runs `test_norm` (both RMSNorm variants) and `test_rope` (partial
  and full rotation), each against a CPU reference.
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
