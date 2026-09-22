# haskell-infer-demo

A demo LLM inference framework written in **Haskell** with a C/CUDA backend,
targeting **Qwen3.8-27B** (hybrid Full-Attention + GatedDeltaNet architecture).

## Highlights

- **Haskell orchestration**: model definition, GPU partitioning and generation loop;
  a C ABI connects to the native weight loader and GPU operators.
- **Multi-GPU layer partitioning**: 64 layers split across 2–8 GPUs (tested on 2× A40 46 GB).
  Correctness-first; no tensor parallelism.
- **Hybrid architecture support**: 16 full-attention layers (GQA, partial RoPE,
  output gate) + 48 GatedDeltaNet layers (causal conv1d, gated delta rule,
  gated RMSNorm).
- **Single-request greedy decoding**: CLI with streaming token output.
- **Three-language build**: Haskell (Cabal) + C/CUDA (CMake) + Rust (Cargo).

## Architecture

```
Haskell (GHC 9.6)
├── Descriptor.hs        Model descriptor: typed record + flat JSON codec
├── Descriptor/Adapter/  Family adapters (HF config.json → descriptor)
├── Placement.hs         Placement policies (layer-wise today)
├── Model.hs             Per-layer plan: (mixer, ffn, device)
├── Runtime.hs           Engine lifecycle, weight loading
├── Generation.hs        Greedy decode loop, streaming output
├── Tokenizer.hs         FFI → Rust tokenizer
└── FFI/Engine.hs        FFI → C engine API
         │
         │ foreign import ccall
         ▼
C/CUDA (sm_86, CUDA 12.9)
├── model_desc.c     Strict parser for the descriptor (no family knowledge)
├── engine.cu        Multi-GPU forward, bounded chunked prefill
├── triton/          FLA-derived chunk kernels and upstream FLA decode AOT
└── kernels/
    ├── gemm.cu             cuBLAS BF16 matrix multiplication
    ├── flashinfer_norm.cu  FlashInfer GemmaRMSNorm and partial RoPE
    ├── attention.cu        FlashInfer causal attention, KV cache and output gate
    ├── gdn_conv.cu         Native causal-conv1d adapter (width 4)
    ├── fla_gdn.cu          FLA chunked prefill and recurrent decode
    ├── gdn_norm.cu         Model-specific gated norm and residual operations
    ├── silu.cu             FlashInfer SiLU × mul
    └── embedding.cu        Token embedding lookup
         │
         │ C ABI
         ▼
Rust (tokenizer-ffi)
└── lib.rs           HuggingFace tokenizers wrapper
```

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| GHC | 9.6+ | via [ghcup](https://www.haskell.org/ghcup/) |
| Cabal | 3.10+ | installed with ghcup |
| CUDA Toolkit | 12.9.1 | nvcc targeting sm_86 (A40) |
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
Runtime requires neither Python nor PyTorch. `CUDA_ARCH` defaults to `86`;
other architectures need separate compilation and numerical validation.

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

- `ctest --test-dir csrc/build-libs` — `test_model_desc` (CPU: descriptor parsing,
  validation, canonical echo) plus the operator-level GPU regressions:
  `test_attention` (causal GQA, KV write, output gate), `test_gdn` (FLA recurrent
  decode, causal-conv1d, gated norm), `test_library_ops` (FLA chunk pipeline
  T=1..128 vs PyTorch recurrent on both GPUs, GemmaRMSNorm, partial RoPE).
- `tests/test_engine.py` — full 27B vs independently generated reference logits
  (20/20 argmax), state reset, invalid-input/capacity checks, chunk-split
  self-consistency.
- `tests/test_longseq.py` — 433-token prompt chunk-split consistency plus
  128-token generation coherence (repetition-rate and tail-degradation checks).
- `tests/capture_logits.py` — records greedy logits for fixed prompts and compares
  two captures bitwise; the gate for refactors that must not change numerics.
- `cabal test infer-tests` — descriptor round-trip, layer plan and placement
  (no GPU); with `INFER_MODEL_DIR` set it also checks the adapter still
  reproduces `descriptors/*.json`.

## Usage

```bash
# Phase 1: FFI verification (no model weights needed)
cabal run haskell-infer-demo -- hello-gpu --device 0 --value 42

# Show model configuration (descriptor + placement; no weights needed)
cabal run haskell-infer-demo -- show-config --descriptor descriptors/qwen38-27b.json

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

## Design Decisions

See [docs/design.md](docs/design.md) for the full architecture rationale.

Key choices:
- **Layer-wise partitioning** over tensor parallelism (correctness first)
- **Library-backed operators** with a small native C ABI, not a wrapper around a serving engine
- **Chunked prefill and recurrent decode** with reusable per-device scratch and state
- **BF16** throughout (no quantization)
- **Greedy decoding** only (no sampling)

## License

Project code: MIT. The vendored FLA-derived chunk kernels retain the Apache-2.0
license in `csrc/triton/LICENSE`; causal-conv1d retains its license in
`csrc/third_party/causal_conv1d/LICENSE`. FlashInfer and upstream FLA retain their
respective dependency licenses.
