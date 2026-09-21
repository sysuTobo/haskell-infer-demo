# haskell-infer-demo

A demo LLM inference framework written in **Haskell** with a C/CUDA backend,
targeting **Qwen3.8-27B** (hybrid Full-Attention + GatedDeltaNet architecture).

## Highlights

- **Haskell orchestration**: model definition, GPU partitioning, generation loop,
  weight loading — all in pure Haskell with FFI to a thin C/CUDA layer.
- **Multi-GPU layer partitioning**: 64 layers split across 2–8 GPUs (L20 46 GB).
  Correctness-first; no tensor parallelism.
- **Hybrid architecture support**: 16 full-attention layers (GQA, partial RoPE,
  output gate) + 48 GatedDeltaNet layers (causal conv1d, gated delta rule,
  gated RMSNorm).
- **Single-request greedy decoding**: CLI with streaming token output.
- **Three-language build**: Haskell (Cabal) + C/CUDA (CMake) + Rust (Cargo).

## Architecture

```
Haskell (GHC 9.6)
├── Config.hs        Model dimensions, GPU partition
├── Model.hs         Layer type ADT (Attention | GDN)
├── Runtime.hs       Engine lifecycle, weight loading
├── Generation.hs    Greedy decode loop, streaming output
├── Safetensors.hs   Weight file parser (mmap)
├── Tokenizer.hs     FFI → Rust tokenizer
└── FFI/Engine.hs    FFI → C engine API
         │
         │ foreign import ccall
         ▼
C/CUDA (sm_89, CUDA 12.x)
├── engine.cu        Multi-GPU forward pass
├── memory.cu        GPU alloc/copy/peer
└── kernels/
    ├── rmsnorm.cu   GemmaRMSNorm (weight+1, f32)
    ├── rope.cu      Partial rotary (64/256 dims)
    ├── attention.cu Naive SDPA + KV cache + GQA
    ├── gdn_conv.cu  Causal conv1d (kernel=4)
    ├── gdn_delta_rule.cu  Gated delta rule (recurrent)
    ├── gdn_norm.cu  RMSNormGated + weight prep
    ├── silu.cu      SiLU × mul (MLP gate)
    ├── embedding.cu Token embedding lookup
    └── argmax.cu    Vocab argmax
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
| CUDA Toolkit | 12.x | nvcc targeting sm_89 (L20) |
| CMake | 3.18+ | for CUDA build |
| Rust | 1.75+ | for tokenizer-ffi |
| GPU | 2× L20 46GB | or equivalent with ≥25 GB per card |

## Build

```bash
./scripts/build.sh          # Build all (Rust → CUDA → Haskell)
./scripts/build.sh --tests  # Include CUDA kernel unit tests
./scripts/build.sh --clean  # Remove all build artifacts
```

## Usage

```bash
# Phase 1: FFI verification (no model weights needed)
cabal run haskell-infer-demo -- hello-gpu --device 0 --value 42

# Show model configuration
cabal run haskell-infer-demo -- show-config

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

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding, build system, FFI hello-world | Code written |
| 2 | Basic CUDA kernels (norm, RoPE, SiLU, embed, argmax) | Code written |
| 3 | Full attention layer (naive SDPA, KV cache, GQA) | Code written |
| 4 | GDN kernels (conv1d, delta rule, gated norm) | Code written |
| 5 | Haskell model definition, weight loading | Code written |
| 6 | Multi-GPU engine integration | Working (2x A40) |
| 7 | Tokenizer + CLI + generation loop | Code written |
| 8 | End-to-end validation + docs | Forward pass stable, numerical debugging pending |

## Design Decisions

See [docs/design.md](docs/design.md) for the full architecture rationale.

Key choices:
- **Layer-wise partitioning** over tensor parallelism (correctness first)
- **Naive attention** over flash attention (simplicity over speed)
- **Recurrent prefill** (per-token) over chunked (reuses decode path)
- **BF16** throughout (no quantization)
- **Greedy decoding** only (no sampling)

## License

MIT
