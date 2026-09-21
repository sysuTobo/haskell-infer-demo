# Design: Haskell Inference Framework Demo

## Goal

Demonstrate that Haskell can serve as the orchestration layer for a production-style
LLM inference engine, with C/CUDA handling tensor computation. Target model:
Qwen3.8-27B (hybrid Full-Attention + GatedDeltaNet, 64 layers, ~50 GiB BF16).

## Constraints

- Single request at a time (no batching)
- Greedy decoding only (no sampling)
- 2–8 GPUs (L20 46 GB, sm_89), layer-wise partitioning
- Pure text (no vision/multimodal)
- CLI interface with streaming output
- Correctness over performance

## Architecture

### Three-layer design

```
┌─────────────────────────────────────────┐
│  Haskell: orchestration & control flow  │
│  (model def, GPU partition, gen loop)   │
├─────────────────────────────────────────┤
│  C/CUDA: tensor computation             │
│  (kernels, cuBLAS, memory, multi-GPU)   │
├─────────────────────────────────────────┤
│  Rust: tokenizer (HuggingFace crate)    │
└─────────────────────────────────────────┘
```

**Why this split?**
- Haskell excels at pure logic: model configuration, partitioning, state machine
  for the generation loop. Type safety catches wiring errors at compile time.
- CUDA kernels must be C/C++ — no practical Haskell GPU compute path exists.
- The tokenizer ecosystem is in Rust/Python; Rust gives us a C ABI with zero
  runtime dependency beyond libstdc++.

### FFI boundary

The C API (`engine.h`) is model-level, not op-level:

```c
EngineHandle *engine_create(model_dir, config);
int engine_prefill(engine, token_ids, n, out_logits);
int engine_decode(engine, token_id, out_logits);
void engine_reset(engine);
void engine_destroy(engine);
```

Haskell never touches CUDA pointers directly. It passes configuration and
receives logits. This keeps the FFI surface small and auditable.

### Multi-GPU layer partitioning

64 layers split contiguously across N devices:
- Device 0: embedding + layers[0..K-1]
- Device N-1: layers[M..63] + final_norm + lm_head
- Cross-device: `cudaMemcpyPeerAsync` of activation [1, 5120] BF16 (10 KB)

KV cache and GDN state are device-local — no cross-device state sharing.
This is simpler than tensor parallelism and sufficient for single-request demo.

### Model architecture (Qwen3.8-27B)

| Component | Detail |
|-----------|--------|
| Layers | 64 total: 16 full-attention + 48 GDN |
| Pattern | Every 4th layer (3,7,11,...,63) is attention |
| Hidden | 5120 |
| Heads | 24 q-heads, 4 kv-heads (GQA 6:1) |
| Head dim | 256 |
| RoPE | Partial: 64 of 256 dims rotated, theta=1e7 |
| Attention gate | out = attn × sigmoid(gate) |
| MLP | gate_up [5120→34816] → SiLU×mul → down [17408→5120] |
| Norm | GemmaRMSNorm (weight+1, f32 accumulation) |
| GDN | conv1d(k=4) → delta_rule → gated_norm → out_proj |
| GDN state | conv [10240×3] BF16 + SSM [48×128×128] F32 per layer |
| Vocab | 248320, lm_head not tied |
| EOS | 248046, 248044 |

### GDN (GatedDeltaNet) layer

The most complex component. Per-layer forward:

```
x → in_proj_qkvz [5120→16384] → split q(2048), k(2048), v(6144), z(6144)
x → in_proj_ba   [5120→96]    → split b(48), a(48)

[q,k,v] → causal_conv1d(k=4, state) → conv_out
conv_out → l2norm + gating(b,a) → prepared

prepared → gated_delta_rule(q,k,v, ssm_state) → delta_out
           (decode: recurrent; prefill: per-token recurrent)

delta_out → RMSNormGated(z) → normed
normed → out_proj [6144→5120] → layer_output
```

**Delta rule (per v-head h, k-head kh = h/3):**
```
delta = v[h] - k[kh] @ S[h]
S[h] = alpha[h] * S[h] + beta[h] * outer(k[kh], delta)
o[h] = q[kh] @ S[h]
```

State S is [48, 128, 128] F32 = 3 MiB per layer, 147 MiB total.

**First version simplification:** prefill processes tokens one at a time
(same recurrent kernel as decode). This is O(n) kernel launches for n prompt
tokens but guarantees correctness and code reuse. Chunked prefill is a
future optimization.

### Weight loading

Safetensors format: 8-byte LE header length → JSON header → raw BF16 data.

Haskell parses the header (aeson), mmaps the data section, and passes
pointers to the C engine which uploads to the correct device via
`cudaMemcpyAsync`.

Weight name mapping (from kern reference):
- `model.language_model.embed_tokens.weight` → device 0
- `model.language_model.layers.{i}.*` → device per partition
- `lm_head.weight` → last device
- `model.language_model.norm.weight` → last device (converted to weight+1 f32)

Derived weights computed on-device after loading:
- `weight_p1 = float(norm_weight) + 1.0` (GemmaRMSNorm)
- RoPE cos/sin tables (generated from theta and max_position)

### Memory budget (2× L20, 4096 context)

Per device:
- Weights: ~25 GiB (half of 50 GiB)
- KV cache: 16 attn layers × 4096 tokens × 4 kv_heads × 256 dim × 2 (K+V) × 2 bytes = ~256 MiB
- GDN state: 48 layers × (61 KB conv + 3 MiB SSM) = ~147 MiB
- Activations + scratch: ~100 MiB
- **Total: ~25.5 GiB** → fits in 46 GB with headroom

## Testing strategy

| Level | Method | Pass criteria |
|-------|--------|---------------|
| Kernel | CUDA unit test vs CPU reference | ≤1 ulp BF16 |
| Layer | Fixed input, compare vs PyTorch eager | max abs diff < 1e-2 |
| Logits | Short prompt prefill vs HF/vLLM | top-1 token match |
| E2E | 5 prose prompts × 200 tokens greedy | ≥50 token prefix match |
| Multi-GPU | Same prompt on 2/4/8 GPUs | byte-identical output |

Note: we do NOT require bit-exact match with vLLM. cuBLAS algorithm selection
on thin shapes (N=96) causes 1-ulp differences that flip near-tie argmax.
This is documented in the kern project's Qwen3.8 bringup.

## What this demo is NOT

- Not a production inference engine
- Not optimized (naive attention, per-token prefill, no CUDA graphs)
- Not quantized (BF16 only)
- Not batched (single request)
- Not multimodal (text only)
- Not a Haskell GPU compute framework (Haskell orchestrates, CUDA computes)

## Future work

1. Chunked prefill for GDN (reuse FLA algorithm)
2. Flash attention for full-attention layers
3. CUDA graph capture for decode
4. Tensor parallelism for lower latency
5. Sampling (temperature, top-p)
6. HTTP API (servant/wai)
7. Streaming SSE output
