# Design: Haskell Inference Framework Demo

## Goal

Demonstrate that Haskell can serve as the orchestration layer for a production-style
LLM inference engine, with C/CUDA handling tensor computation. Target model:
Qwen3.8-27B (hybrid Full-Attention + GatedDeltaNet, 64 layers, ~50 GiB BF16).

## Constraints

- Single request at a time (no batching)
- Greedy decoding only (no sampling)
- 2–8 GPUs, layer-wise partitioning (verified on 2× A40 46 GB, sm_86)
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

### Hardware targets and cross-device communication

One shared library serves every target. `CMAKE_CUDA_ARCHITECTURES` is
`86;89;90a` (SASS) plus `90-virtual` (PTX): the PTX lets a device newer than the
built list JIT the nvcc-compiled kernels, which is the only forward-compatibility
mechanism available. The FLA chunk kernels are Triton AOT cubins with no PTX
equivalent, so `triton/build_aot.py` compiles them per architecture and the
launcher picks the image matching the current device's compute capability,
raising an explicit error when none was built.

Cross-device work runs through `csrc/collective.cu`. There is no NCCL: at 2-8
devices inside one process, the transfers and the reduction are cheaper than a
communicator and the demo stays dependency-free. Two rules:

- A copy never synchronizes the producer's stream. The producer records an event
  and the consumer's stream waits on it, so the GPUs stay busy while the
  activation crosses devices.
- Peer access is probed once at startup and reported; when it is unavailable
  (A40 pairs on PCIe have no P2P) the copies still work, staged through the host
  by the driver.

The all-reduce used by tensor/expert parallel placement is leader-based: every
follower copies its partial into the leader's staging buffer, the leader adds it
(elementwise bf16) and broadcasts the result. It is exercised by
`test_collective`, which also checks that a copy issued without host
synchronization still observes data produced asynchronously on the source stream.

Verification status per target: sm_86 and sm_89 are verified at runtime (sm_89
through the operator suite on an L20, which exercises the sm_89 Triton cubins);
sm_90a is compile- and artifact-verified (`cuobjdump` shows the cubins, and the
loader's arch table lists 90) until H200 hardware is available. Building without
a cubin for the device is an explicit error, not a driver failure.

### Model descriptor

Everything the engine needs to know about an architecture travels in one *flat*
JSON document, the model descriptor, produced by a family adapter in Haskell
(`src/Infer/Descriptor/Adapter/`) from the checkpoint's `config.json`:

- dimensions (hidden/intermediate/vocab, head counts, RoPE, norms, GDN layout);
- `layer_mixers` / `layer_ffns`: an explicit per-layer kind list, so C never
  infers the layout arithmetically;
- `role_templates`: the weight-name template for each role (`%d` = layer,
  `%e` = expert), which replaces the string-building loop that used to live in C.

The C side (`csrc/model_desc.c`) parses it strictly: unknown keys, missing keys,
wrong types and inconsistent dimensions are hard errors. Roles and their expected
tensor shapes are compiled into C; which tensors a family has is data. Committed
snapshots live in `descriptors/`, and a test re-derives them from the model
directory so a transformers upgrade shows up as a failing test rather than wrong
logits after a ten-minute load.

The descriptor is deliberately *not* a C struct: it spans variable-length
per-layer data, and a struct had to be mirrored field-for-field in Haskell, C and
two Python tests.

### FFI boundary

The C API (`engine.h`) is model-level, not op-level:

```c
EngineHandle *engine_create(model_dir, descriptor_json, num_devices, devices, layer_devices);
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

### Tensor-parallel placement (replicated)

`--tp N` keeps the whole model on every rank and splits only the weights the
descriptor marks as shardable (the per-role rules a family adapter derives):
attention q/k/v by head blocks (`out_heads`), o_proj and the MLP down projection
by input columns (`in_dim`), gate/up by output rows (`out_dim`). Roles without a
rule — norms, embeddings, GDN, MoE — stay whole: GDN keeps the head counts its
AOT cubins were compiled with, and MoE keeps complete experts until expert
parallelism lands. Every rank runs the full forward pass on replicated
activations and a sublayer whose weights were split all-reduces its partial
output (leader reduce + broadcast, `collective.cu`) before the residual add, so
the residual stream stays identical across ranks.

The TP logits are *not* bit-identical to the layer-wise split: split-K GEMM sums
and the BF16 all-reduce reorder additions. The gate for the equivalence test
(`tests/test_tp.py`) is identical greedy tokens plus logit RMS ≤ 0.05, well
inside the engine-vs-PyTorch RMS (0.02–0.04 for this family); the layer-wise
path itself stays bit-identical to the historical baseline.

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

### Mixture-of-experts feed-forward

A layer whose `layer_ffns` entry is `moe` routes through a sparse FFN instead of
the dense MLP. The arithmetic follows the reference implementation's order:
router logits in BF16 (as the reference linear layer emits them), scoring in
FP32, top-k selection with the lowest index winning ties, optional
renormalization, then the per-expert MLP. Routed expert tensors are fused into
one contiguous block per expert so the expert loop is a plain GEMM per slice.

Two implementation choices worth knowing:

- Experts run one at a time over their packed token slices and empty slices are
  skipped. For single-request decode at most `top_k` experts are non-empty, so
  this is both simple and cheap; a capacity-padded batched GEMM is the
  optimization for throughput, not for correctness.
- The per-expert token counts live on the device, so the loop pulls the offsets
  back once per MoE layer (one small sync). Padding every expert to a common
  capacity would remove it.

The expert's per-slice layout is `[gate; up]` because the activation kernel
consumes that adjacency, which in turn requires the expert width to be a multiple
of 8; unsupported widths are rejected with an explicit error.

### What differs between families

The layer kinds share one path, so a new family is descriptor data plus, at most,
one kernel:

| | Qwen3.8-27B | Qwen3 / Mixtral | Qwen3-MoE |
|---|---|---|---|
| mixer | GDN + full attention | full attention | full attention |
| q/k norm | yes | yes (Mixtral: no) | yes |
| attention output gate | yes | no | no |
| RMSNorm | Gemma (weight + 1) | plain | plain |
| RoPE | partial (64 of 256) | full (128 of 128) | full (128 of 128) |
| FFN | dense | dense | routed experts + `mlp.gate` router |
| token embeddings | untied | tied (Qwen3-4B) | untied |

Two families are verified end to end against independent PyTorch references
(Qwen3.8-27B bitwise against its own baseline plus a 0.02-0.04 logit rms;
Qwen3-4B and Qwen3-30B-A3B with every greedy token matching and a logit rms of
0.1-0.4, the spread coming from BF16 reassociation: the reference accumulates
expert outputs in BF16, this engine accumulates in FP32 and rounds once).

### Prefill batching

The descriptor's `max_chunk` (default 128, validated against the kernels' hard
limit) sets the prefill batch size: activation buffers, position buffers and the
chunking loop all follow it, so a model that wants smaller batches only changes
data. Kernel-side limits stay where they belong -- FLA's chunk pipeline rejects
more than 128 tokens per call regardless of the descriptor.

### Layer kinds

A layer is `norm -> mixer -> residual -> norm -> ffn -> residual`. The mixer
(`full_attn`, `gdn`, later `mla`) and the feed-forward kind (`dense`, later `moe`)
come from the descriptor and select a row in the dispatch tables in
`csrc/layer_dispatch.cu`. The engine's layer loop calls `forward_layer` and knows
nothing about kinds, so a new kind is a descriptor field plus a kernel file plus
a table row -- not a change to the loop.

Buffers are registered while loading (`LayerWeights::owned` for allocation,
`reset_zero` for per-sequence state), which makes `engine_destroy` and
`engine_reset` kind-agnostic; the hand-written free list they replaced would not
have survived per-expert weights.

### Weight loading

Safetensors format: 8-byte LE header length → JSON header → raw BF16 data.
The C loader (`safetensors_loader.cu`) scans the shards, expands the descriptor's
role templates, validates each tensor's shape against the role's expectation in
terms of the model dimensions, and uploads it to the owning device.

Weight placement is a role property, not a family property:
- embedding → first configured device
- final norm and `lm_head` → last configured device
- everything else → the device owning that layer

With the template scheme, the Qwen3.5 layout (`model.language_model.` prefix,
`lm_head` unprefixed, `linear_attn.*` for GDN layers) is data in the descriptor
rather than code in the engine.

Derived weights computed on-device after loading:
- `weight_p1 = float(norm_weight) + 1.0` (GemmaRMSNorm)
- the GDN gated-norm weight cast to F32 (raw weight: this variant does *not* add 1)
- RoPE cos/sin tables (generated from theta and max_position)

### Memory budget (2× A40, 4096 context)

Per device:
- Weights: ~25 GiB (half of 50 GiB)
- KV cache: 16 attn layers × 4096 tokens × 4 kv_heads × 256 dim × 2 (K+V) × 2 bytes = ~256 MiB
- GDN state: 48 layers × (61 KB conv + 3 MiB SSM) = ~147 MiB
- Activations + scratch: ~100 MiB
- **Total: ~25.5 GiB** → fits in 46 GB with headroom

## Testing strategy

| Level | Test | Pass criteria |
|-------|------|---------------|
| Descriptor | `ctest -R test_model_desc` (CPU) + Haskell spec | strict parse, reject bad/missing keys, adapter reproduces the snapshot |
| Kernel | `ctest`: test_attention / test_gdn / test_library_ops | vs CPU/PyTorch reference, BF16 tolerances |
| Engine | `tests/test_engine.py` vs independent PyTorch logits | argmax in the reference's max set, logit rms ≤ 0.1 |
| Chunking | same prompt, different prefill splits | top-1 equal, rms ≤ 5 (state-loss guard) |
| Refactor | `tests/capture_logits.py --compare` | bitwise identical (same build, same prompt) |
| Long sequence | `tests/test_longseq.py` | chunk-split self-consistency + no repetition collapse |

Note: we do NOT require bit-exact match with PyTorch. BF16 operator reassociation
and cuBLAS algorithm selection move individual logits by ~1e-2, which can flip a
near-tie argmax; the reference's tied maxima are therefore accepted as a set.

## What this demo is NOT

- Not a production inference engine
- Not optimized (naive attention, per-token prefill, no CUDA graphs)
- Not quantized (BF16 only)
- Not batched (single request)
- Not multimodal (text only)
- Not a Haskell GPU compute framework (Haskell orchestrates, CUDA computes)

## Future work

1. More model families (MoE, MLA) through new layer kinds in the descriptor
2. More GPU targets (sm_89/sm_90a) via multi-arch SASS plus runtime cubin choice
3. Expert parallel placement (shard whole experts across ranks) on top of the two
   policies that exist today: layer-wise partitioning and replicated tensor
   parallel. Note the numeric contract: layer-wise placement does not split
   intra-layer reductions, while tensor parallel splits the contraction dimension
   across ranks (measured equivalent within the documented RMS gate) and expert
   parallel changes how expert outputs are combined.
4. CUDA graph capture for decode, vocab-parallel argmax
5. Sampling (temperature, top-p), HTTP API, streaming SSE
6. A training framework (SFT, PPO, GRPO, DAPO, on-policy distillation) on small
   models, orchestrated in Haskell over the same region library as inference, and
   the numerical execution contract that makes train/inference bitwise agreement
   structural rather than disciplinary — see `plan-numeric-contract.md` for the
   staged plan, the audit of what the descriptor already pins versus what is
   still implicit, and what does not exist yet
