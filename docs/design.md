# Design: Haskell Inference Framework Demo

## Goal

Demonstrate that Haskell can serve as the orchestration layer for a production-style
LLM inference engine, with C/CUDA handling tensor computation. Target model:
Qwen3.8-27B (hybrid Full-Attention + GatedDeltaNet, 64 layers, ~50 GiB BF16).

## Constraints

These describe the current inference engine, not the proposed trainer in
[plan-numeric-contract.md](plan-numeric-contract.md).

- Single request at a time (chunked prefill, no multi-request batching)
- Greedy decoding only; stochastic rollout sampling is planned, not implemented
- 1–8 GPUs; layer-wise placement and separate TP/EP policies (verified on 2× A40 46 GB, sm_86); combined TP+EP is rejected
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

The C side (`csrc/model_desc.c`) parses it strictly: unknown keys, missing required
keys, wrong types and inconsistent dimensions are hard errors. Roles and their
expected tensor shapes are compiled into C; which tensors a family has is data.
Committed snapshots live in `descriptors/`. The model-directory round-trip test
in `tests/Spec.hs` currently re-derives only `qwen38-27b.json`, conditional on
`INFER_MODEL_DIR`; it is not an all-family snapshot gate. Other adapters have
structural tests, and each new training target needs an explicit snapshot and
runtime compatibility check.

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
TP rule — norms, embeddings, GDN — stay whole: GDN keeps the head counts its AOT
cubins were compiled with. MoE layers under TP are rejected and directed to the
existing EP policy. Every rank runs the full forward pass on replicated
activations and a sublayer whose weights were split all-reduces its partial
output (leader reduce + broadcast, `collective.cu`) before the residual add, so
the residual stream stays identical across ranks.

The TP logits are *not* bit-identical to the layer-wise split: split-K GEMM sums
and the BF16 all-reduce reorder additions. The gate for the equivalence test
(`tests/test_tp.py`) is identical greedy tokens plus logit RMS ≤ 0.05, comparable
to the engine-vs-PyTorch RMS (0.02–0.04 for this family), not stricter than it.
The historical layer-wise refactor gate is separate from a still-needed test of
bitwise invariance across different layer-wise placements.

### Expert-parallel placement

`--ep N` splits whole experts across the ranks: rank r holds experts
@[r * E\/ep, (r + 1) * E\/ep)@ and loads only those tensors (the descriptor
marks the routed-expert roles `out_experts`). The router and the shared experts
stay replicated, so every rank computes the same top-k and the same shared
output; the routed part is a partial sum that is all-reduced before the shared
experts are added — fewer experts per rank must not change how often the shared
contribution is counted, which is why the reduce sits between the two halves of
the MoE (`forward_moe_routed` / `forward_moe_shared`). Entries routed to another
rank are localized away (`kernel_moe_localize_ids` -> -1 slots the combine
skips), so the packed expert work stays proportional to the local experts.

Combined TP+EP placement needs subgroup collectives and is rejected for now;
MoE layers under `--tp` point at `--ep`. Verified with `tests/test_tp.py --ep 2`
on Qwen3-30B-A3B (see the README gate list).

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

The most complex component. Per-layer forward for this family's separate
projection layout (the Qwen3-Next adapter also supports fused projections):

```
x → in_proj_qkv [5120→10240] → q(2048), k(2048), v(6144)
x → in_proj_z   [5120→6144]  → z
x → in_proj_b / in_proj_a   → b(48), a(48)

[q,k,v] → causal_conv1d(k=4, state) → BF16 → SiLU → conv_out
conv_out → Q/K L2 normalization + head expansion + gating(b,a) → prepared

prepared → gated_delta_rule(q,k,v, ssm_state) → delta_out
           (tokens == 1: recurrent; tokens > 1: chunkwise pipeline)

delta_out → RMSNormGated(z) → normed
normed → out_proj [6144→5120] → layer_output
```

**Delta rule (per v-head h, k-head kh = floor(h/3)):**
```
D = alpha[h] * S[h]
delta = v[h] - k[kh] @ D
S[h] = D + outer(k[kh], beta[h] * delta)
o[h] = (q[kh] / sqrt(128)) @ S[h]
```

Here q/k are the L2-normalized vectors; `alpha = exp(g)` and
`g = -exp(A_log) * softplus(a + dt_bias)`. The implementation rounds sigmoid(b)
through BF16 before using it as FP32 beta. State S is [48, 128, 128] FP32 = 3 MiB
per layer, 144 MiB across 48 layers. `tests/kernels/test_gdn.cu::delta_reference`
records the decay-before-prediction and Q-scale convention. The equations specify
the mathematical function; kernel reductions and dtype boundaries additionally
specify its numerical execution.

Chunked prefill already exists. `max_chunk` bounds each engine call internally,
while the GDN pipeline groups work in 64-token chunks. A one-token tail takes the
recurrent path. Different decompositions are not assumed bitwise equivalent;
the numerical-contract plan separates state-correctness tests from invariance
claims. Any new CPR forward must be paired with a revalidated backward.

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

### MLA (multi-head latent attention) layer

A layer whose mixer is `mla` follows DeepSeek-V2. The KV cache holds the
compressed latent plus the shared rope key — `kv_lora_rank + qk_rope_head_dim`
values per token, 1152 bytes for V2-Lite against roughly 8 KiB for the
equivalent GQA cache, which is the point of the design. `kv_b_proj` decompresses
latent to per-head `k_nope`/`v` for the whole cached range on every step (the
naive path; absorbing `kv_b` into `q` and `o` is the optimization that avoids
it). Scores scale by `1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`, which is
`1 / sqrt(192)` for V2-Lite.

Three details are load-bearing, and each was settled against the oracle rather
than from the reference source:

- The latent is normalized by an ordinary RMSNorm over the first `kv_lora_rank`
  channels of a wider row. The library norm kernel assumes both operands are
  contiguous `[rows, cols]` blocks, so the slice is staged through a repack block
  (copy out, normalize, copy back); normalizing in place corrupts every row after
  the first.
- RoPE is interleaved, not split-half: the rope slice reads as complex pairs
  `(x[2i], x[2i+1])` whose frequencies are built from `dim = qk_rope_head_dim`,
  applied to the query's rope slice and to the shared key alike.
- The attention kernel is one block per (query token, head) with the scores in
  shared memory and an in-block tree reduction. That reduction is exact only for
  power-of-two block sizes, so the launch rounds up to the next power of two with
  idle lanes carrying identity values. Rounding to multiples of 32 instead
  silently dropped lane groups from the softmax denominator at 96/160/192
  threads — a defect that only fires past 64 cached tokens and is therefore
  invisible to short-prompt gates; `test_mla` covers it by re-cutting the same
  token sequence so the two runs land in different block-size bands.

### What differs between families

The layer kinds share one path, so a new family is descriptor data plus, at most,
one kernel:

| | Qwen3.8-27B | Qwen3 / Mixtral | Qwen3-MoE | DeepSeek-V2-Lite |
|---|---|---|---|---|
| mixer | GDN + full attention | full attention | full attention | MLA |
| q/k norm | yes | yes (Mixtral: no) | yes | no (the latent norm plays that role) |
| attention output gate | yes | no | no | no |
| RMSNorm | Gemma (weight + 1) | plain | plain | plain |
| RoPE | partial (64 of 256) | full (128 of 128) | full (128 of 128) | decoupled partial (64 of 192) |
| FFN | dense | Qwen3: dense; Mixtral: routed experts | routed experts + `mlp.gate` router | layer 0 dense; elsewhere routed experts plus one shared MLP |
| token embeddings | untied | tied (Qwen3-4B) | untied | untied |

Recorded end-to-end comparisons include Qwen3.8-27B (bitwise against its own
baseline and 0.02–0.04 logit RMS against an independent PyTorch reference), plus
Qwen3-4B and Qwen3-30B-A3B (matching greedy tokens, logit RMS 0.1–0.4). These
numbers are not universal gates. One known MoE difference is expert-output
accumulation in BF16 in the reference versus FP32 followed by one rounding in the
engine; it does not explain dense-model differences by itself. Additional
implemented families include Qwen3-Next (verified on a synthetic checkpoint:
16/16 greedy tokens) and DeepSeek-V2 MLA (V2-Lite verified end-to-end against
transformers: 16/16 greedy tokens at 0.04–0.57 logit RMS, and the engine-internal
chunking check agrees to 0.07 with matching top-1). All of these numbers come
from the same 2× A40 sm_86 box; runtime coverage and training support must be
listed separately for each target.

### Prefill batching

The descriptor's `max_chunk` (default 128, validated against the kernels' hard
limit) sets the prefill batch size: activation buffers, position buffers and the
chunking loop all follow it, so a model that wants smaller batches only changes
data. Kernel-side limits stay where they belong -- FLA's chunk pipeline rejects
more than 128 tokens per call regardless of the descriptor.

### Layer kinds

A layer is `norm -> mixer -> residual -> norm -> ffn -> residual`. The implemented
mixers (`full_attn`, `gdn`, `mla`) and feed-forward kinds (`dense`, `moe`) come
from the descriptor and select a row in the dispatch tables in
`csrc/layer_dispatch.cu`. The engine's layer loop calls `forward_layer` and knows
nothing about kinds, so a new kind is a descriptor field plus a kernel file plus
a table row -- not a change to the loop. Inference support does not imply that a
kind has a backward implementation or belongs to the initial trainer scope.

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

The GDN gated-norm weight has an FP32 derived copy created after loading; it uses
the raw weight, without `+1`. Current GemmaRMSNorm passes the raw BF16 weight to
FlashInfer, which applies the Gemma convention, and RoPE uses FlashInfer's
position-ID kernel rather than engine-owned precomputed cos/sin tables.

These are inference-only weights today. A trainer must establish parameter
identity, sum gradients for tied embedding/LM-head roles (currently loaded into
separate allocations), and refresh derived copies after every update before a
new rollout version becomes visible. See the training-runtime stage in the plan.

### Memory budget (2× A40, 4096 context)

Approximate per-device budget for a balanced 32-layer split:
- Weights: roughly 25 GiB; embeddings/LM head can make the split uneven
- KV cache: 8 local attention layers × 4096 × 4 kv_heads × 256 dim × 2 (K+V) × 2 bytes = 128 MiB
- GDN state: 24 local layers × (60 KiB conv + 3 MiB SSM) ≈ 73.4 MiB
- Activations, GDN intermediates, logits and other scratch: shape-dependent; measure allocations for the selected `max_chunk`

The whole model, not each device, has 16 attention and 48 GDN layers. This
inference configuration fits the two-card target; it is not a training budget.
Optimizer states, saved activations and concurrent rollout snapshots require a
separate per-device, per-phase calculation in `plan-numeric-contract.md`.

## Testing strategy

| Level | Test | Pass criteria |
|-------|------|---------------|
| Descriptor | `ctest -R test_model_desc` (CPU) + Haskell spec | strict parsing/structural checks; conditional Qwen3.8 model-directory snapshot round-trip |
| Kernel | `ctest`: test_attention, test_gdn, test_collective, test_moe, test_norm, test_rope, test_mla, test_library_ops | vs CPU/PyTorch reference, BF16 tolerances; test_mla additionally re-cuts one sequence to catch block-shape-dependent defects |
| Engine | `tests/test_engine.py` vs independent PyTorch logits | argmax in the reference's max set; configured `--rms-tolerance` (default 0.1, family-specific overrides) |
| Chunking | same prompt, different prefill splits | top-1 equal, rms ≤ 5 (state-loss guard) |
| Refactor | `tests/capture_logits.py --compare` | numeric arrays identical; caller must currently establish same-build/input provenance because metadata is skipped |
| Long sequence | `tests/test_longseq.py` | chunk-split self-consistency + no repetition collapse |

We do not require bit-exact match with PyTorch. BF16 reassociation and algorithm
selection can move logits; exact reference ties are accepted as a set, while a
near-tie top-1 disagreement is not silently accepted. Different prefill splits
can also change GDN grouping and attention execution; whole-model RMS alone does
not attribute the difference to cuBLAS. Training gradient checks, cross-case
bitwise admission and provenance enforcement are planned gates, not current
coverage; see `plan-numeric-contract.md`.

## What this demo is NOT

- Not a production inference engine
- Not throughput-optimized (FlashInfer attention and chunked prefill exist; CUDA graphs and multi-request scheduling do not)
- Not quantized (BF16 only)
- Not multi-request batched
- Not multimodal (text only)
- Not a Haskell GPU compute framework (Haskell orchestrates, CUDA computes)
- Not yet a trainer or an asynchronous rollout service

## Future work

1. Extend model coverage and per-family snapshot/runtime gates; MoE and MLA
   inference already exist, but their backward paths do not.
2. Runtime validation on sm_90a hardware; multi-arch artifacts and sm_89 operator
   validation already exist.
3. Combined TP+EP with subgroup collectives, if justified. Separate TP and EP
   already exist and alter reduction order; initial training will instead use
   single-device or layer-wise placement with explicitly scoped invariance tests.
4. CUDA graph capture for decode, vocab-parallel argmax.
5. Sampling (temperature, top-p), HTTP API, streaming SSE.
6. Haskell-orchestrated SFT, OPD, GRPO, DAPO, GSPO and PPO over a shared region
   library. This first requires trainable parameter ownership, saved-activation
   lifetimes, backward kernels and independently checked loss/optimizer regions.
7. Bounded-staleness asynchronous RL, after a synchronous baseline. Sharing an
   implementation is not sharing mutable weights: concurrent actors need stable
   versions, cache ownership, recorded behavior logprobs and a validated
   off-policy objective. See [plan-numeric-contract.md](plan-numeric-contract.md)
   for the contract, GSPO objective, resource trade-offs and phased proposal.
