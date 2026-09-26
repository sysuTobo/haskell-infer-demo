# Design: Haskell Inference Framework Demo

## Goal

Demonstrate that Haskell can serve as the orchestration layer for a production-style
LLM inference engine, with C/CUDA handling tensor computation. Target model:
Qwen3.8-27B (hybrid Full-Attention + GatedDeltaNet, 64 layers, ~50 GiB BF16).

## Constraints

These describe the current inference engine, not the proposed trainer in
[plan-numeric-contract.md](plan-numeric-contract.md).

- Single request at a time (chunked prefill, no multi-request batching)
- Greedy decoding on the inference path. The trainer's synchronous rollout *does* sample
  (`engine_rollout_sample`, host FP64 at temperature 1 with no truncation, plan Stage 5),
  but the CLI's own sampler migration is planned, not implemented
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
elementwise and broadcasts the result. The element type is a property of the
data rather than of the transport: the caller states what its buffer holds
(F32, F16, BF16 or either FP8 flavour), every copy is sized from it, and the
reduction accumulates in FP32 and rounds once when storing. Placement reduces
BF16 activations, so that is the type it passes. It is exercised by
`test_collective`, which runs the same reduction once per element type and also
checks that a copy issued without host synchronization still observes data
produced asynchronously on the source stream.

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

### Tokenizer (Rust FFI)

The tokenizer bridge is capacity-explicit rather than fixed-buffer: every
conversion has a length query, text lengths count the terminating NUL, a buffer
that is too small writes *nothing* and reports a distinct code, and an empty
result is legal rather than an error. The Haskell side allocates from that
query, so a long prompt or a long generation is never silently truncated (the
old fixed 4096-byte buffer was), and text crosses the boundary as explicit
UTF-8 instead of whatever the locale happens to be. Failures raise, they are not
flattened into `[]` or `""`.

Streaming output goes through a stream handle that owns a tokenizer clone plus
its decode state, so it borrows nothing from the tokenizer and outlives any
single call. Feeding advances the state exactly once and only buffers text;
reading is a separate `pending`/`drain` pair, so a length query and a retry can
never double-advance or lose a chunk. `finish` flushes an incomplete trailing
byte sequence (decoding it the way whole-sequence decode would, replacement
character and all) and is idempotent; `reset` clears everything for reuse.
Streaming and whole-sequence decode use the same special-token policy.

The pinned `tokenizers` 0.20.4 has a defect in `step_decode_stream`: it drains
its retained ids from a `read_index` that lags one generation behind
`prefix_index`, which mis-emits once chunks arrive back to back and eventually
underflows `ids.len() - prefix_index`. Seeding `read_index` from `prefix_index`
before each step reproduces what upstream changed in 0.22, and the call is
isolated from unwinding so a dependency panic cannot abort the process across
the C ABI.

### Multi-GPU layer partitioning

64 layers split contiguously across N devices:
- Device 0: embedding + layers[0..K-1]
- Device N-1: layers[M..63] + final_norm + lm_head
- Cross-device: `cudaMemcpyPeerAsync` of activation [1, 5120] BF16 (10 KB)

KV cache and GDN state are device-local — no cross-device state sharing.

The CLI resolves the topology once, in `resolveTopology`: `--tp`/`--ep` on the
command line (both must be ≥ 1) decide the policy, and the descriptor handed to
the engine is rewritten to match — a pipelined run is single-rank (`tp_size =
ep_size = 1`) even when the snapshot was captured with `tp_size > 1`, and a
replicated run overwrites both counts and the rank indices. Without that, a
descriptor carrying `tp_size: 2` run with `--tp 1` would have Haskell report a
layer-wise split while the engine, which selects its execution mode from the
descriptor, ran a replicated TP forward.

### Tensor-parallel placement (replicated)

`--tp N` divides the rank's dimensions (attention heads, KV heads, the dense
MLP), so the roles that describe those dimensions have to be split. The
descriptor parser enforces that before anything is allocated: a `tp_size > 1`
document whose q/k/v, o, gate/up/down, or MLA q/kv_b/o roles do not carry the
rule their layout needs (`out_heads`, `in_dim`, `out_dim`) is rejected, because
an unsplit tensor under divided dimensions makes the forward read the wrong rows
and, for an output projection, past the end of its destination buffer. GDN roles
are not divided, so they stay unconstrained.


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
and the all-reduce (which carries BF16 activations) reorder additions. The gate for the equivalence test
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

The routed partial stays in FP32 all the way through the merge: each rank writes
it unrounded (`forward_moe_routed_f32`), the collective merges FP32 buffers with
the type it is told the data has, and a single cast turns the merged value into
this rank's activation. Storing the partial in the BF16 activation first and
merging it as BF16 rounds twice, where the single-rank path — which sums every
expert in FP32 and rounds once when it stores the activation — rounds once; that
extra rounding is exactly what a placement-equivalence run measures. With the
partial kept in FP32, what remains between the two placements is FP32
reassociation, not a precision step this path alone pays.

Combined TP+EP placement needs subgroup collectives and is rejected for now;
MoE layers under `--tp` point at `--ep`. `tests/test_tp.py --ep 2` on
Qwen3-30B-A3B is the gate: identical greedy tokens and logit RMS under the same
0.05 the TP case uses (see the README gate list).

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

#### Where the GDN forward rounds (required before differentiating it)

The equations above specify the function; a backward needs the *execution*, because a
derivative that consumes a differently-rounded operand is the gradient of a different
function. Stage 4 of `plan-numeric-contract.md` asks for this table before any
derivative is written, and `tests/kernels/test_gdn.cu` plus the Stage-4 gate pin each
row. Reading order is the conv output → prepare → core → gated norm:

| Step | Exact arithmetic | Rounding boundary |
|---|---|---|
| conv1d | `bias[c] + Σ_j w[c,j] x[t-(k-1)+j]`, state oldest-first, FP32 accumulation | output rounded to BF16 once, at the store (`causal_conv1d_update.cu`); the state keeps BF16 |
| conv SiLU | `x/(1+e^-x)` in FP32 from the BF16 input | result rounded to BF16 at the store (`kernel_silu_inplace`) |
| Q/K L2 norm | `x * rsqrt(Σ_d x_d² + eps)`, eps **1e-6 inside** the rsqrt, per (token, key head); no mean | rounded to BF16 once, when written into the head-expanded buffer; the three value heads sharing a key head each recompute the same norm rather than sharing it |
| v in prepare | a plain copy of the conv output | none beyond its existing BF16 storage |
| g (log-decay) | `-exp(A_log[h]) * softplus(a + dt_bias)`, softplus with the `x > 20 → x` branch | `a`, `dt_bias`, `A_log` are BF16 inputs; **g is stored FP32** |
| beta | `sigmoid(b)` | rounded **through BF16 and widened back to FP32** before use, so the core's beta is a BF16 value |
| core | the decay-before-prediction recurrence above, state FP32 | the chunkwise cubin rounds the state to BF16 for each MMA operand and accumulates in FP32; the recurrent path is FP32 throughout |
| gated norm | `s = rsqrt(mean(x²)+eps)`; `n = x*s`; `y = n * w * swish(z)` with the **raw** weight (no +1) | three boundaries: `x*s` → BF16, then `*w` → BF16, then `*swish(z)` → BF16 |

Two consequences the backward has to respect. `beta`'s BF16 rounding is *inside* the
sigmoid's chain: the convention this project adopted is that a cast is identity for
gradient propagation, so `d_beta` passes through unchanged while the sigmoid's own
derivative is evaluated at the pre-cast value. And the gated norm's weight gradient
belongs to the BF16 source weight, not to the FP32 effective copy the forward reads
(Stage 3's derived-copy registry), so a publication has to refresh that copy.

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

The combine writes the activation rounded to BF16. Under expert parallelism it
instead writes an unrounded FP32 partial, which the ranks then merge (see
Expert-parallel placement): that is the only arithmetic difference between the
two placements.

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
- The scores live in dynamic shared memory, so one key per cached token, on top
  of a fixed 256-float reduction array. That budget — the device's per-block
  limit, minus what the driver reserves, minus the reduction — is the *only*
  thing bounding the cache length, since the kernel does not opt into the larger
  carve-out. `kernel_mla_max_seq_len()` derives it, `engine_create` refuses a
  `max_seq_len` above it before allocating any cache, and `forward_mla_layer`
  refuses a call above it, so a too-long request fails where it was asked for
  instead of at an opaque launch error. 16K contexts do not fit and are rejected;
  this is not long-context support.

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

### Engine-capability validation

Structural validation accepts any self-consistent descriptor, so a checkpoint
with a GDN head layout the AOT cubins were not built for would otherwise reach
the first forward pass and index past its buffers. `engine_create` therefore runs
`model_desc_check_runtime_support` after `model_desc_validate`: a model with GDN
layers must have `gdn_head_dim` 128, `gdn_num_k_heads` 16 and `gdn_num_v_heads`
32 or 48 (the recurrent decode kernels exist only for those two counts), matching
`csrc/triton/build_aot.py`. It is a separate function from validation because
such a descriptor still describes its checkpoint faithfully — it is the engine,
not the descriptor, that cannot run it.

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

Safetensors format: 8-byte LE header length → JSON header → raw BF16 data. The
loader is split so the risky half is testable without a GPU: `safetensors.cpp`
holds the parsing and validation (no CUDA, covered by `test_safetensors` on CPU)
and `safetensors_loader.cu` only moves bytes.

Parsing is bounded and strict rather than string-matched: a cursor over the
header the file itself declares (header size checked against the real file size
and a sanity bound before anything is allocated), duplicate or unknown fields
rejected, unknown dtype tags rejected instead of defaulting to BF16, checked
arithmetic on every shape product, and each tensor's data offsets checked both
against the data section and against the exact byte count its shape and dtype
imply. A directory scan fails as a whole — no partially populated index — and a
tensor the text model never loads is still indexed: Qwen3.8-27B ships a rank-5
Conv3D vision patch embedding next to its 1198 text tensors, so the parser's rank
bound exists to keep the metadata struct fixed-size, not to describe the roles.
The upload helpers each take the destination capacity and re-check the byte
range against the file before a copy, so a short read or a failed copy is
reported rather than passed along.

Once parsed, the loader expands the descriptor's role templates, validates each
tensor's shape against the role's expectation in terms of the model dimensions,
and uploads it to the owning device. Every buffer a layer owns is registered with
that layer *at the moment it is allocated* (`alloc_owned`), before the upload or
any later allocation can fail: `engine_destroy` frees exactly what was
registered, so a failed initialization cannot strand device memory, and a
registration that itself fails frees on the owning device. Context-owned buffers
keep their single owner (the per-device fields the teardown walks) so nothing is
registered twice.

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

### Execution manifest and capture provenance

The descriptor is portable architecture: it carries no runtime facts, and its text
mixes placement into the architecture (`tp_size`/`ep_size`, the shard plan).
Bitwise comparison needs a different object, so `engine_manifest` answers *which
numerical execution did this run observe* as three separate content identities over
canonical JSON blocks, plus the provenance a comparison has to agree on:

- `semantic_id` — dimensions, layer/role semantics, the tied-role relations
  (derived from identical role templates, so Qwen3-4B's `embed=lmHead` is recorded)
  and the mathematical conventions;
- `numerical_policy_id` — the region/case → implementation binding table with its
  own digest, the effective constants (`effective_rms_eps`, `max_chunk`,
  `fla_chunk_size`), dtype and rounding boundaries, the attention split-KV setting,
  the GEMM algorithm policy (recorded as *unpinned*, not as a default) and the
  cross-device collective;
- `deployment_id` — placement, devices, the per-layer owner, the shard plan and the
  allocation/transfer choices;
- `weights.parameter_manifest_sha256` — the tensor index in sorted name order, an
  identity separate from the architecture; the raw-content hash is reported as
  `null` unless a caller pays the minutes of I/O for it, which is a different
  statement from "the content differs";
- provenance — the build facts (from a header the build step generates over its own
  artefacts, never a caller-supplied label) and the runtime facts (CUDA/driver/
  cuBLAS versions, per-device capability and UUID, and the kernel path the build's
  target lists select for that capability).

Constants that describe both the function and its realization are projected into
both identities rather than omitted from one: the declared `rms_eps` is semantic,
the FP32 value the kernels actually read is numerical. Enabling TP/EP moves the
numerical policy as well as the placement, which is consistent with placement
equivalence being gated on tokens plus an RMS band rather than bitwise equality. An
unestablished fact is reported as `unknown`/`unavailable`/`unsupported`/
`unspecified` and *refuses* strict admission: agreeing on an unknown is not
establishing a fact.

Comparison therefore has modes rather than a tolerance: strict (the identities, the
parameter identity, the placement unless a scoped claim is declared, and the
provenance all agree — and then the numeric arrays still have to be bitwise
identical), diagnostic (a deliberate difference reported for attribution, never a
contract pass) and legacy (a document without a manifest version stays numerically
comparable with an explicit `legacy/unverified` result and no invented identity).
The field ownership table, the projections and the region determinism registry are
in [manifest-contract.md](manifest-contract.md).

The canonical encoding is what makes the digests checkable by another
implementation: keys sorted by byte value, no whitespace, integers bare,
non-integer constants as decimal strings, printable ASCII only. `test_manifest`
pins the identity matrix without a GPU, `test_manifest_hashes` re-derives every
digest from the parsed document with `hashlib`, and the Haskell side parses the
same documents (`tests/ManifestSpec.hs`, `haskell-infer-demo manifest-compare`).
Two contract violations were caught this way by the engine's own document: a
missing top-level `manifest_version` and a region flag emitted as an integer where
the contract fixes a boolean.

### Region inventory: what a region reads, writes and may be compared across

The manifest's region registry answers "what ran, and is it repeatable?" for a
*capture*. A second, larger table answers the question the trainer needs:
`csrc/regions.c` inventories every in-scope forward region of the dense/dense-hybrid
path with its inputs, outputs, persistent state, saved-for-backward values and case
availability, and registers each reachable pair of execution cases (chunked
prefill, recurrent prefill, one-token tail, decode) as `exact`, a quantified
`exception`, `unverified` or `not_applicable`.

Three design rules make it usable rather than decorative. `exact` is claimed only
where the manifest registry already says `deterministic` — a single elementwise
pass, a row gather or a permutation — so a library's reduction order can never be
blessed by a fixture that happened to agree once; `ctest test_region_cases` enforces
that by requiring bitwise equality of output *and* persistent state, and
`ctest test_region_inventory` ties the two registries together in both directions.
A number is carried only by a measured `exception`; an `unverified` pair reports
what the fixture measured and claims nothing. And the trainer traversal
(`train_forward`, `eval_no_autograd`, `recompute`, `backward`) is registered as
*unavailable* from every region, because these cases are the inference engine's four
traversals and the trainer's is not one of them: an inventory cell for it would claim
coverage this registry's harnesses cannot produce. That is now a statement about
fixtures rather than about the API — Stages 3–5 built the traversal (the teacher-forced
forward, the SFT step, the backward, the losses and the synchronous rollout), and its
coverage is registered where its fixtures are instead of in a region's case list
(Stage 4's region table, walked against this inventory in both directions, and Stage 5's
step and rollout gates). Registering `train_forward` per region would need fixtures that
compare the training forward against the inference forward region by region, which do
not exist yet and which the gap list names rather than this column advertising. The
inventory is deliberately not
projected into `numerical_policy_id`: which test claims have been made is not part
of the numerical identity a capture observed. See
[plan-numeric-contract.md](plan-numeric-contract.md) Stage 1 and
[worklog.md](worklog.md) for the measured pair-by-pair evidence.

Stage 2 then asked what those measurements mean, and the answers are narrower than
the *shared-implementation* argument would suggest. Whole-layer placement is
bitwise inert on a fixed build and a same-architecture device pair (claim A), and
the GDN decomposition difference is entirely the core's, not prepare's (claim B,
51/51 prepare comparisons bitwise identical). But the library-backed regions are
*not* shape-invariant: cuBLAS changes its accumulation order with M, costing up to
5.4e-3 relative on a BF16 output and 3.0e-5 on an FP32 one (claim D), and
FlashInfer's attention tiling is bitwise stable for 202 of 264 measured tilings and
within half a BF16 ULP otherwise (claim C). That is why those pairs carry measured
`exception` bounds instead of an exactness claim, and it is the concrete answer to
"does sharing an implementation establish anything by construction": it does not.
The attention forward/backward pair (claim E) is feasible — the forward can return
the base-2 LSE a backward needs while leaving the output bitwise unchanged — and the
guarantee a trainer may claim about gradients is spelled out separately in the plan,
because a forward per-row property does not carry over to accumulated dW.

### The training runtime's ownership model

Stage 3's prerequisite is that a trainer can share the inference engine's weights
without either side copying them or surprising the other. The shape is: **the store
owns the parameter's identity, the engine owns its buffer.** A store resolves the
descriptor's roles into logical parameters (two roles with one template are one
parameter with two readers), gives each a version, and points its compute slot at the
engine's own BF16 weight - so a published update is visible to an inference forward by
construction rather than by a copy, and a reader can never observe a half-updated set
because an update requires exclusive ownership.

That leaves the two things a copy would otherwise hide. A *derived* copy - the FP32
GDN norm weight, cast from its BF16 source at load - would silently disagree with a
published update, so the store marks derived copies stale on publication and refuses
to close the window while one is stale; the refresh is enforced, not remembered. And a
*tied* parameter has one master but more than one reader buffer, so a publication
walks every alias: writing one and not the other would leave the model with two
versions of one weight, which is exactly the failure the tie is supposed to make
impossible.

The same section's retention rules are the other half. A training step keeps the
mixer, ffn and residual activations of every layer (the residual stream is updated in
place, so "read it later" is not an option) and the GDN chunk-boundary states under a
full-sequence schedule, with alias rules and per-consumer free points. Stage 4's
backward consumes them; what Stage 3 fixes is that they exist, are accounted for, and
cannot be freed while a consumer still holds them.

### The backward, the losses and the optimizer

Stage 4's shape is the same split as Stage 3: **C owns the contract and the
arithmetic, and the things a derivative can silently get wrong are rejections rather
than comments.** Four of them, in `csrc/include/backward.h` and `csrc/backward.c`:

- *Which forward was differentiated.* The plan requires the GDN forward's rounding
  boundaries to be documented before differentiating it, and the sub-section above is
  that table. It matters because a derivative that consumes a differently-rounded
  operand is the gradient of a different function: the output gate's gate-gradient uses
  the exact sigmoid while its attention-gradient uses the BF16-rounded one, and the GDN
  gated norm's weight gradient belongs to the BF16 source rather than Stage 3's FP32
  derived copy.
- *What a cast does.* A BF16 cast is a step function, so "identity for gradient
  propagation" is a choice the engine reports (`backward_cast_mode`) and the gate pins
  by comparing a whole fixture against a reference whose casts behave the same way. It
  is not finite-differenced, because that would produce a number for a function with no
  derivative there.
- *What a backward is allowed to read.* Each region's saved values are a table
  (`backward_required_values`), and a caller that runs a region's backward declares its
  retained set and is refused **by name** (`backward_check_retained`) when a value is
  neither retained nor recomputable — the CPU gate drives that refusal directly, which is
  the contract a region harness runs under. Stage 2 is why the distinction is explicit: the
  attention core's natural saving is one base-2 LSE, and the probabilities are a
  *recomputation* from it rather than a retained `[T,T]` tensor. The trainer's own step does
  not consult that table: it keeps Stage 3's retention plan and the SFT gate is what
  exercises it, so routing the step's plan through the table is an open item (named in the
  gaps list) rather than a property this bullet should claim.
- *What the optimizer's order is.* AdamW's bias correction, its decoupled decay and
  the position of `eps` are all part of the step rather than implementation detail,
  because each changes the result by more than the FP32 noise floor near a zero moment.
  The gate compares the whole step — parameters, both moments and the published BF16 —
  against an independent FP64 implementation of PyTorch's order.

The two paired regions are where "pairing" stops being free, and the honest split is
recorded rather than smoothed over: attention's backward recomputes P from the forward's
base-2 LSE (so the probabilities are never retained) while the forward's PV product
rounds P to BF16, which is the ~1e-3 residual the gate reports on dV; and the GDN core
backward differentiates the recurrence the model's design specifies rather than the
cubin's `(I+A)^{-1}`/BF16-MMA decomposition of it. Both are alignment work the plan
assigns to Stage 6, and saying so is what keeps a passing gate from reading as a
bitwise pairing.

### The training step and the synchronous baseline

Stages 3 and 4 built the ownership objects and the region backwards; Stage 5 chains them
into a step, and the chaining is where the interesting decisions are:

- **The step retains boundaries, not activations.** A layer keeps its input residual, its
  mixer output and its feed-forward output; everything finer (the normalised input, the
  projections' inputs, the attention's queries) is *recomputed* by the sublayer's own
  backward from those three values into the layer workspace. That trades arithmetic for
  memory, which is the side of the plan's "saved or recomputed statistics" a small-memory
  trainer wants; retaining the fine values is the same code path with a larger
  `step_plan`.
- **A gradient's operand is the value the forward read.** A GEMM's dX and dW consume the
  FP32 *widening* of the BF16 weight (`float(bf16(w))`), never the FP32 master: the two
  differ by the publication rounding, and a cast is identity for gradient propagation, so
  the faithful operand is the rounded one.
- **The residual's identity branch is explicit.** `r1 = r0 + m` and `r2 = r1 + f` are BF16
  adds whose gradient is the identity in FP32, so the incoming gradient is seeded into the
  running residual gradient and each sublayer's internal backward *adds* to it. Not doing
  that (accumulating into whatever the buffer held) produced a *growing* loss, which is how
  the wiring bug was found.
- **The loss keeps both normalisations and both log-probabilities.** The masked cross
  entropy's mean is taken inside the loss, and the record keeps the model's log-probability
  *and* the sampler's, because with a transformation the objective has to say which one it
  optimises.
- **A phase is a state machine, not a flag.** SFT holds a step's retained values and the
  optimizer state; a rollout borrows the parameter version through Stage 3's context object
  (so the store itself refuses an update while it reads) and holds a KV cache and the GDN
  state instead. A phase boundary resets the sequence, and a record is bound to the version
  it was generated under — reading it after an update is refused, which is the plan's
  "never reconstruct an old denominator using updated weights".
- **The rollout's version is read, not asserted.** `engine_rollout_sample` opens a *rollout*
  `TrainContext` for the whole generation and stamps the record with the version that
  context borrowed, so a caller cannot label a completion with a version the engine did not
  read and the store cannot write an update underneath it. The generation path and the
  teacher-forced trainer path are the same weights through different code (a KV cache
  against a causal mask), so the gate measures the gap between their log-probabilities
  (4.8e-07 on the tiny fixture) instead of assuming the two denominators are one.
- **A group is one version and one configuration, and the reward is the caller's.** G
  completions of one prompt differ only in their seed, are recorded into one `TrainGroup`
  whose version is pinned by its first member (the engine can generate a record under a newer
  version and the group refuses it), and each carries a reward the caller's *deterministic*
  verifier computes — no reward model exists to be trained. The advantages then come from
  `backward_group_advantage` over those rewards, and the plan's degenerate case is a refusal
  the caller has to waive explicitly (two members that scored the same are a zero-variance
  group).
- **Offloading the optimizer state is a declaration, not a guess.** `train_loop` can neither
  move nor free the store's buffers, so the plan's "retain or explicitly offload it according
  to the budget" is split: the *budget* says what a phase may hold (0 optimizer bytes for a
  rollout, whose step reads none of it) and the *loop* records what a caller actually shed,
  queryable at the boundary. A phase starts at nothing and the boundary clears it, so no
  phase inherits another's answer. The engine does not move the buffers yet, which the plan's
  Stage-5 limits state.

The step is wired for the attention and dense-MLP path on one device; the GDN mixer's
backward and the placements are named in the plan's Stage-5 status as the remaining work
rather than implied to exist, and a group's G completions are collected by looping (there
is no batched group driver).

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

`scripts/build.sh --tests` runs every layer in order — `cargo test --locked
--offline`, the whole CTest suite, then `cabal test all --enable-tests` — and
stops at the first failure, so one command answers "is the tree green".

| Level | Test | Pass criteria |
|-------|------|---------------|
| Descriptor | `ctest -R test_model_desc` (CPU) + Haskell spec | strict parsing/structural checks, the tp-role coverage rule; conditional Qwen3.8 model-directory snapshot round-trip |
| Checkpoint | `ctest -R test_safetensors` (CPU) | malformed headers, offsets, shapes and dtypes rejected without allocating; higher-rank tensors indexed; row-gather and slice capacity arithmetic |
| Kernel | `ctest`: test_attention, test_gdn, test_collective, test_moe, test_norm, test_rope, test_mla, test_library_ops | vs CPU/PyTorch reference, BF16 tolerances; test_mla additionally re-cuts one sequence to catch block-shape-dependent defects; test_collective runs the same all-reduce once per element type |
| Generation (CPU) | `cabal test infer-generation-tests` | budget/EOS/error semantics of the real generation loop, with a scriptable engine stub instead of a GPU |
| Backward | `ctest -R test_backward` (CPU) + `test_backward_kernels` (GPU) | every Stage-4 region's backward against a double-precision definition or a central difference of it; the attention pair against an analytic reading of the device's own base-2 LSE; the GDN core backward with a nonzero initial state across one and three chunks; the losses and AdamW against independent FP64; and a checkpoint resume bitwise equal to an uninterrupted run |
| SFT step | `ctest -R test_sft` (GPU) + `test_train_loop` (CPU) | the first step's loss against a `transformers` training run (6.3e-05 relative), the per-parameter gradient direction against torch (cosine 0.999), the fixture overfitting, a bitwise state round-trip, and the phase/budget/record/sampler contract |
| Resource safety | `ctest -R test_engine_resources` | repeated failing creations return no handle, explain the error and move no device memory; a valid checkpoint still builds afterwards |
| Engine | `tests/test_engine.py` vs independent PyTorch logits | argmax in the reference's max set; configured `--rms-tolerance` (default 0.1, family-specific overrides) |
| Chunking | same prompt, different prefill splits | top-1 equal, rms ≤ 5 (state-loss guard) |
| Manifest | `ctest -R test_manifest` (CPU, plus the Python re-derivation), `tests/ManifestSpec.hs` | canonical form re-serializes byte-for-byte in a second implementation; every digest re-derives; a semantic/numerical change moves its own id only, a deployment-only change moves `deployment_id` only, and an unestablished provenance value is refused rather than defaulted |
| Refactor | `tests/capture_logits.py --compare` | numeric arrays bitwise identical in every mode; strict admission additionally requires matching identities and provenance, a placement-only difference needs `--deployment-scoped` and is reported as scoped, and a capture without a manifest reports `legacy/unverified` (exit 2) instead of a pass |
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
- Not an asynchronous rollout service (a *synchronous* SFT/rollout step exists; staleness,
  batching and a reward model do not)

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
