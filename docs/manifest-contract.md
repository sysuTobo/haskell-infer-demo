# The execution manifest

The architecture descriptor ([design.md](design.md), `descriptors/*.json`) says
what a model *is*. It deliberately carries no runtime facts, and it contains both
architecture and placement fields. The **execution manifest** says which numerical
execution a capture actually observed, as three separate content identities plus
the provenance a bitwise comparison has to agree on.

The manifest is a **proposal-implementing** artifact of Stage 0 of
[plan-numeric-contract.md](plan-numeric-contract.md). It is not a claim that any
two implementations agree; it is the bookkeeping that makes such a claim
checkable, and it makes an unestablished fact visible instead of defaulting it.

## Where it comes from

| Surface | Purpose |
|---|---|
| `engine_manifest(engine, buf, len)` | Canonical manifest for this engine (C ABI, `csrc/include/manifest.h`) |
| `engine_manifest_version()` | Manifest wire version (`1`) |
| `haskell-infer-demo manifest --model-dir DIR [--write FILE] [--check]` | Print/summarise it; `--check` re-queries and requires byte-identical answers |
| `haskell-infer-demo manifest-compare LEFT RIGHT [--mode strict\|diagnostic] [--deployment-scoped]` | The admission verdict for two documents |
| `tests/capture_logits.py` | Records the manifest with a capture and compares three ways |

Triton/FLA versions, the archive revision, the CUDA toolkit and target lists, the
toolchain flag digest and the hashes of the generated kernels all come from
`csrc/gen_build_info.cmake`, which runs at *build* time over the artefacts the
build produced. CUDA runtime/driver/cuBLAS versions, per-device name, compute
capability, UUID and the selected kernel path come from CUDA queries at manifest
time. Nothing is a caller-supplied label: `manifest --descriptor` changes the
architecture, never the provenance.

## Canonical encoding

Every digest is a SHA-256 over the canonical text of one block, so a consumer in
another language can re-derive it instead of trusting the emitter:

- keys sorted by byte value, no whitespace, `,`/`:` separators;
- integers bare, `true`/`false`/`null` as JSON literals;
- **non-integer constants as decimal strings** (`"%.17g"`). Decimal-to-JSON float
  formatting is the one place two languages legitimately disagree, and an identity
  must not depend on a `printf` convention;
- strings restricted to printable ASCII, so no escaping convention can differ.

Each identity block is an object `{"fields": {…}, "<name>_id": "<sha256>"}`, so the
digest is over `fields` alone and needs no "remove the digest field first" rule.
The committed region table is hashed separately and its digest
(`numerical_policy.fields.regions_sha256`) is part of the numerical identity.

Verified by construction, not by assertion: `tests/manifest_test.c` checks
determinism, `tests/test_manifest_hashes.py` re-parses, re-canonicalises and
re-hashes every block with `hashlib` (and re-checks the identity matrix), and
`tests/ManifestSpec.hs` parses the same documents on the Haskell side.

## Field ownership

Three identities, and the rule for each field. This table is the specification:
a field that appears in two identities is *projected* into both deliberately.

### `semantic_id` — what the architecture means

| Field | Source |
|---|---|
| `family`, `model_type`, `num_layers`, `hidden_size`, `intermediate_size`, `vocab_size` | descriptor |
| `num_heads`, `num_kv_heads`, `head_dim`, `rotary_dim`, `rotary_theta` (declared, as a string) | descriptor |
| `rms_eps_declared` (declared value, as a string) | descriptor |
| `norm_style` (`gemma_weight_plus_one` / `plain_rms`), `attn_qk_norm`, `attn_output_gate`, `q_gate_interleave` | descriptor |
| `rope_convention` (`partial_split_half`), `gdn_decay_order` (`decay_before_prediction`), `gdn_q_scale` | engine convention (`docs/design.md`) |
| `gdn_*` head layout and `mla_*` / `moe_*` dimensions and switches | descriptor |
| `eos_tokens`, `layer_mixers`, `layer_ffns` | descriptor |
| `roles`, `role_templates`, `tied_roles` | descriptor; `tied_roles` is *derived* — the pairs of roles whose templates are identical (Qwen3-4B's `embed=lmHead`), because tying is a semantic fact the descriptor only implies |
| `max_position_embeddings`, `max_seq_len` | descriptor |

Not here, on purpose:

- `tp_size`, `tp_rank`, `ep_size`, `ep_rank`, `role_shards` — placement (see
  `deployment_id`), although they live in the descriptor document;
- `fla_chunk_size`, `max_chunk` — realization (`numerical_policy_id`);
- the weight values — a separate identity (see below);
- the sampling temperature and seed — request config and replay data, not a model
  identity (see below).

### `numerical_policy_id` — how the semantics are realized

`max_chunk`, `fla_chunk_size`, `effective_rms_eps`, `effective_rope_theta`,
`attention_split_kv` (`disabled_null_workspace`), `attention_workspace`,
`norm_impl`, `gemm_algorithm_policy` (`cublas_default_heuristic_unpinned`),
`gemm_output_type`, `lm_head_output_type`, `cast_boundaries`,
`gdn_recurrent_impl`, `mla_impl`, `moe_combine`, `moe_ep_merge`, `collective`,
`fusion`, `backward` (`not_implemented`), `regions_sha256`, `sampling_sha256`.

Two things are worth stating explicitly.

**Projection.** `rms_eps` and `rotary_theta` describe both the mathematical
function and its realization, so they appear in *both* identities: the declared
value in `semantic_id`, the FP32-rounded value the kernels actually read as
`effective_rms_eps` / `effective_rope_theta` in `numerical_policy_id`. Deleting one
of the two would silently omit a constant rather than resolve it.

**Enabling replicated placement is a numerical change.** TP/EP split the heads and
reduce across devices, so the collective and the expert partial merge enter the
numerical policy at `tp_size`/`ep_size > 1`. That is why `test_tp.py` gates on
tokens plus an RMS band and not on bitwise equality: the reduction order genuinely
differs (`docs/design.md`, "Tensor-parallel placement").

**Unknown settings are not defaults.** `gemm_algorithm_policy` says *unpinned*:
cuBLAS picks its own algorithm and workspace, so an exact claim about the GEMM
region is unsupported until Stage 2 pins or replaces it. The field records that
instead of hiding it.

### `deployment_id` — where it ran and how memory moved

`placement` (`layer_split` / `replicated_tp_ep`), `devices` (CUDA ordinals),
`layer_device` (the per-layer owner), `ep_size`, `declared_tp_rank`,
`declared_ep_rank`, `role_shards`, `allocations`, `transfer`.

Device *identity* (UUID, capability) is provenance, not deployment: the placement
is a choice, the device is a fact about the run.

### Weights identity

`weights.parameter_manifest_sha256` is the SHA-256 of a canonical tensor index —
name, dtype, shape and byte count per tensor in sorted name order, plus the sorted
shard basenames (not their mount paths) — computed once at `engine_create`.
`weights.content_sha256` is `null` unless a caller asked for the raw bytes to be
hashed: hashing a 50 GiB checkpoint is minutes of I/O, and "not hashed" is a
different statement from "different content". Both are reported; `null` is never
treated as equal or as unequal by itself — strict admission compares the
parameter-manifest digest.

### Provenance

`provenance.build.*` (from the generated header) and `provenance.runtime.*`
(CUDA runtime, driver, cuBLAS, per-device facts). `kernel_path` and
`triton_cubin_arch` are the *selection rule* applied to the build's target lists
plus the device's capability, not an observation of the launched binary — which is
not queryable per kernel. The field name says `kernel_path`, "selected".

Four spellings mean *not established*: `unknown`, `unavailable`, `unsupported`,
`unspecified`. A strict comparison refuses them **even when both sides agree on
one**, because agreeing on an unknown does not establish a fact. That is what
makes a CPU-only or un-provisioned build unable to pass strict admission — by
design.

## Region registry and determinism

`regions` binds each region to the implementation that ran it and records, per
region, `determinism` (`deterministic`, `unverified`, `not_implemented`),
`mechanism` and `rng_dependency`. The two axes are separate: a seed does not order
atomics, and ignoring measured results in GDN and BF16 output is not the same as
confirmed measurement.

The table is deliberately conservative. `deterministic` is claimed only where the
implementation is a single elementwise pass with no cross-thread reduction
(embedding gather, RoPE, Q/gate de-interleave, the output gate, residual add,
SiLU-multiply, the FP32 MoE combine, the host argmax). Everything whose reduction
or tiling order comes from a library (FlashInfer, cuBLAS, the AOT FLA cubins) or
crosses devices is `unverified`, and the backward regions are `not_implemented`
rather than assumed. Stage 2's experiments are what would establish the rest.

## Sampling

The generation-only selection policy is a numerical-policy field: the `sampling`
block carries `mode` (`greedy`), `transform` (`argmax_lowest_token_id`),
`transform_version`, `arithmetic` and `rng` (`none`), and its digest
(`numerical_policy.fields.sampling_sha256`) is part of the numerical identity — a
capture taken under a different sampler arithmetic is a different numerical policy,
and a comparison refuses it rather than overlooking the field. Greedy consumes no
random word, so there is nothing to record per request. When the temperature
sampler of [plan-numeric-contract.md](plan-numeric-contract.md) (T0–T4) lands, its
transform and arithmetic join this block, while the resolved temperature and seed
of one request belong to the capture's replay data — so **changing only the
temperature changes no model, weight or policy identity**.

## Comparison modes

| Mode | Admission | Exit |
|---|---|---|
| `strict` (default) | semantic, numerical-policy and parameter identities equal; placement equal unless a scoped claim is declared; provenance established and equal; numeric arrays bitwise identical (`max_abs == 0`) | 0 admitted (1 otherwise) |
| `strict --deployment-scoped` | as above, but a placement-only difference (including `descriptor.sha256`, whose canonical text carries the placement fields) is reported as a *scoped exception*, never as an identity-level pass | 0 |
| `diagnostic` | a deliberate manifest difference is reported for attribution and is explicitly *not* a contract pass | 3 |
| legacy | one document has no `manifest_version`: the numeric arrays stay comparable, no identity is established and none is invented | 2 |

The descriptor reference is scoped rather than an identity because the descriptor's
canonical text contains `tp_size`/`ep_size`/`role_shards`; classifying it as an
identity would make a placement-only change indistinguishable from a semantic one.
A capture is only comparable under `strict` when it was taken the same way — a
different GPU, driver, toolkit, build or set of shards is not a platform difference
to be waved through.

The numeric gate is unchanged and absolute: a manifest verdict never loosens it,
and a bitwise failure is a failure in every mode.

## Where it is implemented and tested

| Piece | File |
|---|---|
| SHA-256, canonical writer, identities, region registry | `csrc/sha256.c`, `csrc/manifest.c`, `csrc/include/manifest.h` |
| Build provenance generation | `csrc/gen_build_info.cmake`, `csrc/triton/build_aot.py` |
| Engine query, descriptor/parameter digests, device facts | `csrc/engine.cu`, `csrc/include/engine.h` |
| FFI, parse, diff, admission modes, CLI | `src/Infer/FFI/Engine.hs`, `src/Infer/Manifest.hs`, `src/Main.hs` |
| Gates | `ctest test_manifest`, `ctest test_manifest_hashes` (CPU), `tests/ManifestSpec.hs` (`cabal test infer-tests`), `tests/test_manifest_compare_cli.py`, `tests/capture_logits.py` |
