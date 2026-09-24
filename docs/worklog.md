# Work Log

A running snapshot of what this project can do today, what has been verified and
where it is knowingly incomplete. [README.md](../README.md) describes the
component layout and how to build and test; [design.md](design.md) holds the
architecture rationale; [plan-numeric-contract.md](plan-numeric-contract.md) is
the proposal for the trainer, RL and inference-optimization work that is **not**
implemented here.

Last updated: 2026-09-25.

## Verified today

Everything below was run and passed on 2× A40 46 GB (sm_86) unless a line says
otherwise. No GPU result here is implied by a document edit alone.

| Gate | Result |
|---|---|
| `cargo test --locked --offline` | 11/11 |
| `ctest --test-dir csrc/build-libs` | 13/13 — `test_model_desc`, `test_safetensors`, `test_manifest`, `test_manifest_hashes`, `test_engine_resources`, `test_collective`, `test_attention`, `test_gdn`, `test_moe`, `test_mla`, `test_norm`, `test_rope`, `test_library_ops` |
| `cabal test all --enable-tests` | `infer-tests` 66/66, `infer-generation-tests` 15/15 |
| `manifest --model-dir <27B> --gpus 0,1 --check` | exit 0: a 12216-byte canonical document carrying all three identities plus the parameter identity, build/runtime provenance fully established, two queries byte-identical, and every digest re-derived independently by `tests/manifest_check.py` |
| Two independent 27B captures, strict comparison | bitwise identical (`max_abs == 0`) and verdict `admitted` |
| Pre-refactor golden vs a fresh 27B capture | bitwise identical (`max_abs == 0`) with verdict `legacy/unverified` (exit 2) — the older capture's numeric arrays are compared, but nothing about its identity is invented |
| Qwen3.8-27B golden capture | bitwise identical to the pre-refactor baseline (`max_abs == 0`) |
| `tests/test_engine.py` (27B vs independent PyTorch logits) | 20/20 greedy tokens; logit RMS 0.02–0.04 |
| `tests/test_longseq.py` | 433-token chunk-split self-consistency (RMS 0.029) and 128-token generation coherence |
| `tests/test_tp.py --devices 0,1` (TP2) | identical greedy tokens, per-step logit RMS ≤ 0.05 |

The manifest rows ran on the real 27B with `semantic_id`
`890c5472…fabde`, `numerical_policy_id` `b0da2057…d1c551`, `deployment_id`
`56f6e4aa…d6f4d` and `parameter_manifest_sha256` `4cb768d4…bbcc` over 1199
tensors (device ordinals 0,1; 32 layers each; layer-split placement), from a build
whose own revision the manifest reports as `1f23880`.

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
  region, determinism/mechanism/RNG-dependency separately (24 regions: 9
  deterministic by construction, 13 unverified because a library or cross-device
  reduction order is not established, 2 `not_implemented` — every backward region).
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

- **The manifest's per-device `kernel_path` and `triton_cubin_arch` are a
  selection rule, not an observation.** Which binary the driver actually launched
  is not queryable per kernel, so the manifest reports what the build's target
  lists plus the device's compute capability select for it, and the field names say
  `selected`.
- **No region has an established reduction order beyond the elementwise ones.** The
  registry marks 13 of 24 regions `unverified`, and the GEMM algorithm policy is
  recorded as `cublas_default_heuristic_unpinned`: an exact claim about that region
  is unsupported until the plan's Stage 2 pins or replaces it.
- **`weights.content_sha256` is null** unless a caller chooses to hash 50 GiB of
  tensor data; the parameter-manifest digest over the tensor index is what strict
  admission compares.
- **Backward regions do not exist**, so the registry records them as
  `not_implemented` rather than assuming a determinism verdict for them.
- **Expert-parallel equivalence re-run is pending.** The EP-vs-layer-split check
  after the FP32 merge landed was stopped before it finished. The gate is
  unchanged (`test_tp.py --ep 2`: identical greedy tokens, logit RMS ≤ 0.05).
- **sm_90a has no runtime gate** — compile- and artifact-verified only.
- **Long context is not supported.** The MLA attention kernel's shared-memory
  budget caps the cached sequence length (a 16K context does not fit) and both
  `engine_create` and the kernel entry reject anything longer.
- **Greedy decoding only.** There is no temperature/seed sampling, no top-k/top-p
  and no batching; a CPU sampler migration is proposed in
  plan-numeric-contract.md (milestones T0–T4) but not implemented.
- **Performance is not optimized** (correctness-first): decode is a single-token
  full forward, and the logits are computed for the last position only.

## Documents

| File | Role |
|---|---|
| [README.md](../README.md) | Build, test entry points, usage, phase status, model weights |
| [design.md](design.md) | Architecture rationale, per-family layout differences, testing strategy |
| [manifest-contract.md](manifest-contract.md) | The execution manifest: canonical encoding, field ownership and projections, the region determinism registry, and the comparison modes |
| [plan-numeric-contract.md](plan-numeric-contract.md) | **Proposal, not implemented** (Stage 0 is implemented; see above) — trainer (SFT/OPD/GRPO/DAPO/GSPO/PPO), bounded-staleness async RL, temperature-sampling migration, and an inference-optimization track (fusion, W4A16, speculative decoding) |
| [reference-output.json](reference-output.json) | Transformers reference tokens for the 27B debugging prompt |
