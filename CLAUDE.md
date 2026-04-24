# llzk-to-shlo

## Design Philosophy

llzk-to-shlo lowers [LLZK](https://github.com/project-llzk/llzk-lib) circuit IR
into [StableHLO](https://github.com/fractalyze/stablehlo) so that ZK witness
generation can ride on the same ML compiler infrastructure everyone else already
optimizes. The end of the pipeline is GPU execution through
[open-zkx](https://github.com/fractalyze/open-zkx)'s `stablehlo_runner`, which
gives batched witness generation a single kernel launch instead of one per
proof.

**Core principles:**

- **Reuse ML compiler infrastructure.** StableHLO is already a well-supported IR
  with GPU backends, fusion, and batching. Every design choice prefers
  expressing ZK primitives as StableHLO ops over rolling our own runtime.
- **Batch first.** Witness generation is embarrassingly parallel across proofs.
  Every lowering must survive a leading batch dimension being added by
  `BatchStablehlo`; per-op bridging rules live in
  [`docs/BATCH_STABLEHLO.md`](docs/BATCH_STABLEHLO.md).
- **GPU correctness is the gate.** LIT + unit tests prove IR shape; the real
  correctness signal is `batch[i] == single[i]` against circom's native C++
  witness on real circuits. When a lowering looks right on paper but disagrees
  with circom, circom wins — circom is the source of truth.
- **Frontend-agnostic target.** LLZK is the stable contract; Circom is one
  producer. Do not leak Circom-specific assumptions into LlzkToStablehlo passes.

## Pipeline Overview

```
Circom (.circom)
   |  circom --llzk concrete [--llzk_plaintext]
   v
LLZK IR (.llzk)
   |  llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo
   v
StableHLO IR (.mlir)
   |  llzk-to-shlo-opt --batch-stablehlo="batch-size=N"
   v
Batched StableHLO IR (.mlir)
   |  open-zkx stablehlo_runner (GPU)
   v
N witnesses from one kernel launch
```

Two passes sit at the core:

- **SimplifySubComponents** removes pod dispatch patterns and flattens component
  calls into direct `function.call`, making the IR legible to the type-aware
  conversion pass that follows.
- **LlzkToStablehlo** is the heavy pass — it runs pre-passes (input-pod
  elimination, while-carry promotion, SSA-ification of `array.write`,
  `array.insert`, and `struct.writem`), the main partial conversion (LLZK op
  patterns → StableHLO), and post-passes (`scf.while` → `stablehlo.while`,
  `scf.if` → `stablehlo.select`, reconnecting `func.call` results to
  `pod.read @comp` consumers, residual LLZK cleanup, arith → stablehlo, DCE,
  while loop vectorization).

Three vectorization phases are applied after conversion — independent while
loops, 2D carry while, and nested-while inner loops — each turning serial
row-by-row computation into parallel tensor ops. Benchmarks and the exact
rewrite shapes are in [`docs/BATCH_STABLEHLO.md`](docs/BATCH_STABLEHLO.md).

## LLZK as a Moving Contract

LLZK is versioned upstream and changes break us. Two things that have bitten us
before and will again:

- **Test fixtures are consumer-owned IR.** Hand-written `.mlir` test files live
  in this repo and get parsed directly. When LLZK changes IR syntax (e.g. v1.x
  `struct.def @Foo<[]>` → v2.0 `struct.def @Foo`), the automatic IR migrator
  upstream does not touch parser input — consumer fixtures must be hand-migrated
  in the same bump.
- **BUILD glue must move with upstream.** New `.td` files in LLZK (new dialects,
  new interfaces) do not appear in `third_party/llzk/llzk.BUILD` automatically.
  When bumping, diff `include/llzk/Dialect/**/*.td` against the BUILD's
  `gentbl_cc_library` targets and add any missing inc-gen rules; the build will
  fail late with a missing `.h.inc` include otherwise.
- **`createEmptyTemplateRemoval` uses `applyFullConversion` over a narrow op
  list.** The pass's conversion target only handles ops in
  `OpClassesWithStructTypes` (struct/array/function/global/constrain/
  polymorphic — see `lib/Dialect/Polymorphic/Transforms/SharedImpl.cpp`).
  Anything else — `pod.*` ops, `llzk.nondet` results with struct types, our own
  synthesized ops — must be **gone or already in stripped form** before this
  pass runs, or its legality walk fails (or worse, crashes if our pre-strip
  leaves the IR in a half-stripped state). When you write or reorder a pre-pass,
  the rule is: clean residual pod traffic first, then pre-strip `<[]>` only on
  ops *outside* that tuple, then run template removal. See
  `SimplifySubComponents.cpp` for the canonical ordering.
- **`<[]>` (empty params) vs no params on `!struct.type`.** Template removal
  rewrites `<[]>` to no-params on the ops it covers but leaves SSA values on
  uncovered ops alone (`llzk.nondet`, `scf.while` block args, etc.). Mixing
  forms on either side of a use-def edge produces unresolved
  `builtin.unrealized_conversion_cast` errors at the next
  `applyPartial- Conversion`. Strip on the uncovered ops; don't strip on the
  covered ones (that desyncs upstream's bookkeeping).

See [`docs/CIRCUIT_COVERAGE.md`](docs/CIRCUIT_COVERAGE.md) for how a
frontend/LLZK mismatch surfaces at the user-visible level (per-circuit
pass/fail, per-stage error categories).

## Conventions & Background

Top-level docs:

- [E2E Lowering Guide](docs/E2E_LOWERING_GUIDE.md) — how each LLZK op lowers to
  StableHLO (type conversion, pod elimination, while loop transformation,
  post-passes)
- [Batch StableHLO](docs/BATCH_STABLEHLO.md) — IR-level batching: leading batch
  dimension, op-by-op rules, data-dependent indexing, GPU benchmarks
- [Circuit Coverage](docs/CIRCUIT_COVERAGE.md) — full 123-circuit analysis
  (pass/fail per stage, error categories, affected circuit families)
- [GPU Profiling](docs/GPU_PROFILING.md) — Nsight Systems profiling of kernel
  launches, memory transfers, batch vs sequential
- [CI Setup Guide](docs/CI_SETUP_GUIDE.md) — self-hosted runner configuration;
  **circom is built via Nix flake**, not apt — MLIR 20 dev libraries come from
  the flake
- [llzk-status](llzk-status.md) — current conversion/GPU/LIT/semantic counts and
  known limitations

External sources of truth:

- [project-llzk/llzk-lib](https://github.com/project-llzk/llzk-lib) — LLZK
  dialect definitions (pinned in `third_party/llzk/workspace.bzl`)
- [project-llzk/circom](https://github.com/project-llzk/circom), `llzk` branch —
  Circom frontend with the LLZK backend
- [fractalyze/stablehlo](https://github.com/fractalyze/stablehlo) — StableHLO
  dialect, ZK fork with prime field types
- [fractalyze/open-zkx](https://github.com/fractalyze/open-zkx) —
  `stablehlo_runner` GPU execution target
