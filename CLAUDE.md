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
  witness on real circuits. `@constrain` is erased during lowering, so GPU code
  computes witnesses with no internal alarm — circom is the only catch for a
  miscompile.
- **Frontend-agnostic target.** LLZK is the stable contract; Circom is one
  producer. Do not leak Circom-specific assumptions into LlzkToStablehlo passes.

## Pipeline Overview

```
Circom (.circom)
   |  circom --llzk concrete --llzk_plaintext --stabilize
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
loops, 2D carry while, and nested-while inner loops. Benchmarks and the exact
rewrite shapes are in [`docs/BATCH_STABLEHLO.md`](docs/BATCH_STABLEHLO.md).

## Why the Pipeline Looks This Way

Four splits look arbitrary in code but exist because merging would break
correctness:

- **Pod dispatch elimination is mandatory, not optional cleanup.** A single
  Circom `lt.in[0] <== v1; lt.in[1] <== v2` compiles to ~40 lines of LLZK state
  machine (input-counter + `!pod.type<[...]>` pending-inputs record + delayed
  `function.call`). Conversion patterns can't reliably match across this
  boilerplate, so `SimplifySubComponents` must flatten it first.
- **`SimplifySubComponents` runs to a fixed point because component nesting is
  arbitrary.** Each pod layer must be peeled before the next becomes
  pattern-matchable. Phases that early-return for internal-state safety (e.g.
  `unpackPodWhileCarry` — pointer invalidation when its inner SmallVector
  contains a chained-while it erased inline) need their own inner
  `while (phase(block))` loop at the driver site if a *same-iteration* later
  phase consumes their complete output. The outer fixed point is too coarse:
  with N independent candidates only the first gets processed before destructive
  later phases (`eliminatePodDispatch` Phase 5) run.
- **While-loop transformation is four phases because LLZK is mutable, StableHLO
  is SSA, and loop bodies can mutate outer arrays.** `array.write %outer[%i]`
  inside `scf.while` must flow through carry tuples to lower to a functional
  `stablehlo.while`. The phase ordering (array-to-carry promotion → SSA-ify
  writes → main conversion → `scf.while → stablehlo.while`) is forced. The
  shared walker `processBlockForArrayMutations` is reused with different
  `latest` trackers per pre-pass — it MUST gate mutating ops on the target being
  currently tracked, or untracked writes get rewritten orphan and silently
  DCE'd.
- **Post-passes exist because `applyPartialConversion` does 1:1 op replacement,
  not region restructuring.** Rewriting `scf.while` → `stablehlo.while`,
  reconnecting `func.call` results to `pod.read @comp` consumers, and
  vectorizing independent while loops all move/delete regions — partial
  conversion can't express that.

## LLZK as a Moving Contract

LLZK is versioned upstream and changes break us. The full catalogue of
silent-miscompile and hang traps — each with reproduction, diagnostic, and the
landed fix — lives in [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md),
organized by where the bug fires:

- **Walker traps** — `scf.while`/`scf.if`/`scf.execute_region` array carry,
  passthrough detection, body-arg SSA-ification. See
  [Walker traps](docs/LOWERING_PITFALLS.md#walker-traps).
- **SimplifySubComponents driver-ordering traps** — struct-of-pods materializer
  invariants, `eliminatePodDispatch` Phase 1/4/5 hoister and erasure ordering,
  `processNested` recursion guards, SSC convergence with the outer fixed-point
  loop. See
  [SimplifySubComponents driver-ordering traps](docs/LOWERING_PITFALLS.md#simplifysubcomponents-driver-ordering-traps).
- **Pod-dispatch silent miscompile signals** — surviving pod-typed iter-args,
  sub-call count diffs, multi-carry stale operands, writerless dispatch pods.
  See
  [Pod-dispatch silent miscompile signals](docs/LOWERING_PITFALLS.md#pod-dispatch-silent-miscompile-signals).
- **Upstream-LLZK contract drift** — circom PR #378's same-named template wrap,
  `<[]>` vs no-params, `llzk.nondet : index`, test-fixture and BUILD-glue
  migration rules. See
  [Upstream-LLZK contract drift](docs/LOWERING_PITFALLS.md#upstream-llzk-contract-drift).
- **MLIR C++ API gotchas** — `arith.cmpi`/`cmpf` predicates live in
  `properties`, not the discardable attribute dict; `gentbl_cc_library`
  include-set hygiene; consolidate-the-`tblgen`-rule advice. See
  [MLIR C++ API gotchas](docs/LOWERING_PITFALLS.md#mlir-c-api-gotchas).
- **APInt arithmetic traps** — `getSExtValue` UB on `getBitWidth() > 64`,
  `operator==` bit-width-mismatch assertion. See
  [APInt arithmetic traps](docs/LOWERING_PITFALLS.md#apint-arithmetic-traps).
- **Diagnostic foot-guns** — `awk`-slicing a `stablehlo.while` body, m3 gate
  repro without `BUILD.bazel` edits. See
  [Diagnostic foot-guns](docs/LOWERING_PITFALLS.md#diagnostic-foot-guns).

Two quick rules worth remembering at every keystroke:

- **Verify against the lowered StableHLO, NOT the simplified LLZK.** Many
  LLZK-level "fixes" are absorbed by downstream conversion phases; the bytes
  that matter are the ones GPU runs. When the LLZK changes but the lowered MLIR
  is identical, the upstream fix didn't earn its keep.
- **Structural IR cleanup ≠ runtime fix.** Closing the iter-arg chain at
  SimplifySubComponents (lowered `func.call (%cst, %cst)` count → 0) is
  necessary but NOT sufficient. Always re-run the GPU correctness gate AFTER
  each structural improvement. Structural metrics improving without a runtime
  delta means the data-flow disconnect is downstream (`LlzkToStablehlo` main
  conversion, vectorization, or `BatchStablehlo`).

For already-landed fixes, `git blame` plus the per-trap
`~/.claude/knowledge/llzk-to-shlo-*.md` knowledge notes carry the full
implementation history.

See [`docs/CIRCUIT_COVERAGE.md`](docs/CIRCUIT_COVERAGE.md) for how a
frontend/LLZK mismatch surfaces at the user-visible level (per-circuit
pass/fail, per-stage error categories).

## Load-Bearing Invariants

Cross-cutting assumptions the pipeline relies on. Each can be violated by an
innocent-looking change without breaking the build — the result still compiles,
runs on GPU, and silently produces wrong witnesses. The full list with rationale
lives in [`docs/load-bearing-invariants.md`](docs/load-bearing-invariants.md),
grouped into:

- Correctness gate hierarchy (sister-circuit families aren't differential
  references; `<==` vs `<--` is a security boundary; `@constrain` is erased at
  lowering).
- Batching invariants (`BatchStablehlo` must see every signal; trip counts must
  be compile-time-constant; immutable circom signals are what makes
  vectorization sound).
- Lowering equivalences (`array.new` and `llzk.nondet` both lower to
  `dense<0>`).
- Witness layout & verifier asymmetries (`getMemberFlatSize` recursion,
  `MemberReadOp` pub-from-outside rule, `--witness-layout-anchor` ordering).

## Conventions

Project-wide conventions grouped by topic file under
[`docs/conventions/`](docs/conventions/). The umbrella index lives at
[`docs/conventions.md`](docs/conventions.md); jump straight into:

- [`docs/conventions/m3-fixture.md`](docs/conventions/m3-fixture.md) — JSON
  schema, positional ordering, `@main` arg order with `public [...]`.
- [`docs/conventions/m3-correctness-gate.md`](docs/conventions/m3-correctness-gate.md)
  — `.json.gate` format, `.wtns` layout caveats, `circom --O0` rule, gate
  enrollment, cache invalidation, `zkx_*` flag prefix.
- [`docs/conventions/docs-style.md`](docs/conventions/docs-style.md) —
  `mdformat-footnote` is NOT installed; use inline numeric markers.

## Top-level docs

- [E2E Lowering Guide](docs/E2E_LOWERING_GUIDE.md) — how each LLZK op lowers to
  StableHLO (type conversion, pod elimination, while loop transformation,
  post-passes).
- [Batch StableHLO](docs/BATCH_STABLEHLO.md) — IR-level batching: leading batch
  dimension, op-by-op rules, data-dependent indexing, GPU benchmarks.
- [Witness Layout Anchor](docs/WITNESS_LAYOUT_ANCHOR.md) — the
  `--witness-layout-anchor` / `--verify-witness-layout` dialect + pass contract.
- [Circuit Coverage](docs/CIRCUIT_COVERAGE.md) — full 123-circuit analysis
  (pass/fail per stage, error categories, affected circuit families).
- [GPU Profiling](docs/GPU_PROFILING.md) — Nsight Systems profiling of kernel
  launches, memory transfers, batch vs sequential.
- [CI Setup Guide](docs/CI_SETUP_GUIDE.md) — self-hosted runner configuration;
  **circom is built via Nix flake**, not apt — MLIR 20 dev libraries come from
  the flake.
- [llzk-status](llzk-status.md) — current conversion/GPU/LIT/semantic counts and
  known limitations.

## External sources of truth

- [project-llzk/llzk-lib](https://github.com/project-llzk/llzk-lib) — LLZK
  dialect definitions (pinned in `third_party/llzk/workspace.bzl`).
- [project-llzk/circom](https://github.com/project-llzk/circom), `llzk` branch —
  Circom frontend with the LLZK backend.
- [fractalyze/stablehlo](https://github.com/fractalyze/stablehlo) — StableHLO
  dialect, ZK fork with prime field types.
- [fractalyze/open-zkx](https://github.com/fractalyze/open-zkx) —
  `stablehlo_runner` GPU execution target.
