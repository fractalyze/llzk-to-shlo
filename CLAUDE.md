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
  with circom, circom wins — circom is the source of truth. This matters because
  `@constrain` functions are erased during lowering (see "Load-Bearing
  Invariants"); GPU code only computes witnesses and can never self-verify, so a
  miscompile in `@compute` would produce "self-consistent" wrong output with no
  internal alarm. An external reference (circom) is the only catch.
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

## Why the Pipeline Looks This Way

The pipeline's shape (two passes, fixed-point iteration, four-phase while
lowering, a separate post-pass phase) is not stylistic — each split exists
because merging the phases would break correctness. Four decisions that look
arbitrary in the code but aren't:

- **Pod dispatch elimination is mandatory, not optional cleanup.** A single
  Circom line like `lt.in[0] <== v1; lt.in[1] <== v2` compiles to ~40 lines of
  LLZK state machine (an input-counter, a `!pod.type<[...]>` pending-inputs
  record, and a delayed `function.call` fired only when all inputs have
  arrived). Conversion patterns in `LlzkToStablehlo` cannot reliably match
  across this boilerplate, so `SimplifySubComponents` must flatten it into
  direct `function.call` first. This is why we have two passes — removing
  `SimplifySubComponents` isn't a speed win, it silently breaks conversion.
- **`SimplifySubComponents` runs to a fixed point because component nesting is
  arbitrary.** `GreaterThan` calls `LessThan` calls `Num2Bits` — each pod layer
  has to be peeled before the next becomes pattern-matchable. The pass is seven
  internal phases (−1, 0, 1–5) inside a `repeat-until-no-change` loop; dropping
  the outer loop compiles fine on toy circuits and fails on multi-level ones.
- **While-loop transformation is four phases because LLZK is mutable, StableHLO
  is SSA, and loop bodies can mutate outer arrays.** A Circom pattern like
  `signal bits[N]; for (i=0..N) { bits[i] <-- …; }` lowers to LLZK with
  `array.write %outer[%i]` *inside* `scf.while`. StableHLO's `while` is purely
  functional — all mutation must flow through carry tuples. The four phases
  (array-to-carry promotion → SSA-ification of writes → main conversion →
  `scf.while → stablehlo.while`) bridge this gap; the ordering is forced, not
  chosen. Nested loops additionally have to skip values defined in a parent
  while body during capture detection, or promotion introduces a domination
  violation.
- **Post-passes exist because `applyPartialConversion` does 1:1 op replacement,
  not region restructuring.** The main pass handles `felt.add → stablehlo.add`
  cleanly. Rewriting `scf.while` into `stablehlo.while`, reconnecting
  `func.call` results back to `pod.read` consumers, and vectorizing independent
  while loops all require moving or deleting regions — which partial conversion
  can't express. That's why post-passes are a distinct phase, not "optional
  optimizations."

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
  `applyPartialConversion`. Strip on the uncovered ops; don't strip on the
  covered ones (that desyncs upstream's bookkeeping).
- **`llzk.nondet : index` is dialect-conversion-illegal.** The conversion target
  only legalizes nondet for felt / array / struct kinds. When scrubbing residual
  `pod.read` ops whose result type is `index` (the dispatch-pod `@count`
  countdown is the canonical case), substitute `arith.constant 0 : index`
  instead — the surrounding cmpi/scf.if scaffold is structurally dead once
  `resolveArrayPodCompReads` has hoisted the `function.call`, so 0 keeps the
  cmpi false and DCE collapses the dead branch.
- **`replaceRemainingPodOps` (Phase 5) clobbers `unpackPodWhileCarry`'s field
  discovery.** Phase 5 nondets every `pod.read` in a block — including reads of
  pod-typed `scf.while` block args, which is the field-discovery input the next
  outer fixed-point iteration of `unpackPodWhileCarry` needs. Gate
  `eliminatePodDispatch` on the block having no pod-typed block args; once the
  carry is unpacked, the args become non-pod and dispatch elimination proceeds
  normally. Symptom of getting this wrong: multi-record input pods (e.g. keccak
  `<[@a: array, @b: array]>` carries) survive into dialect conversion and fail
  `pod.new` legalization.

See [`docs/CIRCUIT_COVERAGE.md`](docs/CIRCUIT_COVERAGE.md) for how a
frontend/LLZK mismatch surfaces at the user-visible level (per-circuit
pass/fail, per-stage error categories).

## Load-Bearing Invariants

Five cross-cutting assumptions the pipeline relies on. Each one can be violated
by an innocent-looking change without breaking the build — the result still
compiles, still runs on GPU, and silently produces wrong witnesses. Treat these
as load-bearing; test changes in the neighborhood against circom's C++ witness,
not against the lowered IR alone.

- **Circom's `<==` vs `<--` is a security boundary, not a style choice.** `<==`
  emits both `@compute` (witness computation) and `@constrain` (the soundness
  check). `<--` emits only `@compute` and *requires* a separate explicit
  constraint elsewhere — e.g. `Num2Bits` must add `out[i] * (out[i] - 1) === 0`
  after `out[i] <-- (in >> i) & 1`. Dropping the follow-up constraint does not
  fail any test in this repo — the witness validates against itself — it just
  makes the circuit unsound in the prover.
- **`@constrain` functions and all `constrain.eq` ops are erased during
  lowering.** GPU code only runs witness generation; constraint satisfaction is
  the prover's job downstream. The lowered StableHLO has no way to catch a
  broken constraint, which is why circom is the correctness gate (see "Design
  Philosophy").
- **Batched witness output must include every signal — public outputs *and*
  private internals.** `BatchStablehlo`'s leading `N` dimension aliases the full
  per-witness state across `N` proofs. Pruning the output to "just the public
  signals" looks like a clean optimization and silently misaligns the batch
  axis.
- **Circom while loops have compile-time trip counts; all batch elements iterate
  the same number of times.** `BatchStablehlo` extracts the loop predicate from
  element `[0]` and reuses it for the whole batch. A future frontend that emits
  batch-divergent trip counts would break this — there is no runtime fallback
  for per-lane divergent loops.
- **Circom signals are immutable, which is what makes vectorization sound.**
  Auto-vectorization turns `while (i<N) { out[i] = f(a[i]) }` into `out = f(a)`
  only because `out[i]` can't be reassigned — the canonical iterative pattern is
  `chain[N][K+1]`-style arrays, not mutable accumulators. A future frontend that
  lowers mutable iteration through LLZK would need the vectorization phase
  disabled, not extended.

## Conventions & Background

### Markdown footnotes in docs/

The pre-commit `mdformat` hook runs with `mdformat-gfm` and
`mdformat-frontmatter` only — **no** `mdformat-footnote` plugin. Raw GFM
footnote definitions (`[^id]: text`) are escaped to `\[^id\]: text` on
autoformat, breaking rendered references. When a docs page (e.g.
`docs/M3_REPORT.md`) needs footnote-like notes, use **inline numeric markers**
(`¹` `²` … `⁶` …) in the cell or sentence and a "Notes:" sub-list immediately
under the table. The first M3 report fill (`aes_256_encrypt` rows in §4)
established this convention; mirror it for new rows. If proper footnotes become
necessary, add `mdformat-footnote` to `.pre-commit-config.yaml` `mdformat`
`additional_dependencies` first, then switch styles in one PR.

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
