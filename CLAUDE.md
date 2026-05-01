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
  Phases that early-return for internal-state safety (e.g. `unpackPodWhileCarry`
  returns after one whileOp to avoid pointer invalidation when its inner
  SmallVector contains a chained-while it erased inline) must be wrapped in
  their own inner `while (phase(block))` loop at the driver site if a
  *same-iteration* later phase consumes their complete output. The outer fixed
  point is too coarse: with N independent candidates, only the first gets
  processed before the destructive later phase runs (`eliminatePodDispatch`
  Phase 5 nondets cross-block readers before iter 2 can rerun the materializer
  for siblings). PR #32 hit this for chi (25 dispatch pods) — same shape recurs
  whenever a function fans out to many independent sub-component instances.
- **While-loop transformation is four phases because LLZK is mutable, StableHLO
  is SSA, and loop bodies can mutate outer arrays.** A Circom pattern like
  `signal bits[N]; for (i=0..N) { bits[i] <-- …; }` lowers to LLZK with
  `array.write %outer[%i]` *inside* `scf.while`. StableHLO's `while` is purely
  functional — all mutation must flow through carry tuples. The four phases
  (array-to-carry promotion → SSA-ification of writes → main conversion →
  `scf.while → stablehlo.while`) bridge this gap; the ordering is forced, not
  chosen. Nested loops additionally have to skip values defined in a parent
  while body during capture detection, or promotion introduces a domination
  violation. The shared walker `processBlockForArrayMutations` is called by
  multiple pre-passes (`convertArrayWritesToSSA`, `convertWhileBodyArgsToSSA`,
  `liftScfIfWithArrayWrites`) with different `latest` trackers each time — it
  MUST gate mutating ops on the target being currently tracked. Eagerly
  rewriting an untracked write to result-bearing form leaves an orphaned op
  whose yield is never re-routed; downstream DCE silently erases it, dropping
  the write. PR #30 added that gate; widening the tracker on the caller side
  would also work but is harder to keep correct across passes.
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
- **Pod-array iter-arg survival post-simplify is a silent miscompile signal.**
  Run `bazel run //tools:llzk-to-shlo-opt -- --simplify-sub-components <input>`
  and `grep -nE "scf.while.*x !pod"` the output. Any surviving pod-typed
  `scf.while` carry means `flattenPodArrayWhileCarry` skipped that loop —
  usually because (a) the source array is multi-dim and the per-field type
  builder dropped inner dims, or (b) the carrier is nested deeper than 1
  block-arg-chain hop, so pod.read/pod.write field discovery returned empty.
  Downstream Phase 5 nondets the cross-iteration `pod.read [@a]` reads, the
  resulting `function.call @Sub::@compute(%nondet, %nondet)` lowers to
  `XOR(0,0) = 0`, and the parent struct.member's witness slot fills with zeros
  that look correct on paper. AES `@xor_2` (4992 felts, 4-level nest) is the
  canonical case. Pair this grep with the lowered StableHLO grep
  `func.call @<Sub>_<Sub>_compute(%cst.*=0` — both should be empty for a
  cleanly-flattened circuit.
- **Dispatch pod emitted as `llzk.nondet : !pod.type<[@count, @comp, @params]>`
  instead of `pod.new {@count = const_N}` is a silent miscompile.** Earlier
  circom-llzk emits initialized
  `%c3 = arith.constant 3 : index; %pod = pod.new {@count = %c3} : <[@count, @comp, @params]>`.
  After project-llzk/circom PR #390 era (2026-04-30+) the same pod can come in
  as
  `%nondet = llzk.nondet : !pod.type<[@count: index, @comp: !struct.type<...>, @params: !pod.type<[]>]>`.
  Inside the input-collection scf.while body, `pod.read [@count]` then yields a
  garbage `index`, `cmpi eq 0` is never true, the buried
  `function.call @<Sub>::@compute(...)` inside the scf.if branch never fires,
  and `pod.read [@comp]` returns a nondet struct whose `@out` reads zero —
  `@main`'s structural witness slot fills with const-0. PR #48
  (`erase dead llzk.nondet during conversion for empty pod types`) handled the
  trivial empty-pod variant; the dispatch variant is a separate fix.
  **Diagnostic recipe**: post-`--simplify-sub-components` lowered StableHLO
  `@main` grep — if `@main` body has only `stablehlo.constant ... 0` ops +
  reshapes + dynamic_update_slice with no `func.call`, the dispatch elimination
  silently dropped the call. (The
  `llzk.nondet : !pod.type<[@count, @comp, @params]>` pod-creation op may still
  survive post-simplify even when fixed — what matters is whether the @comp
  readback resolves to a real `function.call`, not whether the dispatch nondet
  itself was DCE'd.) Canonical case at memo time: `iden3_get_subject_location`
  2026-04-30. **Fix shape (landed PR-TBD 2026-04-30)**: in
  `SimplifySubComponents.cpp:materializeScalarPodCompField`, extend the
  candidate filter to include `llzk.nondet` alongside `pod.new`. The rest of the
  helper (writer-while detection, post-while iter-arg projection, cross-block
  @comp reader replacement) is definer-agnostic and works unchanged once the
  candidate is admitted. No `arith.constant N` synthesis is needed — the helper
  materializes a tail call after the writer-while whose operands are projected
  from post-while results, so the count countdown's structural deadness becomes
  irrelevant.
- **Writerless `llzk.nondet` dispatch pod ⇒ zero-arg substruct call must be
  synthesized.** A subset of the `llzk.nondet` dispatch shape above has only
  readers (no `pod.write %pod[@comp] = ...` anywhere — circom dropped the inline
  call entirely for constant-table sub-components). Canonical case: keccak's
  `RC_0` round-constant struct used by 7 keccak chips
  (chi/iota3/iota10/rhopi/round0/round20/theta), surfaced once PR #49 fixed the
  bazel cache invalidation that was masking the regression. Filter widening
  alone (admit `llzk.nondet`) doesn't help — `materializeScalarPodCompField`
  bails when `writers.empty()`. Fix: walk the @comp struct ref + append
  `@compute`, look up the resulting `function.def` via
  `SymbolTable::lookupSymbolIn(module, callee)`, and only synthesize a
  function-scope `function.call @<Sub>::@compute()` when `getNumInputs() == 0` —
  the zero-arg gate prevents inventing operands for arg-bearing dispatches that
  just happen to be missing their writer in this iteration. Use the top-level
  module (walk past LLZK v2's per-component `builtin.module` wrappers via
  `getTopLevelModule` so SymbolTable can reach sibling components).
- **`APInt::getSExtValue()` on a felt constant is a silent miscompile.**
  `FeltConstPattern::matchAndRewrite` (`FeltPatterns.cpp`) historically called
  `feltConstAttr.getValue().getSExtValue()`, which is UB at `getBitWidth() > 64`
  — for any felt constant ≥ 2^63 the call returned the low 64 bits. Power-of-two
  constants (`1 << 252` is the canonical case: LessThan(252)'s offset) truncated
  to `0`, so every comparator chain that depended on the offset miscomputed.
  iden3_intest was the minimal repro; querytest dodged via Mux3 masking.
  Diagnostic recipe: lowered StableHLO
  `grep "value = dense<" | grep -v "dense<[0-9]>"` — a bn128 felt circuit with
  comparators MUST emit at least one
  `dense<7237005577332262213973186563042994240829374041602535252466099000494570602496>`
  (= 2^252) per `LessThan` instance; if those large constants are missing,
  felt.const lowering broke. Fix shape (PR #55): APInt overload of
  `LlzkToStablehloTypeConverter::createConstantAttr` using
  `APInt::zextOrTrunc(storageWidth)` — zero-extension is correct because field
  elements are unsigned, ranged `[0, p)` with `p < 2^254`. When touching any
  MLIR APInt extraction in this codebase (sister site: `convertToIndexTensor` in
  `TypeConversion.cpp`, which guards correctly via
  `getSignificantBits() > 64 → bail`), prefer the APInt path over `getSExtValue`
  / `getZExtValue` unless you've verified the source bitwidth is bounded ≤ 64.
- **`processNested` only recurses into scf.while regions, NOT scf.if.** The
  recursive walker in `runOnOperation` skips any non-scf.while op while visiting
  children, and `flattenPodArrayWhileCarry(block)` itself uses non-recursive
  `for (Operation &op : block)`. So a pod-array-carrying scf.while buried inside
  an scf.if branch is invisible to both. AES `@AES256Encrypt_6::compute` has
  carriers at depth 5 inside scf.if branches (e.g.
  `%148:4 = scf.if %147 -> (..., <13,4,3,32 x !pod>)`); without a separate
  module-level pass these survive to lowering. Fix shape: a post-main-loop
  straggler pass that uses `module.walk(scf::WhileOp)` to find scf.if-buried
  carriers and re-invokes `flattenPodArrayWhileCarry` on their containing block.
  Convergence-safe (same idempotency rule as flatten itself). Don't try to fix
  this by adding scf.if recursion to `processNested` — that path also runs
  `eliminatePodDispatch`, whose pod block-arg gating assumes scf.while
  semantics.
- **Structural IR cleanup ≠ runtime fix on its own.** Closing the iter-arg chain
  at SimplifySubComponents level (lowered
  `func.call @<Sub>_compute (%cst, %cst)` count goes to 0, post-simplify shows
  full per-field connectivity) is necessary but NOT sufficient. The 2026-04-28
  AES investigation produced a fully-connected chain end-to-end with a
  byte-identical runtime witness vs the disconnected baseline (md5 match,
  312/14852 vs circom's 8141/16193). Always re-run the runtime metric AFTER each
  structural improvement; "expected to fix" ≠ "did fix". When three structural
  metrics improve (XOR-nondet 5→0, 17 fewer 4D survivors, 15 fewer 3D survivors)
  without a runtime delta, the data-flow disconnect lives in a layer you weren't
  editing — typically LlzkToStablehlo main conversion, the 3 post-conversion
  vectorization phases, or BatchStablehlo.
- **A new SimplifySubComponents transform must converge with
  `eliminatePodDispatch`.** The outer `while (changed)` loop in `runOnOperation`
  re-runs all phases plus `processNested` recursion until no pass returns
  `true`. If your transform produces IR that `eliminatePodDispatch` keeps
  re-modifying every iteration (e.g. a residual pod.read or pod.new it nondets /
  DCEs), the loop never settles. Symptom: unit tests pass on toy IR but CI hangs
  for tens of minutes on real circuits (PR #37 hit this on `aes_256_ctr` /
  `aes_256_key_expansion` / `maci_*` from a recursive `expandPodArrayWhile` ↔
  `rewritePodArrayUsesInBlock` mutual recursion). Diagnostic recipe: temporarily
  wrap the outer loop with an iter counter + abort at iter > 50, log each pass's
  `changed` return per iter, find the pass that reports `1` after the others
  have settled — that's the one whose input IR your transform broke. Fix shape:
  emit IR that's idempotent under all five `eliminatePodDispatch` phases
  (extract calls / replace reads / erase writem / erase dead pods / replace
  remaining), or rely on the existing outer-fixed-point + `processNested` to
  revisit nested constructs across iterations rather than recursing inside your
  own helper.
- **Result-bearing scf.if with tracked-array result slots + `%nondet_*` branch
  yields breaks the carry chain.** LLZK's `<--` (compute-only assignment)
  semantics produces scf.ifs whose array result slots get yielded as
  `llzk.nondet` placeholders in both branches — the actual writes happen via
  inner whiles inside each branch using the parent's tracked carry as init.
  `liftScfIfWithArrayWrites` (LlzkToStablehlo.cpp:508) early-returns at line 513
  when `getNumResults() != 0`, so the lift never extends the chain through this
  if. After applyPartialConversion the nondet arrays become const-zero tensors;
  the post-pass at line 1985+ inlines branches into selects, but selects on the
  array slots pick between two const-zero tensors → outer carry reads zero.
  Inner-while modifications inside each branch never escape the if boundary. AES
  `xor_2`/`xor_3` slots are the canonical case; lowered `%1#16`/`%1#17` are
  `tensor<13x4x3x32>` / `tensor<13x4x32>` reading from offsets 452/10692 with
  0/4992 + 0/1664 nonzero. Pattern is GENERIC, not AES-specific: 47-circuit
  sweep (2026-04-28) found 5 hits across 2 families (3 AES + fpmultiply +
  signed_fp_carry_mod_p). **Wrong diagnoses to avoid**: (1) line 424 scf.while
  opaque — line 424 already rebinds latest correctly for inner whiles whose init
  matches a tracked array; verified via lowered IR (`%1:18` slot 13 yields
  `%71#3` modified). (2) Sibling-while operand-rewire in scf.while branch —
  disproven via `llvm::errs()` debug print: 27 inner whiles seen, 0 rewire
  fires. **Diagnostic recipe**: dump `--simplify-sub-components` LLZK, search
  `scf\.if.*->.*!array`, check whether each branch's outer scf.yield uses
  `%nondet_*` at the array slot; scan all 47 circuits with
  `python3 /tmp/scan_pattern.py /tmp/llzk_sweep/*.llzk` to confirm prevalence
  before assuming AES-only. **Fix shape (2026-04-28 night-2 implemented at
  `LlzkToStablehlo.cpp:617-738`)**: append NEW tail result slots typed
  `!array<x !felt>` (matching tracked-key types) — do NOT try to rewrite
  existing slots. The original `!array<x !pod>` slots (placeholder dispatch
  arrays from circom's input-counter pattern) and `!array<x !felt>` tracked
  carriers are NOT type-equal pre-conversion, so any type-match-based rewrite
  gate never fires; post-conversion both collapse to the same `tensor<...>`
  shape via `getArrayDimensions()` (which ignores element type), but the helper
  runs pre-conversion. Process each branch with `branchLatest` seeded from
  `parentLatest`, find liveKeys (any tracked key whose branchLatest differs from
  parentLatest in either branch), classify each via reuse-by-yield-match
  (`thenYield[i] == thenLatest[key] && elseYield[i] == elseLatest[key]` →
  existing slot already covers this key) vs append. Idempotent across the
  dual-walker invocations (`promoteArraysToWhileCarry`'s
  `convertArrayWritesToSSA` seeds `latest` with captured arrays only;
  `convertWhileBodyArgsToSSA` later seeds with all body args — second walk
  reuses first walk's appended slots and appends new ones for keys not in first
  walk's seed). Reuse path must reference `newIf.getResult(i)` not
  `oldIf.getResult(i)` because oldIf gets erased on append — first version
  segfaulted on this. **CAVEAT — necessary but not sufficient**: chain extension
  alone leaves AES at 312/14852 (byte-identical to baseline) because the
  inner-loop data source is
  `%nondet_303 = llzk.nondet : !struct<@Num2Bits_2>; struct.readm @out` which
  lowers to const-zero. The remaining gap is a separate `eliminatePodDispatch`
  array-pod gating bug — the `function.call @Num2Bits_2::@compute(...)` that
  should materialize `%nondet_303` doesn't reach this layer.

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

### M3 fixture convention

The M3 measurement harness (`bench/m3/`) feeds the SAME JSON fixture to both
`gpu_zkx` (`m3_runner --input_json=...`) and `cpu_circom` (the circom witness
binary in `run_baseline.sh`). Schema is circom's native form:
`{<signal_name>: scalar | flat-array}`, one top-level key per circuit input
signal. Fixtures live at `bench/m3/inputs/<TARGET>.json` where `<TARGET>`
matches the bazel alias in `bench/m3/run.sh:60-78`.

GPU-side parameter mapping is **positional in JSON insertion order** — MLIR
lowering strips circom signal names (parameters surface as `%arg0`, `%arg1`, …).
`bench/m3/json_input.cc` uses `nlohmann::ordered_json` (NOT plain
`nlohmann::json`, which iterates alphabetically) to preserve order; the
`KeyOrderMatters` test pins the contract. When adding a new circuit fixture, the
JSON key order must match the order of `func.func @main`'s parameters in the
lowered StableHLO output (see `bazel-bin/examples/<TARGET>.stablehlo.mlir`).

Fixtures store **one witness's worth of values**, not N copies. `m3_runner`
auto-tiles the per-witness tokens across the leading batch dim added by
`--batch-stablehlo`: `LiteralFromDecStrings` accepts
`tokens.size() * shape.dimensions(0) == num_elements` and replicates via
`tokens[i % token_count]`. So a fixture's `in[1600]` array fills
`tensor<N, 1600>` at any N≥1 without fixture changes. `cpu_circom`
(`run_baseline.sh`) feeds the same single-witness fixture to N sequential circom
invocations — the same file drives both backends with no tiling indirection.

`@open_zkx//zkx/tools/stablehlo_runner/stablehlo_runner_main.cc`'s
`ParseInputLiteral` / `ParseInputLiteralsFromJson` are private to that binary's
anonymous namespace; we cannot reuse them via include and have ported the
equivalents into `bench/m3/json_input.cc`. Don't chase a "share the helper"
refactor — it's been considered.

### M3 correctness gate convention

The `bench/m3/` gate (PR #20) opts in per-circuit via a
`bench/m3/inputs/<TARGET>.json.gate` sentinel. Sentinel content is the `.wtns`
wire-index list (one per output Literal element, space- or comma-separated); an
empty file defaults to contiguous `[1..1+N)`. `m3_runner` reads the index list
through `--gate_wtns_indices=...` and byte-compares the GPU output Literal
against `wtns.Witness(idx)` for each declared index.

**Layout caveat**: circom does NOT always assign wire IDs as
`[const, outputs, inputs, intermediates]`. `MontgomeryDouble`'s `.wtns`
interleaves outputs around the echoed-input wires
(`[const, out0, out1, in0, in1, out2, out3]`), so its sentinel is `1 2 5 6`
rather than `1 2 3 4`. For each new gated circuit: decode the `.wtns` integers
to confirm input wire positions (echoed inputs match the JSON fixture's values
verbatim) — or run the gate with provisional indices and read the
mismatch-hex-diff to recover the actual layout.

`witness_compare`'s API accepts duplicate indices, which is required for
circuits whose GPU output flattens public + private intermediate signals into
one tensor. e.g. `keccak_pad` emits `tensor<2176>` = `out[1088] || out2[1088]`;
the `.wtns` file only stores public wires, so positions in the private half map
onto any 0-valued public wire (`out2[264..1080)` are zero by template
construction; `out2[1087]=0` mirrors `wtns[1]=0`). See `docs/M3_REPORT.md` §4.4
footnote ¹⁵ for the `keccak_pad` row. The gate rejects tuple shapes / N>1
batched outputs; it is N=1 single-tensor only — `run.sh` auto-skips at N>1 with
an actionable message rather than failing the comparator on length mismatch, so
a circuit can be both gated (at N=1) and measured (at N>1) without operator
intervention. Extending the gate to N>1 byte-equality requires N-tiling
`gate_wtns_indices` against `[N, K]` outputs; that work is intentionally
separate so a divergence at any N>1 does not retroactively block the N=1-only
entries.

A close variant: when `@main` reduces to
`dynamic_update_slice(zeros<N>, %result<M>, 0)` (M < N — the result tensor
occupies a prefix and zeros pad the rest), the trailing N−M positions are
sentinel-equivalent to `keccak_pad`'s private half — assign each pad position
any `.wtns` index whose value decodes to 0 (a single shared index typically
works for the whole pad). The keccak chi/round0/round20/theta/iota3/iota10/
rhopi cluster all share this shape under the standard `in[1600]` fixture; see
footnote ¹⁹.

**Every newly gated circuit must be added to the CI regression test in the same
PR that lands the fixture.** `//bench/m3:m3_correctness_gate_test` (`gpu`-tagged
`sh_test`) runs `m3_runner --correctness_gate=true` against each chip in its
`data` block; CI executes this via `.bazelrc.ci`'s `--test_tag_filters=""` so a
future lowering regression on any gated chip surfaces in the PR's check status
instead of going silent until the next manual `bench/m3/run.sh`. To add a chip:
(1) extend the `CHIPS=(...)` array in `bench/m3/m3_correctness_gate_test.sh`,
(2) append the matching `//examples:<chip>` target plus the `.json` /
`.json.gate` / `.wtns` trio to the `data = [...]` list in
`bench/m3/BUILD.bazel`. Skip this step ⇒ the gate is a manual-checkpoint-only
artifact and the next silent-zero regression goes unnoticed; mirroring
`MontgomeryDouble` / `keccak_pad` / `keccak_squeeze` / `iden3_is_expirable` /
`iden3_is_updatable` in §4.4 of the M3 report without the matching `data =`
entry is the convention violation.

**Circom binary swaps don't invalidate bazel's cache; the gate can pass on stale
LLZK.** `third_party/circom/workspace.bzl` writes a wrapper `exec`-ing the
resolved circom path. Bazel hashes the wrapper text, not the underlying binary
content. Updating `/usr/local/bin/circom` (or whatever `CIRCOM_PATH` points at)
without changing the path string leaves wrapper text identical → all downstream
`circom_to_llzk`, stablehlo conversion, and `m3_correctness_gate_test` outputs
cache-hit on prior runs. Symptom: CI shows
`//bench/m3:m3_correctness_gate_test (cached) PASSED in <past time>`. Local
invalidation: `--disk_cache=` or
`--repo_env=CIRCOM_PATH=/usr/local/./bin/circom` (slight path perturbation
forces repository_rule re-eval). Proper fix tracked in
`memory/bazel-circom-content-hash-and-keccak-regression-followup.md`.

**Don't ship a gate sentinel before its baseline is currently green.** A new
`.json.gate` whose byte-equal compare fails at the time of landing buys zero
regression protection: enabling it turns CI red, toggling it off ships a dead
skip mechanism, and committing the artifacts without wiring them in leaves
unused files. Bundle the sentinel + `data=[...]` + `CHIPS` updates into the same
PR as the lowering / compute fix that flips the metric red → green, so the diff
produces one bisectable point. Pre-staged artifacts (`.wtns` + `.json.gate`) in
a worktree without committing are fine; what's harmful is landing a gate-half-PR
ahead of the fix-half-PR.

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
