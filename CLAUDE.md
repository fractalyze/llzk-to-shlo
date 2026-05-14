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

LLZK is versioned upstream and changes break us. Each rule below is a
silent-miscompile or hang trap; for already-landed fixes, git blame +
`~/.claude/knowledge/llzk-to-shlo-*.md` carry the implementation history.

- **Test fixtures are consumer-owned IR.** Hand-written `.mlir` test files get
  parsed directly. The upstream IR migrator does not touch parser input —
  consumer fixtures must be hand-migrated in the same bump.
- **BUILD glue must move with upstream.** New `.td` files (new dialects /
  interfaces) don't appear in `third_party/llzk/llzk.BUILD` automatically. When
  bumping, diff `include/llzk/Dialect/**/*.td` against `gentbl_cc_library`
  targets and add missing inc-gen rules.
- **`createEmptyTemplateRemoval` uses `applyFullConversion` over a narrow op
  list.** Its conversion target only handles ops in `OpClassesWithStructTypes`
  (struct/array/function/global/constrain/polymorphic). Anything else (`pod.*`,
  `llzk.nondet` results with struct types, our synthesized ops) MUST be gone or
  already in stripped form before this pass. Order: clean residual pod traffic
  first, pre-strip `<[]>` only on ops *outside* that tuple, then run template
  removal.
- **project-llzk/circom PR #378's same-named `poly.template` wrap leaves
  `module @X { function.def @X }` shells after `EmptyTemplateRemoval`.** circom
  v2 (post-#378) emits every `function.def` / `struct.def` inside a same-named
  `poly.template @X` to track polymorphic typing. `EmptyTemplateRemoval`
  rewrites that to `builtin.module @X { function.def @X }`; the inner symbol now
  shadows the wrapping module's symbol in the parent's `SymbolTable`, and the
  next pass that walks it (LlzkToStablehlo conversion in particular) trips with
  `redefinition of symbol named '<X>'`. Fix surface is
  `flattenSingleEntityWrapperModules` in `SimplifySubComponents.cpp`: hoist the
  same-named single child to module level, erase the empty wrapper, then use
  `AttrTypeReplacer` to rewrite `@X::@X[::@method]` → `@X[::@method]` so refs
  nested in types (e.g. `!struct.type<@X::@X>`) are also caught — a plain
  attribute walk misses those.
- **`<[]>` (empty params) vs no params on `!struct.type`.** Template removal
  rewrites `<[]>` to no-params on covered ops but leaves SSA values on uncovered
  ops alone. Mixing forms across a use-def edge produces unresolved
  `builtin.unrealized_conversion_cast`. Strip on uncovered ops; don't strip on
  covered ones.
- **`llzk.nondet : index` is dialect-conversion-illegal.** The conversion target
  legalizes nondet for felt/array/struct only. For residual `pod.read` ops with
  `index` result type (dispatch-pod `@count` countdown), substitute
  `arith.constant 0 : index` — the surrounding cmpi/scf.if scaffold is
  structurally dead once the call is hoisted; 0 keeps cmpi false and DCE
  collapses the dead branch.
- **`replaceRemainingPodOps` (Phase 5) clobbers `unpackPodWhileCarry`'s field
  discovery.** Phase 5 nondets every `pod.read` in a block — including reads of
  pod-typed `scf.while` block args, which are field-discovery input for the next
  outer iteration. Gate `eliminatePodDispatch` on the block having no pod-typed
  block args. Symptom: multi-record input pods (e.g. keccak
  `<[@a: array, @b: array]>` carries) survive into dialect conversion.
- **Pod-array iter-arg survival post-simplify is a silent miscompile signal** —
  a surviving pod-typed `scf.while` carry plus the `func.call (%cst…)`
  blanket-nondet grep are the paired diagnostic. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#pod-array-iter-arg-survival-post-simplify).
- **Multi-carry chips need rewrite-back between consecutive same-instance
  compute calls** — without `dynamic_update_slice %iterArg` between back-to-back
  calls, the second call reads stale operands and silently miscompiles. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#multi-carry-chips-need-rewrite-back-between-consecutive-same-instance-calls).
- **Sub-component call-count diff (SSC vs lowered StableHLO) is a distinct
  silent-miscompile signal** — N→0 or N→partial drop between
  `function.call @<Sub>::@compute` count in SSC output vs `func.call @<Sub>` in
  lowered StableHLO locates the Phase 1 hoist gap. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#sub-component-call-count-diff-ssc-vs-lowered-stablehlo).
- **Struct-of-pods carrier `!pod<[@idx_0..@idx_K-1: T]>` is a SEPARATE shape
  from array-of-pods `!array<K x !pod<...>>` — `flattenPodArrayWhileCarry` only
  handles the latter, and the survivor is load-bearing at the dispatch firing
  site.** Fix surface is multi-stage: `convertStructOfPodsToArrayOfPods` when
  inner type is uniform, `materializeStructOfPodsCompField` when not. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#struct-of-pods-carrier-survives-flattening-when-inner-type-is-non-uniform)
  for the four-stage chain, and
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#struct-of-pods-materializer-requires-three-coordination-invariants)
  for the materializer's own invariants.
- **`processNested`'s `hasArrayOfPods` / `hasPodBlockArg` guard is four-branch,
  not two-branch** — mirror the structure when adding a new guard-gated dispatch
  elimination phase. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#processnesteds-hasarrayofpods--haspodblockarg-guard-is-four-branch).
- **SSC rewriters rebuilding region-bearing ops via `takeBody` need a parallel
  `rebuiltOps` set** — `toErase.count(parent)` returns false after a rebuild
  because `parent` is the NEW op, leading to exponential rebuild and heap
  corruption. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#ssc-rewriters-rebuilding-region-bearing-ops-need-a-rebuiltops-set).
- **`llzk.nondet` dispatch pod (no `pod.new`) is a silent miscompile** —
  post-project-llzk/circom-#390 the dispatch pod can arrive as `llzk.nondet`
  rather than `pod.new`, and `materializeScalarPodCompField`'s candidate filter
  must cover both. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#llzknondet-dispatch-pod-no-podnew).
- **Writerless `llzk.nondet` dispatch pod ⇒ synthesize zero-arg substruct
  call.** Subset of the above with no `pod.write %pod[@comp] = ...` anywhere
  (circom dropped the inline call for constant-table sub-components, e.g.
  keccak's `RC_0`). `materializeScalarPodCompField` bails on `writers.empty()`.
  Walk the @comp struct ref + `@compute`, look up the `function.def` via
  `SymbolTable::lookupSymbolIn(module, callee)`, and synthesize a function-scope
  `function.call @<Sub>::@compute()` only when `getNumInputs() == 0`. Use
  `getTopLevelModule` to walk past LLZK v2's per-component `builtin.module`
  wrappers.
- **`APInt::getSExtValue()` on a felt constant is a silent miscompile.** UB at
  `getBitWidth() > 64` — the call returns the low 64 bits. `1 << 252` (LessThan
  offset) truncates to `0`. Use `APInt::zextOrTrunc(storageWidth)` —
  zero-extension is correct because field elements are unsigned, ranged `[0, p)`
  with `p < 2^254`. Diagnostic: `grep "value = dense<" | grep -v "dense<[0-9]>"`
  — bn128 felt circuits with comparators MUST emit at least one
  `dense<7237005577332262213973186563042994240829374041602535252466099000494570602496>`
  (= 2^252) per `LessThan`. Sister site to audit: `convertToIndexTensor` in
  `TypeConversion.cpp`.
- **`APInt::operator==` / `!=` asserts on bit-width mismatch — normalize at
  construction when collecting APInts from heterogeneous sources.**
  `felt.const`'s `FeltConstAttr` uses `APIntParameter`'s minimum-bits-needed
  sizing (e.g. `felt.const 1` → 4-bit APInt); `arith.constant : index` is always
  64-bit. Comparing them directly asserts in dbg and is UB in opt (slow-case
  `EqualSlowCase` reads past one side's word boundary, often producing a corrupt
  pointer that segfaults much later — the original bucket-2 S1-cluster
  `Type::getContext` UAF inside `EmptyTemplateRemoval` was a downstream symptom
  of this). When building a `SmallVector<APInt>` for comparison across sources,
  apply `zextOrTrunc(kCommonWidth)` at the push_back sites (typically 64 for
  index-typed values). Reference site:
  `SimplifySubComponents.cpp::outerIndexConstValues`.
- **`unpackPodWhileCarry` gates on a fixed set of handleable pod-value use
  shapes** — any use outside the handled set must short-circuit before IR
  mutation, or the cleanup trips on `use_empty()`. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#unpackpodwhilecarry-gates-on-a-fixed-set-of-handleable-pod-value-use-shapes).
- **`processNested` only recurses into scf.while regions, NOT scf.if.**
  `flattenPodArrayWhileCarry(block)` itself is non-recursive
  (`for (Operation &op : block)`). Pod-array-carrying `scf.while` buried inside
  an `scf.if` branch is invisible. Reach them with a post-main-loop straggler
  pass using `module.walk(scf::WhileOp)` — don't add scf.if recursion to
  `processNested`, that path also runs `eliminatePodDispatch`, whose pod
  block-arg gating assumes scf.while semantics.
- **Structural IR cleanup ≠ runtime fix on its own.** Closing the iter-arg chain
  at SimplifySubComponents (lowered `func.call (%cst, %cst)` count → 0) is
  necessary but NOT sufficient. Always re-run the GPU correctness gate AFTER
  each structural improvement. When structural metrics improve without a runtime
  delta, the data-flow disconnect is downstream — typically LlzkToStablehlo main
  conversion, the 3 vectorization phases, or BatchStablehlo.
- **A new SimplifySubComponents transform must converge with
  `eliminatePodDispatch`** — IR that the dispatch loop keeps re-modifying hangs
  CI on real circuits while unit tests pass on toy IR. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#a-new-ssc-transform-must-converge-with-eliminatepoddispatch).
- **Result-bearing `scf.if` with tracked-array result slots + `%nondet_*` branch
  yields breaks the carry chain.** LLZK's `<--` produces scf.ifs whose array
  result slots get yielded as `llzk.nondet` placeholders in both branches; the
  actual writes happen via inner whiles inside each branch using the parent's
  tracked carry as init. After applyPartialConversion the nondet arrays become
  const-zero tensors; selects over them pick between two const-zero tensors.
  `liftScfIfWithArrayWrites` early-returns on `getNumResults() != 0` and handles
  void ifs only — result-bearing ifs go through
  `extendResultBearingScfIfArrayChain`, which must append NEW tail result slots
  typed `!array<x !felt>` (matching tracked-key types) — do NOT rewrite existing
  slots (the original `!array<x !pod>` placeholders and tracked
  `!array<x !felt>` carriers aren't type-equal pre-conversion). Idempotent
  across the dual-walker invocations (`convertArrayWritesToSSA` +
  `convertWhileBodyArgsToSSA`). Reuse path must reference `newIf.getResult(i)`
  not `oldIf.getResult(i)` — `oldIf` gets erased on append.
- **Don't reshape the `<--` cascade from SSC — downstream rewriters depend on
  its shape.** Recognize structurally-dead pod dispatch bundles via use-trace at
  the scf.while-carrier drop decision instead. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#dont-reshape-the----cascade-from-ssc).
- **`eraseDeadPodAndCountOps` Phase 4 must walk regions transitively before
  erasing a region-bearing candidate** — `isAllResultsUnused(op)` is necessary
  but not sufficient when Phase 1 has hoisted a call referencing inner-region
  SSA. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#erasedeadpodandcountops-phase-4-must-walk-regions-transitively).
- **`isOpAndNestedResultsExternallyUnused` is blind to side effects on outer
  values from zero-result inner ops** — pair every region-bearing synthesizer
  with a `hasNonPodArrayWriteInBody` carve-out at `eraseDeadPodAndCountOps`. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#isopandnestedresultsexternallyunused-is-blind-to-side-effects-on-outer-values).
- **`extractCallsFromScfIf` Phase 1's directArgs path is non-idempotent** —
  guard the hoist against inside-scf.if defs via `Operation::isAncestor`, or the
  outer fixed-point re-hoists ~50+ duplicates per `maci_*` chip. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#extractcallsfromscfif-phase-1-directargs-path-is-non-idempotent).
- **`inlineInputPodCarries` must dedupe `toErase`** — a single
  `pod.write %pod = %value` is a user of both operands, so naive `push_back`
  double-frees on cleanup. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#inlineinputpodcarries-must-dedupe-toerase).
- **`convertWhileBodyArgsToSSA` (LlzkToStablehlo.cpp:823-855) absorbs
  forwarder-vs-nested-result yield discrepancies at lowering time.** It walks
  each scf.while body, runs `processBlockForArrayMutations` to track per-block
  latest SSA carriers (lines 491-509 rebind `latest[blockArg]` to the inner
  scf.while's matching result), then rewrites the body's yield
  operand-by-operand using `latest`. So an LLZK body yielding `%arg_blockarg`
  (forwarder shape) and an LLZK body yielding `%nested_while.result(_)`
  (nested-result shape) lower to the same StableHLO. Verify any LLZK-level
  yield-shape fix by diffing the lowered StableHLO, not the simplified LLZK —
  the LLZK may change while the lowered MLIR is identical, meaning the fix
  didn't earn its keep. Concrete fix sites for body-yield correctness are this
  pass and `processBlockForArrayMutations`, not `expandPodArrayWhile`'s yield
  rewriter.
- **Phantom rebind via a read-only inner-while capture** — silent miscompile
  when an inner `scf.while` passthroughs a captured array carrier and the rebind
  at `processBlockForArrayMutations:491-509` taints the outer yield. See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#phantom-rebind-via-a-read-only-inner-while-capture).
- **`collapseRedundantWhileCarrierPairs` zero-init transitivity also requires
  every enclosing while to be passthrough.** The pass classifies a
  `stablehlo.while` slot as DEAD when its yield is a literal pass-through of the
  body argument and its init traces back through enclosing-while body-args to a
  zero-splat constant — then RAUWs the dead result with a sibling LIVE result of
  identical type. The DEAD-collapse semantics depend on the dead slot being
  "always zero on every iteration", which holds at THIS while only if no
  intermediate enclosing while mutates the carrier. A parent while whose yield
  differs from its body arg at the same slot index breaks this — body args on
  later iterations carry the parent's mutated value, not init. Canonical
  violator: AES `xor_2[i][j][k].b` is zero-initialized at the main while but
  actively written across rounds, so any inner-level passthrough isn't truly
  "always zero." `isZeroSplatTransitively` MUST reject the trace whenever a
  visited parent slot is non-passthrough — otherwise the inner RAUW silently
  merges `xor_2 .a` and `xor_2 .b` and the lowered body yields the same SSA
  value at distinct .a/.b slots.
- **`materializePodArrayCompField`'s drain treats K pub felt members as an extra
  outer dim, not as separate `struct.member`s** — K>1 path prepends K in
  declaration order matching circom's `.wtns` layout; K=0 falls through to a
  recursive flatten that promotes inner writem-targeted members to `{llzk.pub}`.
  See
  [`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#materializepodarraycompfields-k-pub-felt-drain-treats-k-as-an-outer-dim).

See [`docs/CIRCUIT_COVERAGE.md`](docs/CIRCUIT_COVERAGE.md) for how a
frontend/LLZK mismatch surfaces at the user-visible level (per-circuit
pass/fail, per-stage error categories).

## Load-Bearing Invariants

Cross-cutting assumptions the pipeline relies on. Each can be violated by an
innocent-looking change without breaking the build — the result still compiles,
runs on GPU, and silently produces wrong witnesses. Test changes in the
neighborhood against circom's C++ witness, not against the lowered IR alone.

- **Sister-circuit families (AES variants, SHA variants, etc.) are NOT
  differential references for each other.** Treating one as "the working one"
  because it diverges from a known-broken sibling is unsound — both can be
  silently broken in different output positions. Always pin the reference
  against an independent backend (circom `.wtns` via the m3 correctness gate)
  before treating it as ground truth.
- **Circom's `<==` vs `<--` is a security boundary.** `<==` emits both
  `@compute` and `@constrain`. `<--` emits only `@compute` and *requires* a
  separate explicit constraint (e.g. `Num2Bits` must add
  `out[i] * (out[i] - 1) === 0` after `out[i] <-- (in >> i) & 1`). Dropping the
  follow-up constraint does not fail any test in this repo — it just makes the
  circuit unsound in the prover.
- **`@constrain` functions and all `constrain.eq` ops are erased during
  lowering.** GPU code only runs witness generation; constraint satisfaction is
  the prover's job downstream. The lowered StableHLO has no way to catch a
  broken constraint.
- **Batched witness output must include every signal — public outputs *and*
  private internals.** `BatchStablehlo`'s leading `N` dim aliases the full
  per-witness state across `N` proofs. Pruning to "just the public signals"
  silently misaligns the batch axis.
- **Circom while loops have compile-time trip counts; all batch elements iterate
  the same number of times.** `BatchStablehlo` extracts the predicate from
  element `[0]` and reuses it for the whole batch. A future frontend emitting
  batch-divergent trip counts would break this.
- **Circom signals are immutable, which is what makes vectorization sound.**
  Auto-vectorization turns `while (i<N) { out[i] = f(a[i]) }` into `out = f(a)`
  only because `out[i]` can't be reassigned — the canonical iterative pattern is
  `chain[N][K+1]`-style arrays, not mutable accumulators.
- **`array.new` (no operands) and `llzk.nondet` lower to the same `dense<0>`
  constant.** `ArrayPatterns.cpp:ArrayNewPattern` and
  `RemovalPatterns.cpp:LlzkNonDetPattern` both call
  `tc.createConstantAttr(tensorType, 0, rewriter)`. Switching one for the other
  in `SimplifySubComponents` is a no-op at lowering time. Before chasing a
  refactor on "orphan nondet" or "wrong init source", verify the lowered
  StableHLO is actually different (run the gate, observe whether bytes flip). If
  not, the bug is downstream — likely in witness-output assembly.
- **Witness-output orphan-wire detection lives in `--verify-witness-layout`, not
  the lowering pattern.** VerifyWLA does an exact per-anchor check against
  `wla.layout`: every signal must have a covering `dynamic_update_slice` chunk
  in `@main`, splat-zero is rejected on `output` signals, and *permitted* on
  `internal` signals — a legitimately zero internal is indistinguishable from an
  orphan at the lowering boundary, so m3 byte-equality is the actual correctness
  gate there. The legacy heuristic (`StructPatterns.cpp:StructWriteMPattern`
  length>=8 splat-zero check) is default-off and survives only as
  `flag-orphan-zero-writes=true` for diagnostic spelunking and Wave 1 regression
  fixtures.
- **`getMemberFlatSize` recurses through struct-typed members.** For
  `<N x !struct<Inner>>` the helper returns `N * recursive_flat(Inner)`, where
  `recursive_flat(Inner)` sums each writem-targeted, non-pod member's flat size
  (recursively). The writem-targeted filter matches the offset-map invariant in
  `LlzkToStablehlo.cpp::registerStructFieldOffsets`. Two practical consequences:
  (1) inner struct-array slots like webb's
  `@inTree : <16 x !struct<ManyMerkleProof_275>>` (where MMP has zero pub felts
  → PR #97 multi-pub flatten doesn't apply) report the real felt footprint
  (1,440 = 16 × 90 for MMP_275's @hasher 30 + @switcher 60), so the lowering can
  later land real per-cell values; (2) chips already covered by PR #97 multi-pub
  flatten still see `<N x !felt>` slots and report `N` — total unchanged. The
  recursion requires `ModuleOp` access to look up inner struct defs, so callers
  pass the top-level module. Cycle guard via a `visited` set on leaf names; LLZK
  struct definitions are tree-shaped today, but a self-referential type returns
  0 instead of looping. The `--stabilize` flag pinned in
  `examples/e2e.bzl::_circom_to_llzk_impl` guarantees inner struct leaf names
  are unique within a chip module, which is what makes the flat walk-and-match
  lookup correct.
- **LLZK `struct.readm` (`MemberReadOp`) on a non-pub member is illegal from
  outside the member's parent struct.** `MemberReadOp::verifySymbolUses` (llzk
  struct dialect Ops.cpp) rejects `struct.readm %callResult[@<priv>]` when the
  enclosing op's parent struct ≠ the member's defining struct AND the member
  lacks `{llzk.pub}`. WLA, in contrast, exposes writem-targeted members in the
  witness footprint regardless of pub-ness (see `getMemberFlatSize` recursion
  above). This is a real asymmetry: the witness layer treats them as observable,
  the struct verifier blocks the external read. The bridge for dispatch-pod
  drain paths that need to externally extract these values is **member
  promotion**: any pass that needs to emit `struct.readm %callResult[@<field>]`
  from a *parent* struct's @compute must first set `PublicAttr::name` on the
  inner struct.def's matching `struct.member` op (mutation site:
  `SimplifySubComponents.cpp::findRecursiveWritemMembers` with `promote=true`).
  Promotion is semantically aligned with the WLA contract already in effect —
  the slots WERE observable in the witness; pub makes the LLZK verifier agree.
  Side-effect audit: WLA's pub/internal kind distinction
  (`WitnessLayoutAnchor.cpp:67-91`) operates on the *main* struct's members, not
  on inner struct members, so promoting inner writem-targets is invisible to
  WLA's signal kind classification.
- **`--witness-layout-anchor` MUST run after `--simplify-sub-components`.**
  Running before SSC trips upstream's `applyFullConversion` (it rejects unknown
  ops). `--verify-witness-layout` runs after `--llzk-to-stablehlo` and asserts
  each `dynamic_update_slice` chunk in `@main` matches a `wla.layout` entry.
  Full dialect + pass contract in
  [`docs/WITNESS_LAYOUT_ANCHOR.md`](docs/WITNESS_LAYOUT_ANCHOR.md).
- **MLIR dialect `gentbl_cc_library` outputs need a wide MLIR-header set even
  when manual code in the .cpp/.h doesn't reference them.** Generated
  `<Dialect>Ops.{h,cpp}.inc` and `<Dialect>Attrs.{h,cpp}.inc` transitively
  reference `DialectBytecodeWriter` (`mlir/Bytecode/BytecodeOpInterface.h`),
  `getProperties()` (`mlir/IR/OpDefinition.h` + `mlir/IR/Builders.h`),
  `OpAsmPrinter` (`mlir/IR/OpImplementation.h`), `Diagnostics.h`,
  `SideEffectInterfaces.h` (when ops use `Pure` etc.), and
  `DialectImplementation.h`. cpplint will keep flagging these as "unused" — DO
  NOT drop them. Mirror the include set in
  `llzk_to_shlo/Dialect/WLA/WLA.{h,cpp}` when adding a new dialect.
- **Consolidate `gentbl_cc_library` rules into one per-dialect TableGen file.**
  Mirror `llzk_to_shlo/Dialect/WLA/BUILD.bazel`'s `wla_inc_gen` rule which
  produces 8 outputs (dialect/enum/attr/op decls + defs) from a single
  `mlir-tblgen` invocation. The 4-rule split re-invokes `mlir-tblgen` 4× per
  clean build for no benefit.

## Conventions & Background

### M3 fixture convention

The M3 measurement harness (`bench/m3/`) feeds the SAME JSON fixture to both
`gpu_zkx` (`m3_runner --input_json=...`) and `cpu_circom` (the circom witness
binary in `run_baseline.sh`). Schema is circom's native form:
`{<signal_name>: scalar | flat-array}`, one top-level key per circuit input
signal. Fixtures live at `bench/m3/inputs/<TARGET>.json` where `<TARGET>`
matches the bazel alias in `bench/m3/run.sh`.

GPU-side parameter mapping is **positional in JSON insertion order** — MLIR
lowering strips circom signal names (parameters surface as `%arg0`, `%arg1`, …).
`bench/m3/json_input.cc` uses `nlohmann::ordered_json` (NOT plain
`nlohmann::json`, which iterates alphabetically) to preserve order; the
`KeyOrderMatters` test pins the contract. When adding a new circuit fixture,
JSON key order must match the order of `func.func @main`'s parameters in the
lowered StableHLO output (see `bazel-bin/examples/<TARGET>.stablehlo.mlir`).

Fixtures store **one witness's worth of values**, not N copies. `m3_runner`
auto-tiles per-witness tokens across the leading batch dim added by
`--batch-stablehlo`: `LiteralFromDecStrings` accepts
`tokens.size() * shape.dimensions(0) == num_elements` and replicates via
`tokens[i % token_count]`. So a fixture's `in[1600]` array fills
`tensor<N, 1600>` at any N≥1 without fixture changes.

`stablehlo_runner_main.cc`'s `ParseInputLiteral` / `ParseInputLiteralsFromJson`
are private to that binary's anonymous namespace; we cannot reuse them via
include and have ported the equivalents into `bench/m3/json_input.cc`. Don't
chase a "share the helper" refactor.

### M3 correctness gate convention

The `bench/m3/` gate opts in per-circuit via a
`bench/m3/inputs/<TARGET>.json.gate` sentinel. Sentinel content is the `.wtns`
wire-index list (one per output Literal element, space- or comma-separated); an
empty file defaults to contiguous `[1..1+N)`. `m3_runner` reads the index list
through `--gate_wtns_indices=...` and byte-compares the GPU output Literal
against `wtns.Witness(idx)` for each declared index.

**Layout caveat**: circom does NOT always assign wire IDs as
`[const, outputs, inputs, intermediates]`. `MontgomeryDouble`'s `.wtns`
interleaves outputs around echoed-input wires, so its sentinel is `1 2 5 6`. For
each new gated circuit, decode the `.wtns` integers to confirm input wire
positions (echoed inputs match the JSON fixture's values verbatim) — or run the
gate with provisional indices and read the mismatch-hex-diff.

**Internal-wire sentinels require `circom --O0`.** circom's default `-O1`
simplifier removes any wire whose value is fully derivable from already-emitted
wires, marking it `-1` in the `.sym` file's second column. Roughly half the
internal wires for a Poseidon-heavy chip (every `hasher.out`, `signature.out`,
`inKeypair.hasher.out`, etc.) drop to `-1` under `-O1` — there's no `.wtns` wire
to compare against, so a sentinel that maps GPU output[i] → a folded signal must
either duplicate-index to a zero-value wire (CLAUDE.md "duplicate indices"
trick) or rebuild the ground-truth witness with `-O0` so every signal gets an
explicit wire ID. For new gated chips whose GPU output covers internal slots
beyond the input region (any chip past keccak-class complexity, e.g. webb's
`Transaction` chain), commit a `-O0`-generated `.wtns` instead of the `-O1`
default. The `-O0` `.wtns` is roughly 2× the `-O1` size but the trade is worth
the byte-equality coverage.

`witness_compare`'s API accepts duplicate indices, required for circuits whose
GPU output flattens public + private intermediate signals into one tensor (e.g.
`keccak_pad` emits `tensor<2176>` = `out[1088] || out2[1088]`). When `@main`
reduces to `dynamic_update_slice(zeros<N>, %result<M>, 0)` with M < N, assign
each trailing pad position any `.wtns` index whose value decodes to 0.

The gate rejects tuple shapes / N>1 batched outputs; it is N=1 single-tensor
only. `run.sh` auto-skips at N>1 with an actionable message.

**Constraint-only templates lower to `tensor<0>` — gateable as shape-stability
anchors via vacuous PASS.** `witness_compare` early-returns OkStatus when
`num_elements == 0`, so a chip with no public signals can still be gated with an
empty `.json.gate`; a future regression that turns the output into `tensor<N>`
with N>0 surfaces at the existing element-count != index-count check.

**Every newly gated circuit must be added to the CI regression test in the same
PR that lands the fixture.** `//bench/m3:m3_correctness_gate_test` (`gpu`-tagged
`sh_test`) runs `m3_runner --correctness_gate=true` against each chip in its
`data` block. (1) Extend `CHIPS=(...)` in
`bench/m3/m3_correctness_gate_test.sh`, (2) append the matching
`//examples:<chip>` target plus the `.json` / `.json.gate` / `.wtns` trio to
`data = [...]` in `bench/m3/BUILD.bazel`. Skip ⇒ the gate is a
manual-checkpoint-only artifact.

**Pre-enrollment bug repro and `awk`-slicing foot-guns** — see
[`docs/LOWERING_PITFALLS.md`](docs/LOWERING_PITFALLS.md#diagnostic-foot-guns)
for `m3_runner` direct-invocation recipe (no CHIPS / BUILD.bazel edits) and the
brace-balanced extractor convention.

**Circom binary swaps don't invalidate bazel's cache.**
`third_party/circom/workspace.bzl` writes a wrapper `exec`-ing the resolved
circom path. Bazel hashes the wrapper text, not binary content. Updating
`/usr/local/bin/circom` without changing the path string leaves wrapper text
identical → all downstream `circom_to_llzk`, stablehlo conversion, and
`m3_correctness_gate_test` outputs cache-hit. Symptom:
`m3_correctness_gate_test (cached) PASSED in <past time>`. Local invalidation:
`--disk_cache=` or `--repo_env=CIRCOM_PATH=/usr/local/./bin/circom` (path
perturbation forces repository_rule re-eval).

**Don't ship a gate sentinel before its baseline is currently green.** A new
`.json.gate` whose byte-equal compare fails at landing buys zero regression
protection. Bundle the sentinel + `data=[...]` + `CHIPS` updates into the same
PR as the lowering / compute fix that flips the metric red → green.

**`circom_to_llzk` passes `--stabilize` so multi-sub-component composite chips
have deterministic `struct.member` ordering.** Without the flag,
project-llzk/circom seeds its symbol-table / sub-component registry from a
default Rust `HashMap` `RandomState`, so any chip whose main `struct.def`
declares ≥2 sub-component-derived members shuffles output ordering per process
(proven 2026-05-04: 3 fresh runs, 3 distinct LLZK md5s on `splicer_test`).
Position-based `.json.gate` sentinels map GPU output offsets to `.wtns` wire
indices via that ordering, so non-deterministic emission would drift the gate
every CI run. `--stabilize` sorts the iteration order into something
reproducible — `splicer_test.llzk` md5 is byte-stable across runs once enabled.
Single-output chips (`@out` only) are immune either way. Keep the flag pinned in
`examples/e2e.bzl::_circom_to_llzk_impl`; dropping it returns the gate to its
drifting state.

Methodology trap: `bazel clean --expunge` does NOT wipe `--disk_cache=PATH`
configured in user-level `~/.bazelrc`; pass `--disk_cache=` (empty) when probing
for variance, otherwise the second run cache-hits the first.

### Markdown footnotes in docs/

The pre-commit `mdformat` hook runs with `mdformat-gfm` and
`mdformat-frontmatter` only — **no** `mdformat-footnote`. Raw GFM footnote
definitions (`[^id]: text`) are escaped to `\[^id\]: text` on autoformat. Use
**inline numeric markers** (`¹` `²` …) and a "Notes:" sub-list immediately under
the table. To switch styles, add `mdformat-footnote` to
`.pre-commit-config.yaml` `mdformat` `additional_dependencies` first.

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
