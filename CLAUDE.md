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
- **Pod-array iter-arg survival post-simplify is a silent miscompile signal.**
  Diagnostic:
  `bazel run //tools:llzk-to-shlo-opt -- --simplify-sub-components <input>` then
  `grep -nE "scf.while.*x !pod"`. Any surviving pod-typed `scf.while` carry
  means `flattenPodArrayWhileCarry` skipped that loop. Downstream Phase 5
  nondets the cross-iteration `pod.read [@a]`, the resulting
  `function.call @<Sub>::@compute(%nondet, %nondet)` lowers to `XOR(0,0) = 0`,
  and the parent struct.member's witness slot fills with zeros. Pair with the
  lowered StableHLO grep `func.call @<Sub>_<Sub>_compute(%cst.*=0` — both should
  be empty for a cleanly-flattened circuit.
- **Multi-carry chips need rewrite-back between consecutive same-instance
  compute calls.** Chips holding multiple input pod-arrays alive across N+ outer
  iterations of `SimplifySubComponents` (canonical case `maci_splicer` with 4
  input pod-arrays) emit 2 compute calls per loop iter (one per `<==` input
  write); without a `dynamic_update_slice %iterArg` between calls, the 2nd call
  reads `[0, latest_write]` instead of `[1st_write, 2nd_write]`. Sister
  single-carry chips (`maci_quin_selector`) emit the rewrite-back natively.
  Diagnostics (BOTH must be 0):
  `grep -cE 'func.call @<Sub>_.*compute\(%cst' <chip>.stablehlo.mlir` (blanket
  nondet of load-bearing `pod.read [@in]`); and
  `grep -nE 'dynamic_update_slice %iterArg_<input-pod-shaped-args>'`
  (accumulator never rewired — runtime bug even when the first metric is clean).
  Structural fix surface lives in three coupled callsites:
  `eraseDeadPodAndCountOps` (guard rewrite-back chain across phases),
  `flattenPodArrayWhileCarry` (resort `fieldOrder` to record-declaration order),
  and the post-`runOnOperation` rewire (contiguous-mixed-type pre-pass before
  homogeneous-sub-run match).
- **`llzk.nondet` dispatch pod (no `pod.new`) is a silent miscompile.** Earlier
  circom-llzk emits `pod.new {@count = const_N}`; post project-llzk/circom PR
  #390 the same pod can come in as
  `llzk.nondet : !pod.type<[@count: index, ...]>`. Inside the input-collection
  scf.while, `pod.read [@count]` yields garbage, the buried
  `function.call @<Sub>::@compute(...)` never fires, and `@main` fills with
  const-0. Diagnostic: post-`--simplify-sub-components` lowered StableHLO
  `@main` body of only `stablehlo.constant ... 0` + reshapes +
  dynamic_update_slice with no `func.call` ⇒ dispatch elimination silently
  dropped the call. `materializeScalarPodCompField`'s candidate filter must
  include `llzk.nondet` alongside `pod.new`.
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
  `eliminatePodDispatch`.** The outer `while (changed)` re-runs all phases plus
  `processNested` until no pass returns `true`. If your transform produces IR
  that `eliminatePodDispatch` keeps re-modifying every iteration, the loop never
  settles. Symptom: unit tests pass on toy IR but CI hangs for tens of minutes
  on real circuits. Diagnostic: wrap the outer loop with an iter counter + abort
  \> 50, log each phase's `changed` per iter — the phase still returning `1`
  after the others settle is the broken one. Emit IR that's idempotent under all
  five `eliminatePodDispatch` phases (extract calls / replace reads / erase
  writem / erase dead pods / replace remaining), or rely on the existing outer
  fixed-point + `processNested` to revisit nested constructs across iterations
  rather than recursing inside your own helper.
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

See [`docs/CIRCUIT_COVERAGE.md`](docs/CIRCUIT_COVERAGE.md) for how a
frontend/LLZK mismatch surfaces at the user-visible level (per-circuit
pass/fail, per-stage error categories).

## Load-Bearing Invariants

Cross-cutting assumptions the pipeline relies on. Each can be violated by an
innocent-looking change without breaking the build — the result still compiles,
runs on GPU, and silently produces wrong witnesses. Test changes in the
neighborhood against circom's C++ witness, not against the lowered IR alone.

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
- **Witness-output orphan-wire detection runs by default
  (`flag-orphan-zero-writes=true`).** The assertion in
  `StructPatterns.cpp:StructWriteMPattern::matchAndRewrite` aborts conversion
  when a `struct.writem`'s value resolves (through `lookThroughCast` +
  `stablehlo::ReshapeOp` + `stablehlo::ConvertOp`) to a splat-zero
  `stablehlo::ConstantOp` of length >= 8. Opt out for diagnostic spelunking with
  `flag-orphan-zero-writes=false`.
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

**Multi-sub-component composite chips ⇒ gate against a checked-in golden LLZK,
NOT against live circom output.** project-llzk/circom emits non-deterministic
`struct.member` ordering per process when the main `struct.def` has ≥2
sub-component-derived members (suspected default Rust `HashMap` `RandomState`;
single-host per-process variance proven 2026-05-04 — 3 fresh runs, 3 distinct
LLZK md5s). Position-based `.json.gate` files map GPU output offsets to `.wtns`
wire indices via `struct.member` declaration order; live circom drifts the gate
per build. For these chips, commit a frozen `examples/<chip>_llzk.llzk.golden`
(generate once with
`bazel build --disk_cache= --config=cuda_clang_official //examples:<chip>_llzk`
and `cp` from `bazel-bin/`) and use the `golden_llzk_to_stablehlo` macro in
`examples/e2e.bzl`. Single-output chips (`@out` only) are immune. Pinning a CI
runner does NOT stabilize layout — every fresh `circom` reseeds. Post-emission
canonicalization in `llzk-to-shlo` is insufficient — `scf.while` iter-arg +
`array.new` orderings shuffle under the same seed and break the lowering's
position-based pattern match (`tryClaimRun` et al.).

`//bench/m3:circom_determinism_tripwire_test` runs `circom` twice and asserts
NOT byte-equal — passes today, FAILS once upstream fixes determinism. A red
tripwire is the signal to retire the golden indirection (delete `.llzk.golden`,
switch back to `circom_to_stablehlo`, drop the `golden_llzk_to_stablehlo` macro
and the tripwire test).

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
