# Load-Bearing Invariants

Cross-cutting assumptions the pipeline relies on. Each can be violated by an
innocent-looking change without breaking the build — the result still compiles,
runs on GPU, and silently produces wrong witnesses. Test changes in the
neighborhood against circom's C++ witness, not against the lowered IR alone.

## Correctness gate hierarchy

- **Sister-circuit families (AES variants, SHA variants, etc.) are NOT
  differential references for each other.** Treating one as "the working one"
  because it diverges from a known-broken sibling is unsound — both can be
  silently broken in different output positions. Always pin the reference
  against an independent backend (circom `.wtns` via the m3 correctness gate)
  before treating it as ground truth.
- **`@constrain` functions and all `constrain.eq` ops are erased during
  lowering.** GPU code only runs witness generation; constraint satisfaction is
  the prover's job downstream. The lowered StableHLO has no way to catch a
  broken constraint.
- **Circom's `<==` vs `<--` is a security boundary.** `<==` emits both
  `@compute` and `@constrain`. `<--` emits only `@compute` and *requires* a
  separate explicit constraint (e.g. `Num2Bits` must add
  `out[i] * (out[i] - 1) === 0` after `out[i] <-- (in >> i) & 1`). Dropping the
  follow-up constraint does not fail any test in this repo — it just makes the
  circuit unsound in the prover.
- **Structural IR cleanup ≠ runtime fix on its own.** Closing the iter-arg chain
  at SimplifySubComponents (lowered `func.call (%cst, %cst)` count → 0) is
  necessary but NOT sufficient. Always re-run the GPU correctness gate AFTER
  each structural improvement. When structural metrics improve without a runtime
  delta, the data-flow disconnect is downstream — typically `LlzkToStablehlo`
  main conversion, the 3 vectorization phases, or `BatchStablehlo`.

## Batching invariants

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

## Lowering equivalences worth knowing

- **`array.new` (no operands) and `llzk.nondet` lower to the same `dense<0>`
  constant.** `ArrayPatterns.cpp:ArrayNewPattern` and
  `RemovalPatterns.cpp:LlzkNonDetPattern` both call
  `tc.createConstantAttr(tensorType, 0, rewriter)`. Switching one for the other
  in `SimplifySubComponents` is a no-op at lowering time. Before chasing a
  refactor on "orphan nondet" or "wrong init source", verify the lowered
  StableHLO is actually different (run the gate, observe whether bytes flip). If
  not, the bug is downstream — likely in witness-output assembly.
- **Void mutation ops are SSA-ified in TWO places with separate filters.**
  `processBlockForArrayMutations` (`LlzkToStablehlo.cpp:479`) handles
  mutations inside `scf.while` / `scf.if` / `scf.execute_region` bodies and
  gates `array.insert` / `array.extract` behind an `includeInsertExtract`
  flag. `convertWritemToSSA` (`LlzkToStablehlo.cpp:2145`) handles
  function-body-scope mutations and uses an explicit op-name allowlist
  (`struct.writem`, `array.write`, `array.insert`). A divergence between
  these allowlists is the canonical "void mutation silently dropped" hazard:
  the lowering pattern in `ArrayPatterns.cpp:replaceWithDUS` creates a fresh
  `stablehlo.dynamic_update_slice` for void inserts, but if upstream SSA-
  fication didn't fire, downstream consumers still reference the original
  unmodified array and the new DUS gets DCE'd. When adding a new void
  mutation op, update BOTH passes; when chasing a dropped-write bug, audit
  BOTH filters before declaring the root cause. Caveat: function-parameter
  destination arrays are short-circuited by `isPromotableCarryType` after
  `convertAllFunctions` (line 2452) morphs their types to `tensor` — that's
  a known limitation, not a bug in the allowlist.

## Witness layout & verifier asymmetries

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
  [`WITNESS_LAYOUT_ANCHOR.md`](WITNESS_LAYOUT_ANCHOR.md).
