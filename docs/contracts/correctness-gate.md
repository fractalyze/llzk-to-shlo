# The correctness gate

`@constrain` functions and all `constrain.eq` ops are erased during lowering.
GPU code only runs witness generation; constraint satisfaction is the prover's
job downstream. The lowered StableHLO has no internal alarm for a miscompile —
circom's native C++ witness is the only external catch. This is why the design
philosophy (see [design/philosophy.md](../design/philosophy.md)) treats GPU
byte-equality against circom's `.wtns` as the authoritative correctness signal,
not the IR itself.

## The gate hierarchy

Sister-circuit families (AES variants, SHA variants, etc.) are NOT differential
references for each other. Treating one family member as "the working one"
because it diverges from a known-broken sibling is unsound — both can be
silently broken in different output positions. The only sound reference is an
independent backend: circom's native C++ `.wtns` file, compared at the m3
correctness gate.

The gate checks two things: `batch[i] == single[i]` (the batching pass did not
diverge the per-proof output from a single-proof run), and byte-equality between
the GPU output and the circom `.wtns` at the positions the gate sentinel
specifies. LIT and unit tests prove IR shape — they cannot catch a miscompile
because the IR can be structurally well-formed while computing a wrong witness
value.

One specific security boundary to keep in mind: circom's `<==` emits both
`@compute` and `@constrain`; `<--` emits only `@compute` and requires a separate
explicit constraint (e.g. `Num2Bits` must add `out[i] * (out[i] - 1) === 0`
after `out[i] <-- (in >> i) & 1`). Dropping the follow-up constraint fails no
test in this repo — it just makes the circuit unsound at the prover. Lowering
faithfully preserves the `@compute` path either way; the distinction matters
only at the constraint layer, which the GPU never sees.

For the mechanics of the m3 gate — fixture JSON schema, `.wtns` layout, sentinel
format, enrollment, and cache invalidation — see
[development/correctness-gate-harness.md](../development/correctness-gate-harness.md)
and
[development/correctness-gate-fixture.md](../development/correctness-gate-fixture.md).

## Two rules at every keystroke

**Rule 1: verify against the lowered StableHLO, not the simplified LLZK.** Many
LLZK-level changes are absorbed by downstream conversion phases. The bytes that
matter are the ones the GPU runs. When the LLZK changes but the lowered MLIR is
byte-identical, the upstream change did not earn its keep — the real bug is
downstream.

This is a frequent source of wasted sessions: a structural improvement in the
simplified LLZK looks correct and the LIT suite stays green, but the lowered
StableHLO is identical and the gate still fails. The conversion in
[passes/llzk-to-stablehlo/README.md](../passes/llzk-to-stablehlo/README.md) runs
several pre-passes, a main partial conversion, and post-passes — any of those
phases can absorb or mask an upstream LLZK-level change.

**Rule 2: structural IR cleanup is necessary but not sufficient.** Closing the
iter-arg chain at SimplifySubComponents (driving the lowered
`func.call (%cst, %cst)` count to 0) is a required precondition for a correct
lowering, but it does not guarantee the gate passes. Always re-run the gate
after a structural change. When structural metrics improve without a gate delta,
the data-flow disconnect is downstream — typically the
[SimplifySubComponents](../passes/simplify-sub-components.md) pass, the main
[LlzkToStablehlo](../passes/llzk-to-stablehlo/README.md) conversion, the three
vectorization phases, or `BatchStablehlo`.
