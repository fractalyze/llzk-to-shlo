# Witness-layout-anchor (wla)

The `wla` dialect carries the canonical witness layout as a machine-readable
spec. Without it, the lowering reconstructs the circom witness layout from
whatever IR survives `simplify-sub-components`. When an upstream pass orphans a
`struct.member`'s contributing SSA — silently, because the failure mode is a
`stablehlo.constant dense<0>` of the right shape — the output looks correct on
paper and disagrees with circom only at runtime, after the gate. The `wla`
dialect dissolves this failure mode: `wla.layout` captures offset and length of
every signal from the LLZK input _before any rewriting_, and
`--verify-witness-layout` enforces that the final `dynamic_update_slice` chain
matches it exactly. A `dense<0>` on a slot the spec marks live becomes a build
failure with an exact diagnostic naming the missing signal, rather than a silent
wrong-witness.

## Op shape

`wla.layout` has no operands, no results, and no runtime semantics. It lives at
module scope and is opaque to both `simplify-sub-components` and
`llzk-to-stablehlo`: neither pass has patterns matching `wla.*`, and
module-scope ops outside the conversion target survive partial conversion
unchanged.

```mlir
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out",      output,   offset = 1, length = 128>,
    #wla.signal<"%arg0",     input,    offset = 129, length = 128>,
    #wla.signal<"%arg1",     input,    offset = 257, length = 1920>,
    #wla.signal<"@xor_1",   internal, offset = 2177, length = 128>
  ]
}
```

The `#wla.signal` attribute has four fields:

1. **`name`** — a reference into the LLZK input. Convention: `@`-prefixed for
   `struct.member` symbols, `%`-prefixed for function parameters, literal
   `"const_one"` for circom's reserved constant-1 wire. The op verifier does not
   enforce this convention; it is the anchor pass's emission contract.
1. **`kind`** — one of `output`, `input`, or `internal`. The verify pass applies
   class-specific source checks: inputs must source from a function-parameter
   slice; `dense<0>` is rejected on `output` signals but permitted on `internal`
   signals (a legitimately zero internal is indistinguishable from an orphan at
   the lowering boundary, so the m3 byte-equality gate is the actual correctness
   check there).
1. **`offset`** — zero-based start position in the flat witness tensor.
1. **`length`** — number of prime-field elements the signal occupies. The op
   verifier enforces `length > 0` and `offset >= 0`; everything else is the
   anchor pass's emission contract.

## Cross-entry invariants

The following invariants are the anchor pass's emission contract and the verify
pass's enforcement target. They are documented here rather than enforced in the
op verifier so the anchor pass can ship incrementally.

1. **Single instance per module.** Exactly one `wla.layout` op, child of the
   top-level `ModuleOp`.
1. **Sorted by offset.** The `signals` array is sorted ascending on `offset`
   (binary-search-friendly).
1. **No overlap.** For adjacent entries `a, b`,
   `a.offset + a.length <= b.offset`.
1. **Optional const-one head.** When present, the first entry is the constant-1
   wire: `name = "const_one"`, `kind = internal`, `offset = 0`, `length = 1`.
   Whether circom emits this slot is chip-dependent; the anchor pass mirrors
   what the LLZK input declares.
1. **Canonical block order.** After the optional const-one entry, signals appear
   in `output*, input*, internal*` order, mirroring circom's
   `[const, outputs, inputs, internals]` flat layout convention.

**`getMemberFlatSize` recursion.** The anchor pass must agree with the lowering
on per-member flat size. For `<N x !struct<Inner>>` the helper returns
`N * recursive_flat(Inner)`, where `recursive_flat(Inner)` sums each
writem-targeted, non-pod member's flat size recursively. The recursion requires
`ModuleOp` access to look up inner struct defs, and a `visited` set guards
against self-referential types. The `--stabilize` flag pinned in
`examples/e2e.bzl` guarantees inner struct leaf names are unique within a chip
module, which is what makes the flat walk-and-match lookup correct. Reuse
`getMemberFlatSize` from
`llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h` rather than
reimplementing it.

**`MemberReadOp` pub-from-outside rule.** `struct.readm` (`MemberReadOp`) on a
non-pub member is illegal from outside the member's parent struct:
`MemberReadOp::verifySymbolUses` rejects the read when the enclosing op's parent
struct differs from the member's defining struct and the member lacks
`{llzk.pub}`. The WLA layout, by contrast, exposes writem-targeted members in
the witness footprint regardless of pub-ness. This asymmetry is intentional —
the witness layer treats the slots as observable (they appear in the output),
while the struct verifier enforces encapsulation. The bridge is **member
promotion**: any pass that needs to emit `struct.readm %callResult[@<field>]`
from a parent struct's `@compute` must first set `PublicAttr::name` on the inner
struct.def's matching `struct.member` op. Promotion is semantically aligned with
the WLA contract already in effect — the slots were already observable in the
witness; pub makes the LLZK verifier agree. WLA's own pub/internal kind
distinction operates on the main struct's members, not on inner struct members,
so promoting inner writem-targets is invisible to WLA's signal kind
classification.

**Pass ordering.** `--witness-layout-anchor` must run after
`--simplify-sub-components`. Running before SSC trips upstream's
`applyFullConversion`, which rejects unknown ops. `--verify-witness-layout` runs
after `--llzk-to-stablehlo` and asserts each `dynamic_update_slice` chunk in
`@main` matches a `wla.layout` entry.

## Background

Before the `wla` dialect, witness-output orphan detection relied on a heuristic
in `StructPatterns.cpp:StructWriteMPattern`: any `stablehlo.constant dense<0>`
splat with `length >= 8` triggered an abort. This catches the most common orphan
case, but cannot distinguish a legitimately zero member from one whose
contributing SSA was silently dropped. The heuristic survives as
`flag-orphan-zero-writes=true` for diagnostic spelunking and regression
fixtures, and becomes redundant once `--verify-witness-layout` ships.

The architectural reason the witness-output assembly stage is the wrong place to
reconstruct the layout is that it must infer layout from whatever IR survived
the rewriting passes — passes that may have silently dropped a contributing SSA
chain. Carrying the layout forward from the LLZK input as an explicit spec
shifts the invariant from "infer at output time" to "verify at output time
against the pre-rewriting record".
