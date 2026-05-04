# Witness-Layout-Anchor (`wla`) Dialect

> Status: **spec only**. Two passes consume this dialect, both in follow-up PRs:
> `--witness-layout-anchor` emits `wla.layout` at pipeline start;
> `--verify-witness-layout` consumes it at pipeline end. Neither ships in this
> PR.

## Why this exists

Today, `llzk-to-shlo`'s witness-output assembly reconstructs the canonical
circom witness layout from whatever IR survives `simplify-sub-components`. When
an upstream pass orphans a `struct.member`'s contributing SSA — silently,
because the failure mode is a `stablehlo.constant dense<0>` of the right shape —
the output looks correct on paper and disagrees with circom only at runtime,
after the gate test. PR #67 added a heuristic loud-failure (`length >= 8`
splat-zero ⇒ abort) which catches the most common case but cannot distinguish a
*legitimately* zero member from an orphaned one.

The `wla` dialect carries the canonical layout as a *spec*: one `wla.layout` op
at module scope holds the offset and length of every signal, sourced from the
LLZK input *before* any rewriting. The lowering becomes machine-checkable: every
chunk in the final `stablehlo.dynamic_update_slice` chain must source as the
spec requires, and `dense<0>` for a slot the spec marks live ⇒ build fails with
an exact diagnostic naming the missing signal.

## Scope of this PR

- New dialect `wla` (`llzk_to_shlo/Dialect/WLA/`).
- One op: `wla.layout`.
- One attribute: `#wla.signal`.
- One enum: `wla.kind` over `output | input | internal` (printed as bare keyword
  inside the `#wla.signal` attribute).
- Op-level verifier: each signal has `length > 0` and `offset >= 0`.
- Roundtrip + verifier-diagnostic LIT fixtures under `tests/Dialect/WLA/`.
- Dialect registered in `llzk-to-shlo-opt`.

Explicitly **not** in scope (these ship in their own follow-up PRs):

- The `--witness-layout-anchor` pass that emits `wla.layout`.
- The `--verify-witness-layout` pass that consumes it.
- Pipeline wiring in `examples/e2e.bzl`.
- Removal of the `flag-orphan-zero-writes` opt-in option from PR #67; that flag
  becomes redundant once verify-witness-layout ships.

## Op shape

```mlir
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out",      output,   offset = 1, length = 128>,
    #wla.signal<"%arg0",     input,    offset = 129, length = 128>,
    #wla.signal<"%arg1",     input,    offset = 257, length = 1920>,
    #wla.signal<"@xor_1",    internal, offset = 2177, length = 128>
  ]
}
```

`wla.layout` has no operands, no results, and no runtime semantics. It is opaque
to `simplify-sub-components` and `llzk-to-stablehlo`: neither pass has patterns
matching `wla.*`, and module-scope ops outside the conversion target survive
partial conversion unchanged. Preservation requires no extra plumbing.

### `#wla.signal` fields

| Field    | Type   | Meaning                                                                                                                                                                                                                                                                                                                    |
| -------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`   | string | Reference into the LLZK input. By convention: leading `@` for `struct.member` symbols (`"@out"`, `"@xor_1"`); leading `%` for function parameters (`"%arg0"`); literal `"const_one"` for circom's reserved const-1 wire. The op verifier does **not** enforce this convention — it is the anchor pass's emission contract. |
| `kind`   | enum   | One of `output`, `input`, `internal`. Lets the verify pass apply class-specific source checks (e.g. inputs must source from a function-parameter slice).                                                                                                                                                                   |
| `offset` | i64    | Zero-based start position in the flat witness tensor.                                                                                                                                                                                                                                                                      |
| `length` | i64    | Number of `pf` (prime field) elements the signal occupies.                                                                                                                                                                                                                                                                 |

## Cross-entry invariants

The op verifier handles per-entry sanity only (`length > 0`, `offset >= 0`). The
following invariants are the anchor pass's emission contract and the verify
pass's enforcement target — documented here, not enforced in the op verifier, so
the anchor pass can ship incrementally:

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
   in `output*, input*, internal*` order (mirroring circom's
   `[const, outputs, inputs, internals]` flat layout convention).

## Reuse pointers for the follow-up passes

- **Flat sizing.** The anchor pass MUST agree with the lowering on per-member
  flat size. Reuse `getMemberFlatSize` from
  `llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h` (already public
  after PR #67); do not reimplement.
- **Chunk walking.** The verify pass walks the terminal `dynamic_update_slice`
  chain in `@main` and matches each chunk against the layout. The chunk-walking
  helpers (`collectChunks` / `extractScalarConstant` / `lookThroughReshapes` /
  `describeSourceOp` / `ChunkInfo`) currently live anonymous-namespace-private
  in `tools/witness_layout_audit.cc` (PR #68); the verify pass should lift these
  into a shared header (e.g. `llzk_to_shlo/Util/WitnessChunkWalker.{h,cpp}`)
  rather than copy.
- **Splat-zero detection.** Reuse `isZeroSplatConstant` from `TypeConversion.h`
  (PR #67). The same helper backs the existing `flag-orphan-zero-writes`
  heuristic and the audit tool.

## Background

The architectural rationale (why the witness-output assembly stage is the wrong
place to reconstruct the layout, and why a spec-carrying dialect dissolves the
silent-`dense<0>` failure mode) is in this repo's `CLAUDE.md` under "Pod-array
iter-arg survival post-simplify is a silent miscompile signal" and
"Result-bearing scf.if with tracked-array result slots + `%nondet_*` branch
yields breaks the carry chain". Wave 1 PRs that built up to this dialect: PR #67
(loud-failure assertion + helpers), PR #68 (audit tool + chunk walker), PR #70
(upstream `@num2bits_*` fix).
