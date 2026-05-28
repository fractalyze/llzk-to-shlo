# Contracts

Boundaries this project must not silently cross — the rules where a violation
shows up as wrong witnesses, not as a build failure. Each file names the
contract, its enforcement mechanism, and what breaks when it slips.

- [`correctness-gate.md`](correctness-gate.md) — Witness equality vs circom's
  `.wtns` is the only catch for silent GPU miscompiles. LIT and unit tests prove
  IR shape; they cannot prove witness correctness.
- [`upstream-llzk-drift.md`](upstream-llzk-drift.md) — Managing LLZK version
  pins and the silent-miscompile traps each upstream bump introduces. Match ops
  by type, never by mnemonic string.
- [`witness-layout-anchor.md`](witness-layout-anchor.md) — The `wla` dialect as
  canonical witness layout spec, anchored before lowering. Turns
  silent-orphan-SSA failures into build errors with a diagnostic naming the
  missing signal.

Parent: [`../README.md`](../README.md) (narrative spine).
