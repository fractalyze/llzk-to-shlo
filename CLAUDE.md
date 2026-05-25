# llzk-to-shlo

`llzk-to-shlo` lowers LLZK circuit IR into StableHLO so that ZK witness
generation rides ML-compiler infrastructure and runs batched on GPU.

**Orientation:** start at [`docs/README.md`](docs/README.md) — the narrative
spine that lays out the design bets, the pipeline shape, the contracts, and the
traps.

## Rules at every keystroke

These are the three rules an agent touching this codebase will hit constantly.
Each one is summarized in one paragraph below; the full explanation lives in the
linked doc.

**1. Match LLZK and MLIR ops by type, never by mnemonic string.** Use the typed
C++ API (`isa<>`, `dyn_cast<>`, `llvm::TypeSwitch`); never write
`op->getName().getStringRef() == "mnemonic"`. A renamed or misspelled mnemonic
compiles without error and silently never matches — the pattern becomes a no-op
and produces wrong witnesses with no build failure. The struct dialect's C++
namespace is `llzk::component` (not `struct`); LLZK `function.call` is
`llzk::function::CallOp`, distinct from upstream `func::CallOp`. →
[`docs/contracts/upstream-llzk-drift.md`](docs/contracts/upstream-llzk-drift.md)

**2. Verify against the lowered StableHLO, not the simplified LLZK.** Many
LLZK-level "fixes" are absorbed by downstream conversion phases. The bytes that
actually run on the GPU are what the correctness gate compares — when the LLZK
changes but the lowered MLIR is identical, the upstream fix didn't earn its
keep. →
[`docs/contracts/correctness-gate.md`](docs/contracts/correctness-gate.md)

**3. Structural IR cleanup is necessary but not sufficient.** Closing an
iter-arg chain at SimplifySubComponents (lowered `func.call (%cst, %cst)` count
→ 0) is a real improvement, but always re-run the correctness gate afterward.
Structural metrics improving without a gate delta means the data-flow disconnect
is downstream — in `LlzkToStablehlo` main conversion, in a vectorization phase,
or in `BatchStablehlo`. →
[`docs/contracts/correctness-gate.md`](docs/contracts/correctness-gate.md)

## Map of the docs

- [`docs/design/`](docs/design/philosophy.md) — the four bets and why the
  pipeline is shaped this way.
- [`docs/passes/`](docs/passes/simplify-sub-components.md) — per-pass mechanics:
  SimplifySubComponents, LlzkToStablehlo, BatchStablehlo.
- [`docs/contracts/`](docs/contracts/correctness-gate.md) — correctness gate,
  witness-layout-anchor, upstream LLZK drift.
- [`docs/development/`](docs/development/conventions.md) — conventions, the
  correctness-gate harness and fixture, CI and build, dev-time gotchas.
