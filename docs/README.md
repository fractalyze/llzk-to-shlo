# llzk-to-shlo

`llzk-to-shlo` lowers LLZK circuit IR into StableHLO so that ZK witness
generation rides ML-compiler infrastructure and runs batched in one GPU kernel
launch. The pipeline runs Circom source through the LLZK frontend, two core
lowering passes (`SimplifySubComponents` + `LlzkToStablehlo`), a batching pass
(`BatchStablehlo`), and then open-zkx's `stablehlo_runner` on GPU — collapsing N
independent proof inputs into a single dispatch instead of one per proof.

## The bet

Four design decisions shaped every trade-off in the codebase.

**(a) Reuse ML compiler infrastructure.** StableHLO already has GPU backends,
fusion, loop vectorization, and a production ecosystem of tooling. Expressing ZK
field arithmetic and array ops as StableHLO ops means the GPU backend, the
batching pass, and downstream optimizers all work without ZK-specific changes.
The cost is that ZK primitives must fit inside StableHLO's type system and SSA
constraints.

**(b) Batch first.** Witness generation is embarrassingly parallel: each proof
input is independent and the arithmetic per witness is fast. The dominant cost
is per-execution dispatch overhead — JIT compilation, stream allocation, host
synchronization — not the arithmetic itself. Amortizing that overhead across N
inputs makes total runtime approximately equal to one single-input execution
regardless of N. The correct optimization target is eliminating per-execution
overhead, not microoptimizing per-kernel arithmetic.

**(c) GPU correctness is the gate.** `@constrain` is erased at lowering — it
exists for the ZK verifier, not for witness computation. GPU code runs with no
internal correctness alarm: a miscompile produces wrong witnesses silently. The
only catch is circom's native C++ witness generator. LIT and unit tests prove IR
shape; they cannot substitute for the correctness gate.

**(d) Frontend-agnostic.** LLZK is the stable contract between this project and
its frontends. Pod dispatch elimination lives in a dedicated pre-pass
(`SimplifySubComponents`) because it is a Circom artifact, not an LLZK
primitive. Patterns in `LlzkToStablehlo` that are Circom-shaped rather than
LLZK-shaped are bugs.

→ [design/philosophy.md](design/philosophy.md)

## The shape

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

The four splits are each forced by a structural necessity, not a refactoring
preference: pod-dispatch elimination is a required precondition for pattern
matching (not optional cleanup); `SimplifySubComponents` runs to a fixed point
because component nesting is arbitrary; while-loop transformation is four
ordered phases because LLZK is mutable and StableHLO is SSA; post-passes exist
because `applyPartialConversion` does 1:1 op replacement and cannot express
region-level restructuring.

→ [design/pipeline-shape.md](design/pipeline-shape.md)

## The contracts you must not break

**LLZK is a moving target.** Match LLZK and MLIR ops using the typed C++ API
(`isa<>`, `dyn_cast<>`, `llvm::TypeSwitch`) — never by
`op->getName().getStringRef() == "mnemonic"`. A misspelled or renamed mnemonic
compiles without error and silently never matches, turning the pattern into a
no-op. Note that the struct dialect's C++ namespace is `llzk::component`, not
`struct`. Each upstream bump introduces 2–4 new pod/scf shapes the pattern set
hasn't seen; hand-migrate test fixtures and BUILD glue in the same change.

→ [contracts/upstream-llzk-drift.md](contracts/upstream-llzk-drift.md)

**Circom is the only catch.** GPU runs `@constrain`-erased code; a miscompile
produces wrong witnesses silently with no build failure and no LIT failure.
Sister-circuit families (AES variants, SHA variants) are not valid differential
references for each other — the only sound reference is circom's native C++
`.wtns` file, compared byte-for-byte at the m3 correctness gate. LIT and unit
tests prove IR shape, not witness correctness.

→ [contracts/correctness-gate.md](contracts/correctness-gate.md)

**Witness layout is a contract with the verifier.** The `wla` dialect captures
the offset and length of every signal from the LLZK input before any rewriting.
Without it, a pass that silently orphans a contributing SSA produces a
`stablehlo.constant dense<0>` of the right shape — which looks correct on paper
and disagrees with circom only after the gate. `--verify-witness-layout` turns
that silent wrong-witness into a build failure with a diagnostic naming the
missing signal.

→ [contracts/witness-layout-anchor.md](contracts/witness-layout-anchor.md)

## The traps

Silent miscompiles cluster in a few well-known places: walker traps inside
`scf.while`/`scf.if`/`scf.execute_region` array carry; `SimplifySubComponents`
driver-ordering (phase inner-loop discipline, fixed-point coarseness);
pod-dispatch silent-miscompile signals (surviving pod-typed iter-args, sub-call
count diffs, writerless dispatch pods); and lowering equivalences where
`array.new` and `llzk.nondet` both lower to `dense<0>`. Two rules apply at every
keystroke: verify against the lowered StableHLO, not the simplified LLZK
(downstream phases often absorb LLZK-level changes, so the bytes that matter are
the ones the GPU runs); and structural IR cleanup is necessary but not
sufficient (always re-run the correctness gate after a structural change —
structural metrics improving without a gate delta means the data-flow disconnect
is downstream).

→ [passes/simplify-sub-components.md](passes/simplify-sub-components.md),
[passes/llzk-to-stablehlo/while-loop-transformation.md](passes/llzk-to-stablehlo/while-loop-transformation.md),
[contracts/correctness-gate.md](contracts/correctness-gate.md)

## Where to dig

- [design/](design/philosophy.md) — the four bets and why the pipeline is shaped
  this way.
- [passes/](passes/simplify-sub-components.md) — per-pass mechanics: SSC,
  LlzkToStablehlo ([op lowering](passes/llzk-to-stablehlo/op-lowering.md) +
  [while-loop transformation](passes/llzk-to-stablehlo/while-loop-transformation.md)),
  [BatchStablehlo](passes/batch-stablehlo.md).
- [contracts/](contracts/correctness-gate.md) — the correctness gate,
  witness-layout-anchor, upstream-LLZK drift.
- [development/](development/conventions.md) — conventions, the correctness-gate
  harness and fixture, CI and build setup, MLIR API gotchas, APInt arithmetic,
  diagnostics.
