# Pipeline shape

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

## The two core passes

**`SimplifySubComponents`** eliminates LLZK pod dispatch patterns and flattens
component calls into direct function calls, leaving clean LLZK that the
type-aware conversion pass can process. Mechanics are in
[`../passes/simplify-sub-components.md`](../passes/simplify-sub-components.md).

**`LlzkToStablehlo`** is the main lowering pass: it converts LLZK ops to
StableHLO equivalents, handling mutable arrays, loop-carry promotion, and
post-conversion structural transformations. Mechanics are in
[`../passes/llzk-to-stablehlo/README.md`](../passes/llzk-to-stablehlo/README.md).

## Why the splits exist

Four splits look arbitrary from the code but each exists because merging the
adjacent stages would break correctness.

**1. Pod dispatch elimination is mandatory, not optional cleanup.** The ~40-line
LLZK pod state machine that Circom emits for each `component` call crosses
multiple blocks via an input counter, pending-inputs record, and a conditional
`function.call` gated on `count == 0`. Conversion patterns in `LlzkToStablehlo`
match one or a small cluster of ops at a time; they cannot reliably match across
this multi-block boilerplate. The stage boundary is not a refactoring
convenience — it is a structural precondition for pattern matching. See
[`../passes/simplify-sub-components.md`](../passes/simplify-sub-components.md)
for pod dispatch phase details.

**2. `SimplifySubComponents` runs to a fixed point because component nesting is
arbitrary.** Circuits like `GreaterThan → LessThan → Num2Bits` have pod layers N
deep; each fixed-point iteration peels one layer. Within a single iteration,
phases that early-return for internal-state safety (specifically
`unpackPodWhileCarry`, which invalidates its inner `SmallVector` when it erases
a chained-while inline) need their own inner `while (phase(block))` loop at the
driver site when a same-iteration later phase consumes their complete output.
The outer fixed point is too coarse: with N independent candidates the outer
loop processes only the first before destructive later phases
(`eliminatePodDispatch` Phase 5) run on an incomplete intermediate state. See
[`../passes/simplify-sub-components.md`](../passes/simplify-sub-components.md).

**3. While-loop transformation is four phases because LLZK is mutable, StableHLO
is SSA, and loop bodies mutate outer arrays.** An `array.write %outer[%i]`
inside `scf.while` must flow through carry tuples to lower to a functional
`stablehlo.while`. The four phases — array-to-carry promotion, SSA conversion of
`array.write`/`struct.writem`, main partial conversion,
`scf.while → stablehlo.while` — must happen in this order: you cannot SSA-ify
writes before you know which arrays are carry candidates; you cannot do the main
conversion before writes are SSA; you cannot convert the while shell before the
body is in StableHLO. The shared walker `processBlockForArrayMutations` is
reused with different `latest` trackers per pre-pass — it must gate mutating ops
on the target array being currently tracked, or untracked writes get rewritten
as orphan ops and silently DCE'd. See
[`../passes/llzk-to-stablehlo/while-loop-transformation.md`](../passes/llzk-to-stablehlo/while-loop-transformation.md).

**4. Post-passes exist because `applyPartialConversion` does 1:1 op replacement,
not region restructuring.** Converting a `scf.while` shell to `stablehlo.while`,
reconnecting `func.call` results to cross-block `pod.read @comp` consumers, and
vectorizing iteration-independent loops all involve moving or deleting regions —
operations that partial conversion's 1:1 replacement model cannot express. These
transformations must run after the main conversion completes. See
[`../passes/llzk-to-stablehlo/README.md`](../passes/llzk-to-stablehlo/README.md)
for post-pass ordering and invariants.
