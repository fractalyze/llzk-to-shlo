# While-loop transformation

LLZK's `scf.while` is mutable: loop bodies can write to arrays defined outside
the loop. StableHLO's `stablehlo.while` is functional: every value that crosses
a loop boundary must be an explicit carry argument. Bridging this requires four
ordered phases. The ordering is forced — collapsing them or running them in any
other sequence silently miscompiles or produces unverifiable IR.

## Step 1: promote captured arrays to while carry

Arrays defined outside a `scf.while` but modified inside are detected and added
as carry arguments. The init value is the array at loop entry; the body receives
it as a new block arg; each write yields the updated tensor; the yield
propagates it to the next iteration.

Domination safety governs the promotion boundary: arrays defined in a parent
while body are not promoted. The parent body's block args are not in scope at
the inner while's init-value evaluation site (which executes before the body
runs), so promoting them would produce an SSA use before definition.

`array.write` is converted from a void in-place op to a value-returning op
during this step. SSA-ification of writes requires the carry to exist first —
the write's return value is what flows to the yield; without the carry slot
there is nowhere for the chain tip to go.

## Step 2: mutable to SSA conversion

`processBlockForArrayMutations` is the shared walker used by both
`convertArrayWritesToSSA` (step 1's companion) and `convertWhileBodyArgsToSSA`.
It is reused with different `latest` trackers per pre-pass. The critical
invariant:

**The walker MUST gate mutating ops on the target being currently tracked, or
untracked writes get rewritten orphan and silently DCE'd.**

If the walker rewrites an `array.write` targeting an untracked array, it emits a
new `array.write` whose result has no downstream consumer — the carry slot for
that array doesn't exist. The orphaned write gets eliminated by DCE, and the
array silently retains its initial value for every subsequent iteration. This
produces wrong witness output with no IR-level error.

The `latest` map determines which arrays are tracked. Walker 1
(`convertArrayWritesToSSA`) and Walker 2 (`convertWhileBodyArgsToSSA`) must
agree on the state of SSA-ified chain links. If Walker 2's `latest` doesn't
reflect chains Walker 1 already built, it re-SSA-ifies the same position and the
earlier chain tip gets orphaned.

`struct.writem` SSA conversion is a separate pre-pass (`convertWritemToSSA`) for
function-body top-level writes, because the scope (function body vs while body)
determines which `latest` tracker applies. Both must be updated when adding a
new void mutation op — see `op-lowering.md` for the two-allowlist invariant.

## Step 3: scf.while to stablehlo.while (post-pass)

After `applyPartialConversion` has converted the body ops to StableHLO, the
while shell itself is converted. The pre-pass phase (steps 1–2) left the shell
as `scf.while` because `applyPartialConversion` does 1:1 op replacement only —
it cannot change block argument layouts. Nested whiles are converted
innermost-first to avoid invalidating region pointers during the walk.

The structural change: `scf.condition(%cond) %carry...` splits into a separate
condition region that returns only the predicate, and a do region that returns
the updated carry. StableHLO's while has this two-region shape; LLZK's scf.while
has a single fused region.

## Post-passes: scf.if to stablehlo.select; vectorization

### scf.if → stablehlo.select

StableHLO has no conditional branching. Both branches are inlined and
`stablehlo.select` chooses the result. This cannot be a 1:1 conversion pattern
because it moves ops between regions — the `then` and `else` bodies are hoisted
into the parent block. It runs as a post-pass after `applyPartialConversion` so
the ops being inlined are already in StableHLO form.

### Vectorization (three phases)

Three auto-vectorization patterns are applied after DCE:

1. **1D element-wise**: `while(i<N) { out[i] = f(a[i], b[i]) }` →
   `out = f(a, b)`. Safe because circom signals are immutable — `out[i]` is
   written once and never reassigned. The canonical iterative shape is a
   `chain[N][K+1]`-style array, not a mutable accumulator.

1. **2D carry write**: `while(i<N) { acc[i, col] = f(a[i], b[i]) }` →
   column-wise vector ops. The batch dimension flows through the carry tuple.

1. **Nested 2D**: `while(j) { while(i) { ... } }` — inner while is vectorized
   independently once the outer while's carry is in canonical form.

Vectorization is sound only because Circom signals are immutable. If the
frontend emitted batch-divergent trip counts, the predicate extraction (`[0]`
element gate) would fail. See
[`../../contracts/correctness-gate.md`](../../contracts/correctness-gate.md).

## Walker traps

### Read-only inner-while capture causes phantom rebind

`findCapturedArrays` returns array captures unconditionally (needed for
keccak/AES carry semantics). An inner `scf.while` whose body only READS a
captured carrier still gets the carrier appended as a carry arg by
`promoteArraysToWhileCarry`, and the body yields the block arg directly
(passthrough). The rebind at `processBlockForArrayMutations` fires anyway:
`init == latest[%cap]` matches, so `latest[%cap]` gets rebound to
`inner.result(captured_slot)`. This rebind is semantically a no-op (passthrough
means `inner.result(i) == init` at every iter), but the SSA pointer shifts.
Downstream `array.insert`/`array.write` chains in sibling `scf.if` branches
build on `inner.result(...)`. The outer `scf.while`'s yield slot then lands on
the inner-while's result instead of the chain tip — that slot becomes a
passthrough of `dense<0>` in the lowered StableHLO.

Diagnostic: post-walker trace — `defOp=scf.while` for a block-arg-typed slot at
the explicit yield rewrite is the smoking gun. Also: grep for
`stablehlo.return.*%iterArg_<N>` at the outer while's terminator — a passthrough
where mutations were expected.

### Result-bearing `scf.if` with tracked-array slots and `%nondet_*` yields

LLZK's `<--` can produce `scf.if` ops whose array result slots are yielded as
`llzk.nondet` placeholders in both branches; the actual writes happen via inner
whiles using the parent's tracked carry as init. `liftScfIfWithArrayWrites`
early-returns on `getNumResults() != 0` and handles void ifs only —
result-bearing ifs go through `extendResultBearingScfIfArrayChain`, which must
append NEW tail result slots (typed `!array<x !felt>`, matching the tracked key
type) rather than rewriting existing slots. The two invocation invariants:

1. Idempotent across the dual-walker invocations — the second walker must
   observe the first walker's appended slots without double-adding.
1. Reuse paths must reference `newIf.getResult(i)`, not `oldIf.getResult(i)` —
   `oldIf` is erased on append.

### `processBlockForArrayMutations` must dispatch on `scf.execute_region`

Without an explicit `scf.execute_region` handler, the region is opaque to the
walker. SSC's `materializeStructOfPodsCompField` wraps deep K-dispatch cascades
(e.g. iden3 Poseidon3's 56-deep `MixS_*` cascade) inside
`scf.execute_region → (!array<K x !pod>)`. Every inner cascade `scf.if` is
invisible to `extendResultBearingScfIfArrayChain`, and every
`array.insert %carrier[%cK]` inside is left behind — the carrier chain silently
drops cascade-arm writes. Mirror `extendResultBearingScfIfArrayChain` via
`extendExecuteRegionArrayChain`, appending NEW tail slots without touching the
pod-typed yield that downstream pod-dispatch elimination still expects.

### `processNested` recurses into `scf.while` only, not `scf.if`

Pod-array-carrying `scf.while`s buried inside `scf.if` branches are invisible to
the main driver. A post-main-loop straggler pass uses
`module.walk(scf::WhileOp)` to reach them. Do NOT add `scf.if` recursion to
`processNested` itself — that path also runs `eliminatePodDispatch`, whose pod
block-arg gating assumes `scf.while` semantics; nesting an `scf.if` recursion
would double-apply dispatch elimination on already-flattened blocks.

### `collapseRedundantWhileCarrierPairs` zero-init transitivity

The pass classifies a `stablehlo.while` slot as DEAD when its yield is a literal
passthrough of the body argument and its init traces back through
enclosing-while body-args to a zero-splat constant — then RAUWs the dead result
to a sibling LIVE result of identical type. The DEAD-collapse semantics hold at
a given while only if no intermediate enclosing while mutates the carrier. A
parent while whose yield differs from its body arg at the same slot index breaks
this: body args on later iterations carry the parent's mutated value, not the
init.

`isZeroSplatTransitively` must reject the trace whenever a visited parent slot
is non-passthrough. Every enclosing while in the trace must itself be a
passthrough at the same slot index for the inner RAUW to be sound.

### `collapseRedundantWhileCarrierPairs` LIVE→DEAD body-arg pairing

The original "both inits are zero-splat constants" safety argument covers only
round 0. Past iteration 0 a LIVE slot diverges from init by definition (yield ≠
body arg). RAUW'ing a DEAD reader (which expected always-zero init) onto the
LIVE result silently corrupts every downstream consumer of the DEAD slot after
the first iteration.

Pair-eligibility requires that the LIVE slot's yield transitively references the
DEAD body argument — a small DAG traversal from `returnOp.getOperand(live)`
through defining-op operands, checking whether `body.getArgument(dead)` is
reachable. Only pairs that satisfy this carry-same-logical-signal invariant (the
original AES `@xor_3$inputs[@a]/[@b]` shape) are redirected.
