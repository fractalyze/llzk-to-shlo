# LlzkToStablehlo

`LlzkToStablehlo` is the heavy conversion pass. It receives LLZK IR that
`SimplifySubComponents` has already cleaned of pod dispatch and produces
StableHLO IR ready for GPU execution. The work is divided into three groups:
pre-passes that make the IR structurally lowerable, a main partial conversion
that replaces LLZK ops 1:1, and post-passes that restructure regions and clean
up what partial conversion cannot express. See `op-lowering.md` for type and op
mechanics; see `while-loop-transformation.md` for the four-phase while
treatment.

## Phase order and why it's forced

### Pre-passes

The pre-passes exist because `applyPartialConversion` does 1:1 op replacement
and cannot restructure regions, reorder block arguments, or look through
unprepared LLZK shapes. Their collective job is to make the IR structurally
lowerable before the main conversion sees it. Concretely they do three things:
(a) eliminate constructs that partial conversion cannot handle — `$inputs` pod
struct members, arrays defined outside a `scf.while` that are written inside
(which must become explicit carry arguments), and void `struct.writem` /
`array.write` mutation ops (which must become SSA chains before 1:1 patterns can
fire); (b) lower control structures whose results need typed wiring —
`dispatchCallHoisting` moves `function.call` ops out of `scf.if` regions before
conversion to avoid SSA domination errors, and `convertAllFunctions` turns
`function.def` into `func.func` so the carry-promotion walker can traverse
function bodies; (c) materialize layout information —
`registerStructFieldOffsets` precomputes per-struct member offsets for the
flattened tensor layout, which the `struct.readm`/`struct.writem` patterns
consume at rewrite time and cannot derive themselves. The internal order among
pre-passes is forced by producer/consumer dependencies: array-to-carry promotion
must run before SSA-ification of writes (the carry slot must exist before the
chain tip can flow to a yield), and both must run before the type-aware patterns
in the main conversion. See
[while-loop-transformation.md](while-loop-transformation.md) for the promotion
and SSA-ification mechanics.

### Main conversion

`applyPartialConversion` fires all LLZK-to-StableHLO conversion patterns in one
driver: the type converter maps `!felt.type`, `!struct.type`, `!array.type`, and
`i1` to tensor forms, and the op patterns replace each LLZK op with its
StableHLO equivalent. Everything that is region-shaped is already in canonical
form thanks to the pre-passes, so the conversion patterns only need to match
individual ops, not regions. The pass is a single driver call rather than
multiple staged calls because pattern interactions inside one fixed-point driver
let dependent rewrites converge without manual inter-group ordering — if the
patterns were split across separate `applyPartialConversion` calls, each group's
preconditions would have to be enforced externally, and the ordering would
replicate the pre/post-pass contract without any of its structural guarantees.
See [op-lowering.md](op-lowering.md) for the type and op pattern details.

### Post-passes

Post-passes exist precisely because `applyPartialConversion` is 1:1 and
structurally cannot move or delete regions. After the main conversion, the while
shells are still `scf.while` (only the body ops became StableHLO), conditionals
are still `scf.if` (StableHLO has no `if` construct), and
`unrealized_conversion_cast` chains bridge `func.call` results to their
`pod.read @comp` consumers. The post-passes resolve all of this: `scf.while` is
converted to `stablehlo.while` (innermost-first to avoid invalidating region
pointers), `scf.if` is inlined into `stablehlo.select`, cast chains from
`func.call` results to `pod.read @comp` consumers are reconnected, residual pod
and array ops that have no live consumer are replaced with zero tensors,
`arith.ori`/`arith.andi` ops synthesized by `RemovalPatterns`' i1-scalar
fallback (when `bool.or`/`bool.and` operands inside `scf.while`/`if` bodies have
not yet been tensor-converted) are lifted to their `stablehlo` equivalents, dead
code is eliminated, and finally independent while loops are vectorized in three
phases. The internal order is forced by the same producer/consumer rule:
structural transformations (`scf.while` → `stablehlo.while`, `scf.if` →
`stablehlo.select`) must complete before cleanups that assume the new region
shape (reconnection, residual erasure, DCE), and vectorization runs last because
it depends on the final SSA shape that DCE produces. See
[while-loop-transformation.md](while-loop-transformation.md) for the
`scf.while`/`scf.if` conversion and vectorization phases.
