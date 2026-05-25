# SimplifySubComponents

Pod dispatch is the LLZK encoding of a Circom `component` call: a `!pod.type`
pending-inputs record, an input-counter, conditional `scf.if (count == 0)`
firing, and deferred result reads via `pod.read`. A single `lt.in[0] <== v1` in
Circom expands to roughly 40 lines of this state machine. `LlzkToStablehlo`
can't match across this boilerplate, so `SimplifySubComponents` must flatten it
— converting every `function.call @Sub::@compute` from conditional pod dispatch
into a direct call — before type-aware conversion begins.

## The 6 phases

The pass repeatedly applies these six phases inside a fixed-point loop until no
phase reports a change:

| Phase | Entry point                     | Why it exists                                                                  |
| ----- | ------------------------------- | ------------------------------------------------------------------------------ |
| -1    | `flattenPodArrayWhileCarry`     | Splits `array<N x pod>` iter-args into per-field arrays so Phase 0 can proceed |
| 0     | `unpackPodWhileCarry`           | Expands a pod-typed `scf.while` carry into one carry slot per field            |
| 1     | `extractCallsFromScfIf`         | Core: hoists `function.call` out of `scf.if (count == 0)` firing guards        |
| 2     | `replacePodReads`               | Replaces `pod.read` with the tracked SSA value written before the call         |
| 3     | `eraseStructWritemForPodValues` | Removes `struct.writem` on pod- or struct-typed values (no longer needed)      |
| 4     | `eraseDeadPodAndCountOps`       | Removes the now-unused pod/arith/scf.if scaffolding                            |
| 5     | `replaceRemainingPodOps`        | Nondets surviving `pod.read`; deletes orphan `pod.new`                         |

**Why fixed-point?** Components nest arbitrarily (e.g.
`GreaterThan → LessThan → Num2Bits`). Each outer iteration peels one nesting
level. A phase that must complete fully before the next phase has an invariant
worth maintaining — see Driver-ordering traps below.

**Phase 1 is the core transformation.** It tracks pod field values through the
block, finds `function.call` inside a dispatch-countdown `scf.if`, and hoists
the call to the parent block with the pod's tracked argument values substituted
as direct arguments. After Phase 1, the buried call is live at block scope and
subsequent phases can clean up the pod scaffolding.

## Code organization

The SSC implementation is split across five TUs, each exposing only its
driver-called entry points while keeping helpers file-private:

**`PodDispatchPhases.{h,cpp}`** — the single-block dispatch-elimination logic
(Phases 1–5), orchestrated by `eliminatePodDispatch`. Answers: given one
`function.def` block, how do you eliminate its pod dispatch?

**`StructOfPodsConversion.{h,cpp}`** — the struct-of-pods rewrite:
`convertStructOfPodsToArrayOfPods` folds `!pod<[@idx_0..@idx_K-1: T]>` carriers
into `!array<K x T>`, and `materializeStructOfPodsCompField` handles
non-uniform-inner carriers that the fold can't collapse.

**`PodArrayWhileCarry.{h,cpp}`** — while-carry flattening pre-passes (Phases
-1/0): `flattenPodArrayWhileCarry`, `flattenPodArrayScfIfResults`, and
`unpackPodWhileCarry`. These run before `eliminatePodDispatch` becomes
applicable because dispatch elimination assumes carried values are felt-typed.

**`PodArrayMaterialize.{h,cpp}`** — bridges a dispatched `function.call` result
to its cross-block `pod.read [@comp]`/`struct.readm` readers
(`materializePodArrayCompField`, `materializePodArrayInputPodField`,
`materializeScalarPodCompField`).

**`PodModuleCleanup.{h,cpp}`** — post-materialization module-wide cleanup:
resolving array-sourced pod reads, unwrapping `$inputs` channels, lifting
const-index post-while calls, stripping empty struct params and same-named
wrapper modules, and erasing dead pod-typed carrier slots. Eight driver-called
entry points; helpers are file-private.

**`SimplifySubComponents.cpp`** — the outer driver. Owns `runOnOperation`, the
module-wide fixed-point loop, and the sequencing of while-carry pre-passes,
`eliminatePodDispatch`, and the cleanup family. Helpers shared across TUs are
declared in `SimplifySubComponentsInternal.h` (library-private).

The split is behavior-preserving: the driver still calls
`populateExternallyLiveMembers(module)` once and drives `eliminatePodDispatch`
through the fixed-point loop exactly as before the split.

## Driver-ordering traps

**Phase 1 and Phase 5 must not run concurrently on the same block.** Phase 5
nondets every `pod.read` in a block, including reads of pod-typed `scf.while`
block args. Those reads are the field-discovery input for `unpackPodWhileCarry`
on the next outer iteration. Gate Phase 5 on the block having no pod-typed block
args; if it fires too early, the unpacker finds only nondets and multi-record
input pods (e.g. keccak's `<[@a: array, @b: array]>` carries) survive into
dialect conversion.

**`processNested` recursion must not include `scf.if` branches.**
`flattenPodArrayWhileCarry` is non-recursive; region recursion happens via
`processNested`, which dispatches on `scf.while` only. Adding `scf.if` recursion
would double-apply `eliminatePodDispatch` on already-flattened blocks (its pod
block-arg gating assumes `scf.while` semantics). Instead, a post-main-loop
straggler pass uses `module.walk(scf::WhileOp)` to reach whiles buried inside
`scf.if` branches.

**A same-iteration consumer of Phase 0 needs its own inner `while(phase(block))`
loop.** Phase 0 (`unpackPodWhileCarry`) early-returns for internal-state safety
when its inner SmallVector contains a chained-while it erased inline
(pointer-invalidation risk). The outer fixed point is too coarse with N
independent candidates: only the first candidate gets processed before
`eliminatePodDispatch` Phase 5 runs and nondets the remaining pod reads. Any
pass that must fully complete before a later same-iteration phase consumes its
output must drive its own convergence loop.

**`extractCallsFromScfIf` Phase 1 directArgs path requires dominance, not just
ancestry.** The tracker resolves hoisted-call args from tracked pod.write
values. "Is this arg available at the insertion point?" requires
`DominanceInfo::properlyDominates(arg, insertionPoint)`, not merely
`!dominanceScope->isAncestor(def)`. Ancestry is necessary but not sufficient:
sibling dispatch sites with interleaved carrier writes can have a `%v` available
in the outer block but defined positionally after the hoist target. Anchor
`DominanceInfo` on the enclosing `function.def`, not `block.getParentOp()` —
when SSC recurses into inner `scf.while` or `scf.if` bodies, `getParentOp()`
returns the inner region's owner, and values in outer enclosing regions fall
outside that root and are over-rejected.

**Phase 4 erasure must walk regions transitively for external users.**
`isAllResultsUnused(op)` is not sufficient when `op` carries inner ops whose
results are consumed by users outside `op`. Phase 1 can hoist a `function.call`
before its enclosing `scf.if` using `directArgs` — those direct args may be
defined inside the `scf.if` body. Phase 4 then sees the `scf.if` result unused,
region-clears it, and destroys the inner producer that the hoisted call still
references. Use `isOpAndNestedResultsExternallyUnused` instead.

**Phase 4 is also blind to void side-effects on outer values.**
`isOpAndNestedResultsExternallyUnused` only walks SSA results. LLZK's
`array.write` / `array.insert` / `struct.writem` produce no results, so a void
side effect on an outer-scoped array (e.g. a per-field iter-arg from
`flattenPodArrayWhileCarry`) is invisible. The carve-out at
`eraseDeadPodAndCountOps` via `hasNonPodArrayWriteInBody` is load-bearing: any
new region-bearing synthesizer that writes to outer values from inside its
region must ensure its enclosing op shape is in the carve-out's accepted-name
set and that the helper recognizes the mutating op.

**Rebuilding region-bearing ops requires a `rebuiltOps` set.** When rebuilding
`scf.while` / `scf.if` / `scf.execute_region` via `builder.create(...)` +
`newRegion.takeBody(oldRegion)`, the old op goes into `toErase` but its body
(including yield ops) now lives inside the new op. Guarding "already rebuilt?"
with `toErase.count(parent)` always returns false — the new op has a different
pointer. Maintain a parallel `DenseSet<Operation *> rebuiltOps`, populated at
each `builder.create(...)`, and probe that instead.

## Silent-miscompile signals

These miscompiles are invisible at build time and pass LIT regression; only the
correctness gate against circom's native C++ witness (`batch[i] == single[i]`)
catches them. See `../contracts/correctness-gate.md` for the gate contract.

**Surviving pod-typed iter-arg.** If `flattenPodArrayWhileCarry` skips a loop
(use-shape gate trips), the cross-iteration `pod.read [@a]` gets nondetted by
Phase 5. The resulting `function.call @Sub::@compute(%nondet, %nondet)` lowers
to `XOR(0,0) = 0`, filling the parent struct.member's witness slot with zeros.
Diagnostic: `grep -nE "scf.while.*x !pod" <chip>.ssc.llzk` must be empty for a
cleanly-flattened circuit.

**Sub-component call-count diff.** When Phase 1 fails to hoist a `function.call`
out of a dispatch-countdown `scf.if` with a statically-false predicate
(`arith.subi 0, 1 = MAX_SIZE_T`), the call sits in a dead branch and
`--llzk-to-stablehlo` DCEs the entire `scf.if`. The caller has zero compute
calls for that sub-component family. Compare post-SSC
`grep -cE 'function\.call @<Sub>.*::@compute' <chip>.ssc.llzk` against the
lowered `grep -cE 'func\.call @<Sub>' <chip>.stablehlo.mlir`; an N→0 drop
locates the hoist gap.

**Multi-carry stale operands.** Chips holding multiple input pod-arrays alive
across N outer SSC iterations emit two compute calls per loop iter (one per
`<==` input write). Without a `dynamic_update_slice %iterArg` rewrite between
calls, the second call reads `[0, latest_write]` instead of
`[1st_write, 2nd_write]`. The accumulator is never rewired. Diagnostic:
`grep -nE 'dynamic_update_slice %iterArg_<input-pod-shaped-args>' <chip>.stablehlo.mlir`
must be non-zero for any chip with multi-record input carries.

**Writerless dispatch pod.** When a dispatch pod has no `pod.write %pod[@comp]`
anywhere (circom dropped the inline call for a constant-table sub-component,
e.g. keccak's `RC_0` round-constant generator), `materializeScalarPodCompField`
bails on `writers.empty()` and the struct.readm consumers read from a
`llzk.nondet : !struct<@RC_0>`, producing `dense<0>` in the lowered output.
Diagnostic: `llzk.nondet : !pod.type<[..., @comp: !struct<@<C>>, ...]>` with
zero `pod.write %pod[@comp]` users in post-SSC IR. Fix: synthesize a zero-arg
`function.call @<Sub>::@compute()` when `getNumInputs() == 0`.

**`llzk.nondet` dispatch pod (no `pod.new`).** Post project-llzk/circom PR #390,
a dispatched sub-component pod can arrive as
`llzk.nondet : !pod.type<[@count: index, ...]>` instead of `pod.new`. The
`@count` read yields garbage, the `function.call` never fires, and `@main` fills
with const-0. `materializeScalarPodCompField`'s candidate filter must include
`llzk.nondet` alongside `pod.new`.
