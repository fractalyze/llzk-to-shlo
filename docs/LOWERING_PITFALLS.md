# LLZK to StableHLO Lowering Pitfalls

Silent-miscompile traps and diagnostic foot-guns surfaced while debugging real
circuits. Each entry is a forensic note: the trap, how it manifests, how it was
fixed, and how to recognize it again.

## Walker traps

### Phantom rebind via a read-only inner-while capture

**Trap.** `findCapturedArrays` returns array captures unconditionally (needed to
preserve keccak / AES carry semantics). An inner `scf.while` whose body only
READS the captured carrier still gets the carrier appended as an iter-arg by
`promoteArraysToWhileCarry`, and the body yields the block arg directly
(passthrough). The rebind at `processBlockForArrayMutations:491-509` fires
anyway: `init == latest[%cap]` matches, so `latest[%cap]` gets rebound to
`inner.result(captured_slot)`.

This rebind is SEMANTICALLY a no-op (passthrough means `inner.result(i) == init`
at every iter), but the SSA pointer shifts and downstream `array.insert` /
`array.write` chains in sibling `scf.if` branches build on `inner.result(...)`.
The outer `scf.while`'s yield slot then lands on the inner-while's result
instead of the chain tip → that slot becomes a passthrough of `dense<0>` in the
lowered StableHLO.

**Canonical case.** webb `@Poseidon_137_compute` outer slot 5 (`%array_67` /
`@mix` carrier) dead; the inner-while whose body has
`array.extract %array_67[..]` (read only) is the load-bearing rebind source.

**Diagnostic.** Post-walker trace `latest[blockArg]` defOp at the explicit yield
rewrite — `defOp=scf.while` for a block-arg-typed slot is the smoking gun. Quick
repro: extract the lowered function body and grep for
`stablehlo.return.*%iterArg_<N>` at the OUTER while's terminator — a passthrough
where mutations were expected.

**Fix (landed).** Three coupled changes in `LlzkToStablehlo.cpp` preserve
`while_paired_carrier_no_false_collapse`:

1. `processBlockForArrayMutations` mirrors the SSA state of result-bearing
   `array.write` / `array.insert` / `struct.writem` chain links into `latest`.
   Walker 2's view stays at the chain tip when walker 1 already SSA-fied the
   mutation.
1. The same walker propagates a fresh SSA-fy to subsequent same-block uses of
   `arr` via `replaceUsesWithIf`, gated by `isBeforeInBlock(newOp)` to preserve
   dominance. Without this, walker 1's static pinning of N in-branch operands to
   the same pre-chain value made walker 2 catch only the first SSA-fy.
1. `convertWhileBodyArgsToSSA`'s yield rewrite now indexes by slot via
   `body.getArgument(i)`, so chain tips reach the yield even when the
   `scf.while` rebind has shifted the operand off the body arg.

The three "obvious" fixes tried before (yield-by-slot-index alone,
passthrough-skip-guard on the rebind, post-order whileOps) each break
`while_paired_carrier_no_false_collapse` in isolation. They only work together
because (1) keeps walker 2's `latest` consistent with walker-1-built chains so
(3)'s by-slot-index rewrite is byte-equivalent to the existing paired-carrier
behavior.

### Result-bearing `scf.if` with tracked-array slots + `%nondet_*` yields

**Trap.** LLZK's `<--` produces scf.ifs whose array result slots get yielded as
`llzk.nondet` placeholders in both branches; the actual writes happen via inner
whiles inside each branch using the parent's tracked carry as init. After
`applyPartialConversion` the nondet arrays become const-zero tensors; selects
over them pick between two const-zero tensors. `liftScfIfWithArrayWrites`
early-returns on `getNumResults() != 0` and handles void ifs only — result-
bearing ifs go through `extendResultBearingScfIfArrayChain`.

**Canonical case.** Any chip where a `<--` cascade produces result-bearing
scf.ifs whose array-typed result slots are placeholders. Both branches yield
`llzk.nondet`; the real writes happen via inner whiles.

**Diagnostic.** Lowered StableHLO: a `stablehlo.select` between two
`broadcast(constant_0)` tensors at the array-slot result of an scf.if
descendant.

**Fix (landed).** `extendResultBearingScfIfArrayChain` appends NEW tail result
slots typed `!array<x !felt>` (matching tracked-key types) — do NOT rewrite
existing slots (the original `!array<x !pod>` placeholders and tracked
`!array<x !felt>` carriers aren't type-equal pre-conversion). Two invariants:

1. Idempotent across the dual-walker invocations (`convertArrayWritesToSSA` +
   `convertWhileBodyArgsToSSA`) — the second walker must observe the first
   walker's appended slots and not double-add.
1. Reuse path must reference `newIf.getResult(i)`, not `oldIf.getResult(i)` —
   `oldIf` gets erased on append.

### `convertWhileBodyArgsToSSA` absorbs forwarder-vs-nested-result yield discrepancies

**Trap.** `convertWhileBodyArgsToSSA` (`LlzkToStablehlo.cpp:823-855`) walks each
scf.while body, runs `processBlockForArrayMutations` to track per-block latest
SSA carriers (lines 491-509 rebind `latest[blockArg]` to the inner scf.while's
matching result), then rewrites the body's yield operand-by-operand using
`latest`. So an LLZK body yielding `%arg_blockarg` (forwarder shape) and an LLZK
body yielding `%nested_while.result(_)` (nested-result shape) lower to the same
StableHLO.

**Canonical case.** Any LLZK body-yield "fix" upstream of this pass — the fix
may flip the LLZK shape between forwarder and nested-result form while the
lowered StableHLO is byte-identical.

**Diagnostic.** Diff the lowered StableHLO, NOT the simplified LLZK. If the LLZK
changes but the lowered MLIR is identical, the upstream fix didn't earn its
keep.

**Fix surface.** The concrete fix sites for body-yield correctness are
`convertWhileBodyArgsToSSA` and `processBlockForArrayMutations`, not
`expandPodArrayWhile`'s yield rewriter. Avoid LLZK-level reshape fixes upstream
of this pass for yield-correctness bugs.

### `collapseRedundantWhileCarrierPairs` zero-init transitivity

**Trap.** The pass classifies a `stablehlo.while` slot as DEAD when its yield is
a literal pass-through of the body argument and its init traces back through
enclosing-while body-args to a zero-splat constant — then RAUWs the dead result
with a sibling LIVE result of identical type. The DEAD-collapse semantics depend
on the dead slot being "always zero on every iteration", which holds at THIS
while only if no intermediate enclosing while mutates the carrier. A parent
while whose yield differs from its body arg at the same slot index breaks this —
body args on later iterations carry the parent's mutated value, not init.

**Canonical case.** AES `xor_2[i][j][k].b` is zero-initialized at the main while
but actively written across rounds, so any inner-level passthrough isn't truly
"always zero." A naive zero-init trace through the body args would silently
merge `xor_2 .a` and `xor_2 .b`.

**Diagnostic.** Lowered StableHLO body yields the same SSA value at distinct
`.a` / `.b` slots, which surfaces as a witness mismatch at the `.b` field's
slots after the m3 byte-equality gate.

**Fix (landed).** `isZeroSplatTransitively` MUST reject the trace whenever a
visited parent slot is non-passthrough. Every enclosing while in the trace must
itself be a passthrough at the same slot index for the inner RAUW to be sound.

### `collapseRedundantWhileCarrierPairs` LIVE→DEAD body-arg link

**Trap.** The pass partitions zero-init tensor slots into DEAD (passthrough) and
LIVE (computed yield), then RAUWs the dead result to a sibling LIVE result of
identical type. The original safety argument — "both inits are zero-splat
constants, so round 0 starts identical for both slots" — only covers round 0.
Past iteration 0 a LIVE slot diverges from init by definition (yield ≠ body
arg), so post-loop the LIVE result is some computed value, not zero. RAUW'ing a
DEAD reader (which expected the always-zero init) onto that LIVE result silently
corrupts every downstream consumer of the DEAD slot.

**Canonical false-positive case.** Poseidon3's `Poseidon_*::@compute` outer
scf.while threads three iter-args: a counter (LIVE, init=0, yields counter+1
each iter, final = N), an input-copy array, and a capacity-init scalar (DEAD,
yield = body arg, always 0). The counter and capacity are both zero-init
`tensor<!pf>` scalars; without a guard, the pass pairs them and rewrites the
post-loop `func.call @PoseidonEx_*::@compute(%0#1, %0#2)` into `(%0#1, %0#0)`.
The Poseidon3 permutation then runs with capacity = N (= 3 for `Poseidon3`)
instead of 0, miscompiling every consumer of the hash output.

**Diagnostic.** Lowered StableHLO `func.call` whose operand list has a
result-index drop (e.g. `%0#2` → `%0#0`) on a same-typed sibling iter-arg. m3
byte-equality fails on the chip's first wire after the capacity-misrouted call's
output. A focused way to spot it: grep the lowered MLIR for
`call @<sub>_compute(... %X#0 ...)` where the same while is also referenced
later in the same operand position via `%X#K` with K > 0 — a mismatch suggests
the RAUW fired.

**Fix (landed).** Pair-eligibility requires that the LIVE slot's yield
transitively references the DEAD body argument. The walker is a small DAG
traversal from `returnOp.getOperand(live)` through defining-op operands,
checking whether `body.getArgument(dead)` is reachable. Only pairs that satisfy
this carry-same-logical-signal invariant — the original AES
`@xor_3$inputs[@a]/[@b]` shape, where the LIVE XOR consumes the DEAD body arg —
are redirected.

### `processNested` only recurses into `scf.while`, not `scf.if`

**Trap.** `flattenPodArrayWhileCarry(block)` itself is non-recursive
(`for (Operation &op : block)`); region recursion happens via `processNested`,
which dispatches on `scf.while` only. Pod-array-carrying `scf.while`s buried
inside an `scf.if` branch are invisible to the main driver and survive into the
dialect-conversion phase.

**Diagnostic.** Lowered StableHLO has a surviving `!pod.type<...>` iter-arg
shape AND the chip's `scf.if`-branched fixture (typical: branches in `@compute`
that wrap a dispatched while). Confirm with
`grep '!pod.type' bazel-bin/examples/<chip>.stablehlo.mlir` — must be 0.

**Fix (landed).** A post-main-loop straggler pass uses
`module.walk(scf::WhileOp)` to reach the buried whiles. Do NOT add `scf.if`
recursion to `processNested` itself — that path also runs
`eliminatePodDispatch`, whose pod block-arg gating assumes `scf.while`
semantics; nesting an `scf.if` recursion into the same walker double-applies
dispatch elimination on already-flattened blocks.

### `processBlockForArrayMutations` must dispatch on `scf.execute_region`

**Trap.** The walker dispatches on `scf.if` (via
`extendResultBearingScfIfArrayChain`) but historically did NOT dispatch on
`scf.execute_region`. SSC's `materializeStructOfPodsCompField` wraps deep
K-dispatch cascades (e.g. iden3 Poseidon3's 56-deep `MixS_*` cascade) inside
`scf.execute_region -> (!array<K x !pod>)`. The region's existing yield slot is
the pod-typed dispatch array, NOT the felt-typed carrier that the cascade arms
mutate. Without an explicit handler the region is opaque to the walker, every
inner cascade `scf.if` is invisible to `extendResultBearingScfIfArrayChain`, and
every `array.insert %carrier[%cK]` inside is left behind — the carrier chain
silently drops cascade-arm writes and the chip miscompiles.

**Diagnostic.** Lowered MLIR for an iden3 Poseidon3 or similar chip shows a
`<56x4 x !pf>` carrier with only the original 6 cascade writes (not 259) AND
`dynamic_update_slice` count for the carrier far below the cascade depth.

**Fix (landed in PR #117).** Mirror the `extendResultBearingScfIfArrayChain`
shape via `extendExecuteRegionArrayChain` — append NEW tail slots typed
`!array<x !felt>`, never rewrite the pod-typed slot. The pod-typed yield stays
untouched so downstream pod-dispatch elimination still has its expected shape.

### Counter-gated `select(EQ iter, K, passthrough, computed)` may be a faithful lift, not a polarity bug

**Trap.** A lowered StableHLO shape like
`select(broadcast(EQ counter, K), %body_arg_passthrough, %computed_chain)` at an
inner-while slot yield is tempting to read as "inverted polarity in the lift —
fix `extendResultBearingScfIfArrayChain`". In practice the lift's THEN yield
defaults to `parentLatest.lookup(key)` (= the body-arg passthrough) **when the
THEN region didn't write the tracked carrier**, while ELSE yield gets the
cascade tip. The resulting select faithfully encodes the LLZK source structure
"this branch doesn't write this carrier" — flipping it would break LLZK semantic
equivalence.

**Diagnostic recipe (which case is this?).**

1. Identify the LLZK-source role of the slot's carrier. Map outer-while-slot →
   inner-while-arg → uses in body. Find every `array.write` / `array.insert`
   targeting that carrier across BOTH THEN and ELSE.
1. **Case A — lift bug (rare):** both branches write the carrier in LLZK source
   but the lifted select has THEN/ELSE swapped or one side mis-yielded. Suspect
   surface: `extendResultBearingScfIfArrayChain`'s `liveKeys` classification,
   `extendYield`'s lookup direction.
1. **Case B — lift faithful, source asymmetric (common):** only one branch
   writes; the other's passthrough is by design. Suspect surface is NOT the lift
   — look downstream at the consumer that reads the carrier at the missing-write
   iteration.

**Canonical case (Case B).** Webb `@Poseidon_137_compute` inner-while slot 3
yield = `select(EQ a260, 0, %arg288_passthrough, %cascade_tip)` (lowered IR
`webb_poseidon137_main.mlir:50588`). The OUTERMOST `scf.if(a260 == 0)`'s THEN
branch (input-load) doesn't write %arg288 (= "carrier-for=out" Ark output
accumulator); only the ELSE branch's 68-deep `(a260==K)` cascade writes it. Lift
faithfully passthroughs THEN. At outer iter 0, downstream sbox-while reads
%arg288 = passthrough = initial `array.new` (lowers to `dense<0>`), computes
`sbox(0)=0`, and `dynamic_update_slice`s row 0 of the input-bearing ark$inputs
carrier with zeros → input erased.

**Fix surface for Case B.** Not the lift. Options (each non-trivial): (a) inject
the missing-iter write into THEN at the source / SSC level so the lifted select
picks a meaningful then-branch value; (b) gate the downstream consumer OFF at
the missing-iter so it doesn't read; (c) re-route the consumer at the
missing-iter to read a different carrier that DOES have a meaningful value (e.g.
ark$inputs instead of post-Ark accumulator). Compare to a sister LLZK circuit
(e.g. iden3 Poseidon family) to see whether the asymmetric-write source
structure is unique to the failing chip.

**Bug-class cousins.** Same emission shape as AES `aes_256_encrypt`'s
`iterArg_139` counter-gated select chain (see knowledge memos
`llzk-aes-encrypt-counter-gated-select-chain-discovery.md` and AES stage4-10
series). The AES family's residual was NOT closed by a polarity flip — stage10
(2026-05-04) showed gate moves between coincidental values without a clean
polarity-style fix.

## SimplifySubComponents driver-ordering traps

### Struct-of-pods materializer requires three coordination invariants

**Trap.** `materializeStructOfPodsCompField` (`SimplifySubComponents.cpp:2755`)
sits inside the SSC outer fixed-point loop, *before* `eliminatePodDispatch` in
the driver, and runs against Phase 1's (`extractCallsFromScfIf`) incremental
call-hoists. Three coordination invariants make this ordering safe; missing any
one of them silently regresses the structural fix.

**Canonical case.** Webb Poseidon's 68-round Ark cascade
(`webb_poseidon_vanchor_{2_2,16_2,16_8,2_8}`). Each `@idx_K` resolves to a
distinct `!struct<@Ark_K>` — non-uniform-inner shape that
`convertStructOfPodsToArrayOfPods` cannot fold, so the carrier stays alive on
the `scf.while` iter-arg through dispatch elimination. The materializer
allocates a parallel `<N x ...inner>` felt carrier per pub field `@F`, writes
each hoisted call's `struct.readm [@F]` value at slot K after the call, and
rewrites reader-side `struct.readm [@F]` of `llzk.nondet : !struct<@<C>>` to
`array.read|extract %carrier_F[%cK]`.

**Invariant 1 — writer matcher rejects calls whose enclosing block parent is
`scf.if`.** On the first outer fixed-point iter, dispatched calls are still
buried in dispatch-firing scf.if cascade arms (Phase 1 hoists them later in the
same iter). Emitting the materializer's `struct.readm + array.write %carrier[K]`
chain there lands them in branches that Phase 4 makes statically false (the
`arith.subi 0, 1` predicate trap), and `--llzk-to-stablehlo` DCEs the entire
scf.if. Net effect: only the cascade's K=0 arm's carrier writes survive; K!=0
arms silently produce zero output. Wait until Phase 1 hoists (block parent flips
to `scf.while`), then the matcher fires on iter 2+.

**Invariant 2 — carriers tagged with `mSoPCF.carrier-for = "<F>"` and reused
across outer iters.** Phase 1 hoists incrementally (~1 call per outer iter on
the 68-round cascade). Without dedupe, each materializer invocation allocates a
fresh `array.new` for the same `@F` — ~70 zombie carriers accumulate per compute
body. `promoteArraysToWhileCarry` only threads ONE carrier as an iter-arg
through the scf.while, so the rest stay orphaned and downstream DCEs them along
with the writer-side struct.readm chains feeding them. The custom attribute is
correctly stripped during dialect conversion (verified: 0 occurrences in lowered
StableHLO).

**Invariant 3 — carrier dim N from a pre-scan of ALL writer-pattern calls.** N
is `max(K) + 1` over the writer set. Phase 1's incremental hoisting means the
first successful materializer invocation sees only a partial subset of writers —
if that subset's max K is 1 but the cascade's true max is 67, the carrier is
allocated `<2 x ...>` and later inserts at K=67 land out-of-bounds. The pre-scan
walks ALL function.calls matching the writer pattern (including ones still
buried in dispatch scf.if cascade arms) and uses that complete set for the N
computation. The pre-scan still gates on the `array.extract %arr[%cK : index]`
operand pattern with `%cK = arith.constant`, so unrelated function.calls
(top-level `@Poseidon::@compute`, Mix calls with non-constant K) are correctly
excluded.

**Diagnostic.** Post-SSC dump:
`grep -c "array.new.*mSoPCF.carrier-for" <chip>.ssc.llzk` — should be exactly
the count of pub fields reached by the materializer (typically 1 for Poseidon's
`@out`). Counts >1 with the same `"<F>"` value indicate invariant 2 violations.

`grep -nE "function.call.*::@<C>::@compute" <chip>.ssc.llzk` per cascade class
`<C>` — should match `pre-scan-N`'s max K + 1; a smaller count means the
pre-scan path was skipped.

**Fix (landed).** Commit `d2e5d3f`. All three invariants are encoded as guards
in the matcher (Invariant 1: block-parent check at the writer matcher entry;
Invariant 2: `kCarrierAttr = "mSoPCF.carrier-for"` lookup; Invariant 3: two-pass
walk where the first pass builds `preScanClassToK` / `preScanMaxK`).

**Generalization.** Any future SSC pass that operates on Phase-1-hoisted output
and allocates per-(class, field) state must mirror this 3-invariant coordination
— OR be moved to AFTER `eliminatePodDispatch` in the driver (which interacts
with the 4-branch `hasArrayOfPods` / `hasPodBlockArg` guard and is structurally
riskier). Failure mode in either case is silent: the chip builds, the LIT
regression suite passes, and only an end-to-end witness gate catches the
zero-output positions.

**LIT-fixture gotcha.** The SSC block runs only when `hasPod=true`, and
`eliminateInputPods` (pre-step) DCEs any `pod.new` whose users are all
`pod.write` (or empty). A minimal positive test must seed at least one
non-`pod.write` consumer of the dummy pod (e.g. a single `pod.read`) to keep it
alive long enough to gate the materializer. The fixture
`tests/Conversion/LlzkToStablehlo/struct_of_pods_comp_field_materialize.mlir`
includes a `%dummy_read = pod.read %dummy[@d]` for exactly this reason.

### `deriveReaderK` needs the pod-dispatch chain fallback for post-cascade readers

**Trap.** `deriveReaderK` (inside `materializeStructOfPodsCompField`) has to
disambiguate K for multi-K-per-class readers. The original walk-up-`scf.if`
strategy works for readers nested inside cascade-arm `then` branches (predicate
`arith.cmpi eq %expr, %c<K>` keyed on K). But a class can have readers OUTSIDE
the cascade — e.g. iden3 Poseidon3's post-cascade Sigma_F loop sits in a sibling
`scf.while` body that reads the final round's Mix output. Those readers have no
enclosing K-eq predicate; the walk runs to the function root and returns
`nullopt`. Reader stays as `pod.read [@comp] → struct.readm`, Phase 5 nondets
it, downstream lowering reads `dense<0>`, silent miscompile (a Sigma_F call fed
by zero instead of by the cascade output).

**Canonical case.** iden3 Poseidon3 (`PoseidonEx_146::@compute` + 3 siblings in
`iden3_query_sig`). Mix_81 sits at K={0,1,2,4,5,6} (multi-K; Mix_85 at K=3). The
post-cascade Sigma_F loop's K=6 reader fails the scf.if walk — the only
structural cue is the pre-Phase-5 chain
`pod.read %mix_pod[@idx_6] → pod.read [@comp] → struct.readm [@out]`.

**Fix (landed via PR #121, `44a8d90`).** `deriveReaderK` falls back to calling
the existing `extractDispatchK` helper (case (b)/(c) wrapper —
`pod.read [@outer] ← array.read [%cK]` or `← pod.read [@idx_K]`) on the readm's
source. `extractDispatchK` gained a `StringRef outerRecord` parameter so the
same helper serves both the writer side (`"in"`) and the new reader-side
fallback (`"comp"`); K extraction logic lives in ONE place. Result is gated on
`K ∈ classToKs[className]` (same membership check as the scf.if walk uses)
before being returned. Ordering: try scf.if walk first (preserves pre-fix
behavior for cascade-arm readers), fall back to chain extraction on failure.

**Diagnostic.** Post-SSC dump:
`grep -c "llzk.nondet : !struct.type<@<C>>" <chip>.ssc.llzk` — if a multi-K
class `<C>` shows N residual `nondet → struct.readm` readers after the
materializer runs, that's the count of unresolved readers that will lower to
`dense<0>`. (Constraint-call inputs are benign and also show up in this count;
filter by `grep -B0 -A1` against the following `struct.readm` to isolate readm
targets vs `@constrain` inputs.)

**Generalization.** Any new dispatch shape that surfaces a third outer record
name beyond `@in`/`@comp` (e.g. a `@view` cell) gets added via a single new
`extractDispatchK(…, "<new>")` call site, not a third copy of the chain walker.
Resist re-implementing the 3-case chain in line — the helper is already general.

**LIT.**
`tests/Conversion/LlzkToStablehlo/struct_of_pods_comp_field_materialize_post_cascade.mlir`
pins the contract (same class at K={0,1} + readers at function body top level
with no surrounding scf.if). Stripping the fallback makes the CHECK fail on the
`Sub_A_compute` calls being DCE'd.

### `extractCallsFromScfIf` Phase 1 directArgs path is non-idempotent

**Trap.** When the inner call's operands aren't `pod.read` (so `inputPodFields`
is empty and `hasDirectArgs` fires), Phase 1 builds the hoisted call with
operands taken directly from the inner call. If any direct arg is defined INSIDE
the scf.if's regions, the hoisted call sits outside but references inner SSA —
an invalid use that Phase 4 normally cleans up by erasing the scf.if. With the
`isOpAndNestedResultsExternallyUnused` gate (below) blocking the erase, the
scf.if survives — and Phase 1's next outer fixed-point iter sees the same scf.if
\+ inner function.call and re-hoists a duplicate. The process never converges.

**Canonical case.** `maci_*` spin-out chips accumulate ~50+ duplicate hoisted
calls per outer fixed-point pass before the loop trip count gives up.

**Diagnostic.** Wrap the outer fixed-point loop with an iter counter and abort
on iter > 50; log Phase 1's `changed` per iter. If Phase 1 keeps returning
`true` while every other phase has settled, this trap is firing.

**Fix (landed).** Skip the hoist when any directArg's defining op is inside `op`
via `Operation::isAncestor`. The dispatch stays unflattened for this scf.if, but
the pass converges.

### SSC call hoisters need positional dominance

**Trap.** `extractCallsFromScfIf` and `materializeScalarPodCompField` resolve
hoisted-call args by walking the latest `pod.write %carrier[@<field>] = %v` per
tracked field. The "is this arg available at the insertion point?" check has
historically been `!dominanceScope->isAncestor(def)` — true whenever `def` sits
outside the scf.if's region. Necessary, but not sufficient: when the tracker
captures multiple sibling dispatch sites with carrier writes interleaved between
them, an earlier scf.if's hoist can pick up a `%v` from a later write that lives
in the same enclosing block but is defined positionally LATER than the insertion
point. The hoisted call ends up referencing an SSA value the verifier rejects as
out-of-order.

**Canonical case.** iden3 cluster — `iden3_auth*`, `iden3_query_mtp*`,
`iden3_query_sig*`, `iden3_id_ownership_sig*`, `iden3_state_transition*` (13
chips). Sequential `pod.write %carrier[@<field>] = %v` between dispatch sites;
the first scf.if's hoist tries to take a `%v` defined two writes later.

**Diagnostic.** Post-SSC verifier fails with "operand #N is not defined before
this op" / "use of value before its definition" on a `function.call` hoisted out
of an scf.if. The hoisted call sits BEFORE one of its operand's producers in the
same enclosing block.

**Fix (landed in PR #112).** Add
`DominanceInfo::properlyDominates(arg, insertionOp)` as an additional gate
alongside the existing `!isAncestor` check in `extractCallsFromScfIf`. Anchor
the `DominanceInfo` root on the enclosing `function.def`, NOT
`block.getParentOp()` — when SSC recurses into inner `scf.while` or `scf.if`
bodies, `block.getParentOp()` returns the inner region's owner, and values
defined in outer enclosing regions fall outside that root. `DominanceInfo`
queries on those values then over-reject and the guard wrongly rejects every
cross-region candidate. `function.def` is the largest legal SSA scope for a
hoisted call.

**Sibling not yet fixed.** `materializeScalarPodCompField`
(`SimplifySubComponents.cpp:3461-3466`) carries the same
`!dominanceScope->isAncestor(def)`-only check. No chip in the corpus currently
triggers it, but it's the same bug class — extend the guard the same way if a
future chip surfaces it.

### `eraseDeadPodAndCountOps` Phase 4 must walk regions transitively

**Trap.** `isAllResultsUnused(op)` is necessary but NOT sufficient when `op`
carries inner ops whose results are consumed by users OUTSIDE `op`. Phase 1
(`extractCallsFromScfIf`) can hoist a `function.call` BEFORE an enclosing scf.if
using `directArgs` (operands that aren't `pod.read`) — those direct args may be
defined inside the scf.if body (e.g. `array.read %arr[%idx]` against an
outer-scoped array). The hoisted call is then an external user of an
inner-scf.if value. Phase 4 sees the scf.if's own result unused (Phase 2 RAUWd
its pod-typed yield away), region-clears it, and trips
`~Operation: operation destroyed but still has uses` during the inner producer's
destruction.

**Canonical case.** Any chip combining directArgs-hoisting (Phase 1) with a
result-unused scf.if (Phase 2 RAUW). The crash manifests as an assertion at
op-destruction time, not at the structural rewrite.

**Diagnostic.** ASAN / dbg traces with `~Operation` assertion firing on a
producer inside an scf.if region. Run `bazel test -c dbg //tests:lit_tests` to
surface it before opt builds silently UAF.

**Fix (landed).** Gate the Phase 4 erase via
`isOpAndNestedResultsExternallyUnused` — walks each region of `op`, checks every
inner op's result-users via `Operation::isAncestor` to confirm they're all
inside `op` before the erase.

### `isOpAndNestedResultsExternallyUnused` is blind to side effects on outer values

**Trap.** The gate only walks `inner->getResults()` for external users; LLZK's
`array.write` / `array.insert` / `struct.writem` produce no SSA results, so a
side effect on an outer-scoped array (e.g. a per-field iter-arg created by
`flattenPodArrayWhileCarry`) is invisible. Phase 4 then erases the enclosing
void-result region as "dead", silently dropping the writes — and any per-field
iter-arg fed by them stays nondet for every iteration.

**Canonical case.** Webb `@ManyMerkleProof_275 @switcher$inputs` @L iter-0 /
iter-i>0 conditional rewrite synthesizes a void scf.if whose only side effect is
`array.write %perFieldArr[%i] = %src` — without a side-effect carve-out, Phase 4
silently drops every @L write.

**Diagnostic.** Witness mismatch with the per-field iter-arg holding its init
(nondet) value forever; the witness output at @L slots reads as `dense<0>` in
the post-conversion StableHLO. There is no structural symptom in the simplified
LLZK — the writes vanish silently.

**Fix (landed).** Pair every region-bearing synthesizer with a side-effect
carve-out at `eraseDeadPodAndCountOps`. The carve-out at
`eraseDeadPodAndCountOps:633` (`hasNonPodArrayWriteInBody` preserves `scf.for` /
`scf.if` whose body does non-pod array.write / array.insert) is the load-bearing
escape hatch. When adding a new region-bearing synthesizer that writes to outer
values from inside its region, audit that the enclosing op shape (scf.for /
scf.if / future scf.execute_region) is in the carve-out's accepted-name set, AND
that the helper recognizes the mutating op (array.write or array.insert
depending on the field type).

### SSC rewriters rebuilding region-bearing ops need a `rebuiltOps` set

**Trap.** When you rebuild scf.while / scf.if / scf.execute_region by
`builder.create(...)` + `newRegion.takeBody(oldRegion)`, the OLD op goes into
`toErase` but its body (including the yield ops) now lives inside the NEW op. A
subsequent cascade visit that asks `yieldOp->getParentOp()` returns the NEW op,
NOT the OLD op — so guarding "is this op already rebuilt?" with
`toErase.count(parent)` always returns false, and the rewriter re-rebuilds on
every yield touch. Exponential IR growth, then heap-corruption crash during
cleanup.

**Canonical case.** `StructOfPodsRewriter::rebuildIfResultSlot` /
`rebuildExecuteRegion` / `rewriteWhileOperand` — each rebuilds a region-bearing
op and is reached from multiple cascade visits per outer iter.

**Diagnostic.** Memory growth + glibc heap corruption
(`free(): invalid pointer`) during SSC, with the SSC outer fixed-point counter
increasing unboundedly. dbg builds may surface
`~Operation: operation destroyed but still has uses` during cleanup.

**Fix (landed).** Maintain a parallel `DenseSet<Operation *> rebuiltOps`
populated at each rebuild's `builder.create(...)` call site, and probe THAT in
the "already rebuilt?" guard. Each rebuild call inserts into both `toErase`
(old) and `rebuiltOps` (new).

### `processNested`'s `hasArrayOfPods` / `hasPodBlockArg` guard is four-branch

**Trap.** Original code skipped `eliminatePodDispatch` entirely when either flag
fired, on the premise that Phase 5 would clobber pod field discovery. But Phase
1 (`extractCallsFromScfIf`) is benign — it only HOISTS function.calls out of
dispatch-firing scf.ifs, never blanket-nondets pod.reads. Phase 2
(`replacePodReads`) is also benign on array-of-pods carriers (block args already
unpacked) but tears `pod.read %arg[@field]` read-back patterns through
*pod*-typed block args.

**Canonical case.** The fourth case (`hasPodBlockArg=true`) is REQUIRED for
non-uniform-inner struct-of-pods carriers (webb Poseidon's 68-round Ark cascade
where each `@idx_K` resolves to a distinct `@Ark_K` struct class —
`convertStructOfPodsToArrayOfPods` cannot rewrite this shape, so the carrier
stays alive on the iter-arg and Phase 1 is the only way to hoist the dispatched
calls out of the statically-false dispatch scf.ifs).

**Diagnostic.** The Phase 2 read-back regression is pinned by
`unpack_pod_while_carry_block_arg_*.mlir`. The Phase 1 hoist gap surfaces in the
call-count diff (post-SSC vs. lowered) — see "Sub-component call-count diff"
section below.

**Fix (landed).** Split the guard four ways:

- `!hasArrayOfPods && !hasPodBlockArg` → full `eliminatePodDispatch`.
- `hasArrayOfPods && !hasPodBlockArg` → Phase 1 + Phase 2 with a local
  `trackedPodValues` map (post-Option-B carriers).
- `hasPodBlockArg` (with or without `hasArrayOfPods`) → Phase 1 ONLY with a
  local tracker; Phase 2 skipped to preserve the pod-block-arg read-back
  regression contract.

When adding a new guard-gated dispatch elimination phase, mirror the four-branch
structure; collapsing branches either misses hoists post-Option-B or trips the
Phase 2 read-back regression.

### Struct-of-pods carrier survives flattening when inner type is non-uniform

**Trap.** Circom emits two parallel carrier shapes for some chips: array-of-pods
`!array<K x !pod<...>>` (runtime-indexed iter-arg access) and struct-of-pods
`!pod<[@idx_0..@idx_K-1: T]>` (compile-time-indexed dispatch reads).
`flattenPodArrayWhileCarry` only handles the former; the latter survives
unflattened. The original LLZK body reads from the struct-of-pods form at the
dispatch firing site, so orphaning it drops the data-flow into the buried
`function.call`, leaving the call with a fresh `llzk.nondet` operand.
`unpackPodWhileCarry` doesn't fire either because the scf.while typically
carries multiple pod iter-args (dispatch-pod-of-pods + input-pod-of-pods pair),
tripping its `podCarryIndices.size() != 1` gate.

**Canonical case.** Webb Poseidon: `%arg2: !pod<[@idx_0..@idx_67: T]>` for
compile-time dispatch reads AND `%arg5: !array<68 x T>` for runtime iter-arg
access. Flatten handles `%arg5`; `%arg2` survives.

**Diagnostic.** `grep -nE "scf\.while.*!pod\.type<\[@idx_0:" <chip>.ssc.llzk` —
any match is a surviving struct-of-pods carrier.

**Fix surface (multi-stage).** Convert when possible, materialize when not.

1. `convertStructOfPodsToArrayOfPods` rewrites `!pod<[@idx_N..]>` →
   `!array<K x T>` with synthesized `arith.constant N : index`, placed before
   `materializePodArrayInputPodField`, so existing flatten handles the rest.
   **Necessary but not sufficient** — after the rewrite, the dispatched
   `function.call`'s operand is defined inside the dispatch-firing `scf.if`, and
   Phase 1's `dominatesScfIf` guard bails on the hoist. Extend Phase 1 to
   clone-hoist pure direct-arg defining ops (array.extract, arith.constant,
   cast.toindex) recursively before the scf.if.
1. **Not always possible** — non-uniform-inner carriers (each `@idx_K` resolves
   to a distinct `@comp: !struct<@Sub_K>`, e.g. webb Poseidon's 68 distinct
   `@Ark_K` round-constant classes) fail `matchStructOfPodsShape`'s
   uniform-inner check. For these, `convertStructOfPodsToArrayOfPods` no-ops and
   the carrier stays on the iter-arg. `processNested`'s `hasPodBlockArg=true`
   branch must run Phase 1 with a local tracker.
1. **Phase 1 hoist alone is also not sufficient** — on a `hasPodBlockArg=true`
   block, the writer-side carrier produces 70 hoisted calls at the inner-while
   body, but the reader-side cascade in a SIBLING inner-while body reads
   `struct.readm [@out]` from a pre-existing
   `llzk.nondet : !struct.type<@Ark_K>` (NOT the hoisted call results). The
   writer↔reader dispatch link was already severed by an earlier phase.
   Diagnostic: `grep -c "pod\.read.*\[@comp\]" <chip>.ssc.llzk` ⇒ 0 means
   consumers are cleared;
   `grep -c "nondet.*!struct\.type<@Sub_" <chip>.ssc.llzk` shows how many were
   nondetted in place.
1. **Completion fix** — `materializeStructOfPodsCompField`: allocate a parallel
   per-@F felt array, fill it from the hoisted call results (sized to K, indexed
   by the cascade key), thread through writer↔reader scf.while iter-args, RAUW
   reader-side struct.readm chains. The K-distinct-struct-class shape is
   preserved (each writer has its own call slot); only the @F-typed `!felt`
   outputs are unified. See the "Struct-of-pods materializer requires three
   coordination invariants" section above for the materializer's own invariants.

### `unpackPodWhileCarry` gates on a fixed set of handleable pod-value use shapes

**Trap.** The expansion rewires pod.read / pod.write uses + the body terminator
at the expand slot + post-while `pod.read` / `pod.write` / `struct.writem` users
\+ immediate chained `scf.while` users. Any other use (nested `scf.yield`
carrying the pod up through enclosing whiles, function.call operand, scf.if
branch yield) survives the rewires and trips `use_empty()` at the subsequent
`eraseArgument` / `op.erase()`.

**Canonical case.** Any chained-of-chained scf.whiles, or pod-typed operands to
function.calls inside the carry body.

**Diagnostic.** Crash in `eraseArgument` / `op.erase()` with `use_empty()`
assert during `unpackPodWhileCarry` cleanup; the surviving use is the unhandled
shape.

**Fix (landed).** Gate via a use-shape check before any IR mutation; if any use
is outside the handleable set, `continue` and let the outer fixed point retry.
Apply the same gate to chained whiles with `allowChained=false` —
chained-of-chained scf.whiles aren't handled by the branch.

### `inlineInputPodCarries` must dedupe `toErase`

**Trap.** A single `pod.write %pod = %value` is a user of BOTH `%pod` (operand
0\) and `%value` (operand 1) when both are pod-typed and traced into `podValues`
via scf.while carry walks. Naive `toErase.push_back(user)` accumulation pushes
the same pod.write twice; the trailing `op->erase()` loop double-frees on the
second visit.

**Canonical case.** Any chip with pod-typed inputs threaded through scf.while as
both `%pod` and `%value`-side operands of the same write (common when an input
pod is also a carry).

**Diagnostic.** Heap-use-after-free at `Operation::getBlock()` during
`inlineInputPodCarries`' final erase loop. Reliably caught in ASAN / dbg-build
LIT runs.

**Fix (landed).** Maintain a parallel `DenseSet<Operation *>` and add via a
helper that no-ops on duplicates. Same bug surface for `pod.read` users in the
less common shape where a `pod.read` appears in two podValues' use lists.

### Don't reshape the `<--` cascade from SSC

**Trap.** Tempting fix for orphan `pod.new : <[]>` survival: walk `scf.if` /
`scf.execute_region` ops post-Phase-5 nondet, drop pod-typed result slots, and
replace dropped uses with `llzk.nondet`. This converges cleanly inside SSC and
yields well-formed IR, but trips `--llzk-to-stablehlo`'s scf.if rewriters at
unrelated locations (`empty block: expect at least a terminator` on adjacent
non-pod scf.execute_region); fully erasing all-pod scf.ifs hangs the conversion.
The cascade carries values that `extendResultBearingScfIfArrayChain` /
`convertArrayWritesToSSA` consume during their tracked-array shape match — any
upstream reshape breaks the type-equality invariants they rely on.

**Canonical case.** Orphan empty-template pod.new survival on a chip with
adjacent non-pod scf.execute_region scaffolding.

**Diagnostic.** `--llzk-to-stablehlo` fails with "empty block: expect at least a
terminator" at a location unrelated to the SSC reshape site, or hangs on a
fully-erased all-pod scf.if.

**Fix (landed).** If a pod dispatch bundle is structurally dead, recognize it
via use-trace at the scf.while-carrier drop decision instead — don't restructure
the scf.if / scf.execute_region scaffolding.

### `materializePodArrayCompField`'s K-pub-felt drain treats K as an outer dim

**Trap.** When a dispatched sub-component struct exposes K>1 `{llzk.pub}` felt
members (Switcher's `@outL/@outR`, BitElementMulAny's `@dblOut/@addOut`), the
parent's `struct.member @F' : <D x !struct>` flips to `<D, K x ...inner>` (one
extra K dim prepended in declaration order — same shape circom's `.wtns` emits
per chip iteration, contiguous `[f0_i, f1_i, …]` per cell).

**Canonical case.** K=1 path stays byte-identical to the original single-pub
layout so AES sister chips never observe the change. K>1 path is exercised by
Switcher and BitElementMulAny in webb's Transaction chain. K=0 zero-pub-felt
inner structs with at least one writem-targeted non-pod member hit the recursive
flatten path (MMP_275 `<30 x !felt>` @hasher + `<30, 2 x !felt>` @switcher).

**Diagnostic.** Post-lowering `@constrain` repair failure: partial conversion
trips on a struct-typed operand into an erased `Sub::@constrain` callee. The
mismatch is at the consumer-side `struct.readm @<f_j>` rewrite, not at the
struct.member shape itself.

**Fix (landed).** Two invariants for K>1, plus a recursive K=0 flatten path:

1. K>1 requires uniform inner type across all pub fields. Mixed (e.g. one scalar
   \+ one `<M x !felt>`) needs a flat-felt concat path that no bucket-1 chip
   exhibits today.
1. `@constrain` must be repaired in lockstep — `array.read %parent[%i]` becomes
   `array.extract` of the `<K x ...>` slice, and each per-pub
   `struct.readm @<f_j>` consumer gets rewritten to
   `array.read|extract %slice[%c_j]` using the field's declaration-order index.
1. K=0 fall-through: `findRecursiveWritemMembers` walks writem-targeted non-pod
   members in declaration order, promotes each to `{llzk.pub}` on the inner
   struct.def (load-bearing — LLZK's `MemberReadOp::verifySymbolUses` rejects
   external reads of private members), allocates destFelt at
   `<D × totalFlat × !felt>`, and unrolls each member's natural dim shape
   row-major into `arith.constant` + `array.read` + `array.write` triplets per
   writer site. Mixed-shape members are handled by this path; the K>1
   uniform-shape constraint is preserved unchanged for sister chips.
   `@constrain` repair for K=0 replaces inner `struct.readm @<member>` with a
   typed `llzk.nondet` placeholder — heterogeneous member shapes can't be
   re-projected from a flat felt slice, and the placeholder's only downstream
   consumer is the now-erased sibling `Sub::@constrain` call.

### A new SSC transform must converge with `eliminatePodDispatch`

**Trap.** The outer `while (changed)` re-runs all phases plus `processNested`
until no pass returns `true`. If your transform produces IR that
`eliminatePodDispatch` keeps re-modifying every iteration, the loop never
settles.

**Canonical case.** Unit tests pass on toy IR but CI hangs for tens of minutes
on real circuits. The new pass and `eliminatePodDispatch` ping-pong over the
same IR shape.

**Diagnostic.** Wrap the outer loop with an iter counter + abort > 50; log each
phase's `changed` per iter. The phase still returning `1` after the others
settle is the broken one.

**Fix (landed).** Emit IR that's idempotent under all five
`eliminatePodDispatch` phases (extract calls / replace reads / erase writem /
erase dead pods / replace remaining), or rely on the existing outer fixed-point
\+ `processNested` to revisit nested constructs across iterations rather than
recursing inside your own helper.

### `replaceRemainingPodOps` (Phase 5) clobbers `unpackPodWhileCarry`'s field discovery

**Trap.** Phase 5 nondets every `pod.read` in a block — including reads of
pod-typed `scf.while` block args. Those reads are the field-discovery input for
`unpackPodWhileCarry` on the NEXT outer iteration, and once they become
`llzk.nondet` the unpacker can't recover the field layout. Symptom: multi-record
input pods (e.g. keccak's `<[@a: array, @b: array]>` carries) survive into
dialect conversion.

**Diagnostic.** Lowered MLIR retains `!pod.type<[@a:..., @b:...]>` iter-args on
`scf.while` for a chip that should have been fully unpacked, plus a
blanket-nondet `func.call(%cst...)` grep on the lowered call sites.

**Fix (landed).** Gate `eliminatePodDispatch` Phase 5 on the block having no
pod-typed block args. Skip the phase for that outer iter; the unpacker gets a
clean run on the next iter and dispatch elimination resumes once the carries are
felt-typed.

### Writerless `llzk.nondet` dispatch pod ⇒ synthesize zero-arg substruct call

**Trap.** Subset of the `llzk.nondet` dispatch-pod case (PR #390) where there's
no `pod.write %pod[@comp] = ...` anywhere — circom dropped the inline call for
constant-table sub-components (e.g. keccak's `RC_0` round-constant generator).
`materializeScalarPodCompField` bails on `writers.empty()`, so the post-loop
@comp readback consumer sees a `llzk.nondet : !struct<@RC_0>` and the
struct.readm reads garbage.

**Diagnostic.** The chip's lowered MLIR has a `dense<0>` constant feeding a
downstream slot that should be a fixed table value. m3 byte-equality fails on
every cell of that table's range. Look for an
`llzk.nondet : !pod.type<[@count, @comp: !struct<@<C>>, ...]>` with zero
`pod.write %pod[@comp] = ...` users in the post-SSC IR.

**Fix (landed).** Walk the `@comp` struct ref + `@compute`, look up the
`function.def` via `SymbolTable::lookupSymbolIn(module, callee)`, and synthesize
a function-scope `function.call @<Sub>::@compute()` only when
`getNumInputs() == 0`. Use `getTopLevelModule` to walk past LLZK v2's
per-component `builtin.module` wrappers (otherwise the lookup hits the outer
wrapper module rather than the chip module).

## Pod-dispatch silent miscompile signals

These miscompile silently: the chip builds, the LIT regression suite passes, and
only the m3 byte-equality gate against circom's `.wtns` catches the wrong-output
positions. Each signal points to a different gap in dispatch-pod elimination.

### Pod-array iter-arg survival post-simplify

**Trap.** Phase 5 (`replaceRemainingPodOps`) nondets every `pod.read` in a
block, including reads of pod-typed `scf.while` block args. When
`flattenPodArrayWhileCarry` skips a loop (e.g. the use-shape gate trips), the
cross-iteration `pod.read [@a]` is nondetted, the resulting
`function.call @<Sub>::@compute(%nondet, %nondet)` lowers to `XOR(0,0) = 0`, and
the parent struct.member's witness slot fills with zeros.

**Canonical case.** Any chip whose `flattenPodArrayWhileCarry` precondition
trips — e.g. multi-record input pods like keccak `<[@a: array, @b: array]>`
carries.

**Diagnostic.** Pair of greps that must BOTH be empty for a cleanly-flattened
circuit:

```bash
# Post-simplify, surviving pod-typed scf.while carry
bazel run //tools:llzk-to-shlo-opt -- --simplify-sub-components <input> \
  | grep -nE "scf.while.*x !pod"

# Lowered StableHLO, blanket-nondet of load-bearing pod.read
grep -cE 'func.call @<Sub>_<Sub>_compute\(%cst' <chip>.stablehlo.mlir
```

**Fix surface.** Either close the `flattenPodArrayWhileCarry` gate (the
preferred path — keep the carry threading) or gate Phase 5 on the block having
no pod-typed block args. The latter is structurally riskier because it leaves
more pod traffic alive into dialect conversion.

### Multi-carry chips need rewrite-back between consecutive same-instance calls

**Trap.** Chips holding multiple input pod-arrays alive across N+ outer iters of
SSC emit 2 compute calls per loop iter (one per `<==` input write). Without a
`dynamic_update_slice %iterArg` between calls, the 2nd call reads
`[0, latest_write]` instead of `[1st_write, 2nd_write]`. The accumulator is
never rewired.

**Canonical case.** `maci_splicer` with 4 input pod-arrays. Sister single-carry
chips (`maci_quin_selector`) emit the rewrite-back natively.

**Diagnostic.** BOTH metrics must be 0 (the first is the `%cst`-operand signal —
independent of the rewrite-back gap, but co-emitting):

```bash
grep -cE 'func.call @<Sub>_.*compute\(%cst' <chip>.stablehlo.mlir
grep -nE 'dynamic_update_slice %iterArg_<input-pod-shaped-args>' <chip>.stablehlo.mlir
```

The second metric must be NON-zero (one DUS per same-instance call boundary) for
a chip with multi-record input carries; a zero count is the runtime bug even
when the first metric is clean.

**Fix surface.** Three coupled callsites: `eraseDeadPodAndCountOps` guards the
rewrite-back chain across phases, `flattenPodArrayWhileCarry` resorts
`fieldOrder` to record-declaration order, and the post-`runOnOperation` rewire
wraps a contiguous-mixed-type pre-pass before the homogeneous-sub-run match.

### Sub-component call-count diff (SSC vs lowered StableHLO)

**Trap.** When `eliminatePodDispatch` Phase 1 (`extractCallsFromScfIf`) fails to
hoist a `function.call @Sub::@compute(%collected)` out of a dispatch-countdown
`scf.if` whose predicate is `arith.cmpi eq (arith.subi 0, 1) 0` (statically
FALSE because `@count` was replaced with `arith.constant 0 : index` and unsigned
`subi 0, 1 = MAX_SIZE_T`), the call sits in a dead branch and
`--llzk-to-stablehlo` DCEs the entire scf.if. The CALLER then has zero compute
calls for that sub-component family.

This signal is distinct from the `%cst`-operand diagnostics above (which still
have calls, just with nondet operands) and distinct from the `@main`-is-all-zero
pattern (which fires for missing top-level calls, not buried sub-calls).

**Canonical case.** Webb `webb_poseidon_vanchor_2_2`'s `@Poseidon_137_compute`
lowered body has 0 `func.call @Ark` vs 70 in SSC.

**Diagnostic.** For each `@Sub`, compare:

```bash
grep -cE 'function\.call @<Sub>.*::@compute' <chip>.ssc.llzk
grep -cE 'func\.call @<Sub>' <chip>.stablehlo.mlir   # scope to enclosing callee body
```

An N→0 (or N→partial) drop locates the hoist gap. Also confirm `%arg0` refs in
the callee's lowered body — a function declaring `%arg0: tensor<K x …>` with
**0** body references means the data-bearing chain (`array.read %arg0[i]` →
`array.write %nondet[i]` → `@Sub::@compute(%nondet)`) lost its terminal consumer
to the hoist gap.

**Fix surface.** The hoist gap is upstream of dispatch elimination — usually a
guarded skip in `processNested`'s four-branch guard (see SSC driver-ordering
traps above) or a missing carrier rewrite (see "Struct-of-pods carrier survives
flattening" above).

### `llzk.nondet` dispatch pod (no `pod.new`)

**Trap.** Earlier circom-llzk emits `pod.new {@count = const_N}`; post
project-llzk/circom PR #390 the same pod can come in as
`llzk.nondet : !pod.type<[@count: index, ...]>`. Inside the input-collection
scf.while, `pod.read [@count]` yields garbage, the buried
`function.call @<Sub>::@compute(...)` never fires, and `@main` fills with
const-0. `materializeScalarPodCompField`'s candidate filter must include
`llzk.nondet` alongside `pod.new`.

**Canonical case.** Any chip built against post-PR-#390 circom-llzk where the
front-end has dropped the explicit `pod.new` constructor for a dispatched
sub-component.

**Diagnostic.** Post-`--simplify-sub-components` lowered StableHLO `@main` body
of only `stablehlo.constant ... 0` + reshapes + `dynamic_update_slice` with no
`func.call` ⇒ dispatch elimination silently dropped the call.

**Fix (landed).** Extend `materializeScalarPodCompField`'s candidate filter to
include `llzk.nondet`. The writerless subset (no `pod.write %pod[@comp]`
anywhere) still goes through the inline-handled
`Writerless llzk.nondet dispatch pod ⇒ synthesize zero-arg substruct call` rule
retained in CLAUDE.md.

## Upstream-LLZK contract drift

### Test fixtures are consumer-owned IR

**Trap.** Hand-written `.mlir` test files (everything under `tests/`) are parsed
directly by `llzk-to-shlo-opt`. The upstream IR migrator does NOT touch parser
input — it rewrites in-tree LLZK IR via the dialect's loader/printer, not
arbitrary `.mlir` fixtures.

**Fix.** When bumping LLZK upstream, hand-migrate every consumer fixture in the
same change. Skipping this lands a green build whose fixtures parse on the old
dialect text shape but break post-merge for anyone trying to add a new test.

### BUILD glue must move with upstream

**Trap.** New `.td` files (new dialects / interfaces) added upstream don't
appear in `third_party/llzk/llzk.BUILD` automatically. The Bazel build skips the
inc-gen rules for missing `.td` paths silently.

**Fix.** When bumping, diff `include/llzk/Dialect/**/*.td` against
`gentbl_cc_library` targets in `third_party/llzk/llzk.BUILD` and add inc-gen
rules for every missing TableGen file. The diff is small (typically 1-2 files
per bump) but easy to miss.

### `createEmptyTemplateRemoval` uses `applyFullConversion` over a narrow op list

**Trap.** Its conversion target only handles ops in `OpClassesWithStructTypes`
(`struct`/`array`/`function`/`global`/`constrain`/`polymorphic`). Anything else
(`pod.*`, `llzk.nondet` results with struct types, our synthesized ops) MUST be
gone or already in stripped form before this pass runs — `applyFullConversion`
rejects unhandled ops with a `failed to legalize` error.

**Fix.** Order: clean residual pod traffic first, pre-strip `<[]>` only on ops
*outside* that tuple, then run template removal. The pre-strip site is in SSC's
epilogue.

### `<[]>` (empty params) vs no params on `!struct.type`

**Trap.** Template removal rewrites `<[]>` to no-params on covered ops but
leaves SSA values on uncovered ops alone. Mixing the two forms across a use-def
edge produces an unresolved `builtin.unrealized_conversion_cast` that
`applyPartialConversion` won't legalize.

**Fix.** Strip `<[]>` on uncovered ops (any op not in
`OpClassesWithStructTypes`); don't strip on covered ones — template removal will
do that itself.

### `llzk.nondet : index` is dialect-conversion-illegal

**Trap.** The conversion target legalizes nondet for `felt`/`array`/`struct`
only. Residual `pod.read` ops with `index` result type (dispatch-pod `@count`
countdown after Phase 5) survive as `llzk.nondet : index`, which
`applyPartialConversion` then fails to legalize.

**Fix.** Substitute `arith.constant 0 : index` for these residuals — the
surrounding `cmpi`/`scf.if` scaffold is structurally dead once the call is
hoisted; `0` keeps `cmpi` false and DCE collapses the dead branch.

### project-llzk/circom PR #378's same-named `poly.template` wrap

**Trap.** Circom v2 (post-#378) emits every `function.def` / `struct.def` inside
a same-named `poly.template @X` to track polymorphic typing.
`EmptyTemplateRemoval` rewrites that to `builtin.module @X { function.def @X }`;
the inner symbol now shadows the wrapping module's symbol in the parent's
`SymbolTable`, and the next pass that walks it (LlzkToStablehlo conversion in
particular) trips with `redefinition of symbol named '<X>'`.

**Canonical case.** Any chip whose LLZK was produced by post-#378 circom.
Pre-#378 circom doesn't emit the same-named wrap, so older bench artifacts
escape.

**Diagnostic.** Lowering aborts with `redefinition of symbol named '<X>'` during
`applyPartialConversion` in `--llzk-to-stablehlo`, where `<X>` is the inner
same-named struct or function. The error site is downstream of the actual wrap;
the source is the `EmptyTemplateRemoval`-produced module shell.

**Fix (landed).** `flattenSingleEntityWrapperModules` in
`SimplifySubComponents.cpp`: hoist the same-named single child to module level,
erase the empty wrapper, then use `AttrTypeReplacer` to rewrite
`@X::@X[::@method]` → `@X[::@method]` so refs nested in types (e.g.
`!struct.type<@X::@X>`) are also caught — a plain attribute walk misses those.

## MLIR C++ API gotchas

### `arith.cmpi` / `arith.cmpf` predicates live in op `properties`, NOT discardable attrs

**Trap.** `def->getAttrOfType<IntegerAttr>("predicate")` returns null on
post-Properties-migration MLIR ops. The predicate is stored in MLIR's per-op
`properties` storage and is invisible to the discardable-attribute dict API. A
pass that reads the predicate via `getAttr("predicate")` silently treats every
`cmpi`/`cmpf` as "predicate unknown" and falls through to a default branch.

**Canonical case.** `materializeStructOfPodsCompField`'s `deriveReaderK` walks
`arith.cmpi eq, %expr, %cK` predicates to disambiguate cascade arms. Reading the
predicate via `getAttr` silently always missed → every reader fell through to
the std::nullopt branch.

**Diagnostic.** A pass that should match `arith.cmpi eq` predicates behaves as
if no predicate matches, even on trivial fixtures. The fallback path activates
on every call site.

**Fix.** Use the typed accessor:

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
if (auto cmp = dyn_cast<arith::CmpIOp>(def))
  if (cmp.getPredicate() == arith::CmpIPredicate::eq) { ... }
```

The print-form fallback (`Operation::print` into a string + `s.find(...)`) works
but is O(op_size) per check — measured at **>60× slowdown** (13s → 14+ min) on
`iden3_query_sig`'s SSC pass alone. For any modern-MLIR op declaring inherent
attrs via TableGen `Properties`, prefer the generated typed accessor over
`getAttr(name)`.

### `gentbl_cc_library` outputs need a wide MLIR-header set

**Trap.** Generated `<Dialect>Ops.{h,cpp}.inc` and `<Dialect>Attrs.{h,cpp}.inc`
transitively reference `DialectBytecodeWriter`
(`mlir/Bytecode/BytecodeOpInterface.h`), `getProperties()`
(`mlir/IR/OpDefinition.h` + `mlir/IR/Builders.h`), `OpAsmPrinter`
(`mlir/IR/OpImplementation.h`), `Diagnostics.h`, `SideEffectInterfaces.h` (when
ops use `Pure` etc.), and `DialectImplementation.h`. The manual `.cpp`/`.h` code
rarely uses these symbols directly, so cpplint flags them as "unused" — but
removing them breaks compilation of the generated `.inc` body.

**Fix.** Mirror the include set in `llzk_to_shlo/Dialect/WLA/WLA.{h,cpp}` when
adding a new dialect, and suppress cpplint with
`// NOLINT(build/include_what_you_use)` where needed.

### Consolidate `gentbl_cc_library` rules into one per-dialect TableGen file

Mirror `llzk_to_shlo/Dialect/WLA/BUILD.bazel`'s `wla_inc_gen` rule — it produces
8 outputs (dialect/enum/attr/op decls + defs) from a single `mlir-tblgen`
invocation. The naive 4-rule split re-invokes `mlir-tblgen` 4× per clean build
for no benefit, and per-rule outputs must declare the same `td_includes`
everywhere.

## APInt arithmetic traps

### `APInt::getSExtValue()` on a felt constant is a silent miscompile

**Trap.** UB at `getBitWidth() > 64` — the call returns the low 64 bits.
`1 << 252` (LessThan offset on bn128) truncates to `0`.

**Canonical case.** bn128 felt circuits emitting `LessThan` comparators with the
`1 << 252` offset constant. The truncated `0` falls through the comparator's
sign-bit check and the witness output for that LessThan slot is wrong.

**Diagnostic.**
`grep "value = dense<" <chip>.stablehlo.mlir | grep -v "dense<[0-9]>"` — bn128
circuits with comparators MUST emit at least one
`dense<7237005577332262213973186563042994240829374041602535252466099000494570602496>`
(= 2^252) per `LessThan`. Zero hits ⇒ a `getSExtValue()` call truncated it
upstream.

**Fix (landed).** Use `APInt::zextOrTrunc(storageWidth)` — zero-extension is
correct because field elements are unsigned, ranged `[0, p)` with `p < 2^254`.
Sister site to audit: `convertToIndexTensor` in `TypeConversion.cpp`.

### `APInt::operator==` / `!=` asserts on bit-width mismatch

**Trap.** `felt.const`'s `FeltConstAttr` uses `APIntParameter`'s minimum-
bits-needed sizing (e.g. `felt.const 1` → 4-bit APInt); `arith.constant : index`
is always 64-bit. Comparing them directly asserts in dbg and is UB in opt
(slow-case `EqualSlowCase` reads past one side's word boundary, often producing
a corrupt pointer that segfaults much later).

**Canonical case.** The original bucket-2 S1-cluster `Type::getContext` UAF
inside `EmptyTemplateRemoval` was a downstream symptom of an `APInt` `slow-case`
word-boundary read — the corrupt pointer surfaced many ops later.

**Diagnostic.** ASAN-instrumented dbg trips at `APInt::EqualSlowCase` with
mismatched `getBitWidth()`. opt builds bury the symptom in a delayed segfault.

**Fix (landed).** Normalize at construction when collecting APInts from
heterogeneous sources. Apply `zextOrTrunc(kCommonWidth)` at the push_back sites
(typically 64 for index-typed values). Reference site:
`SimplifySubComponents.cpp::outerIndexConstValues`.

## Diagnostic foot-guns

### `awk`-slicing a `stablehlo.while` body for a writeback grep

Naive `awk '/stablehlo.while.*iterArg/{f=1} /stablehlo.return/&&f{f=0} f'` stops
at the FIRST inner `stablehlo.return` and misses everything past it — which is
most of any real circuit, since Poseidon-class bodies nest 2-3 levels deep.
False-`0` from this pattern has wasted multiple sessions chasing the wrong
"missing writeback" target.

Slice by `func.func` boundary or use a brace-balanced extractor instead.

Sister gotcha when grepping for a `dynamic_update_slice %iterArg_<N>` writeback:
the cascade form `%47 → %67 → %87 → %107` is structurally one rewrite-back chain
on the same position; only the LAST write at that iter index survives (rest are
overwritten), so seeing N cascaded updates is not a bug — verify by counting
distinct base operands, not total `dynamic_update_slice` instances.

### Reproducing m3 gate failures without `BUILD.bazel` edits

Call `bazel-bin/bench/m3/m3_runner` directly with the chip's fixture quadruple —
no CHIPS / `BUILD.bazel` edits needed. Once `//examples:<chip>` and
`//bench/m3:m3_runner` are built, iterate on the lowered IR + fixtures in ~2 s
without rebuilding the gate test target.

```bash
chip=<chip>
mlir=bazel-bin/examples/${chip}.stablehlo.mlir
json=bench/m3/inputs/${chip}.json
wtns=bench/m3/inputs/${chip}.wtns
gate=bench/m3/inputs/${chip}.json.gate
indices=$(tr -s '[:space:]' ' ' < "$gate" | sed 's/^ *//;s/ *$//')
bazel-bin/bench/m3/m3_runner "$mlir" --circuit="$chip" --N=1 \
  --iterations=1 --warmups=0 --input_json="$json" \
  --correctness_gate=true --gate_wtns_path="$wtns" \
  --gate_wtns_indices="$indices"
```

Add `--zkx_dump_to=<dir> --zkx_dump_hlo_as_text=true` to capture the optimized
HLO (`module_0001.main.sm_12.0_gpu_after_optimizations.txt`) — critical for
diagnosing dead-carry iter-arg bugs because the optimizer folds those carriers
to `broadcast(constant_0)` and that's visible in the post-opt fused_computation
bodies.

### Repeating address frame in opt-binary stack ≠ stack overflow

When `llzk-to-shlo-opt` (release build) segfaults and the dumped backtrace shows
the same `0x...XXXXX` address repeating 3-5+ times, the obvious read is "deep
recursion / stack overflow". It almost never is. The pre-stripped opt build
inlines and hash-cons-merges template instantiations heavily, and
`mlir::detail::walk` — instantiated separately for every distinct walk-callback
type — collapses to a tight cluster of identical-looking PC values when walking
nested IR regions (functions inside struct.def inside module, scf.if inside
scf.while inside body, …). What looks like 4× recursion of "the same function"
is actually 4 different `walk<T>` template instantiations called legitimately
during a single forward pass.

The fix is to rebuild `-c dbg` and re-run before drawing any structural
conclusion from a repeating address:

```bash
bazel build --config=cuda_clang_official -c dbg //tools:llzk-to-shlo-opt
ulimit -s unlimited
bazel-out/k8-dbg/bin/tools/llzk-to-shlo-opt <flags> <input> 2>&1 | head -30
```

A pre-existing dbg binary at `bazel-out/k8-dbg/bin/...` from an older HEAD may
be unusable because the project occasionally ships ASan-instrumented dbg
configs, which can trip a Linux 4.12+ kernel/ASLR conflict at startup
(`AddressSanitizer: Shadow memory range interleaves with an existing memory mapping`).
When that happens, force a fresh build rather than chasing `setarch -R`
workarounds — those mask separate bugs in the older code.

Historical example: `webb_batch_merkle_4`'s opt-binary stack showed
`0x...30034a` repeated 4× and was originally diagnosed as deep recursion / stack
overflow in `SimplifySubComponents`. The dbg-build retry symbolized it as a
single-frame use-after-free at `SimplifySubComponents.cpp:2142`'s
`for (OpOperand &use : ...getUses())` outer loop — entirely unrelated to
recursion depth.
