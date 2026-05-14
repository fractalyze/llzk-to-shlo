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
