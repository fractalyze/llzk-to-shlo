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
