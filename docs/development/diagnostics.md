# Diagnostic foot-guns

## `awk`-slicing a `stablehlo.while` body for a writeback grep

Naive `awk '/stablehlo.while.*iterArg/{f=1} /stablehlo.return/&&f{f=0} f'` stops
at the FIRST inner `stablehlo.return` and misses everything past it — which is
most of any real circuit, since Poseidon-class bodies nest 2-3 levels deep. A
false-zero from this pattern has sent multiple sessions chasing the wrong
"missing writeback" target.

Slice by `func.func` boundary or use a brace-balanced extractor instead.

Sister gotcha when grepping for a `dynamic_update_slice %iterArg_<N>` writeback:
the cascade form `%47 → %67 → %87 → %107` is structurally one rewrite-back chain
on the same position; only the LAST write at that iter index survives (the rest
are overwritten). Seeing N cascaded updates is not a bug — verify by counting
distinct base operands, not total `dynamic_update_slice` instances.

## Reproducing m3 gate failures without `BUILD.bazel` edits

Call `bazel-bin/bench/m3/m3_runner` directly with the chip's fixture quadruple.
No CHIPS / `BUILD.bazel` edits are needed. Once `//examples:<chip>` and
`//bench/m3:m3_runner` are built, iterating on the lowered IR and fixtures takes
~2 s per run without rebuilding the gate test target. See
[./correctness-gate-harness.md](correctness-gate-harness.md) for enrollment
mechanics and the full `m3_runner` flag reference.

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
to `broadcast(constant_0)` and that is visible in the post-opt
`fused_computation` bodies.
