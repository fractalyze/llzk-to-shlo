# Passes

The three core lowering passes that take LLZK IR to batched StableHLO ready for
GPU execution.

- [`simplify-sub-components.md`](simplify-sub-components.md) —
  `SimplifySubComponents` (SSC). Flattens pod dispatch (Circom component
  encoding, ~40 lines per `<==` in SSA) into a form `LlzkToStablehlo` can match.
  Runs to a fixed point because component nesting is arbitrary.
- [`llzk-to-stablehlo/`](llzk-to-stablehlo/README.md) — `LlzkToStablehlo`, the
  heavy conversion pass. Split into pre-passes, a partial-conversion main, and
  post-passes. See [op-lowering.md](llzk-to-stablehlo/op-lowering.md) for type
  and op patterns and
  [while-loop-transformation.md](llzk-to-stablehlo/while-loop-transformation.md)
  for the four-phase while bridge.
- [`batch-stablehlo.md`](batch-stablehlo.md) — `BatchStablehlo`. Prepends a
  leading batch dimension `N` to every tensor; turns single-witness StableHLO
  into an N-witness kernel launch with the `batch-size=N` parameter.

The witness-layout passes (`WitnessLayoutAnchor` + `VerifyWitnessLayout`) are
documented under
[`../contracts/witness-layout-anchor.md`](../contracts/witness-layout-anchor.md)
because their role is contract enforcement, not lowering.

Parent: [`../README.md`](../README.md) (narrative spine).
