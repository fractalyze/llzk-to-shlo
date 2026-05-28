# Design

The four bets that shape every trade-off in this codebase, and the pipeline
shape they force.

- [`philosophy.md`](philosophy.md) — The four design bets: reuse ML compiler
  infrastructure, batch first, GPU correctness is the gate, frontend-agnostic.
- [`pipeline-shape.md`](pipeline-shape.md) — Why the pipeline splits where it
  does (pod-dispatch elimination, SSC fixed point, four-phase while-loop
  transformation, post-passes).
- [`failure-modes.md`](failure-modes.md) — Structural families of lowering
  failures (Circom→LLZK timeouts, SSC matching gaps, while-loop FFI mismatches)
  and where to look for fixes.

Parent: [`../README.md`](../README.md) (narrative spine).
