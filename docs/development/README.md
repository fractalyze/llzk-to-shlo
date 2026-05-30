# Development

Day-to-day developer references: conventions, the correctness-gate harness,
MLIR/APInt traps, and CI setup. Each file exists because violating it has broken
us at least once.

- [`conventions.md`](conventions.md) — Load-bearing project-wide conventions
  grouped by topic.
- [`ci-and-build.md`](ci-and-build.md) — Nix-flake-based runner setup and CI
  configuration.
- [`correctness-gate-harness.md`](correctness-gate-harness.md) — Gate mechanics
  (`m3_correctness_gate_test`), enrollment flow, `.json.gate` sentinel format,
  fixture management.
- [`correctness-gate-fixture.md`](correctness-gate-fixture.md) — JSON fixture
  schema consumed by both `m3_runner` and the circom witness binary.
- [`mlir-api-gotchas.md`](mlir-api-gotchas.md) — `cmpi`/`cmpf` predicates live
  in op properties post-Migration, not in the attribute dict; reading via
  `getAttr` silently breaks.
- [`apint-arithmetic.md`](apint-arithmetic.md) — `APInt::getSExtValue()` on 64+
  bit field constants truncates silently (e.g. `1 << 252` → `0` on bn128).
- [`diagnostics.md`](diagnostics.md) — Foot-guns when slicing `stablehlo.while`
  bodies with `awk` and similar tools; false-zero misses most real circuits.
- [`docs-style.md`](docs-style.md) — No mdformat-footnote in pre-commit; use
  inline numeric markers (`¹` `²` …) instead of raw GFM footnote definitions.

Parent: [`../README.md`](../README.md) (narrative spine).
