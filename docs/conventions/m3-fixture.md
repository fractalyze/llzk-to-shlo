# M3 fixture convention

The M3 measurement harness (`bench/m3/`) feeds the SAME JSON fixture to both
`gpu_zkx` (`m3_runner --input_json=...`) and `cpu_circom` (the circom witness
binary in `run_baseline.sh`). Schema is circom's native form:
`{<signal_name>: scalar | flat-array}`, one top-level key per circuit input
signal. Fixtures live at `bench/m3/inputs/<TARGET>.json` where `<TARGET>`
matches the bazel alias in `bench/m3/run.sh`.

## Multi-dimensional signals flatten to 1-D row-major

Multi-dimensional circom signals (`signal input in[ops][n]`) flatten to a 1-D
row-major array in JSON — `bench/m3/json_input.cc` rejects nested arrays
(`Input "X" must be flat`), and `main_c` accepts the flat form identically to
the nested form for ND signals, so the single-fixture rule survives at any
arity. Empirically verified 2026-05-15 on `binsum` (`in[2][2]`) during the PR
#108 batch-A enrollment.

## GPU parameter mapping is positional in JSON insertion order

MLIR lowering strips circom signal names (parameters surface as `%arg0`,
`%arg1`, …). `bench/m3/json_input.cc` uses `nlohmann::ordered_json` (NOT plain
`nlohmann::json`, which iterates alphabetically) to preserve order; the
`KeyOrderMatters` test pins the contract. When adding a new circuit fixture,
JSON key order must match the order of `func.func @main`'s parameters in the
lowered StableHLO output (see `bazel-bin/examples/<TARGET>.stablehlo.mlir`).

## The lowered `@main` arg order is NOT circom source order with `public [...]`

When the chip declares a non-trivial `public [...]` set, the lowering pulls
publics to the front and groups them by type:

1. Public scalars in `public [...]` list order
1. Public arrays in `public [...]` list order
1. Privates in circom source declaration order

Chips with no `public [...]` (e.g. `iden3_id_ownership_sig`,
`iden3_state_transition`) preserve source order; chips with mixed scalar+array
publics (e.g. `iden3_query_mtp` with `public [..., value, timestamp]` where
`value[64]` is the only array) place the array after all public scalars, not at
its public-list index position.

First-attempt fixtures authored from source order surface as
`INVALID_ARGUMENT: Expected 1 values for shape bn254_sf[] but got 32` at
`json_input.cc` — when this happens, re-derive the key order by diffing the
lowered MLIR `@main` shape pattern against the circom template's public set.

## Fixtures store one witness; `m3_runner` auto-tiles

Fixtures store **one witness's worth of values**, not N copies. `m3_runner`
auto-tiles per-witness tokens across the leading batch dim added by
`--batch-stablehlo`: `LiteralFromDecStrings` accepts
`tokens.size() * shape.dimensions(0) == num_elements` and replicates via
`tokens[i % token_count]`. So a fixture's `in[1600]` array fills
`tensor<N, 1600>` at any N≥1 without fixture changes.

## Don't refactor toward sharing `stablehlo_runner_main.cc` helpers

`stablehlo_runner_main.cc`'s `ParseInputLiteral` / `ParseInputLiteralsFromJson`
are private to that binary's anonymous namespace; we cannot reuse them via
include and have ported the equivalents into `bench/m3/json_input.cc`. Don't
chase a "share the helper" refactor.
