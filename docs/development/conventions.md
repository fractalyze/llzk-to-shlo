# Conventions

Project-wide conventions grouped by topic. Every entry is load-bearing — each
one exists because violating it has bitten us before. Add new entries to the
file that fits best, or add a new topic file when an existing one doesn't cover
the concern.

## Correctness gate

- [correctness-gate-harness.md](correctness-gate-harness.md) — **Gate harness
  convention**: `.json.gate` sentinel format, `.wtns` wire-id layout caveats,
  internal-wire `-O0` requirement, gate-test enrollment workflow, the
  cache-invalidation gotcha around `third_party/circom/workspace.bzl`, and the
  `zkx_*` (NOT `xla_*`) flag prefix for HLO dumps.
- [correctness-gate-fixture.md](correctness-gate-fixture.md) — **Gate fixture
  convention**: JSON schema, positional ordering, `@main` arg-order rule for
  chips with non-trivial `public [...]` sets, fixture replication across the
  batch dim, and the `stablehlo_runner_main.cc` helper-port note.

## Build and CI

- [ci-and-build.md](ci-and-build.md) — **CI and build**: prerequisites
  (LLVM/MLIR 20.x, Bazel 7, `.bazelrc.user`), circom via the Nix flake (why the
  nix dev shell is required), runner setup steps, and the `-c opt` vs `-c dbg`
  assert trap.

## Development gotchas

- [mlir-api-gotchas.md](mlir-api-gotchas.md) — **MLIR C++ API gotchas**:
  `arith.cmpi`/`cmpf` predicates in `properties` (not the discardable attr
  dict), `gentbl_cc_library` include-set hygiene, and
  consolidate-the-tblgen-rule advice.
- [apint-arithmetic.md](apint-arithmetic.md) — **APInt arithmetic traps**:
  `getSExtValue()` UB on `getBitWidth() > 64`, and `operator==` bit-width
  mismatch assertion.
- [diagnostics.md](diagnostics.md) — **Diagnostic foot-guns**: `awk`-slicing a
  `stablehlo.while` body (brace counting fails on nested whiles), and
  reproducing m3 gate failures without `BUILD.bazel` edits.

## Documentation style

- [docs-style.md](docs-style.md) — **Markdown footnotes**: the
  `mdformat-footnote` plugin is NOT installed, so use inline numeric markers
  (`¹` `²` …) and a "Notes:" sub-list rather than raw GFM `[^id]:` footnote
  definitions (the autoformatter escapes them to `\[^id\]: ...`).
