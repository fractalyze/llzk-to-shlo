# Conventions

Project-wide conventions grouped by topic. Every entry is load-bearing — each
one exists because violating it has bitten us before. Add new entries to the
file that fits best, or add a new topic file when an existing one doesn't cover
the concern.

## M3 measurement & correctness gate

The `bench/m3/` harness is the project's primary correctness signal: it feeds
the same fixture to GPU (`m3_runner`) and to circom's C++ witness binary, and
byte-compares the outputs against a `.wtns` ground truth. The fixture and gate
formats have several traps that have cost real debugging time.

- [`conventions/m3-fixture.md`](conventions/m3-fixture.md) — **M3 fixture
  convention**: JSON schema, positional ordering, `@main` arg-order rule for
  chips with non-trivial `public [...]` sets, fixture replication across the
  batch dim, and the `stablehlo_runner_main.cc` helper-port note.
- [`conventions/m3-correctness-gate.md`](conventions/m3-correctness-gate.md) —
  **M3 correctness gate convention**: `.json.gate` sentinel format, `.wtns`
  wire-id layout caveats, internal-wire `-O0` requirement, gate-test enrollment
  workflow, the cache-invalidation gotcha around
  `third_party/circom/workspace.bzl`, and the `zkx_*` (NOT `xla_*`) flag prefix
  for HLO dumps.

## Documentation style

- [`conventions/docs-style.md`](conventions/docs-style.md) — **Markdown
  footnotes**: the `mdformat-footnote` plugin is NOT installed, so use inline
  numeric markers (`¹` `²` …) and a "Notes:" sub-list rather than raw GFM
  `[^id]:` footnote definitions (the autoformatter escapes them to
  `\[^id\]: ...`).
