# M3 correctness gate convention

The `bench/m3/` gate opts in per-circuit via a
`bench/m3/inputs/<TARGET>.json.gate` sentinel. Sentinel content is the `.wtns`
wire-index list (one per output Literal element, space- or comma-separated); an
empty file defaults to contiguous `[1..1+N)`. `m3_runner` reads the index list
through `--gate_wtns_indices=...` and byte-compares the GPU output Literal
against `wtns.Witness(idx)` for each declared index.

## `.wtns` wire layout caveat

Circom does NOT always assign wire IDs as
`[const, outputs, inputs, intermediates]`. `MontgomeryDouble`'s `.wtns`
interleaves outputs around echoed-input wires, so its sentinel is `1 2 5 6`. For
each new gated circuit, decode the `.wtns` integers to confirm input wire
positions (echoed inputs match the JSON fixture's values verbatim) — or run the
gate with provisional indices and read the mismatch-hex-diff.

## Internal-wire sentinels require `circom --O0`

Circom's default `-O1` simplifier removes any wire whose value is fully
derivable from already-emitted wires, marking it `-1` in the `.sym` file's
second column. Roughly half the internal wires for a Poseidon-heavy chip (every
`hasher.out`, `signature.out`, `inKeypair.hasher.out`, etc.) drop to `-1` under
`-O1` — there's no `.wtns` wire to compare against, so a sentinel that maps GPU
output[i] → a folded signal must either duplicate-index to a zero-value wire
(`witness_compare`'s "duplicate indices" trick) or rebuild the ground-truth
witness with `-O0` so every signal gets an explicit wire ID.

For new gated chips whose GPU output covers internal slots beyond the input
region (any chip past keccak-class complexity, e.g. webb's `Transaction` chain),
commit a `-O0`-generated `.wtns` instead of the `-O1` default. The `-O0` `.wtns`
is roughly 2× the `-O1` size but the trade is worth the byte-equality coverage.

## Duplicate indices, padding, and tuple/N>1 restrictions

`witness_compare`'s API accepts duplicate indices, required for circuits whose
GPU output flattens public + private intermediate signals into one tensor (e.g.
`keccak_pad` emits `tensor<2176>` = `out[1088] || out2[1088]`). When `@main`
reduces to `dynamic_update_slice(zeros<N>, %result<M>, 0)` with M < N, assign
each trailing pad position any `.wtns` index whose value decodes to 0.

The gate rejects tuple shapes / N>1 batched outputs; it is N=1 single-tensor
only. `run.sh` auto-skips at N>1 with an actionable message.

**Constraint-only templates lower to `tensor<0>` — gateable as shape-stability
anchors via vacuous PASS.** `witness_compare` early-returns OkStatus when
`num_elements == 0`, so a chip with no public signals can still be gated with an
empty `.json.gate`; a future regression that turns the output into `tensor<N>`
with N>0 surfaces at the existing `element-count != index-count` check.

## Enrollment workflow

**Every newly gated circuit must be added to the CI regression test in the same
PR that lands the fixture.** `//bench/m3:m3_correctness_gate_test` (`gpu`-tagged
`sh_test`) runs `m3_runner --correctness_gate=true` against each chip in its
`data` block:

1. Extend `CHIPS=(...)` in `bench/m3/m3_correctness_gate_test.sh`.
1. Append the matching `//examples:<chip>` target plus the `.json` /
   `.json.gate` / `.wtns` trio to `data = [...]` in `bench/m3/BUILD.bazel`.

Skip ⇒ the gate is a manual-checkpoint-only artifact.

**Pre-enrollment bug repro and `awk`-slicing foot-guns** — see
[`../LOWERING_PITFALLS.md#diagnostic-foot-guns`](../LOWERING_PITFALLS.md#diagnostic-foot-guns)
for the `m3_runner` direct-invocation recipe (no CHIPS / BUILD.bazel edits) and
the brace-balanced extractor convention.

**Don't ship a gate sentinel before its baseline is currently green.** A new
`.json.gate` whose byte-equal compare fails at landing buys zero regression
protection. Bundle the sentinel + `data=[...]` + `CHIPS` updates into the same
PR as the lowering / compute fix that flips the metric red → green.

## Circom binary swaps don't invalidate Bazel's cache

`third_party/circom/workspace.bzl` writes a wrapper `exec`-ing the resolved
circom path. Bazel hashes the wrapper text, not binary content. Updating
`/usr/local/bin/circom` without changing the path string leaves wrapper text
identical → all downstream `circom_to_llzk`, stablehlo conversion, and
`m3_correctness_gate_test` outputs cache-hit. Symptom:
`m3_correctness_gate_test (cached) PASSED in <past time>`.

Local invalidation:

- `--disk_cache=` (empty), OR
- `--repo_env=CIRCOM_PATH=/usr/local/./bin/circom` (path perturbation forces
  repository_rule re-eval).

## `circom_to_llzk` `--stabilize` is load-bearing

`circom_to_llzk` passes `--stabilize` so multi-sub-component composite chips
have deterministic `struct.member` ordering. Without the flag,
project-llzk/circom seeds its symbol-table / sub-component registry from a
default Rust `HashMap` `RandomState`, so any chip whose main `struct.def`
declares ≥2 sub-component-derived members shuffles output ordering per process
(proven 2026-05-04: 3 fresh runs, 3 distinct LLZK md5s on `splicer_test`).

Position-based `.json.gate` sentinels map GPU output offsets to `.wtns` wire
indices via that ordering, so non-deterministic emission would drift the gate
every CI run. `--stabilize` sorts the iteration order into something
reproducible — `splicer_test.llzk` md5 is byte-stable across runs once enabled.
Single-output chips (`@out` only) are immune either way. Keep the flag pinned in
`examples/e2e.bzl::_circom_to_llzk_impl`; dropping it returns the gate to its
drifting state.

**Methodology trap:** `bazel clean --expunge` does NOT wipe `--disk_cache=PATH`
configured in user-level `~/.bazelrc`; pass `--disk_cache=` (empty) when probing
for variance, otherwise the second run cache-hits the first.

## HLO dumps use `zkx_*`, not `xla_*`

Dumping post-XLA HLO from `m3_runner` uses the `zkx_*` flag prefix, NOT `xla_*`.
open-zkx renames the XLA flag namespace end-to-end:
`zkx::AppendDebugOptionsFlags` (called from `bench/m3/m3_runner_main.cc:345`)
registers the renamed flags as CLI options AND triggers
`ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", ...)` in `AllocateFlags`, so the
env-var path is alive too — it just lives under `ZKX_FLAGS`, not `XLA_FLAGS`.

Equivalent invocations:

```bash
# CLI form
bazel-bin/bench/m3/m3_runner \
  "--zkx_dump_to=$DUMP" "--zkx_dump_hlo_as_text=true" \
  --circuit=<chip> --N=1 --iterations=1 --warmups=0 \
  --input_json=bench/m3/inputs/<chip>.json \
  bazel-bin/examples/<chip>.stablehlo.mlir

# Env-var form (parsed by AllocateFlags via call_once on first
# AppendDebugOptionsFlags; same flag list, same effect)
ZKX_FLAGS="--zkx_dump_to=$DUMP --zkx_dump_hlo_as_text=true" \
  bazel-bin/bench/m3/m3_runner \
  --circuit=<chip> --N=1 --iterations=1 --warmups=0 \
  --input_json=bench/m3/inputs/<chip>.json \
  bazel-bin/examples/<chip>.stablehlo.mlir
```

The `module_0001.main.sm_*_gpu_after_optimizations.txt` dump carries
`source_file="-" source_line=<N>` metadata mapping each HLO op back to the
lowered MLIR line number — invaluable for tracing which while-body corresponds
to which `stablehlo.while` in the chip. List all flags via
`bazel-bin/bench/m3/m3_runner --help | grep -iE 'xla|dump'`.

The `xla_*` CLI flags and `XLA_FLAGS` env var both silently no-op (the former is
parsed as a positional MLIR file path and fails with `NOT_FOUND`; the latter is
never read because open-zkx hard-codes `ZKX_FLAGS` as the env-var name in
`open-zkx/zkx/debug_options_flags.cc::AllocateFlags`).
