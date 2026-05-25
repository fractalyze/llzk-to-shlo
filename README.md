# llzk-to-shlo

LLZK → StableHLO conversion for GPU-accelerated ZK witness generation.

## Overview

This project converts [LLZK](https://github.com/project-llzk/llzk-lib) circuit
IR to [StableHLO](https://github.com/fractalyze/stablehlo), enabling GPU
execution of ZK witness generation through existing ML compiler infrastructure.
The end of the pipeline is GPU execution via open-zkx's `stablehlo_runner`,
which batches N witness computations into a single kernel launch.

```
Circom (.circom)
    |  circom --llzk concrete
    v
LLZK IR (.llzk)
    |  llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo
    v
StableHLO IR (.mlir)
    |  llzk-to-shlo-opt --batch-stablehlo="batch-size=N"
    v
Batched StableHLO IR (.mlir)
    |  stablehlo_runner (GPU)
    v
N witnesses in a single GPU kernel launch
```

**The narrative orientation lives at [`docs/README.md`](docs/README.md).**

## Quick start

### Prerequisites

- LLVM/MLIR 20.x (`llvm-20-dev`, `libmlir-20-dev` on Debian/Ubuntu)
- Bazel 7.x
- [project-llzk/circom](https://github.com/project-llzk/circom), **`llzk`
  branch**, for LLZK v2.0.0 compatibility (for E2E examples). Earlier circom
  builds emit 1.x IR that no longer parses.

See [`docs/development/ci-and-build.md`](docs/development/ci-and-build.md) for
the nix-flake-based runner setup.

### Build

Create `.bazelrc.user` in the project root with your local toolchain paths:

```bash
# .bazelrc.user — adjust paths to match your environment
build --action_env=CC=/usr/bin/clang-20
build --action_env=CXX=/usr/bin/clang++-20
build --action_env=LOCAL_CUDA_PATH=/usr/local/cuda-12.9
build --features=-header_modules --features=-module_maps
build --host_features=-header_modules --host_features=-module_maps
```

Then build and test:

```bash
# Build the optimization tool
bazel build //tools:llzk-to-shlo-opt

# Run all CI tests (LIT + smoke)
bazel test //...
```

### Run an example

```bash
mkdir -p build

# 1. Circom -> LLZK (concrete mode eliminates polymorphism)
circom examples/multiplier2/main.circom --llzk concrete -o build/

# 2. LLZK -> StableHLO
bazel run //tools:llzk-to-shlo-opt -- \
  --simplify-sub-components \
  --llzk-to-stablehlo="prime=bn254" \
  `pwd`/build/main_llzk/main.llzk -o `pwd`/build/main.mlir

# 3. StableHLO -> Batched (N=1000 inputs)
bazel run //tools:llzk-to-shlo-opt -- \
  --batch-stablehlo="batch-size=1000" \
  `pwd`/build/main.mlir -o `pwd`/build/main_batch.mlir

# 4. GPU execution (random inputs for quick test)
bazel run --config cuda_clang @open_zkx//zkx/tools/stablehlo_runner:stablehlo_runner_main -- \
  `pwd`/build/main_batch.mlir \
  --use_random_inputs --print_output=true

# 4b. GPU execution (explicit inputs via JSON)
#   JSON format: {"inputs": [[v0, v1, ...], [v0, v1, ...]]}
#   Each inner array corresponds to one function parameter.
#   For batch-size=4 with 2 scalar inputs: each array has 4 elements.
cat > build/inputs.json << 'JSON'
{"inputs": [[3, 5, 7, 11], [2, 4, 6, 8]]}
JSON
bazel run --config cuda_clang @open_zkx//zkx/tools/stablehlo_runner:stablehlo_runner_main -- \
  `pwd`/build/main_batch.mlir \
  --input_json=`pwd`/build/inputs.json --print_output=true
```

### Prime field options

```bash
--llzk-to-stablehlo="prime=2013265921:i32"   # BabyBear (i32 storage)
--llzk-to-stablehlo="prime=bn254"            # BN254 (i256 storage)
```

## Examples (Bazel targets)

All circuits from circom-benchmarks are available as Bazel targets. Each
`circom_to_stablehlo` target also generates an intermediate `_llzk` target.

```bash
# Build all passing circuits
bazel build //examples/...

# Build a specific circuit's StableHLO
bazel build //examples:montgomerydouble
cat bazel-bin/examples/montgomerydouble.stablehlo.mlir

# Build only the intermediate LLZK IR
bazel build //examples:montgomerydouble_llzk
cat bazel-bin/examples/montgomerydouble_llzk.llzk

# Build a failing circuit (tagged manual — must specify explicitly)
bazel build //examples:zksql_delete  # will fail at circom stage
```

Failing circuits are tagged `manual` with comments indicating the failure
reason.

## Testing

```bash
# CI tests (fast, no external dependencies)
bazel test //...

# Full E2E regression (requires circom + circom-benchmarks)
bazel test //tests:batch_e2e_tests --test_tag_filters=manual
```

Test categories:

- **LIT IR-shape tests** (`//tests:lit_tests`) — FileCheck patterns on IR
  structure after each pass.
- **Batch smoke tests** (`//tests:batch_smoke_tests`) — inline LLZK → StableHLO
  → Batch, fast CI gate.
- **Batch E2E tests** (`//tests:batch_e2e_tests`) — full pipeline through
  circom-benchmarks; requires circom + circom-benchmarks; tagged `manual`.
- **GPU witness-correctness tests** (`//tests:witness_correctness_tests`) —
  single vs batch GPU output comparison.
- **m3 correctness gate** (`//bench/m3:m3_correctness_gate_test`) — GPU output
  byte-equal vs circom `.wtns`; see
  [`docs/contracts/correctness-gate.md`](docs/contracts/correctness-gate.md) and
  [`docs/development/correctness-gate-harness.md`](docs/development/correctness-gate-harness.md).

## Documentation

Start at [`docs/README.md`](docs/README.md) — the narrative spine.

- [`docs/design/`](docs/design/philosophy.md) — the four design bets and why the
  pipeline is shaped this way.
- [`docs/passes/`](docs/passes/simplify-sub-components.md) — per-pass mechanics
  (SSC, LlzkToStablehlo, BatchStablehlo).
- [`docs/contracts/`](docs/contracts/correctness-gate.md) — correctness gate,
  witness-layout-anchor, upstream LLZK drift.
- [`docs/development/`](docs/development/conventions.md) — conventions, m3 gate,
  build/CI, dev-time gotchas.

## Dependencies

- [llzk-lib](https://github.com/project-llzk/llzk-lib) **v2.0.0** — LLZK dialect
  definitions (felt, struct, array, pod, poly, ...)
- [stablehlo](https://github.com/fractalyze/stablehlo) — StableHLO dialect (ZK
  fork with prime field types)
- [circom-benchmarks](https://github.com/project-llzk/circom-benchmarks) — E2E
  test circuits
- LLVM/MLIR 20.x — Compiler infrastructure

## License

Apache License 2.0
