# llzk-to-shlo

LLZK to StableHLO conversion for GPU-accelerated ZK witness generation.

## Overview

This project converts [LLZK](https://github.com/project-llzk/llzk-lib) circuit
IR to [StableHLO](https://github.com/fractalyze/stablehlo), enabling GPU
execution of ZK witness generation through existing ML compiler infrastructure.

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

### Key Results

- **Coverage**: 45/46 circuits from
  [circom-benchmarks](https://github.com/project-llzk/circom-benchmarks)
  successfully convert through the full pipeline (LLZK -> StableHLO -> Batch)
- **GPU Correctness**: BN254 and BabyBear witnesses verified against circom
  native C++ output
- **Performance**: Batch witness generation achieves up to **95,000x speedup**
  over sequential execution (RTX 5090, N=100K)

## Documentation

| Document                                         | Description                                                                                                                |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| [E2E Lowering Guide](docs/E2E_LOWERING_GUIDE.md) | How LLZK lowers to StableHLO: type conversion, operation patterns, pod elimination, while loop transformation, post-passes |
| [Batch StableHLO](docs/BATCH_STABLEHLO.md)       | IR-level batch witness generation: adding leading batch dimension, op-by-op rules, data-dependent indexing, GPU benchmarks |
| [Circuit Coverage](docs/CIRCUIT_COVERAGE.md)     | Full 123-circuit analysis: pass/fail per stage, error categories, affected circuit families                                |
| [GPU Profiling](docs/GPU_PROFILING.md)           | Nsight Systems profiling: kernel launches, memory transfers, batch vs sequential analysis                                  |
| [CI Setup Guide](docs/CI_SETUP_GUIDE.md)         | CI environment configuration                                                                                               |

## Quick Start

### Prerequisites

- LLVM/MLIR 20.x (`llvm-20-dev`, `libmlir-20-dev` on Debian/Ubuntu)
- Bazel 7.x
- [project-llzk/circom](https://github.com/project-llzk/circom), **`llzk`
  branch**, for LLZK v2.0.0 compatibility (for E2E examples). Earlier circom
  builds emit 1.x IR that no longer parses.

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

### Run Example

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

**Prime field options**:

```bash
--llzk-to-stablehlo="prime=2013265921:i32"   # BabyBear (i32 storage)
--llzk-to-stablehlo="prime=bn254"            # BN254 (i256 storage)
```

## Architecture

### Passes

| Pass                  | Flag                        | Description                                                                                     |
| --------------------- | --------------------------- | ----------------------------------------------------------------------------------------------- |
| SimplifySubComponents | `--simplify-sub-components` | Removes pod dispatch patterns, converting component calls to direct `function.call`             |
| LlzkToStablehlo       | `--llzk-to-stablehlo`       | Converts LLZK operations to StableHLO with type conversion, SSA transformation, and post-passes |
| BatchStablehlo        | `--batch-stablehlo`         | Adds leading batch dimension N to all tensors for parallel GPU witness generation               |

### Internal Pipeline (LlzkToStablehlo)

```
Pre-passes:
  1. eliminateInputPods         -- remove $inputs pod struct members
  2. inlineInputPodCarries      -- unwrap single-field pod while carries
  3. dispatchCallHoisting       -- hoist function.call from scf.if
  4. registerStructFieldOffsets -- compute member offsets for flattening
  5. convertAllFunctions        -- function.def -> func.func
  6. promoteArraysToWhileCarry  -- captured arrays -> while carry
  7. convertWhileBodyArgsToSSA  -- array.write in while -> SSA chain
  8. convertWritemToSSA         -- struct.writem -> SSA chain

Main pass:
  9. applyPartialConversion     -- LLZK ops -> StableHLO ops (1:1 patterns)

Post-passes:
  10. scf.while -> stablehlo.while
  11. scf.if -> stablehlo.select
  12. func.call reconnection
  13. residual LLZK op cleanup
  14. arith -> stablehlo conversion
  15. dead code elimination
  16. while loop vectorization
```

## Coverage

```
circom-benchmarks (6897550c): 123 entry points
  Circom -> LLZK (concrete):  46
  LLZK -> StableHLO:          45 / 46  (97.8%)
  StableHLO -> Batch(N=4):    45 / 45  (100%)
```

The one unsupported circuit (PointCompress) is an ed25519 circuit that produces
21K lines of LLZK IR, exceeding conversion time limits.

### GPU Correctness

- **BN254**: MontgomeryDouble verified against circom native C++ witness (4/4
  fields match)
- **BabyBear**: 9 manual circuits + 39 circom-benchmark circuits verified
  (`batch[i] == single[i]`)

## Testing

| Test                                  | What it verifies                                               | CI    |
| ------------------------------------- | -------------------------------------------------------------- | ----- |
| `//tests:lit_tests`                   | IR pattern FileCheck (11 batch + 5 lowering)                   | Yes   |
| `//tests:batch_smoke_tests`           | 6 inline LLZK -> StableHLO -> Batch (~0.1s)                    | Yes   |
| `//tests:batch_e2e_tests`             | 46 circom-benchmarks full pipeline (manual)                    | Local |
| `//tests:witness_correctness_tests`   | GPU single vs batch comparison                                 | GPU   |
| `//bench/m3:m3_correctness_gate_test` | 22-chip GPU output byte-equal vs circom `.wtns` (M3 gate, N=1) | GPU   |

```bash
# CI tests (fast, no external dependencies)
bazel test //...

# Full E2E regression (requires circom + circom-benchmarks)
bazel test //tests:batch_e2e_tests --test_tag_filters=manual
```

## Examples (Bazel Targets)

All 123 circuits from circom-benchmarks are available as Bazel targets. Each
`circom_to_stablehlo` target also generates an intermediate `_llzk` target.

```bash
# Build all passing circuits (45 + simple)
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
reason. See [docs/CIRCUIT_COVERAGE.md](docs/CIRCUIT_COVERAGE.md) for the full
list.

## Performance

RTX 5090, BabyBear prime, `stablehlo_runner` GPU execution:

### Sigma (x^5) -- simple circuit

| N       | Sequential (est.) | Batched | Speedup |
| ------- | ----------------- | ------- | ------- |
| 100     | 24.7s             | 263ms   | 94x     |
| 1,000   | 4.1min            | 273ms   | 906x    |
| 10,000  | 41.4min           | 247ms   | 10,044x |
| 100,000 | 6.9hr             | 259ms   | 95,598x |

### MontgomeryDouble -- EC point doubling

| N      | Sequential (est.) | Batched | Speedup |
| ------ | ----------------- | ------- | ------- |
| 1,000  | 5.7min            | 364ms   | 940x    |
| 10,000 | 57min             | 333ms   | 10,269x |

Batched execution time is nearly constant (~250-350ms) regardless of N. The
bottleneck in sequential execution is GPU kernel launch overhead (~250ms
per-run); batching amortizes this to a single launch.

## Dependencies

- [llzk-lib](https://github.com/project-llzk/llzk-lib) **v2.0.0** -- LLZK
  dialect definitions (felt, struct, array, pod, poly, ...)
- [stablehlo](https://github.com/fractalyze/stablehlo) -- StableHLO dialect (ZK
  fork with prime field types)
- [circom-benchmarks](https://github.com/project-llzk/circom-benchmarks) -- E2E
  test circuits
- LLVM/MLIR 20.x -- Compiler infrastructure

## License

Apache License 2.0
