# llzk-to-shlo

LLZK to StableHLO conversion pass for witness generation IR.

## Overview

This project provides an MLIR conversion pass that lowers LLZK dialect
operations (from `compute()` functions) to StableHLO IR. This enables hardware
acceleration of witness generation through StableHLO's optimization and lowering
pipelines.

## Scope

- **Converts**: `compute()` functions (WitnessGen trait)
- **Does not convert**: `constrain()` functions (constraint generation)

## Type Conversion

| LLZK Type               | StableHLO Type                             |
| ----------------------- | ------------------------------------------ |
| `!felt.type`            | `tensor<!field.pf<prime>>`                 |
| `!array.type<N x felt>` | `tensor<N x !field.pf<prime>>`             |
| `!struct.type<@Name>`   | `tensor<M x !field.pf<prime>>` (flattened) |

## Operation Mapping

### Felt Operations

| LLZK         | StableHLO                  |
| ------------ | -------------------------- |
| `felt.add`   | `stablehlo.add`            |
| `felt.sub`   | `stablehlo.subtract`       |
| `felt.mul`   | `stablehlo.multiply`       |
| `felt.div`   | `stablehlo.divide`         |
| `felt.neg`   | `stablehlo.negate`         |
| `felt.inv`   | `stablehlo.divide` (1 / x) |
| `felt.const` | `stablehlo.constant`       |

### Struct Operations

| LLZK            | StableHLO                                      |
| --------------- | ---------------------------------------------- |
| `struct.new`    | `stablehlo.constant` (zero-initialized tensor) |
| `struct.readf`  | `stablehlo.slice` + `stablehlo.reshape`        |
| `struct.writef` | `stablehlo.dynamic_update_slice`               |

### Array Operations

| LLZK            | StableHLO                                           |
| --------------- | --------------------------------------------------- |
| `array.new`     | `stablehlo.constant` or `stablehlo.concatenate`     |
| `array.read`    | `stablehlo.dynamic_slice` + `stablehlo.reshape`     |
| `array.write`   | `stablehlo.dynamic_update_slice`                    |
| `array.extract` | `stablehlo.dynamic_slice`                           |
| `array.insert`  | `stablehlo.dynamic_update_slice`                    |
| `array.len`     | `arith.constant` (static) or `tensor.dim` (dynamic) |

## Building

```bash
# Build all targets
bazel build //...

# Build the optimization tool
bazel build //tools:llzk-to-shlo-opt

# Run tests
bazel test //tests/...
```

## Usage

```bash
# Convert LLZK IR to StableHLO (default: BabyBear prime 2013265921 with i64 storage)
llzk-to-shlo-opt --llzk-to-stablehlo input.mlir -o output.mlir

# Specify prime and storage type (format: <value>:<type>)
llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" input.mlir  # i32 storage
llzk-to-shlo-opt --llzk-to-stablehlo="prime=2147483647:i64" input.mlir  # i64 storage

# Type can be omitted (defaults to i64)
llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921" input.mlir
```

## Example

Input (LLZK):

```mlir
!felt = !felt.type
func.func @compute(%a: !felt, %b: !felt) -> !felt {
  %sum = felt.add %a, %b : !felt
  %result = felt.mul %sum, %a : !felt
  return %result : !felt
}
```

Output (StableHLO with `prime=2013265921:i32`):

```mlir
!pf = !field.pf<2013265921 : i32>
func.func @compute(%a: tensor<!pf>, %b: tensor<!pf>) -> tensor<!pf> {
  %sum = stablehlo.add %a, %b : tensor<!pf>
  %result = stablehlo.multiply %sum, %a : tensor<!pf>
  return %result : tensor<!pf>
}
```

## E2E Examples (Circom → StableHLO)

E2E examples are provided using
[circom-benchmarks](https://github.com/project-llzk/circom-benchmarks).

### Prerequisites

Install the LLZK-enabled circom compiler:

```bash
git clone https://github.com/project-llzk/circom.git
cd circom
cargo build --release

# Option 1: Add to PATH
export PATH="$PWD/target/release:$PATH"

# Option 2: Set CIRCOM_PATH environment variable
export CIRCOM_PATH="$PWD/target/release/circom"
```

**Note:** Building circom requires LLVM 20.x. See
[project-llzk/circom](https://github.com/project-llzk/circom) for details.

### Running E2E Examples

```bash
# Build a specific example (Circom → LLZK → StableHLO)
bazel build //examples:num2bits

# View the generated StableHLO IR
cat bazel-bin/examples/num2bits.stablehlo.mlir

# Build intermediate LLZK IR only
bazel build //examples:num2bits_llzk
cat bazel-bin/examples/num2bits.llzk
```

### Available Examples

| Target                 | Description        |
| ---------------------- | ------------------ |
| `//examples:num2bits`  | Number to bits     |
| `//examples:fulladder` | Full adder circuit |
| `//examples:decoder`   | Decoder circuit    |
| `//examples:binsum`    | Binary sum         |

## Dependencies

- [llzk-lib](https://github.com/project-llzk/llzk-lib) - LLZK dialect
  definitions
- [stablehlo](https://github.com/fractalyze/stablehlo) - StableHLO dialect
- [prime-ir](https://github.com/fractalyze/prime-ir) - Prime field types
- LLVM/MLIR - Compiler infrastructure

## License

Apache License 2.0
