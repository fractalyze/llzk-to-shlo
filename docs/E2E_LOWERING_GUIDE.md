# Circom to StableHLO E2E Lowering Guide

## Overview

The E2E pipeline from Circom to StableHLO consists of two main tools:

1. **circom (project-llzk/circom)**: Circom → LLZK IR conversion (frontend)
1. **llzk-to-shlo**: LLZK IR → StableHLO conversion (backend)

## E2E Pipeline

```
Circom (.circom)
    │
    ▼  [circom --llzk]
LLZK IR (.mlir)
    │
    ▼  [llzk-to-shlo-opt --llzk-to-stablehlo]
StableHLO IR (.mlir)
    │
    ▼  [stablehlo tools]
Hardware-specific code (CPU, GPU, etc.)
```

______________________________________________________________________

## Step 1: Circom → LLZK IR

### Installation

```bash
# Clone project-llzk/circom repository
git clone https://github.com/project-llzk/circom.git
cd circom

# Build circom (requires LLVM 20.x)
cargo build --release

# Option 1: Add to PATH
export PATH="$PWD/target/release:$PATH"

# Option 2: Set CIRCOM_PATH environment variable
export CIRCOM_PATH="$PWD/target/release/circom"
```

**Note:** Building circom requires LLVM 20.x. See
[project-llzk/circom](https://github.com/project-llzk/circom) for details.

### Usage

```bash
# Compile Circom to LLZK IR
circom circuit.circom --llzk templated -o output_dir

# Options:
#   --llzk templated  : Generate templated LLZK IR (default)
#   --llzk concrete   : Generate concrete LLZK IR
#   --llzk_passes "..." : Apply MLIR pass pipeline
#   --prime <curve>   : Prime to use (bn128, bls12377, goldilocks, etc.)
```

### Output

The file `output_dir/<circuit>_llzk/<circuit>.llzk` will be generated.

**Example LLZK IR:**

```mlir
!felt = !felt.type
func.func @compute(%a: !felt, %b: !felt) -> !felt {
  %sum = felt.add %a, %b : !felt
  %result = felt.mul %sum, %a : !felt
  return %result : !felt
}
```

______________________________________________________________________

## Step 2: LLZK IR → StableHLO

### Installation

```bash
cd llzk-to-shlo

# Build
bazel build //tools:llzk-to-shlo-opt

# Run tests
bazel test //tests/...
```

### Usage

```bash
# Convert LLZK IR to StableHLO (default: BabyBear prime 2013265921, i64 storage)
llzk-to-shlo-opt --llzk-to-stablehlo input.mlir -o output.mlir

# Specify prime and storage type (format: <value>:<type>)
llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" input.mlir  # i32 storage
llzk-to-shlo-opt --llzk-to-stablehlo="prime=2147483647:i64" input.mlir  # i64 storage

# If type is omitted, defaults to i64
llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921" input.mlir
```

### Output

**Example StableHLO IR:**

```mlir
!pf = !field.pf<2147483647 : i64>
func.func @compute(%a: tensor<!pf>, %b: tensor<!pf>) -> tensor<!pf> {
  %sum = stablehlo.add %a, %b : tensor<!pf>
  %result = stablehlo.multiply %sum, %a : tensor<!pf>
  return %result : tensor<!pf>
}
```

______________________________________________________________________

## Type Conversion Table

| LLZK Type               | StableHLO Type                             |
| ----------------------- | ------------------------------------------ |
| `!felt.type`            | `tensor<!field.pf<prime>>`                 |
| `!array.type<N x felt>` | `tensor<N x !field.pf<prime>>`             |
| `!struct.type<@Name>`   | `tensor<M x !field.pf<prime>>` (flattened) |

______________________________________________________________________

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

### Array Operations

| LLZK          | StableHLO                                       |
| ------------- | ----------------------------------------------- |
| `array.new`   | `stablehlo.constant` or `stablehlo.concatenate` |
| `array.read`  | `stablehlo.dynamic_slice` + `stablehlo.reshape` |
| `array.write` | `stablehlo.dynamic_update_slice`                |

### Struct Operations

| LLZK            | StableHLO                                      |
| --------------- | ---------------------------------------------- |
| `struct.new`    | `stablehlo.constant` (zero-initialized tensor) |
| `struct.readf`  | `stablehlo.slice` + `stablehlo.reshape`        |
| `struct.writef` | `stablehlo.dynamic_update_slice`               |

______________________________________________________________________

## Complete E2E Example

### 1. Write Circom Circuit

```circom
// quadratic.circom
pragma circom 2.0.0;

template Quadratic() {
    signal input a;
    signal input b;
    signal input x;
    signal output y;

    signal x_squared;
    x_squared <-- x * x;
    y <-- a * x_squared + b;
}

component main = Quadratic();
```

### 2. Compile to LLZK IR

```bash
circom quadratic.circom --llzk templated -o build/
```

### 3. Convert to StableHLO

```bash
llzk-to-shlo-opt --llzk-to-stablehlo build/quadratic_llzk/quadratic.llzk -o quadratic_shlo.mlir
```

______________________________________________________________________

## Bazel E2E Examples

Pre-configured E2E examples are available using
[circom-benchmarks](https://github.com/project-llzk/circom-benchmarks):

```bash
# Option 1: circom in PATH (auto-detected by Bazel)
export PATH="/path/to/circom/target/release:$PATH"
bazel build //examples:num2bits

# Option 2: Set CIRCOM_PATH environment variable
CIRCOM_PATH=/path/to/circom bazel build //examples:num2bits

# View generated StableHLO
cat bazel-bin/examples/num2bits.stablehlo.mlir

# Build LLZK IR only
bazel build //examples:num2bits_llzk
```

Available targets:

- `//examples:num2bits` - Number to bits conversion
- `//examples:fulladder` - Full adder circuit
- `//examples:decoder` - Decoder circuit
- `//examples:binsum` - Binary sum

______________________________________________________________________

## Important Notes

1. **Scope**: `llzk-to-shlo` only converts `compute()` functions. `constrain()`
   functions are not converted.

1. **Prime Fields**: The default prime is BabyBear (2013265921) with i64 storage
   type. You can specify prime and storage type using the format
   `--llzk-to-stablehlo="prime=<value>:<type>"` (e.g., `prime=2013265921:i32`).

1. **Dependencies**:

   - [llzk-lib](https://github.com/project-llzk/llzk-lib) - LLZK dialect
     definitions
   - [stablehlo](https://github.com/fractalyze/stablehlo) - StableHLO dialect
   - [prime-ir](https://github.com/fractalyze/prime-ir) - Prime field types

______________________________________________________________________

## Key Files Reference

**llzk-to-shlo:**

- `llzk-to-shlo/tools/llzk-to-shlo-opt.cpp` - CLI tool
- `llzk-to-shlo/llzk_to_shlo/Conversion/LlzkToStablehlo/` - Conversion passes
- `llzk-to-shlo/tests/Conversion/LlzkToStablehlo/` - Test examples

**circom (project-llzk):**

- `llzk_backend/src/codegen.rs` - `generate_llzk()` entry point
- `circom/src/input_user.rs` - CLI flags (`--llzk`, `--llzk_passes`)
