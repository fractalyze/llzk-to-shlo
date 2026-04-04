# Batch StableHLO: IR-Level Batch Witness Generation

This document describes the `--batch-stablehlo` pass, which transforms StableHLO
IR to process N inputs simultaneously in a single GPU kernel launch.

## Motivation

The standard pipeline generates one witness per input:

```mlir
@main(%x: tensor<!pf>) -> tensor<3x!pf>
```

Real ZK provers execute the same circuit thousands to millions of times. GPU
throughput comes from batch parallelism, not single-input speed. The batch pass
adds a leading dimension N to all tensors:

```mlir
// Before (single input)
@main(%x: tensor<!pf>) -> tensor<3x!pf>

// After (N inputs batched)
@main(%x: tensor<Nx!pf>) -> tensor<Nx3x!pf>
```

One GPU kernel launch produces N witnesses simultaneously.

## How It Works

### Leading Dimension Insertion

Every tensor gets a leading batch dimension N:

| Original Type     | Batched Type        |
| ----------------- | ------------------- |
| `tensor<!pf>`     | `tensor<Nx!pf>`     |
| `tensor<Mx!pf>`   | `tensor<NxMx!pf>`   |
| `tensor<MxKx!pf>` | `tensor<NxMxKx!pf>` |

### Implementation Structure

The pass operates in two passes per function:

```
batchFunction(funcOp, N):
  1. Update func signature (tensor<M> -> tensor<NxM>)
  2. Update block argument types
  3. Pass 1: Process non-constant ops (element-wise, reshape, slice, etc.)
  4. Pass 2: Broadcast constants (data vs index discrimination)
```

The two-pass constant handling is critical: constants used as
`dynamic_slice`/`dynamic_update_slice` indices must NOT be broadcast (they
remain scalar), while data constants must be broadcast to match batch
dimensions.

______________________________________________________________________

## Op-by-Op Transformation Rules

### Element-wise Ops (simplest)

Only the result type changes. Broadcasting handles the rest automatically:

```mlir
// Before
%0 = stablehlo.add %a, %b : tensor<!pf>

// After (N=4)
%0 = stablehlo.add %a, %b : tensor<4x!pf>
```

Supported: add, subtract, multiply, divide, negate, power, abs, convert, rem,
and, or, xor, shift_left, shift_right_logical, not.

### Constants -> broadcast_in_dim

Constants are broadcast along the batch dimension:

```mlir
// Before
%c = stablehlo.constant dense<42> : tensor<i32>

// After
%c = stablehlo.constant dense<42> : tensor<i32>          // original kept
%bc = stablehlo.broadcast_in_dim %c, dims = []
    : (tensor<i32>) -> tensor<4xi32>                     // batch dim added
```

**Index constant discrimination**: Constants used as `dynamic_slice` or
`dynamic_update_slice` start indices are NOT broadcast. The pass uses
`isSliceIndexUse()` to check whether a constant flows into a slice index
operand.

### dynamic_slice / dynamic_update_slice

**Case 1: Constant indices** (common case)

Prepend a batch dim index of 0:

```mlir
// Before
stablehlo.dynamic_slice %t, %idx, sizes=[1]

// After
%zero = stablehlo.constant dense<0> : tensor<i32>
stablehlo.dynamic_slice %t, %zero, %idx, sizes=[N, 1]
```

**Case 2: Data-dependent indices** (e.g., AES lookup tables)

When indices are per-batch-element (`tensor<N>`), `dynamic_slice` cannot accept
them. The pass rewrites to a **one-hot gather** pattern:

```
One-Hot Read (Gather):
  iota = [0, 1, ..., M-1]              // table indices
  mask = compare(iota, idx, EQ)        // NxM boolean mask
  selected = table * mask              // zero out non-selected
  result = reduce_sum(dim=1)           // collapse to N values
```

Detection: `hasAnyBatchedIndex()` checks if any index operand has a batched type
(rank increased after batching).

**Case 3: Multiple data-dependent indices** (e.g., AES 2D table)

For multi-dimensional lookups where both row and column indices are
per-batch-element, AND the per-dimension one-hot masks:

```mlir
// tensor<4x256> table, both indices batched
mask_row = compare(iota_row, row_idx, EQ)  // tensor<Nx4>
mask_col = compare(iota_col, col_idx, EQ)  // tensor<Nx256>
combined = and(mask_row, mask_col)          // tensor<Nx4x256> (broadcast)
selected = table * combined
result = reduce_sum(dim=1, dim=2)           // tensor<N>
```

### One-Hot Write (Scatter)

`dynamic_update_slice` with data-dependent indices uses the same one-hot pattern
in reverse:

```mlir
// select(mask, broadcast(update_value), original_tensor)
// mask=true positions get the update, rest keeps original
```

### stablehlo.while (fixed trip count)

Circom while loops have fixed iteration counts (all batch elements execute the
same number of iterations). The batch dimension is added to carry types, and the
condition predicate is extracted from a single batch element:

```mlir
// After batching: carry has batch dim
stablehlo.while(%carry: tensor<Nx33x!pf>) {
  // cond: all batch elements identical, extract element[0]
  %pred_N = stablehlo.compare LT, ...  // tensor<Nxi1>
  %pred_1 = stablehlo.slice [0:1]      // tensor<1xi1>
  %pred = stablehlo.reshape             // tensor<i1>
  stablehlo.return %pred
} do {
  // body: all ops have batch dim applied
  ...
}
```

### func.call, compare, select

- **func.call**: Callee is also batched, so only the result type changes
- **compare**: Result becomes `tensor<Nxi1>`
- **select**: Predicate and result both get batch dim

### LLZK Residual Ops

If pod dispatch was not fully eliminated, residual LLZK dialect ops (`pod.new`,
`array.new`, etc.) may remain in the StableHLO output. These are dead code and
are skipped by the batch pass.

______________________________________________________________________

## Usage

```bash
# 1. LLZK -> StableHLO
llzk-to-shlo-opt --simplify-sub-components \
  --llzk-to-stablehlo="prime=2013265921:i32" circuit.llzk -o circuit.mlir

# 2. StableHLO -> Batched (N=1000)
llzk-to-shlo-opt --batch-stablehlo="batch-size=1000" \
  circuit.mlir -o circuit_batch.mlir

# 3. GPU execution
stablehlo_runner circuit_batch.mlir \
  --input_json=batch_inputs.json --print_output=true
```

______________________________________________________________________

## Performance Benchmarks

RTX 5090, BabyBear prime, `stablehlo_runner` GPU execution.

### Sigma (x^5) -- simple circuit

| N       | Sequential (est.) | Batched | Speedup     |
| ------- | ----------------- | ------- | ----------- |
| 100     | 24.7s             | 263ms   | **94x**     |
| 1,000   | 4.1min            | 273ms   | **906x**    |
| 10,000  | 41.4min           | 247ms   | **10,044x** |
| 100,000 | 6.9hr             | 259ms   | **95,598x** |

### MontgomeryDouble -- EC point doubling

| N      | Sequential (est.) | Batched | Speedup     |
| ------ | ----------------- | ------- | ----------- |
| 1,000  | 5.7min            | 364ms   | **940x**    |
| 10,000 | 57min             | 333ms   | **10,269x** |

**Key observation**: Batched execution time is nearly constant (~250-350ms)
regardless of N. Sequential time scales linearly with N. The bottleneck in
sequential execution is GPU kernel launch overhead (~250ms per run); batching
amortizes this to a single launch.

Sequential estimates are extrapolated from single-input GPU execution time
(~247ms for Sigma, ~342ms for MontgomeryDouble) multiplied by N.

______________________________________________________________________

## Verification

### Coverage

```
circom-benchmarks (6897550c): 123 entry points
  Circom -> LLZK (concrete):  46
  LLZK -> StableHLO:          45 / 46  (97.8%)
  StableHLO -> Batch(N=4):    45 / 45  (100%)
```

### GPU Correctness

- **BN254**: circom native C++ witness == GPU output (MontgomeryDouble, 4/4
  fields match)
- **BabyBear**: 9 manual circuits + 39 circom-benchmark circuits verified
  (`batch[i] == single[i]` for all i)

### Test Suite

| Test                                | What it verifies                                     | CI    |
| ----------------------------------- | ---------------------------------------------------- | ----- |
| `//tests:lit_tests`                 | IR pattern FileCheck (11 batch tests)                | Yes   |
| `//tests:batch_smoke_tests`         | 6 inline LLZK circuits -> StableHLO -> Batch (~0.1s) | Yes   |
| `//tests:batch_e2e_tests`           | 46 circom-benchmarks full pipeline (manual)          | Local |
| `//tests:witness_correctness_tests` | GPU single vs batch comparison                       | GPU   |

______________________________________________________________________

## Known Limitations

| Item                   | Status                                                                           |
| ---------------------- | -------------------------------------------------------------------------------- |
| PointCompress (1/46)   | LLZK conversion timeout (21K-line ed25519 circuit)                               |
| BN254 full correctness | Only MontgomeryDouble verified; remaining circuits need BN254 witness comparison |
