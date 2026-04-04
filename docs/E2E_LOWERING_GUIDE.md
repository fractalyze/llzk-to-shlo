# LLZK to StableHLO Lowering Guide

This document describes how LLZK IR (from Circom circuits) is lowered to
StableHLO IR for GPU execution. It covers type conversion, operation patterns,
pod dispatch elimination, while loop transformation, and post-passes.

## Pipeline Overview

```
Circom (.circom)
    |  circom --llzk concrete
    v
LLZK IR (.llzk)
    |  --simplify-sub-components (pod dispatch elimination)
    v
LLZK IR (clean, no pods)
    |  --llzk-to-stablehlo (pre-passes + main + post-passes)
    v
StableHLO IR (.mlir)
    |  stablehlo_runner / open-zkx GPU backend
    v
GPU execution (witness output)
```

Each stage:

| Stage           | Tool                     | What it does                                                   |
| --------------- | ------------------------ | -------------------------------------------------------------- |
| Circom -> LLZK  | `circom --llzk concrete` | template -> struct, signal -> felt, `<==` -> compute/constrain |
| Pod elimination | `SimplifySubComponents`  | Removes pod dispatch patterns (component -> function.call)     |
| Type conversion | `TypeConverter`          | felt -> tensor, struct -> flatten, array -> tensor             |
| Op conversion   | `ConversionPatterns`     | felt.add -> stablehlo.add, struct -> slice/update, etc.        |
| While transform | pre/post-passes          | carry promotion -> SSA -> stablehlo.while                      |
| GPU execution   | `stablehlo_runner`       | StableHLO -> MHLO -> HLO -> CUDA kernel                        |

______________________________________________________________________

## Type Conversion

All ZK data becomes tensors with `!field.pf<prime>` element type.

### felt -> scalar tensor

```mlir
// LLZK
%x : !felt.type

// StableHLO (0-dimensional scalar tensor)
%x : tensor<!field.pf<2013265921:i32>>
```

### array -> tensor (shape preserved)

```mlir
!array.type<8 x felt>    ->  tensor<8 x !pf>
!array.type<4,3 x felt>  ->  tensor<4x3 x !pf>
```

### struct -> flattened tensor

Structs are flattened into a 1D tensor. Each felt member occupies 1 slot, each
`array<N>` member occupies N slots, and nested structs are recursively
flattened.

```
struct @LessThan {
  @out : !felt.type                  ->  1 slot  (offset 0)
  @n2b : !struct.type<@Num2Bits>     -> 33 slots (offset 1..33)
}
-> tensor<34 x !pf>    (1 + 33 = 34)
```

Member access becomes offset-based slicing:

```mlir
// LLZK: access by name
struct.readm %self[@x4]

// StableHLO: access by offset
stablehlo.slice %self [1:2:1]   // offset 1, size 1
```

### pod -> removed

Pod types have no StableHLO equivalent. They are eliminated by
`SimplifySubComponents` before conversion (see Pod Dispatch Elimination below).

### i1 -> tensor\<i1>

Boolean values used as `scf.while` carry are converted from bare `i1` to
`tensor<i1>`. The `llzk.nondet : i1` pre-pass replaces these with
`arith.constant false` before type conversion.

### Summary

| LLZK Type               | StableHLO Type    | Rule                           |
| ----------------------- | ----------------- | ------------------------------ |
| `!felt.type`            | `tensor<!pf>`     | 0-d scalar tensor              |
| `!array.type<N x felt>` | `tensor<N x !pf>` | shape preserved                |
| `!struct.type<@Name>`   | `tensor<K x !pf>` | flatten (K = total felt count) |
| `!pod.type<...>`        | (removed)         | eliminated in pre-pass         |
| `i1`                    | `tensor<i1>`      | while carry only               |

______________________________________________________________________

## Operation Patterns

### Arithmetic (1:1 mapping)

| LLZK           | StableHLO                                     |
| -------------- | --------------------------------------------- |
| `felt.add`     | `stablehlo.add`                               |
| `felt.sub`     | `stablehlo.subtract`                          |
| `felt.mul`     | `stablehlo.multiply`                          |
| `felt.div`     | `stablehlo.divide`                            |
| `felt.neg`     | `stablehlo.negate`                            |
| `felt.inv %a`  | `stablehlo.divide(1, %a)`                     |
| `felt.const N` | `stablehlo.constant dense<N>`                 |
| `felt.pow`     | `stablehlo.power` (exponent converted to i32) |

### Bitwise Operations (3-step conversion)

`!field.pf` has no bitwise operations. Convert to integer, perform the
operation, convert back:

```mlir
// felt.bit_and %a, %b
%a_i32 = stablehlo.convert %a : tensor<!pf> -> tensor<i32>   // step 1
%b_i32 = stablehlo.convert %b : tensor<!pf> -> tensor<i32>
%r_i32 = stablehlo.and %a_i32, %b_i32 : tensor<i32>         // step 2
%r     = stablehlo.convert %r_i32 : tensor<i32> -> tensor<!pf>  // step 3
```

Same pattern applies to: `felt.shr` (shift_right_logical), `felt.shl`
(shift_left), `felt.bit_or` (or), `felt.bit_xor` (xor).

### Integer Division/Remainder (3-step conversion)

Unsigned integer operations use the same 3-step pattern as bitwise ops:

| LLZK           | StableHLO (via i32)   | Example                |
| -------------- | --------------------- | ---------------------- |
| `felt.umod`    | `stablehlo.remainder` | FullAdder: `(a+b) % 2` |
| `felt.uintdiv` | `stablehlo.divide`    | FullAdder: `(a+b) / 2` |

Note: `felt.div` is **field division** (modular inverse), while `felt.uintdiv`
is **integer division**. Both map to `stablehlo.divide` but at different type
levels.

### Struct Operations (offset-based slicing)

```mlir
// struct.new -> zero tensor
%self = stablehlo.constant dense<0> : tensor<3x!pf>

// struct.readm -> slice + reshape
%sliced = stablehlo.slice %self [1:2:1] : tensor<3x!pf> -> tensor<1x!pf>
%val = stablehlo.reshape %sliced : tensor<1x!pf> -> tensor<!pf>

// struct.writem -> dynamic_update_slice (SSA chain)
%val_1d = stablehlo.reshape %val : tensor<!pf> -> tensor<1x!pf>
%idx = stablehlo.constant dense<2> : tensor<i32>
%self_new = stablehlo.dynamic_update_slice %self, %val_1d, %idx
```

LLZK's mutable `struct.writem` becomes an SSA chain of `dynamic_update_slice`
operations:

```mlir
// LLZK: same %self mutated repeatedly
struct.writem %self[@x2] = %x2
struct.writem %self[@x4] = %x4
struct.writem %self[@out] = %x5

// StableHLO: SSA chain, each producing a new tensor
%s1 = stablehlo.dynamic_update_slice %s0, %x2, [0]
%s2 = stablehlo.dynamic_update_slice %s1, %x4, [1]
%s3 = stablehlo.dynamic_update_slice %s2, %x5, [2]
return %s3
```

### Array Operations

```mlir
// array.read -> dynamic_slice + reshape
%idx_i32 = stablehlo.convert %idx : tensor<!pf> -> tensor<i32>
%sliced = stablehlo.dynamic_slice %arr, %idx_i32, sizes=[1]
%elem = stablehlo.reshape %sliced : tensor<1x!pf> -> tensor<!pf>

// array.write -> dynamic_update_slice
%val_1d = stablehlo.reshape %val : tensor<!pf> -> tensor<1x!pf>
%idx_i32 = stablehlo.convert %idx
%arr_new = stablehlo.dynamic_update_slice %arr, %val_1d, %idx_i32
```

### Boolean and Comparison

| LLZK                       | StableHLO                      |
| -------------------------- | ------------------------------ |
| `bool.cmp lt(%a, %b)`      | `stablehlo.compare LT`         |
| `bool.and`                 | `stablehlo.and`                |
| `bool.or`                  | `stablehlo.or`                 |
| `bool.not %a : i1`         | `arith.xori %a, true` (scalar) |
| `bool.not %a : tensor<i1>` | `stablehlo.not %a` (tensor)    |

### Control Flow

| LLZK                         | StableHLO                                  |
| ---------------------------- | ------------------------------------------ |
| `scf.if`                     | `stablehlo.select` (both branches inlined) |
| `function.call @A::@compute` | `func.call @A_compute` (name flattened)    |
| `cast.toindex`               | `stablehlo.convert` (field -> i32)         |

### Removed Operations

| LLZK                          | Handling                      | Reason                           |
| ----------------------------- | ----------------------------- | -------------------------------- |
| `struct.def`, `struct.member` | Deleted                       | Only used for offset calculation |
| `function.def @constrain`     | Deleted                       | Not needed for witness gen       |
| `constrain.eq`                | Deleted                       | Verification only                |
| `bool.assert`                 | Deleted                       | Runtime unnecessary              |
| `felt.nondet`, `llzk.nondet`  | `stablehlo.constant dense<0>` | Zero-initialized                 |

### cast.toindex Constant Preservation

`cast.toindex` converts felt constants to array indices. The conversion extracts
the actual constant value from the defining `felt.const` op instead of using a
fallback of 0, which would cause incorrect array writes:

```mlir
// LLZK
%c1 = felt.const 1
%i1 = cast.toindex %c1        // index = 1

// StableHLO (correct: extracts value 1)
%i1 = stablehlo.constant dense<1> : tensor<i32>
```

______________________________________________________________________

## Pod Dispatch Elimination

Circom's `component` calls generate ~40 lines of pod dispatch in LLZK (pod.new,
pod.read, pod.write, counter logic, scf.if conditional execution). Since
`!pod.type` has no StableHLO equivalent, all pod patterns must be removed before
conversion.

### SimplifySubComponents (6-phase, fixed-point)

Runs repeatedly until no more changes:

| Phase | Name                          | What it does                                                    |
| ----- | ----------------------------- | --------------------------------------------------------------- |
| -1    | flattenPodArrayWhileCarry     | Split `array<N x pod>` into per-field arrays                    |
| 0     | unpackPodWhileCarry           | Expand pod while carry into individual fields                   |
| 1     | extractCallsFromScfIf         | **Core**: extract `function.call` from `scf.if` to parent block |
| 2     | replacePodReads               | Replace `pod.read` with tracked values                          |
| 3     | eraseStructWritemForPodValues | Remove pod/struct type writem                                   |
| 4     | eraseDeadPodAndCountOps       | Remove unused pod/arith/scf.if                                  |
| 5     | replaceRemainingPodOps        | Replace remaining pod.read -> nondet, delete pod.new            |

**Why fixed-point?** Sub-components can be nested multiple levels deep
(GreaterThan -> LessThan -> Num2Bits). Each iteration peels one level.

**Phase 1 is the core transformation**: it tracks pod field values through the
block, finds `function.call` inside `scf.if (count == 0)`, and hoists the call
outside with tracked arguments.

```mlir
// Before (~40 lines of pod dispatch)
%pod = pod.new { @count = 2 }
pod.write %pod_in[@in][0] = %val1
pod.write %pod_in[@in][1] = %val2
scf.if count==0 { function.call @LessThan::@compute(pod_in[@in]) }
result = pod.read %pod[@comp]

// After (clean function call)
%inputs = array.new : <2 x !felt.type>
array.write %inputs[0] = %val1
array.write %inputs[1] = %val2
%result = function.call @LessThan::@compute(%inputs)
```

### Additional Pre-passes (in LlzkToStablehlo)

Some pod patterns survive `SimplifySubComponents` and are handled by internal
pre-passes:

1. **eliminateInputPods**: Removes `$inputs` pod struct members (constrain-only,
   not needed for witness gen)

1. **inlineInputPodCarries**: Unwraps single-field input pods used as while
   carry, replacing the pod wrapper with the inner type directly

1. **Dispatch call hoisting**: For array-of-dispatch-pods patterns, hoists
   `function.call` and its operands from `scf.if` to the parent block to avoid
   SSA domination errors

1. **resolveArrayPodCompReads**: When multiple dispatches exist in the same
   while body, matches `pod.read @comp` to the correct call result using result
   type (not just the last call)

______________________________________________________________________

## While Loop Transformation

LLZK's `scf.while` is mutable (can modify arrays defined outside the loop).
StableHLO's `stablehlo.while` is immutable (all values must be passed as carry).
This is the most complex part of the conversion.

### Step 1: Promote Captured Arrays to While Carry

Arrays defined outside the loop but modified inside are detected and added as
while carry arguments:

```mlir
// Before: external array modified inside loop
%arr = llzk.nondet : !array.type<33 x !felt.type>
scf.while (%i = %c0) {
  array.write %arr[%i] = %bit    // external array!
  scf.yield %next_i
}

// After: array promoted to carry
scf.while (%i = %c0, %arr_carry = %arr) {
  %arr_new = array.write %arr_carry[%i] = %bit   // SSA: returns new value
  scf.yield %next_i, %arr_new                    // propagate via carry
}
```

**Domination safety**: Arrays defined in a parent while body are not promoted
(would cause SSA domination violation since the init value is evaluated before
the body).

### Step 2: Mutable -> SSA Conversion

`array.write` and `struct.writem` are void operations (in-place mutation). In
SSA form, each write produces a new value, and subsequent operations use the
latest version:

```mlir
// Before: same %arr mutated
array.write %arr[0] = %a    // void
array.write %arr[1] = %b    // void

// After: SSA chain
%arr1 = array.write %arr[0] = %a    // returns new array
%arr2 = array.write %arr1[1] = %b   // uses arr1, returns arr2
```

`array.write` SSA conversion happens during Step 1 (carry promotion requires
return values). `struct.writem` SSA conversion happens in a separate pass for
function body top-level writes.

### Step 3: scf.while -> stablehlo.while (post-pass)

After the main pattern conversion transforms body operations to StableHLO, the
while loop shell is converted:

```mlir
// Before (body is StableHLO, shell is still scf)
scf.while (%i = %c0, %arr = %init) : (tensor<!pf>, tensor<33x!pf>) {
  %cond = stablehlo.compare LT, %i, %c33
  scf.condition(%cond) %i, %arr
} do {
  stablehlo.return %next_i, %arr_new
}

// After
stablehlo.while (%i = %c0, %arr = %init) : (tensor<!pf>, tensor<33x!pf>) {
  %cond = stablehlo.compare LT, %i, %c33
  stablehlo.return %cond              // predicate only
} do {
  stablehlo.return %next_i, %arr_new  // carry
}
```

Nested while loops are converted innermost-first to avoid invalidating region
pointers.

______________________________________________________________________

## Post-passes

After the main conversion pass (`applyPartialConversion`), several post-passes
handle structural transformations and cleanup:

| #   | Post-pass                    | What it does                                                  |
| --- | ---------------------------- | ------------------------------------------------------------- |
| 1   | scf.while -> stablehlo.while | Convert loop shell (body already converted)                   |
| 2   | scf.if -> stablehlo.select   | Inline both branches, select by predicate                     |
| 3   | func.call reconnection       | Remove unrealized_conversion_cast chains from pod.read @comp  |
| 4   | Residual LLZK cleanup        | Replace remaining pod/array ops with zero tensors (dead code) |
| 5   | arith -> stablehlo           | Convert arith.ori/andi to stablehlo.or/and (tensor context)   |
| 6   | Dead code elimination        | Remove unused ops (fixed-point iteration)                     |
| 7   | While loop vectorization     | Convert iteration-independent loops to vector ops             |
| 8   | Final DCE                    | Clean up after vectorization                                  |

### scf.if -> stablehlo.select

StableHLO has no `if` construct. Both branches are inlined and a `select`
chooses the result:

```mlir
// Before
%r = scf.if %cond -> tensor<!pf> {
  %then = stablehlo.add %a, %b
  scf.yield %then
} else {
  %else = stablehlo.multiply %a, %b
  scf.yield %else
}

// After
%then = stablehlo.add %a, %b
%else = stablehlo.multiply %a, %b
%r = stablehlo.select %cond, %then, %else
```

### While Loop Vectorization

Detects iteration-independent while loops and converts them to vector
operations:

```mlir
// Before: sequential
stablehlo.while(i < N) { out[i] = f(in[i]); i++ }

// After: vectorized
out = f(in)   // N elements processed in parallel
```

Three patterns are supported:

1. **1D element-wise**: `out[i] = f(a[i], b[i])` -> `out = f(a, b)`
1. **2D column write**: `acc[i, col] = f(a[i], b[i])` -> column-wise vector ops
1. **Nested 2D**: outer while(j) { inner while(i) { ... } } -> vector ops

______________________________________________________________________

## Building and Testing

### Build

```bash
bazel build //tools:llzk-to-shlo-opt
```

### Tests

```bash
# All CI tests (LIT + smoke, ~10s)
bazel test //...

# LIT tests only (IR pattern FileCheck)
bazel test //tests:lit_tests

# Batch smoke tests (6 inline circuits, ~0.1s)
bazel test //tests:batch_smoke_tests

# Full E2E regression (requires circom + circom-benchmarks, ~15min)
bazel test //tests:batch_e2e_tests --test_tag_filters=manual

# GPU witness correctness (requires GPU + stablehlo_runner)
bazel test //tests:witness_correctness_tests
```

### E2E Example (Bazel)

```bash
# Build a specific example (Circom -> LLZK -> StableHLO)
bazel build //examples:num2bits

# View generated StableHLO
cat bazel-bin/examples/num2bits.stablehlo.mlir

# View intermediate LLZK IR
bazel build //examples:num2bits_llzk
cat bazel-bin/examples/num2bits.llzk
```

Available example targets: `num2bits`, `fulladder`, `decoder`, `binsum`.

______________________________________________________________________

## Key Files Reference

**Conversion passes**:

- `llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkToStablehlo.cpp` -- Main lowering
  pass (pre-passes, main conversion, post-passes)
- `llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.cpp` -- Pod
  dispatch elimination (6-phase fixed-point)
- `llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.cpp` -- Type converter
  (felt/struct/array/i1 -> tensor)
- `llzk_to_shlo/Conversion/LlzkToStablehlo/FeltPatterns.cpp` -- Arithmetic,
  bitwise, integer division patterns
- `llzk_to_shlo/Conversion/LlzkToStablehlo/StructPatterns.cpp` -- Struct access
  patterns (offset-based slicing)
- `llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.cpp` -- Array access
  patterns (dynamic_slice/update_slice)
- `llzk_to_shlo/Conversion/LlzkToStablehlo/RemovalPatterns.cpp` -- Deleted ops
  (constrain, assert, nondet, bool.not)
- `llzk_to_shlo/Conversion/BatchStablehlo/BatchStablehlo.cpp` -- Batch pass
  (leading dimension insertion)

**Tools**:

- `llzk_to_shlo/tools/llzk-to-shlo-opt.cpp` -- CLI tool

**Tests**:

- `tests/Conversion/LlzkToStablehlo/` -- Lowering LIT tests
- `tests/Conversion/BatchStablehlo/` -- Batch LIT tests
- `tests/batch_smoke_test.sh` -- CI-friendly batch smoke tests
- `tests/batch_e2e_test.sh` -- Full circom-benchmarks regression
