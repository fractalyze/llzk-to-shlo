// RUN: llzk-to-shlo-opt --llzk-to-stablehlo %s | FileCheck %s

// Full conversion test: Verify LLZK felt operations are converted to StableHLO
// Note: Function signatures remain as !felt.type since func.func is marked legal.
// Felt operations inside the function are converted with unrealized_conversion_cast.

// Verify felt operations are converted
// CHECK-NOT: felt.add
// CHECK-NOT: felt.mul
// CHECK-NOT: felt.const

// CHECK-LABEL: func.func @compute_add
// CHECK: stablehlo.add
// CHECK: return
func.func @compute_add(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %sum = felt.add %a, %b : !felt.type
  return %sum : !felt.type
}

// CHECK-LABEL: func.func @compute_mul
// CHECK: stablehlo.multiply
func.func @compute_mul(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %prod = felt.mul %a, %b : !felt.type
  return %prod : !felt.type
}

// CHECK-LABEL: func.func @compute_const
// CHECK: stablehlo.constant
// CHECK-SAME: dense<42>
func.func @compute_const() -> !felt.type {
  %c = felt.const 42
  return %c : !felt.type
}

// CHECK-LABEL: func.func @compute_combined
// CHECK: stablehlo.add
// CHECK: stablehlo.multiply
// CHECK: stablehlo.subtract
func.func @compute_combined(%a: !felt.type, %b: !felt.type, %c: !felt.type) -> !felt.type {
  %sum = felt.add %a, %b : !felt.type
  %prod = felt.mul %sum, %c : !felt.type
  %diff = felt.sub %prod, %a : !felt.type
  return %diff : !felt.type
}
