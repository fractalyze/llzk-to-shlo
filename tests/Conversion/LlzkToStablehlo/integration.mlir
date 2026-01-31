// RUN: llzk-to-shlo-opt --llzk-to-stablehlo %s | FileCheck %s

// Integration test: Simple compute function conversion using felt operations

// Verify no LLZK operations remain in output
// CHECK-NOT: !felt.type
// CHECK-NOT: felt.add
// CHECK-NOT: felt.sub
// CHECK-NOT: felt.mul
// CHECK-NOT: felt.div
// CHECK-NOT: felt.neg

// CHECK-LABEL: func.func @compute_sum_and_product
// CHECK: stablehlo.add
// CHECK: stablehlo.multiply
// CHECK: return
func.func @compute_sum_and_product(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %sum = felt.add %a, %b : !felt.type
  %result = felt.mul %sum, %a : !felt.type
  return %result : !felt.type
}

// CHECK-LABEL: func.func @compute_quadratic
// CHECK: stablehlo.multiply
// CHECK: stablehlo.multiply
// CHECK: stablehlo.add
func.func @compute_quadratic(%a: !felt.type, %x: !felt.type, %b: !felt.type) -> !felt.type {
  // Computes a * x² + b
  %x_squared = felt.mul %x, %x : !felt.type
  %ax2 = felt.mul %a, %x_squared : !felt.type
  %result = felt.add %ax2, %b : !felt.type
  return %result : !felt.type
}

// CHECK-LABEL: func.func @compute_with_div
// CHECK: stablehlo.add
// CHECK: stablehlo.divide
func.func @compute_with_div(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %sum = felt.add %a, %b : !felt.type
  %result = felt.div %sum, %a : !felt.type
  return %result : !felt.type
}

// CHECK-LABEL: func.func @compute_with_neg
// CHECK: stablehlo.negate
func.func @compute_with_neg(%a: !felt.type) -> !felt.type {
  %neg = felt.neg %a : !felt.type
  return %neg : !felt.type
}
