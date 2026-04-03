// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test data-dependent dynamic_slice batching via one-hot selection.
// When the index is computed from batched data (not a constant), the pass
// converts dynamic_slice to iota + compare + multiply + reduce_sum.

// CHECK-LABEL: func.func @lookup
// CHECK-SAME: (%arg0: tensor<4x8xi32>, %arg1: tensor<4xi32>) -> tensor<4x1xi32>
//
// One-hot selection pattern:
// CHECK: stablehlo.iota
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.broadcast_in_dim
// CHECK: stablehlo.compare EQ
// CHECK: stablehlo.convert
// CHECK: stablehlo.multiply
// CHECK: stablehlo.reduce
// CHECK: stablehlo.reshape
func.func @lookup(%table: tensor<8xi32>, %idx: tensor<i32>) -> tensor<1xi32> {
  %0 = stablehlo.dynamic_slice %table, %idx, sizes = [1]
      : (tensor<8xi32>, tensor<i32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
