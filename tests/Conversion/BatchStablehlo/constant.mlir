// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test stablehlo.constant batching: constant → broadcast_in_dim to add batch
// dimension.

// CHECK-LABEL: func.func @const_scalar
// CHECK: %[[C:.*]] = stablehlo.constant dense<42> : tensor<i32>
// CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim %[[C]]
// CHECK-SAME: dims = []
// CHECK-SAME: : (tensor<i32>) -> tensor<4xi32>
// CHECK: stablehlo.add %arg0, %[[B]] : tensor<4xi32>
func.func @const_scalar(%a: tensor<i32>) -> tensor<i32> {
  %c = stablehlo.constant dense<42> : tensor<i32>
  %0 = stablehlo.add %a, %c : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @const_1d
// CHECK: %[[C:.*]] = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
// CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim %[[C]]
// CHECK-SAME: dims = [1]
// CHECK-SAME: : (tensor<3xi32>) -> tensor<4x3xi32>
// CHECK: stablehlo.add %arg0, %[[B]] : tensor<4x3xi32>
func.func @const_1d(%a: tensor<3xi32>) -> tensor<3xi32> {
  %c = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  %0 = stablehlo.add %a, %c : tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: func.func @const_unused
// CHECK: stablehlo.constant dense<7>
// CHECK: stablehlo.broadcast_in_dim
func.func @const_unused() -> tensor<i32> {
  %c = stablehlo.constant dense<7> : tensor<i32>
  return %c : tensor<i32>
}
