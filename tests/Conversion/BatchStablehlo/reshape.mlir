// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test reshape batching: leading N added to target shape.

// CHECK-LABEL: func.func @reshape_1d_to_scalar
// CHECK-SAME: (%arg0: tensor<4x1xi32>) -> tensor<4xi32>
// CHECK: stablehlo.reshape %arg0 : (tensor<4x1xi32>) -> tensor<4xi32>
func.func @reshape_1d_to_scalar(%a: tensor<1xi32>) -> tensor<i32> {
  %0 = stablehlo.reshape %a : (tensor<1xi32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @reshape_2d
// CHECK-SAME: (%arg0: tensor<4x6xi32>) -> tensor<4x2x3xi32>
// CHECK: stablehlo.reshape %arg0 : (tensor<4x6xi32>) -> tensor<4x2x3xi32>
func.func @reshape_2d(%a: tensor<6xi32>) -> tensor<2x3xi32> {
  %0 = stablehlo.reshape %a : (tensor<6xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}
