// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Verify batch dimension is correctly prepended to function signatures,
// constant broadcasting, and dynamic_slice index adjustment.

// Function signature: tensor<3xi32> → tensor<4x3xi32>
// CHECK-LABEL: func.func @batch_add
// CHECK-SAME: (%arg0: tensor<4x3xi32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xi32>
// CHECK: stablehlo.add %arg0, %arg1 : tensor<4x3xi32>

func.func @batch_add(%a: tensor<3xi32>, %b: tensor<3xi32>) -> tensor<3xi32> {
  %0 = stablehlo.add %a, %b : tensor<3xi32>
  return %0 : tensor<3xi32>
}

// Constant: broadcast_in_dim adds batch dimension
// CHECK-LABEL: func.func @batch_const
// CHECK-SAME: () -> tensor<4x3xi32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<42> : tensor<3xi32>
// CHECK: stablehlo.broadcast_in_dim %[[CST]]
// CHECK-SAME: dims = [1]
// CHECK-SAME: -> tensor<4x3xi32>

func.func @batch_const() -> tensor<3xi32> {
  %c = stablehlo.constant dense<42> : tensor<3xi32>
  return %c : tensor<3xi32>
}

// Scalar constant: broadcast_in_dim with dims = []
// CHECK-LABEL: func.func @batch_scalar_const
// CHECK-SAME: () -> tensor<4xi32>
// CHECK: %[[CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
// CHECK: stablehlo.broadcast_in_dim %[[CST]]
// CHECK-SAME: dims = []
// CHECK-SAME: -> tensor<4xi32>

func.func @batch_scalar_const() -> tensor<i32> {
  %c = stablehlo.constant dense<7> : tensor<i32>
  return %c : tensor<i32>
}
