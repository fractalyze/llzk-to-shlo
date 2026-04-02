// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test dynamic_slice and dynamic_update_slice batching: batch dim index 0
// prepended to start indices, slice_sizes gets leading N.

// CHECK-LABEL: func.func @dynamic_slice
// CHECK-SAME: (%arg0: tensor<4x8xi32>) -> tensor<4x1xi32>
// CHECK-DAG: %[[IDX:.*]] = stablehlo.constant dense<3>
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0>
// CHECK: stablehlo.dynamic_slice %arg0, %[[ZERO]], %[[IDX]]
// CHECK-SAME: sizes = [4, 1]
// CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>, tensor<i32>) -> tensor<4x1xi32>
func.func @dynamic_slice(%t: tensor<8xi32>) -> tensor<1xi32> {
  %idx = stablehlo.constant dense<3> : tensor<i32>
  %0 = stablehlo.dynamic_slice %t, %idx, sizes = [1]
      : (tensor<8xi32>, tensor<i32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @dynamic_update_slice
// CHECK-SAME: (%arg0: tensor<4x8xi32>, %arg1: tensor<4x1xi32>) -> tensor<4x8xi32>
// CHECK-DAG: %[[IDX:.*]] = stablehlo.constant dense<5>
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0>
// CHECK: stablehlo.dynamic_update_slice %arg0, %arg1, %[[ZERO]], %[[IDX]]
// CHECK-SAME: : (tensor<4x8xi32>, tensor<4x1xi32>, tensor<i32>, tensor<i32>) -> tensor<4x8xi32>
func.func @dynamic_update_slice(%t: tensor<8xi32>, %u: tensor<1xi32>) -> tensor<8xi32> {
  %idx = stablehlo.constant dense<5> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %t, %u, %idx
      : (tensor<8xi32>, tensor<1xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
