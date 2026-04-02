// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test that multiple functions and func.call are correctly batched.

// CHECK-LABEL: func.func @helper
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.multiply
func.func @helper(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.multiply %a, %b : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<4xi32>) -> tensor<4xi32>
// CHECK: %[[R:.*]] = call @helper(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.add %[[R]], %arg0 : tensor<4xi32>
func.func @main(%x: tensor<i32>) -> tensor<i32> {
  %sq = func.call @helper(%x, %x) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = stablehlo.add %sq, %x : tensor<i32>
  return %0 : tensor<i32>
}
