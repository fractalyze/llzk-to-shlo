// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test compare and select batching.

// CHECK-LABEL: func.func @compare_and_select
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: %[[CMP:.*]] = stablehlo.compare LT, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
// CHECK: %[[SEL:.*]] = stablehlo.select %[[CMP]], %arg0, %arg1 : tensor<4xi1>, tensor<4xi32>
// CHECK: return %[[SEL]] : tensor<4xi32>
func.func @compare_and_select(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %cmp = stablehlo.compare LT, %a, %b : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %sel = stablehlo.select %cmp, %a, %b : tensor<i1>, tensor<i32>
  return %sel : tensor<i32>
}
