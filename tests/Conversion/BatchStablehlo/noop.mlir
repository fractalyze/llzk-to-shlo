// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=1" %s | FileCheck %s

// Test that batch-size=1 is a no-op.

// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%arg0: tensor<i32>) -> tensor<i32>
// CHECK: stablehlo.add %arg0, %arg0 : tensor<i32>
func.func @identity(%a: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.add %a, %a : tensor<i32>
  return %0 : tensor<i32>
}
