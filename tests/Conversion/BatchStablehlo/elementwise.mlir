// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Test element-wise ops: result types get leading batch dim, no structural
// change needed.

// CHECK-LABEL: func.func @add_scalar
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.add %arg0, %arg1 : tensor<4xi32>
func.func @add_scalar(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.add %a, %b : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @add_1d
// CHECK-SAME: (%arg0: tensor<4x3xi32>, %arg1: tensor<4x3xi32>) -> tensor<4x3xi32>
// CHECK: stablehlo.add %arg0, %arg1 : tensor<4x3xi32>
func.func @add_1d(%a: tensor<3xi32>, %b: tensor<3xi32>) -> tensor<3xi32> {
  %0 = stablehlo.add %a, %b : tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: func.func @mul_scalar
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.multiply %arg0, %arg1 : tensor<4xi32>
func.func @mul_scalar(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.multiply %a, %b : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @negate
// CHECK-SAME: (%arg0: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.negate %arg0 : tensor<4xi32>
func.func @negate(%a: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.negate %a : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @subtract
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.subtract %arg0, %arg1 : tensor<4xi32>
func.func @subtract(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.subtract %a, %b : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @divide
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.divide %arg0, %arg1 : tensor<4xi32>
func.func @divide(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.divide %a, %b : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @power
// CHECK-SAME: (%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32>
// CHECK: stablehlo.power %arg0, %arg1 : tensor<4xi32>
func.func @power(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.power %a, %b : tensor<i32>
  return %0 : tensor<i32>
}
