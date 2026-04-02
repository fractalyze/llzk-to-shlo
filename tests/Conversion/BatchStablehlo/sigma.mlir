// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=1000" %s | FileCheck %s

// Test Sigma circuit (x⁵) after llzk-to-stablehlo conversion.
// Verifies that the batch pass correctly transforms a real circuit output.

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<1000xi32>) -> tensor<1000x3xi32>
//
// Constants become broadcast_in_dim:
// CHECK: stablehlo.constant dense<0> : tensor<3xi32>
// CHECK: stablehlo.broadcast_in_dim
// CHECK-SAME: -> tensor<1000x3xi32>
//
// x² = x * x
// CHECK: stablehlo.multiply %arg0, %arg0 : tensor<1000xi32>
//
// dynamic_update_slice gets batch dim 0 index:
// CHECK: stablehlo.dynamic_update_slice
// CHECK-SAME: tensor<1000x3xi32>
//
// x⁴ = x² * x²
// CHECK: stablehlo.multiply
//
// x⁵ = x⁴ * x
// CHECK: stablehlo.multiply
//
// Final return is batched:
// CHECK: return %{{.*}} : tensor<1000x3xi32>

func.func @main(%arg0: tensor<i32>) -> tensor<3xi32> {
  // Struct init (zero tensor)
  %init = stablehlo.constant dense<0> : tensor<3xi32>

  // x² = x * x
  %x2 = stablehlo.multiply %arg0, %arg0 : tensor<i32>

  // Write x² to struct[1] (in2)
  %x2_r = stablehlo.reshape %x2 : (tensor<i32>) -> tensor<1xi32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %s1 = stablehlo.dynamic_update_slice %init, %x2_r, %c1
      : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>

  // x⁴ = x² * x²
  %x4 = stablehlo.multiply %x2, %x2 : tensor<i32>

  // Write x⁴ to struct[2] (in4)
  %x4_r = stablehlo.reshape %x4 : (tensor<i32>) -> tensor<1xi32>
  %c2 = stablehlo.constant dense<2> : tensor<i32>
  %s2 = stablehlo.dynamic_update_slice %s1, %x4_r, %c2
      : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>

  // x⁵ = x⁴ * x
  %x5 = stablehlo.multiply %x4, %arg0 : tensor<i32>

  // Write x⁵ to struct[0] (out)
  %x5_r = stablehlo.reshape %x5 : (tensor<i32>) -> tensor<1xi32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %s3 = stablehlo.dynamic_update_slice %s2, %x5_r, %c0
      : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>

  return %s3 : tensor<3xi32>
}
