// Synthetic StableHLO module for witness_layout_audit smoke test.
// All chunks are non-zero — represents a "clean" lowered output.
//
// Layout:
//   chunk 0  offset=[0]  length=1   stablehlo.add        (non-zero)
//   chunk 1  offset=[1]  length=3   stablehlo.multiply   (non-zero)
//   chunk 2  offset=[4]  length=4   stablehlo.subtract   (non-zero)
module {
  func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<4xi32>) -> tensor<8xi32> {
    %zero8 = stablehlo.constant dense<0> : tensor<8xi32>

    %scalar0 = stablehlo.constant dense<7> : tensor<i32>
    %scalar1 = stablehlo.constant dense<11> : tensor<i32>
    %s0 = stablehlo.add %scalar0, %scalar1 : tensor<i32>
    %s0r = stablehlo.reshape %s0 : (tensor<i32>) -> tensor<1xi32>

    %three3 = stablehlo.constant dense<3> : tensor<3xi32>
    %v1 = stablehlo.multiply %arg0, %three3 : tensor<3xi32>

    %onez = stablehlo.constant dense<1> : tensor<4xi32>
    %v2 = stablehlo.subtract %arg1, %onez : tensor<4xi32>

    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>

    %u0 = stablehlo.dynamic_update_slice %zero8, %s0r, %c0 : (tensor<8xi32>, tensor<1xi32>, tensor<i32>) -> tensor<8xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %v1, %c1 : (tensor<8xi32>, tensor<3xi32>, tensor<i32>) -> tensor<8xi32>
    %u2 = stablehlo.dynamic_update_slice %u1, %v2, %c4 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    return %u2 : tensor<8xi32>
  }
}
