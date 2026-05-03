// Synthetic StableHLO module for witness_layout_audit smoke test.
// chunk 1 is a length-16 splat-zero — the AES-class silent-fallback shape
// we want the audit tool to flag (the audit reports every splat-zero
// regardless of length; PR #67's loud-failure assertion separately uses a
// length>=8 heuristic to skip intentional small zero-inits).
//
// Layout:
//   chunk 0  offset=[0]   length=1   stablehlo.add        (non-zero)
//   chunk 1  offset=[1]   length=16  stablehlo.constant   (splat zero)
//   chunk 2  offset=[17]  length=4   stablehlo.subtract   (non-zero)
module {
  func.func @main(%arg0: tensor<4xi32>) -> tensor<21xi32> {
    %zero21 = stablehlo.constant dense<0> : tensor<21xi32>

    %scalar0 = stablehlo.constant dense<5> : tensor<i32>
    %scalar1 = stablehlo.constant dense<9> : tensor<i32>
    %s0 = stablehlo.add %scalar0, %scalar1 : tensor<i32>
    %s0r = stablehlo.reshape %s0 : (tensor<i32>) -> tensor<1xi32>

    // The orphan: a fresh splat-zero where a real wire should have been.
    // length 16 (>= 8) — length<8 is treated as intentional zero-init.
    %orphan = stablehlo.constant dense<0> : tensor<16xi32>

    %onez = stablehlo.constant dense<1> : tensor<4xi32>
    %v2 = stablehlo.subtract %arg0, %onez : tensor<4xi32>

    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c17 = stablehlo.constant dense<17> : tensor<i32>

    %u0 = stablehlo.dynamic_update_slice %zero21, %s0r, %c0 : (tensor<21xi32>, tensor<1xi32>, tensor<i32>) -> tensor<21xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %orphan, %c1 : (tensor<21xi32>, tensor<16xi32>, tensor<i32>) -> tensor<21xi32>
    %u2 = stablehlo.dynamic_update_slice %u1, %v2, %c17 : (tensor<21xi32>, tensor<4xi32>, tensor<i32>) -> tensor<21xi32>
    return %u2 : tensor<21xi32>
  }
}
