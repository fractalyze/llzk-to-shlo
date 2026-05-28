// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// Canonical layout the anchor pass actually emits: a `const_one`
// internal head at offset 0, then output, input, internal in block
// order. The const_one head is `internal`-kind but precedes the
// outputs by design, so the block-order check must skip it rather than
// reject the internal-before-output sequence.

// CHECK-LABEL: wla.layout
// CHECK-LABEL: func.func @main
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out", output, offset = 1, length = 4>,
    #wla.signal<"%arg0", input, offset = 5, length = 4>,
    #wla.signal<"@xor", internal, offset = 9, length = 4>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<13xi32> {
    %base = stablehlo.constant dense<0> : tensor<13xi32>
    %one = stablehlo.constant dense<1> : tensor<1xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %c9 = stablehlo.constant dense<9> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %one, %c0 : (tensor<13xi32>, tensor<1xi32>, tensor<i32>) -> tensor<13xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %arg0, %c1 : (tensor<13xi32>, tensor<4xi32>, tensor<i32>) -> tensor<13xi32>
    %u2 = stablehlo.dynamic_update_slice %u1, %arg0, %c5 : (tensor<13xi32>, tensor<4xi32>, tensor<i32>) -> tensor<13xi32>
    %u3 = stablehlo.dynamic_update_slice %u2, %arg0, %c9 : (tensor<13xi32>, tensor<4xi32>, tensor<i32>) -> tensor<13xi32>
    return %u3 : tensor<13xi32>
  }
}
