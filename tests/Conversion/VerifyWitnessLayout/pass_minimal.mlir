// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// `wla.layout` declares one output @ [0,1) and one internal @ [1,16);
// the DUS chain in @main fills both slots with non-zero data.
// The verify pass should silent-OK and pass `@main` through unchanged.

// CHECK-LABEL: wla.layout
// CHECK-LABEL: func.func @main
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 1>,
    #wla.signal<"@internal_xor", internal, offset = 1, length = 16>
  ]

  func.func @main(%arg0: tensor<i32>, %arg1: tensor<16xi32>) -> tensor<17xi32> {
    %base = stablehlo.constant dense<0> : tensor<17xi32>

    %seven = stablehlo.constant dense<7> : tensor<i32>
    %out_val = stablehlo.add %arg0, %seven : tensor<i32>
    %out_r = stablehlo.reshape %out_val : (tensor<i32>) -> tensor<1xi32>

    %ones = stablehlo.constant dense<1> : tensor<16xi32>
    %xor_val = stablehlo.add %arg1, %ones : tensor<16xi32>

    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>

    %u0 = stablehlo.dynamic_update_slice %base, %out_r, %c0 : (tensor<17xi32>, tensor<1xi32>, tensor<i32>) -> tensor<17xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %xor_val, %c1 : (tensor<17xi32>, tensor<16xi32>, tensor<i32>) -> tensor<17xi32>
    return %u1 : tensor<17xi32>
  }
}
