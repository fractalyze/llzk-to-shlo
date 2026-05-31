// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// The positive side of the input source-from-funcparam invariant: an `input`
// signal whose covering chunk copies a function parameter in through a
// `stablehlo.reshape` (the canonical anchored form). The check looks through
// reshapes to the block argument, so this is a valid funcparam source and
// verify OKs and passes `@main` through unchanged.

// CHECK-LABEL: wla.layout
// CHECK-LABEL: func.func @main
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 4>,
    #wla.signal<"%arg0", input, offset = 4, length = 4>
  ]

  func.func @main(%arg0: tensor<2x2xi32>) -> tensor<8xi32> {
    %base = stablehlo.constant dense<0> : tensor<8xi32>
    %three = stablehlo.constant dense<3> : tensor<4xi32>
    %out = stablehlo.add %three, %three : tensor<4xi32>
    %arg_flat = stablehlo.reshape %arg0 : (tensor<2x2xi32>) -> tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %arg_flat, %c4 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    return %u1 : tensor<8xi32>
  }
}
