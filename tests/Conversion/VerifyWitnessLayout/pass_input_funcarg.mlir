// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// Positive side of the input↔function-parameter invariant: an `input` signal
// `%arg0` (length 4) maps to @main's block argument 0, here a `tensor<2x2xi32>`
// whose flat element count (2*2) matches the signal length. Inputs are block
// args, not witness chunks, so verify checks the correspondence (flattening the
// arg type) rather than a covering chunk, then consumes (erases) the layout.

// CHECK-NOT: wla.layout
// CHECK-LABEL: func.func @main
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 4>,
    #wla.signal<"%arg0", input, offset = 4, length = 4>
  ]

  func.func @main(%arg0: tensor<2x2xi32>) -> tensor<4xi32> {
    %base = stablehlo.constant dense<0> : tensor<4xi32>
    %out = stablehlo.reshape %arg0 : (tensor<2x2xi32>) -> tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
    return %u0 : tensor<4xi32>
  }
}
