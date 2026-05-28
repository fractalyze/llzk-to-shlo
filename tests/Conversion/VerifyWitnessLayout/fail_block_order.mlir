// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` is sorted and non-overlapping, but the input signal
// precedes the output signal. Circom's flat witness is laid out
// `const, outputs, inputs, internals`; an input before an output means
// the anchor pass mis-classified a signal, so the verify pass must
// reject the block ordering.

// CHECK: error: 'wla.layout' op signal `@out` (kind=output) breaks canonical block order
module {
  wla.layout signals = [
    #wla.signal<"%arg0", input, offset = 0, length = 4>,
    #wla.signal<"@out", output, offset = 4, length = 4>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<8xi32> {
    %base = stablehlo.constant dense<0> : tensor<8xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %arg0, %c0 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %arg0, %c4 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    return %u1 : tensor<8xi32>
  }
}
