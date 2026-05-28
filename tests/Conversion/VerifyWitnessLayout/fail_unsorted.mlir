// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` lists @b (offset 8) before @a (offset 0): the signals
// array is not sorted ascending by offset. The spec is malformed
// independent of @main, so the verify pass must reject it before the
// per-chunk match.

// CHECK: error: 'wla.layout' op signals must be sorted by ascending offset
module {
  wla.layout signals = [
    #wla.signal<"@b", output, offset = 8, length = 4>,
    #wla.signal<"@a", output, offset = 0, length = 4>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<12xi32> {
    %base = stablehlo.constant dense<0> : tensor<12xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %arg0, %c0 : (tensor<12xi32>, tensor<4xi32>, tensor<i32>) -> tensor<12xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %arg0, %c8 : (tensor<12xi32>, tensor<4xi32>, tensor<i32>) -> tensor<12xi32>
    return %u1 : tensor<12xi32>
  }
}
