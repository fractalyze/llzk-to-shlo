// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` declares an input @ [8,16) but no DUS chunk in the
// chain covers offset 8 (the chain only writes [0,8)). The verify
// pass must report the missing coverage rather than silently passing.

// CHECK: error: 'wla.layout' op signal `%arg0` (kind=input, offset=8, length=8) has no covering `dynamic_update_slice` chunk
module {
  wla.layout signals = [
    #wla.signal<"%arg0", input, offset = 8, length = 8>
  ]

  func.func @main(%arg0: tensor<8xi32>) -> tensor<16xi32> {
    %base = stablehlo.constant dense<0> : tensor<16xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %arg0, %c0 : (tensor<16xi32>, tensor<8xi32>, tensor<i32>) -> tensor<16xi32>
    return %u0 : tensor<16xi32>
  }
}
