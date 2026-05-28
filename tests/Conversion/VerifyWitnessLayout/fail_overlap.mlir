// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` is sorted by offset but @a [0,8) and @b [4,8) overlap:
// `a.offset + a.length` (8) exceeds the next entry's offset (4). Two
// signals claiming the same witness slot is a layout bug; the verify
// pass must reject it.

// CHECK: error: 'wla.layout' op signals overlap
module {
  wla.layout signals = [
    #wla.signal<"@a", output, offset = 0, length = 8>,
    #wla.signal<"@b", output, offset = 4, length = 4>
  ]

  func.func @main(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %base = stablehlo.constant dense<0> : tensor<8xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %arg0, %c0 : (tensor<8xi32>, tensor<8xi32>, tensor<i32>) -> tensor<8xi32>
    return %u0 : tensor<8xi32>
  }
}
