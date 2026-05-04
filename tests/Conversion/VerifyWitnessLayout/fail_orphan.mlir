// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` declares an output @ [0,16); the DUS chain emits
// `dense<0>` for that slot — the AES-class silent-fallback signature.
// The verify pass must fail loudly, naming the offending signal and
// its kind so the orphaned upstream wire can be located.

// CHECK: error: 'wla.layout' op signal `@out` (kind=output, offset=0, length=16) is sourced by a splat-zero constant
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 16>
  ]

  func.func @main(%arg0: tensor<i32>) -> tensor<16xi32> {
    %base = stablehlo.constant dense<0> : tensor<16xi32>
    // Orphan: a fresh splat-zero where a real wire should have been.
    %orphan = stablehlo.constant dense<0> : tensor<16xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %orphan, %c0 : (tensor<16xi32>, tensor<16xi32>, tensor<i32>) -> tensor<16xi32>
    return %u0 : tensor<16xi32>
  }
}
