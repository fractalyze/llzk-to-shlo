// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` declares an output that, compacted into @main's witness, lands
// at offset 0 — but @main's DUS chain only writes [8,12), leaving the output
// slot uncovered (a dropped/regressed wire). The verify pass must report the
// missing coverage rather than silently passing. (Inputs are block args, not
// chunks, so the "no covering chunk" check is scoped to output/internal.)

// CHECK: error: 'wla.layout' op output signal `@out` (length=8) has no covering `dynamic_update_slice` chunk at @main witness offset 0
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 8>
  ]

  func.func @main() -> tensor<16xi32> {
    %base = stablehlo.constant dense<0> : tensor<16xi32>
    %v = stablehlo.constant dense<5> : tensor<4xi32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %v, %c8 : (tensor<16xi32>, tensor<4xi32>, tensor<i32>) -> tensor<16xi32>
    return %u0 : tensor<16xi32>
  }
}
