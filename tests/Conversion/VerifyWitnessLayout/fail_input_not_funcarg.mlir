// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` declares input `%arg0` with length 8, but @main's block
// argument 0 is `tensor<4xi32>` (flat size 4). Inputs are realized as @main
// block arguments (not witness chunks), so verify checks that `%argN` maps to
// arg N with a matching flat element count; a regressed/external layout that
// mis-sizes the input must be rejected. The `@out` output is computed from the
// input and lands at compacted offset 0 — fine; the input check is what fires.

// CHECK: error: 'wla.layout' op input signal `%arg0` length 8 does not match @main arg 0 flat size 4
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 4>,
    #wla.signal<"%arg0", input, offset = 4, length = 8>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %base = stablehlo.constant dense<0> : tensor<4xi32>
    %out = stablehlo.add %arg0, %arg0 : tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
    return %u0 : tensor<4xi32>
  }
}
