// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` declares an input @ [4,8). Its covering `dynamic_update_slice`
// chunk in @main fills that slot from a `stablehlo.constant`, not from @main's
// block argument. An `input`-kind signal must source from a function parameter
// (the WLA contract's input source-from-funcparam invariant); verify must
// reject this regressed/external layout rather than silent-OK. The `@out`
// chunk below is compute-sourced (`stablehlo.add`) on purpose — the check is
// scoped to `input`-kind, so a computed output/internal chunk is fine.

// CHECK: error: 'wla.layout' op input signal `%arg0` (offset=4, length=4) does not source from a function parameter; its covering chunk's value is `stablehlo.constant`
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 4>,
    #wla.signal<"%arg0", input, offset = 4, length = 4>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<8xi32> {
    %base = stablehlo.constant dense<0> : tensor<8xi32>
    %out = stablehlo.add %arg0, %arg0 : tensor<4xi32>
    %bogus = stablehlo.constant dense<7> : tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %bogus, %c4 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    return %u1 : tensor<8xi32>
  }
}
