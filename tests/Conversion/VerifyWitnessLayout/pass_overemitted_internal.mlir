// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// The anchor over-emits `internal` signals that the lowering legitimately
// elides from @main's DUS chain (dead struct sub-component members). Here
// `@dummy` is a layout internal with no covering chunk in @main. Verify must
// tolerate it (internals are deferred to the m3 byte-equality gate), checking
// only the output (`@out` at compacted offset 0) and the input (`%arg0` ↔ block
// arg 0), then consuming (erasing) the layout. Mirrors the layout that
// `examples/lessthan_bounded` emits.

// CHECK-NOT: wla.layout
// CHECK-LABEL: func.func @main
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out", output, offset = 1, length = 1>,
    #wla.signal<"%arg0", input, offset = 2, length = 2>,
    #wla.signal<"@dummy", internal, offset = 4, length = 1>
  ]

  func.func @main(%arg0: tensor<2xi32>) -> tensor<1xi32> {
    %base = stablehlo.constant dense<0> : tensor<1xi32>
    %out = stablehlo.constant dense<5> : tensor<1xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
    return %u0 : tensor<1xi32>
  }
}
