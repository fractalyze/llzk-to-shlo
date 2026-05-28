// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// The `const_one` head is the reserved constant-1 wire: circom pins it to
// slot 0. Here it sits at offset 5, so the layout is malformed. This is a
// pure-layout property, so the verify pass must reject it before examining
// @main.

// CHECK: error: 'wla.layout' op `const_one` head must be at offset 0
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 5, length = 1>,
    #wla.signal<"@out", output, offset = 6, length = 4>
  ]

  func.func @main() -> tensor<10xi32> {
    %base = stablehlo.constant dense<0> : tensor<10xi32>
    return %base : tensor<10xi32>
  }
}
