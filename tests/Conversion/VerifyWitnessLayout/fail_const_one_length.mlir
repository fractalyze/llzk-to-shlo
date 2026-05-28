// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// The `const_one` head occupies a single prime-field element (length 1).
// Here it claims length 2, so the layout is malformed independent of @main.

// CHECK: error: 'wla.layout' op `const_one` head must have length 1
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 2>,
    #wla.signal<"@out", output, offset = 2, length = 4>
  ]

  func.func @main() -> tensor<6xi32> {
    %base = stablehlo.constant dense<0> : tensor<6xi32>
    return %base : tensor<6xi32>
  }
}
