// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// Exactly one reserved constant-1 wire exists per witness. A second signal
// named `const_one` is malformed regardless of where it sits.

// CHECK: error: 'wla.layout' op layout has more than one `const_one` signal
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out", output, offset = 1, length = 4>,
    #wla.signal<"const_one", internal, offset = 5, length = 1>
  ]

  func.func @main() -> tensor<6xi32> {
    %base = stablehlo.constant dense<0> : tensor<6xi32>
    return %base : tensor<6xi32>
  }
}
