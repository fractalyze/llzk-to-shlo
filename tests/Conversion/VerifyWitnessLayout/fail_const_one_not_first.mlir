// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// The `const_one` head must lead the layout (circom reserves slot 0). Here
// an output signal precedes it, so the reserved wire is misplaced.

// CHECK: error: 'wla.layout' op `const_one` must be the first layout entry
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 4>,
    #wla.signal<"const_one", internal, offset = 4, length = 1>
  ]

  func.func @main() -> tensor<5xi32> {
    %base = stablehlo.constant dense<0> : tensor<5xi32>
    return %base : tensor<5xi32>
  }
}
