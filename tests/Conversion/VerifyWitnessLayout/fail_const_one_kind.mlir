// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// The reserved constant-1 wire is an `internal` signal that heads the
// layout. Declaring `const_one` as `output` mis-classifies the reserved
// wire; the block-order exemption keys on name+kind, so without this check
// a mislabeled head would be silently accepted.

// CHECK: error: 'wla.layout' op `const_one` head must be `internal`-kind
module {
  wla.layout signals = [
    #wla.signal<"const_one", output, offset = 0, length = 1>,
    #wla.signal<"@out", output, offset = 1, length = 4>
  ]

  func.func @main() -> tensor<5xi32> {
    %base = stablehlo.constant dense<0> : tensor<5xi32>
    return %base : tensor<5xi32>
  }
}
