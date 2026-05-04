// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// No `wla.layout` op in the module — the verify pass silent-no-ops.
// This is the steady-state path while `--witness-layout-anchor`
// (TRACK 3) is being incrementally rolled out per chip.

// CHECK-NOT: wla.layout
// CHECK-LABEL: func.func @main
module {
  func.func @main(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %ones = stablehlo.constant dense<1> : tensor<8xi32>
    %r = stablehlo.add %arg0, %ones : tensor<8xi32>
    return %r : tensor<8xi32>
  }
}
