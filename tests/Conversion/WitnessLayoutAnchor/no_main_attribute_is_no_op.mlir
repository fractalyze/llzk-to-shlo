// RUN: llzk-to-shlo-opt --witness-layout-anchor %s | FileCheck %s

// Modules without an `llzk.main` attribute (e.g. helper / fixture
// modules) are non-chip and the pass MUST silently no-op rather than
// emit an empty layout op.

module {
  // CHECK-LABEL: module
  // CHECK-NOT: wla.layout
  // CHECK-NOT: wla.signal
  func.func @helper(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0 : tensor<i32>
  }
}
