// RUN: llzk-to-shlo-opt %s | FileCheck %s
// RUN: llzk-to-shlo-opt %s | llzk-to-shlo-opt | FileCheck %s

// Parse/print/parse stability for `wla.layout`, `#wla.signal`, kind enum.

// CHECK-LABEL: module
module {
  // CHECK: wla.layout signals = [
  // CHECK-SAME: #wla.signal<"const_one", internal, offset = 0, length = 1>
  // CHECK-SAME: #wla.signal<"@out", output, offset = 1, length = 128>
  // CHECK-SAME: #wla.signal<"%arg0", input, offset = 129, length = 128>
  // CHECK-SAME: #wla.signal<"%arg1", input, offset = 257, length = 1920>
  // CHECK-SAME: #wla.signal<"@xor_1", internal, offset = 2177, length = 128>
  // CHECK-SAME: ]
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out", output, offset = 1, length = 128>,
    #wla.signal<"%arg0", input, offset = 129, length = 128>,
    #wla.signal<"%arg1", input, offset = 257, length = 1920>,
    #wla.signal<"@xor_1", internal, offset = 2177, length = 128>
  ]
}
