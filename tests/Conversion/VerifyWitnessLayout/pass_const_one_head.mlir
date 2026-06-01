// RUN: llzk-to-shlo-opt --verify-witness-layout %s -o - | FileCheck %s

// Canonical layout the anchor pass emits: a `const_one` internal head at
// offset 0, then output, input, internal in circom's full-witness block order.
// In lowered @main the const_one wire is implicit (never written), the input is
// a block argument (read, never written), and only the output + internal are
// materialized — compacted to offsets 0 and 4. Verify skips const_one, checks
// `%arg0` against @main's block arg 0, matches @out/@xor at their compacted
// offsets, then consumes (erases) the layout.

// CHECK-NOT: wla.layout
// CHECK-LABEL: func.func @main
module {
  wla.layout signals = [
    #wla.signal<"const_one", internal, offset = 0, length = 1>,
    #wla.signal<"@out", output, offset = 1, length = 4>,
    #wla.signal<"%arg0", input, offset = 5, length = 4>,
    #wla.signal<"@xor", internal, offset = 9, length = 4>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<8xi32> {
    %base = stablehlo.constant dense<0> : tensor<8xi32>
    %seven = stablehlo.constant dense<7> : tensor<4xi32>
    %out = stablehlo.add %arg0, %seven : tensor<4xi32>
    %ones = stablehlo.constant dense<1> : tensor<4xi32>
    %xor = stablehlo.add %arg0, %ones : tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %xor, %c4 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    return %u1 : tensor<8xi32>
  }
}
