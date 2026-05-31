// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// An `input` signal whose covering chunk is a splat-zero constant (an
// orphaned upstream wire). It is both splat-zero and not a function parameter,
// but the splat-zero "orphaned wire" diagnosis is the precise root cause, so
// verify reports only that — not the redundant source-from-funcparam error.

// CHECK: error: 'wla.layout' op signal `%arg0` (kind=input, offset=4, length=4) is sourced by a splat-zero constant
// CHECK-NOT: does not source from a function parameter
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 4>,
    #wla.signal<"%arg0", input, offset = 4, length = 4>
  ]

  func.func @main(%arg0: tensor<4xi32>) -> tensor<8xi32> {
    %base = stablehlo.constant dense<0> : tensor<8xi32>
    %out = stablehlo.add %arg0, %arg0 : tensor<4xi32>
    %orphan = stablehlo.constant dense<0> : tensor<4xi32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %u0 = stablehlo.dynamic_update_slice %base, %out, %c0 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    %u1 = stablehlo.dynamic_update_slice %u0, %orphan, %c4 : (tensor<8xi32>, tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
    return %u1 : tensor<8xi32>
  }
}
