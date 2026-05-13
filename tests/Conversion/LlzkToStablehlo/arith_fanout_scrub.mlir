// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: the LlzkToStablehlo target keeps `arith::ArithDialect` legal
// during partial conversion, so an `array.write %carrier[%dyn_idx] = %v`
// emitted with a felt-typed dynamic index inside a stablehlo.while body
// survives as a per-row select fan-out:
//
//   %idx = unrealized_conversion_cast %iter_i32 : tensor<i32> to index
//   %true = arith.constant true
//   %true_t = unrealized_conversion_cast %true : i1 to tensor<i1>
//   %cN = arith.constant N : index
//   %cmp = arith.cmpi eq, %idx, %cN : index
//   %cmp_t = unrealized_conversion_cast %cmp : i1 to tensor<i1>
//   %flag = arith.andi %cmp_t, %true_t : tensor<i1>
//   %sel = stablehlo.select %flag, %v, %prev
//
// The ZKX HLO translator rejects every surviving arith.* op, so the
// post-pass must scrub the chain into stablehlo.compare + stablehlo.and
// on tensor<i32> / tensor<i1>. Without the scrub, webb_poseidon_vanchor
// (and any chip with similar fan-out) fails m3 enrollment with
// "'arith.constant' op can't be translated to ZKX HLO".

// CHECK-LABEL: func.func @main
// CHECK-NOT: arith.constant
// CHECK-NOT: arith.cmpi
// CHECK-NOT: arith.andi
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: stablehlo.compare  EQ
// CHECK: stablehlo.and

module {
  func.func @main(%iter32: tensor<i32>, %v0: tensor<i32>, %v1: tensor<i32>, %v2: tensor<i32>, %seed: tensor<i32>) -> tensor<i32> {
    %idx = builtin.unrealized_conversion_cast %iter32 : tensor<i32> to index
    %true = arith.constant true
    %true_t = builtin.unrealized_conversion_cast %true : i1 to tensor<i1>
    %c0 = arith.constant 0 : index
    %cmp0 = arith.cmpi eq, %idx, %c0 : index
    %cmp0_t = builtin.unrealized_conversion_cast %cmp0 : i1 to tensor<i1>
    %flag0 = arith.andi %cmp0_t, %true_t : tensor<i1>
    %r0 = stablehlo.select %flag0, %v0, %seed : tensor<i1>, tensor<i32>
    %c1 = arith.constant 1 : index
    %cmp1 = arith.cmpi eq, %idx, %c1 : index
    %cmp1_t = builtin.unrealized_conversion_cast %cmp1 : i1 to tensor<i1>
    %flag1 = arith.andi %cmp1_t, %true_t : tensor<i1>
    %r1 = stablehlo.select %flag1, %v1, %r0 : tensor<i1>, tensor<i32>
    %c2 = arith.constant 2 : index
    %cmp2 = arith.cmpi eq, %idx, %c2 : index
    %cmp2_t = builtin.unrealized_conversion_cast %cmp2 : i1 to tensor<i1>
    %flag2 = arith.andi %cmp2_t, %true_t : tensor<i1>
    %r2 = stablehlo.select %flag2, %v2, %r1 : tensor<i1>, tensor<i32>
    return %r2 : tensor<i32>
  }
}
