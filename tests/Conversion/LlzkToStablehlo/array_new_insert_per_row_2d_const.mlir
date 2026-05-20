// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: a 2D constant matrix initialized via `array.new <Empty>` plus
// per-row `array.insert` of literal sub-arrays (the LLZK shape emitted by
// circom's `Expression::ArrayInLine` for inline 2D literals — Poseidon MDS,
// AES SBox tables, etc.) must lower so subsequent reads see the populated
// matrix, NOT the empty `dense<0>` result of `array.new`.
//
// Without `array.insert` being threaded through SSA at function-body scope
// by `convertWritemToSSA`, the void `array.insert` ops produce orphan
// `stablehlo.dynamic_update_slice` ops that get DCE'd. Downstream
// `array.read` then sees the original empty matrix and the result is
// uniformly zero.
//
// Canonical victim: Webb `@Mix_69_compute` returning all-zero (MDS matrix
// stays at `dense<0>`).

// CHECK-LABEL: func.func @main
// `array.new` for the 2x3 matrix lowers to a constant zero tensor.
// CHECK: %[[EMPTY:.*]] = stablehlo.constant{{.*}}dense<0> : tensor<2x3xi32>
// Each row is built with the Values constructor; both lower to a chain of
// reshape + concatenate. Match the two row constants by their values.
// CHECK: stablehlo.constant{{.*}}dense<11>
// CHECK: stablehlo.constant{{.*}}dense<12>
// CHECK: stablehlo.constant{{.*}}dense<13>
// CHECK: %[[ROW0:.*]] = stablehlo.concatenate
// CHECK: stablehlo.constant{{.*}}dense<21>
// CHECK: stablehlo.constant{{.*}}dense<22>
// CHECK: stablehlo.constant{{.*}}dense<23>
// CHECK: %[[ROW1:.*]] = stablehlo.concatenate
// First insert writes row 0 into the empty matrix.
// CHECK: %[[AFTER0:.*]] = stablehlo.dynamic_update_slice %[[EMPTY]],
// Second insert writes row 1; its destination must be the first insert's
// result, not the empty matrix. Data-flow correctness load-bearing here.
// CHECK: %[[AFTER1:.*]] = stablehlo.dynamic_update_slice %[[AFTER0]],
// The read of M[1, 2] (= 23) must come from the populated chain tip,
// not from the empty matrix.
// CHECK: stablehlo.dynamic_slice %[[AFTER1]],

module attributes {llzk.lang, llzk.main = !struct.type<@MdsLike<[]>>} {
  struct.def @MdsLike {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@MdsLike<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@MdsLike<[]>>

      // Build M[2][3] = [[11, 12, 13], [21, 22, 23]] via array.new (Empty)
      // + per-row array.insert of inline 1D literals. This is exactly what
      // circom's gen_context.rs:2566 emits for an inline 2D literal.
      %M = array.new : <2,3 x !felt.type>

      %c11 = felt.const 11 : !felt.type
      %c12 = felt.const 12 : !felt.type
      %c13 = felt.const 13 : !felt.type
      %row0 = array.new %c11, %c12, %c13 : <3 x !felt.type>

      %c21 = felt.const 21 : !felt.type
      %c22 = felt.const 22 : !felt.type
      %c23 = felt.const 23 : !felt.type
      %row1 = array.new %c21, %c22, %c23 : <3 x !felt.type>

      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      array.insert %M[%i0] = %row0 : <2,3 x !felt.type>, <3 x !felt.type>
      array.insert %M[%i1] = %row1 : <2,3 x !felt.type>, <3 x !felt.type>

      // Read M[1, 2] — should be 23. Before the fix this reads from the
      // empty matrix and returns 0.
      %i2 = arith.constant 2 : index
      %v = array.read %M[%i1, %i2] : <2,3 x !felt.type>, !felt.type

      struct.writem %self[@out] = %v : <@MdsLike<[]>>, !felt.type
      function.return %self : !struct.type<@MdsLike<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@MdsLike<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
