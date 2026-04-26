// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: when two adjacent scf.while loops both reference the same
// `llzk.nondet`-defined array and the second loop reads what the first
// loop wrote, `promoteArraysToWhileCarry` must thread the first while's
// updated-array carry into the second while's init. Otherwise the second
// while reads from the original `llzk.nondet` zero-init and the data
// chain is broken — caught downstream by the M3 keccak_pad gate where
// the public output `out[i] <== out2[i]` was always zero because the
// `out2`-writing loop's result wasn't connected to the `out`-writing
// loop's `out2`-reading body.

// CHECK-LABEL: func.func @main
// CHECK: %[[FIRST:.*]]:{{[0-9]+}} = stablehlo.while
// CHECK: %[[SECOND:.*]]:{{[0-9]+}} = stablehlo.while({{.*}}%[[FIRST]]#{{[0-9]+}}{{.*}}
// The second while's array carry must be initialized from the first
// while's array carry result, not a freshly emitted constant.

module attributes {llzk.lang, llzk.main = !struct.type<@CopyChain<[]>>} {
  struct.def @CopyChain {
    struct.member @out : !array.type<8 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<8 x !felt.type>) -> !struct.type<@CopyChain<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@CopyChain<[]>>
      %tmp = llzk.nondet : !array.type<8 x !felt.type>
      %dst = llzk.nondet : !array.type<8 x !felt.type>
      %felt_const_0 = felt.const  0 : !felt.type
      %felt_const_8 = felt.const  8 : !felt.type
      %felt_const_1 = felt.const  1 : !felt.type
      // First while: tmp[i] = arg0[i] for i in 0..8
      %0 = scf.while (%i = %felt_const_0) : (!felt.type) -> !felt.type {
        %lt = bool.cmp lt(%i, %felt_const_8) : !felt.type, !felt.type
        scf.condition(%lt) %i : !felt.type
      } do {
      ^bb0(%i: !felt.type):
        %idx = cast.toindex %i : !felt.type
        %v = array.read %arg0[%idx] : <8 x !felt.type>, !felt.type
        %widx = cast.toindex %i : !felt.type
        array.write %tmp[%widx] = %v : <8 x !felt.type>, !felt.type
        %inc = felt.add %i, %felt_const_1 : !felt.type, !felt.type
        scf.yield %inc : !felt.type
      }
      // Second while: dst[i] = tmp[i] for i in 0..8
      %1 = scf.while (%j = %felt_const_0) : (!felt.type) -> !felt.type {
        %lt = bool.cmp lt(%j, %felt_const_8) : !felt.type, !felt.type
        scf.condition(%lt) %j : !felt.type
      } do {
      ^bb0(%j: !felt.type):
        %ridx = cast.toindex %j : !felt.type
        %v = array.read %tmp[%ridx] : <8 x !felt.type>, !felt.type
        %widx = cast.toindex %j : !felt.type
        array.write %dst[%widx] = %v : <8 x !felt.type>, !felt.type
        %inc = felt.add %j, %felt_const_1 : !felt.type, !felt.type
        scf.yield %inc : !felt.type
      }
      struct.writem %self[@out] = %dst : <@CopyChain<[]>>, !array.type<8 x !felt.type>
      function.return %self : !struct.type<@CopyChain<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@CopyChain<[]>>, %arg1: !array.type<8 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
