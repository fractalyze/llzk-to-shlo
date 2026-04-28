// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: inside an outer scf.while body, when a tracked array carry
// is updated mid-block by `array.write` (or `array.insert`) and then
// reused as the init operand of a NESTED scf.while in the same body,
// `processBlockForArrayMutations` must rewire the nested while's init
// operand from the pre-update block arg to the post-update SSA value.
// Otherwise the nested stablehlo.while reads from the pre-update carrier
// (block-arg / dense<0> after lowering) and the data chain breaks at the
// scf.while boundary.
//
// Caught downstream on AES `aes_256_encrypt`: a Num2Bits_2 result was
// inserted via `array.insert %arr[i,j,k] = %row` mid-body, and the
// immediately-following `scf.while` reading that slice initialized from
// the pre-insert carry — so the nested stablehlo.while iter-arg pinned
// to the all-zero outer carry instead of the just-updated array.
// Pre-fix: 312/14852 nonzero output cells; post-fix: chain connects.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// Outer write produces a dynamic_update_slice; the nested
// stablehlo.while in the same body must initialize from THAT result,
// not from the pre-update outer iter-arg.
// CHECK: %[[POST:.*]] = stablehlo.dynamic_update_slice %iterArg
// CHECK: stablehlo.while({{.*}} = %[[POST]]

module attributes {llzk.lang, llzk.main = !struct.type<@RewireInner<[]>>} {
  struct.def @RewireInner {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<4 x !felt.type>) -> !struct.type<@RewireInner<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@RewireInner<[]>>
      %dst = llzk.nondet : !array.type<4 x !felt.type>
      %inner_init = llzk.nondet : !array.type<4 x !felt.type>
      %f0 = felt.const  0 : !felt.type
      %f1 = felt.const  1 : !felt.type
      %f4 = felt.const  4 : !felt.type
      // Outer while carries %arr as a tracked-array iter-arg. Body:
      //   1. array.write %arr[%idx] = arg0[%idx]           ← updates latest[%arr]
      //   2. nested scf.while init = %arr                  ← without fix, reads pre-update
      //   3. nested body reads %arr2[%idx] and writes %dst ← post-fix: sees the just-written value
      // The nested-while-init rewire is the load-bearing fix.
      %0:2 = scf.while (%i = %f0, %arr = %inner_init) : (!felt.type, !array.type<4 x !felt.type>) -> (!felt.type, !array.type<4 x !felt.type>) {
        %lt = bool.cmp lt(%i, %f4) : !felt.type, !felt.type
        scf.condition(%lt) %i, %arr : !felt.type, !array.type<4 x !felt.type>
      } do {
      ^bb0(%i: !felt.type, %arr: !array.type<4 x !felt.type>):
        %idx = cast.toindex %i : !felt.type
        %v = array.read %arg0[%idx] : <4 x !felt.type>, !felt.type
        array.write %arr[%idx] = %v : <4 x !felt.type>, !felt.type
        // Nested while: init operand %arr must be rewired to the post-write
        // SSA value so the inner read sees the just-written cell.
        %1:2 = scf.while (%j = %f0, %arr2 = %arr) : (!felt.type, !array.type<4 x !felt.type>) -> (!felt.type, !array.type<4 x !felt.type>) {
          %ltj = bool.cmp lt(%j, %f4) : !felt.type, !felt.type
          scf.condition(%ltj) %j, %arr2 : !felt.type, !array.type<4 x !felt.type>
        } do {
        ^bb0(%j: !felt.type, %arr2: !array.type<4 x !felt.type>):
          %idx_j = cast.toindex %j : !felt.type
          %v_inner = array.read %arr2[%idx_j] : <4 x !felt.type>, !felt.type
          array.write %dst[%idx_j] = %v_inner : <4 x !felt.type>, !felt.type
          %j1 = felt.add %j, %f1 : !felt.type, !felt.type
          scf.yield %j1, %arr2 : !felt.type, !array.type<4 x !felt.type>
        }
        %i1 = felt.add %i, %f1 : !felt.type, !felt.type
        scf.yield %i1, %arr : !felt.type, !array.type<4 x !felt.type>
      }
      struct.writem %self[@out] = %dst : <@RewireInner<[]>>, !array.type<4 x !felt.type>
      function.return %self : !struct.type<@RewireInner<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@RewireInner<[]>>, %arg1: !array.type<4 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
