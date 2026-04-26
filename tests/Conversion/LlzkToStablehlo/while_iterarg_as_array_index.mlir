// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: when an scf.while iter-arg is the array index for `array.read`
// or `array.write`, the LLZK→StableHLO conversion must thread the loop
// counter into the resulting `stablehlo.dynamic_slice` /
// `stablehlo.dynamic_update_slice` index. Previously, `convertToIndexTensor`'s
// trace path (cast.toindex → look-through to a tensor felt) failed for an
// scf.while block-arg operand and fell into the "last resort: dense<0>"
// fallback. The lowered while body's slice/update_slice operations all
// pinned to position 0, making the loop a degenerate clobber.

// Caught downstream by the M3 correctness gate on keccak_pad: GPU output
// was uniformly zero where circom's witness held the alternating padded
// bits.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// The lowered while body's slice index must come from the iter-arg
// (a stablehlo.convert from the iter-arg's tensor<!pf> form to tensor<i32>),
// not a freshly emitted constant-0.
// CHECK: stablehlo.convert{{.*}} -> tensor<i32>
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.dynamic_update_slice

module attributes {llzk.lang, llzk.main = !struct.type<@CopyArr<[]>>} {
  struct.def @CopyArr {
    struct.member @out : !array.type<8 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<8 x !felt.type>) -> !struct.type<@CopyArr<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@CopyArr<[]>>
      %dst = llzk.nondet : !array.type<8 x !felt.type>
      %felt_const_0 = felt.const  0 : !felt.type
      %felt_const_8 = felt.const  8 : !felt.type
      %felt_const_1 = felt.const  1 : !felt.type
      %0 = scf.while (%i = %felt_const_0) : (!felt.type) -> !felt.type {
        %lt = bool.cmp lt(%i, %felt_const_8) : !felt.type, !felt.type
        scf.condition(%lt) %i : !felt.type
      } do {
      ^bb0(%i: !felt.type):
        %idx_r = cast.toindex %i : !felt.type
        %v = array.read %arg0[%idx_r] : <8 x !felt.type>, !felt.type
        %idx_w = cast.toindex %i : !felt.type
        array.write %dst[%idx_w] = %v : <8 x !felt.type>, !felt.type
        %inc = felt.add %i, %felt_const_1 : !felt.type, !felt.type
        scf.yield %inc : !felt.type
      }
      struct.writem %self[@out] = %dst : <@CopyArr<[]>>, !array.type<8 x !felt.type>
      function.return %self : !struct.type<@CopyArr<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@CopyArr<[]>>, %arg1: !array.type<8 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
