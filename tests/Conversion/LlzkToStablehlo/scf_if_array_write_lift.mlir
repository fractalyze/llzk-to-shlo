// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: a void scf.if containing `array.write` to an outer array
// inside a scf.while body must be lifted to result-bearing form so the
// array update threads through the surrounding scf.yield. Otherwise the
// `convertArrayWritesToSSA` walk does not recurse into scf.if regions
// and the void scf.if is erased by the dead-code branch in the
// scf.if → stablehlo.select post-pass — silently dropping the write.
//
// This is the shape that keccak_squeeze / keccak_chi / keccak_round0 /
// keccak_round20 / keccak_theta all hit: `for (i=0; i<2*N; i++) { if (i<N)
// { out[i] <== in[i]; } }`. Caught by the M3 correctness gate (uniformly
// zero GPU output where circom's witness held the copied input).

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// The lifted scf.if becomes a stablehlo.select that picks between the
// array carry and the slot-updated array, threading the result through
// the scf.yield that turns into the while's carry.
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.select
// No scf.if must survive into the lowered IR.
// CHECK-NOT: scf.if

module attributes {llzk.lang, llzk.main = !struct.type<@MinSqueeze<[]>>} {
  struct.def @MinSqueeze {
    struct.member @out : !array.type<8 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<16 x !felt.type>) -> !struct.type<@MinSqueeze<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@MinSqueeze<[]>>
      %dst = llzk.nondet : !array.type<8 x !felt.type>
      %f0 = felt.const  0 : !felt.type
      %f1 = felt.const  1 : !felt.type
      %f8 = felt.const  8 : !felt.type
      %f16 = felt.const  16 : !felt.type
      %0 = scf.while (%i = %f0) : (!felt.type) -> !felt.type {
        %lt = bool.cmp lt(%i, %f16) : !felt.type, !felt.type
        scf.condition(%lt) %i : !felt.type
      } do {
      ^bb0(%i: !felt.type):
        %guard = bool.cmp lt(%i, %f8) : !felt.type, !felt.type
        scf.if %guard {
          %idx = cast.toindex %i : !felt.type
          %v = array.read %arg0[%idx] : <16 x !felt.type>, !felt.type
          array.write %dst[%idx] = %v : <8 x !felt.type>, !felt.type
        } else {
        }
        %inc = felt.add %i, %f1 : !felt.type, !felt.type
        scf.yield %inc : !felt.type
      }
      struct.writem %self[@out] = %dst : <@MinSqueeze<[]>>, !array.type<8 x !felt.type>
      function.return %self : !struct.type<@MinSqueeze<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@MinSqueeze<[]>>, %arg1: !array.type<16 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
