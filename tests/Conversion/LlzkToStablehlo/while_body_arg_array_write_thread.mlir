// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: when an scf.while body contains writes to BOTH a body block
// arg (e.g. carried over from `flattenPodArrayWhileCarry`'s pod-array
// decomposition) AND an outer-scope captured array, the
// `processBlockForArrayMutations` walker inside
// `convertArrayWritesToSSA` (called from `promoteArraysToWhileCarry`)
// must NOT eagerly convert writes for arrays it doesn't track. Eager
// conversion leaves the new result-bearing op orphaned (latest map not
// updated for untracked arrays, yield not re-routed), and the later
// `convertWhileBodyArgsToSSA` pass skips the now-result-bearing write —
// so its result is unused and downstream DCE drops the write entirely.
// Caught by the M3 correctness gate on keccak_chi / keccak_round0 /
// keccak_round20 / keccak_theta where the input-capture write inside
// XorArraySingle/AndArray/XorArray helper bodies came back as dense<0>.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.dynamic_slice %arg0
// The body-block-arg write must thread through the yield, so the read of
// the same block arg position post-write reads the just-written value.
// CHECK: %[[ARG1_DUS:.*]] = stablehlo.dynamic_update_slice %iterArg
// CHECK: stablehlo.dynamic_slice %[[ARG1_DUS]]
// And the captured outer-scope array's write threads too.
// CHECK: stablehlo.dynamic_update_slice %iterArg
// The yield must include the updated block-arg value; otherwise the
// while result for that carry stays at the dense<0> init.
// CHECK: stablehlo.return %{{.*}}, %[[ARG1_DUS]]

module attributes {llzk.lang, llzk.main = !struct.type<@WriteThroughBlockArg<[]>>} {
  struct.def @WriteThroughBlockArg {
    struct.member @out : !array.type<8 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<8 x !felt.type>) -> !struct.type<@WriteThroughBlockArg<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@WriteThroughBlockArg<[]>>
      %outer = llzk.nondet : !array.type<8 x !felt.type>
      %inner_init = llzk.nondet : !array.type<8 x !felt.type>
      %f0 = felt.const  0 : !felt.type
      %f1 = felt.const  1 : !felt.type
      %f8 = felt.const  8 : !felt.type
      // While carries an existing felt-array iter-arg %arr (would have been
      // synthesized by flattenPodArrayWhileCarry from a pod-array carry in
      // a real circuit). Body reads %arg0[i], writes both %arr[i]=%v and
      // %outer[i]=%arr[i]. With the bug, promote's pre-conversion strips
      // %arr's write but leaves no consumer; convertWhileBodyArgsToSSA
      // skips it; the read of %arr[i] post-write returns dense<0>; the
      // %outer write copies dense<0> instead of %v.
      %0:2 = scf.while (%i = %f0, %arr = %inner_init) : (!felt.type, !array.type<8 x !felt.type>) -> (!felt.type, !array.type<8 x !felt.type>) {
        %lt = bool.cmp lt(%i, %f8) : !felt.type, !felt.type
        scf.condition(%lt) %i, %arr : !felt.type, !array.type<8 x !felt.type>
      } do {
      ^bb0(%i: !felt.type, %arr: !array.type<8 x !felt.type>):
        %idx = cast.toindex %i : !felt.type
        %v = array.read %arg0[%idx] : <8 x !felt.type>, !felt.type
        array.write %arr[%idx] = %v : <8 x !felt.type>, !felt.type
        %v2 = array.read %arr[%idx] : <8 x !felt.type>, !felt.type
        array.write %outer[%idx] = %v2 : <8 x !felt.type>, !felt.type
        %inc = felt.add %i, %f1 : !felt.type, !felt.type
        scf.yield %inc, %arr : !felt.type, !array.type<8 x !felt.type>
      }
      struct.writem %self[@out] = %outer : <@WriteThroughBlockArg<[]>>, !array.type<8 x !felt.type>
      function.return %self : !struct.type<@WriteThroughBlockArg<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@WriteThroughBlockArg<[]>>, %arg1: !array.type<8 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
