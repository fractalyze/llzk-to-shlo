// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the keccak iota3 / iota10 / round0 / round20 cross-while drop
// site post project-llzk/circom PR #390 (2026-04-30). Two distinct shapes
// share the same root cause and must both materialize:
//
//   1. WRITER-BEARING `llzk.nondet` dispatch pod. The pod-creation site is
//      `llzk.nondet : !pod.type<[@count, @comp: !struct<@Sub>, @params]>`
//      (NOT `pod.new {@count = const_N}`). `materializeScalarPodCompField`
//      must admit `llzk.nondet` to its candidate filter — otherwise the
//      scf.if writers are detected but the post-while tail call is never
//      synthesized, the cross-while reader falls through Phase 5
//      (`replaceRemainingPodOps`), and `@main` lowers without a substruct
//      call. Pinned by `pod_a` below.
//
//   2. NO-WRITER zero-arg `llzk.nondet` dispatch pod. circom #390 dropped
//      the inline `function.call @Sub::@compute()` materialization for
//      constant-table sub-components (keccak's RC_0). Without a writer to
//      extract operands from, `materializeScalarPodCompField` must
//      synthesize a zero-arg call at function scope, gated on the
//      @compute method actually being zero-arg via the module SymbolTable.
//      Pinned by `pod_b` below.
//
// Both calls must reach lowered `@main`; without the fix, neither shape
// produces a substruct call and the reader-while reads from `dense<0>`
// constants.

// CHECK-LABEL: func.func @main
// `pod_a`: writer-bearing dispatch — tail call uses post-while iter-args.
// CHECK-DAG: %[[CALL_A:.*]] = call @SubArg_0_compute(
// `pod_b`: no-writer zero-arg dispatch — synthesized at function scope.
// CHECK-DAG: %[[CALL_B:.*]] = call @SubZero_0_compute(
// CHECK: stablehlo.while
// Reader-while body slices from BOTH call results, not from a constant.
// CHECK-DAG: stablehlo.slice %[[CALL_A]] [0:2]
// CHECK-DAG: stablehlo.slice %[[CALL_B]] [0:2]
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  // 2-arg sub-component (writer-bearing dispatch).
  struct.def @SubArg_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !array.type<2 x !felt.type>) -> !struct.type<@SubArg_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@SubArg_0<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %a0 = array.read %arg0[%c0] : <2 x !felt.type>, !felt.type
      %b0 = array.read %arg1[%c0] : <2 x !felt.type>, !felt.type
      %s0 = felt.add %a0, %b0 : !felt.type, !felt.type
      array.write %nondet[%c0] = %s0 : <2 x !felt.type>, !felt.type
      %a1 = array.read %arg0[%c1] : <2 x !felt.type>, !felt.type
      %b1 = array.read %arg1[%c1] : <2 x !felt.type>, !felt.type
      %s1 = felt.add %a1, %b1 : !felt.type, !felt.type
      array.write %nondet[%c1] = %s1 : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet : <@SubArg_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@SubArg_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@SubArg_0<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
  // 0-arg sub-component (no-writer zero-arg dispatch — RC_0 analog).
  struct.def @SubZero_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute() -> !struct.type<@SubZero_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@SubZero_0<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %k0 = felt.const 7
      %k1 = felt.const 11
      array.write %nondet[%c0] = %k0 : <2 x !felt.type>, !felt.type
      array.write %nondet[%c1] = %k1 : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet : <@SubZero_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@SubZero_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@SubZero_0<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<4 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<4 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Writer-bearing dispatch pod (post-#390 shape: llzk.nondet, NOT pod.new).
      %pod_a = llzk.nondet : !pod.type<[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>
      // Zero-arg dispatch pod — no writer, only readers.
      %pod_b = llzk.nondet : !pod.type<[@count: index, @comp: !struct.type<@SubZero_0<[]>>, @params: !pod.type<[]>]>
      %pod_in = pod.new : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>

      // Writer-while: dispatch pod_a, inputs from %arg0[0..3].
      %fc0 = felt.const 0
      %0:2 = scf.while (%iter = %fc0, %carry = %pod_in) : (!felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>) -> (!felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>) {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      } do {
      ^bb0(%iter: !felt.type, %carry: !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>):
        %idx = cast.toindex %iter : !felt.type
        %va = array.read %arg0[%idx] : <4 x !felt.type>, !felt.type
        %ca = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %ca[%idx] = %va : <2 x !felt.type>, !felt.type
        pod.write %carry[@a] = %ca : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt1 = pod.read %pod_a[@count] : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt1m = arith.subi %cnt1, %c1 : index
        pod.write %pod_a[@count] = %cnt1m : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, index
        %fire1 = arith.cmpi eq, %cnt1m, %c0 : index
        scf.if %fire1 {
          %a_in = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res1 = function.call @SubArg_0::@compute(%a_in, %b_in) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@SubArg_0<[]>>
          pod.write %pod_a[@comp] = %res1 : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@SubArg_0<[]>>
        }
        %fc2 = felt.const 2
        %iter_b = felt.add %iter, %fc2 : !felt.type, !felt.type
        %idx_b = cast.toindex %iter_b : !felt.type
        %vb = array.read %arg0[%idx_b] : <4 x !felt.type>, !felt.type
        %cb = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %cb[%idx] = %vb : <2 x !felt.type>, !felt.type
        pod.write %carry[@b] = %cb : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt2 = pod.read %pod_a[@count] : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2m = arith.subi %cnt2, %c1 : index
        pod.write %pod_a[@count] = %cnt2m : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, index
        %fire2 = arith.cmpi eq, %cnt2m, %c0 : index
        scf.if %fire2 {
          %a_in2 = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in2 = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res2 = function.call @SubArg_0::@compute(%a_in2, %b_in2) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@SubArg_0<[]>>
          pod.write %pod_a[@comp] = %res2 : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@SubArg_0<[]>>
        }
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      }

      // Reader scf.while: cross-block readback from BOTH dispatch pods.
      %fc0_2 = felt.const 0
      %1 = scf.while (%iter = %fc0_2) : (!felt.type) -> !felt.type {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %comp_a = pod.read %pod_a[@comp] : <[@count: index, @comp: !struct.type<@SubArg_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@SubArg_0<[]>>
        %out_a = struct.readm %comp_a[@out] : <@SubArg_0<[]>>, !array.type<2 x !felt.type>
        %idx = cast.toindex %iter : !felt.type
        %va = array.read %out_a[%idx] : <2 x !felt.type>, !felt.type
        array.write %nondet_out[%idx] = %va : <4 x !felt.type>, !felt.type
        %comp_b = pod.read %pod_b[@comp] : <[@count: index, @comp: !struct.type<@SubZero_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@SubZero_0<[]>>
        %out_b = struct.readm %comp_b[@out] : <@SubZero_0<[]>>, !array.type<2 x !felt.type>
        %vb = array.read %out_b[%idx] : <2 x !felt.type>, !felt.type
        %fc2 = felt.const 2
        %iter_off = felt.add %iter, %fc2 : !felt.type, !felt.type
        %idx_off = cast.toindex %iter_off : !felt.type
        array.write %nondet_out[%idx_off] = %vb : <4 x !felt.type>, !felt.type
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<4 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<4 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
