// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the keccak chi/round0/round20/theta cross-while drop site.
// PR #29's `materializeScalarPodCompField` covers a single pod.new
// candidate; this test pins the multi-pod case where N independent pods
// each have their own writer-while plus a shared cross-block reader.
//
// Without the fix, `unpackPodWhileCarry` returns after unpacking ONE
// writer-while (the early return at the call site is a workaround for
// chained-while pointer invalidation in its inner SmallVector). The
// outer fixed-point loop's iter 1 sees only the first writer-while
// unpacked, the materializer succeeds for that pod's tail call, and
// `eliminatePodDispatch` Phase 5 (`replaceRemainingPodOps`) then
// `llzk.nondet`s every cross-block `pod.read` before iter 2 can rerun
// the materializer for the sibling pods.
//
// The fix drives `unpackPodWhileCarry` to its own fixed point inside
// the dispatch driver, so all writer-while pod-carries are unpacked
// before `materializeScalarPodCompField` surveys candidates. Both
// pods then materialize tail calls in iter 1.
//
// CHECK lines pin the contract that fails for `pod_b` without the fix:
//   - BOTH `func.call @Sub_0_compute(...)` results survive lowering with
//     non-zero consumers; without the fix, only one tail call appears.
//   - Reader-while body slices from BOTH call results, not from `dense<0>`.

// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[CALL_A:.*]] = call @Sub_0_compute(
// CHECK-DAG: %[[CALL_B:.*]] = call @Sub_0_compute(
// CHECK: stablehlo.while
// Reader-while slices from BOTH tail-call results.
// CHECK-DAG: stablehlo.slice %[[CALL_A]] [0:2]
// CHECK-DAG: stablehlo.slice %[[CALL_B]] [0:2]
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Sub_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
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
      struct.writem %self[@out] = %nondet : <@Sub_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_0<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<8 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<4 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      // TWO independent dispatch pods. Each has its own writer-while.
      // The reader-while consumes both — without the fix, only one of
      // the materializations succeeds in iter 1.
      %pod_a = pod.new { @count = %c4 } : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      %pod_b = pod.new { @count = %c4 } : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      %pod_in_a = pod.new : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      %pod_in_b = pod.new : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>

      // Writer-while #1: dispatch pod_a, inputs from %arg0[0..3].
      %fc0 = felt.const 0
      %0:2 = scf.while (%iter = %fc0, %carry = %pod_in_a) : (!felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>) -> (!felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>) {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      } do {
      ^bb0(%iter: !felt.type, %carry: !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>):
        %idx = cast.toindex %iter : !felt.type
        %va = array.read %arg0[%idx] : <8 x !felt.type>, !felt.type
        %ca = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %ca[%idx] = %va : <2 x !felt.type>, !felt.type
        pod.write %carry[@a] = %ca : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt1 = pod.read %pod_a[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt1m = arith.subi %cnt1, %c1 : index
        pod.write %pod_a[@count] = %cnt1m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire1 = arith.cmpi eq, %cnt1m, %c0 : index
        scf.if %fire1 {
          %a_in = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res1 = function.call @Sub_0::@compute(%a_in, %b_in) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod_a[@comp] = %res1 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        %fc2 = felt.const 2
        %iter_b = felt.add %iter, %fc2 : !felt.type, !felt.type
        %idx_b = cast.toindex %iter_b : !felt.type
        %vb = array.read %arg0[%idx_b] : <8 x !felt.type>, !felt.type
        %cb = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %cb[%idx] = %vb : <2 x !felt.type>, !felt.type
        pod.write %carry[@b] = %cb : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt2 = pod.read %pod_a[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2m = arith.subi %cnt2, %c1 : index
        pod.write %pod_a[@count] = %cnt2m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire2 = arith.cmpi eq, %cnt2m, %c0 : index
        scf.if %fire2 {
          %a_in2 = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in2 = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res2 = function.call @Sub_0::@compute(%a_in2, %b_in2) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod_a[@comp] = %res2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      }

      // Writer-while #2: dispatch pod_b, inputs from %arg0[4..7].
      %fc0_2 = felt.const 0
      %1:2 = scf.while (%iter = %fc0_2, %carry = %pod_in_b) : (!felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>) -> (!felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>) {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      } do {
      ^bb0(%iter: !felt.type, %carry: !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>):
        %fc4 = felt.const 4
        %iter_a = felt.add %iter, %fc4 : !felt.type, !felt.type
        %idx_a = cast.toindex %iter_a : !felt.type
        %idx_w = cast.toindex %iter : !felt.type
        %va = array.read %arg0[%idx_a] : <8 x !felt.type>, !felt.type
        %ca = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %ca[%idx_w] = %va : <2 x !felt.type>, !felt.type
        pod.write %carry[@a] = %ca : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt1 = pod.read %pod_b[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt1m = arith.subi %cnt1, %c1 : index
        pod.write %pod_b[@count] = %cnt1m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire1 = arith.cmpi eq, %cnt1m, %c0 : index
        scf.if %fire1 {
          %a_in = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res1 = function.call @Sub_0::@compute(%a_in, %b_in) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod_b[@comp] = %res1 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        %fc6 = felt.const 6
        %iter_b = felt.add %iter, %fc6 : !felt.type, !felt.type
        %idx_b = cast.toindex %iter_b : !felt.type
        %vb = array.read %arg0[%idx_b] : <8 x !felt.type>, !felt.type
        %cb = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %cb[%idx_w] = %vb : <2 x !felt.type>, !felt.type
        pod.write %carry[@b] = %cb : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt2 = pod.read %pod_b[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2m = arith.subi %cnt2, %c1 : index
        pod.write %pod_b[@count] = %cnt2m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire2 = arith.cmpi eq, %cnt2m, %c0 : index
        scf.if %fire2 {
          %a_in2 = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in2 = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res2 = function.call @Sub_0::@compute(%a_in2, %b_in2) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod_b[@comp] = %res2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      }

      // Reader scf.while: cross-block readback FROM BOTH POD COMPS.
      %fc0_3 = felt.const 0
      %2 = scf.while (%iter = %fc0_3) : (!felt.type) -> !felt.type {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %comp_a = pod.read %pod_a[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %out_a = struct.readm %comp_a[@out] : <@Sub_0<[]>>, !array.type<2 x !felt.type>
        %idx = cast.toindex %iter : !felt.type
        %va = array.read %out_a[%idx] : <2 x !felt.type>, !felt.type
        array.write %nondet_out[%idx] = %va : <4 x !felt.type>, !felt.type
        %comp_b = pod.read %pod_b[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %out_b = struct.readm %comp_b[@out] : <@Sub_0<[]>>, !array.type<2 x !felt.type>
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
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<8 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
