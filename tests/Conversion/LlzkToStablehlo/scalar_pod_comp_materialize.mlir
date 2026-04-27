// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the keccak iota3 / iota10 lane-0 zero residual. Shape:
//
//   1. `%pod = pod.new : <[@count, @comp: !struct, @params]>` at function scope
//      (SCALAR — not `array.new : <N x !pod<...>>`, which is the variant
//      `materializePodArrayCompField` covers).
//   2. A first scf.while ("writer") that decrements `@count`, fires
//      `function.call @Comp::@compute` from inside an scf.if, and writes
//      `pod.write %pod[@comp] = %res`. Call operands are body-block iter-args
//      mutated in-place by `array.write` per iteration (post-`unpackPodWhileCarry`
//      shape).
//   3. A *separate* sibling scf.while ("reader") that reads back via
//      `pod.read %pod[@comp]; struct.readm %comp[@out]`.
//
// Before the fix, `extractCallsFromScfIf` correctly hoists the call inside the
// writer-while body, but its tracker is per-block — the reader-while body's
// `pod.read %pod[@comp]` is in a different block, has no tracked entry, and
// Phase 5 (`replaceRemainingPodOps`) substitutes a fresh `llzk.nondet :
// !struct<...>`. The lowered helper silently slices from a `dense<0>` tensor.
//
// `materializeScalarPodCompField` emits a tail `function.call @F(<post-while
// operand values>)` after the writer-while and rewrites the cross-block
// reader's `pod.read [@comp]` to consume it. The substantively-firing call's
// operands are exactly the writer-while's post-loop iter-arg state, so the
// tail-call result matches the LLZK semantics.
//
// CHECK lines pin three contracts that all fail without the fix:
//   - `func.call @Sub_0_compute(...)` survives lowering with consumed result.
//   - The reader while body slices from the call result, not `dense<0>`.
//   - The function return is connected through the materialized scalar.

// CHECK-LABEL: func.func @main
// Writer-while populates two 2-felt arrays.
// CHECK: stablehlo.while
// Tail call after the writer-while consumes its post-loop iter-arg state —
// without the fix the call is absent from @main and the reader-while reads a
// zero constant.
// CHECK: %[[CALL:.*]] = call @Sub_0_compute(
// Reader-while slices from the tail-call result, not a `dense<0>` constant.
// CHECK: stablehlo.while
// CHECK: stablehlo.slice %[[CALL]] [0:2]
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
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<4 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      // SCALAR dispatch pod (the case PR #25 misses).
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %pod = pod.new { @count = %c4 } : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      // Carry pod that gets unpacked to two 2-felt arrays in writer-while.
      %pod_in = pod.new : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      // Writer scf.while: 2 iters. Body decrements @count twice, second
      // decrement substantively fires call once @count reaches 0.
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
        // First decrement + scf.if.
        %cnt1 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt1m = arith.subi %cnt1, %c1 : index
        pod.write %pod[@count] = %cnt1m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire1 = arith.cmpi eq, %cnt1m, %c0 : index
        scf.if %fire1 {
          %a_in = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res1 = function.call @Sub_0::@compute(%a_in, %b_in) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod[@comp] = %res1 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        // Read second-half input into @b.
        %fc2 = felt.const 2
        %iter_b = felt.add %iter, %fc2 : !felt.type, !felt.type
        %idx_b = cast.toindex %iter_b : !felt.type
        %vb = array.read %arg0[%idx_b] : <4 x !felt.type>, !felt.type
        %cb = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %cb[%idx] = %vb : <2 x !felt.type>, !felt.type
        pod.write %carry[@b] = %cb : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        // Second decrement + scf.if. SUBSTANTIVELY fires when @count hits 0
        // at iter 1 (after 4 decrements: 4→3→2→1→0). %carry @a/@b are fully
        // populated at that point — exactly what the tail call observes.
        %cnt2 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2m = arith.subi %cnt2, %c1 : index
        pod.write %pod[@count] = %cnt2m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire2 = arith.cmpi eq, %cnt2m, %c0 : index
        scf.if %fire2 {
          %a_in2 = pod.read %carry[@a] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b_in2 = pod.read %carry[@b] : <[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %res2 = function.call @Sub_0::@compute(%a_in2, %b_in2) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod[@comp] = %res2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !pod.type<[@a: !array.type<2 x !felt.type>, @b: !array.type<2 x !felt.type>]>
      }
      // Reader scf.while: cross-block readback.
      %fc0_2 = felt.const 0
      %1 = scf.while (%iter = %fc0_2) : (!felt.type) -> !felt.type {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %comp = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %out = struct.readm %comp[@out] : <@Sub_0<[]>>, !array.type<2 x !felt.type>
        %idx = cast.toindex %iter : !felt.type
        %v = array.read %out[%idx] : <2 x !felt.type>, !felt.type
        array.write %nondet_out[%idx] = %v : <2 x !felt.type>, !felt.type
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<4 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
