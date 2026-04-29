// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the AES Num2Bits_2 input-pod data-flow gap. Shape:
//
//   1. `%arr_in = array.new : <N x !pod<[@in: !felt]>>` at function scope —
//      the per-instance input-pod array Circom emits to stage the dispatch's
//      operand. (Distinct from the sibling dispatch pod array
//      `<N x !pod<[@count, @comp:T, @params]>>` covered by PR #41.)
//   2. The writer scf.while threads `%arr_in` through 2+ nested scf.while
//      iter-args. The innermost body does:
//         %cell = array.read %arr_in_iter[idx]
//         pod.write %cell[@in] = %src           // %src is the input value
//         array.write %arr_in_iter[idx] = %cell
//      and the firing scf.if (count countdown == 0) does:
//         %read = pod.read %cell[@in]
//         function.call @Sub_0::@compute(%read)
//
// PR #41 + PR #42 closed the OUTPUT-side flow (the dispatched call's
// `@out : !array<K x !felt>` reaches the parent witness via materialized
// per-field arrays + scf.while init-rewire). They did NOT cover the
// INPUT-side flow: `flattenPodArrayWhileCarry` at multi-level nesting
// disconnects the per-iteration `%src` from the firing-site `%read`,
// `rewriteArrayPodCountCompInReads` substitutes `pod.read %cell[@in]`
// with `llzk.nondet`, and the firing call sees zero. Symptom: AES
// `aes_256_encrypt` runtime stuck at 344/14852 instead of ≥ 3500.
//
// `materializePodArrayInputPodField` runs BEFORE `flattenPodArrayWhileCarry`,
// while the writer/reader are still SSA-paired through `%cell`, and rewires
// the firing-site call's operand to consume `%src` directly. The CHECK chain
// below pins the dataflow positively: the writer's `felt.add %a, 3` is the
// SBox-style source, and the dispatched `Sub_0_compute` must consume that
// `stablehlo.add` result rather than a `stablehlo.constant` placeholder.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: %[[SRC:[^ ]+]] = stablehlo.add
// CHECK: func.call @Sub_0_compute(%[[SRC]])
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  // Sub_0::@compute takes a felt input; the witness gap is whether the
  // call's operand is the writer's `%src` or a const-zero placeholder.
  struct.def @Sub_0 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Sub_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
      %felt2 = felt.const 2
      %0 = felt.mul %arg0, %felt2 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Sub_0<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_0<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    struct.member @sub_0$inputs : !array.type<2 x !pod.type<[@in: !felt.type]>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %arr_count = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>
      %arr_in = array.new : <2 x !pod.type<[@in: !felt.type]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %arr_count[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %arr_count[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      }
      // Outer scf.while threads %arr_count + %arr_in as iter-args (matching
      // the AES `@AES256Encrypt_6::@compute` writer-while shape).
      %felt0 = felt.const 0
      %0:2 = scf.while (%iter = %felt0, %arr_in_iter = %arr_in)
          : (!felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>)
          -> (!felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>) {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter, %arr_in_iter
            : !felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>
      } do {
      ^bb0(%iter: !felt.type, %arr_in_iter: !array.type<2 x !pod.type<[@in: !felt.type]>>):
        %idx = cast.toindex %iter
        %a = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %felt3 = felt.const 3
        // %src is the per-iteration input value to be threaded through the
        // input-pod array into the firing-site call.
        %src = felt.add %a, %felt3 : !felt.type, !felt.type
        // Writer pattern.
        %cell = array.read %arr_in_iter[%idx] : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        pod.write %cell[@in] = %src : <[@in: !felt.type]>, !felt.type
        array.write %arr_in_iter[%idx] = %cell : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        // Counter-pod countdown — fires the dispatch on count==0.
        %dp = array.read %arr_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          // Firing site: read the staged input and dispatch.
          %read = pod.read %cell[@in] : <[@in: !felt.type]>, !felt.type
          %res = function.call @Sub_0::@compute(%read) : (!felt.type) -> !struct.type<@Sub_0<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
          array.write %arr_count[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        } else {
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next, %arr_in_iter
            : !felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>
      }
      // Reader scf.while: walks %arr_count to recover @out into %nondet_out.
      %felt0_2 = felt.const 0
      %1 = scf.while (%iter = %felt0_2) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %dp2 = array.read %arr_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %out = struct.readm %comp[@out] : <@Sub_0<[]>>, !felt.type
        array.write %nondet_out[%idx] = %out : <2 x !felt.type>, !felt.type
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
