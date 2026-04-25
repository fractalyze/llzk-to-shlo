// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the LLZK v2 bump's Group A+B coverage gap.
// Two shapes are tested:
//   (A) array-of-dispatch-pods `@count` countdown in one scf.while, with
//       `pod.read @comp` in a *separate* post-loop scf.while that re-walks
//       the same array. The post-loop read can't be redirected to the
//       hoisted call (its SSA scope is the first loop's body), so it must
//       be replaced with `llzk.nondet` of the comp struct type.
//   (B) array-of-input-pods carried through scf.while with a `pod.read
//       @in` consumer; same justification as the scalar `eliminateInputPods`
//       path — the sub-component's `@constrain` re-derives the value.
// Either shape leaves residual `pod.*` ops if not handled — they fail
// dialect-conversion legality.

// CHECK-LABEL: func.func @main
// CHECK-NOT: pod.
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Comp_0 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Comp_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Comp_0<[]>>
      %0 = felt.mul %arg0, %arg0 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Comp_0<[]>>, !felt.type
      function.return %self : !struct.type<@Comp_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Comp_0<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    struct.member @sub$inputs : !array.type<2 x !pod.type<[@in: !felt.type]>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %dispatch_arr = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>
      %input_arr = array.new : <2 x !pod.type<[@in: !felt.type]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      // Initialize @count = 1 on every dispatch pod.
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %dispatch_arr[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %dispatch_arr[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
      }

      // Dispatch-firing loop: reads @count, decrements, fires the call
      // when post-decrement equals zero. The pod.read @count survives
      // resolveArrayPodCompReads — it's the Group A target.
      %felt0 = felt.const 0
      %0:2 = scf.while (%iter = %felt0, %carry = %input_arr) : (!felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>) -> (!felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>) {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter, %carry : !felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>
      } do {
      ^bb0(%iter: !felt.type, %carry: !array.type<2 x !pod.type<[@in: !felt.type]>>):
        %idx = cast.toindex %iter
        %xval = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %elem = array.read %carry[%idx] : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        pod.write %elem[@in] = %xval : <[@in: !felt.type]>, !felt.type
        array.write %carry[%idx] = %elem : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        %dp = array.read %dispatch_arr[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          %ax = pod.read %elem[@in] : <[@in: !felt.type]>, !felt.type
          %res = function.call @Comp_0::@compute(%ax) : (!felt.type) -> !struct.type<@Comp_0<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
          array.write %dispatch_arr[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        } else {
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !array.type<2 x !pod.type<[@in: !felt.type]>>
      }

      // Separate post-loop scf.while that reads `pod.read @comp` from the
      // already-dispatched array. resolveArrayPodCompReads can't redirect
      // these — the function.call SSA value is local to the dispatch loop's
      // body. The rewrite must replace each surviving pod.read @comp with
      // llzk.nondet of the comp struct type.
      %felt0_2 = felt.const 0
      %1 = scf.while (%iter = %felt0_2) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %dp = array.read %dispatch_arr[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp[@comp] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
        %out = struct.readm %comp[@out] : <@Comp_0<[]>>, !felt.type
        array.write %nondet_out[%idx] = %out : <2 x !felt.type>, !felt.type
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }

      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      struct.writem %self[@sub$inputs] = %0#1 : <@Main_1<[]>>, !array.type<2 x !pod.type<[@in: !felt.type]>>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
