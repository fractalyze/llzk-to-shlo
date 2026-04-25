// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: `pod.new` of a single-field pod, threaded through `scf.while`
// as a dead carry (no `pod.read`/`pod.write` users). circom v2 emits this
// shape for sub-component dispatch slots whose body never reads the pod.
// `inlineInputPodCarries` previously skipped this case (its `innerType`
// lookup required a `pod.read` user) and the surviving `pod.new` failed
// dialect legalization.

// CHECK-NOT: pod.
// CHECK-NOT: !pod.type

// CHECK-LABEL: func.func @main
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@DeadPodCarry<[]>>} {
  struct.def @DeadPodCarry {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@DeadPodCarry<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@DeadPodCarry<[]>>
      %pod = pod.new : <[@dummy: !array.type<4 x !felt.type>]>
      %felt_const_0 = felt.const  0 : !felt.type
      %0:2 = scf.while (%arg1 = %pod, %arg2 = %felt_const_0) : (!pod.type<[@dummy: !array.type<4 x !felt.type>]>, !felt.type) -> (!pod.type<[@dummy: !array.type<4 x !felt.type>]>, !felt.type) {
        %felt_const_4 = felt.const  4 : !felt.type
        %lt = bool.cmp lt(%arg2, %felt_const_4) : !felt.type, !felt.type
        scf.condition(%lt) %arg1, %arg2 : !pod.type<[@dummy: !array.type<4 x !felt.type>]>, !felt.type
      } do {
      ^bb0(%arg1: !pod.type<[@dummy: !array.type<4 x !felt.type>]>, %arg2: !felt.type):
        %felt_const_1 = felt.const  1 : !felt.type
        %inc = felt.add %arg2, %felt_const_1 : !felt.type, !felt.type
        scf.yield %arg1, %inc : !pod.type<[@dummy: !array.type<4 x !felt.type>]>, !felt.type
      }
      struct.writem %self[@out] = %arg0 : <@DeadPodCarry<[]>>, !felt.type
      function.return %self : !struct.type<@DeadPodCarry<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@DeadPodCarry<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
