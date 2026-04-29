// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// SimplifySubComponents Phase 5 (`replaceRemainingPodOps`) blanket-substitutes
// every surviving `pod.read` with `llzk.nondet` of the read's result type,
// including `pod.read [@params] : !pod.type<[]>` from a dispatcher's empty
// template-params field. Those reads are dead by construction (empty pod
// carries no value) and `LlzkNonDetPattern` previously failed to legalize
// the resulting `llzk.nondet : !pod.type<[]>` because there is no numeric
// tensor element type to materialize a zero for.
//
// This minimal repro inlines the post-SSC shape directly: a dead
// `llzk.nondet : !pod.type<[]>` that should be erased rather than rewritten
// to `stablehlo.constant`.

// CHECK-NOT: llzk.nondet
// CHECK-LABEL: func.func @main

module attributes {llzk.lang, llzk.main = !struct.type<@DeadEmptyPodNondet<[]>>} {
  struct.def @DeadEmptyPodNondet {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%in: !felt.type) -> !struct.type<@DeadEmptyPodNondet<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@DeadEmptyPodNondet<[]>>
      // Dead llzk.nondet of empty pod: no users, unrepresentable element type.
      %dead = "llzk.nondet"() : () -> !pod.type<[]>
      struct.writem %self[@out] = %in : <@DeadEmptyPodNondet<[]>>, !felt.type
      function.return %self : !struct.type<@DeadEmptyPodNondet<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@DeadEmptyPodNondet<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
