// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for `eraseDeadPodAndCountOps` Phase 4: when the candidate is a
// region-bearing op (`scf.if`) whose own results are unused, its body may
// contain an op whose result is consumed by a `function.call` that
// `extractCallsFromScfIf` (Phase 1) hoisted BEFORE the scf.if via the
// `directArgs` path. Without the gate, region-clearing the scf.if destroys
// the inner producer while the hoisted call still uses it — MLIR aborts
// with "operation destroyed but still has uses". Companion idempotence
// guard in `extractCallsFromScfIf` prevents the hoist from firing when any
// direct-arg is defined inside the scf.if (otherwise the outer fixed point
// would spin re-hoisting a duplicate every iter once the gate refuses to
// erase).
//
// Pattern: an `scf.if` returning a single-record `!pod.type<[@out: !felt]>`
// whose body extracts a felt from an outer `array` via `array.read` (a
// direct, non-pod.read value) and feeds it to `function.call @Sub::@compute`.
// Phase 1's `directArgs` branch would hoist the call to before the scf.if;
// without the guard, the hoist uses an SSA value defined inside the scf.if;
// without the gate, Phase 4 then region-clears the scf.if and trips MLIR.

// With the fix in place: Phase 1's `directArgs` guard skips the hoist
// (the `array.read` operand is defined inside the scf.if), so Phase 4
// can erase the dead scf.if cleanly — its body has no external users.
// `CHECK-NOT: scf.if` between the function header and the return asserts
// the erase actually fired; pre-fix the pass aborts before any output is
// produced, so neither CHECK matches.
// CHECK-LABEL: struct.def @Main_1
// CHECK-LABEL: function.def @compute(%arg0
// CHECK-NOT: scf.if
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Sub {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Sub<[]>>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub<[]>>
      function.return %self : !struct.type<@Sub<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub<[]>>, %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type<"bn128">>) -> !struct.type<@Main_1<[]>>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet = llzk.nondet : !felt.type<"bn128">
      %c0 = arith.constant 0 : index
      %fz = felt.const 0 : <"bn128">
      %cond = bool.cmp eq(%nondet, %fz) : !felt.type<"bn128">, !felt.type<"bn128">

      // Result-bearing scf.if returning a single-record pod. The body's
      // `array.read` against the function's arg array is a direct,
      // non-pod.read producer that feeds the inner `function.call`. After
      // `materializeScalarPodCompField` resolves the `@comp` chain, the
      // scf.if's own result becomes unused — Phase 4 then considers it for
      // erase, which trips the gate (inner array.read has external user
      // via Phase-1-hoisted call).
      %r = scf.if %cond -> (!pod.type<[@out: !struct.type<@Sub<[]>>]>) {
        %elem = array.read %arg0[%c0] : <2 x !felt.type<"bn128">>, !felt.type<"bn128">
        %res = function.call @Sub::@compute(%elem) : (!felt.type<"bn128">) -> !struct.type<@Sub<[]>>
        %p = pod.new : <[@out: !struct.type<@Sub<[]>>]>
        pod.write %p[@out] = %res : <[@out: !struct.type<@Sub<[]>>]>, !struct.type<@Sub<[]>>
        scf.yield %p : !pod.type<[@out: !struct.type<@Sub<[]>>]>
      } else {
        %p2 = pod.new : <[@out: !struct.type<@Sub<[]>>]>
        scf.yield %p2 : !pod.type<[@out: !struct.type<@Sub<[]>>]>
      }

      struct.writem %self[@out] = %nondet : <@Main_1<[]>>, !felt.type<"bn128">
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type<"bn128">>)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
