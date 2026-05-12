// RUN: llzk-to-shlo-opt --simplify-sub-components --verify-each %s | FileCheck %s

// Regression for `replacePodReads` Phase 2 transitive-RAUW use-after-erase.
// Phase 1 sets `tracker[%pod_b][@in] = %outer` where `%outer` is itself a
// `pod.read` result. Phase 2's `block.walk` queues `%outer` for erase, then
// rewires the inner scf.if's `pod.read %pod_b[@in]` to `%outer` — adding a
// new use of an op the erase loop is about to destroy. MLIR aborts with
// "'pod.read' op operation destroyed but still has uses".
//
// Fix: walk the tracker chain to a non-pod.read terminal before RAUW. In
// this fixture the terminal is `%arg0`, so the inner call ends up calling
// `@Sub::@compute(%arg0)` directly.

// CHECK-LABEL: struct.def @Main_1
// CHECK-LABEL: function.def @compute(%arg0: !felt.type<"bn128">)
// CHECK-NOT: pod.read
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Sub {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Sub<[]>>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub<[]>>
      function.return %self : !struct.type<@Sub<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub<[]>>, %arg1: !felt.type<"bn128">)
        attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Main_1<[]>>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>

      // Outer pod initialized with %arg0 at @x. Phase 1 sets
      //   trackedPodValues[%pod_a][@x] = %arg0
      %pod_a = pod.new { @x = %arg0 } : <[@x: !felt.type<"bn128">]>

      // Inner dispatch pod with a 1-tick countdown. Phase 1 sets
      //   trackedPodValues[%pod_b][@count] = %c1
      // The non-count traffic at @in is the one driving the bug below.
      %c1 = arith.constant 1 : index
      %pod_b = pod.new { @count = %c1 }
          : <[@count: index, @comp: !struct.type<@Sub<[]>>, @in: !felt.type<"bn128">]>

      // OUTER pod.read — feeds the inner pod's @in slot via pod.write.
      // Phase 1 then sets trackedPodValues[%pod_b][@in] = %outer (a
      // pod.read result — the transitive-RAUW seed).
      %outer = pod.read %pod_a[@x]
          : <[@x: !felt.type<"bn128">]>, !felt.type<"bn128">
      pod.write %pod_b[@in] = %outer
          : <[@count: index, @comp: !struct.type<@Sub<[]>>, @in: !felt.type<"bn128">]>, !felt.type<"bn128">

      // Countdown -> fire on count == 0.
      %cnt = pod.read %pod_b[@count]
          : <[@count: index, @comp: !struct.type<@Sub<[]>>, @in: !felt.type<"bn128">]>, index
      %cnt2 = arith.subi %cnt, %c1 : index
      pod.write %pod_b[@count] = %cnt2
          : <[@count: index, @comp: !struct.type<@Sub<[]>>, @in: !felt.type<"bn128">]>, index
      %c0 = arith.constant 0 : index
      %fire = arith.cmpi eq, %cnt2, %c0 : index

      // Inner pod.read + function.call. Without the chain-resolve fix,
      // Phase 2 RAUWs %inner_r -> %outer; the function.call then holds a
      // dangling reference to %outer when the erase loop destroys it.
      scf.if %fire {
        %inner_r = pod.read %pod_b[@in]
            : <[@count: index, @comp: !struct.type<@Sub<[]>>, @in: !felt.type<"bn128">]>, !felt.type<"bn128">
        %res = function.call @Sub::@compute(%inner_r)
            : (!felt.type<"bn128">) -> !struct.type<@Sub<[]>>
        pod.write %pod_b[@comp] = %res
            : <[@count: index, @comp: !struct.type<@Sub<[]>>, @in: !felt.type<"bn128">]>, !struct.type<@Sub<[]>>
      }

      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !felt.type<"bn128">)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
