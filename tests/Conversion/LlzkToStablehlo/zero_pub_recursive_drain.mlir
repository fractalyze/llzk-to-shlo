// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Phase 3 recursive K=0 drain. Mirrors `materializePodArrayCompField`'s
// new path triggered when the dispatched inner struct exposes NO
// `{llzk.pub}` felt members (so the K=1/K>1 multi-pub flatten falls
// through) but has writem-targeted, non-pod, heterogeneous-shape members
// with non-zero recursive flat size — webb's `@ManyMerkleProof_275`
// shape (`@hasher : <30 x !felt>` + `@switcher : <30, 2 x !felt>`).
//
// What this pins (post-SSC):
//   1. Inner @Sub's previously-private writem-targeted members @a / @b
//      get promoted to `{llzk.pub}` so the parent's writer-emit
//      `struct.readm %callResult[@a/@b]` is legal under LLZK's
//      MemberReadOp visibility verifier.
//   2. Parent's `@drain` struct.member is retyped from `<2 x !struct<@Sub>>`
//      to a flat `<2, 5 x !felt>` (totalFlat = @a (2) + @b (3) = 5).
//   3. Parent emits a `<2, 5 x !felt>` destFelt allocation and one
//      `array.write` per element per writer site (per-instance,
//      row-major over each writem-targeted member's natural shape).
//
// CHECK-LABEL: struct.def @Sub
// Both writem-targeted members are promoted to pub for external readm.
// CHECK: struct.member @a : !array.type<2 x !felt.type{{.*}}{llzk.pub}
// CHECK: struct.member @b : !array.type<3 x !felt.type{{.*}}{llzk.pub}
//
// CHECK-LABEL: struct.def @Main
// Parent drain member retyped: 2 instances × 5 felts/instance = 10 cells.
// CHECK: struct.member @drain : !array.type<2,5 x !felt.type
//
// CHECK-LABEL: function.def @compute
// destFelt allocated with the new flat shape.
// CHECK: array.new {{.*}}: <2,5 x !felt.type
// At each writer site, the @a + @b members get read out externally;
// values are unrolled into per-element `array.write` ops into destFelt.
// CHECK: struct.readm {{.*}}[@a]
// CHECK: array.write
// CHECK: struct.readm {{.*}}[@b]
// CHECK: array.write

module attributes {llzk.lang, llzk.main = !struct.type<@Main<[]>>} {
  // Inner Sub has TWO writem-targeted non-pod members of DIFFERENT
  // shapes (mixed-shape, so K>1 uniform-shape gate rejects). Neither
  // is pub — Phase 3 promotes both during the K=0 drain.
  struct.def @Sub {
    struct.member @a : !array.type<2 x !felt.type>
    struct.member @b : !array.type<3 x !felt.type>
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Sub<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub<[]>>
      %nondet_a = llzk.nondet : !array.type<2 x !felt.type>
      %nondet_b = llzk.nondet : !array.type<3 x !felt.type>
      struct.writem %self[@a] = %nondet_a : <@Sub<[]>>, !array.type<2 x !felt.type>
      struct.writem %self[@b] = %nondet_b : <@Sub<[]>>, !array.type<3 x !felt.type>
      function.return %self : !struct.type<@Sub<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main {
    // Inputs to the dispatched sub-components — populated by the writer.
    struct.member @drain$inputs : !array.type<2 x !pod.type<[@in: !felt.type]>>
    // The drain destination: a struct array of @Sub. Without Phase 3,
    // its writem is splat-zero (K=0-pub-felt path skipped pre-fix).
    struct.member @drain : !array.type<2 x !struct.type<@Sub<[]>>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Main<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main<[]>>
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>
      %inputs = array.new : <2 x !pod.type<[@in: !felt.type]>>
      %destStruct = array.new : <2 x !struct.type<@Sub<[]>>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %array[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
      }
      // Writer scf.while: dispatches Sub::@compute(arg0[i]) per iter.
      %felt0 = felt.const 0
      %0 = scf.while (%iter = %felt0) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %a = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %ip = array.read %inputs[%idx] : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        pod.write %ip[@in] = %a : <[@in: !felt.type]>, !felt.type
        array.write %inputs[%idx] = %ip : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        %dp = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          %res = function.call @Sub::@compute(%a) : (!felt.type) -> !struct.type<@Sub<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub<[]>>
          array.write %array[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      // Drain: copy each dispatch-pod's @comp into %destStruct.
      %felt0_b = felt.const 0
      %1 = scf.while (%iter2 = %felt0_b) : (!felt.type) -> !felt.type {
        %felt2_b = felt.const 2
        %cond2 = bool.cmp lt(%iter2, %felt2_b)
        scf.condition(%cond2) %iter2 : !felt.type
      } do {
      ^bb0(%iter2: !felt.type):
        %idx2 = cast.toindex %iter2
        %dp2 = array.read %array[%idx2] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub<[]>>
        array.write %destStruct[%idx2] = %comp : <2 x !struct.type<@Sub<[]>>>, !struct.type<@Sub<[]>>
        %felt1_b = felt.const 1
        %next2 = felt.add %iter2, %felt1_b : !felt.type, !felt.type
        scf.yield %next2 : !felt.type
      }
      struct.writem %self[@drain$inputs] = %inputs : <@Main<[]>>, !array.type<2 x !pod.type<[@in: !felt.type]>>
      struct.writem %self[@drain] = %destStruct : <@Main<[]>>, !array.type<2 x !struct.type<@Sub<[]>>>
      function.return %self : !struct.type<@Main<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
