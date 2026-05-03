// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for the post-flatten parent-child rewire pass when the parent
// scf.while flattens TWO pod-array carriers of DIFFERENT shapes that are
// threaded into a single child scf.while as adjacent iter-args. AES rounds-
// loop is the canonical case: `<13,4,3,32 x !pod<[@a,@b]>>` (post-SubBytes
// MC-coeff bits) and `<13,4,32 x !pod<[@a,@b]>>` (post-MC bit-decomp) are
// both threaded through `%98:9` → `%102:8` and end up as 4 adjacent nondets
// in the post-flatten child's init list with a heterogeneous type sequence
// `<13,4,3,32>×2 + <13,4,32>×2`.
//
// Before the fix, the rewire pass at `SimplifySubComponents.cpp` (post-flatten
// rewire) tried to match the entire 4-element run as one contiguous run
// against parent body args, which never matched because the parent has
// these types at non-contiguous positions [0,1] + [8,9]. The rewire silently
// no-op'd, leaving the child's pod-array iter-args wired to local nondets
// that lower to const-zero. Downstream XOR(0, RK_bit) propagated the zero
// state through every iteration of the parent loop.
//
// The fix breaks the contiguous nondet scan on type transition, splitting
// into homogeneous-type sub-runs that each match their own parent slot
// pair. After the fix, the child's slot 0/1 wires to parent body args
// matching `<3 x !felt>×2` (post-flatten of the first pod-array) and slot
// 2/3 wires to body args matching `<5 x !felt>×2` (post-flatten of the
// second).

// The lowered child's init list must reference parent body block args, not
// fresh local nondets, for both type groups. Pod-array carriers are gone.
// CHECK-LABEL: function.def @compute
// CHECK: %[[OUTER:.*]]:5 = scf.while
// CHECK-SAME: !array.type<3 x !felt
// CHECK-SAME: !array.type<5 x !felt
// CHECK: ^bb0
// CHECK-SAME: %[[BA1:[^:]+]]: !array.type<3 x !felt
// CHECK-SAME: %[[BA2:[^:]+]]: !array.type<3 x !felt
// CHECK-SAME: %[[BA3:[^:]+]]: !array.type<5 x !felt
// CHECK-SAME: %[[BA4:[^:]+]]: !array.type<5 x !felt
// CHECK: scf.while (%{{[^=]*}} = %[[BA1]], %{{[^=]*}} = %[[BA2]], %{{[^=]*}} = %[[BA3]], %{{[^=]*}} = %[[BA4]]
// CHECK-NOT: pod.read
// CHECK-NOT: pod.write
// CHECK-NOT: pod.new
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@HeteroFlatten<[]>>} {
  struct.def @HeteroFlatten {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<3 x !felt.type>, %arg1: !array.type<3 x !felt.type>, %arg2: !array.type<5 x !felt.type>, %arg3: !array.type<5 x !felt.type>) -> !struct.type<@HeteroFlatten<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@HeteroFlatten<[]>>
      %arr3 = array.new : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      %arr5 = array.new : <5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      %felt0 = felt.const 0
      // Outer while: carries both pod-arrays. flattenPodArrayWhileCarry
      // expands each into 2 per-field felt arrays (4 total new slots).
      %0:3 = scf.while (%i = %felt0, %a3 = %arr3, %a5 = %arr5) : (!felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>) -> (!felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>) {
        %felt1 = felt.const 1
        %cond = bool.cmp lt(%i, %felt1)
        scf.condition(%cond) %i, %a3, %a5 : !felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      } do {
      ^bb0(%i: !felt.type, %a3: !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, %a5: !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>):
        %felt0_1 = felt.const 0
        // Inner while: threads both pod-arrays as adjacent iter-args. After
        // flatten, slots 0..3 become 4 adjacent nondets of types
        // <3>×2 + <5>×2 — the heterogeneous run that defeats the original
        // single-run matcher.
        %1:3 = scf.while (%j3 = %a3, %j5 = %a5, %k = %felt0_1) : (!array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type) -> (!array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type) {
          %felt3 = felt.const 3
          %cond = bool.cmp lt(%k, %felt3)
          scf.condition(%cond) %j3, %j5, %k : !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type
        } do {
        ^bb0(%j3: !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, %j5: !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, %k: !felt.type):
          %idx3 = cast.toindex %k
          %elem3 = array.read %j3[%idx3] : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !pod.type<[@a: !felt.type, @b: !felt.type]>
          %a_val3 = array.read %arg0[%idx3] : <3 x !felt.type>, !felt.type
          pod.write %elem3[@a] = %a_val3 : <[@a: !felt.type, @b: !felt.type]>, !felt.type
          %b_val3 = array.read %arg1[%idx3] : <3 x !felt.type>, !felt.type
          pod.write %elem3[@b] = %b_val3 : <[@a: !felt.type, @b: !felt.type]>, !felt.type
          array.write %j3[%idx3] = %elem3 : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !pod.type<[@a: !felt.type, @b: !felt.type]>

          %elem5 = array.read %j5[%idx3] : <5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !pod.type<[@a: !felt.type, @b: !felt.type]>
          %a_val5 = array.read %arg2[%idx3] : <5 x !felt.type>, !felt.type
          pod.write %elem5[@a] = %a_val5 : <[@a: !felt.type, @b: !felt.type]>, !felt.type
          %b_val5 = array.read %arg3[%idx3] : <5 x !felt.type>, !felt.type
          pod.write %elem5[@b] = %b_val5 : <[@a: !felt.type, @b: !felt.type]>, !felt.type
          array.write %j5[%idx3] = %elem5 : <5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !pod.type<[@a: !felt.type, @b: !felt.type]>

          %felt1 = felt.const 1
          %k_next = felt.add %k, %felt1 : !felt.type, !felt.type
          scf.yield %j3, %j5, %k_next : !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type
        }
        %felt1 = felt.const 1
        %i_next = felt.add %i, %felt1 : !felt.type, !felt.type
        scf.yield %i_next, %1#0, %1#1 : !felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !array.type<5 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      }
      struct.writem %self[@out] = %felt0 : <@HeteroFlatten<[]>>, !felt.type
      function.return %self : !struct.type<@HeteroFlatten<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@HeteroFlatten<[]>>, %arg1: !array.type<3 x !felt.type>, %arg2: !array.type<3 x !felt.type>, %arg3: !array.type<5 x !felt.type>, %arg4: !array.type<5 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
