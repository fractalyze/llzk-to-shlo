// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Test: flattenPodArrayWhileCarry converts array-of-pods with felt fields
// into per-field felt arrays. The while carry changes from pod array to
// individual felt arrays.

// Verify pod.read/pod.write/pod.new are eliminated from compute function.
// struct.member declarations may still reference pod.type.
// CHECK-LABEL: function.def @compute
// CHECK-NOT: pod.read
// CHECK-NOT: pod.write
// CHECK-NOT: pod.new
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@PodFlatten<[]>>} {
  struct.def @PodFlatten<[]> {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @inputs : !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !array.type<2 x !felt.type>) -> !struct.type<@PodFlatten<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@PodFlatten<[]>>
      %arr = array.new : <2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>
      %felt0 = felt.const 0
      %0:2 = scf.while (%arg2 = %felt0, %arg3 = %arr) : (!felt.type, !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>) -> (!felt.type, !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>) {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%arg2, %felt2)
        scf.condition(%cond) %arg2, %arg3 : !felt.type, !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>
      } do {
      ^bb0(%arg2: !felt.type, %arg3: !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>):
        %idx = cast.toindex %arg2
        %elem = array.read %arg3[%idx] : <2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>, !pod.type<[@x: !felt.type, @y: !felt.type]>
        %xval = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        pod.write %elem[@x] = %xval : <[@x: !felt.type, @y: !felt.type]>, !felt.type
        %yval = array.read %arg1[%idx] : <2 x !felt.type>, !felt.type
        pod.write %elem[@y] = %yval : <[@x: !felt.type, @y: !felt.type]>, !felt.type
        array.write %arg3[%idx] = %elem : <2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>, !pod.type<[@x: !felt.type, @y: !felt.type]>
        %felt1 = felt.const 1
        %next = felt.add %arg2, %felt1 : !felt.type, !felt.type
        scf.yield %next, %arg3 : !felt.type, !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>
      }
      %felt0_1 = felt.const 0
      struct.writem %self[@out] = %felt0_1 : <@PodFlatten<[]>>, !felt.type
      struct.writem %self[@inputs] = %0#1 : <@PodFlatten<[]>>, !array.type<2 x !pod.type<[@x: !felt.type, @y: !felt.type]>>
      function.return %self : !struct.type<@PodFlatten<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@PodFlatten<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
