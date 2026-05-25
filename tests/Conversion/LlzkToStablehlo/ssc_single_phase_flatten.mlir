// RUN: llzk-to-shlo-opt --simplify-sub-components="test-phase=flattenPodArrayWhileCarry" %s | FileCheck %s

// Task 5 (arch-hardening): isolated-phase test for the SSC `test-phase`
// option. With `test-phase=flattenPodArrayWhileCarry` the pass runs ONLY that
// one phase entry point on each function body and returns, instead of the full
// fixed-point pipeline. This pins `flattenPodArrayWhileCarry`'s documented
// contract (PodArrayWhileCarry.h) in isolation:
//
//   Precondition: a block contains an `scf.while` carrying an
//     `array<N x !pod<[@f: felt, ...]>>` iter-arg whose pod record list is
//     non-empty and uniformly felt-flattenable.
//   Postcondition: that slot is split into per-field `array<N x felt>` carries
//     on the while; no pod-typed carry remains.
//
// This fixture would FAIL without the new option: the bare phase name in the
// `--simplify-sub-components="test-phase=..."` flag would not parse (the pass
// had no options before Task 5), so `llzk-to-shlo-opt` would error out and the
// FileCheck would never see a flattened `scf.while`.

module attributes {llzk.lang} {
  struct.def @Main_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Main_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_0<[]>>
      %array = array.new : <2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      %felt0 = felt.const 0
      // scf.while carrying the array-of-pods iter-arg. flattenPodArrayWhileCarry
      // must split this into two per-field `array<2 x felt>` carries.
      %0 = scf.while (%carry = %array) : (!array.type<2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>) -> !array.type<2 x !pod.type<[@a: !felt.type, @b: !felt.type]>> {
        %felt1 = felt.const 1
        %cond = bool.cmp lt(%felt0, %felt1)
        scf.condition(%cond) %carry : !array.type<2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      } do {
      ^bb0(%carry: !array.type<2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>):
        %idx = cast.toindex %felt0
        %a = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %dp = array.read %carry[%idx] : <2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !pod.type<[@a: !felt.type, @b: !felt.type]>
        pod.write %dp[@a] = %a : <[@a: !felt.type, @b: !felt.type]>, !felt.type
        pod.write %dp[@b] = %a : <[@a: !felt.type, @b: !felt.type]>, !felt.type
        array.write %carry[%idx] = %dp : <2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !pod.type<[@a: !felt.type, @b: !felt.type]>
        scf.yield %carry : !array.type<2 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      }
      struct.writem %self[@out] = %arg0 : <@Main_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_0<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}

// The while survives (only this one phase ran — no dispatch elimination, no
// template removal), but its pod-array carry is gone: the scf.while now carries
// two per-field `array<2 x felt>` slots, the body uses direct
// `array.read`/`array.write`, and no `pod.read`/`pod.write` against the carry
// remains.
//
// CHECK-LABEL: function.def @compute
// Postcondition proper: no pod-typed carry survives on the while.
// CHECK-NOT: scf.while {{.*}}!pod.type
// The flattened while carries two felt-array slots, not the pod-array.
// CHECK: scf.while {{.*}}!array.type<2 x !felt.type>, !array.type<2 x !felt.type>
// CHECK: scf.condition
// Body access is direct array I/O — no pod traffic survives on the carry.
// CHECK: array.read
// CHECK: array.write
// CHECK-NOT: pod.read
// CHECK-NOT: pod.write
// CHECK: scf.yield
