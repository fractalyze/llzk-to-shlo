// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for `unpackPodWhileCarry`: the post-while pod result is
// consumed by an enclosing `scf.while`'s `scf.yield` (not handleable
// by `replacePostWhilePodUsers`, which only addresses pod.read /
// pod.write / struct.writem, nor by the chained-while branch, which
// only handles scf.while users). Without the use-shape gate, the
// transformed inner scf.while is erased while the outer `scf.yield`
// still references its old pod result, tripping `use_empty()` inside
// `Operation::~Operation`.
//
// Shape: nested scf.while where the inner pod result is yielded by
// the outer's body terminator at a non-pod-carry slot of the outer.
//
// Without the gate the dbg/asan build aborts inside
// `unpackPodWhileCarry:whileOp->erase()`. With the gate the transform
// skips the inner while (the outer's yield use isn't a recognized
// shape) and the pass completes.

// Without the gate `Operation::~Operation` asserts on the leftover
// scf.yield use of the inner pod result and the function body never
// reaches stdout. The two scf.while matches below confirm both
// nested loops survive pass-completion.
// CHECK-LABEL: struct.def @Main_1
// CHECK: scf.while
// CHECK: scf.while
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Main_1 {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !felt.type<"bn128">
      %pod_init = pod.new : <[@a: !felt.type<"bn128">]>
      %felt_zero = felt.const 0 : <"bn128">
      %felt_one = felt.const 1 : <"bn128">
      %limit = felt.const 4 : <"bn128">

      // Outer scf.while carries the pod across iterations. The body
      // contains an inner scf.while whose pod result feeds the
      // outer's scf.yield.
      %0:2 = scf.while (%outer_iter = %felt_zero, %outer_pod = %pod_init)
          : (!felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>)
          -> (!felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>) {
        %outer_cond = bool.cmp lt(%outer_iter, %limit) : !felt.type<"bn128">, !felt.type<"bn128">
        scf.condition(%outer_cond) %outer_iter, %outer_pod : !felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>
      } do {
      ^bb_outer(%outer_iter: !felt.type<"bn128">, %outer_pod: !pod.type<[@a: !felt.type<"bn128">]>):
        // Inner scf.while: also pod-carrying, but its uses inside the
        // body ARE handleable (only pod.read / pod.write). The
        // unpacker COULD unpack the inner if its post-while uses were
        // also handleable. They aren't — the outer scf.yield below
        // consumes %inner#1 (pod result) at slot 1, which isn't a
        // pod.read / pod.write / struct.writem / scf.while shape.
        %inner:2 = scf.while (%i_iter = %felt_zero, %i_pod = %outer_pod)
            : (!felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>)
            -> (!felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>) {
          %inner_cond = bool.cmp lt(%i_iter, %limit) : !felt.type<"bn128">, !felt.type<"bn128">
          scf.condition(%inner_cond) %i_iter, %i_pod : !felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>
        } do {
        ^bb_inner(%i_iter: !felt.type<"bn128">, %i_pod: !pod.type<[@a: !felt.type<"bn128">]>):
          %old = pod.read %i_pod[@a] : <[@a: !felt.type<"bn128">]>, !felt.type<"bn128">
          %new = felt.add %old, %felt_one : !felt.type<"bn128">, !felt.type<"bn128">
          pod.write %i_pod[@a] = %new : <[@a: !felt.type<"bn128">]>, !felt.type<"bn128">
          %i_next = felt.add %i_iter, %felt_one : !felt.type<"bn128">, !felt.type<"bn128">
          scf.yield %i_next, %i_pod : !felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>
        }

        // Offending use: the outer's scf.yield consumes %inner#1
        // (post-while pod result of the inner scf.while). This is
        // not handleable by either replacePostWhilePodUsers or the
        // chained-while branch.
        %outer_next = felt.add %outer_iter, %felt_one : !felt.type<"bn128">, !felt.type<"bn128">
        scf.yield %outer_next, %inner#1 : !felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>
      }

      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !felt.type<"bn128">
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
      function.return
    }
  }
}
