// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for `unpackPodWhileCarry`: the pod-typed scf.while block
// arg is consumed by an op that isn't `pod.read` / `pod.write` / the
// body terminator at the expand position. `expandWhileRegionArgs`'s
// `replacePodOpsOnValue` only rewires pod.read / pod.write uses, and
// `expandTerminatorArg` only substitutes one terminator slot. Any
// other use of the pod block arg survives the rewires; the subsequent
// `blk.eraseArgument` then trips `use_empty()` because the
// BlockArgument still has a live user.
//
// Shape: an `scf.if` inside the while body yields the pod block arg
// back as its result, leaving an scf.yield reference to the pod arg
// that neither `replacePodOpsOnValue` nor `expandTerminatorArg`
// addresses.
//
// Without the use-shape gate the dbg/asan build aborts inside
// `expandWhileRegionArgs:269` (Block::eraseArgument). With the gate
// the transform bails, the IR is left untouched for this while, and
// the pass completes — the chip would surface a downstream
// `pod.new not yet supported` error or similar, not a crash.

// Without the gate the pass aborts inside `Block::eraseArgument` and
// the function body never reaches stdout. The scf.while + scf.if
// matches below distinguish pass-completion from silent body-drop.
// CHECK-LABEL: struct.def @Main_1
// CHECK: scf.while
// CHECK: scf.if
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
      %true = arith.constant true

      // Single-pod-carry scf.while. The pod block arg is yielded by an
      // inner scf.if (back to itself), creating a scf.yield use of the
      // pod arg that lives outside the body terminator's expand slot.
      // `unpackPodWhileCarry` cannot rewire it; the use-shape gate
      // bails here.
      %0:2 = scf.while (%iter = %felt_zero, %pod_iter = %pod_init)
          : (!felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>)
          -> (!felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>) {
        %cond = bool.cmp lt(%iter, %limit) : !felt.type<"bn128">, !felt.type<"bn128">
        scf.condition(%cond) %iter, %pod_iter : !felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>
      } do {
      ^bb0(%iter: !felt.type<"bn128">, %pod_iter: !pod.type<[@a: !felt.type<"bn128">]>):
        %old = pod.read %pod_iter[@a] : <[@a: !felt.type<"bn128">]>, !felt.type<"bn128">
        %new = felt.add %old, %felt_one : !felt.type<"bn128">, !felt.type<"bn128">
        pod.write %pod_iter[@a] = %new : <[@a: !felt.type<"bn128">]>, !felt.type<"bn128">

        // Nested scf.if whose yields reference %pod_iter directly —
        // the unhandleable use that wedges the unpacker without the
        // gate.
        %routed_pod = scf.if %true -> (!pod.type<[@a: !felt.type<"bn128">]>) {
          scf.yield %pod_iter : !pod.type<[@a: !felt.type<"bn128">]>
        } else {
          scf.yield %pod_iter : !pod.type<[@a: !felt.type<"bn128">]>
        }

        %next = felt.add %iter, %felt_one : !felt.type<"bn128">, !felt.type<"bn128">
        scf.yield %next, %routed_pod : !felt.type<"bn128">, !pod.type<[@a: !felt.type<"bn128">]>
      }

      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !felt.type<"bn128">
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
      function.return
    }
  }
}
