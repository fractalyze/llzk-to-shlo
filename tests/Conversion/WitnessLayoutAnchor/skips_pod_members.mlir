// RUN: llzk-to-shlo-opt --witness-layout-anchor %s | FileCheck %s

// `*$inputs` pod-typed scaffolding members emitted by circom for
// sub-component input collection are NOT real circom signals — they get
// erased by `--simplify-sub-components`. The anchor pass must drop both
// scalar `!pod.type<...>` and `!array<... x !pod.type>` members so the
// emitted layout matches the post-simplify member set the lowering
// registers offsets for.

module attributes {llzk.lang, llzk.main = !struct.type<@Mixed::@Mixed<[]>>} {
  // The pod scratch members must be ABSENT from the emitted layout.
  // CHECK-LABEL: module
  // CHECK: wla.layout signals = [
  // CHECK-SAME: #wla.signal<"const_one", internal, offset = 0, length = 1>
  // CHECK-SAME: #wla.signal<"@out", output, offset = 1, length = 1>
  // CHECK-SAME: #wla.signal<"%arg0", input, offset = 2, length = 1>
  // CHECK-SAME: #wla.signal<"@kept", internal, offset = 3, length = 3>
  // CHECK-SAME: ]
  // The pod scratch members must be ABSENT from the layout entries —
  // checked on the layout op line itself, not the struct.def declaration
  // where the member names appear for unrelated reasons.
  // CHECK-NOT: signal<"@pod_scratch"
  // CHECK-NOT: signal<"@pod_array_scratch"
  module @Mixed {
    struct.def @Mixed {
      struct.member @out : !felt.type<"bn128"> {llzk.pub}
      struct.member @kept : !array.type<3 x !felt.type<"bn128">>
      struct.member @pod_scratch : !pod.type<[@a: !felt.type<"bn128">]>
      struct.member @pod_array_scratch
          : !array.type<2 x !pod.type<[@b: !felt.type<"bn128">]>>
      function.def @compute(%arg0: !felt.type<"bn128">)
          -> !struct.type<@Mixed::@Mixed<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@Mixed::@Mixed<[]>>
        %v = llzk.nondet : !array.type<3 x !felt.type<"bn128">>
        struct.writem %self[@kept] = %v
          : <@Mixed::@Mixed<[]>>, !array.type<3 x !felt.type<"bn128">>
        function.return %self : !struct.type<@Mixed::@Mixed<[]>>
      }
      function.def @constrain(
          %arg0: !struct.type<@Mixed::@Mixed<[]>>,
          %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
}
