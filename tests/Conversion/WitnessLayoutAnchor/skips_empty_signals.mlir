// RUN: llzk-to-shlo-opt --witness-layout-anchor %s | FileCheck %s

// Empty sub-component members (writem-targeted struct types with no
// non-pod members of their own) have `flat_size = 0`. The `wla.layout`
// op verifier rejects length <= 0, so the anchor must drop these at
// emit time. The skipped signal contributes 0 to the running offset so
// subsequent signals stay aligned with `@main`'s chunk layout.

// CHECK-LABEL: module
// CHECK: wla.layout signals = [
// CHECK-SAME: #wla.signal<"const_one", internal, offset = 0, length = 1>
// CHECK-SAME: #wla.signal<"@out", output, offset = 1, length = 2>
// CHECK-SAME: #wla.signal<"%arg0", input, offset = 3, length = 1>
// CHECK-NOT: "@empty_sub"
// CHECK-SAME: ]

module attributes {llzk.lang, llzk.main = !struct.type<@WithEmpty::@WithEmpty<[]>>} {
  module @EmptySub {
    struct.def @EmptySub {
      function.def @compute() -> !struct.type<@EmptySub::@EmptySub<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@EmptySub::@EmptySub<[]>>
        function.return %self : !struct.type<@EmptySub::@EmptySub<[]>>
      }
      function.def @constrain(%arg0: !struct.type<@EmptySub::@EmptySub<[]>>) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
  module @WithEmpty {
    struct.def @WithEmpty {
      struct.member @out : !array.type<2 x !felt.type<"bn128">> {llzk.pub}
      struct.member @empty_sub : !struct.type<@EmptySub::@EmptySub<[]>>
      function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@WithEmpty::@WithEmpty<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@WithEmpty::@WithEmpty<[]>>
        %sub = llzk.nondet : !struct.type<@EmptySub::@EmptySub<[]>>
        struct.writem %self[@empty_sub] = %sub
          : <@WithEmpty::@WithEmpty<[]>>, !struct.type<@EmptySub::@EmptySub<[]>>
        function.return %self : !struct.type<@WithEmpty::@WithEmpty<[]>>
      }
      function.def @constrain(
          %arg0: !struct.type<@WithEmpty::@WithEmpty<[]>>,
          %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
}
