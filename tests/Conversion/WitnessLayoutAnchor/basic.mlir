// RUN: llzk-to-shlo-opt --witness-layout-anchor %s | FileCheck %s

// Smallest LLZK shape the anchor pass cares about: one main struct
// (named via `llzk.main`) with one `{llzk.pub}` output member, one
// non-pub internal member written via `struct.writem`, and a
// `@compute` taking two function-arg inputs. The struct lives inside a
// named `builtin.module` shell which mirrors the post-`simplify-sub-components`
// IR shape the pass actually consumes.

module attributes {llzk.lang, llzk.main = !struct.type<@Toy::@Toy<[]>>} {
  // CHECK-LABEL: module
  // CHECK: wla.layout signals = [
  // CHECK-SAME: #wla.signal<"const_one", internal, offset = 0, length = 1>
  // CHECK-SAME: #wla.signal<"@out", output, offset = 1, length = 4>
  // CHECK-SAME: #wla.signal<"%arg0", input, offset = 5, length = 1>
  // CHECK-SAME: #wla.signal<"%arg1", input, offset = 6, length = 8>
  // CHECK-SAME: #wla.signal<"@scratch", internal, offset = 14, length = 2>
  // CHECK-SAME: ]
  module @Toy {
    struct.def @Toy {
      struct.member @out : !array.type<4 x !felt.type<"bn128">> {llzk.pub}
      struct.member @scratch : !array.type<2 x !felt.type<"bn128">>
      function.def @compute(
          %arg0: !felt.type<"bn128">,
          %arg1: !array.type<8 x !felt.type<"bn128">>) -> !struct.type<@Toy::@Toy<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@Toy::@Toy<[]>>
        %v = llzk.nondet : !array.type<2 x !felt.type<"bn128">>
        struct.writem %self[@scratch] = %v
          : <@Toy::@Toy<[]>>, !array.type<2 x !felt.type<"bn128">>
        function.return %self : !struct.type<@Toy::@Toy<[]>>
      }
      function.def @constrain(
          %arg0: !struct.type<@Toy::@Toy<[]>>,
          %arg1: !felt.type<"bn128">,
          %arg2: !array.type<8 x !felt.type<"bn128">>) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
}
