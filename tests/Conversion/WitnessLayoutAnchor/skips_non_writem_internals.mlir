// RUN: llzk-to-shlo-opt --witness-layout-anchor %s | FileCheck %s

// Internal `struct.member` declarations that have no `struct.writem` in
// the main struct's body must be skipped — they correspond to the
// dispatch scaffolding that `--simplify-sub-components` later erases.
// `{llzk.pub}` outputs are emitted unconditionally regardless of writem
// presence (their write happens via the `@compute` return value chain,
// not always via a literal `struct.writem` against `%self`).

module attributes {llzk.lang, llzk.main = !struct.type<@Outputs::@Outputs<[]>>} {
  // CHECK-LABEL: module
  // CHECK: wla.layout signals = [
  // CHECK-SAME: #wla.signal<"const_one", internal, offset = 0, length = 1>
  // CHECK-SAME: #wla.signal<"@a", output, offset = 1, length = 1>
  // CHECK-SAME: #wla.signal<"@b", output, offset = 2, length = 2>
  // CHECK-SAME: #wla.signal<"%arg0", input, offset = 4, length = 1>
  // CHECK-SAME: ]
  // The internal `@unwritten` member must be ABSENT from the layout
  // entries — checked on the layout op line itself, not the struct.def
  // declaration where the member name appears for unrelated reasons.
  // CHECK-NOT: signal<"@unwritten"
  module @Outputs {
    struct.def @Outputs {
      struct.member @a : !felt.type<"bn128"> {llzk.pub}
      struct.member @b : !array.type<2 x !felt.type<"bn128">> {llzk.pub}
      struct.member @unwritten : !felt.type<"bn128">
      function.def @compute(%arg0: !felt.type<"bn128">)
          -> !struct.type<@Outputs::@Outputs<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@Outputs::@Outputs<[]>>
        function.return %self : !struct.type<@Outputs::@Outputs<[]>>
      }
      function.def @constrain(
          %arg0: !struct.type<@Outputs::@Outputs<[]>>,
          %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
}
