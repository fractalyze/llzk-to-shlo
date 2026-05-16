// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: `collapseRedundantWhileCarrierPairs` must NOT redirect a
// DEAD scalar-passthrough slot's external uses onto a sibling LIVE scalar
// slot just because both share `tensor<!pf>` type and zero-init.
//
// Shape: a Poseidon3-style `@compute` whose enclosing scf.while threads
// (a) an iteration counter (LIVE, init=0, yields counter+1), (b) an
// input array being copied (LIVE), and (c) an immutable capacity-init
// scalar (DEAD passthrough, init=0, yields the unchanged body arg).
// Slots (a) and (c) are both zero-init `tensor<!pf>` scalars; without
// the LIVE.yield → DEAD body-arg link guard, the pass would pair them
// and RAUW post-loop uses of (c) to (a)'s result. Downstream uses of
// the capacity then observe the final counter value (= N) instead of 0
// — silently miscompiling any consumer reading the capacity slot.
//
// CHECK-LABEL: func.func @main
// CHECK: %[[W:.+]]:3 = stablehlo.while
// The body yields (counter+1, updated-input, capacity-passthrough). The
// third slot's yield must be the body argument unchanged.
// CHECK: stablehlo.return %{{.+}}, %{{.+}}, %iterArg{{[_0-9]*}} : tensor<!pf_babybear>, tensor<3x!pf_babybear>, tensor<!pf_babybear>
// Post-loop: the @capacity output store MUST flow %W#2 (the unchanged
// zero-init) into its DUS, not %W#0 (the post-loop counter = N). Without
// the LIVE.yield → DEAD body-arg link guard, the previous behavior was
// `stablehlo.reshape %W#0` here, silently writing the counter into the
// capacity output slot.
// CHECK: %[[CAP:.+]] = stablehlo.reshape %[[W]]#2 : (tensor<!pf_babybear>) -> tensor<1x!pf_babybear>
// CHECK: stablehlo.dynamic_update_slice %{{.+}}, %[[CAP]]

module attributes {llzk.lang, llzk.main = !struct.type<@ScalarPassthrough<[]>>} {
  struct.def @ScalarPassthrough {
    struct.member @inputs : !array.type<3 x !felt.type> {llzk.pub}
    struct.member @capacity : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<3 x !felt.type>) -> !struct.type<@ScalarPassthrough<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ScalarPassthrough<[]>>
      %c0 = felt.const  0 : !felt.type
      %c1 = felt.const  1 : !felt.type
      %c3 = felt.const  3 : !felt.type
      %nondet_in = llzk.nondet : !array.type<3 x !felt.type>
      // While: 3 iter-args of types (felt, <3 x felt>, felt). The
      // first is a counter (LIVE), the second is an array-copy carrier
      // (LIVE), the third is a passthrough capacity (DEAD).
      %r:3 = scf.while (%i = %c0, %buf = %nondet_in, %cap = %c0) : (!felt.type, !array.type<3 x !felt.type>, !felt.type) -> (!felt.type, !array.type<3 x !felt.type>, !felt.type) {
        %lt = bool.cmp lt(%i, %c3) : !felt.type, !felt.type
        scf.condition(%lt) %i, %buf, %cap : !felt.type, !array.type<3 x !felt.type>, !felt.type
      } do {
      ^bb0(%i: !felt.type, %buf: !array.type<3 x !felt.type>, %cap: !felt.type):
        %idx = cast.toindex %i : !felt.type
        %v = array.read %arg0[%idx] : <3 x !felt.type>, !felt.type
        array.write %buf[%idx] = %v : <3 x !felt.type>, !felt.type
        %iinc = felt.add %i, %c1 : !felt.type, !felt.type
        scf.yield %iinc, %buf, %cap : !felt.type, !array.type<3 x !felt.type>, !felt.type
      }
      struct.writem %self[@inputs] = %r#1 : <@ScalarPassthrough<[]>>, !array.type<3 x !felt.type>
      // Load-bearing assertion: %r#2 must remain the unchanged capacity
      // (post-loop value = init = 0), not collapsed to %r#0 (counter = 3).
      struct.writem %self[@capacity] = %r#2 : <@ScalarPassthrough<[]>>, !felt.type
      function.return %self : !struct.type<@ScalarPassthrough<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ScalarPassthrough<[]>>, %arg1: !array.type<3 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
