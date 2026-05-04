// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: `collapseRedundantWhileCarrierPairs` must NOT collapse a
// passthrough/live carrier pair at an inner stablehlo.while when an
// enclosing while's slot for the same carrier is mutated. Without the
// "every enclosing slot is passthrough" guard, the inner-level RAUW
// silently rewrites all uses of the dead-result to the live-result —
// then the outer body's yield gets two identical SSA values at distinct
// slots, dropping all data flow on the dead-collapsed slot.
//
// This is the AES `aes_256_encrypt` `xor_2[i][j][k].a/.b` shape — both
// fields are modified at the main while level, but at the inner k×l×m
// nest only `.a` is written each iteration while `.b` is read and
// passes through. The bottom-up pass would otherwise collapse `.b` into
// `.a` at every inner while, then the main while ends up yielding
// `<.a-result>, <.a-result>` at the slots intended to hold .a and .b.
//
// CHECK-LABEL: func.func @main
// CHECK: %0:3 = stablehlo.while
// Inside outer body, the inner stablehlo.while is where the bad collapse
// would have fired (.b is passthrough at this level only).
// CHECK: %[[INNER:.+]]:3 = stablehlo.while
// Inner-while body must keep .a-modified and .b-passthrough at distinct
// slots — the bug-shape was the inner while's slot 2 result being
// RAUW'd into slot 1's result, then the outer DUS that reads slot 2
// would silently read from slot 1 instead.
// CHECK: stablehlo.return %{{.+}}, %{{.+}}, %iterArg_{{[0-9]+}} : tensor<!pf_babybear>, tensor<4x!pf_babybear>, tensor<4x!pf_babybear>
// After the inner while, the outer body MUST DUS into %INNER#2 (the
// .b passthrough), NOT %INNER#1 (the .a result). This is the
// load-bearing assertion for the fix.
// CHECK: stablehlo.dynamic_update_slice %[[INNER]]#2,
// Outer body terminator yields counter+1, then .a (= INNER#1), then .b
// (= the DUS-of-INNER#2 above) — all three distinct values.
// CHECK: stablehlo.return %{{.+}}, %[[INNER]]#1, %{{.+}} : tensor<!pf_babybear>, tensor<4x!pf_babybear>, tensor<4x!pf_babybear>

module attributes {llzk.lang, llzk.main = !struct.type<@PairedCarriers<[]>>} {
  struct.def @PairedCarriers {
    struct.member @a : !array.type<4 x !felt.type> {llzk.pub}
    struct.member @b : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<4 x !felt.type>) -> !struct.type<@PairedCarriers<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@PairedCarriers<[]>>
      %a = llzk.nondet : !array.type<4 x !felt.type>
      %b = llzk.nondet : !array.type<4 x !felt.type>
      %c0 = felt.const  0 : !felt.type
      %c1 = felt.const  1 : !felt.type
      %c2 = felt.const  2 : !felt.type
      // Outer while: i = 0..2. Modifies BOTH `a` (via inner while) and
      // `b` (directly at outer level). This is the load-bearing shape
      // that triggers the bad collapse on the inner while.
      %0 = scf.while (%i = %c0) : (!felt.type) -> !felt.type {
        %lt = bool.cmp lt(%i, %c2) : !felt.type, !felt.type
        scf.condition(%lt) %i : !felt.type
      } do {
      ^bb0(%i: !felt.type):
        // Inner while: j = 0..2. Reads `b[i+j]` and writes `a[i+j] = a + b`.
        // `b` is passthrough at this level (read-only).
        %1 = scf.while (%j = %c0) : (!felt.type) -> !felt.type {
          %lt2 = bool.cmp lt(%j, %c2) : !felt.type, !felt.type
          scf.condition(%lt2) %j : !felt.type
        } do {
        ^bb0(%j: !felt.type):
          %ij = felt.add %i, %j : !felt.type, !felt.type
          %idx = cast.toindex %ij : !felt.type
          %va = array.read %a[%idx] : <4 x !felt.type>, !felt.type
          %vb = array.read %b[%idx] : <4 x !felt.type>, !felt.type
          %sum = felt.add %va, %vb : !felt.type, !felt.type
          array.write %a[%idx] = %sum : <4 x !felt.type>, !felt.type
          %jinc = felt.add %j, %c1 : !felt.type, !felt.type
          scf.yield %jinc : !felt.type
        }
        // Outer-level write to `b` makes the outer's `.b` slot non-passthrough.
        // Without this, the carrier IS truly always-zero-init-passthrough at
        // every level and the collapse is genuinely safe.
        %i_idx = cast.toindex %i : !felt.type
        array.write %b[%i_idx] = %i : <4 x !felt.type>, !felt.type
        %iinc = felt.add %i, %c1 : !felt.type, !felt.type
        scf.yield %iinc : !felt.type
      }
      struct.writem %self[@a] = %a : <@PairedCarriers<[]>>, !array.type<4 x !felt.type>
      struct.writem %self[@b] = %b : <@PairedCarriers<[]>>, !array.type<4 x !felt.type>
      function.return %self : !struct.type<@PairedCarriers<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@PairedCarriers<[]>>, %arg1: !array.type<4 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
