// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: a 2D `!array<N,M x !felt>` allocated at function scope,
// captured by an outer scf.while, mutated inside a result-bearing
// scf.if (a slot of the if yields `llzk.nondet` placeholder of the
// matching array type — the LLZK `<--` pattern). The write inside the
// branch is `array.insert %arr[%idx] = %row` whose first operand is the
// captured 2D carrier (rewritten to a body-arg by
// promoteArraysToWhileCarry). After lowering, the per-iteration write
// MUST surface as a `dynamic_update_slice %iterArg` inside the outer
// stablehlo.while; otherwise the carry is dead and the final value of
// `%arr` (consumed by struct.writem after the loop) lowers to a
// dense<0> constant. Mirrors webb_poseidon_vanchor's
// @Poseidon_137_compute slot-5 dead-carry.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// The 2D carrier slot must take a dynamic_update_slice writeback at
// some iter-arg position (the slot position depends on the carry
// ordering after promoteArraysToWhileCarry).
// CHECK: dynamic_update_slice %iterArg

module attributes {llzk.lang, llzk.main = !struct.type<@MixCarry2D<[]>>} {
  struct.def @MixCarry2D {
    struct.member @out : !array.type<3,2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@MixCarry2D<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@MixCarry2D<[]>>
      // Outer-scope 2D array — the captured carrier mirroring %array_67.
      %mix = array.new : <3,2 x !felt.type>
      // %nondet acts as the placeholder yielded in the scf.if's array slot
      // (LLZK `<--` shape).
      %nondet_init = llzk.nondet : !array.type<3,2 x !felt.type>
      %f0 = felt.const  0 : !felt.type
      %f1 = felt.const  1 : !felt.type
      %f2 = felt.const  2 : !felt.type
      %f3 = felt.const  3 : !felt.type
      // Outer scf.while: counter %i = 0..3 plus a 2D nondet carrier
      // %carry — mirrors the outer 68-iter Transaction while at LLZK
      // line 1650 with %arg3:!array<68,5> = %nondet_513.
      %0:2 = scf.while (%i = %f0, %carry = %nondet_init) : (!felt.type, !array.type<3,2 x !felt.type>) -> (!felt.type, !array.type<3,2 x !felt.type>) {
        %lt = bool.cmp lt(%i, %f3) : !felt.type, !felt.type
        scf.condition(%lt) %i, %carry : !felt.type, !array.type<3,2 x !felt.type>
      } do {
      ^bb0(%i: !felt.type, %carry: !array.type<3,2 x !felt.type>):
        %eq0 = bool.cmp eq(%i, %f0) : !felt.type, !felt.type
        // Result-bearing scf.if whose array slot yields %nondet on the
        // then branch and the actual write on the else branch.
        // BOTH branches yield the same-typed %nondet at the array slot —
        // post-pass extendResultBearingScfIfArrayChain must append a NEW
        // tail slot threading the array.insert through.
        %if_res:2 = scf.if %eq0 -> (!felt.type, !array.type<3,2 x !felt.type>) {
          %nd_then = llzk.nondet : !array.type<3,2 x !felt.type>
          scf.yield %i, %nd_then : !felt.type, !array.type<3,2 x !felt.type>
        } else {
          // Mutate the captured outer 2D array %mix via array.insert.
          // Build the inserted row from %arg0 (2 felts).
          %row = llzk.nondet : !array.type<2 x !felt.type>
          %z = cast.toindex %f0 : !felt.type
          %v0 = array.read %arg0[%z] : <2 x !felt.type>, !felt.type
          array.write %row[%z] = %v0 : <2 x !felt.type>, !felt.type
          %idx = cast.toindex %i : !felt.type
          array.insert %mix[%idx] = %row : <3,2 x !felt.type>, <2 x !felt.type>
          %nd_else = llzk.nondet : !array.type<3,2 x !felt.type>
          scf.yield %i, %nd_else : !felt.type, !array.type<3,2 x !felt.type>
        }
        %i1 = felt.add %i, %f1 : !felt.type, !felt.type
        scf.yield %i1, %carry : !felt.type, !array.type<3,2 x !felt.type>
      }
      // Post-loop read of the captured 2D — the new (appended) scf.while
      // result must replace this reference.
      struct.writem %self[@out] = %mix : <@MixCarry2D<[]>>, !array.type<3,2 x !felt.type>
      function.return %self : !struct.type<@MixCarry2D<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@MixCarry2D<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
