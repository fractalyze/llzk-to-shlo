// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test all 6 bool.cmp predicates. Each comparison is used as a while loop
// condition to prevent dead code elimination.

// CHECK-NOT: bool.cmp

// CHECK-LABEL: func.func @main

// CHECK: stablehlo.compare EQ
// CHECK: stablehlo.compare NE
// CHECK: stablehlo.compare LT
// CHECK: stablehlo.compare LE
// CHECK: stablehlo.compare GT
// CHECK: stablehlo.compare GE

module attributes {llzk.lang, llzk.main = !struct.type<@CmpTest<[]>>} {
  struct.def @CmpTest<[]> {
    struct.member @out_eq : !felt.type {llzk.pub}
    struct.member @out_ne : !felt.type {llzk.pub}
    struct.member @out_lt : !felt.type {llzk.pub}
    struct.member @out_le : !felt.type {llzk.pub}
    struct.member @out_gt : !felt.type {llzk.pub}
    struct.member @out_ge : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@CmpTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@CmpTest<[]>>
      %zero = felt.const 0
      %one = felt.const 1

      // EQ predicate
      %r_eq = scf.while (%i = %zero) : (!felt.type) -> !felt.type {
        %c = bool.cmp eq(%i, %a)
        scf.condition(%c) %i : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %n = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %n : !felt.type
      }
      struct.writem %self[@out_eq] = %r_eq : <@CmpTest<[]>>, !felt.type

      // NE predicate
      %r_ne = scf.while (%i = %zero) : (!felt.type) -> !felt.type {
        %c = bool.cmp ne(%i, %a)
        scf.condition(%c) %i : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %n = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %n : !felt.type
      }
      struct.writem %self[@out_ne] = %r_ne : <@CmpTest<[]>>, !felt.type

      // LT predicate
      %r_lt = scf.while (%i = %zero) : (!felt.type) -> !felt.type {
        %c = bool.cmp lt(%i, %a)
        scf.condition(%c) %i : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %n = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %n : !felt.type
      }
      struct.writem %self[@out_lt] = %r_lt : <@CmpTest<[]>>, !felt.type

      // LE predicate
      %r_le = scf.while (%i = %zero) : (!felt.type) -> !felt.type {
        %c = bool.cmp le(%i, %a)
        scf.condition(%c) %i : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %n = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %n : !felt.type
      }
      struct.writem %self[@out_le] = %r_le : <@CmpTest<[]>>, !felt.type

      // GT predicate
      %r_gt = scf.while (%i = %zero) : (!felt.type) -> !felt.type {
        %c = bool.cmp gt(%i, %a)
        scf.condition(%c) %i : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %n = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %n : !felt.type
      }
      struct.writem %self[@out_gt] = %r_gt : <@CmpTest<[]>>, !felt.type

      // GE predicate
      %r_ge = scf.while (%i = %zero) : (!felt.type) -> !felt.type {
        %c = bool.cmp ge(%i, %a)
        scf.condition(%c) %i : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %n = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %n : !felt.type
      }
      struct.writem %self[@out_ge] = %r_ge : <@CmpTest<[]>>, !felt.type

      function.return %self : !struct.type<@CmpTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@CmpTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
