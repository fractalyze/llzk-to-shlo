// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test bool.cmp conversion to stablehlo.compare with multiple predicates.
// Results are used as scf.while conditions to prevent dead code elimination.

// CHECK-NOT: bool.cmp

// CHECK-LABEL: func.func @main

// LT predicate inside while condition
// CHECK: stablehlo.compare LT

module attributes {llzk.lang, llzk.main = !struct.type<@BoolTest<[]>>} {
  struct.def @BoolTest<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %limit: !felt.type) -> !struct.type<@BoolTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@BoolTest<[]>>
      %zero = felt.const 0
      %one = felt.const 1

      // Use bool.cmp lt in a while loop (result is used as condition)
      %result = scf.while (%iter = %zero) : (!felt.type) -> !felt.type {
        %cond = bool.cmp lt(%iter, %limit)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%arg: !felt.type):
        %next = felt.add %arg, %one : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }

      struct.writem %self[@out] = %result : <@BoolTest<[]>>, !felt.type
      function.return %self : !struct.type<@BoolTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@BoolTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
