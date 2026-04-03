// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test felt.inv conversion to stablehlo.divide(1, x).

// CHECK-NOT: felt.inv
// CHECK-LABEL: func.func @main
// CHECK: dense<1>
// CHECK: stablehlo.divide
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@InvTest<[]>>} {
  struct.def @InvTest<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@InvTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@InvTest<[]>>
      %result = felt.inv %arg0 : !felt.type
      struct.writem %self[@out] = %result : <@InvTest<[]>>, !felt.type
      function.return %self : !struct.type<@InvTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@InvTest<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
