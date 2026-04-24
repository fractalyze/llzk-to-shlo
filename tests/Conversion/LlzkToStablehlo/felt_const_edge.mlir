// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test felt.const with edge values: 0 and 1.

// CHECK-NOT: felt.const
// CHECK-LABEL: func.func @main
// CHECK-DAG: dense<0>
// CHECK-DAG: dense<1>
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@ConstEdge<[]>>} {
  struct.def @ConstEdge {
    struct.member @zero_out : !felt.type {llzk.pub}
    struct.member @one_out : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@ConstEdge<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ConstEdge<[]>>
      %zero = felt.const 0
      struct.writem %self[@zero_out] = %zero : <@ConstEdge<[]>>, !felt.type
      %one = felt.const 1
      struct.writem %self[@one_out] = %one : <@ConstEdge<[]>>, !felt.type
      function.return %self : !struct.type<@ConstEdge<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ConstEdge<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
