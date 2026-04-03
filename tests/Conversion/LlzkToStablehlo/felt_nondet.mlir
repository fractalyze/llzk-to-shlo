// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test llzk.nondet conversion to stablehlo.constant(0).

// CHECK-NOT: llzk.nondet

// CHECK-LABEL: func.func @main
// llzk.nondet produces a zero-initialized array
// CHECK: dense<0>
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@NonDetTest<[]>>} {
  struct.def @NonDetTest<[]> {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute() -> !struct.type<@NonDetTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@NonDetTest<[]>>
      %arr = llzk.nondet : !array.type<4 x !felt.type>
      struct.writem %self[@out] = %arr : <@NonDetTest<[]>>, !array.type<4 x !felt.type>
      function.return %self : !struct.type<@NonDetTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@NonDetTest<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
