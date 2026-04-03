// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test that cast.toindex preserves constant values from felt.const.
// Previously, bare index values fell through to a fallback that always
// emitted dense<0>, causing array.write to wrong indices.

// CHECK-NOT: cast.toindex

// CHECK-LABEL: func.func @main
//
// First write: index 0
// CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[C0]]
//
// Second write: index 1 (not 0!)
// CHECK: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[C1]]

module attributes {llzk.lang, llzk.main = !struct.type<@TwoElem<[]>>} {
  struct.def @TwoElem<[]> {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@TwoElem<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@TwoElem<[]>>
      %arr = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = felt.const 0
      %i0 = cast.toindex %c0
      array.write %arr[%i0] = %a : <2 x !felt.type>, !felt.type
      %c1 = felt.const 1
      %i1 = cast.toindex %c1
      array.write %arr[%i1] = %b : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %arr : <@TwoElem<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@TwoElem<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@TwoElem<[]>>, %a: !felt.type, %b: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
