// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test array.new with element arguments (concatenation path).
// When elements are provided, array.new reshapes each to [1,...],
// concatenates, then reshapes to the final array type.

// CHECK-NOT: array.new
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.reshape
// CHECK: stablehlo.reshape
// CHECK: stablehlo.concatenate
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@NewElems<[]>>} {
  struct.def @NewElems {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@NewElems<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@NewElems<[]>>
      %arr = array.new %a, %b : <2 x !felt.type>
      struct.writem %self[@out] = %arr : <@NewElems<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@NewElems<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@NewElems<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
