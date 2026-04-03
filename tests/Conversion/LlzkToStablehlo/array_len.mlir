// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test array.len conversion. Circom arrays are statically sized, so
// array.len resolves at compile time. The result is used as a valid
// array index (len - 1 = last element).

// CHECK-NOT: array.len
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.reshape
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@LenTest<[]>>} {
  struct.def @LenTest<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arr: !array.type<5 x !felt.type>) -> !struct.type<@LenTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@LenTest<[]>>
      %c0 = arith.constant 0 : index
      // array.len resolves to 5 at compile time
      %len = array.len %arr, %c0 : <5 x !felt.type>
      // len - 1 = 4, a valid index for the last element
      %c1 = arith.constant 1 : index
      %last = arith.subi %len, %c1 : index
      %val = array.read %arr[%last] : <5 x !felt.type>, !felt.type
      struct.writem %self[@out] = %val : <@LenTest<[]>>, !felt.type
      function.return %self : !struct.type<@LenTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@LenTest<[]>>, %arg1: !array.type<5 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
