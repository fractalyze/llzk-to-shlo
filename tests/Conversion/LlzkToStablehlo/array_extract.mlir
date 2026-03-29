// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test: array.extract converts to stablehlo.dynamic_slice + reshape.
// array.extract reads a sub-array (row) from a 2D array.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.reshape
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@ExtractTest<[]>>} {
  struct.def @ExtractTest<[]> {
    struct.member @out : !array.type<3 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<4,3 x !felt.type>, %arg1: !felt.type) -> !struct.type<@ExtractTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ExtractTest<[]>>
      %idx = cast.toindex %arg1
      %row = array.extract %arg0[%idx] : <4,3 x !felt.type>
      struct.writem %self[@out] = %row : <@ExtractTest<[]>>, !array.type<3 x !felt.type>
      function.return %self : !struct.type<@ExtractTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ExtractTest<[]>>, %arg1: !array.type<4,3 x !felt.type>, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
