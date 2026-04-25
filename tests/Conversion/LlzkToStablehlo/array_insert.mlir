// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test array.insert conversion to stablehlo.dynamic_update_slice.
// array.insert inserts a sub-array (row) into a 2D array.

// CHECK-NOT: array.insert
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.reshape
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@InsertTest<[]>>} {
  struct.def @InsertTest {
    struct.member @out : !array.type<4,3 x !felt.type> {llzk.pub}
    function.def @compute(%dest: !array.type<4,3 x !felt.type>, %row: !array.type<3 x !felt.type>, %idx_felt: !felt.type) -> !struct.type<@InsertTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@InsertTest<[]>>
      %idx = cast.toindex %idx_felt
      array.insert %dest[%idx] = %row : <4,3 x !felt.type>, <3 x !felt.type>
      struct.writem %self[@out] = %dest : <@InsertTest<[]>>, !array.type<4,3 x !felt.type>
      function.return %self : !struct.type<@InsertTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@InsertTest<[]>>, %arg1: !array.type<4,3 x !felt.type>, %arg2: !array.type<3 x !felt.type>, %arg3: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
