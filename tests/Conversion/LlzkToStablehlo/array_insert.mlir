// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test array.insert conversion: array.insert inserts a sub-array into a 2D
// array. The destination is an array.new (the case that actually exercises
// convertWritemToSSA — function-parameter destinations are short-circuited
// by the pod-skip guard because convertAllFunctions has already converted
// the parameter type to tensor by the time convertWritemToSSA runs).
//
// The CHECK pattern asserts data-flow correctness: the read of the dest
// array after the insert MUST come from the insert's dynamic_update_slice
// result, not from the original empty array.new (which lowers to dense<0>).
// Without that strictness, the silent-drop bug (orphan dynamic_update_slice
// when array.insert is void) would pass the test.

// CHECK-NOT: array.insert
// CHECK-LABEL: func.func @main
// array.new lowers to a dense<0> tensor of the 2D shape.
// CHECK: %[[EMPTY:.*]] = stablehlo.constant{{.*}}dense<0> : tensor<4x3xi32>
// array.insert lowers to dynamic_update_slice writing the row into the
// empty matrix. Its dest must be the empty matrix (no other source exists).
// CHECK: %[[INSERTED:.*]] = stablehlo.dynamic_update_slice %[[EMPTY]],
// struct.writem flattens %dest into the output member. Its source must
// trace back to the insert's chain tip via a reshape, not to the original
// empty matrix — otherwise the void array.insert lowers to an orphan
// dynamic_update_slice that DCE drops, and the reshape consumes %[[EMPTY]].
// CHECK: %[[FLAT:.*]] = stablehlo.reshape %[[INSERTED]]
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %[[FLAT]]

module attributes {llzk.lang, llzk.main = !struct.type<@InsertTest<[]>>} {
  struct.def @InsertTest {
    struct.member @out : !array.type<4,3 x !felt.type> {llzk.pub}
    function.def @compute(%row: !array.type<3 x !felt.type>, %idx_felt: !felt.type) -> !struct.type<@InsertTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@InsertTest<[]>>
      %dest = array.new : <4,3 x !felt.type>
      %idx = cast.toindex %idx_felt
      array.insert %dest[%idx] = %row : <4,3 x !felt.type>, <3 x !felt.type>
      struct.writem %self[@out] = %dest : <@InsertTest<[]>>, !array.type<4,3 x !felt.type>
      function.return %self : !struct.type<@InsertTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@InsertTest<[]>>, %arg1: !array.type<3 x !felt.type>, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
