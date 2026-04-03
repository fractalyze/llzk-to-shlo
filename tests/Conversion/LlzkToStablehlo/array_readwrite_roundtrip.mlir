// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Verify array write-then-read roundtrip: write a value at an index,
// read from the same index, and the result feeds into the return.

// CHECK-LABEL: func.func @main
// array.new → zero tensor
// CHECK: %[[ARR:.*]] = stablehlo.constant{{.*}}dense<0> : tensor<4xi32>
// array.write → dynamic_update_slice on %[[ARR]]
// CHECK: %[[WROTE:.*]] = stablehlo.dynamic_update_slice %[[ARR]]
// array.read → dynamic_slice from the written array
// CHECK: stablehlo.dynamic_slice %[[WROTE]]
// CHECK: stablehlo.reshape
// read result used in struct.writem
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@ArrRoundTrip<[]>>} {
  struct.def @ArrRoundTrip<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%val: !felt.type) -> !struct.type<@ArrRoundTrip<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ArrRoundTrip<[]>>
      %arr = array.new : <4 x !felt.type>
      %idx_felt = felt.const 2
      %idx = cast.toindex %idx_felt
      array.write %arr[%idx] = %val : <4 x !felt.type>, !felt.type
      %read = array.read %arr[%idx] : <4 x !felt.type>, !felt.type
      struct.writem %self[@out] = %read : <@ArrRoundTrip<[]>>, !felt.type
      function.return %self : !struct.type<@ArrRoundTrip<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ArrRoundTrip<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
