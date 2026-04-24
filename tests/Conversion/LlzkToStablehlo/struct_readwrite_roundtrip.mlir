// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Verify struct write-then-read roundtrip: the value written to a field
// can be read back, and the slice offset matches the write offset.

// CHECK-LABEL: func.func @main
// struct.new → zero tensor (2 fields)
// CHECK: %[[SELF:.*]] = stablehlo.constant{{.*}}dense<0> : tensor<2xi32>
// writem @x at offset 0
// CHECK: %[[UPD:.*]] = stablehlo.dynamic_update_slice %[[SELF]]
// readm @x → slice [0:1] from the updated tensor
// CHECK: stablehlo.slice %[[UPD]] [0:1]
// CHECK: stablehlo.reshape
// writem @y at offset 1, using the read result
// CHECK: dense<1> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %[[UPD]]
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@RoundTrip<[]>>} {
  struct.def @RoundTrip {
    struct.member @x : !felt.type {llzk.pub}
    struct.member @y : !felt.type {llzk.pub}
    function.def @compute(%val: !felt.type) -> !struct.type<@RoundTrip<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@RoundTrip<[]>>
      struct.writem %self[@x] = %val : <@RoundTrip<[]>>, !felt.type
      %read = struct.readm %self[@x] : <@RoundTrip<[]>>, !felt.type
      struct.writem %self[@y] = %read : <@RoundTrip<[]>>, !felt.type
      function.return %self : !struct.type<@RoundTrip<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@RoundTrip<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
