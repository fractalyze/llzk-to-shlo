// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Verify struct field offset constants are correct.
// Struct layout: @a (offset 0), @b (offset 1), @arr (offset 2, size 3).
// Flattened size = 1 + 1 + 3 = 5.

// CHECK-LABEL: func.func @main
// struct.new → flattened tensor with correct size
// CHECK: dense<0> : tensor<5xi32>
// writem @a → DUS at offset 0
// CHECK: %[[OFF_A:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[OFF_A]]
// writem @b → DUS at offset 1
// CHECK: %[[OFF_B:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[OFF_B]]
// writem @arr → DUS at offset 2
// CHECK: %[[OFF_C:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[OFF_C]]
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@OffsetTest<[]>>} {
  struct.def @OffsetTest<[]> {
    struct.member @a : !felt.type {llzk.pub}
    struct.member @b : !felt.type {llzk.pub}
    struct.member @arr : !array.type<3 x !felt.type> {llzk.pub}
    function.def @compute(%x: !felt.type, %y: !felt.type, %v: !array.type<3 x !felt.type>) -> !struct.type<@OffsetTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@OffsetTest<[]>>
      struct.writem %self[@a] = %x : <@OffsetTest<[]>>, !felt.type
      struct.writem %self[@b] = %y : <@OffsetTest<[]>>, !felt.type
      struct.writem %self[@arr] = %v : <@OffsetTest<[]>>, !array.type<3 x !felt.type>
      function.return %self : !struct.type<@OffsetTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@OffsetTest<[]>>, %arg1: !felt.type, %arg2: !felt.type, %arg3: !array.type<3 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
