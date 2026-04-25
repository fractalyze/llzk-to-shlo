// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test struct with array member. struct.writem with array value exercises
// the multi-dimensional flatten path in StructWriteMPattern.

// CHECK-NOT: struct.writem
// CHECK-LABEL: func.func @main
// struct.new → zero-initialized flattened tensor (1 scalar + 3 array = 4)
// CHECK: dense<0> : tensor<4xi32>
// struct.writem @x (scalar) → reshape + dynamic_update_slice at offset 0
// CHECK: stablehlo.dynamic_update_slice
// struct.writem @arr (1D array) → dynamic_update_slice at offset 1
// CHECK: dense<1>
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@StructArr<[]>>} {
  struct.def @StructArr {
    struct.member @x : !felt.type {llzk.pub}
    struct.member @arr : !array.type<3 x !felt.type> {llzk.pub}
    function.def @compute(%a: !felt.type, %v: !array.type<3 x !felt.type>) -> !struct.type<@StructArr<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@StructArr<[]>>
      struct.writem %self[@x] = %a : <@StructArr<[]>>, !felt.type
      struct.writem %self[@arr] = %v : <@StructArr<[]>>, !array.type<3 x !felt.type>
      function.return %self : !struct.type<@StructArr<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@StructArr<[]>>, %arg1: !felt.type, %arg2: !array.type<3 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
