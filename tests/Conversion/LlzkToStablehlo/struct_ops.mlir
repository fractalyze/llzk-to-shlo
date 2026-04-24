// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test struct operation patterns: struct.new, struct.readm, struct.writem.
// struct.new → zero-initialized flattened tensor
// struct.readm → stablehlo.slice + reshape
// struct.writem → stablehlo.dynamic_update_slice

// CHECK-NOT: struct.new
// CHECK-NOT: struct.readm
// CHECK-NOT: struct.writem
// CHECK-NOT: struct.def
// CHECK-NOT: struct.member

// CHECK-LABEL: func.func @main
// struct.new → stablehlo.constant dense<0>
// CHECK: dense<0>
// struct.writem @x → stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// struct.writem @y → stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// struct.readm @x → stablehlo.slice + reshape
// CHECK: stablehlo.slice
// CHECK: stablehlo.reshape
// struct.writem @z (using readm result) → stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@StructOpsTest<[]>>} {
  struct.def @StructOpsTest {
    struct.member @x : !felt.type {llzk.pub}
    struct.member @y : !felt.type {llzk.pub}
    struct.member @z : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@StructOpsTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@StructOpsTest<[]>>
      struct.writem %self[@x] = %a : <@StructOpsTest<[]>>, !felt.type
      struct.writem %self[@y] = %b : <@StructOpsTest<[]>>, !felt.type
      // Read back @x so it's not dead code
      %read_x = struct.readm %self[@x] : <@StructOpsTest<[]>>, !felt.type
      // Use readm result in writem to prevent DCE
      struct.writem %self[@z] = %read_x : <@StructOpsTest<[]>>, !felt.type
      function.return %self : !struct.type<@StructOpsTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@StructOpsTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
