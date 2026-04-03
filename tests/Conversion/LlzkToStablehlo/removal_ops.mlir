// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test removal/erase patterns: cast.toindex, constrain.eq erasure,
// and @constrain function erasure.

// Verify all constraint-related ops are erased
// CHECK-NOT: constrain.eq
// CHECK-NOT: cast.toindex
// No @constrain function should remain
// CHECK-NOT: func.func @{{.*}}constrain

// cast.toindex → stablehlo.convert (field → i32 index tensor)
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.convert {{.*}} tensor<i32>
// CHECK: stablehlo.dynamic_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@RemovalTest<[]>>} {
  struct.def @RemovalTest<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %arr: !array.type<4 x !felt.type>) -> !struct.type<@RemovalTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@RemovalTest<[]>>

      // cast.toindex → stablehlo.convert, then use as array index
      %idx = cast.toindex %a
      %val = array.read %arr[%idx] : <4 x !felt.type>, !felt.type

      struct.writem %self[@out] = %val : <@RemovalTest<[]>>, !felt.type
      function.return %self : !struct.type<@RemovalTest<[]>>
    }
    // @constrain function should be entirely erased
    function.def @constrain(%arg0: !struct.type<@RemovalTest<[]>>, %arg1: !felt.type, %arg2: !array.type<4 x !felt.type>) attributes {function.allow_constraint} {
      %x = struct.readm %arg0[@out] : <@RemovalTest<[]>>, !felt.type
      constrain.eq %x, %arg1 : !felt.type
      function.return
    }
  }
}
