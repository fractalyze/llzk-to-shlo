// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test array operation patterns: array.new, array.read, array.write.

// CHECK-NOT: array.new
// CHECK-NOT: array.read
// CHECK-NOT: array.write

// CHECK-LABEL: func.func @main
// array.new → stablehlo.constant dense<0>
// CHECK: dense<0>
// array.write → stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// array.write → stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// array.read → stablehlo.dynamic_slice + reshape (result used in writem)
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.reshape
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@ArrayOpsTest<[]>>} {
  struct.def @ArrayOpsTest {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@ArrayOpsTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ArrayOpsTest<[]>>

      // array.new: zero-initialized 4-element array
      %arr = array.new : <4 x !felt.type>

      // array.write at index 0
      %idx0 = felt.const 0
      %i0 = cast.toindex %idx0
      array.write %arr[%i0] = %arg0 : <4 x !felt.type>, !felt.type

      // array.write at index 1
      %idx1 = felt.const 1
      %i1 = cast.toindex %idx1
      array.write %arr[%i1] = %arg1 : <4 x !felt.type>, !felt.type

      // array.read: read back element and use it (prevents DCE)
      %val = array.read %arr[%i0] : <4 x !felt.type>, !felt.type
      struct.writem %self[@out] = %val : <@ArrayOpsTest<[]>>, !felt.type

      function.return %self : !struct.type<@ArrayOpsTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ArrayOpsTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
