// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Verify felt.inv produces divide(1, x) with constant 1 as FIRST operand.
// This is semantically important: inv(x) = 1/x, not x/1.

// CHECK-LABEL: func.func @main
// CHECK: %[[ONE:.*]] = stablehlo.constant{{.*}}dense<1>
// CHECK: stablehlo.divide %[[ONE]], %arg0
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@InvSem<[]>>} {
  struct.def @InvSem<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%x: !felt.type) -> !struct.type<@InvSem<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@InvSem<[]>>
      %inv = felt.inv %x : !felt.type
      struct.writem %self[@out] = %inv : <@InvSem<[]>>, !felt.type
      function.return %self : !struct.type<@InvSem<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@InvSem<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
