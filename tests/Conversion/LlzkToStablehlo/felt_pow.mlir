// UNSUPPORTED: true
// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test felt.pow conversion to stablehlo.power.
// felt.pow requires function.def with function.allow_non_native_field_ops.
//
// Disabled: open-zkx's stablehlo.power verifier rejects heterogeneous types
// (field base + i32 exponent). Requires upstream fix in open-zkx.

// CHECK-NOT: felt.pow
// CHECK-LABEL: func.func @main
// CHECK: stablehlo.power
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@PowTest<[]>>} {
  struct.def @PowTest {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%base: !felt.type, %exp: !felt.type) -> !struct.type<@PowTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@PowTest<[]>>
      %result = felt.pow %base, %exp : !felt.type, !felt.type
      struct.writem %self[@out] = %result : <@PowTest<[]>>, !felt.type
      function.return %self : !struct.type<@PowTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@PowTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
