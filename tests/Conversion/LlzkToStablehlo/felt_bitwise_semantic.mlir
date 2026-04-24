// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Verify felt.shr conversion chain: field → int → shift → field.
// The convert ops must have correct type signatures.

// CHECK-LABEL: func.func @main
// Convert both operands from field to integer
// CHECK: %[[A_INT:.*]] = stablehlo.convert %arg0 : (tensor<!pf{{.*}}>) -> tensor<i32>
// CHECK: %[[B_INT:.*]] = stablehlo.convert %arg1 : (tensor<!pf{{.*}}>) -> tensor<i32>
// Perform integer shift on converted values
// CHECK: %[[SHIFT:.*]] = stablehlo.shift_right_logical %[[A_INT]], %[[B_INT]] : tensor<i32>
// Convert result back to field
// CHECK: stablehlo.convert %[[SHIFT]] : (tensor<i32>) -> tensor<!pf{{.*}}>
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@ShrSem<[]>>} {
  struct.def @ShrSem {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@ShrSem<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ShrSem<[]>>
      %result = felt.shr %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %result : <@ShrSem<[]>>, !felt.type
      function.return %self : !struct.type<@ShrSem<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ShrSem<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
