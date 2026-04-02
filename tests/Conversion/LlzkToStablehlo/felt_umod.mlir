// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test felt.umod and felt.uintdiv lowering.
// Both convert to integer, perform the op, and convert back to field.

// CHECK-NOT: felt.umod
// CHECK-NOT: felt.uintdiv

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.convert
// CHECK: stablehlo.remainder
// CHECK: stablehlo.convert
// CHECK: stablehlo.convert
// CHECK: stablehlo.divide
// CHECK: stablehlo.convert
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@FullAdder<[]>>} {
  struct.def @FullAdder<[]> {
    struct.member @val : !felt.type {llzk.pub}
    struct.member @carry : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@FullAdder<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@FullAdder<[]>>
      %sum = felt.add %a, %b : !felt.type, !felt.type
      %two = felt.const 2
      %val = felt.umod %sum, %two : !felt.type, !felt.type
      struct.writem %self[@val] = %val : <@FullAdder<[]>>, !felt.type
      %carry = felt.uintdiv %sum, %two : !felt.type, !felt.type
      struct.writem %self[@carry] = %carry : <@FullAdder<[]>>, !felt.type
      function.return %self : !struct.type<@FullAdder<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@FullAdder<[]>>, %a: !felt.type, %b: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
