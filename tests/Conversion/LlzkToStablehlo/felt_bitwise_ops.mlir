// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test felt bitwise and unsigned integer operations.
// These convert field → integer, perform the integer op, convert back.

// CHECK-NOT: felt.shr
// CHECK-NOT: felt.bit_and
// CHECK-NOT: felt.umod
// CHECK-NOT: felt.uintdiv

// --- felt.shr: field → convert(i32) → shift_right_logical → convert(field)
// CHECK-LABEL: func.func @ShrTest_compute
// CHECK: stablehlo.convert
// CHECK: stablehlo.convert
// CHECK: stablehlo.shift_right_logical
// CHECK: stablehlo.convert
// CHECK: return

// --- felt.bit_and: field → convert(i32) → and → convert(field)
// CHECK-LABEL: func.func @BitAndTest_compute
// CHECK: stablehlo.convert
// CHECK: stablehlo.convert
// CHECK: stablehlo.and
// CHECK: stablehlo.convert
// CHECK: return

// --- felt.umod: field → convert(i32) → remainder → convert(field)
// CHECK-LABEL: func.func @UmodTest_compute
// CHECK: stablehlo.convert
// CHECK: stablehlo.convert
// CHECK: stablehlo.remainder
// CHECK: stablehlo.convert
// CHECK: return

// --- felt.uintdiv: field → convert(i32) → divide → convert(field)
// CHECK-LABEL: func.func @UintdivTest_compute
// CHECK: stablehlo.convert
// CHECK: stablehlo.convert
// CHECK: stablehlo.divide
// CHECK: stablehlo.convert
// CHECK: return

// --- Top-level calls all sub-components
// CHECK-LABEL: func.func @main
// CHECK: call @ShrTest_compute
// CHECK: call @BitAndTest_compute
// CHECK: call @UmodTest_compute
// CHECK: call @UintdivTest_compute
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Top<[]>>} {

  struct.def @ShrTest {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@ShrTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ShrTest<[]>>
      %result = felt.shr %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %result : <@ShrTest<[]>>, !felt.type
      function.return %self : !struct.type<@ShrTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@ShrTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @BitAndTest {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@BitAndTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@BitAndTest<[]>>
      %result = felt.bit_and %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %result : <@BitAndTest<[]>>, !felt.type
      function.return %self : !struct.type<@BitAndTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@BitAndTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @UmodTest {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@UmodTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@UmodTest<[]>>
      %result = felt.umod %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %result : <@UmodTest<[]>>, !felt.type
      function.return %self : !struct.type<@UmodTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@UmodTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @UintdivTest {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@UintdivTest<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@UintdivTest<[]>>
      %result = felt.uintdiv %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %result : <@UintdivTest<[]>>, !felt.type
      function.return %self : !struct.type<@UintdivTest<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@UintdivTest<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @Top {
    struct.member @shr : !struct.type<@ShrTest<[]>>
    struct.member @bitand : !struct.type<@BitAndTest<[]>>
    struct.member @umod : !struct.type<@UmodTest<[]>>
    struct.member @uintdiv : !struct.type<@UintdivTest<[]>>
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@Top<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Top<[]>>
      %s0 = function.call @ShrTest::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@ShrTest<[]>>
      struct.writem %self[@shr] = %s0 : <@Top<[]>>, !struct.type<@ShrTest<[]>>
      %s1 = function.call @BitAndTest::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@BitAndTest<[]>>
      struct.writem %self[@bitand] = %s1 : <@Top<[]>>, !struct.type<@BitAndTest<[]>>
      %s2 = function.call @UmodTest::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@UmodTest<[]>>
      struct.writem %self[@umod] = %s2 : <@Top<[]>>, !struct.type<@UmodTest<[]>>
      %s3 = function.call @UintdivTest::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@UintdivTest<[]>>
      struct.writem %self[@uintdiv] = %s3 : <@Top<[]>>, !struct.type<@UintdivTest<[]>>
      function.return %self : !struct.type<@Top<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Top<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
