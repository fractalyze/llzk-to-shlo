// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Sub-component test: GreaterThan(32) with pod read-modify-write pattern.
// Tests SimplifySubComponents extracting function.call from scf.if dispatch.

// Verify no pod/scf.if operations remain
// CHECK-NOT: pod.
// CHECK-NOT: scf.if

// CHECK-LABEL: func.func @main
// CHECK: call @LessThan_1_compute
// CHECK: return

// CHECK-LABEL: func.func @LessThan_1_compute
// CHECK: call @Num2Bits_0_compute
// CHECK: return

// CHECK-LABEL: func.func @Num2Bits_0_compute
// CHECK: stablehlo.while
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@GreaterThan_2<[]>>} {
  struct.def @GreaterThan_2 {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @lt : !struct.type<@LessThan_1<[]>>
    struct.member @lt$inputs : !pod.type<[@in: !array.type<2 x !felt.type>]>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@GreaterThan_2<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@GreaterThan_2<[]>>
      %c2 = arith.constant 2 : index
      %pod = pod.new { @count = %c2 }  : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>
      %pod_0 = pod.new : <[@in: !array.type<2 x !felt.type>]>
      %felt_const_32 = felt.const  32
      %felt_const_1 = felt.const  1
      %0 = cast.toindex %felt_const_1
      %1 = array.read %arg0[%0] : <2 x !felt.type>, !felt.type
      %2 = pod.read %pod_0[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
      %felt_const_0 = felt.const  0
      %3 = cast.toindex %felt_const_0
      array.write %2[%3] = %1 : <2 x !felt.type>, !felt.type
      pod.write %pod_0[@in] = %2 : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
      %4 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, index
      %c1 = arith.constant 1 : index
      %5 = arith.subi %4, %c1 : index
      pod.write %pod[@count] = %5 : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, index
      %c0 = arith.constant 0 : index
      %6 = arith.cmpi eq, %5, %c0 : index
      scf.if %6 {
        %17 = pod.read %pod_0[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %18 = function.call @LessThan_1::@compute(%17) : (!array.type<2 x !felt.type>) -> !struct.type<@LessThan_1<[]>>
        pod.write %pod[@comp] = %18 : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, !struct.type<@LessThan_1<[]>>
      } else {
      }
      %felt_const_0_1 = felt.const  0
      %7 = cast.toindex %felt_const_0_1
      %8 = array.read %arg0[%7] : <2 x !felt.type>, !felt.type
      %9 = pod.read %pod_0[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
      %felt_const_1_2 = felt.const  1
      %10 = cast.toindex %felt_const_1_2
      array.write %9[%10] = %8 : <2 x !felt.type>, !felt.type
      pod.write %pod_0[@in] = %9 : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
      %11 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, index
      %c1_3 = arith.constant 1 : index
      %12 = arith.subi %11, %c1_3 : index
      pod.write %pod[@count] = %12 : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, index
      %c0_4 = arith.constant 0 : index
      %13 = arith.cmpi eq, %12, %c0_4 : index
      scf.if %13 {
        %17 = pod.read %pod_0[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %18 = function.call @LessThan_1::@compute(%17) : (!array.type<2 x !felt.type>) -> !struct.type<@LessThan_1<[]>>
        pod.write %pod[@comp] = %18 : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, !struct.type<@LessThan_1<[]>>
      } else {
      }
      %14 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, !struct.type<@LessThan_1<[]>>
      %15 = struct.readm %14[@out] : <@LessThan_1<[]>>, !felt.type
      struct.writem %self[@out] = %15 : <@GreaterThan_2<[]>>, !felt.type
      struct.writem %self[@lt$inputs] = %pod_0 : <@GreaterThan_2<[]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
      %16 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@LessThan_1<[]>>, @params: !pod.type<[]>]>, !struct.type<@LessThan_1<[]>>
      struct.writem %self[@lt] = %16 : <@GreaterThan_2<[]>>, !struct.type<@LessThan_1<[]>>
      function.return %self : !struct.type<@GreaterThan_2<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@GreaterThan_2<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      function.return
    }
  }
  struct.def @LessThan_1 {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @n2b : !struct.type<@Num2Bits_0<[]>>
    struct.member @n2b$inputs : !pod.type<[@in: !felt.type]>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@LessThan_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@LessThan_1<[]>>
      %c1 = arith.constant 1 : index
      %pod = pod.new { @count = %c1 }  : <[@count: index, @comp: !struct.type<@Num2Bits_0<[]>>, @params: !pod.type<[]>]>
      %pod_0 = pod.new : <[@in: !felt.type]>
      %felt_const_0_1 = felt.const  0
      %1 = cast.toindex %felt_const_0_1
      %2 = array.read %arg0[%1] : <2 x !felt.type>, !felt.type
      %felt_const_4294967296 = felt.const  4294967296
      %3 = felt.add %2, %felt_const_4294967296 : !felt.type, !felt.type
      %felt_const_1_2 = felt.const  1
      %4 = cast.toindex %felt_const_1_2
      %5 = array.read %arg0[%4] : <2 x !felt.type>, !felt.type
      %6 = felt.sub %3, %5 : !felt.type, !felt.type
      pod.write %pod_0[@in] = %6 : <[@in: !felt.type]>, !felt.type
      %7 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Num2Bits_0<[]>>, @params: !pod.type<[]>]>, index
      %c1_3 = arith.constant 1 : index
      %8 = arith.subi %7, %c1_3 : index
      pod.write %pod[@count] = %8 : <[@count: index, @comp: !struct.type<@Num2Bits_0<[]>>, @params: !pod.type<[]>]>, index
      %c0 = arith.constant 0 : index
      %9 = arith.cmpi eq, %8, %c0 : index
      scf.if %9 {
        %16 = pod.read %pod_0[@in] : <[@in: !felt.type]>, !felt.type
        %17 = function.call @Num2Bits_0::@compute(%16) : (!felt.type) -> !struct.type<@Num2Bits_0<[]>>
        pod.write %pod[@comp] = %17 : <[@count: index, @comp: !struct.type<@Num2Bits_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Num2Bits_0<[]>>
      } else {
      }
      %felt_const_1_4 = felt.const  1
      %10 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@Num2Bits_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Num2Bits_0<[]>>
      %11 = struct.readm %10[@out] : <@Num2Bits_0<[]>>, !array.type<33 x !felt.type>
      %felt_const_32_5 = felt.const  32
      %12 = cast.toindex %felt_const_32_5
      %13 = array.read %11[%12] : <33 x !felt.type>, !felt.type
      %14 = felt.sub %felt_const_1_4, %13 : !felt.type, !felt.type
      struct.writem %self[@out] = %14 : <@LessThan_1<[]>>, !felt.type
      struct.writem %self[@n2b$inputs] = %pod_0 : <@LessThan_1<[]>>, !pod.type<[@in: !felt.type]>
      %15 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@Num2Bits_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Num2Bits_0<[]>>
      struct.writem %self[@n2b] = %15 : <@LessThan_1<[]>>, !struct.type<@Num2Bits_0<[]>>
      function.return %self : !struct.type<@LessThan_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@LessThan_1<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      function.return
    }
  }
  struct.def @Num2Bits_0 {
    struct.member @out : !array.type<33 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Num2Bits_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Num2Bits_0<[]>>
      %nondet = llzk.nondet : !array.type<33 x !felt.type>
      %felt_const_0 = felt.const  0
      %felt_const_1 = felt.const  1
      %felt_const_0_0 = felt.const  0
      %0:3 = scf.while (%arg1 = %felt_const_1, %arg2 = %felt_const_0_0, %arg3 = %felt_const_0) : (!felt.type, !felt.type, !felt.type) -> (!felt.type, !felt.type, !felt.type) {
        %felt_const_33_1 = felt.const  33
        %1 = bool.cmp lt(%arg2, %felt_const_33_1)
        scf.condition(%1) %arg1, %arg2, %arg3 : !felt.type, !felt.type, !felt.type
      } do {
      ^bb0(%arg1: !felt.type, %arg2: !felt.type, %arg3: !felt.type):
        %1 = felt.shr %arg0, %arg2 : !felt.type, !felt.type
        %felt_const_1_1 = felt.const  1
        %2 = felt.bit_and %1, %felt_const_1_1 : !felt.type, !felt.type
        %3 = cast.toindex %arg2
        array.write %nondet[%3] = %2 : <33 x !felt.type>, !felt.type
        %4 = cast.toindex %arg2
        %5 = array.read %nondet[%4] : <33 x !felt.type>, !felt.type
        %6 = felt.mul %5, %felt_const_1 : !felt.type, !felt.type
        %7 = felt.add %arg3, %6 : !felt.type, !felt.type
        %8 = felt.add %arg1, %arg1 : !felt.type, !felt.type
        %felt_const_1_2 = felt.const  1
        %9 = felt.add %arg2, %felt_const_1_2 : !felt.type, !felt.type
        scf.yield %8, %9, %7 : !felt.type, !felt.type, !felt.type
      }
      struct.writem %self[@out] = %nondet : <@Num2Bits_0<[]>>, !array.type<33 x !felt.type>
      function.return %self : !struct.type<@Num2Bits_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Num2Bits_0<[]>>, %arg1: !felt.type) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      function.return
    }
  }
}
