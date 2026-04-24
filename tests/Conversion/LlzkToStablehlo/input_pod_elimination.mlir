// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test that $inputs pod struct members are eliminated and do not appear
// in the StableHLO output. These pods store sub-component input parameters
// only needed by constrain functions.

// CHECK-NOT: pod.
// CHECK-NOT: $inputs

// CHECK-LABEL: func.func @main
// CHECK: call @Inner_compute
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Outer<[]>>} {
  struct.def @Inner {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Inner<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Inner<[]>>
      %sq = felt.mul %arg0, %arg0 : !felt.type, !felt.type
      struct.writem %self[@out] = %sq : <@Inner<[]>>, !felt.type
      function.return %self : !struct.type<@Inner<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Inner<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Outer {
    struct.member @result : !felt.type {llzk.pub}
    struct.member @inner : !struct.type<@Inner<[]>>
    struct.member @inner$inputs : !pod.type<[@in: !felt.type]>
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Outer<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Outer<[]>>
      %c1 = arith.constant 1 : index
      %pod = pod.new { @count = %c1 }  : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>
      %pod_0 = pod.new : <[@in: !felt.type]>
      pod.write %pod_0[@in] = %arg0 : <[@in: !felt.type]>, !felt.type
      %0 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, index
      %c1_1 = arith.constant 1 : index
      %1 = arith.subi %0, %c1_1 : index
      pod.write %pod[@count] = %1 : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, index
      %c0 = arith.constant 0 : index
      %2 = arith.cmpi eq, %1, %c0 : index
      scf.if %2 {
        %5 = pod.read %pod_0[@in] : <[@in: !felt.type]>, !felt.type
        %6 = function.call @Inner::@compute(%5) : (!felt.type) -> !struct.type<@Inner<[]>>
        pod.write %pod[@comp] = %6 : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, !struct.type<@Inner<[]>>
      } else {
      }
      %3 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, !struct.type<@Inner<[]>>
      %4 = struct.readm %3[@out] : <@Inner<[]>>, !felt.type
      struct.writem %self[@result] = %4 : <@Outer<[]>>, !felt.type
      struct.writem %self[@inner$inputs] = %pod_0 : <@Outer<[]>>, !pod.type<[@in: !felt.type]>
      struct.writem %self[@inner] = %3 : <@Outer<[]>>, !struct.type<@Inner<[]>>
      function.return %self : !struct.type<@Outer<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Outer<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
