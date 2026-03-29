// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test: dispatch scf.if inside while body with array-of-pods.
// The function.call is extracted from the dispatch scf.if and the
// pod.read @comp is replaced with the call result.

// CHECK-LABEL: func.func @main
// CHECK: call @MiMC_0_compute
// CHECK-NOT: pod.
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Multi_1<[]>>} {
  struct.def @MiMC_0<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@MiMC_0<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@MiMC_0<[]>>
      %0 = felt.add %arg0, %arg1 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@MiMC_0<[]>>, !felt.type
      function.return %self : !struct.type<@MiMC_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@MiMC_0<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Multi_1<[]> {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @comp : !struct.type<@MiMC_0<[]>>
    struct.member @comp$inputs : !pod.type<[@x: !felt.type, @k: !felt.type]>
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@Multi_1<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Multi_1<[]>>
      %c1 = arith.constant 1 : index
      %pod = pod.new { @count = %c1 } : <[@count: index, @comp: !struct.type<@MiMC_0<[]>>, @params: !pod.type<[]>]>
      %pod_inp = pod.new : <[@x: !felt.type, @k: !felt.type]>
      pod.write %pod_inp[@x] = %arg0 : <[@x: !felt.type, @k: !felt.type]>, !felt.type
      pod.write %pod_inp[@k] = %arg1 : <[@x: !felt.type, @k: !felt.type]>, !felt.type
      %0 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@MiMC_0<[]>>, @params: !pod.type<[]>]>, index
      %c1_1 = arith.constant 1 : index
      %1 = arith.subi %0, %c1_1 : index
      pod.write %pod[@count] = %1 : <[@count: index, @comp: !struct.type<@MiMC_0<[]>>, @params: !pod.type<[]>]>, index
      %c0 = arith.constant 0 : index
      %2 = arith.cmpi eq, %1, %c0 : index
      scf.if %2 {
        %7 = pod.read %pod_inp[@x] : <[@x: !felt.type, @k: !felt.type]>, !felt.type
        %8 = pod.read %pod_inp[@k] : <[@x: !felt.type, @k: !felt.type]>, !felt.type
        %9 = function.call @MiMC_0::@compute(%7, %8) : (!felt.type, !felt.type) -> !struct.type<@MiMC_0<[]>>
        pod.write %pod[@comp] = %9 : <[@count: index, @comp: !struct.type<@MiMC_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@MiMC_0<[]>>
      } else {
      }
      %3 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@MiMC_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@MiMC_0<[]>>
      %4 = struct.readm %3[@out] : <@MiMC_0<[]>>, !felt.type
      struct.writem %self[@out] = %4 : <@Multi_1<[]>>, !felt.type
      struct.writem %self[@comp$inputs] = %pod_inp : <@Multi_1<[]>>, !pod.type<[@x: !felt.type, @k: !felt.type]>
      %5 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@MiMC_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@MiMC_0<[]>>
      struct.writem %self[@comp] = %5 : <@Multi_1<[]>>, !struct.type<@MiMC_0<[]>>
      function.return %self : !struct.type<@Multi_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Multi_1<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
