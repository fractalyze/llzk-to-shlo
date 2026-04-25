// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Test: dispatch scf.if inside a while body with array-of-pods dispatch.
// The function.call is extracted from scf.if and pod.read @comp is
// replaced with the call result. The func.call is preserved through
// the void scf.if erasure and reconnected to stablehlo.slice consumers.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: func.call @Comp_0_compute
// CHECK: stablehlo.slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Comp_0 {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @aux : !felt.type
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Comp_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Comp_0<[]>>
      %0 = felt.mul %arg0, %arg0 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Comp_0<[]>>, !felt.type
      struct.writem %self[@aux] = %arg0 : <@Comp_0<[]>>, !felt.type
      function.return %self : !struct.type<@Comp_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Comp_0<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    struct.member @sub : !array.type<2 x !struct.type<@Comp_0<[]>>>
    struct.member @sub$inputs : !array.type<2 x !pod.type<[@x: !felt.type]>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %array[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %c1_init = arith.constant 1 : index
        pod.write %p[@count] = %c1_init : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
      }
      %arr_inp = array.new : <2 x !pod.type<[@x: !felt.type]>>
      %felt0 = felt.const 0
      %1:2 = scf.while (%iter = %felt0, %carry = %arr_inp) : (!felt.type, !array.type<2 x !pod.type<[@x: !felt.type]>>) -> (!felt.type, !array.type<2 x !pod.type<[@x: !felt.type]>>) {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter, %carry : !felt.type, !array.type<2 x !pod.type<[@x: !felt.type]>>
      } do {
      ^bb0(%iter: !felt.type, %carry: !array.type<2 x !pod.type<[@x: !felt.type]>>):
        %idx = cast.toindex %iter
        %xval = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %elem = array.read %carry[%idx] : <2 x !pod.type<[@x: !felt.type]>>, !pod.type<[@x: !felt.type]>
        pod.write %elem[@x] = %xval : <[@x: !felt.type]>, !felt.type
        array.write %carry[%idx] = %elem : <2 x !pod.type<[@x: !felt.type]>>, !pod.type<[@x: !felt.type]>
        %dp = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        %c1_2 = arith.constant 1 : index
        %cnt2 = arith.subi %cnt, %c1_2 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        %c0_2 = arith.constant 0 : index
        %eq = arith.cmpi eq, %cnt2, %c0_2 : index
        scf.if %eq {
          %ax = pod.read %elem[@x] : <[@x: !felt.type]>, !felt.type
          %res = function.call @Comp_0::@compute(%ax) : (!felt.type) -> !struct.type<@Comp_0<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
          array.write %array[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        } else {
        }
        %dp2 = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
        %out = struct.readm %comp[@out] : <@Comp_0<[]>>, !felt.type
        array.write %nondet[%idx] = %out : <2 x !felt.type>, !felt.type
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !array.type<2 x !pod.type<[@x: !felt.type]>>
      }
      struct.writem %self[@out] = %nondet : <@Main_1<[]>>, !array.type<2 x !felt.type>
      struct.writem %self[@sub$inputs] = %1#1 : <@Main_1<[]>>, !array.type<2 x !pod.type<[@x: !felt.type]>>
      %arr_sub = array.new : <2 x !struct.type<@Comp_0<[]>>>
      struct.writem %self[@sub] = %arr_sub : <@Main_1<[]>>, !array.type<2 x !struct.type<@Comp_0<[]>>>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
