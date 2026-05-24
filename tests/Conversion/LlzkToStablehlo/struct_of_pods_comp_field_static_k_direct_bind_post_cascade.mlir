// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Static-K direct-binding regression for post-cascade multi-K readers.
//
// This is the same-class / multi-K shape that forces
// `materializeStructOfPodsCompField` to resolve reader K through
// `deriveReaderK` Strategy (A): walk back from
// `struct.readm <- pod.read[@comp] <- array.read[%cK]`.
//
// The direct-binding fast path is only legal when the chosen writer-side
// value dominates each reader. This fixture keeps both writers hoisted in the
// function body and places the readers at top level, so the dominance gate is
// satisfied and the pass may bypass the `mSoPCF.carrier-for` carrier.
//
// CHECK-LABEL: function.def @compute(
// CHECK-NOT: "mSoPCF.carrier-for"
// CHECK-DAG: %[[CALL0:.*]] = function.call @Sub_A::@compute(
// CHECK-DAG: %[[OUT0:.*]] = struct.readm %[[CALL0]][@out]
// CHECK-DAG: %[[CALL1:.*]] = function.call @Sub_A::@compute(
// CHECK-DAG: %[[OUT1:.*]] = struct.readm %[[CALL1]][@out]
// CHECK: array.write %nondet[%c0] = %[[OUT0]]
// CHECK: array.write %nondet[%c1] = %[[OUT1]]

module attributes {llzk.lang, llzk.main = !struct.type<@Main_0<[]>>} {
  struct.def @Sub_A {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_A<[]>>
      %c0 = arith.constant 0 : index
      %v = array.read %arg0[%c0] : <1 x !felt.type>, !felt.type
      struct.writem %self[@out] = %v : <@Sub_A<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_A<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_A<[]>>, %arg1: !array.type<1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2, 1 x !felt.type>) -> !struct.type<@Main_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_0<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %dummy = pod.new : <[@d: index]>
      %dummy_read = pod.read %dummy[@d] : <[@d: index]>, index

      %ext_a = array.extract %arg0[%c0] : <2, 1 x !felt.type>
      %call_a = function.call @Sub_A::@compute(%ext_a) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>
      %ext_b = array.extract %arg0[%c1] : <2, 1 x !felt.type>
      %call_b = function.call @Sub_A::@compute(%ext_b) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>

      %carrier = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>>

      %slot_a = array.read %carrier[%c0] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>
      %comp_a = pod.read %slot_a[@comp] : <[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_A<[]>>
      %read_a = struct.readm %comp_a[@out] : <@Sub_A<[]>>, !felt.type
      array.write %nondet_out[%c0] = %read_a : <2 x !felt.type>, !felt.type

      %slot_b = array.read %carrier[%c1] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>
      %comp_b = pod.read %slot_b[@comp] : <[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_A<[]>>
      %read_b = struct.readm %comp_b[@out] : <@Sub_A<[]>>, !felt.type
      array.write %nondet_out[%c1] = %read_b : <2 x !felt.type>, !felt.type

      struct.writem %self[@out] = %nondet_out : <@Main_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_0<[]>>, %arg1: !array.type<2, 1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
