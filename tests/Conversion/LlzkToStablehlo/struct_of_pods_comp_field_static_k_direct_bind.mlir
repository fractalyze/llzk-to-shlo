// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Static-K fast path for `materializeStructOfPodsCompField`.
//
// When every reader can be resolved to a constant K and the corresponding
// writer call has already been hoisted into the function body, the pass should
// bypass the per-field `mSoPCF.carrier-for` array entirely and wire the
// writer-side `struct.readm %call[@out]` values directly into the reader uses.
//
// This fixture stages that canonical post-hoist form directly:
// - writer calls sit in the function body with constant-K `array.extract` args
// - readers are plain `llzk.nondet -> struct.readm` chains
// - no runtime-K cascade remains, so the carrier path is unnecessary
//
// CHECK-LABEL: function.def @compute(
// CHECK-NOT: "mSoPCF.carrier-for"
// CHECK-DAG: %[[CALL_A:.*]] = function.call @Sub_A::@compute(
// CHECK-DAG: %[[OUT_A:.*]] = struct.readm %[[CALL_A]][@out]
// CHECK-DAG: %[[CALL_B:.*]] = function.call @Sub_B::@compute(
// CHECK-DAG: %[[OUT_B:.*]] = struct.readm %[[CALL_B]][@out]
// CHECK: array.write %nondet[%c0] = %[[OUT_A]]
// CHECK: array.write %nondet[%c1] = %[[OUT_B]]

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
  struct.def @Sub_B {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<1 x !felt.type>) -> !struct.type<@Sub_B<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_B<[]>>
      %c0 = arith.constant 0 : index
      %v = array.read %arg0[%c0] : <1 x !felt.type>, !felt.type
      struct.writem %self[@out] = %v : <@Sub_B<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_B<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_B<[]>>, %arg1: !array.type<1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2, 1 x !felt.type>) -> !struct.type<@Main_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_0<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %dummy = pod.new : <[@d: index]>
      %dummy_read = pod.read %dummy[@d] : <[@d: index]>, index
      %ext_a = array.extract %arg0[%c0] : <2, 1 x !felt.type>
      %call_a = function.call @Sub_A::@compute(%ext_a) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>
      %ext_b = array.extract %arg0[%c1] : <2, 1 x !felt.type>
      %call_b = function.call @Sub_B::@compute(%ext_b) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_B<[]>>
      %ndet_a = llzk.nondet : !struct.type<@Sub_A<[]>>
      %read_a = struct.readm %ndet_a[@out] : <@Sub_A<[]>>, !felt.type
      array.write %nondet[%c0] = %read_a : <2 x !felt.type>, !felt.type
      %ndet_b = llzk.nondet : !struct.type<@Sub_B<[]>>
      %read_b = struct.readm %ndet_b[@out] : <@Sub_B<[]>>, !felt.type
      array.write %nondet[%c1] = %read_b : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet : <@Main_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_0<[]>>, %arg1: !array.type<2, 1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
