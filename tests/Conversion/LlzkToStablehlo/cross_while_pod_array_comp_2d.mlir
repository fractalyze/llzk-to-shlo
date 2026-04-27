// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Multi-dim regression for `materializePodArrayCompField`. PR #25 lifted
// the cross-while @comp field for 1D `array.new : <N x !pod>` arrays;
// rank > 1 was silently broken because:
//   - The writer scan picked `array.write`'s value with `getOperand(2)`,
//     which on `array.write %arr[%i, %j] = %v` returns the index `%j`
//     (an `index` SSA value), not the pod value `%v`. The subsequent
//     `pod.write` match failed, `writers` stayed empty, and
//     materialization was skipped.
//   - The reader rewrite passed only `arrayRead->getOperand(1)` as the
//     index for the per-field `array.read`, which would emit a single-
//     index read on a multi-rank `array<D1, D2 x !felt>` per-field array.
// Symptom on AES-style sub-component aggregation fields: the cross-while
// `pod.read [@comp]` was nondet'd to zero, the @main DUS chain emitted
// `dense<0>`, and the GPU witness was wrong.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: func.call @Comp_0_compute
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.while
// CHECK-NOT: dense<0> : tensor<2x2xi256>
// CHECK: stablehlo.dynamic_slice
// CHECK: stablehlo.dynamic_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_2<[]>>} {
  struct.def @Comp_0 {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @aux : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@Comp_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Comp_0<[]>>
      %0 = felt.add %arg0, %arg1 : !felt.type, !felt.type
      %1 = felt.mul %arg0, %arg1 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Comp_0<[]>>, !felt.type
      struct.writem %self[@aux] = %1 : <@Comp_0<[]>>, !felt.type
      function.return %self : !struct.type<@Comp_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Comp_0<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_2 {
    struct.member @out : !array.type<2,2 x !felt.type> {llzk.pub}
    struct.member @aux : !array.type<2,2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2,2 x !felt.type>, %arg1: !array.type<2,2 x !felt.type>) -> !struct.type<@Main_2<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_2<[]>>
      %nondet_out = llzk.nondet : !array.type<2,2 x !felt.type>
      %nondet_aux = llzk.nondet : !array.type<2,2 x !felt.type>
      %array = array.new : <2,2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i_init = %c0 to %c2 step %c1 {
        scf.for %j_init = %c0 to %c2 step %c1 {
          %p = array.read %array[%i_init, %j_init] : <2,2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
          pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
          array.write %array[%i_init, %j_init] = %p : <2,2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        }
      }
      // Writer: outer scf.while iterates i, inner scf.while iterates j.
      // Dispatch fire occurs at (%i, %j) inside scf.if.
      %felt0_wi = felt.const 0
      %0 = scf.while (%iter_i = %felt0_wi) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter_i, %felt2)
        scf.condition(%cond) %iter_i : !felt.type
      } do {
      ^bb0(%iter_i: !felt.type):
        %i = cast.toindex %iter_i
        %felt0_wj = felt.const 0
        %inner_w = scf.while (%iter_j = %felt0_wj) : (!felt.type) -> !felt.type {
          %felt2 = felt.const 2
          %cond = bool.cmp lt(%iter_j, %felt2)
          scf.condition(%cond) %iter_j : !felt.type
        } do {
        ^bb0(%iter_j: !felt.type):
          %j = cast.toindex %iter_j
          %a = array.read %arg0[%i, %j] : <2,2 x !felt.type>, !felt.type
          %b = array.read %arg1[%i, %j] : <2,2 x !felt.type>, !felt.type
          %dp = array.read %array[%i, %j] : <2,2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
          %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
          %cnt2 = arith.subi %cnt, %c1 : index
          pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
          %fire = arith.cmpi eq, %cnt2, %c0 : index
          scf.if %fire {
            %res = function.call @Comp_0::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@Comp_0<[]>>
            pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
            array.write %array[%i, %j] = %dp : <2,2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
          } else {
          }
          %felt1_j = felt.const 1
          %next_j = felt.add %iter_j, %felt1_j : !felt.type, !felt.type
          scf.yield %next_j : !felt.type
        }
        %felt1_i = felt.const 1
        %next_i = felt.add %iter_i, %felt1_i : !felt.type, !felt.type
        scf.yield %next_i : !felt.type
      }
      // Reader: nested scf.while over the same array, drains @out + @aux.
      // This is the cross-while data flow that gets nondet'd without the
      // multi-dim writer-detection fix.
      %felt0_ri = felt.const 0
      %1 = scf.while (%iter_i = %felt0_ri) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter_i, %felt2)
        scf.condition(%cond) %iter_i : !felt.type
      } do {
      ^bb0(%iter_i: !felt.type):
        %i = cast.toindex %iter_i
        %felt0_rj = felt.const 0
        %inner_r = scf.while (%iter_j = %felt0_rj) : (!felt.type) -> !felt.type {
          %felt2 = felt.const 2
          %cond = bool.cmp lt(%iter_j, %felt2)
          scf.condition(%cond) %iter_j : !felt.type
        } do {
        ^bb0(%iter_j: !felt.type):
          %j = cast.toindex %iter_j
          %dp2 = array.read %array[%i, %j] : <2,2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
          %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
          %out = struct.readm %comp[@out] : <@Comp_0<[]>>, !felt.type
          %aux = struct.readm %comp[@aux] : <@Comp_0<[]>>, !felt.type
          array.write %nondet_out[%i, %j] = %out : <2,2 x !felt.type>, !felt.type
          array.write %nondet_aux[%i, %j] = %aux : <2,2 x !felt.type>, !felt.type
          %felt1_j = felt.const 1
          %next_j = felt.add %iter_j, %felt1_j : !felt.type, !felt.type
          scf.yield %next_j : !felt.type
        }
        %felt1_i = felt.const 1
        %next_i = felt.add %iter_i, %felt1_i : !felt.type, !felt.type
        scf.yield %next_i : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_2<[]>>, !array.type<2,2 x !felt.type>
      struct.writem %self[@aux] = %nondet_aux : <@Main_2<[]>>, !array.type<2,2 x !felt.type>
      function.return %self : !struct.type<@Main_2<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_2<[]>>, %arg1: !array.type<2,2 x !felt.type>, %arg2: !array.type<2,2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
