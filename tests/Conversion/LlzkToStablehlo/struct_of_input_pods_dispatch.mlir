// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for the struct-of-input-pods carrier shape that surfaced in
// webb_poseidon_vanchor_{2_2,16_2,16_8,2_8}. The input-pod-of-pods carrier
// `!pod<[@idx_0..@idx_K-1: !pod<[@in: !array<P x !felt>]>]>` was unflattened
// by `flattenPodArrayWhileCarry` (which only handles `!array<K x !pod<...>>`),
// dropping the operand chain into the dispatched function.call.
//
// The new pre-pass `convertStructOfPodsToArrayOfPods` rewrites that carrier
// to `!array<K x !pod<[@in: !array<P x !felt>]>>` with synthesized
// `arith.constant N : index` indices, so the existing flatten/unpack
// infrastructure handles the rest. K=1 keeps the fixture small while
// preserving the struct-of-pods shape (top-level type starts `!pod<[@idx_0:`
// not `!array<1 x`, so the seed predicate still fires).

// CHECK-LABEL: struct.def @Main_1
// CHECK-NOT: !pod.type<[@idx_0:
// CHECK: array.new
// CHECK-SAME: !array
// CHECK-NOT: pod.read {{.*}}[@idx_
// CHECK-NOT: pod.write {{.*}}[@idx_

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Sub_0 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>)
        -> !struct.type<@Sub_0<[]>>
        attributes {function.allow_non_native_field_ops,
                    function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
      %c0 = arith.constant 0 : index
      %v0 = array.read %arg0[%c0] : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %v0 : <@Sub_0<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%self: !struct.type<@Sub_0<[]>>,
                            %arg0: !array.type<2 x !felt.type>)
        attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>)
        -> !struct.type<@Main_1<[]>>
        attributes {function.allow_non_native_field_ops,
                    function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      // Build the struct-of-pods input carrier (K=1).
      %inp = pod.new
          : <[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>

      %felt0 = felt.const 0
      %felt2 = felt.const 2

      // Outer scf.while threads the struct-of-pods carrier as iter-arg.
      // `pod.read %i[@idx_0]` reads the inner pod (compile-time @idx_0),
      // `pod.read %inner[@in]` reads the felt array, `array.write` mutates
      // a position, then writebacks store the rebound pod via
      // `pod.write %i[@idx_0] = %inner`.
      %0:2 = scf.while (%pos = %felt0, %i = %inp)
          : (!felt.type,
             !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>)
          -> (!felt.type,
              !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>) {
        %cond = bool.cmp lt(%pos, %felt2)
            : !felt.type, !felt.type
        scf.condition(%cond) %pos, %i
            : !felt.type,
              !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>
      } do {
      ^bb0(%pos: !felt.type,
           %i: !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>):
        %idx = cast.toindex %pos : !felt.type
        %v = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %i0 = pod.read %i[@idx_0]
            : !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>,
              !pod.type<[@in: !array.type<2 x !felt.type>]>
        %arr = pod.read %i0[@in]
            : !pod.type<[@in: !array.type<2 x !felt.type>]>,
              !array.type<2 x !felt.type>
        array.write %arr[%idx] = %v : <2 x !felt.type>, !felt.type
        %i0_rb = pod.read %i[@idx_0]
            : !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>,
              !pod.type<[@in: !array.type<2 x !felt.type>]>
        pod.write %i0_rb[@in] = %arr
            : !pod.type<[@in: !array.type<2 x !felt.type>]>,
              !array.type<2 x !felt.type>
        pod.write %i[@idx_0] = %i0_rb
            : !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>,
              !pod.type<[@in: !array.type<2 x !felt.type>]>
        %felt1 = felt.const 1
        %next = felt.add %pos, %felt1 : !felt.type, !felt.type
        scf.yield %next, %i
            : !felt.type,
              !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>
      }

      // Read back the input pod and dispatch the sub-component.
      %final_inner = pod.read %0#1[@idx_0]
          : !pod.type<[@idx_0: !pod.type<[@in: !array.type<2 x !felt.type>]>]>,
            !pod.type<[@in: !array.type<2 x !felt.type>]>
      %final_arr = pod.read %final_inner[@in]
          : !pod.type<[@in: !array.type<2 x !felt.type>]>,
            !array.type<2 x !felt.type>
      %r = function.call @Sub_0::@compute(%final_arr)
          : (!array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
      %out_val = struct.readm %r[@out]
          : <@Sub_0<[]>>, !felt.type
      struct.writem %self[@out] = %out_val : <@Main_1<[]>>, !felt.type
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%self: !struct.type<@Main_1<[]>>,
                            %arg0: !array.type<2 x !felt.type>)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
