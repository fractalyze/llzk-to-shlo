// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32 flag-orphan-zero-writes=true" %s | FileCheck %s

// Regression for the AES `aes_256_encrypt` `@num2bits_2` orphan that PR #67's
// loud-failure assertion uncovered. Shape:
//
//   1. `%array = array.new : <D x !pod<[@count, @comp:T, @params]>>` at
//      function scope, dispatching component `@Sub` whose
//      `@out : !array<K x !felt>` (array-typed, NOT scalar).
//   2. A writer scf.while/scf.for that fires `function.call @Sub::@compute`
//      from inside an scf.if and stores the result via
//      `pod.write %dp[@comp] = %res; array.write %array[%i] = %dp`.
//   3. Post-loop scf.for that DRAINS the dispatch-pod array into a parallel
//      struct array via whole-struct copy:
//        `%dp2 = array.read %array[%j];
//         %comp = pod.read %dp2[@comp] : !struct<@Sub>;
//         array.write %destStruct[%j] = %comp`
//      (NO `struct.readm @out` — direct whole-struct write).
//   4. `struct.writem %self[@member] = %destStruct : <D x !struct<@Sub>>`.
//
// Distinguishes from the existing `cross_while_pod_array_comp_array_typed_out`
// fixture (which exercises the reader path: `struct.readm @out + array.read`).
// This fixture exercises the DRAIN path (whole-struct write into a struct
// array that flows into struct.writem of an output member).

// CHECK-LABEL: func.func @main
// Hoisted call survives lowering with its result consumed.
// CHECK: func.call @Sub_compute
// Writer materializes into a 2D `tensor<D, K>` via dynamic_update_slice
// instead of a `dense<0>` placeholder.
// CHECK: stablehlo.dynamic_update_slice
// No all-zero placeholder for the writem's value (i.e. the orphan-zero
// fallback in StructWriteMPattern is NOT taken).
// CHECK-NOT: dense<0> : tensor<6xi32>
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main<[]>>} {
  // Sub::@out is array-typed (the variant the original drainReader path
  // missed). `{llzk.pub}` marks @out as the public output for the
  // discriminator that picks @out over any internal felt-or-felt-array
  // auxiliaries.
  struct.def @Sub {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Sub<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %felt_const_2 = felt.const 2
      %0 = felt.add %arg0, %arg0 : !felt.type, !felt.type
      %1 = felt.mul %arg0, %felt_const_2 : !felt.type, !felt.type
      array.write %nondet[%c0] = %0 : <2 x !felt.type>, !felt.type
      array.write %nondet[%c1] = %1 : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet : <@Sub<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Sub<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main {
    // Inputs to the dispatched sub-components — populated by the writer.
    struct.member @member$inputs : !array.type<3 x !pod.type<[@in: !felt.type]>>
    // The drain destination: a struct array of @Sub. Without the fix,
    // its writem is the orphan-zero target.
    struct.member @member : !array.type<3 x !struct.type<@Sub<[]>>>
    function.def @compute(%arg0: !array.type<3 x !felt.type>) -> !struct.type<@Main<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main<[]>>
      %array = array.new : <3 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>
      %inputs = array.new : <3 x !pod.type<[@in: !felt.type]>>
      %destStruct = array.new : <3 x !struct.type<@Sub<[]>>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i = %c0 to %c3 step %c1 {
        %p = array.read %array[%i] : <3 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <3 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
      }
      // Writer scf.while: dispatches Sub::@compute(arg0[i]) per iteration
      // into %array[i].@comp. Also stages @in into %inputs[i] for the
      // parent's `@member$inputs` slot.
      %felt0 = felt.const 0
      %0 = scf.while (%iter = %felt0) : (!felt.type) -> !felt.type {
        %felt3 = felt.const 3
        %cond = bool.cmp lt(%iter, %felt3)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %a = array.read %arg0[%idx] : <3 x !felt.type>, !felt.type
        %ip = array.read %inputs[%idx] : <3 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        pod.write %ip[@in] = %a : <[@in: !felt.type]>, !felt.type
        array.write %inputs[%idx] = %ip : <3 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        %dp = array.read %array[%idx] : <3 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          %res = function.call @Sub::@compute(%a) : (!felt.type) -> !struct.type<@Sub<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub<[]>>
          array.write %array[%idx] = %dp : <3 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        } else {
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      // DRAIN scf.for: walks %array, reads back .@comp (whole struct, NOT
      // .@out), writes into %destStruct. This is the pattern that AES
      // `aes_256_encrypt` emits for `@num2bits_2`.
      scf.for %j = %c0 to %c3 step %c1 {
        %dp2 = array.read %array[%j] : <3 x !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Sub<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub<[]>>
        array.write %destStruct[%j] = %comp : <3 x !struct.type<@Sub<[]>>>, !struct.type<@Sub<[]>>
      }
      struct.writem %self[@member$inputs] = %inputs : <@Main<[]>>, !array.type<3 x !pod.type<[@in: !felt.type]>>
      struct.writem %self[@member] = %destStruct : <@Main<[]>>, !array.type<3 x !struct.type<@Sub<[]>>>
      function.return %self : !struct.type<@Main<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main<[]>>, %arg1: !array.type<3 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
