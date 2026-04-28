// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the AES Num2Bits_2-style sub-component shape, where the
// dispatched component's `@out` is itself array-typed (`!array<K x !felt>`)
// rather than a scalar felt. Shape:
//
//   1. `%array = array.new : <N x !pod<[@count, @comp:T, @params]>>` at
//      function scope, dispatching component `@Sub_0` whose
//      `@out : !array<K x !felt>`.
//   2. A first scf.while ("writer") that decrements @count, fires
//      `function.call @Sub_0::@compute` from inside an scf.if, and writes
//      `pod.write %dp[@comp] = %res; array.write %array[%i] = %dp`.
//   3. A *separate* sibling scf.while ("reader") that walks the same array
//      via `array.read %array[%j]; pod.read %dp2[@comp]; struct.readm
//      %comp[@out]` (yielding an `!array<K x !felt>`) and then `array.read`
//      individual elements out of that.
//
// PR #25 (`materializePodArrayCompField`) was scalar-only on the reader
// filter (line 753-769) and the per-field dim builder (line 855-880),
// rejecting `@out : !array<K x !felt>`. The fix accepts array-typed `@out`
// by combining the dispatch dims with the field's inner dims via
// `getArrayDimensions(innerArr)` and emitting `array.insert`/`array.extract`
// at writer/reader sites instead of `array.write`/`array.read`. AES
// `aes_256_encrypt` Num2Bits_2 (`@out : !array<32 x !felt>`) was the
// primary failing case.
//
// CHECK lines below pin three contracts that all fail without the fix:
//   - `func.call @Sub_0_compute(...)` survives lowering with its result
//     consumed (without the fix, the call's result is dead and the call
//     is DCE'd; the reader hits a `dense<0>` tensor).
//   - The reader-while body slices from a materialized 2D `tensor<N,K>`
//     populated by the writer, not from a `dense<0>` constant.
//   - The function output is connected through the materialization, not
//     through `llzk.nondet`/zero constants.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: func.call @Sub_0_compute
// Writer-while updates the materialized 2D array via dynamic_update_slice.
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.while
// Reader-while slices the materialized array — no all-zero placeholder.
// CHECK-NOT: dense<0> : tensor<1x2xi256>
// CHECK: stablehlo.dynamic_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  // Sub_0::@out is array-typed (the variant `materializePodArrayCompField`
  // missed before the array-typed `@out` extension).
  struct.def @Sub_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@Sub_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %0 = felt.add %arg0, %arg1 : !felt.type, !felt.type
      %1 = felt.mul %arg0, %arg1 : !felt.type, !felt.type
      array.write %nondet[%c0] = %0 : <2 x !felt.type>, !felt.type
      array.write %nondet[%c1] = %1 : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet : <@Sub_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_0<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !array.type<2 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<4 x !felt.type>
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %array[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      }
      // Writer scf.while: dispatches Sub_0::@compute(arg0[i], arg1[i]) per
      // iteration into %array[i].@comp. The materializer must populate a
      // per-field `array.new : <2,2 x !felt>` for `@out` (NOT a 1D scalar
      // array, which is the felt-typed `@out` shape covered by PR #25).
      %felt0 = felt.const 0
      %0 = scf.while (%iter = %felt0) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %a = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %b = array.read %arg1[%idx] : <2 x !felt.type>, !felt.type
        %dp = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          %res = function.call @Sub_0::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@Sub_0<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
          array.write %array[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        } else {
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      // Reader scf.while: walks %array, reads back .@comp.@out (array-typed)
      // and copies into %nondet_out. Without the fix, the reader's
      // `struct.readm %comp[@out]` is replaced by `llzk.nondet : !array<2 x !felt>`
      // by Phase 5; with the fix, the readm is rewritten to
      // `array.extract %comp_out_array[%idx]`.
      %felt0_2 = felt.const 0
      %1 = scf.while (%iter = %felt0_2) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %dp2 = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %out = struct.readm %comp[@out] : <@Sub_0<[]>>, !array.type<2 x !felt.type>
        // Walk the inner array and copy into %nondet_out.
        %v0 = array.read %out[%c0] : <2 x !felt.type>, !felt.type
        %v1 = array.read %out[%c1] : <2 x !felt.type>, !felt.type
        %felt2_w = felt.const 2
        %iter2 = felt.mul %iter, %felt2_w : !felt.type, !felt.type
        %felt1_w = felt.const 1
        %iter2p1 = felt.add %iter2, %felt1_w : !felt.type, !felt.type
        %widx0 = cast.toindex %iter2
        %widx1 = cast.toindex %iter2p1
        array.write %nondet_out[%widx0] = %v0 : <4 x !felt.type>, !felt.type
        array.write %nondet_out[%widx1] = %v1 : <4 x !felt.type>, !felt.type
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<4 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
