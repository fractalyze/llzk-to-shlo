// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for circomlib gates.circom XorArray-style helpers (keccak chi /
// iota / rhopi / round / squeeze / theta + AES + iden3 + SHA-256). The shape:
//
//   1. `%array = array.new : <N x !pod<[@count, @comp:T, @params]>>` at
//      function scope.
//   2. A first scf.while ("writer") that decrements @count, fires
//      `function.call @Comp::@compute` from inside an scf.if, and writes
//      `pod.write %dp[@comp] = %res; array.write %array[%i] = %dp`.
//   3. A *separate* sibling scf.while ("reader") that walks the same array
//      via `array.read %array[%j]; pod.read %dp2[@comp]; struct.readm
//      %comp[@out]` and stores the result into the function output.
//
// Before the fix, SimplifySubComponents' `extractCallsFromScfIf` tracker is
// per-SSA-value (`%dp` ≠ `%dp2`), so the cross-while data flow is severed:
// `rewriteArrayPodCountCompInReads` (and Phase 5 as a backstop) replaced the
// reader's `pod.read %dp2[@comp]` with `llzk.nondet`, causing the lowered
// helper to silently return zeros even though `func.call @Comp_0_compute`
// remained in the IR with a dead result. The fix
// (`materializePodArrayCompField`) lifts the `@comp` field of every
// function-scope pod-array dispatch storage into a per-field
// `array.new : <N x T>`, rewires writers to populate it directly, and
// rewires readers to consume from it.
//
// The CHECK lines below pin three contracts that all fail without the fix:
//   - `func.call @Comp_0_compute(...)` survives lowering with its result
//     consumed (today the result is dead and the call gets DCE'd).
//   - The reader while body slices from a non-zero source tensor (today it
//     slices from `dense<0> : tensor<1x...>`).
//   - The function return is connected to the call result through the
//     materialized comp array, not through `llzk.nondet`/zero constants.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: func.call @Comp_0_compute
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.while
// CHECK-NOT: dense<0> : tensor<1xi256>
// CHECK: stablehlo.dynamic_slice
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Comp_0 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@Comp_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Comp_0<[]>>
      %0 = felt.add %arg0, %arg1 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Comp_0<[]>>, !felt.type
      function.return %self : !struct.type<@Comp_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Comp_0<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !array.type<2 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %array[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
      }
      // Writer scf.while: dispatches Comp_0::@compute(arg0[i], arg1[i]) per
      // iteration into %array[i].@comp.
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
        %dp = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          %res = function.call @Comp_0::@compute(%a, %b) : (!felt.type, !felt.type) -> !struct.type<@Comp_0<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
          array.write %array[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        } else {
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      // Reader scf.while: walks %array, reads back .@comp.@out into %nondet_out.
      // This is the cross-while read that gets nondet'd without the fix.
      %felt0_2 = felt.const 0
      %1 = scf.while (%iter = %felt0_2) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %dp2 = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
        %out = struct.readm %comp[@out] : <@Comp_0<[]>>, !felt.type
        array.write %nondet_out[%idx] = %out : <2 x !felt.type>, !felt.type
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
