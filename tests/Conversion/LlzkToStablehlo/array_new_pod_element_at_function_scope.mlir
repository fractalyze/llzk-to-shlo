// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=bn254" %s | FileCheck %s

// Regression for AES-256-encrypt's `component xor_2[13][4][3][32]`-style
// pattern. After SimplifySubComponents clears pod read/write traffic, the
// `array.new : <... x !pod.type<[...]>>` at @compute scope is the only LLZK
// relic left for type conversion. Before the fix that landed alongside this
// test, the array dialect's legality predicate marked any pod-element array
// op as legal — which blocked ArrayNewPattern from firing on `array.new` and
// left it paired with an `unrealized_conversion_cast` to the converted tensor
// type. StableHLO-only consumers (m3_runner, GPU JIT) reject the residual at
// parse time. The fix lets ArrayNewPattern fire because the type converter
// erases the pod element (`tensor<... x !pf>`), so a zero-initialized
// constant is the correct lowering and the cast becomes redundant.

// CHECK-LABEL: func.func @main
// CHECK-NOT: array.new
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Main_1 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      // 1D, single-field pod element (mirrors AES `bits2num_2[16]`).
      %a = array.new : <16 x !pod.type<[@in: !felt.type]>>
      // 4D, multi-field pod element (mirrors AES `xor_2[13][4][3][32]`).
      %b = array.new : <13,4,3,32 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      %z = felt.const 0
      struct.writem %self[@out] = %z : <@Main_1<[]>>, !felt.type
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
