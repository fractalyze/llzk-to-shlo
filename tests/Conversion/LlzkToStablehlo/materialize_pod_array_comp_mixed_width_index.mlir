// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for `materializePodArrayCompField`'s multi-writer
// equal-outer-index fold. `outerIndexConstValues` returns APInts read
// from either `felt.const` (`FeltConstAttr` — typically 254-bit on
// bn128) or `arith.constant` on `index` type (64-bit). When two
// writers' indices come from different sources, the per-slot
// comparison at the all-same-outer-index check has mismatched bit
// widths. `APInt::operator==` then asserts in debug builds and is UB
// in opt builds — the latter producing ill-formed downstream IR that
// crashed inside `EmptyTemplateRemoval` for the 34-chip
// `aes_gcm_siv_*` / `iden3_*` / `maci_batch_*` / `webb_batch_merkle_*`
// / `zksql_*` cluster.
//
// The fix zext-normalizes both APInts to a common width before
// comparing. Indices are non-negative, so zext is correct.
//
// Shape: two function-scope writer chains write to the same dispatch
// slot through different index SSA values. Writer A's index is a
// direct `arith.constant 0 : index` (64-bit). Writer B's index walks
// `cast.toindex %felt0` where `%felt0 = felt.const 0` (254-bit on
// bn128). A cross-block reader inside an `scf.while` body satisfies
// the materialize-fire precondition.
//
// `array.new : <2 x !felt.type<"bn128">>` is the per-field felt array
// the materialize fold synthesizes; its presence confirms the fold
// reached `materializePodArrayCompField`'s per-field allocation step
// after the mixed-width comparison. A bare `function.return` check
// would also pass if the pass silently dropped the body.
// CHECK-LABEL: struct.def @Main_1
// CHECK: array.new {{.*}} <2 x !felt.type<"bn128">>
// CHECK: function.call @Comp_0
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Comp_0 {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Comp_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Comp_0<[]>>
      struct.writem %self[@out] = %arg0 : <@Comp_0<[]>>, !felt.type<"bn128">
      function.return %self : !struct.type<@Comp_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Comp_0<[]>>, %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !felt.type<"bn128"> {llzk.pub}
    function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !felt.type<"bn128">
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>

      // Writer A: outer index sourced from `arith.constant 0 : index`
      // (64-bit APInt).
      %c0_idx = arith.constant 0 : index
      %dp_a = array.read %array[%c0_idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
      %res_a = function.call @Comp_0::@compute(%arg0) : (!felt.type<"bn128">) -> !struct.type<@Comp_0<[]>>
      pod.write %dp_a[@comp] = %res_a : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
      array.write %array[%c0_idx] = %dp_a : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>

      // Writer B: outer index sourced from `cast.toindex %felt1` where
      // `%felt1 = felt.const 1`. The cast.toindex walk-back lands on
      // `felt.const` whose `FeltConstAttr` is built by MLIR's
      // `APIntParameter` parser from the literal `1`, which sizes the
      // APInt to ~4 bits (the minimum-bits-needed rounding). Writer A's
      // index uses `arith.constant 0 : index` which produces a 64-bit
      // APInt. Comparing the two at the materialize fold's all-same-
      // outer-index check (mixed-width values 1 vs 0) trips
      // `APInt::operator!=`'s "Comparison requires equal bit widths"
      // assertion in dbg — UB in opt.
      //
      // Writer B writes to slot 1 (value=1), Writer A to slot 0
      // (value=0): the values themselves differ, so the fold correctly
      // bails out as "not all same outer index" once widths are
      // normalized. The regression is the comparison itself surviving
      // a width mismatch.
      %felt1 = felt.const 1 : <"bn128">
      %idx_felt = cast.toindex %felt1 : !felt.type<"bn128">
      %dp_b = array.read %array[%idx_felt] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
      %res_b = function.call @Comp_0::@compute(%arg0) : (!felt.type<"bn128">) -> !struct.type<@Comp_0<[]>>
      pod.write %dp_b[@comp] = %res_b : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
      array.write %array[%idx_felt] = %dp_b : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>

      // Cross-block reader inside scf.while — satisfies the materialize
      // fire-precondition (at least one cross-block `struct.readm @F`
      // consumer of a `pod.read [@comp]`).
      %felt_zero = felt.const 0 : <"bn128">
      %0 = scf.while (%iter = %felt_zero) : (!felt.type<"bn128">) -> !felt.type<"bn128"> {
        %felt2 = felt.const 2 : <"bn128">
        %cond = bool.cmp lt(%iter, %felt2) : !felt.type<"bn128">, !felt.type<"bn128">
        scf.condition(%cond) %iter : !felt.type<"bn128">
      } do {
      ^bb0(%iter: !felt.type<"bn128">):
        %idx_r = cast.toindex %iter : !felt.type<"bn128">
        %dp_r = array.read %array[%idx_r] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        %comp_r = pod.read %dp_r[@comp] : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Comp_0<[]>>
        %out = struct.readm %comp_r[@out] : <@Comp_0<[]>>, !felt.type<"bn128">
        %felt_one = felt.const 1 : <"bn128">
        %next = felt.add %iter, %felt_one : !felt.type<"bn128">, !felt.type<"bn128">
        scf.yield %next : !felt.type<"bn128">
      }

      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !felt.type<"bn128">
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
      function.return
    }
  }
}
