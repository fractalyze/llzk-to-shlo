// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for `flattenPodArrayScfIfResults`: a result-bearing scf.if
// returning `<D x !pod<[@a, @b]>>` rewrites into N per-field
// `<D x !felt>` result slots. Both branches' yields are replaced with
// per-field `llzk.nondet : <D x !felt>` placeholders. Old pod-array result
// uses are replaced with a single fresh `llzk.nondet : <D x !pod>` so they
// remain well-typed but orphan.
//
// Pre-existing `flattenPodArrayWhileCarry` only walks scf.while iter-args, so
// scf.ifs whose result type list still contains `<D x !pod>` survived the
// while-carry flatten. AES rounds-loop has scf.ifs nested inside an
// already-flattened outer scf.while whose pod-array result slots blocked the
// per-field carry from threading through to outer scf.yields. After this
// rewrite, the new per-field felt-array result slots become promotable
// carries; downstream LlzkToStablehlo's `extendResultBearingScfIfArrayChain`
// then walks each branch and rewrites the branch yields to use the latest
// per-field SSA values.
//
// Two structs exercise distinct paths:
//
//   @SimpleFlatten        — single pod-array result slot (smoke test of the
//                           basic flatten path).
//   @InterspersedFlatten  — pod-array result slot interleaved with non-pod
//                           result slots, exercising the slot-shift bookkeeping
//                           (`newIdxForOldNonFlattened` + `newStartForFlattened`)
//                           when pod and non-pod slots are not contiguous.

// --- @SimpleFlatten ---
// CHECK-LABEL: struct.def @SimpleFlatten
// The scf.if's result type list now has 2 per-field `<3 x !felt>` slots in
// place of the original single `<3 x !pod<...>>` slot.
// CHECK: %{{.*}}:2 = scf.if
// CHECK-SAME: !array.type<3 x !felt
// CHECK-SAME: !array.type<3 x !felt
// No surviving scf.if result whose type contains a pod element.
// CHECK-NOT: scf.if{{.*}}!array.type<{{[^>]*}} x !pod
// Both branches yield 2 per-field felt-array nondets.
// CHECK: scf.yield %{{[^,]+}}, %{{[^:]+}} : !array.type<3 x !felt
// CHECK: scf.yield %{{[^,]+}}, %{{[^:]+}} : !array.type<3 x !felt

// --- @InterspersedFlatten ---
// Original result list: (!felt, <3 x !pod>, !felt). After flatten, the pod
// slot expands into 2 per-field felts at positions 1,2; the trailing !felt
// slot must shift from index 2 to index 3. The CHECK-SAME chain pins the
// post-flatten result type list as (!felt, <3 x !felt>, <3 x !felt>, !felt).
// CHECK-LABEL: struct.def @InterspersedFlatten
// CHECK: %{{.*}}:4 = scf.if
// CHECK-SAME: !felt
// CHECK-SAME: !array.type<3 x !felt
// CHECK-SAME: !array.type<3 x !felt
// CHECK-SAME: !felt
// Both branches yield (felt, nondet, nondet, felt) — the trailing !felt
// operand at the original slot 2 forwards into the new slot 3.
// CHECK: scf.yield %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^:]+}} :
// CHECK-SAME: !felt
// CHECK-SAME: !array.type<3 x !felt
// CHECK-SAME: !array.type<3 x !felt
// CHECK-SAME: !felt

module attributes {llzk.lang, llzk.main = !struct.type<@SimpleFlatten<[]>>} {
  struct.def @SimpleFlatten {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@SimpleFlatten<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@SimpleFlatten<[]>>
      %f0 = felt.const 0
      %f1 = felt.const 1
      %cond = bool.cmp lt(%f0, %f1)
      %t = scf.if %cond -> (!array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>) {
        %arr_t = array.new : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
        scf.yield %arr_t : !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      } else {
        %arr_e = array.new : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
        scf.yield %arr_e : !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
      }
      struct.writem %self[@out] = %f0 : <@SimpleFlatten<[]>>, !felt.type
      function.return %self : !struct.type<@SimpleFlatten<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@SimpleFlatten<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @InterspersedFlatten {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@InterspersedFlatten<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@InterspersedFlatten<[]>>
      %f0 = felt.const 0
      %f1 = felt.const 1
      %cond = bool.cmp lt(%f0, %f1)
      // 3 results: leading !felt, middle <3 x !pod>, trailing !felt. The pod
      // slot expands at index 1 into 2 per-field felts; the trailing !felt
      // slot must shift from index 2 to index 3.
      %t:3 = scf.if %cond -> (!felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type) {
        %arr_t = array.new : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
        scf.yield %f0, %arr_t, %f1 : !felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type
      } else {
        %arr_e = array.new : <3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>
        scf.yield %f1, %arr_e, %f0 : !felt.type, !array.type<3 x !pod.type<[@a: !felt.type, @b: !felt.type]>>, !felt.type
      }
      // Use the leading and trailing !felt results to keep the slot-shift
      // bookkeeping under live-use pressure (otherwise dead-code elimination
      // could drop them and hide a regression).
      %sum = felt.add %t#0, %t#2 : !felt.type, !felt.type
      struct.writem %self[@out] = %sum : <@InterspersedFlatten<[]>>, !felt.type
      function.return %self : !struct.type<@InterspersedFlatten<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@InterspersedFlatten<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
