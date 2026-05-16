// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for `materializeStructOfPodsCompField::deriveReaderK` Strategy
// (A) — pod-dispatch chain K extraction.
//
// Scenario: a class instantiated at MULTIPLE K's (multi-K) with a reader
// that lives OUTSIDE any `scf.if` cascade-arm predicate. The existing
// scf.if-walking path (Strategy B, added in PR #118) walks up enclosing
// `scf.if %572 -> ()` predicates of form `arith.cmpi eq %expr, %c<K>`
// to disambiguate K — but post-cascade readers in sibling-while bodies
// (or at the function body top level) have no such predicate. Without
// Strategy (A), deriveReaderK returns nullopt for these readers and
// they stay as `pod.read [@comp] → struct.readm`, lowering to a
// `dense<0>`-fed downstream Sigma call (or whatever consumer reads the
// readm output).
//
// The canonical post-cascade reader shape, as it appears pre-Phase-5
// (and thus when the materializer runs):
//   %slot = array.read %dispatch_arr[%cK]   // or pod.read [@idx_K]
//   %comp = pod.read %slot[@comp]           // record_name = "comp"
//   %out  = struct.readm %comp[@F]          // ← reader
//
// Strategy (A) recovers K from this chain: walk back from the readm's
// operand 0 to find `pod.read [@comp]`, then read its operand's defining
// op (either `array.read` with a constant K or `pod.read [@idx_K]`),
// and pull K out. Gated on K ∈ classToKs[className].
//
// This fixture stages the materializer's input directly at the
// post-Phase-1-hoist canonical form: two writers for the SAME class at
// different K's (forcing classToKs[Sub_A] = {0, 1} — multi-K), plus
// two readers via the array.read → pod.read [@comp] → struct.readm
// chain. The readers live at the function body top level, NOT inside
// any `scf.if`, so Strategy (B) cannot disambiguate K. Without
// Strategy (A) the rewrite would leave both readers as
// `pod.read [@comp] → struct.readm` (and after Phase 5,
// `llzk.nondet → struct.readm`), which then lowers to `dense<0>` and
// erases the `Sub_A` calls as dead. The CHECK lines pin that both
// `Sub_A_compute` calls survive lowering AND that the readers reach
// them via the materialized carrier (not via dead `dense<0>`).

// CHECK-LABEL: func.func @main
// CHECK-DAG: call @Sub_A_compute(
// CHECK-DAG: call @Sub_A_compute(
// CHECK: return

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

      // `materializeStructOfPodsCompField` only runs inside SSC blocks
      // gated on `hasPod = true`; a single live `pod.read` keeps the
      // dummy alive past `eliminateInputPods`'s orphan DCE so the pass
      // fires on this block.
      %dummy = pod.new : <[@d: index]>
      %dummy_read = pod.read %dummy[@d] : <[@d: index]>, index

      // Writers: SAME class `Sub_A` at K=0 and K=1. The materializer's
      // pre-scan + per-writer extractDispatchK pulls K from the
      // `array.extract %arg0[%cK]` operand (case (a)). After this scan
      // `classToKs[Sub_A] = {0, 1}` — multi-K, which means readers
      // cannot be resolved via the `ks.size() == 1` direct-lookup path
      // and MUST call into `deriveReaderK`.
      %ext_a = array.extract %arg0[%c0] : <2, 1 x !felt.type>
      %call_a = function.call @Sub_A::@compute(%ext_a) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>
      %ext_b = array.extract %arg0[%c1] : <2, 1 x !felt.type>
      %call_b = function.call @Sub_A::@compute(%ext_b) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>

      // Synthetic dispatch carrier built post-`convertStructOfPodsToArrayOfPods`
      // (array-of-pods form). The materializer's deriveReaderK Strategy
      // (A) `array.read %arr[%cK]` inner-op path will pull K out of the
      // reader chain below.
      %carrier = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>>

      // Reader at K=0 — NO surrounding `scf.if`. Strategy (B) cannot
      // disambiguate; Strategy (A) walks readm→pod.read[@comp]
      // →array.read[%c0] and recovers K=0.
      %slot_a = array.read %carrier[%c0] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>
      %comp_a = pod.read %slot_a[@comp] : <[@count: index, @comp: !struct.type<@Sub_A<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_A<[]>>
      %read_a = struct.readm %comp_a[@out] : <@Sub_A<[]>>, !felt.type
      array.write %nondet_out[%c0] = %read_a : <2 x !felt.type>, !felt.type

      // Reader at K=1 — same post-cascade shape with `[%c1]`.
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
