// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for `materializeStructOfPodsCompField` (commit d2e5d3f).
//
// Scenario: a non-uniform-inner struct-of-pods dispatch carrier
// `!pod<[@idx_0: !struct<@Sub_A>, @idx_1: !struct<@Sub_B>]>` where each
// `@idx_K` resolves to a DISTINCT sub-component class. The uniform-inner
// matcher in `convertStructOfPodsToArrayOfPods` cannot fold this shape
// (no single inner type can represent both `@Sub_A` and `@Sub_B`), so
// the dispatch carrier survives the rewrite, `processNested`'s
// `hasPodBlockArg=true` branch runs Phase 1 (`extractCallsFromScfIf`)
// with a local tracker to hoist each dispatched `function.call
// @Sub_K::@compute(%array.extract %carrier[%cK])` out of its
// statically-false dispatch scf.if, and then
// `materializeStructOfPodsCompField` must close the writer↔reader link
// by allocating a parallel per-`@F` carrier of shape `<N x ...inner>`,
// filling it from the hoisted call results, and rewiring reader-side
// `struct.readm %nondet[@F]` chains to read from it.
//
// The matcher fires at *post-Phase-1-hoist* canonical form:
//   - Writer: `function.call @<C>::@compute(%v)` with
//     `%v = array.extract %arr[%cK]` and `%cK = arith.constant <k>`.
//   - Reader: `llzk.nondet : !struct<@<C>>` with a downstream
//     `struct.readm %nondet[@F]` consumer.
// This fixture seeds the IR at that canonical form directly so the test
// exercises the materializer's emission contract independent of the
// upstream dispatch scaffolding.
//
// Without the materializer, both `func.call @Sub_A_compute` and
// `func.call @Sub_B_compute` survive lowering but have no consumer —
// the reader-side `struct.readm` cascades feed off `llzk.nondet` only,
// which lowers to `dense<0>`. Per-position output reads as zero. The
// fix routes both call results through a per-`@out` carrier that the
// readers consume.
//
// CHECK lines pin the contract that fails without the fix: both
// `func.call @Sub_*_compute(...)` survive lowering inside `@main`. Without
// the materializer, the calls have no consumer (the readers `struct.readm`
// off a `llzk.nondet`, which lowers to `dense<0>`) and standard DCE
// erases them — `@main`'s output is then assembled entirely from constant
// zeros and the `%[[A:.*]] =` / `%[[B:.*]] =` patterns never appear.

// CHECK-LABEL: func.func @main
// CHECK-DAG: call @Sub_A_compute(
// CHECK-DAG: call @Sub_B_compute(
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
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // The materializer only runs inside SSC blocks gated on `hasPod = true`.
      // `eliminateInputPods`'s orphan DCE erases pod.news whose only users are
      // pod.writes (or that have no users); a single pod.read keeps the dummy
      // alive past that pre-step. The materializer runs, then later
      // `eliminatePodDispatch` Phase 4 erases this dummy as dead. The real
      // production trigger is the unflattened dispatch scf.while + scf.if
      // cascade described in the file-level comment; omitting that scaffolding
      // here keeps the fixture readable.
      %dummy = pod.new : <[@d: index]>
      %dummy_read = pod.read %dummy[@d] : <[@d: index]>, index
      // Synthetic carrier feeding both writers via `array.extract` — the
      // post-Phase-1-hoist canonical shape after `extractCallsFromScfIf`
      // lifts the dispatched calls out of their dispatch scf.ifs.
      // Writer A: distinct class Sub_A at K=0.
      %ext_a = array.extract %arg0[%c0] : <2, 1 x !felt.type>
      %call_a = function.call @Sub_A::@compute(%ext_a) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>
      // Writer B: distinct class Sub_B at K=1 — non-uniform with A.
      %ext_b = array.extract %arg0[%c1] : <2, 1 x !felt.type>
      %call_b = function.call @Sub_B::@compute(%ext_b) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_B<[]>>
      // Reader A: nondet of Sub_A + readm @out — the writer↔reader gap that
      // the materializer must close by routing through the parallel carrier.
      %ndet_a = llzk.nondet : !struct.type<@Sub_A<[]>>
      %read_a = struct.readm %ndet_a[@out] : <@Sub_A<[]>>, !felt.type
      array.write %nondet_out[%c0] = %read_a : <2 x !felt.type>, !felt.type
      // Reader B: nondet of Sub_B + readm @out.
      %ndet_b = llzk.nondet : !struct.type<@Sub_B<[]>>
      %read_b = struct.readm %ndet_b[@out] : <@Sub_B<[]>>, !felt.type
      array.write %nondet_out[%c1] = %read_b : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet_out : <@Main_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_0<[]>>, %arg1: !array.type<2, 1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
