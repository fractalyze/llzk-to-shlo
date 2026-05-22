// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Phase 4 (`eraseDeadPodAndCountOps`) used to erase any `scf.if` with
// unused results, treating it as count/dispatch bookkeeping. PointCompress's
// nested `@LessThanBounded_5::@compute` puts the comparator's conditional
// witness emission inside such an scf.if:
//     %y = scf.if %cond -> (!felt.type) {
//       struct.writem %self[@out] = %v1
//       scf.yield %v1
//     } else {
//       struct.writem %self[@out] = %v2
//       scf.yield %v2
//     }
// where `%y` is unused. Erasing the scf.if also erases both writems, so
// `collectWritemTargets` returns an empty set, `registerStructFieldOffsets`
// allocates no slot for `@out`, and a downstream `struct.readm @out` in a
// parent's `@compute` fails to legalize with `member offset not found for:
// out`. The fix preserves any `scf.if` (or `scf.for`) whose body contains
// a `struct.writem` writing a felt-typed value — real witness emission.

// CHECK-LABEL: function.def @compute
// CHECK: scf.if
// CHECK: struct.writem
// CHECK: scf.yield
// CHECK: } else {
// CHECK: struct.writem
// CHECK: scf.yield

module attributes {llzk.lang, llzk.main = !struct.type<@CondWritem<[]>>} {
  struct.def @CondWritem {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@CondWritem<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@CondWritem<[]>>
      %felt0 = felt.const 0
      // Dummy pod kept alive by the in-body `pod.read` consumer so that
      // `eliminateInputPods` (which DCEs pod.new chains whose only users
      // are pod.write) leaves it for the dispatch driver to visit. Without
      // any pod op in the @compute block, `eliminatePodDispatch` (and
      // therefore Phase 4) never runs and the regression cannot reproduce.
      %pod = pod.new : !pod.type<[@x: !felt.type]>
      pod.write %pod[@x] = %felt0 : <[@x: !felt.type]>, !felt.type
      %v = pod.read %pod[@x] : <[@x: !felt.type]>, !felt.type
      %cond = bool.cmp lt(%v, %arg0) : !felt.type, !felt.type
      // Yielded `%y` is unused; the only side effect in either branch is
      // the felt-typed `struct.writem %self[@out]`. Without the Phase 4
      // guard the entire scf.if is dropped and `@out` falls out of the
      // writem set.
      %y = scf.if %cond -> (!felt.type) {
        %felt1 = felt.const 1
        struct.writem %self[@out] = %felt1 : <@CondWritem<[]>>, !felt.type
        scf.yield %felt1 : !felt.type
      } else {
        %felt2 = felt.const 0
        struct.writem %self[@out] = %felt2 : <@CondWritem<[]>>, !felt.type
        scf.yield %felt2 : !felt.type
      }
      function.return %self : !struct.type<@CondWritem<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@CondWritem<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
