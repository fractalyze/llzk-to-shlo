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
// out`. The fix preserves an `scf.if`/`scf.for` whose body contains a
// `struct.writem` writing a felt-typed value — but ONLY when the target
// member is read by some function other than the owning struct's
// `@constrain` (the cross-function liveness gate prevents the inverse
// regression on chips like standalone `lessthan_bounded`, where the only
// reader is the doomed `@constrain` and preserving the scf.if would
// allocate a phantom witness slot that breaks the m3 correctness gate
// against circom-native).

// CHECK-LABEL: struct.def @CondWritem
// CHECK-LABEL: function.def @compute
// CHECK: scf.if
// CHECK: struct.writem
// CHECK: scf.yield
// CHECK: } else {
// CHECK: struct.writem
// CHECK: scf.yield

module attributes {llzk.lang, llzk.main = !struct.type<@Parent<[]>>} {
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

  // Parent struct's @compute reads `@out` externally — this is the
  // cross-function liveness signal that gates the preservation. Without
  // this reader, `@out` is only consumed by `@CondWritem::@constrain` (a
  // doomed function), and Phase 4 correctly erases the scf.if to avoid
  // allocating a phantom witness slot.
  struct.def @Parent {
    struct.member @y : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Parent<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Parent<[]>>
      %child = function.call @CondWritem::@compute(%arg0) : (!felt.type) -> !struct.type<@CondWritem<[]>>
      %out_v = struct.readm %child[@out] : <@CondWritem<[]>>, !felt.type
      struct.writem %self[@y] = %out_v : <@Parent<[]>>, !felt.type
      function.return %self : !struct.type<@Parent<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Parent<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
