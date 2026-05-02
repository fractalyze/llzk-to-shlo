// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression for the AES `aes_256_encrypt` [0,128) ciphertext residual.
//
// `materializePodArrayCompField` has TWO independent emission loops:
//   - Loop A populates per-field felt arrays (`perFieldArrays`) for
//     cross-block `struct.readm @F` readers.
//   - Loop B populates per-drain-target felt arrays (`drainPlans`) for
//     `array.write %destArr[*] = %comp; struct.writem %self[@F'] =
//     %destArr` drain readers.
//
// When a sub-component's @comp result is BOTH cross-block-read (via
// `struct.readm @<inner>`) AND drained to a parent struct array (via
// `struct.writem %self[@F'] = %destArr`), and the inner struct exposes
// a single felt member `@<inner>` that matches the field name of Loop
// A's reader, both loops fire for the SAME hoisted `function.call` —
// each emits an `array.write` into a DIFFERENT fresh felt array at
// IDENTICAL outer indices. After dialect conversion, the duplicate
// `dynamic_update_slice` pair collapses on identical RHS and the
// duplicate clobbers a sibling per-field write that should hold a
// distinct value (e.g. AES round-key bit). See
// memory/aes-encrypt-mod16-residual-followup.md.
//
// The fix reuses `perFieldArrays[innerField]` as `drainPlans[X].destFelt`
// when the element type and shape match, and skips Loop B's duplicate
// `array.write` at the writer site. The parent `struct.writem` is
// redirected to the unified felt array (visibly equal contents). The
// `@constrain` repair walk (struct.readm retype + sibling-call erase)
// works against TYPE shape, not the felt-array SSA identity, so reuse
// is correctness-preserving for the constrain side.
//
// CHECK-LABEL: struct.def @Main_1
// The parent member type is flipped from `array<D x !struct>` to
// `array<D x !felt>` because the drain plan reuses the per-field felt
// array.
// CHECK: struct.member @sub : !array.type<2 x !felt.type>
// CHECK-LABEL: function.def @compute
// One unified felt array allocation: the shared (reader + drain)
// per-field felt array. (The original `%dest_arr = array.new : <2 x
// !struct>` survives temporarily as use-empty and is DCE'd downstream.)
// CHECK: %[[FELT_ARR:.+]] = array.new : <2 x !felt.type>
// At the writer site, exactly ONE `array.write` fires for the shared
// field (Loop A's reader-driven write). Loop B's duplicate `array.write`
// to a sibling fresh allocation is suppressed by the `reusedFromPerField`
// gate.
// CHECK: function.call @Comp_0::@compute
// CHECK-NEXT: struct.readm
// CHECK-NEXT: array.write %[[FELT_ARR]]
// CHECK-NOT: array.write %{{.*}}[%{{.*}}] = %{{.*}} : <2 x !felt.type>
// CHECK: scf.yield
// On the consumer side, both the reader's struct.readm rewrite and the
// parent struct.writem reference the SAME `%[[FELT_ARR]]`.
// CHECK: array.read %[[FELT_ARR]]
// CHECK: struct.writem %{{.*}}[@sub] = %[[FELT_ARR]]

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
    // The drain target — a sibling array of the inner struct type. The
    // dispatch loop drains each `pod.read [@comp]` cell into the @sub
    // member array; the parent `struct.writem` is the drain reader's
    // anchor. The inner struct exposes ONE felt member `@out`, so
    // `findInnerFeltMember` succeeds and Loop B builds a drain plan.
    struct.member @sub : !array.type<2 x !struct.type<@Comp_0<[]>>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !array.type<2 x !felt.type>) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>
      %dest_arr = array.new : <2 x !struct.type<@Comp_0<[]>>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 1 on every dispatch slot.
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %array[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Comp_0<[]>>, @params: !pod.type<[]>]>
      }
      // Writer scf.while: dispatches Comp_0::@compute(arg0[i], arg1[i]).
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
      // Cross-block reader (drives Loop A's perFieldArrays for "out") AND
      // drain reader (drives Loop B's drainPlans into %dest_arr). Both
      // bind on the same @out field, so the fix must reuse the felt array
      // and skip Loop B's duplicate `array.write`.
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
        // Reader-side use of @out (Loop A trigger).
        %out = struct.readm %comp[@out] : <@Comp_0<[]>>, !felt.type
        array.write %nondet_out[%idx] = %out : <2 x !felt.type>, !felt.type
        // Drain-side use of the @comp struct (Loop B trigger via
        // `struct.writem %self[@sub] = %dest_arr`).
        array.write %dest_arr[%idx] = %comp : <2 x !struct.type<@Comp_0<[]>>>, !struct.type<@Comp_0<[]>>
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      struct.writem %self[@sub] = %dest_arr : <@Main_1<[]>>, !array.type<2 x !struct.type<@Comp_0<[]>>>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
