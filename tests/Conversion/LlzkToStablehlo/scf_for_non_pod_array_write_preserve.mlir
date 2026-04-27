// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Phase 4 (`eraseDeadPodAndCountOps`) used to erase any `scf.for` with
// unused results, on the assumption that `scf.for` here is dispatch
// counting boilerplate. AES sub-component aggregation fields (e.g.
// `@bits2num_1`, `@xor_2`) emit 0-iter-arg `scf.for` fill loops that
// drain `pod.read [@comp]` results into a felt / struct-element array
// — real witness emission. The fix preserves any `scf.for` whose body
// contains an `array.write` whose target array's element type is not a
// `!pod.type`. Mirrors the AES Loop B shape: the scf.for body holds a
// surviving `pod.read` (kept alive on entry by the dummy pod write) and
// an `array.write` to a felt array (the witness target).

// CHECK-LABEL: function.def @compute
// CHECK: scf.for
// CHECK: array.write
// CHECK: struct.writem

module attributes {llzk.lang, llzk.main = !struct.type<@FillFor<[]>>} {
  struct.def @FillFor {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<4 x !felt.type>) -> !struct.type<@FillFor<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@FillFor<[]>>
      %nondet_out = llzk.nondet : !array.type<4 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      // Dummy pod kept alive by the in-loop `pod.read` consumer so that
      // `eliminateInputPods` (which DCEs pod.new chains whose only users
      // are pod.write) leaves it for the dispatch driver to visit.
      %pod = pod.new : !pod.type<[@x: !felt.type]>
      %felt0 = felt.const 0
      pod.write %pod[@x] = %felt0 : <[@x: !felt.type]>, !felt.type
      // 0-iter-arg fill loop. Body's `array.write` targets %nondet_out
      // (element type !felt — non-pod), which must be preserved. Without
      // the Phase 4 guard the entire scf.for is dropped, the array.write
      // disappears, and %nondet_out remains uninitialized.
      scf.for %i = %c0 to %c4 step %c1 {
        %v = pod.read %pod[@x] : <[@x: !felt.type]>, !felt.type
        array.write %nondet_out[%i] = %v : <4 x !felt.type>, !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@FillFor<[]>>, !array.type<4 x !felt.type>
      function.return %self : !struct.type<@FillFor<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@FillFor<[]>>, %arg1: !array.type<4 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
