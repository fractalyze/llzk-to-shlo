// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for the maci_splicer input-pod accumulator gap. Shape:
//
//   1. Two function-scope `<N x !pod<[@in: !array<K x !felt>]>>` arrays —
//      mirrors maci_splicer's 4 sub-component input-pod arrays
//      (greaterThan/isLeafIndex/mux/quinSelectors $inputs). Each one's
//      $inputs pod has a single `@in` record (multi-record cases like
//      mux's `@c, @s` and quinSelectors' `@in, @index` go through a
//      different helper path; this test pins the single-`@in` shape that
//      goes through `materializePodArrayInputPodField`).
//   2. The writer scf.while threads BOTH arrays as iter-args. Each loop
//      iteration writes K=2 input slots to BOTH arrays (mirroring the
//      `<==` from outer_idx and arg into both sub-components per iter).
//   3. The dispatch counter (init=K) fires the call once per array per
//      iteration, after both writes.
//
// Without the eraseDead guard for user-input pod.write rewrite-back
// chains:
//
//   - iter 1 of the SimplifySubComponents fixed-point: materialize fires
//     on BOTH arrays' pod.writes (RAUW'ing the firing-site pod.read
//     users), then flatten fires on the FIRST pod-array iter-arg (returns
//     true after one), then processNested → eraseDead erases the SECOND
//     array's pod.writes (now "vacuously unused" since their pod.read
//     users were RAUW'd).
//   - iter 2: flatten on the SECOND pod-array now sees no pod.writes to
//     convert into `array.insert`. The cross-iteration mutation chain is
//     severed for that array — each loop iteration reads the original
//     (zero-init) state.
//
// The fix preserves pod.writes whose `cell` is `array.read` and whose
// field is not @count/@comp/@params. Both arrays' rewrite-back chains
// then survive into their respective flatten iterations and convert
// cleanly into `array.insert` (lowered to dynamic_update_slice on the
// per-field carry tensor).
//
// CHECK pin: BOTH per-iter writes must appear as 2D dynamic_update_slice
// on the carry tensor (`tensor<2x2x...>` with two `tensor<i32>` index
// operands). With the bug, the second array's per-iter write is lost
// entirely — its dynamic_update_slice doesn't appear, and the resulting
// `func.call` consumes a stablehlo.constant zero placeholder instead of
// the staged input value.

// Both arrays must produce a 2D rewrite-back per call site (4 writes
// total = 2 arrays * 2 fire sites). With the bug, the first array's
// pod.writes get erased before flatten can convert them, so only 2 of
// the 4 calls have a preceding dynamic_update_slice on the @in carry
// tensor. The four sequenced CHECKs below require all four to be
// present.

// CHECK-LABEL: func.func @main
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_update_slice {{.*}} : (tensor<2x2x{{[^,]+}}>, tensor<1x2x{{[^,]+}}>, tensor<i32>, tensor<i32>)
// CHECK: stablehlo.dynamic_update_slice {{.*}} : (tensor<2x2x{{[^,]+}}>, tensor<1x2x{{[^,]+}}>, tensor<i32>, tensor<i32>)
// CHECK: stablehlo.dynamic_update_slice {{.*}} : (tensor<2x2x{{[^,]+}}>, tensor<1x2x{{[^,]+}}>, tensor<i32>, tensor<i32>)
// CHECK: stablehlo.dynamic_update_slice {{.*}} : (tensor<2x2x{{[^,]+}}>, tensor<1x2x{{[^,]+}}>, tensor<i32>, tensor<i32>)
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  // Sub_0::@compute takes a 2-element array (mirrors GreaterThan's `in[2]`).
  struct.def @Sub_0 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %a0 = array.read %arg0[%c0] : <2 x !felt.type>, !felt.type
      %a1 = array.read %arg0[%c1] : <2 x !felt.type>, !felt.type
      %sum = felt.add %a0, %a1 : !felt.type, !felt.type
      struct.writem %self[@out] = %sum : <@Sub_0<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_0<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Main_1 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    struct.member @sub_a$inputs : !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>
    struct.member @sub_b$inputs : !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %arr_a_count = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>
      %arr_a_in = array.new : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>
      %arr_b_count = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>
      %arr_b_in = array.new : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // Initialize @count = 2 on every dispatch slot for both arrays.
      scf.for %i = %c0 to %c2 step %c1 {
        %pa = array.read %arr_a_count[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        pod.write %pa[@count] = %c2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %arr_a_count[%i] = %pa : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %pb = array.read %arr_b_count[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        pod.write %pb[@count] = %c2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        array.write %arr_b_count[%i] = %pb : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      }
      // Outer scf.while threads BOTH @in pod-arrays + their dispatch
      // arrays. Each loop iteration performs the K=2 write pattern on
      // BOTH arrays. The flatten pass processes one pod-array per call
      // and returns; intervening eraseDead would clobber the OTHER's
      // pod.writes without the fix.
      %felt0 = felt.const 0
      %0:3 = scf.while (%iter = %felt0, %arr_a_iter = %arr_a_in, %arr_b_iter = %arr_b_in)
          : (!felt.type, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>)
          -> (!felt.type, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>) {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter, %arr_a_iter, %arr_b_iter
            : !felt.type, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>
      } do {
      ^bb0(%iter: !felt.type, %arr_a_iter: !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, %arr_b_iter: !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>):
        %idx = cast.toindex %iter
        %felt7 = felt.const 7
        %felt11 = felt.const 11

        // ===== Array A: 2 writes + count countdown + fire =====
        // Slot 0:
        %ca0 = array.read %arr_a_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %sa0 = pod.read %ca0[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %sa0[%c0] = %iter : <2 x !felt.type>, !felt.type
        %ca0b = array.read %arr_a_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        pod.write %ca0b[@in] = %sa0 : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %arr_a_iter[%idx] = %ca0b : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %dpa0 = array.read %arr_a_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %cna0 = pod.read %dpa0[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cna0b = arith.subi %cna0, %c1 : index
        pod.write %dpa0[@count] = %cna0b : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fa0 = arith.cmpi eq, %cna0b, %c0 : index
        scf.if %fa0 {
          %ra = pod.read %ca0b[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %rsa = function.call @Sub_0::@compute(%ra) : (!array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %dpa0[@comp] = %rsa : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
          array.write %arr_a_count[%idx] = %dpa0 : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        }
        // Slot 1:
        %ca1 = array.read %arr_a_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %sa1 = pod.read %ca1[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %sa1[%c1] = %felt7 : <2 x !felt.type>, !felt.type
        %ca1b = array.read %arr_a_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        pod.write %ca1b[@in] = %sa1 : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %arr_a_iter[%idx] = %ca1b : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %dpa1 = array.read %arr_a_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %cna1 = pod.read %dpa1[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cna1b = arith.subi %cna1, %c1 : index
        pod.write %dpa1[@count] = %cna1b : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fa1 = arith.cmpi eq, %cna1b, %c0 : index
        scf.if %fa1 {
          %rab = pod.read %ca1b[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %rsab = function.call @Sub_0::@compute(%rab) : (!array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %dpa1[@comp] = %rsab : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
          array.write %arr_a_count[%idx] = %dpa1 : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        }

        // ===== Array B: 2 writes + count countdown + fire =====
        // Slot 0:
        %cb0 = array.read %arr_b_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %sb0 = pod.read %cb0[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %sb0[%c0] = %iter : <2 x !felt.type>, !felt.type
        %cb0b = array.read %arr_b_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        pod.write %cb0b[@in] = %sb0 : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %arr_b_iter[%idx] = %cb0b : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %dpb0 = array.read %arr_b_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %cnb0 = pod.read %dpb0[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnb0b = arith.subi %cnb0, %c1 : index
        pod.write %dpb0[@count] = %cnb0b : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fb0 = arith.cmpi eq, %cnb0b, %c0 : index
        scf.if %fb0 {
          %rb = pod.read %cb0b[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %rsb = function.call @Sub_0::@compute(%rb) : (!array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %dpb0[@comp] = %rsb : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
          array.write %arr_b_count[%idx] = %dpb0 : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        }
        // Slot 1:
        %cb1 = array.read %arr_b_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %sb1 = pod.read %cb1[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %sb1[%c1] = %felt11 : <2 x !felt.type>, !felt.type
        %cb1b = array.read %arr_b_iter[%idx] : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        pod.write %cb1b[@in] = %sb1 : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %arr_b_iter[%idx] = %cb1b : <2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !pod.type<[@in: !array.type<2 x !felt.type>]>
        %dpb1 = array.read %arr_b_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %cnb1 = pod.read %dpb1[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cnb1b = arith.subi %cnb1, %c1 : index
        pod.write %dpb1[@count] = %cnb1b : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fb1 = arith.cmpi eq, %cnb1b, %c0 : index
        scf.if %fb1 {
          %rbb = pod.read %cb1b[@in] : <[@in: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %rsbb = function.call @Sub_0::@compute(%rbb) : (!array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %dpb1[@comp] = %rsbb : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
          array.write %arr_b_count[%idx] = %dpb1 : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        }

        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next, %arr_a_iter, %arr_b_iter
            : !felt.type, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>, !array.type<2 x !pod.type<[@in: !array.type<2 x !felt.type>]>>
      }

      // Reader: walks both count arrays to recover @out into %nondet_out.
      // We sum a + b at each slot, but for the CHECK we just need both
      // calls' results to flow somewhere.
      %felt0_2 = felt.const 0
      %1 = scf.while (%iter = %felt0_2) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %dpa = array.read %arr_a_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %compa = pod.read %dpa[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %outa = struct.readm %compa[@out] : <@Sub_0<[]>>, !felt.type
        %dpb = array.read %arr_b_count[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
        %compb = pod.read %dpb[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        %outb = struct.readm %compb[@out] : <@Sub_0<[]>>, !felt.type
        %sum = felt.add %outa, %outb : !felt.type, !felt.type
        array.write %nondet_out[%idx] = %sum : <2 x !felt.type>, !felt.type
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
