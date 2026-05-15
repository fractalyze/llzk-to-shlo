// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Contract test for `extractCallsFromScfIf`'s positional-dominance guard
// on resolved call args. The structural shape is one carrier pod with two
// scalar fields shared across two sibling dispatch scf.ifs at funcBlock
// scope, with `pod.write %carrier[@F] = %v` writes interleaved between
// them — the shape that recurs across iden3's `SMTVerifier_*` /
// `verifyClaimSignature_*` dispatch chains.
//
// What this fixture pins: SSC handles the multi-dispatch + interleaved-
// write shape without over-rejecting valid hoists. Both function.call
// sites lift out of their scf.ifs, neither operand crosses an scf.if
// boundary in the wrong direction.
//
// What this fixture does NOT pin: a minimal hand-crafted IR that
// actively triggers the dominance violation. The bug surfaces in the
// full iden3 dispatch chains (multi-iteration outer fixed-point through
// scf.while iter-args + transitive tracker resolution); the minimal
// 2-dispatch form here happens to converge cleanly with or without the
// guard. The 13 iden3 chips in `examples/BUILD.bazel`
// (`iden3_auth*`, `iden3_query_mtp*`, `iden3_query_sig*`,
// `iden3_id_ownership_sig*`, `iden3_state_transition*`) are the tight
// regression coverage — their `circom_to_stablehlo` builds fail mid-SSC
// with `operand does not dominate this use` when the guard is reverted.

// CHECK-LABEL: struct.def @Main_1
// CHECK: function.def @compute
// CHECK-SAME: !struct.type<@Main_1
// CHECK: function.call @Sub_0::@compute(%arg0, %arg1)
// CHECK: function.call @Sub_0::@compute({{.*}}felt_const{{.*}}
// CHECK-NOT: scf.if
// CHECK: function.return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Sub_0 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@Sub_0<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
      %0 = felt.add %arg0, %arg1 : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Sub_0<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_0<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @Main_1 {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type, %arg1: !felt.type) -> !struct.type<@Main_1<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index

      // Dispatch pod for @Sub_0 with count=2 (fires twice).
      %pod_count = pod.new { @count = %c2 } : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      // Carrier pod sharing two scalar input fields across both dispatches.
      %carrier = pod.new : <[@x: !felt.type, @y: !felt.type]>

      // Initial writes BEFORE the first scf.if.
      pod.write %carrier[@x] = %arg0 : <[@x: !felt.type, @y: !felt.type]>, !felt.type
      pod.write %carrier[@y] = %arg1 : <[@x: !felt.type, @y: !felt.type]>, !felt.type

      // First dispatch.
      %cnt1 = pod.read %pod_count[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
      %cnt1m = arith.subi %cnt1, %c1 : index
      pod.write %pod_count[@count] = %cnt1m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
      %fire1 = arith.cmpi eq, %cnt1m, %c0 : index
      scf.if %fire1 {
        %x1 = pod.read %carrier[@x] : <[@x: !felt.type, @y: !felt.type]>, !felt.type
        %y1 = pod.read %carrier[@y] : <[@x: !felt.type, @y: !felt.type]>, !felt.type
        %res1 = function.call @Sub_0::@compute(%x1, %y1) : (!felt.type, !felt.type) -> !struct.type<@Sub_0<[]>>
        pod.write %pod_count[@comp] = %res1 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
      }

      // Inter-dispatch writes: block-scope `felt.const` ops defined
      // AFTER the first scf.if, so they do NOT dominate it.
      %felt_zero = felt.const 0 : !felt.type
      pod.write %carrier[@x] = %felt_zero : <[@x: !felt.type, @y: !felt.type]>, !felt.type
      %felt_one = felt.const 1 : !felt.type
      pod.write %carrier[@y] = %felt_one : <[@x: !felt.type, @y: !felt.type]>, !felt.type

      // Second dispatch.
      %cnt2 = pod.read %pod_count[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
      %cnt2m = arith.subi %cnt2, %c1 : index
      pod.write %pod_count[@count] = %cnt2m : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
      %fire2 = arith.cmpi eq, %cnt2m, %c0 : index
      scf.if %fire2 {
        %x2 = pod.read %carrier[@x] : <[@x: !felt.type, @y: !felt.type]>, !felt.type
        %y2 = pod.read %carrier[@y] : <[@x: !felt.type, @y: !felt.type]>, !felt.type
        %res2 = function.call @Sub_0::@compute(%x2, %y2) : (!felt.type, !felt.type) -> !struct.type<@Sub_0<[]>>
        pod.write %pod_count[@comp] = %res2 : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
      }

      // Drain the dispatch pod's @comp into @main's output.
      %comp = pod.read %pod_count[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
      %out = struct.readm %comp[@out] : <@Sub_0<[]>>, !felt.type
      struct.writem %self[@out] = %out : <@Main_1<[]>>, !felt.type
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
