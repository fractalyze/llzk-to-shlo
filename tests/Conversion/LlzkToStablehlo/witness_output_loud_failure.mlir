// RUN: not llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s 2>&1 | FileCheck %s
// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32 flag-orphan-zero-writes=false" %s -o /dev/null

// Witness-output loud failure: when a struct.member's writem operand traces
// back to a splat-zero stablehlo.constant of length >= 8, the build is aborted
// with a diagnostic. The chunk would otherwise be silently zeroed in the
// witness tensor even though the LLZK declares the member as written, hiding
// an upstream pass that orphaned the wire.
//
// Caught AES `aes_256_encrypt`'s @num2bits_2 (length 16) and @xor_1$inputs
// (length 208) chunks — previously surfaced only as a gate-test mismatch,
// with no signal pointing at the missing wire (~5 sessions of debug drift
// across SimplifySubComponents Stages 7..12).
//
// Heuristic: the length >= 8 threshold lets existing fixtures with small
// intentional zero writes keep passing (felt_const_edge length=1, felt_nondet
// length=4, dispatch_while_body @sub length=2). A future anchor + verify pass
// pair (T3+T4 in llzk-to-shlo-architectural-fix-track-decomposition.md)
// supersedes this with an airtight per-anchor check.

// CHECK: error: witness-output: silent dense<0> fallback for struct.member @out
// CHECK-SAME: offset=0
// CHECK-SAME: length=8

module attributes {llzk.lang, llzk.main = !struct.type<@OrphanWire<[]>>} {
  struct.def @OrphanWire {
    struct.member @out : !array.type<8 x !felt.type> {llzk.pub}
    function.def @compute() -> !struct.type<@OrphanWire<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@OrphanWire<[]>>
      // Synthetic orphan: array.new with no element args lowers to a
      // splat-zero stablehlo.constant. In a real circuit this happens when
      // an upstream pass replaces the writem's wire with a fresh zero array.
      %arr = array.new : <8 x !felt.type>
      struct.writem %self[@out] = %arr : <@OrphanWire<[]>>, !array.type<8 x !felt.type>
      function.return %self : !struct.type<@OrphanWire<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@OrphanWire<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
