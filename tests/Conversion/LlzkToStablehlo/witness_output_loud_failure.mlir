// RUN: not llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32 flag-orphan-zero-writes=true" %s 2>&1 | FileCheck %s
// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s -o /dev/null

// Two RUN lines pin the orphan-zero heuristic's contract end-to-end:
//   - Line 1 (opt-in `flag-orphan-zero-writes=true`): the splat-zero
//     `struct.writem` operand still aborts conversion with the named
//     diagnostic. Wave 1 regression fixture for `aes_256_encrypt`'s
//     `@num2bits_2`/`@xor_1$inputs` orphans.
//   - Line 2 (default off post-WLA+VerifyWLA migration): the same input
//     compiles silently. VerifyWLA, not this heuristic, is now the
//     authoritative orphan check.

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
