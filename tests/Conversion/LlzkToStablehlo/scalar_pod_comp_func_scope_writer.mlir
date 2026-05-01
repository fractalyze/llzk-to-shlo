// RUN: llzk-to-shlo-opt --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression for `materializeScalarPodCompField` function-scope-writer
// shape: 1 in-while writer (@c-fill) + 1 funcBlock-level scf.if writer
// after the while (the @s-write decrement). The substantively-firing call
// is the LATER (function-scope) one — the materializer must accept it as
// a candidate (writerWhile == nullptr), pick it as last-in-source-order,
// and emit the tail call after that scf.if. Mirrors `iden3_get_value_by_index`'s
// @main shape where Num2Bits.out elements are written after the @c-fill.

// CHECK-LABEL: func.func @main
// CHECK: %[[W:.*]]:{{.*}} = stablehlo.while
// Post-while @s update: first operand is the @s-projection of the while
// result (NOT a while-body iterArg), discriminating from the intra-body
// dynamic_update_slice on the @c-fill array.
// CHECK: %[[S_FINAL:.*]] = stablehlo.dynamic_update_slice %[[W]]#{{[0-9]+}}
// CHECK-NEXT: %[[CALL:.*]] = call @Sub_0_compute(%[[W]]#{{[0-9]+}}, %[[S_FINAL]])
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@Main_1<[]>>} {
  struct.def @Sub_0 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%c: !array.type<2 x !felt.type>, %s: !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_0<[]>>
      %nondet = llzk.nondet : !array.type<2 x !felt.type>
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %c0 = array.read %c[%k0] : <2 x !felt.type>, !felt.type
      %s0 = array.read %s[%k0] : <2 x !felt.type>, !felt.type
      %r0 = felt.add %c0, %s0 : !felt.type, !felt.type
      array.write %nondet[%k0] = %r0 : <2 x !felt.type>, !felt.type
      %c1 = array.read %c[%k1] : <2 x !felt.type>, !felt.type
      %s1 = array.read %s[%k1] : <2 x !felt.type>, !felt.type
      %r1 = felt.add %c1, %s1 : !felt.type, !felt.type
      array.write %nondet[%k1] = %r1 : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet : <@Sub_0<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Sub_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_0<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @Main_1 {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<2 x !felt.type>, %arg1: !felt.type) -> !struct.type<@Main_1<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_1<[]>>
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %nondet_out = llzk.nondet : !array.type<2 x !felt.type>
      %pod = llzk.nondet : !pod.type<[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>
      %inputs = pod.new : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>

      // In-while writer: fill @c from %arg0 (2 iters, count countdown).
      %fc0 = felt.const 0
      %0:2 = scf.while (%iter = %fc0, %carry = %inputs) : (!felt.type, !pod.type<[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>) -> (!felt.type, !pod.type<[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>) {
        %fc2 = felt.const 2
        %cond = bool.cmp lt(%iter, %fc2) : !felt.type, !felt.type
        scf.condition(%cond) %iter, %carry : !felt.type, !pod.type<[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>
      } do {
      ^bb0(%iter: !felt.type, %carry: !pod.type<[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>):
        %idx = cast.toindex %iter : !felt.type
        %v = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %ca = pod.read %carry[@c] : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        array.write %ca[%idx] = %v : <2 x !felt.type>, !felt.type
        pod.write %carry[@c] = %ca : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %cnt = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %cntm = arith.subi %cnt, %k1 : index
        pod.write %pod[@count] = %cntm : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cntm, %k0 : index
        scf.if %fire {
          %a = pod.read %carry[@c] : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %b = pod.read %carry[@s] : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
          %r = function.call @Sub_0::@compute(%a, %b) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
          pod.write %pod[@comp] = %r : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
        }
        %fc1 = felt.const 1
        %next = felt.add %iter, %fc1 : !felt.type, !felt.type
        scf.yield %next, %carry : !felt.type, !pod.type<[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>
      }

      // Function-scope writer at funcBlock level. The actual count countdown
      // hits 0 here (both prior in-while iters keep it >0 in this fixture's
      // shape), so the materializer must select THIS writer.
      %sf = pod.read %0#1[@s] : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
      array.write %sf[%k0] = %arg1 : <2 x !felt.type>, !felt.type
      pod.write %0#1[@s] = %sf : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
      %cnt_f = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
      %cnt_fm = arith.subi %cnt_f, %k1 : index
      pod.write %pod[@count] = %cnt_fm : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, index
      %fire_f = arith.cmpi eq, %cnt_fm, %k0 : index
      scf.if %fire_f {
        %a = pod.read %0#1[@c] : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %b = pod.read %0#1[@s] : <[@c: !array.type<2 x !felt.type>, @s: !array.type<2 x !felt.type>]>, !array.type<2 x !felt.type>
        %r = function.call @Sub_0::@compute(%a, %b) : (!array.type<2 x !felt.type>, !array.type<2 x !felt.type>) -> !struct.type<@Sub_0<[]>>
        pod.write %pod[@comp] = %r : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
      }

      %comp = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@Sub_0<[]>>, @params: !pod.type<[]>]>, !struct.type<@Sub_0<[]>>
      %out = struct.readm %comp[@out] : <@Sub_0<[]>>, !array.type<2 x !felt.type>
      %v0 = array.read %out[%k0] : <2 x !felt.type>, !felt.type
      array.write %nondet_out[%k0] = %v0 : <2 x !felt.type>, !felt.type
      %v1 = array.read %out[%k1] : <2 x !felt.type>, !felt.type
      array.write %nondet_out[%k1] = %v1 : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %nondet_out : <@Main_1<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@Main_1<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_1<[]>>, %arg1: !array.type<2 x !felt.type>, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
