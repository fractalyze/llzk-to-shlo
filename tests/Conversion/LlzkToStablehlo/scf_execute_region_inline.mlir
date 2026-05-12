// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression: scf.execute_region wrappers around a felt-typed body with
// a single scf.yield must be inlined post-conversion. LLZK's `<--`
// returning a felt (e.g. the 67-way Ark-constant lookup in
// Poseidon-using chips) emits a felt-array dispatch cascade wrapped in
// scf.execute_region. Main conversion lowers the inner ops to stablehlo
// but no pattern dissolves the wrapper; survivors then block
// stablehlo_runner's module load with "Dialect `scf' not found for
// custom op 'scf.execute_region'".

// CHECK-LABEL: func.func @main
// CHECK-NOT: scf.execute_region
// CHECK-NOT: scf.yield

module attributes {llzk.lang, llzk.main = !struct.type<@MinExecRegion<[]>>} {
  struct.def @MinExecRegion {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@MinExecRegion<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@MinExecRegion<[]>>
      %r = scf.execute_region -> !felt.type {
        %tmp = felt.add %a, %b : !felt.type, !felt.type
        scf.yield %tmp : !felt.type
      }
      struct.writem %self[@out] = %r : <@MinExecRegion<[]>>, !felt.type
      function.return %self : !struct.type<@MinExecRegion<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@MinExecRegion<[]>>, %arg1: !felt.type, %arg2: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
