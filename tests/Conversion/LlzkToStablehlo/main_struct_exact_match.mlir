// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Regression test: `llzk.main = !struct.type<@MainHelper<[]>>` must not make
// `@Main` look like the entry chip just because its symbol is a substring of
// the printed attribute text.

// CHECK-LABEL: func.func @Main_compute
// CHECK-NOT: func.func @MainHelper_compute
// CHECK-LABEL: func.func @main

module attributes {llzk.lang, llzk.main = !struct.type<@MainHelper<[]>>} {
  struct.def @Main {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Main<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main<[]>>
      struct.writem %self[@out] = %arg0 : <@Main<[]>>, !felt.type
      function.return %self : !struct.type<@Main<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @MainHelper {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !felt.type) -> !struct.type<@MainHelper<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@MainHelper<[]>>
      struct.writem %self[@out] = %arg0 : <@MainHelper<[]>>, !felt.type
      function.return %self : !struct.type<@MainHelper<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@MainHelper<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
