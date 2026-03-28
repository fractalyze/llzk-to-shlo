// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// Semantic verification tests: verify StableHLO output produces correct
// computation structure for known circom circuits.

// -----------------------------------------------------------------------
// Sigma circuit: out = x⁵, in2 = x², in4 = x⁴
// Struct layout: [out, in2, in4] → tensor<3x!pf>
//
// Verifies that convertWritemToSSA correctly chains struct.writem ops
// so that the return value reflects ALL writes, not just the first.
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<!pf_babybear_std>) -> tensor<3x!pf_babybear_std>

// CHECK: %[[X2:.*]] = stablehlo.multiply %arg0, %arg0
// in2 = x²
// CHECK: %[[UPD1:.*]] = stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %{{.*}}

// CHECK: %[[X4:.*]] = stablehlo.multiply %[[X2]], %[[X2]]
// in4 = x⁴
// CHECK: %[[UPD2:.*]] = stablehlo.dynamic_update_slice %[[UPD1]], %{{.*}}, %{{.*}}

// CHECK: %[[X5:.*]] = stablehlo.multiply %[[X4]], %arg0
// out = x⁵
// CHECK: %[[UPD3:.*]] = stablehlo.dynamic_update_slice %[[UPD2]], %{{.*}}, %{{.*}}

// Return must be the LAST update (all 3 fields written), not an intermediate.
// CHECK: return %[[UPD3]]

module attributes {llzk.lang, llzk.main = !struct.type<@Sigma_0<[]>>} {
  struct.def @Sigma_0<[]> {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @in2 : !felt.type
    struct.member @in4 : !felt.type
    function.def @compute(%arg0: !felt.type) -> !struct.type<@Sigma_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sigma_0<[]>>
      %0 = felt.mul %arg0, %arg0 : !felt.type, !felt.type
      struct.writem %self[@in2] = %0 : <@Sigma_0<[]>>, !felt.type
      %1 = felt.mul %0, %0 : !felt.type, !felt.type
      struct.writem %self[@in4] = %1 : <@Sigma_0<[]>>, !felt.type
      %2 = felt.mul %1, %arg0 : !felt.type, !felt.type
      struct.writem %self[@out] = %2 : <@Sigma_0<[]>>, !felt.type
      function.return %self : !struct.type<@Sigma_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sigma_0<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
