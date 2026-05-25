// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Regression for mixed batched + constant-index scatter.
//
// The write updates one row selected by a batched index and one column selected
// by a rank-0 constant index. Pre-fix, the scatter rewrite only built the
// one-hot mask for the batched row index and left the constant column to the
// final SelectOp's broadcast, splatting the update across every column.
//
// Post-fix, the constant column participates in the AND-mask the same way
// batchDynamicSliceAsGather already handles mixed indices.
//
// CHECK-LABEL: func.func @mixed_index_scatter
// CHECK-SAME: (%arg0: tensor<4x8x4xi32>, %arg1: tensor<4xi32>, %arg2: tensor<4x1x1xi32>) -> tensor<4x8x4xi32>
// CHECK-DAG: %[[COL:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-DAG: %[[ROW_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<8xi32>
// CHECK-DAG: %[[COL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<4xi32>
// CHECK-DAG: %[[ROW_MASK:.*]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<4x8x4xi32>, tensor<4x8x4xi32>) -> tensor<4x8x4xi1>
// CHECK-DAG: %[[COL_MASK:.*]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<4x8x4xi32>, tensor<4x8x4xi32>) -> tensor<4x8x4xi1>
// CHECK: %[[MASK:.*]] = stablehlo.and %[[ROW_MASK]], %[[COL_MASK]] : tensor<4x8x4xi1>
// CHECK: stablehlo.select %[[MASK]]
// CHECK-SAME: : tensor<4x8x4xi1>, tensor<4x8x4xi32>

func.func @mixed_index_scatter(%operand: tensor<8x4xi32>, %row: tensor<i32>,
                               %update: tensor<1x1xi32>) -> tensor<8x4xi32> {
  %col = stablehlo.constant dense<2> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %operand, %update, %row, %col
      : (tensor<8x4xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<8x4xi32>
  return %0 : tensor<8x4xi32>
}
