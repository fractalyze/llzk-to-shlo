// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Regression for the symmetric DS gap to PR #27's DUS multi-index scatter
// fix. AES-256-key-expansion's round-key derivation
// `dynamic_slice %scratch, %i, %c0, sizes=[1, 1] : (tensor<60x4>,
// tensor<i32>, tensor<i32>) -> tensor<1x1>` mixes one batched index with
// one rank-0 constant index. Pre-fix, `batchDynamicSliceAsGather`
// `continue`d on rank-0 indices, so the constant-index dim leaked through
// the reduce and the trailing reshape rejected `tensor<N x D>` →
// `tensor<N x 1 x 1>`.

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[TABLE:.*]]: tensor<4x60x4xi32>, %[[I:.*]]: tensor<4xi32>) -> tensor<4x1x1xi32>

// Two-dim AND-mask reduction:
// CHECK: stablehlo.compare  EQ
// CHECK-SAME: -> tensor<4x60x4xi1>
// CHECK: stablehlo.compare  EQ
// CHECK-SAME: -> tensor<4x60x4xi1>
// CHECK: stablehlo.and {{.*}} : tensor<4x60x4xi1>
// CHECK: stablehlo.reduce
// CHECK-SAME: dimensions = [1, 2]
// CHECK: stablehlo.reshape
// CHECK-SAME: (tensor<4xi32>) -> tensor<4x1x1xi32>

func.func @main(%table: tensor<60x4xi32>, %i: tensor<i32>) -> tensor<1x1xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %s = stablehlo.dynamic_slice %table, %i, %c0, sizes = [1, 1]
      : (tensor<60x4xi32>, tensor<i32>, tensor<i32>) -> tensor<1x1xi32>
  return %s : tensor<1x1xi32>
}

// -----

// Full-axis row gather: `sizes=[1, dimSize]` keeps the const-index dim
// intact in the result. Mirrors AES-256-key-expansion's
// `dynamic_slice %scratch_60x32, %i, %c0, sizes=[1, 32]`.

// CHECK-LABEL: func.func @row_gather
// CHECK-SAME: (%[[TABLE:.*]]: tensor<4x60x32xi32>, %[[I:.*]]: tensor<4xi32>) -> tensor<4x1x32xi32>

// Single-dim AND-mask: only the batched dim is masked + reduced; dim 2
// (size 32) stays intact.
// CHECK: stablehlo.compare  EQ
// CHECK-SAME: -> tensor<4x60x32xi1>
// CHECK-NOT: stablehlo.and
// CHECK: stablehlo.reduce
// CHECK-SAME: dimensions = [1]
// CHECK: stablehlo.reshape
// CHECK-SAME: (tensor<4x32xi32>) -> tensor<4x1x32xi32>

func.func @row_gather(%table: tensor<60x32xi32>, %i: tensor<i32>)
    -> tensor<1x32xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %s = stablehlo.dynamic_slice %table, %i, %c0, sizes = [1, 32]
      : (tensor<60x32xi32>, tensor<i32>, tensor<i32>) -> tensor<1x32xi32>
  return %s : tensor<1x32xi32>
}
