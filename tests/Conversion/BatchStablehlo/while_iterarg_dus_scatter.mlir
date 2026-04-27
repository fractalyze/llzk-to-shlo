// RUN: llzk-to-shlo-opt --batch-stablehlo="batch-size=4" %s | FileCheck %s

// Regression for the BatchStablehloPass walk-order bug exposed by AES-256:
// an inner stablehlo.while body whose dynamic_update_slice writes one cell
// of a rank-1 iter_arg at a data-dependent index. Pre-fix, the default
// post-order walk visited the DUS BEFORE batchWhile updated the inner
// while's body block-arg type, so batchDynamicUpdateSliceAsScatter sized
// its one-hot mask from the still-unbatched operand. The body return
// then carried unbatched results that disagreed with the carry types
// batchWhile later promoted, and the verifier rejected the while op as
// "expect operands to be compatible with body block return types".
//
// Post-fix (pre-order walk + scatterDim = batchedIdxPos + 1):
//   - inner while body block-arg type is batched before its body ops run
//   - scatter mask shape derives from the batched operand (4x8, not 8)
//   - select / body return types agree with the batched carry types
//
// The shape mirrors AES @main lines 354/365 — slice one cell from a
// batched lookup table at a batched index, write that cell back into a
// batched accumulator at the same index.

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[TABLE:.*]]: tensor<4x8xi32>, %[[DST:.*]]: tensor<4x8xi32>, %[[IDX:.*]]: tensor<4xi32>) -> tensor<4x8xi32>
//
// CHECK: stablehlo.while
// CHECK-SAME: tensor<4xi32>, tensor<4x8xi32>
//
// One-hot scatter mask sized from the batched accumulator (4x8), not the
// per-witness shape (8):
// CHECK: stablehlo.compare EQ
// CHECK-SAME: -> tensor<4x8xi1>
// CHECK: stablehlo.select
// CHECK-SAME: tensor<4x8xi1>, tensor<4x8xi32>
//
// CHECK: return %{{.*}} : tensor<4x8xi32>

func.func @main(%table: tensor<8xi32>, %dst: tensor<8xi32>, %idx: tensor<i32>)
    -> tensor<8xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c10 = stablehlo.constant dense<10> : tensor<i32>
  %0:2 = stablehlo.while(%i = %c0, %acc = %dst) : tensor<i32>, tensor<8xi32>
    cond {
      %cond = stablehlo.compare LT, %i, %c10
        : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      // Read one cell at a data-dependent index — exercises the gather
      // rewrite. The result rides the batched producer chain, so by the
      // time DUS visits it, %slice is already batched.
      %slice = stablehlo.dynamic_slice %table, %idx, sizes = [1]
        : (tensor<8xi32>, tensor<i32>) -> tensor<1xi32>
      // Write that cell back into the rank-1 iter_arg at the same index —
      // exercises the scatter rewrite under a nested while body, the AES
      // failure path.
      %dus = stablehlo.dynamic_update_slice %acc, %slice, %idx
        : (tensor<8xi32>, tensor<1xi32>, tensor<i32>) -> tensor<8xi32>
      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %inext = stablehlo.add %i, %c1 : tensor<i32>
      stablehlo.return %inext, %dus : tensor<i32>, tensor<8xi32>
    }
  return %0#1 : tensor<8xi32>
}
