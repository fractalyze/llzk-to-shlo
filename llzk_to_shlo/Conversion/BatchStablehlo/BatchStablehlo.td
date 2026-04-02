/* Copyright 2026 The llzk-to-shlo Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef LLZK_TO_SHLO_CONVERSION_BATCHSTABLEHLO_BATCHSTABLEHLO_TD_
#define LLZK_TO_SHLO_CONVERSION_BATCHSTABLEHLO_BATCHSTABLEHLO_TD_

include "mlir/Pass/PassBase.td"

def BatchStablehlo : Pass<"batch-stablehlo", "ModuleOp"> {
  let summary = "Add leading batch dimension to StableHLO IR.";
  let description = [{
    Transforms StableHLO IR to process N inputs in parallel by prepending a
    batch dimension to every tensor type.

    Before (single input):
      func.func @main(%x: tensor<M x !pf>) -> tensor<K x !pf>

    After (N inputs batched):
      func.func @main(%x: tensor<N x M x !pf>) -> tensor<N x K x !pf>

    Transformation rules:
    - func.func signature: all tensor args/results get leading dim N
    - stablehlo.constant: wrapped with broadcast_in_dim to add batch dim
    - element-wise ops (add, mul, ...): types updated (auto broadcast)
    - stablehlo.dynamic_slice: batch dim index 0 prepended
    - stablehlo.dynamic_update_slice: batch dim index 0 prepended
    - stablehlo.reshape: target shape gets leading N
    - stablehlo.compare / stablehlo.select: types updated
    - func.call: callee batched in tandem
    - func.return: types updated

    This pass operates on StableHLO IR produced by --llzk-to-stablehlo and
    targets circuits without while loops (Step 1). While loop support is
    planned for a subsequent step.
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
  ];
  let options = [
    Option<"batchSize", "batch-size", "int64_t", /*default=*/"1",
           "Number of inputs to batch (the leading dimension N)">,
  ];
}

#endif  // LLZK_TO_SHLO_CONVERSION_BATCHSTABLEHLO_BATCHSTABLEHLO_TD_
