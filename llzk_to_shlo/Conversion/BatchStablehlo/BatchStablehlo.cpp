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

#include "llzk_to_shlo/Conversion/BatchStablehlo/BatchStablehlo.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_BATCHSTABLEHLO
#include "llzk_to_shlo/Conversion/BatchStablehlo/BatchStablehlo.h.inc"

namespace {

// ===----------------------------------------------------------------------===
// Helpers
// ===----------------------------------------------------------------------===

/// Prepend batch dimension N to a RankedTensorType.
///   tensor<!pf>        → tensor<Nx!pf>
///   tensor<Mx!pf>      → tensor<NxMx!pf>
///   tensor<MxKx!pf>    → tensor<NxMxKx!pf>
RankedTensorType addBatchDim(RankedTensorType type, int64_t batchSize) {
  SmallVector<int64_t> newShape;
  newShape.push_back(batchSize);
  newShape.append(type.getShape().begin(), type.getShape().end());
  return RankedTensorType::get(newShape, type.getElementType());
}

/// Add batch dimension to a type if it is a RankedTensorType.
Type batchType(Type type, int64_t batchSize) {
  if (auto rtt = dyn_cast<RankedTensorType>(type))
    return addBatchDim(rtt, batchSize);
  return type;
}

/// Check if an op is a StableHLO element-wise op (operates independently on
/// each element, so batching only requires changing the type).
bool isElementWiseOp(Operation *op) {
  return isa<stablehlo::AbsOp>(op) || isa<stablehlo::NegOp>(op) ||
         isa<stablehlo::ConvertOp>(op) || isa<stablehlo::AddOp>(op) ||
         isa<stablehlo::SubtractOp>(op) || isa<stablehlo::MulOp>(op) ||
         isa<stablehlo::DivOp>(op) || isa<stablehlo::RemOp>(op) ||
         isa<stablehlo::PowOp>(op) || isa<stablehlo::MaxOp>(op) ||
         isa<stablehlo::MinOp>(op) || isa<stablehlo::AndOp>(op) ||
         isa<stablehlo::OrOp>(op) || isa<stablehlo::XorOp>(op) ||
         isa<stablehlo::ShiftRightLogicalOp>(op) ||
         isa<stablehlo::ShiftLeftOp>(op) || isa<stablehlo::NotOp>(op);
}

// ===----------------------------------------------------------------------===
// Pass implementation
// ===----------------------------------------------------------------------===

class BatchStablehloPass : public impl::BatchStablehloBase<BatchStablehloPass> {
  using BatchStablehloBase::BatchStablehloBase;

  void runOnOperation() override {
    auto module = getOperation();
    int64_t N = batchSize;

    if (N <= 0) {
      module.emitError("batch-size must be positive, got ") << N;
      return signalPassFailure();
    }

    // Trivial: batch-size=1 is a no-op.
    if (N == 1)
      return;

    // Process each function in the module.
    auto result = module.walk([&](func::FuncOp funcOp) -> WalkResult {
      if (failed(batchFunction(funcOp, N)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

  /// Batch a single function: update signature and transform all ops.
  LogicalResult batchFunction(func::FuncOp funcOp, int64_t N) {
    // 1. Update function signature.
    auto funcType = funcOp.getFunctionType();
    SmallVector<Type> newInputTypes, newResultTypes;

    for (Type t : funcType.getInputs())
      newInputTypes.push_back(batchType(t, N));
    for (Type t : funcType.getResults())
      newResultTypes.push_back(batchType(t, N));

    funcOp.setFunctionType(
        FunctionType::get(funcOp.getContext(), newInputTypes, newResultTypes));

    // 2. Update block argument types.
    for (auto &block : funcOp.getBody()) {
      for (auto arg : block.getArguments()) {
        if (auto rtt = dyn_cast<RankedTensorType>(arg.getType()))
          arg.setType(addBatchDim(rtt, N));
      }
    }

    // 3. Walk all ops in the function body and transform them.
    //    Two passes: first handle structural ops (skip constants), then batch
    //    constants. This avoids wrongly batching index constants used by
    //    dynamic_slice/dynamic_update_slice.
    //    Pre-order so an enclosing stablehlo.while is visited before its body
    //    ops — batchWhile updates the body's block-arg types, which inner ops
    //    like batchDynamicUpdateSliceAsScatter then read to size mask/iota.
    SmallVector<Operation *> ops;
    SmallVector<stablehlo::ConstantOp> constants;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == funcOp.getOperation())
        return;
      if (auto constOp = dyn_cast<stablehlo::ConstantOp>(op))
        constants.push_back(constOp);
      else
        ops.push_back(op);
    });

    // Pass 1: batch all non-constant ops.
    for (auto *op : ops) {
      if (failed(batchOp(op, N)))
        return failure();
    }

    // Pass 2: batch constants, but only for non-index uses.
    for (auto constOp : constants) {
      if (failed(batchConstant(constOp, N)))
        return failure();
    }

    return success();
  }

  /// Batch a single operation.
  LogicalResult batchOp(Operation *op, int64_t N) {
    // func.return: just update operand types (already batched from producers).
    if (isa<func::ReturnOp>(op))
      return success();

    // stablehlo.constant: handled in a separate pass (see batchFunction).
    if (isa<stablehlo::ConstantOp>(op))
      return success();

    // Element-wise ops: update result types only.
    if (isElementWiseOp(op))
      return batchElementWise(op, N);

    // stablehlo.compare: update operand handling (result is always tensor<i1>
    // variant with batch dim).
    if (auto cmpOp = dyn_cast<stablehlo::CompareOp>(op))
      return batchCompare(cmpOp, N);

    // stablehlo.select: update result type.
    if (auto selectOp = dyn_cast<stablehlo::SelectOp>(op))
      return batchSelect(selectOp, N);

    // stablehlo.broadcast_in_dim: update for batch dim.
    if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op))
      return batchBroadcastInDim(bcastOp, N);

    // stablehlo.dynamic_slice: prepend batch dim index.
    if (auto sliceOp = dyn_cast<stablehlo::DynamicSliceOp>(op))
      return batchDynamicSlice(sliceOp, N);

    // stablehlo.dynamic_update_slice: prepend batch dim index.
    if (auto dusOp = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op))
      return batchDynamicUpdateSlice(dusOp, N);

    // stablehlo.reshape: update target shape with leading N.
    if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op))
      return batchReshape(reshapeOp, N);

    // stablehlo.slice (static): update for batch dim.
    if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(op))
      return batchStaticSlice(sliceOp, N);

    // stablehlo.concatenate: update for batch dim.
    if (auto concatOp = dyn_cast<stablehlo::ConcatenateOp>(op))
      return batchConcatenate(concatOp, N);

    // stablehlo.iota: update for batch dim.
    if (auto iotaOp = dyn_cast<stablehlo::IotaOp>(op))
      return batchIota(iotaOp, N);

    // func.call: callee will be batched separately, just update result types.
    if (auto callOp = dyn_cast<func::CallOp>(op))
      return batchCall(callOp, N);

    // stablehlo.while: batch carry types and region block args. Inner ops
    // are batched by the enclosing walk (they appear later in the op list).
    if (auto whileOp = dyn_cast<stablehlo::WhileOp>(op))
      return batchWhile(whileOp, N);

    // stablehlo.return: handle cond region predicate reduction.
    if (auto retOp = dyn_cast<stablehlo::ReturnOp>(op))
      return batchStablehloReturn(retOp, N);

    // arith.constant (scalar index for dynamic_slice): leave unchanged.
    if (isa<arith::ConstantOp>(op))
      return success();

    // unrealized_conversion_cast: from i1→tensor<i1> type conversion.
    // Update result type with batch dim.
    if (isa<UnrealizedConversionCastOp>(op))
      return batchElementWise(op, N);

    // Residual LLZK dialect ops (pod.new, pod.read, pod.write, array.new,
    // array.read, array.write on pod-element arrays): these survive as dead
    // code from dispatch patterns. Skip them — they don't produce tensor
    // results that need batching.
    if (op->getName().getDialectNamespace() == "pod" ||
        op->getName().getDialectNamespace() == "array")
      return success();

    return op->emitError("batch-stablehlo: unsupported op '")
           << op->getName() << "'";
  }

  // ----- Op-specific batching -----

  /// Check if a use is a start_index operand of dynamic_slice or
  /// dynamic_update_slice (should NOT be batched).
  static bool isSliceIndexUse(OpOperand &use) {
    Operation *user = use.getOwner();
    if (isa<stablehlo::DynamicSliceOp>(user)) {
      // Operand 0 is the input tensor, operands 1+ are start_indices.
      return use.getOperandNumber() >= 1;
    }
    if (isa<stablehlo::DynamicUpdateSliceOp>(user)) {
      // Operand 0 is input, operand 1 is update, operands 2+ are
      // start_indices.
      return use.getOperandNumber() >= 2;
    }
    return false;
  }

  LogicalResult batchConstant(stablehlo::ConstantOp constOp, int64_t N) {
    // Check if any use needs a batched value (i.e., is not a slice index).
    bool hasDataUses = false;
    for (auto &use : constOp.getResult().getUses()) {
      if (!isSliceIndexUse(use)) {
        hasDataUses = true;
        break;
      }
    }
    if (!hasDataUses)
      return success();

    OpBuilder builder(constOp->getContext());
    builder.setInsertionPointAfter(constOp);

    auto origType = cast<RankedTensorType>(constOp.getType());
    auto batchedType = addBatchDim(origType, N);
    int64_t origRank = origType.getRank();

    // broadcast_in_dim maps original dims [0, ..., origRank-1] to
    // [1, ..., origRank] in the batched tensor (dim 0 is the batch dim).
    SmallVector<int64_t> broadcastDims;
    for (int64_t i = 0; i < origRank; ++i)
      broadcastDims.push_back(i + 1);

    auto bcastOp = builder.create<stablehlo::BroadcastInDimOp>(
        constOp.getLoc(), batchedType, constOp.getResult(),
        builder.getDenseI64ArrayAttr(broadcastDims));

    // Replace only non-index uses with the broadcast result.
    constOp.getResult().replaceUsesWithIf(
        bcastOp.getResult(), [&](OpOperand &use) {
          return &use != &bcastOp->getOpOperand(0) && !isSliceIndexUse(use);
        });
    return success();
  }

  LogicalResult batchElementWise(Operation *op, int64_t N) {
    for (auto result : op->getResults()) {
      if (auto rtt = dyn_cast<RankedTensorType>(result.getType()))
        result.setType(addBatchDim(rtt, N));
    }
    return success();
  }

  LogicalResult batchCompare(stablehlo::CompareOp cmpOp, int64_t N) {
    auto resultType = cast<RankedTensorType>(cmpOp.getType());
    cmpOp.getResult().setType(addBatchDim(resultType, N));
    return success();
  }

  LogicalResult batchSelect(stablehlo::SelectOp selectOp, int64_t N) {
    auto resultType = cast<RankedTensorType>(selectOp.getType());
    selectOp.getResult().setType(addBatchDim(resultType, N));
    return success();
  }

  LogicalResult batchBroadcastInDim(stablehlo::BroadcastInDimOp bcastOp,
                                    int64_t N) {
    auto resultType = cast<RankedTensorType>(bcastOp.getType());
    auto batchedResult = addBatchDim(resultType, N);

    // Shift all broadcast_dimensions by +1 to account for the new batch dim,
    // then prepend dim 0 for the batch dimension.
    auto origDims = bcastOp.getBroadcastDimensions();
    SmallVector<int64_t> newDims;
    newDims.push_back(0); // batch dim maps to batch dim
    for (int64_t d : origDims)
      newDims.push_back(d + 1);

    OpBuilder builder(bcastOp);
    bcastOp.setBroadcastDimensionsAttr(builder.getDenseI64ArrayAttr(newDims));
    bcastOp.getResult().setType(batchedResult);
    return success();
  }

  /// Create a 0-d tensor<i32> zero constant, matching the index type used by
  /// the llzk-to-stablehlo conversion for dynamic_slice start indices.
  Value createZeroIndex(OpBuilder &builder, Location loc) {
    auto type = RankedTensorType::get({}, builder.getI32Type());
    auto attr = DenseElementsAttr::get(type, builder.getI32IntegerAttr(0));
    return builder.create<stablehlo::ConstantOp>(loc, attr);
  }

  /// Check if any start index has been batched (rank > 0 after batching).
  bool hasAnyBatchedIndex(OperandRange indices) {
    for (Value idx : indices) {
      if (auto rtt = dyn_cast<RankedTensorType>(idx.getType()))
        if (rtt.getRank() > 0)
          return true;
    }
    return false;
  }

  LogicalResult batchDynamicSlice(stablehlo::DynamicSliceOp sliceOp,
                                  int64_t N) {
    OpBuilder builder(sliceOp);
    auto loc = sliceOp.getLoc();
    auto resultType = cast<RankedTensorType>(sliceOp.getType());

    // Check if any index is data-dependent (batched to tensor<N>).
    // If so, we must use stablehlo.gather instead of dynamic_slice.
    if (hasAnyBatchedIndex(sliceOp.getStartIndices()))
      return batchDynamicSliceAsGather(sliceOp, N);

    // Prepend a 0 index for the batch dimension.
    auto zero = createZeroIndex(builder, loc);
    SmallVector<Value> newIndices;
    newIndices.push_back(zero);
    newIndices.append(sliceOp.getStartIndices().begin(),
                      sliceOp.getStartIndices().end());

    // Prepend N to slice_sizes.
    auto origSizes = sliceOp.getSliceSizes();
    SmallVector<int64_t> newSizes;
    newSizes.push_back(N);
    newSizes.append(origSizes.begin(), origSizes.end());

    auto batchedResult = addBatchDim(resultType, N);
    auto newOp = builder.create<stablehlo::DynamicSliceOp>(
        loc, batchedResult, sliceOp.getOperand(), newIndices,
        builder.getDenseI64ArrayAttr(newSizes));

    sliceOp.replaceAllUsesWith(newOp.getOperation());
    sliceOp.erase();
    return success();
  }

  /// Convert a batched dynamic_slice with data-dependent indices to
  /// one-hot selection. This handles the case where slice indices are
  /// per-batch-element (e.g., lookup table indexing in AES circuits).
  ///
  /// Original: dynamic_slice %table, %idx, sizes=[1]
  ///   table: tensor<NxMxE>, idx: tensor<N> (batched)
  ///   result: tensor<Nx1xE>
  ///
  /// Lowering (one-hot):
  ///   iota = [0, 1, ..., M-1]                           // tensor<M>
  ///   idx_bcast = broadcast(idx, [N,M])                  // tensor<NxM>
  ///   iota_bcast = broadcast(iota, [N,M])                // tensor<NxM>
  ///   mask = compare(iota_bcast, idx_bcast, EQ)          // tensor<NxM> i1
  ///   mask_E = convert(mask, E)                          // tensor<NxMxE>
  ///   selected = table * mask_E                          // zero-out non-match
  ///   result = reduce_sum(selected, dim=1)               // tensor<NxE>
  ///   result_r = reshape → tensor<Nx1xE>
  ///
  /// Supports arbitrary rank — finds the batched index dimension and applies
  /// one-hot selection along that dimension.
  LogicalResult batchDynamicSliceAsGather(stablehlo::DynamicSliceOp sliceOp,
                                          int64_t N) {
    auto origSizes = sliceOp.getSliceSizes();
    // All slice sizes on batched dimensions must be 1.
    // (Non-batched dimensions can have any size.)

    OpBuilder builder(sliceOp);
    auto loc = sliceOp.getLoc();
    auto resultType = cast<RankedTensorType>(sliceOp.getType());
    auto batchedResult = addBatchDim(resultType, N);

    Value operand = sliceOp.getOperand();
    auto operandType = cast<RankedTensorType>(operand.getType());
    Type elemType = operandType.getElementType();

    SmallVector<int64_t> maskShape(operandType.getShape());
    auto maskType = RankedTensorType::get(maskShape, builder.getI1Type());

    // Build combined one-hot mask for every indexed dim (both batched
    // tensor<N> indices and rank-0 constant indices). Pre-fix, the rewriter
    // skipped rank-0 indices, so a mixed `dynamic_slice ... sizes=[1, 1]`
    // (one batched + one constant index over a 2D operand) reduced only the
    // batched dim and leaked the constant-index dim into the result — the
    // reshape verifier then rejected `tensor<N x 4>` → `tensor<N x 1 x 1>`.
    // AES-256-key-expansion's round-key derivation while body exercises this
    // path against its 60×4 scratch.
    Value combinedMask;
    SmallVector<int64_t> reducedDims; // operand dims to collapse via reduce

    for (auto [i, idx] : llvm::enumerate(sliceOp.getStartIndices())) {
      auto idxType = dyn_cast<RankedTensorType>(idx.getType());
      if (!idxType)
        return sliceOp.emitError(
            "batch-stablehlo: dynamic_slice index is not a tensor");

      int64_t dim = i + 1; // +1 for batch dim
      bool isBatchedIdx = idxType.getRank() > 0;

      // Full-axis copy (`size == dimSize`, only meaningful for non-batched
      // indices since DS clamps `idx + size <= dimSize` ⇒ idx == 0): leave
      // this dim intact in the result, do not contribute to the one-hot
      // AND-mask. Mirrors AES key-expansion's `sizes=[1, 32]` row gather.
      int64_t dimSize = operandType.getDimSize(dim);
      if (!isBatchedIdx && origSizes[i] == dimSize)
        continue;

      // One-hot selection requires slice size 1 on every dim being masked.
      if (origSizes[i] != 1)
        return sliceOp.emitError(
                   "batch-stablehlo: one-hot gather requires slice size 1, "
                   "got ")
               << origSizes[i];
      reducedDims.push_back(dim);
      auto idxElemType = idxType.getElementType();

      auto iotaType = RankedTensorType::get({dimSize}, idxElemType);
      auto iota = builder.create<stablehlo::IotaOp>(
          loc, iotaType, builder.getI64IntegerAttr(0));

      // Batched (rank-1 tensor<N>) index: broadcast along batch dim 0.
      // Rank-0 (constant) index: empty broadcast_dimensions splats to all,
      // but Pass 2's batchConstant later walks the original constant's
      // non-slice-index uses and would rewrite our broadcast_in_dim operand
      // to a rank-N value while leaving broadcast_dimensions empty. Clone
      // the constant locally so Pass 2 doesn't see the use.
      Value idxOperand;
      SmallVector<int64_t> idxBcastDims;
      if (isBatchedIdx) {
        idxOperand = idx;
        idxBcastDims.push_back(0);
      } else {
        auto constOp = idx.getDefiningOp<stablehlo::ConstantOp>();
        if (!constOp)
          return sliceOp.emitError(
              "batch-stablehlo: rank-0 dynamic_slice index must be a "
              "stablehlo.constant");
        idxOperand = builder.clone(*constOp.getOperation())->getResult(0);
      }

      auto idxBcast = builder.create<stablehlo::BroadcastInDimOp>(
          loc, RankedTensorType::get(maskShape, idxElemType), idxOperand,
          builder.getDenseI64ArrayAttr(idxBcastDims));
      auto iotaBcast = builder.create<stablehlo::BroadcastInDimOp>(
          loc, RankedTensorType::get(maskShape, idxElemType), iota,
          builder.getDenseI64ArrayAttr({dim}));

      auto dimMask = builder.create<stablehlo::CompareOp>(
          loc, maskType, iotaBcast, idxBcast,
          stablehlo::ComparisonDirectionAttr::get(
              builder.getContext(), stablehlo::ComparisonDirection::EQ));

      if (!combinedMask)
        combinedMask = dimMask;
      else
        combinedMask = builder.create<stablehlo::AndOp>(loc, maskType,
                                                        combinedMask, dimMask);
    }

    if (!combinedMask)
      return sliceOp.emitError("batch-stablehlo: no indices to gather over");

    // Multiply operand with mask, reduce along every indexed dim.
    auto maskElem = builder.create<stablehlo::ConvertOp>(
        loc, RankedTensorType::get(maskShape, elemType), combinedMask);
    auto selected =
        builder.create<stablehlo::MulOp>(loc, operandType, operand, maskElem);

    // Reduce along every indexed dim at once.
    auto scalarTensorType = RankedTensorType::get({}, elemType);
    auto zeroAttr =
        DenseElementsAttr::get(RankedTensorType::get({}, builder.getI32Type()),
                               builder.getI32IntegerAttr(0));
    auto zero =
        builder.create<stablehlo::ConstantOp>(loc, scalarTensorType, zeroAttr);

    SmallVector<int64_t> reducedShape;
    for (int64_t i = 0; i < operandType.getRank(); ++i) {
      if (llvm::find(reducedDims, i) == reducedDims.end())
        reducedShape.push_back(operandType.getDimSize(i));
    }
    auto reducedType = RankedTensorType::get(reducedShape, elemType);

    auto reduce = builder.create<stablehlo::ReduceOp>(
        loc, TypeRange{reducedType}, ValueRange{selected}, ValueRange{zero},
        builder.getDenseI64ArrayAttr(reducedDims));
    {
      Block &body = reduce.getBody().emplaceBlock();
      body.addArgument(scalarTensorType, loc);
      body.addArgument(scalarTensorType, loc);
      OpBuilder bodyBuilder = OpBuilder::atBlockEnd(&body);
      auto sum = bodyBuilder.create<stablehlo::AddOp>(
          loc, scalarTensorType, body.getArgument(0), body.getArgument(1));
      bodyBuilder.create<stablehlo::ReturnOp>(loc, ValueRange{sum});
    }

    // Reshape to batchedResult.
    auto result = builder.create<stablehlo::ReshapeOp>(loc, batchedResult,
                                                       reduce.getResult(0));

    sliceOp.replaceAllUsesWith(result.getOperation());
    sliceOp.erase();
    return success();
  }

  LogicalResult batchDynamicUpdateSlice(stablehlo::DynamicUpdateSliceOp dusOp,
                                        int64_t N) {
    OpBuilder builder(dusOp);
    auto loc = dusOp.getLoc();
    auto resultType = cast<RankedTensorType>(dusOp.getType());

    // Check if any index is data-dependent (batched).
    if (hasAnyBatchedIndex(dusOp.getStartIndices()))
      return batchDynamicUpdateSliceAsScatter(dusOp, N);

    // Prepend a 0 index for the batch dimension.
    auto zero = createZeroIndex(builder, loc);
    SmallVector<Value> newIndices;
    newIndices.push_back(zero);
    newIndices.append(dusOp.getStartIndices().begin(),
                      dusOp.getStartIndices().end());

    auto batchedResult = addBatchDim(resultType, N);
    auto newOp = builder.create<stablehlo::DynamicUpdateSliceOp>(
        loc, batchedResult, dusOp.getOperand(), dusOp.getUpdate(), newIndices);

    dusOp.replaceAllUsesWith(newOp.getOperation());
    dusOp.erase();
    return success();
  }

  /// Convert a batched dynamic_update_slice with data-dependent indices to
  /// one-hot scatter. Write counterpart of batchDynamicSliceAsGather; mirrors
  /// its multi-index AND-mask pattern.
  ///
  /// Original: dynamic_update_slice %operand, %update, %idx0, %idx1, ...
  ///   operand : tensor<N x D0 x D1 x ... x E>  (already batched)
  ///   update  : tensor<N x 1 x 1 x ... x E>    (typical: SSA-batched, slice 1
  ///                                              on every batched dim) OR
  ///             tensor<1 x 1 x ... x E>        (unbatched, e.g. constant)
  ///   idx_k   : tensor<N> for each batched dim, scalar otherwise
  ///   result: per-batch one-hot scatter — operand[b, idx0[b], idx1[b], ...]
  ///           = update[b, 0, 0, ...]
  ///
  /// Lowering:
  ///   for each batched idx_k (operand dim k+1):
  ///     iota_k       = [0, ..., D_k-1]                    // tensor<D_k>
  ///     idx_bcast_k  = broadcast(idx_k,  dim 0)           // tensor<N x D...>
  ///     iota_bcast_k = broadcast(iota_k, dim k+1)         // tensor<N x D...>
  ///     mask_k       = compare(iota_bcast_k, idx_bcast_k, EQ)  // i1
  ///   combined_mask  = AND(mask_0, mask_1, ...)           // i1, exactly one
  ///                                                          true position
  ///                                                          per batch lane
  ///   update_bcast   = broadcast(update, dims = [0..R)+offset)
  ///                                                       // operand shape
  ///   result = select(combined_mask, update_bcast, operand)
  ///
  /// Pre-fix (single-index scatter) was wrong for ops with multiple batched
  /// start indices: it built a one-hot mask along ONE dim only, leaving the
  /// other batched dim free, so update splatted across that whole dim instead
  /// of touching one position. AES @main's inner while body
  /// `dynamic_update_slice %iter, %v, %i, %j` exercised this path.
  LogicalResult
  batchDynamicUpdateSliceAsScatter(stablehlo::DynamicUpdateSliceOp dusOp,
                                   int64_t N) {
    OpBuilder builder(dusOp);
    auto loc = dusOp.getLoc();
    auto operandType = cast<RankedTensorType>(dusOp.getOperand().getType());

    Value operand = dusOp.getOperand();
    Value update = dusOp.getUpdate();
    Type elemType = operandType.getElementType();

    SmallVector<int64_t> maskShape(operandType.getShape());
    auto maskType = RankedTensorType::get(maskShape, builder.getI1Type());

    // Implicit slice size on operand dim k is update.shape[k + rankOffset]
    // (update tail-aligns with operand). One-hot scatter requires that
    // implicit size to be 1 on every batched dim, mirroring the explicit
    // origSizes[i]==1 guard sister batchDynamicSliceAsGather enforces.
    auto updateType = cast<RankedTensorType>(update.getType());
    int64_t rankOffset = operandType.getRank() - updateType.getRank();

    // Build a one-hot mask per batched index, AND them together. Multi-
    // batched-index parity with batchDynamicSliceAsGather; rank-0 (constant)
    // indices are skipped here and handled by the implicit splat in the
    // SelectOp below — gather diverges and folds rank-0 into the AND-mask
    // (it has no implicit-splat fallback). Generalize this loop the same
    // way if a real circuit needs scatter at a constant index.
    Value combinedMask;
    for (auto [i, idx] : llvm::enumerate(dusOp.getStartIndices())) {
      auto idxType = dyn_cast<RankedTensorType>(idx.getType());
      if (!idxType || idxType.getRank() == 0)
        continue; // Not batched — leave for the implicit splat below.

      int64_t dim = i + 1; // +1 skips operand's leading batch dim.
      int64_t dimSize = operandType.getDimSize(dim);
      int64_t updateDim = dim - rankOffset;
      if (updateDim < 0 || updateDim >= updateType.getRank() ||
          updateType.getDimSize(updateDim) != 1)
        return dusOp.emitError(
                   "batch-stablehlo: one-hot scatter requires update size 1 "
                   "on batched operand dim ")
               << dim;
      auto idxElemType = idxType.getElementType();

      auto iotaType = RankedTensorType::get({dimSize}, idxElemType);
      auto iota = builder.create<stablehlo::IotaOp>(
          loc, iotaType, builder.getI64IntegerAttr(0));

      auto idxBcast = builder.create<stablehlo::BroadcastInDimOp>(
          loc, RankedTensorType::get(maskShape, idxElemType), idx,
          builder.getDenseI64ArrayAttr({0}));
      auto iotaBcast = builder.create<stablehlo::BroadcastInDimOp>(
          loc, RankedTensorType::get(maskShape, idxElemType), iota,
          builder.getDenseI64ArrayAttr({dim}));

      auto dimMask = builder.create<stablehlo::CompareOp>(
          loc, maskType, iotaBcast, idxBcast,
          stablehlo::ComparisonDirectionAttr::get(
              builder.getContext(), stablehlo::ComparisonDirection::EQ));

      if (!combinedMask)
        combinedMask = dimMask;
      else
        combinedMask = builder.create<stablehlo::AndOp>(loc, maskType,
                                                        combinedMask, dimMask);
    }

    if (!combinedMask)
      return dusOp.emitError("batch-stablehlo: no batched index found");

    // Broadcast update to operand shape. update may be either fully batched
    // (rank == operandRank, e.g. an SSA producer that already grew the leading
    // N dim) or unbatched (rank == operandRank - 1, e.g. a constant). The
    // tail-aligning rankOffset computed above maps each update dim to the
    // corresponding operand dim.
    SmallVector<int64_t> updateBcastDims;
    for (int64_t i = 0; i < updateType.getRank(); ++i)
      updateBcastDims.push_back(i + rankOffset);
    auto updateBcast = builder.create<stablehlo::BroadcastInDimOp>(
        loc, operandType, update,
        builder.getDenseI64ArrayAttr(updateBcastDims));

    auto result = builder.create<stablehlo::SelectOp>(
        loc, operandType, combinedMask, updateBcast, operand);

    dusOp.replaceAllUsesWith(result.getOperation());
    dusOp.erase();
    return success();
  }

  LogicalResult batchReshape(stablehlo::ReshapeOp reshapeOp, int64_t N) {
    auto resultType = cast<RankedTensorType>(reshapeOp.getType());
    reshapeOp.getResult().setType(addBatchDim(resultType, N));
    return success();
  }

  LogicalResult batchStaticSlice(stablehlo::SliceOp sliceOp, int64_t N) {
    auto resultType = cast<RankedTensorType>(sliceOp.getType());
    auto batchedResult = addBatchDim(resultType, N);

    OpBuilder builder(sliceOp);

    // Prepend batch dim bounds: start=0, limit=N, stride=1.
    auto origStart = sliceOp.getStartIndices();
    auto origLimit = sliceOp.getLimitIndices();
    auto origStrides = sliceOp.getStrides();

    SmallVector<int64_t> newStart, newLimit, newStrides;
    newStart.push_back(0);
    newStart.append(origStart.begin(), origStart.end());
    newLimit.push_back(N);
    newLimit.append(origLimit.begin(), origLimit.end());
    newStrides.push_back(1);
    newStrides.append(origStrides.begin(), origStrides.end());

    sliceOp.setStartIndicesAttr(builder.getDenseI64ArrayAttr(newStart));
    sliceOp.setLimitIndicesAttr(builder.getDenseI64ArrayAttr(newLimit));
    sliceOp.setStridesAttr(builder.getDenseI64ArrayAttr(newStrides));
    sliceOp.getResult().setType(batchedResult);
    return success();
  }

  LogicalResult batchConcatenate(stablehlo::ConcatenateOp concatOp, int64_t N) {
    auto resultType = cast<RankedTensorType>(concatOp.getType());
    concatOp.getResult().setType(addBatchDim(resultType, N));

    // Shift the concatenation dimension by +1.
    int64_t origDim = concatOp.getDimension();
    concatOp.setDimensionAttr(
        OpBuilder(concatOp).getI64IntegerAttr(origDim + 1));
    return success();
  }

  LogicalResult batchIota(stablehlo::IotaOp iotaOp, int64_t N) {
    auto resultType = cast<RankedTensorType>(iotaOp.getType());
    iotaOp.getResult().setType(addBatchDim(resultType, N));

    // Shift the iota dimension by +1.
    int64_t origDim = iotaOp.getIotaDimension();
    iotaOp.setIotaDimensionAttr(
        OpBuilder(iotaOp).getI64IntegerAttr(origDim + 1));
    return success();
  }

  LogicalResult batchCall(func::CallOp callOp, int64_t N) {
    // Update result types to match the batched callee signature.
    for (auto result : callOp.getResults()) {
      if (auto rtt = dyn_cast<RankedTensorType>(result.getType()))
        result.setType(addBatchDim(rtt, N));
    }
    return success();
  }

  LogicalResult batchWhile(stablehlo::WhileOp whileOp, int64_t N) {
    // 1. Update result types (carry types with batch dim).
    for (auto result : whileOp.getResults()) {
      if (auto rtt = dyn_cast<RankedTensorType>(result.getType()))
        result.setType(addBatchDim(rtt, N));
    }

    // 2. Update block args of cond and body regions.
    for (Region *region : {&whileOp.getCond(), &whileOp.getBody()}) {
      for (auto arg : region->getArguments()) {
        if (auto rtt = dyn_cast<RankedTensorType>(arg.getType()))
          arg.setType(addBatchDim(rtt, N));
      }
    }

    // Inner ops will be batched by the enclosing walk (they are collected
    // after this while op in the ops list).
    return success();
  }

  LogicalResult batchStablehloReturn(stablehlo::ReturnOp retOp, int64_t N) {
    // Check if this return is inside a while cond region.
    auto *parentOp = retOp->getParentOp();
    if (auto whileOp = dyn_cast<stablehlo::WhileOp>(parentOp)) {
      // Is this the cond region (first region)?
      if (retOp->getParentRegion() == &whileOp.getCond()) {
        // The cond must return tensor<i1>. After batching the compare op,
        // the predicate is tensor<Nxi1>. Since circom while loops have
        // fixed trip counts, all batch elements have the same predicate.
        // Extract element [0] to get the scalar tensor<i1>.
        Value pred = retOp.getOperand(0);
        auto predType = dyn_cast<RankedTensorType>(pred.getType());
        if (predType && predType.getRank() == 1 &&
            predType.getDimSize(0) == N) {
          OpBuilder builder(retOp);
          auto scalarType =
              RankedTensorType::get({}, predType.getElementType());
          // slice [0:1] then reshape to scalar
          auto sliced = builder.create<stablehlo::SliceOp>(
              retOp.getLoc(),
              RankedTensorType::get({1}, predType.getElementType()), pred,
              builder.getDenseI64ArrayAttr({0}),
              builder.getDenseI64ArrayAttr({1}),
              builder.getDenseI64ArrayAttr({1}));
          auto scalar = builder.create<stablehlo::ReshapeOp>(
              retOp.getLoc(), scalarType, sliced);
          retOp.setOperand(0, scalar);
        }
      }
    }
    // Body region return and other cases: operand types already updated.
    return success();
  }
};

} // namespace
} // namespace mlir::llzk_to_shlo
