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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Pattern to convert array.new to tensor construction.
class ArrayNewPattern : public ConversionPattern {
public:
  ArrayNewPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "array.new", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType)
      return failure();

    Location loc = op->getLoc();

    // If elements are provided, concatenate them
    if (!operands.empty()) {
      SmallVector<Value> reshapedElements;
      for (Value elem : operands) {
        auto elemType = dyn_cast<RankedTensorType>(elem.getType());
        if (!elemType)
          continue;

        SmallVector<int64_t> newShape = {1};
        for (int64_t dim : elemType.getShape()) {
          newShape.push_back(dim);
        }
        auto reshapedType =
            RankedTensorType::get(newShape, elemType.getElementType());
        auto reshaped =
            rewriter.create<stablehlo::ReshapeOp>(loc, reshapedType, elem);
        reshapedElements.push_back(reshaped);
      }

      if (!reshapedElements.empty()) {
        auto concat = rewriter.create<stablehlo::ConcatenateOp>(
            loc, reshapedElements, /*dimension=*/0);
        rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, tensorType,
                                                          concat);
        return success();
      }
    }

    // No elements provided: create zero-initialized array
    auto zeroAttr = typeConverter->createConstantAttr(tensorType, 0, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       zeroAttr);
    return success();
  }
};

/// Pattern to convert array.read to stablehlo.dynamic_slice.
class ArrayReadPattern : public ConversionPattern {
public:
  ArrayReadPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "array.read", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // First operand is the array, remaining are indices
    if (operands.empty() || op->getNumResults() != 1)
      return failure();

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();
    Value array = operands[0];
    auto indices = operands.drop_front();

    auto arrayType = dyn_cast<RankedTensorType>(array.getType());
    if (!arrayType)
      return failure();

    int64_t rank = arrayType.getRank();

    // Convert index values to scalar i64 tensors for dynamic_slice
    SmallVector<Value> startIndices;
    for (Value idx : indices) {
      Value idxI64;
      if (idx.getType().isIndex()) {
        idxI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                     idx);
      } else {
        idxI64 = idx;
      }
      auto idxTensor = rewriter.create<stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({}, rewriter.getI64Type()), idxI64);
      startIndices.push_back(idxTensor);
    }

    // For remaining dimensions (if indices.size() < rank), use 0
    for (size_t i = indices.size(); i < static_cast<size_t>(rank); ++i) {
      auto zero = rewriter.create<stablehlo::ConstantOp>(
          loc, DenseElementsAttr::get(
                   RankedTensorType::get({}, rewriter.getI64Type()),
                   rewriter.getI64IntegerAttr(0)));
      startIndices.push_back(zero);
    }

    // Slice sizes: 1 for indexed dimensions, full size for others
    SmallVector<int64_t> sliceSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (static_cast<size_t>(i) < indices.size()) {
        sliceSizes.push_back(1);
      } else {
        sliceSizes.push_back(arrayType.getDimSize(i));
      }
    }

    auto sliced = rewriter.create<stablehlo::DynamicSliceOp>(
        loc, array, startIndices, rewriter.getDenseI64ArrayAttr(sliceSizes));

    // Reshape to remove the indexed dimensions
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType, sliced);
    return success();
  }
};

/// Pattern to convert array.write to stablehlo.dynamic_update_slice.
class ArrayWritePattern : public ConversionPattern {
public:
  ArrayWritePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "array.write", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect: array, value, indices...
    if (operands.size() < 2)
      return failure();

    Location loc = op->getLoc();
    Value array = operands[0];
    Value value = operands[1];
    auto indices = operands.drop_front(2);

    auto arrayType = dyn_cast<RankedTensorType>(array.getType());
    if (!arrayType)
      return failure();

    int64_t rank = arrayType.getRank();

    // Convert index values to scalar i64 tensors
    SmallVector<Value> startIndices;
    for (Value idx : indices) {
      Value idxI64;
      if (idx.getType().isIndex()) {
        idxI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                     idx);
      } else {
        idxI64 = idx;
      }
      auto idxTensor = rewriter.create<stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({}, rewriter.getI64Type()), idxI64);
      startIndices.push_back(idxTensor);
    }

    // For remaining dimensions, use 0
    for (size_t i = indices.size(); i < static_cast<size_t>(rank); ++i) {
      auto zero = rewriter.create<stablehlo::ConstantOp>(
          loc, DenseElementsAttr::get(
                   RankedTensorType::get({}, rewriter.getI64Type()),
                   rewriter.getI64IntegerAttr(0)));
      startIndices.push_back(zero);
    }

    // Reshape value to match slice shape (add size-1 dimensions)
    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    if (!valueType)
      return failure();

    SmallVector<int64_t> updateShape;
    for (size_t i = 0; i < indices.size(); ++i) {
      updateShape.push_back(1);
    }
    for (int64_t dim : valueType.getShape()) {
      updateShape.push_back(dim);
    }
    auto updateType =
        RankedTensorType::get(updateShape, valueType.getElementType());
    auto reshapedValue =
        rewriter.create<stablehlo::ReshapeOp>(loc, updateType, value);

    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        op, array, reshapedValue, startIndices);
    return success();
  }
};

/// Pattern to convert array.len to a constant or tensor.dim.
class ArrayLenPattern : public ConversionPattern {
public:
  ArrayLenPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "array.len", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() < 2)
      return failure();

    Value array = operands[0];
    Value dimIdx = operands[1];

    auto arrayType = dyn_cast<RankedTensorType>(array.getType());
    if (!arrayType)
      return failure();

    // If the dimension index is a constant, compute the length statically
    if (auto constOp = dimIdx.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t dim = constOp.value();
      if (dim >= 0 && dim < arrayType.getRank()) {
        int64_t length = arrayType.getDimSize(dim);
        if (length != ShapedType::kDynamic) {
          rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, length);
          return success();
        }
      }
    }

    // For dynamic dimension index, use tensor.dim
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, array, dimIdx);
    return success();
  }
};

} // namespace

void populateArrayToStablehloPatterns(LlzkToStablehloTypeConverter &converter,
                                      RewritePatternSet &patterns,
                                      ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();

  // Mark array dialect operations as illegal
  target.addIllegalDialect("array");

  patterns.add<ArrayNewPattern>(converter, ctx);
  patterns.add<ArrayReadPattern>(converter, ctx);
  patterns.add<ArrayWritePattern>(converter, ctx);
  patterns.add<ArrayLenPattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
