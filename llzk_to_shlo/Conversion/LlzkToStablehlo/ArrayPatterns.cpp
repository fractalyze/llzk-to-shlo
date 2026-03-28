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

    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
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
    auto zeroAttr = tc.createConstantAttr(tensorType, 0, rewriter);

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

    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();
    Value array = lookThroughCast(operands[0]);
    SmallVector<Value> indices;
    for (Value idx : operands.drop_front())
      indices.push_back(lookThroughCast(idx));

    array =
        ensureTensorType(rewriter, array, op->getOperand(0).getType(), tc, loc);
    auto arrayType = dyn_cast<RankedTensorType>(array.getType());
    if (!arrayType)
      return failure();

    int64_t rank = arrayType.getRank();

    SmallVector<Value> startIndices;
    for (Value idx : indices)
      startIndices.push_back(convertToIndexTensor(rewriter, idx, loc));

    // For remaining dimensions (if indices.size() < rank), use 0
    for (size_t i = indices.size(); i < static_cast<size_t>(rank); ++i) {
      startIndices.push_back(createIndexConstant(rewriter, loc, 0));
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
    // Operand order: (array, indices..., value)
    if (operands.size() < 3)
      return failure();

    Location loc = op->getLoc();
    const auto &tc = getConverter(getTypeConverter());

    Value array = lookThroughCast(operands[0]);
    Value value = lookThroughCast(operands.back());
    SmallVector<Value> indices;
    for (Value idx : operands.slice(1, operands.size() - 2))
      indices.push_back(lookThroughCast(idx));

    array =
        ensureTensorType(rewriter, array, op->getOperand(0).getType(), tc, loc);
    // value is the last operand; original type comes from op
    unsigned valIdx = op->getNumOperands() - 1;
    value = ensureTensorType(rewriter, value, op->getOperand(valIdx).getType(),
                             tc, loc);

    auto arrayType = dyn_cast<RankedTensorType>(array.getType());
    if (!arrayType)
      return failure();

    int64_t rank = arrayType.getRank();

    SmallVector<Value> startIndices;
    for (Value idx : indices)
      startIndices.push_back(convertToIndexTensor(rewriter, idx, loc));

    // For remaining dimensions, use 0
    for (size_t i = indices.size(); i < static_cast<size_t>(rank); ++i) {
      startIndices.push_back(createIndexConstant(rewriter, loc, 0));
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

    // Circom arrays are always statically sized. Compute length at compile time
    // and emit a stablehlo.constant (avoids arith/tensor dialect ops).
    // Try to resolve dimension index from the operand.
    int64_t dim = 0; // default to first dimension
    if (auto constOp = dimIdx.getDefiningOp<arith::ConstantIndexOp>())
      dim = constOp.value();

    if (dim >= 0 && dim < arrayType.getRank()) {
      int64_t length = arrayType.getDimSize(dim);
      if (length != ShapedType::kDynamic) {
        rewriter.replaceOp(op,
                           createIndexConstant(rewriter, op->getLoc(), length));
        return success();
      }
    }
    return failure();
  }
};

} // namespace

void populateArrayToStablehloPatterns(LlzkToStablehloTypeConverter &converter,
                                      RewritePatternSet &patterns,
                                      ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();

  // Array dialect legality is set by the caller (LlzkToStablehlo.cpp)
  // to handle dynamic legality for pod-element arrays.

  patterns.add<ArrayNewPattern>(converter, ctx);
  patterns.add<ArrayReadPattern>(converter, ctx);
  patterns.add<ArrayWritePattern>(converter, ctx);
  patterns.add<ArrayLenPattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
