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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructPatterns.h"

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Pattern to convert struct.new to tensor construction.
/// Creates a zero-initialized tensor for the flattened struct.
class StructNewPattern : public ConversionPattern {
public:
  StructNewPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.new", /*benefit=*/1, ctx) {}

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

    // Create a zero-initialized tensor for the struct
    auto zeroAttr = tc.createConstantAttr(tensorType, 0, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       zeroAttr);
    return success();
  }
};

/// Pattern to convert struct.readm to stablehlo.slice.
/// Uses the field offset stored in the type converter.
class StructReadMPattern : public ConversionPattern {
public:
  StructReadMPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.readm", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.empty() || op->getNumResults() != 1)
      return failure();

    const auto &tc = getConverter(getTypeConverter());

    // Get the struct type from the operand
    Type structType = op->getOperand(0).getType();

    // Get field name from the operation attribute
    auto memberNameAttr = op->getAttrOfType<FlatSymbolRefAttr>("member_name");
    if (!memberNameAttr)
      return failure();

    StringRef memberName = memberNameAttr.getValue();

    // Get the member offset in the flattened struct
    auto offset = tc.getFieldOffset(structType, memberName);
    if (!offset) {
      return op->emitError("member offset not found for: ") << memberName;
    }

    Type resultType = tc.convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();
    Value structTensor = operands[0];

    // Determine the size of the field being read
    int64_t fieldSize = 1;
    if (auto resultTensorType = dyn_cast<RankedTensorType>(resultType)) {
      if (resultTensorType.getRank() == 0) {
        fieldSize = 1;
      } else {
        fieldSize = 1;
        for (int64_t dim : resultTensorType.getShape()) {
          if (dim != ShapedType::kDynamic) {
            fieldSize *= dim;
          }
        }
      }
    }

    // Create slice to extract the field
    auto sliced = rewriter.create<stablehlo::SliceOp>(
        loc, structTensor, rewriter.getDenseI64ArrayAttr({*offset}),
        rewriter.getDenseI64ArrayAttr({*offset + fieldSize}),
        rewriter.getDenseI64ArrayAttr({1}));

    // Reshape to the result type
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType, sliced);
    return success();
  }
};

/// Pattern to convert struct.writem to stablehlo.dynamic_update_slice.
class StructWriteMPattern : public ConversionPattern {
public:
  StructWriteMPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.writem", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // struct.writem typically has 2 operands: struct and value
    if (operands.size() < 2)
      return failure();

    const auto &tc = getConverter(getTypeConverter());

    Type structType = op->getOperand(0).getType();

    // Get field name from the operation attribute
    auto memberNameAttr = op->getAttrOfType<FlatSymbolRefAttr>("member_name");
    if (!memberNameAttr)
      return failure();

    StringRef memberName = memberNameAttr.getValue();

    auto offset = tc.getFieldOffset(structType, memberName);
    if (!offset) {
      return op->emitError("member offset not found for: ") << memberName;
    }

    Location loc = op->getLoc();
    Value structTensor = operands[0];
    Value value = operands[1];

    // Flatten the value to 1D if needed
    auto valueType = value.getType();
    if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
      if (tensorType.getRank() == 0) {
        // Scalar tensor: reshape to 1D tensor with size 1
        value = rewriter.create<stablehlo::ReshapeOp>(
            loc, RankedTensorType::get({1}, tensorType.getElementType()),
            value);
      } else if (tensorType.getRank() > 1) {
        // Multi-dimensional: flatten to 1D
        int64_t flatSize = 1;
        for (int64_t dim : tensorType.getShape()) {
          if (dim != ShapedType::kDynamic) {
            flatSize *= dim;
          }
        }
        auto flatType =
            RankedTensorType::get({flatSize}, tensorType.getElementType());
        value = rewriter.create<stablehlo::ReshapeOp>(loc, flatType, value);
      }
    }

    auto startIndex = createIndexConstant(rewriter, loc, *offset);

    // Shared helper handles both the SSA-ified (1-result) and the
    // void (0-result, when promoteArraysToWhileCarry didn't pick the writem
    // up — convertWritemToSSA leaves writems inside scf control flow alone)
    // shapes; the orphan DUS produced for the void case is DCE'd downstream.
    replaceWithDUS(rewriter, op, loc, structTensor, value, startIndex);
    return success();
  }
};

} // namespace

void populateStructToStablehloPatterns(LlzkToStablehloTypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();

  // Mark struct dialect operations as illegal
  target.addIllegalDialect("struct");

  patterns.add<StructNewPattern>(converter, ctx);
  patterns.add<StructReadMPattern>(converter, ctx);
  patterns.add<StructWriteMPattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
