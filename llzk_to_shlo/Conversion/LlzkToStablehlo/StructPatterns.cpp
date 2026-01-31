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

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType)
      return failure();

    // Create a zero-initialized tensor for the struct
    auto zeroAttr = typeConverter->createConstantAttr(tensorType, 0, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       zeroAttr);
    return success();
  }
};

/// Pattern to convert struct.readf to stablehlo.slice.
/// Uses the field offset stored in the type converter.
class StructReadFPattern : public ConversionPattern {
public:
  StructReadFPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.readf", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.empty() || op->getNumResults() != 1)
      return failure();

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());

    // Get the struct type from the operand
    Type structType = op->getOperand(0).getType();

    // Get field name from the operation attribute
    auto fieldNameAttr = op->getAttrOfType<FlatSymbolRefAttr>("field_name");
    if (!fieldNameAttr) {
      fieldNameAttr = op->getAttrOfType<FlatSymbolRefAttr>("fieldName");
    }
    if (!fieldNameAttr)
      return failure();

    StringRef fieldName = fieldNameAttr.getValue();

    // Get the field offset in the flattened struct
    auto offset = typeConverter->getFieldOffset(structType, fieldName);
    if (!offset) {
      return op->emitError("field offset not found for: ") << fieldName;
    }

    Type resultType = typeConverter->convertType(op->getResult(0).getType());
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

/// Pattern to convert struct.writef to stablehlo.dynamic_update_slice.
class StructWriteFPattern : public ConversionPattern {
public:
  StructWriteFPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.writef", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // struct.writef typically has 2 operands: struct and value
    if (operands.size() < 2)
      return failure();

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());

    Type structType = op->getOperand(0).getType();

    // Get field name from the operation attribute
    auto fieldNameAttr = op->getAttrOfType<FlatSymbolRefAttr>("field_name");
    if (!fieldNameAttr) {
      fieldNameAttr = op->getAttrOfType<FlatSymbolRefAttr>("fieldName");
    }
    if (!fieldNameAttr)
      return failure();

    StringRef fieldName = fieldNameAttr.getValue();

    auto offset = typeConverter->getFieldOffset(structType, fieldName);
    if (!offset) {
      return op->emitError("field offset not found for: ") << fieldName;
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

    // Create start indices constant (scalar tensor for 1D struct tensor)
    auto startIndex = rewriter.create<stablehlo::ConstantOp>(
        loc, RankedTensorType::get({}, rewriter.getI64Type()),
        DenseElementsAttr::get(RankedTensorType::get({}, rewriter.getI64Type()),
                               rewriter.getI64IntegerAttr(*offset)));

    // Use dynamic_update_slice to update the struct tensor
    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        op, structTensor, value, ValueRange{startIndex});
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
  patterns.add<StructReadFPattern>(converter, ctx);
  patterns.add<StructWriteFPattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
