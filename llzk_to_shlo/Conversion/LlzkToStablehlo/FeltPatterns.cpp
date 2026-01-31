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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FeltPatterns.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Generic pattern for binary felt operations
class FeltBinaryOpPattern : public ConversionPattern {
public:
  FeltBinaryOpPattern(StringRef opName, StringRef stablehloOpName,
                      TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, opName, /*benefit=*/1, ctx),
        stablehloOpName(stablehloOpName) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2)
      return failure();

    Type resultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    if (stablehloOpName == "stablehlo.add") {
      rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, resultType, operands[0],
                                                    operands[1]);
    } else if (stablehloOpName == "stablehlo.subtract") {
      rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(
          op, resultType, operands[0], operands[1]);
    } else if (stablehloOpName == "stablehlo.multiply") {
      rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, resultType, operands[0],
                                                    operands[1]);
    } else if (stablehloOpName == "stablehlo.divide") {
      rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, resultType, operands[0],
                                                    operands[1]);
    } else {
      return failure();
    }
    return success();
  }

private:
  StringRef stablehloOpName;
};

/// Pattern for felt.neg -> stablehlo.negate
class FeltNegPattern : public ConversionPattern {
public:
  FeltNegPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "felt.neg", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return failure();

    Type resultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, resultType, operands[0]);
    return success();
  }
};

/// Pattern for felt.inv -> stablehlo.divide(1, x)
class FeltInvPattern : public ConversionPattern {
public:
  FeltInvPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "felt.inv", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return failure();

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();

    // Create constant 1
    auto tensorType = cast<RankedTensorType>(resultType);
    auto oneAttr = typeConverter->createConstantAttr(tensorType, 1, rewriter);
    auto one = rewriter.create<stablehlo::ConstantOp>(loc, tensorType, oneAttr);

    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, resultType, one,
                                                  operands[0]);
    return success();
  }
};

/// Pattern for felt.const -> stablehlo.constant
class FeltConstPattern : public ConversionPattern {
public:
  FeltConstPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "felt.const", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    // Get the constant value from the "value" attribute
    // LLZK uses FeltConstAttr which has a getValue() method returning APInt
    auto valueAttr = op->getAttr("value");
    if (!valueAttr)
      return failure();

    auto tensorType = cast<RankedTensorType>(resultType);

    // Handle LLZK FeltConstAttr
    int64_t value = 0;
    if (auto feltConstAttr = dyn_cast<llzk::felt::FeltConstAttr>(valueAttr)) {
      value = feltConstAttr.getValue().getSExtValue();
    } else if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      value = intAttr.getValue().getSExtValue();
    } else {
      return op->emitError("unsupported constant value attribute type");
    }

    auto denseAttr =
        typeConverter->createConstantAttr(tensorType, value, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       denseAttr);
    return success();
  }
};

/// Pattern for felt.nondet -> stablehlo.constant (placeholder zero)
class FeltNonDetPattern : public ConversionPattern {
public:
  FeltNonDetPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "felt.nondet", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    auto tensorType = cast<RankedTensorType>(resultType);
    auto zeroAttr = typeConverter->createConstantAttr(tensorType, 0, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       zeroAttr);
    return success();
  }
};

} // namespace

void populateFeltToStablehloPatterns(LlzkToStablehloTypeConverter &converter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();

  // Mark felt dialect operations as illegal
  target.addIllegalDialect("felt");

  // Add patterns
  patterns.add<FeltConstPattern>(converter, ctx);
  patterns.add<FeltNegPattern>(converter, ctx);
  patterns.add<FeltInvPattern>(converter, ctx);
  patterns.add<FeltNonDetPattern>(converter, ctx);

  // Binary operations
  patterns.add<FeltBinaryOpPattern>("felt.add", "stablehlo.add", converter,
                                    ctx);
  patterns.add<FeltBinaryOpPattern>("felt.sub", "stablehlo.subtract", converter,
                                    ctx);
  patterns.add<FeltBinaryOpPattern>("felt.mul", "stablehlo.multiply", converter,
                                    ctx);
  patterns.add<FeltBinaryOpPattern>("felt.div", "stablehlo.divide", converter,
                                    ctx);
}

} // namespace mlir::llzk_to_shlo
