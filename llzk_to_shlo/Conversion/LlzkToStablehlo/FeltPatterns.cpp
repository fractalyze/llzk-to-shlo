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

/// Generic template pattern for binary felt → stablehlo operations.
template <typename StablehloOp>
class FeltBinaryOpPattern : public ConversionPattern {
public:
  FeltBinaryOpPattern(StringRef opName, TypeConverter &converter,
                      MLIRContext *ctx)
      : ConversionPattern(converter, opName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2)
      return failure();
    Type resultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();
    rewriter.replaceOpWithNewOp<StablehloOp>(op, resultType, operands[0],
                                             operands[1]);
    return success();
  }
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

    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();

    // Create constant 1
    auto tensorType = cast<RankedTensorType>(resultType);
    auto oneAttr = tc.createConstantAttr(tensorType, 1, rewriter);
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
    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
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

    auto denseAttr = tc.createConstantAttr(tensorType, value, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       denseAttr);
    return success();
  }
};

/// Pattern for felt bitwise ops (shr, bit_and).
/// Convert field → integer, perform the integer op, convert back.
///   felt.shr     → convert → shift_right_logical → convert
///   felt.bit_and → convert → and → convert
template <typename StablehloOp>
class FeltBitwiseOpPattern : public ConversionPattern {
public:
  FeltBitwiseOpPattern(StringRef opName, TypeConverter &converter,
                       MLIRContext *ctx)
      : ConversionPattern(converter, opName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2)
      return failure();

    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();
    auto fieldTensorType = cast<RankedTensorType>(resultType);
    auto intTensorType =
        RankedTensorType::get(fieldTensorType.getShape(), tc.getStorageType());

    auto lhsInt =
        rewriter.create<stablehlo::ConvertOp>(loc, intTensorType, operands[0]);
    auto rhsInt =
        rewriter.create<stablehlo::ConvertOp>(loc, intTensorType, operands[1]);
    auto intResult =
        rewriter.create<StablehloOp>(loc, intTensorType, lhsInt, rhsInt);
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, fieldTensorType,
                                                      intResult);
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
    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    auto tensorType = cast<RankedTensorType>(resultType);
    auto zeroAttr = tc.createConstantAttr(tensorType, 0, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       zeroAttr);
    return success();
  }
};

/// Pattern for felt.pow → stablehlo.power.
/// Converts exponent from field to storage integer type (PowOp requires
/// integer exponent).
class FeltPowerPattern : public ConversionPattern {
public:
  FeltPowerPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "felt.pow", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2)
      return failure();

    const auto &tc = getConverter(getTypeConverter());
    Type resultType = tc.convertType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();
    auto fieldTensorType = cast<RankedTensorType>(resultType);
    auto intTensorType =
        RankedTensorType::get(fieldTensorType.getShape(), tc.getStorageType());
    auto expInt =
        rewriter.create<stablehlo::ConvertOp>(loc, intTensorType, operands[1]);
    rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, resultType, operands[0],
                                                  expInt);
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

  // Binary field arithmetic
  patterns.add<FeltBinaryOpPattern<stablehlo::AddOp>>("felt.add", converter,
                                                      ctx);
  patterns.add<FeltBinaryOpPattern<stablehlo::SubtractOp>>("felt.sub",
                                                           converter, ctx);
  patterns.add<FeltBinaryOpPattern<stablehlo::MulOp>>("felt.mul", converter,
                                                      ctx);
  patterns.add<FeltBinaryOpPattern<stablehlo::DivOp>>("felt.div", converter,
                                                      ctx);

  // Power: felt.pow → stablehlo.power(field_base, int_exponent).
  // Exponent is bitcast_converted from field to storage integer type.
  patterns.add<FeltPowerPattern>(converter, ctx);

  // Bitwise operations: field → integer convert → op → field convert
  patterns.add<FeltBitwiseOpPattern<stablehlo::ShiftRightLogicalOp>>(
      "felt.shr", converter, ctx);
  patterns.add<FeltBitwiseOpPattern<stablehlo::ShiftLeftOp>>("felt.shl",
                                                             converter, ctx);
  patterns.add<FeltBitwiseOpPattern<stablehlo::AndOp>>("felt.bit_and",
                                                       converter, ctx);

  // Unsigned integer arithmetic: field → integer convert → op → field convert
  patterns.add<FeltBitwiseOpPattern<stablehlo::RemOp>>("felt.umod", converter,
                                                       ctx);
  patterns.add<FeltBitwiseOpPattern<stablehlo::DivOp>>("felt.uintdiv",
                                                       converter, ctx);
}

} // namespace mlir::llzk_to_shlo
