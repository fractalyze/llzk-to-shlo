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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/RemovalPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Erase struct.def when all nested ops have been converted.
class StructDefErasePattern : public ConversionPattern {
public:
  StructDefErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.def", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (nested.getName().getStringRef() != "struct.member")
            return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Erase struct.member (absorbed into tensor layout).
class StructMemberErasePattern : public ConversionPattern {
public:
  StructMemberErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.member", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Erase function.def @constrain (not needed for witness generation).
class ConstrainFunctionErasePattern : public ConversionPattern {
public:
  ConstrainFunctionErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "function.def", /*benefit=*/2, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto symName = op->getAttrOfType<StringAttr>("sym_name");
    if (!symName || symName.getValue() != "constrain")
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Generic pattern that erases an op unconditionally.
class OpErasePattern : public ConversionPattern {
public:
  OpErasePattern(StringRef opName, TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, opName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert bool.or to stablehlo.or (boolean OR on tensor<i1>).
/// Falls back to arith.ori if operands are i1 scalars (inside scf bodies).
class BoolOrPattern : public ConversionPattern {
public:
  BoolOrPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "bool.or", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2)
      return failure();
    Value lhs = lookThroughCast(operands[0]);
    Value rhs = lookThroughCast(operands[1]);
    // If still i1 scalars (not tensors), fall back to arith
    if (!isa<RankedTensorType>(lhs.getType()) ||
        !isa<RankedTensorType>(rhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::OrIOp>(op, operands[0], operands[1]);
      return success();
    }
    rewriter.replaceOpWithNewOp<stablehlo::OrOp>(op, lhs, rhs);
    return success();
  }
};

/// Convert bool.and to stablehlo.and (boolean AND on tensor<i1>).
class BoolAndPattern : public ConversionPattern {
public:
  BoolAndPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "bool.and", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2)
      return failure();
    Value lhs = lookThroughCast(operands[0]);
    Value rhs = lookThroughCast(operands[1]);
    if (!isa<RankedTensorType>(lhs.getType()) ||
        !isa<RankedTensorType>(rhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::AndIOp>(op, operands[0], operands[1]);
      return success();
    }
    rewriter.replaceOpWithNewOp<stablehlo::AndOp>(op, lhs, rhs);
    return success();
  }
};

/// Convert bool.cmp to stablehlo.compare.
/// Predicate enum: eq=0, ne=1, lt=2, le=3, gt=4, ge=5.
class BoolCmpPattern : public ConversionPattern {
public:
  BoolCmpPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "bool.cmp", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2 || op->getNumResults() != 1)
      return failure();

    auto predicateAttr = op->getAttr("predicate");
    if (!predicateAttr)
      return failure();

    auto predValue = parseBoolCmpPredicate(predicateAttr);
    if (!predValue || *predValue < 0 || *predValue > 5)
      return op->emitError("unsupported bool.cmp predicate");

    using Dir = stablehlo::ComparisonDirection;
    const Dir dirs[] = {Dir::EQ, Dir::NE, Dir::LT, Dir::LE, Dir::GT, Dir::GE};

    auto resultType = RankedTensorType::get({}, rewriter.getI1Type());
    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, resultType, operands[0], operands[1], dirs[*predValue]);
    return success();
  }
};

/// Lower llzk.nondet to zero constant (placeholder).
class LlzkNonDetPattern : public ConversionPattern {
public:
  LlzkNonDetPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "llzk.nondet", /*benefit=*/1, ctx) {}

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

    auto zeroAttr = typeConverter->createConstantAttr(tensorType, 0, rewriter);
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType,
                                                       zeroAttr);
    return success();
  }
};

/// Convert cast.toindex to field → integer → index conversion.
/// Used for array indexing with felt-typed loop variables.
class CastToIndexPattern : public ConversionPattern {
public:
  CastToIndexPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "cast.toindex", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.empty() || op->getNumResults() != 1)
      return failure();

    Location loc = op->getLoc();
    Value input = operands[0];

    // Convert field element to tensor<i32> for use as StableHLO slice index.
    auto i32TensorType = RankedTensorType::get({}, rewriter.getI32Type());
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, i32TensorType, input);
    return success();
  }
};

} // namespace

void populateRemovalPatterns(LlzkToStablehloTypeConverter &converter,
                             RewritePatternSet &patterns,
                             ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();
  target.addIllegalDialect("constrain");
  target.addIllegalDialect("bool");
  target.addIllegalDialect("llzk");

  patterns.add<StructDefErasePattern>(converter, ctx);
  patterns.add<StructMemberErasePattern>(converter, ctx);
  patterns.add<ConstrainFunctionErasePattern>(converter, ctx);
  patterns.add<OpErasePattern>("constrain.eq", converter, ctx);
  patterns.add<OpErasePattern>("bool.assert", converter, ctx);
  patterns.add<BoolCmpPattern>(converter, ctx);
  // bool.or/and → arith.ori/andi (boolean ops on i1, no type conversion)
  patterns.add<BoolOrPattern>(converter, ctx);
  patterns.add<BoolAndPattern>(converter, ctx);
  patterns.add<LlzkNonDetPattern>(converter, ctx);
  patterns.add<CastToIndexPattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
