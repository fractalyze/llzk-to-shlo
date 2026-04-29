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
    // Fallback to arith.ori for i1 scalars that haven't been type-converted
    // to tensors yet (e.g., inside scf.while/if bodies where block args
    // remain as i1 until the structural type conversion runs).
    if (!isa<RankedTensorType>(lhs.getType()) ||
        !isa<RankedTensorType>(rhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::OrIOp>(op, operands[0], operands[1]);
      return success();
    }
    rewriter.replaceOpWithNewOp<stablehlo::OrOp>(op, lhs, rhs);
    return success();
  }
};

/// Convert bool.not to xor with true (boolean NOT).
class BoolNotPattern : public ConversionPattern {
public:
  BoolNotPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "bool.not", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return failure();
    Value input = lookThroughCast(operands[0]);
    Location loc = op->getLoc();
    if (!isa<RankedTensorType>(input.getType())) {
      // Scalar i1: use arith.xori with true
      auto trueVal =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, input, trueVal);
    } else {
      // tensor<i1>: use stablehlo.not
      rewriter.replaceOpWithNewOp<stablehlo::NotOp>(op, input);
    }
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
    // Fallback to arith.andi for i1 scalars that haven't been type-converted
    // to tensors yet (e.g., inside scf.while/if bodies where block args
    // remain as i1 until the structural type conversion runs).
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

    // Dead nondet: erase directly. SimplifySubComponents Phase 5
    // (`replaceRemainingPodOps`) blanket-substitutes every surviving
    // `pod.read` with `llzk.nondet` of the read's result type, including
    // `pod.read [@params] : !pod.type<[]>` from the dispatcher's empty
    // template-params field. Those reads are dead by construction (empty
    // pod carries no value to consume) and the resulting nondet has no
    // numeric element type to materialize a tensor zero for.
    if (op->getResult(0).use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    const auto &tc = getConverter(getTypeConverter());
    Type origType = op->getResult(0).getType();
    Type resultType = tc.convertType(origType);

    // Handle bare scalar types (e.g., i1) that aren't converted by the
    // LLZK type converter: wrap in tensor<>.
    if (!resultType)
      resultType = RankedTensorType::get({}, origType);

    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType)
      return failure();

    // For boolean types, use dense<false>; for integers, use dense<0>.
    Attribute zeroVal;
    if (tensorType.getElementType().isInteger(1))
      zeroVal = DenseElementsAttr::get(tensorType, rewriter.getBoolAttr(false));
    else
      zeroVal = tc.createConstantAttr(tensorType, 0, rewriter);

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, tensorType, zeroVal);
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

    // Convert field element to tensor<i32> for use as StableHLO slice index.
    auto i32TensorType = RankedTensorType::get({}, rewriter.getI32Type());
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, i32TensorType,
                                                      operands[0]);
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
  patterns.add<BoolNotPattern>(converter, ctx);
  patterns.add<LlzkNonDetPattern>(converter, ctx);
  patterns.add<CastToIndexPattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
