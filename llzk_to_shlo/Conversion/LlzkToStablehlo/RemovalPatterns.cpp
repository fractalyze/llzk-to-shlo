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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Check if a function.def is a constrain function.
bool isConstrainFunction(Operation *op) {
  // Check by function name
  if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
    if (symNameAttr.getValue() == "constrain") {
      return true;
    }
  }
  // Check by function kind attribute
  if (auto kindAttr = op->getAttrOfType<StringAttr>("function_kind")) {
    return kindAttr.getValue() == "constrain";
  }
  return false;
}

/// Pattern to erase struct.def operations.
/// Struct definitions become tensors, so the struct.def wrapper is removed.
class StructDefErasePattern : public ConversionPattern {
public:
  StructDefErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.def", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only erase struct.def when all nested operations have been converted.
    // function.def operations are moved to module level during conversion,
    // so we wait until the body is empty.
    if (!op->getRegions().empty()) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          // Check if there are any operations that haven't been converted yet
          for (Operation &nestedOp : block.getOperations()) {
            // Skip struct.field - they are erased separately
            if (nestedOp.getName().getStringRef() == "struct.field") {
              continue;
            }
            // If there's any other nested operation, wait for it to be
            // converted
            return failure();
          }
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to erase struct.field operations.
/// Field definitions are absorbed into the tensor layout during conversion.
class StructFieldErasePattern : public ConversionPattern {
public:
  StructFieldErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "struct.field", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to erase function.def @constrain operations.
/// Constraint functions are not needed at runtime in StableHLO.
class ConstrainFunctionErasePattern : public ConversionPattern {
public:
  ConstrainFunctionErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "function.def", /*benefit=*/2, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only erase constrain functions
    if (!isConstrainFunction(op)) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to erase constrain.eq operations.
/// Constraints are not needed at runtime.
class ConstrainEqErasePattern : public ConversionPattern {
public:
  ConstrainEqErasePattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "constrain.eq", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateRemovalPatterns(LlzkToStablehloTypeConverter &converter,
                             RewritePatternSet &patterns,
                             ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();

  // Mark constrain dialect as illegal
  target.addIllegalDialect("constrain");

  patterns.add<StructDefErasePattern>(converter, ctx);
  patterns.add<StructFieldErasePattern>(converter, ctx);
  patterns.add<ConstrainFunctionErasePattern>(converter, ctx);
  patterns.add<ConstrainEqErasePattern>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
