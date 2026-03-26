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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FunctionPatterns.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Convert function.call to func.call with type conversion.
/// Handles nested symbol references: @Struct::@compute → @Struct_compute.
class FunctionCallToFuncCall : public ConversionPattern {
public:
  FunctionCallToFuncCall(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "function.call", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());

    // Resolve callee name: flat or nested symbol reference
    std::string calleeName;
    if (auto flat = op->getAttrOfType<FlatSymbolRefAttr>("callee")) {
      calleeName = flat.getValue().str();
    } else if (auto nested = op->getAttrOfType<SymbolRefAttr>("callee")) {
      calleeName = nested.getRootReference().getValue().str();
      for (auto ref : nested.getNestedReferences())
        calleeName += "_" + ref.getValue().str();
    } else {
      return failure();
    }

    SmallVector<Type> convertedResults;
    for (Type t : op->getResultTypes()) {
      Type c = typeConverter->convertType(t);
      convertedResults.push_back(c ? c : t);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, FlatSymbolRefAttr::get(op->getContext(), calleeName),
        convertedResults, operands);
    return success();
  }
};

/// Convert function.return to func.return.
class FunctionReturnToReturn : public ConversionPattern {
public:
  FunctionReturnToReturn(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, "function.return", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);
    return success();
  }
};

} // namespace

void populateFunctionToFuncPatterns(LlzkToStablehloTypeConverter &converter,
                                    RewritePatternSet &patterns,
                                    ConversionTarget &target) {
  MLIRContext *ctx = patterns.getContext();
  target.addIllegalDialect("function");
  patterns.add<FunctionCallToFuncCall>(converter, ctx);
  patterns.add<FunctionReturnToReturn>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
