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

/// Get the struct name from a parent struct.def operation.
/// Walks up the operation hierarchy to find struct.def and returns its name.
std::optional<StringRef> getParentStructName(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (parent->getName().getStringRef() == "struct.def") {
      if (auto symNameAttr = parent->getAttrOfType<StringAttr>("sym_name")) {
        return symNameAttr.getValue();
      }
    }
    parent = parent->getParentOp();
  }
  return std::nullopt;
}

/// Check if a function.def is a compute function.
/// Compute functions have the function kind attribute set to "compute" or the
/// name @compute.
bool isComputeFunction(Operation *op) {
  // Check by function name
  if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
    if (symNameAttr.getValue() == "compute") {
      return true;
    }
  }
  // Check by function kind attribute
  if (auto kindAttr = op->getAttrOfType<StringAttr>("function_kind")) {
    return kindAttr.getValue() == "compute";
  }
  return false;
}

/// Pattern to convert function.def @compute to func.func.
/// The result function is named @StructName_compute.
class FunctionDefToFuncOp : public ConversionPattern {
public:
  FunctionDefToFuncOp(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match function.def operations
    if (op->getName().getStringRef() != "function.def") {
      return failure();
    }

    // Only convert compute functions
    if (!isComputeFunction(op)) {
      return failure();
    }

    auto *typeConverter =
        static_cast<const LlzkToStablehloTypeConverter *>(getTypeConverter());

    // Get function type
    auto funcTypeAttr = op->getAttrOfType<TypeAttr>("function_type");
    if (!funcTypeAttr) {
      return op->emitError("function.def missing function_type attribute");
    }
    auto funcType = dyn_cast<FunctionType>(funcTypeAttr.getValue());
    if (!funcType) {
      return op->emitError("function_type attribute is not a FunctionType");
    }

    // Convert function signature types
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    for (Type inputType : funcType.getInputs()) {
      Type converted = typeConverter->convertType(inputType);
      if (!converted) {
        return op->emitError("failed to convert input type");
      }
      convertedInputs.push_back(converted);
    }
    for (Type resultType : funcType.getResults()) {
      Type converted = typeConverter->convertType(resultType);
      if (!converted) {
        return op->emitError("failed to convert result type");
      }
      convertedResults.push_back(converted);
    }

    auto newFuncType =
        FunctionType::get(op->getContext(), convertedInputs, convertedResults);

    // Build the new function name: @StructName_compute
    std::string newName;
    if (auto structName = getParentStructName(op)) {
      newName = structName->str() + "_compute";
    } else {
      newName = "compute";
    }

    // Find the parent module to insert the func.func at module level
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return op->emitError("function.def must be inside a module");
    }

    // Save the current insertion point to restore later
    OpBuilder::InsertionGuard guard(rewriter);

    // Set insertion point to end of module body (before the terminator if any)
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    // Create new func.func operation at module level
    auto funcOp =
        rewriter.create<func::FuncOp>(op->getLoc(), newName, newFuncType);

    // Copy the body region
    Region &srcRegion = op->getRegion(0);
    Region &dstRegion = funcOp.getBody();
    rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());

    // Convert block argument types
    if (!dstRegion.empty()) {
      Block &entryBlock = dstRegion.front();
      for (auto [idx, arg] : llvm::enumerate(entryBlock.getArguments())) {
        arg.setType(convertedInputs[idx]);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert function.return to func.return.
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

  // Mark function dialect operations as illegal
  target.addIllegalDialect("function");

  patterns.add<FunctionDefToFuncOp>(converter, ctx);
  patterns.add<FunctionReturnToReturn>(converter, ctx);
}

} // namespace mlir::llzk_to_shlo
