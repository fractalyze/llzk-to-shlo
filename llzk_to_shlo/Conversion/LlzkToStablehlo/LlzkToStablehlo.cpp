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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkToStablehlo.h"

#include <cstdlib>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FeltPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FunctionPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/RemovalPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_LLZKTOSTABLEHLO
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkToStablehlo.h.inc"

namespace {

/// Known prime field aliases with their modulus and storage bit width.
struct PrimeAlias {
  const char *name;
  const char *modulus; // String because BN254 doesn't fit in uint64_t
  unsigned bitWidth;
};

// clang-format off
const PrimeAlias kPrimeAliases[] = {
    {"bn254",
     "21888242871839275222246405745257275088548364400416034343698204186575808495617",
     256},
};
// clang-format on

/// Resolve prime alias to modulus:bits format.
/// Returns the input unchanged if not an alias.
std::string resolvePrimeAlias(llvm::StringRef primeStr) {
  for (const auto &alias : kPrimeAliases) {
    if (primeStr.equals_insensitive(alias.name)) {
      return std::string(alias.modulus) + ":i" + std::to_string(alias.bitWidth);
    }
  }
  return primeStr.str();
}

/// Parse prime string in format "<value>:<type>" or alias (e.g., "bn254").
/// Returns (primeValue as APInt, storageBitWidth).
std::pair<llvm::APInt, unsigned> parsePrimeString(llvm::StringRef primeStr) {
  // First resolve any alias
  std::string resolved = resolvePrimeAlias(primeStr);

  auto [valueStr, typeStr] = llvm::StringRef(resolved).split(':');

  unsigned bitWidth = 64; // Default
  if (!typeStr.empty() && typeStr.starts_with("i")) {
    bitWidth = std::strtoul(typeStr.drop_front(1).str().c_str(), nullptr, 10);
  }

  // Parse as APInt to handle large primes like BN254
  llvm::APInt primeValue(bitWidth, valueStr, 10);

  return {primeValue, bitWidth};
}

/// Pattern to convert func.return with converted operand types
class ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Check if a type is legal (not an LLZK type that needs conversion).
/// Simply checks if it's a tensor type or a standard type.
bool isTypeLegal(Type type) {
  // Tensor types are legal (these are our target types)
  if (isa<RankedTensorType, UnrankedTensorType>(type))
    return true;
  // Standard scalar types are legal
  if (isa<IntegerType, FloatType, IndexType>(type))
    return true;
  // FunctionType needs recursive check
  if (auto funcType = dyn_cast<FunctionType>(type)) {
    for (Type input : funcType.getInputs()) {
      if (!isTypeLegal(input))
        return false;
    }
    for (Type result : funcType.getResults()) {
      if (!isTypeLegal(result))
        return false;
    }
    return true;
  }
  // Everything else (including LLZK types) is not legal
  return false;
}

/// Add structural conversion patterns for function signatures.
void addStructuralConversionPatterns(
    LlzkToStablehloTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  // Use MLIR's built-in function conversion patterns
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);

  // Add return op pattern
  patterns.add<ReturnOpConversion>(typeConverter, patterns.getContext());

  // For now, mark func.func and func.return as legal
  // The pre-pass handles LLZK function conversion
  // TODO: Re-enable dynamic legality checking with proper null checks
  target.addLegalOp<func::FuncOp>();
  target.addLegalOp<func::ReturnOp>();
}

struct LlzkToStablehlo : impl::LlzkToStablehloBase<LlzkToStablehlo> {
  using LlzkToStablehloBase::LlzkToStablehloBase;

  void runOnOperation() override;
};

void LlzkToStablehlo::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Parse prime string (format: "<value>:<type>", e.g., "2013265921:i32")
  auto [primeValue, storageBitWidth] = parsePrimeString(prime);

  // Create type converter with the specified prime and field type flag
  LlzkToStablehloTypeConverter typeConverter(
      context, primeValue, storageBitWidth, usePrimeFieldType);

  // Set up conversion target
  ConversionTarget target(*context);

  // Mark StableHLO and supporting dialects as legal
  target.addLegalDialect<stablehlo::StablehloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<prime_ir::field::FieldDialect>();

  // Mark func dialect operations as dynamically legal
  target.addLegalDialect<func::FuncDialect>();

  // Mark builtin operations as legal
  target.addLegalOp<ModuleOp>();

  // Mark all LLZK dialects as illegal
  target.addIllegalDialect("struct");
  target.addIllegalDialect("function");
  target.addIllegalDialect("constrain");
  target.addIllegalDialect("felt");
  target.addIllegalDialect("array");
  target.addIllegalDialect("component");

  // Allow unrealized conversion casts during partial conversion
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Collect conversion patterns
  RewritePatternSet patterns(context);

  // Add felt operation patterns
  populateFeltToStablehloPatterns(typeConverter, patterns, target);

  // Add struct operation patterns
  populateStructToStablehloPatterns(typeConverter, patterns, target);

  // Add array operation patterns
  populateArrayToStablehloPatterns(typeConverter, patterns, target);

  // Add function conversion patterns (function.def @compute -> func.func)
  populateFunctionToFuncPatterns(typeConverter, patterns, target);

  // Add removal patterns (erase struct.def, struct.field, constrain.*)
  populateRemovalPatterns(typeConverter, patterns, target);

  // Add structural conversion patterns (functions, etc.)
  addStructuralConversionPatterns(typeConverter, patterns, target);

  // Load func dialect before creating func.func operations
  context->loadDialect<func::FuncDialect>();

  // Pre-pass 1: Find struct.new operations to get the actual struct type and
  // register field offsets from the parent struct.def
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "struct.new") {
      if (op->getNumResults() > 0) {
        Type structType = op->getResult(0).getType();

        // Find the parent struct.def to get field info
        Operation *parent = op->getParentOp();
        while (parent && parent->getName().getStringRef() != "struct.def") {
          parent = parent->getParentOp();
        }

        if (parent) {
          int64_t offset = 0;
          for (Region &region : parent->getRegions()) {
            for (Block &block : region) {
              for (Operation &nestedOp : block) {
                if (nestedOp.getName().getStringRef() == "struct.field") {
                  auto fieldName =
                      nestedOp.getAttrOfType<StringAttr>("sym_name");
                  if (fieldName) {
                    typeConverter.registerFieldOffset(
                        structType, fieldName.getValue(), offset);
                    offset++;
                  }
                }
              }
            }
          }
          typeConverter.registerStructFlattenedSize(structType, offset);
        }
      }
    }
  });

  // Pre-pass 2: Convert function.def @compute to func.func at module level
  // This needs to be done before the main conversion because the conversion
  // framework doesn't properly handle nested operations in unregistered
  // dialects
  SmallVector<Operation *> computeFunctions;
  SmallVector<Operation *> opsToErase;

  // Collect compute functions first (avoid modifying while walking)
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "function.def") {
      // Check if it's a compute function
      if (auto symName = op->getAttrOfType<StringAttr>("sym_name")) {
        if (symName.getValue() == "compute") {
          computeFunctions.push_back(op);
        } else {
          opsToErase.push_back(op);
        }
      }
    }
  });

  // Convert compute functions to func.func
  OpBuilder builder(context);
  for (Operation *funcDefOp : computeFunctions) {
    // Get the function type
    auto funcTypeAttr = funcDefOp->getAttrOfType<TypeAttr>("function_type");
    if (!funcTypeAttr)
      continue;

    auto funcType = dyn_cast<FunctionType>(funcTypeAttr.getValue());
    if (!funcType)
      continue;

    // Convert types
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    for (Type input : funcType.getInputs()) {
      Type converted = typeConverter.convertType(input);
      convertedInputs.push_back(converted ? converted : input);
    }
    for (Type result : funcType.getResults()) {
      Type converted = typeConverter.convertType(result);
      convertedResults.push_back(converted ? converted : result);
    }

    // Get parent struct name
    std::string funcName = "compute";
    if (Operation *parent = funcDefOp->getParentOp()) {
      if (parent->getName().getStringRef() == "struct.def") {
        if (auto symName = parent->getAttrOfType<StringAttr>("sym_name")) {
          funcName = symName.getValue().str() + "_compute";
        }
      }
    }

    // Create func.func at module level
    builder.setInsertionPointToEnd(module.getBody());
    auto newFuncType =
        FunctionType::get(context, convertedInputs, convertedResults);
    auto funcOp = builder.create<func::FuncOp>(funcDefOp->getLoc(), funcName,
                                               newFuncType);

    // Move the body region
    Region &srcRegion = funcDefOp->getRegion(0);
    Region &dstRegion = funcOp.getBody();
    dstRegion.takeBody(srcRegion);

    // Convert block argument types
    if (!dstRegion.empty()) {
      Block &entryBlock = dstRegion.front();
      for (auto [idx, arg] : llvm::enumerate(entryBlock.getArguments())) {
        arg.setType(convertedInputs[idx]);
      }

      // Don't convert function.return here - let the main conversion handle it
      // after struct.new and other operations are converted
    }

    opsToErase.push_back(funcDefOp);
  }

  // Erase the original function.def operations
  for (Operation *op : opsToErase) {
    op->erase();
  }

  // Also erase struct.def operations (they should be empty now)
  SmallVector<Operation *> structsToErase;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "struct.def") {
      structsToErase.push_back(op);
    }
  });
  for (Operation *op : structsToErase) {
    op->erase();
  }

  // Apply partial conversion (full conversion is too strict for func.func)
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

} // namespace mlir::llzk_to_shlo
