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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// ===----------------------------------------------------------------------===
// Prime field parsing
// ===----------------------------------------------------------------------===

struct PrimeAlias {
  const char *name;
  const char *modulus;
  unsigned bitWidth;
};

const PrimeAlias kPrimeAliases[] = {
    // clang-format off
    {"bn254", "21888242871839275222246405745257275088548364400416034343698204186575808495617", 256},
    // clang-format on
};

std::string resolvePrimeAlias(llvm::StringRef primeStr) {
  for (const auto &alias : kPrimeAliases) {
    if (primeStr.equals_insensitive(alias.name))
      return std::string(alias.modulus) + ":i" + std::to_string(alias.bitWidth);
  }
  return primeStr.str();
}

std::pair<llvm::APInt, unsigned> parsePrimeString(llvm::StringRef primeStr) {
  std::string resolved = resolvePrimeAlias(primeStr);
  auto [valueStr, typeStr] = llvm::StringRef(resolved).split(':');
  unsigned bitWidth = 64;
  if (!typeStr.empty() && typeStr.starts_with("i"))
    bitWidth = std::strtoul(typeStr.drop_front(1).str().c_str(), nullptr, 10);
  return {llvm::APInt(bitWidth, valueStr, 10), bitWidth};
}

// ===----------------------------------------------------------------------===
// Utility: check if a struct is the llzk.main entry point
// ===----------------------------------------------------------------------===

bool isMainStruct(ModuleOp module, StringRef structName) {
  auto mainAttr = module->getAttr("llzk.main");
  if (!mainAttr)
    return false;
  std::string s;
  llvm::raw_string_ostream os(s);
  mainAttr.print(os);
  return s.find(structName) != std::string::npos;
}

// ===----------------------------------------------------------------------===
// Structural conversion patterns
// ===----------------------------------------------------------------------===

/// Convert scf.while to stablehlo.while with type conversion.
/// scf.while carries !felt.type values that become tensor<!pf> values.
class ScfWhileToStablehloWhile : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    SmallVector<Type> convertedTypes;
    for (Type t : op.getResultTypes()) {
      Type c = getTypeConverter()->convertType(t);
      convertedTypes.push_back(c ? c : t);
    }

    // Create stablehlo.while with converted init values
    auto whileOp = rewriter.create<stablehlo::WhileOp>(
        op.getLoc(), convertedTypes, adaptor.getInits());

    // Move condition region
    {
      Region &condRegion = whileOp.getCond();
      rewriter.inlineRegionBefore(op.getBefore(), condRegion, condRegion.end());
      // Convert block arg types
      Block &condBlock = condRegion.front();
      for (auto [idx, arg] : llvm::enumerate(condBlock.getArguments())) {
        if (idx < convertedTypes.size())
          arg.setType(convertedTypes[idx]);
      }
      // Replace scf.condition with stablehlo.return of the predicate
      condBlock.walk([&](Operation *termOp) {
        if (auto condOp = dyn_cast<scf::ConditionOp>(termOp)) {
          rewriter.setInsertionPoint(condOp);
          // The predicate needs to be a tensor<i1>
          Value pred = condOp.getCondition();
          if (!isa<RankedTensorType>(pred.getType())) {
            auto tensorPred = rewriter.create<tensor::FromElementsOp>(
                condOp.getLoc(), RankedTensorType::get({}, pred.getType()),
                pred);
            pred = tensorPred;
          }
          rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(condOp,
                                                           ValueRange{pred});
        }
      });
    }

    // Move body region
    {
      Region &bodyRegion = whileOp.getBody();
      rewriter.inlineRegionBefore(op.getAfter(), bodyRegion, bodyRegion.end());
      Block &bodyBlock = bodyRegion.front();
      for (auto [idx, arg] : llvm::enumerate(bodyBlock.getArguments())) {
        if (idx < convertedTypes.size())
          arg.setType(convertedTypes[idx]);
      }
      // Replace scf.yield with stablehlo.return
      bodyBlock.walk([&](Operation *termOp) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(termOp)) {
          rewriter.setInsertionPoint(yieldOp);
          rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(
              yieldOp, yieldOp.getOperands());
        }
      });
    }

    rewriter.replaceOp(op, whileOp.getResults());
    return success();
  }
};

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

void addStructuralConversionPatterns(
    LlzkToStablehloTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  patterns.add<ReturnOpConversion>(typeConverter, patterns.getContext());
  patterns.add<ScfWhileToStablehloWhile>(typeConverter, patterns.getContext());
  target.addLegalOp<func::FuncOp>();
  target.addLegalOp<func::ReturnOp>();
}

// ===----------------------------------------------------------------------===
// Pre-pass helpers
// ===----------------------------------------------------------------------===

/// Register struct member offsets from struct.def for the type converter.
void registerStructFieldOffsets(ModuleOp module,
                                LlzkToStablehloTypeConverter &typeConverter) {
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "struct.new" ||
        op->getNumResults() == 0)
      return;

    Type structType = op->getResult(0).getType();
    Operation *parent = op->getParentOp();
    while (parent && parent->getName().getStringRef() != "struct.def")
      parent = parent->getParentOp();
    if (!parent)
      return;

    int64_t offset = 0;
    for (Region &region : parent->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          if (nested.getName().getStringRef() == "struct.member") {
            if (auto name = nested.getAttrOfType<StringAttr>("sym_name")) {
              typeConverter.registerFieldOffset(structType, name.getValue(),
                                                offset);
              offset++;
            }
          }
        }
      }
    }
    typeConverter.registerStructFlattenedSize(structType, offset);
  });
}

/// Convert a single function.def to func.func at module level.
/// Returns the created func::FuncOp, or nullptr on failure.
func::FuncOp convertFunctionDef(Operation *funcDefOp, StringRef funcName,
                                LlzkToStablehloTypeConverter &typeConverter,
                                OpBuilder &builder, ModuleOp module) {
  auto funcTypeAttr = funcDefOp->getAttrOfType<TypeAttr>("function_type");
  if (!funcTypeAttr)
    return nullptr;
  auto funcType = dyn_cast<FunctionType>(funcTypeAttr.getValue());
  if (!funcType)
    return nullptr;

  SmallVector<Type> convertedInputs, convertedResults;
  for (Type t : funcType.getInputs()) {
    Type c = typeConverter.convertType(t);
    convertedInputs.push_back(c ? c : t);
  }
  for (Type t : funcType.getResults()) {
    Type c = typeConverter.convertType(t);
    convertedResults.push_back(c ? c : t);
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto newType = FunctionType::get(funcDefOp->getContext(), convertedInputs,
                                   convertedResults);
  auto funcOp =
      builder.create<func::FuncOp>(funcDefOp->getLoc(), funcName, newType);

  funcOp.getBody().takeBody(funcDefOp->getRegion(0));
  if (!funcOp.getBody().empty()) {
    Block &entry = funcOp.getBody().front();
    for (auto [idx, arg] : llvm::enumerate(entry.getArguments()))
      arg.setType(convertedInputs[idx]);
  }
  return funcOp;
}

/// Convert all function.def ops to func.func at module level.
/// Compute functions become @main or @StructName_compute.
/// Constrain functions are erased. Helpers keep their name.
void convertAllFunctions(ModuleOp module,
                         LlzkToStablehloTypeConverter &typeConverter,
                         MLIRContext *context) {
  SmallVector<Operation *> computeFuncs, helperFuncs, toErase;

  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "function.def")
      return;
    auto symName = op->getAttrOfType<StringAttr>("sym_name");
    if (!symName)
      return;
    if (symName.getValue() == "compute")
      computeFuncs.push_back(op);
    else if (symName.getValue() == "constrain")
      toErase.push_back(op);
    else
      helperFuncs.push_back(op);
  });

  OpBuilder builder(context);

  for (Operation *op : computeFuncs) {
    // Determine name: main entry → "main", sub-component → "Struct_compute"
    std::string name = "main";
    if (auto *parent = op->getParentOp()) {
      if (parent->getName().getStringRef() == "struct.def") {
        if (auto sn = parent->getAttrOfType<StringAttr>("sym_name")) {
          if (!isMainStruct(module, sn.getValue()))
            name = sn.getValue().str() + "_compute";
        }
      }
    }
    convertFunctionDef(op, name, typeConverter, builder, module);
    toErase.push_back(op);
  }

  for (Operation *op : helperFuncs) {
    auto sn = op->getAttrOfType<StringAttr>("sym_name");
    convertFunctionDef(op, sn ? sn.getValue() : "helper", typeConverter,
                       builder, module);
    toErase.push_back(op);
  }

  for (Operation *op : toErase)
    op->erase();

  // Erase now-empty struct.def shells
  SmallVector<Operation *> structs;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "struct.def")
      structs.push_back(op);
  });
  for (Operation *op : structs)
    op->erase();
}

/// Convert struct.writem from mutable to SSA form.
/// Each writem gets a result, and subsequent uses of the struct value
/// are updated to use the latest write result.
void convertWritemToSSA(ModuleOp module) {
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Block *block) {
      llvm::DenseMap<Value, Value> latestValue;

      for (Operation &op : llvm::make_early_inc_range(block->getOperations())) {
        // Rewire operands to latest SSA values
        for (auto &operand : op.getOpOperands()) {
          auto it = latestValue.find(operand.get());
          if (it != latestValue.end())
            operand.set(it->second);
        }

        if (op.getName().getStringRef() != "struct.writem")
          continue;

        Value structVal = op.getOperand(0);
        OpBuilder b(&op);
        OperationState state(op.getLoc(), "struct.writem");
        state.addOperands(op.getOperands());
        state.addTypes({structVal.getType()});
        for (auto &attr : op.getAttrs())
          state.addAttribute(attr.getName(), attr.getValue());

        Operation *newOp = b.create(state);
        latestValue[structVal] = newOp->getResult(0);
        op.erase();
      }
    });
  });
}

// ===----------------------------------------------------------------------===
// Main pass
// ===----------------------------------------------------------------------===

struct LlzkToStablehlo : impl::LlzkToStablehloBase<LlzkToStablehlo> {
  using LlzkToStablehloBase::LlzkToStablehloBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    auto [primeValue, storageBitWidth] = parsePrimeString(prime);
    LlzkToStablehloTypeConverter typeConverter(
        context, primeValue, storageBitWidth, usePrimeFieldType);

    // Conversion target: legal and illegal dialects
    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect, arith::ArithDialect,
                           tensor::TensorDialect, prime_ir::field::FieldDialect,
                           func::FuncDialect>();
    // SCF is NOT legal — scf.while must be converted to stablehlo.while
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    for (StringRef d : {"struct", "function", "constrain", "felt", "array",
                        "component", "bool", "llzk", "cast", "pod", "poly"})
      target.addIllegalDialect(d);

    // Conversion patterns
    RewritePatternSet patterns(context);
    populateFeltToStablehloPatterns(typeConverter, patterns, target);
    populateStructToStablehloPatterns(typeConverter, patterns, target);
    populateArrayToStablehloPatterns(typeConverter, patterns, target);
    populateFunctionToFuncPatterns(typeConverter, patterns, target);
    populateRemovalPatterns(typeConverter, patterns, target);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    context->loadDialect<func::FuncDialect>();

    // Pre-passes: transform LLZK IR before dialect conversion
    registerStructFieldOffsets(module, typeConverter);
    convertAllFunctions(module, typeConverter, context);
    convertWritemToSSA(module);

    // Strip llzk.* module attributes AFTER function conversion
    // (isMainStruct reads llzk.main during convertAllFunctions)
    SmallVector<StringRef> attrsToRemove;
    for (auto attr : module->getAttrs())
      if (attr.getName().getValue().starts_with("llzk."))
        attrsToRemove.push_back(attr.getName().getValue());
    for (auto name : attrsToRemove)
      module->removeAttr(name);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
