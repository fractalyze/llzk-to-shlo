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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LoweringInfrastructure.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

bool involvesPodType(Type ty) {
  if (isa<llzk::pod::PodType, llzk::component::StructType>(ty))
    return true;
  if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
    return involvesPodType(arrTy.getElementType());
  return false;
}

bool isPromotableCarryType(Type ty) {
  if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
    return !isa<llzk::pod::PodType>(arrTy.getElementType());
  return isa<llzk::component::StructType>(ty);
}

namespace {

// ===----------------------------------------------------------------------===
// Utility: check if a struct is the llzk.main entry point
// ===----------------------------------------------------------------------===

bool isMainStruct(ModuleOp module, Operation *structDefOp) {
  auto mainAttr = module->getAttrOfType<TypeAttr>("llzk.main");
  if (!mainAttr)
    return false;
  auto mainTy = dyn_cast<llzk::component::StructType>(mainAttr.getValue());
  if (!mainTy)
    return false;

  auto structNameAttr = structDefOp->getAttrOfType<StringAttr>("sym_name");
  if (!structNameAttr)
    return false;

  SymbolRefAttr candidate =
      FlatSymbolRefAttr::get(module.getContext(), structNameAttr.getValue());
  if (auto *moduleParent = structDefOp->getParentOp()) {
    if (isa<ModuleOp>(moduleParent)) {
      if (auto moduleName = moduleParent->getAttrOfType<StringAttr>("sym_name"))
        candidate = SymbolRefAttr::get(
            module.getContext(), moduleName.getValue(),
            FlatSymbolRefAttr::get(module.getContext(),
                                   structNameAttr.getValue()));
    }
  }
  return mainTy.getNameRef() == candidate;
}

// ===----------------------------------------------------------------------===
// Structural conversion patterns
// ===----------------------------------------------------------------------===

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

} // namespace

void addStructuralConversionPatterns(
    LlzkToStablehloTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  patterns.add<ReturnOpConversion>(typeConverter, patterns.getContext());
  // SCF type conversion handled by populateSCFStructuralTypeConversions
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

    DenseSet<StringAttr> writemTargets = collectWritemTargets(parent);

    int64_t offset = 0;
    for (Region &region : parent->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          if (nested.getName().getStringRef() == "struct.member") {
            if (auto name = nested.getAttrOfType<StringAttr>("sym_name")) {
              if (!writemTargets.contains(name))
                continue;
              typeConverter.registerFieldOffset(structType, name.getValue(),
                                                offset);
              auto memberTypeAttr = nested.getAttrOfType<TypeAttr>("type");
              offset +=
                  memberTypeAttr
                      ? getMemberFlatSize(memberTypeAttr.getValue(), module)
                      : 1;
            }
          }
        }
      }
    }
    typeConverter.registerStructFlattenedSize(structType, offset);
  });
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
    // or "Module_Struct_compute" when the struct sits inside a named
    // sub-module (LLZK v2 template-removal leaves a `module @X` shell around
    // each `struct.def`, so circom-emitted calls are `@X::@X::@compute`).
    std::string name = "main";
    if (auto *structParent = op->getParentOp()) {
      if (structParent->getName().getStringRef() == "struct.def") {
        if (auto sn = structParent->getAttrOfType<StringAttr>("sym_name")) {
          if (!isMainStruct(module, structParent)) {
            std::string prefix;
            if (auto *moduleParent = structParent->getParentOp()) {
              if (isa<ModuleOp>(moduleParent)) {
                if (auto mn =
                        moduleParent->getAttrOfType<StringAttr>("sym_name"))
                  prefix = mn.getValue().str() + "_";
              }
            }
            name = prefix + sn.getValue().str() + "_compute";
          }
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

} // namespace mlir::llzk_to_shlo
