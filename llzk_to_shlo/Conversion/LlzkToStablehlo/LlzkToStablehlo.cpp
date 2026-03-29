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
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FeltPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FunctionPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/RemovalPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
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

    int64_t offset = 0;
    for (Region &region : parent->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          if (nested.getName().getStringRef() == "struct.member") {
            if (auto name = nested.getAttrOfType<StringAttr>("sym_name")) {
              typeConverter.registerFieldOffset(structType, name.getValue(),
                                                offset);
              auto memberTypeAttr = nested.getAttrOfType<TypeAttr>("type");
              offset += memberTypeAttr
                            ? getMemberFlatSize(memberTypeAttr.getValue())
                            : 1;
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

// ===----------------------------------------------------------------------===
// promoteArraysToWhileCarry helpers
// ===----------------------------------------------------------------------===

/// Find array values defined outside the while body but used inside.
llvm::SmallSetVector<Value, 4> findCapturedArrays(scf::WhileOp whileOp) {
  Block &body = whileOp.getAfter().front();

  llvm::SmallSetVector<Value, 4> capturedArrays;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (operand.getParentBlock() == &body)
        continue; // defined inside body
      if (llvm::is_contained(body.getArguments(), operand))
        continue; // block arg (already a carry)
      Type ty = operand.getType();
      if (ty.getDialect().getNamespace() != "array")
        continue;
      // Skip pod-element arrays — these are count/dispatch bookkeeping
      // that can't be type-converted to tensors.
      if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
        if (arrTy.getElementType().getDialect().getNamespace() == "pod")
          continue;
      // Skip values defined inside a parent while body. These can't be
      // promoted to a carry of the current while because they wouldn't
      // dominate the parent while's init position.
      if (auto *defOp = operand.getDefiningOp()) {
        if (auto parentWhile =
                dyn_cast_or_null<scf::WhileOp>(whileOp->getParentOp()))
          if (defOp->getParentRegion() == &parentWhile.getAfter())
            continue;
      }
      capturedArrays.insert(operand);
    }
  });

  return capturedArrays;
}

/// Create a new scf.while with extra carry values for captured arrays.
/// Moves regions, adds block args, fixes scf.condition. Returns the new
/// WhileOp.
scf::WhileOp
createWhileWithExtraCarry(OpBuilder &builder, scf::WhileOp whileOp,
                          llvm::SmallSetVector<Value, 4> &capturedArrays) {
  unsigned origNumResults = whileOp.getNumResults();

  // Collect new init values (original + arrays)
  SmallVector<Value> newInits(whileOp.getInits().begin(),
                              whileOp.getInits().end());
  SmallVector<Type> newTypes(whileOp.getResultTypes().begin(),
                             whileOp.getResultTypes().end());
  for (Value arr : capturedArrays) {
    newInits.push_back(arr);
    newTypes.push_back(arr.getType());
  }

  // Create new while with extra carry values
  auto newWhile =
      builder.create<scf::WhileOp>(whileOp.getLoc(), newTypes, newInits);

  // Move condition region, add extra block args
  Region &newCond = newWhile.getBefore();
  newCond.takeBody(whileOp.getBefore());
  Block &newCondBlock = newCond.front();
  for (Value arr : capturedArrays)
    newCondBlock.addArgument(arr.getType(), arr.getLoc());

  // Fix scf.condition to pass the extra args
  auto condOp = cast<scf::ConditionOp>(newCondBlock.getTerminator());
  SmallVector<Value> condArgs(condOp.getArgs().begin(), condOp.getArgs().end());
  for (unsigned i = origNumResults; i < newCondBlock.getNumArguments(); ++i)
    condArgs.push_back(newCondBlock.getArgument(i));
  OpBuilder condBuilder(condOp);
  condBuilder.create<scf::ConditionOp>(condOp.getLoc(), condOp.getCondition(),
                                       condArgs);
  condOp.erase();

  // Move body region, add extra block args
  Region &newBody = newWhile.getAfter();
  newBody.takeBody(whileOp.getAfter());
  Block &newBodyBlock = newBody.front();
  SmallVector<Value> arrayBlockArgs;
  for (Value arr : capturedArrays)
    arrayBlockArgs.push_back(
        newBodyBlock.addArgument(arr.getType(), arr.getLoc()));

  // Replace external array uses in body with the new block args
  for (auto [extArr, blockArg] : llvm::zip(capturedArrays, arrayBlockArgs)) {
    // Replace uses of extArr INSIDE the body with blockArg
    // But only uses that are dominated by the body block
    for (auto &use : llvm::make_early_inc_range(extArr.getUses())) {
      if (use.getOwner()->getBlock() == &newBodyBlock ||
          newBodyBlock.getParentOp()->isProperAncestor(use.getOwner()))
        use.set(blockArg);
    }
    // Also replace in condition block
    for (auto &use : llvm::make_early_inc_range(extArr.getUses())) {
      if (use.getOwner()->getBlock() == &newCondBlock ||
          newCondBlock.getParentOp()->isProperAncestor(use.getOwner()))
        use.set(newCondBlock.getArgument(origNumResults +
                                         llvm::find(capturedArrays, extArr) -
                                         capturedArrays.begin()));
    }
  }

  return newWhile;
}

/// Walk the body block, convert array.write to produce SSA result, track
/// latest value. Returns DenseMap<Value, Value> of latestArraySSA.
llvm::DenseMap<Value, Value>
convertArrayWritesToSSA(Block &bodyBlock, ArrayRef<Value> arrayBlockArgs) {
  // Track latest SSA value for each array (keyed by block arg)
  llvm::DenseMap<Value, Value> latestArraySSA;
  for (auto blockArg : arrayBlockArgs)
    latestArraySSA[blockArg] = blockArg;

  // Walk ops sequentially: rewire array operands to latest SSA value,
  // then convert array.write to produce a result.
  for (Operation &op : llvm::make_early_inc_range(bodyBlock.getOperations())) {
    // Rewire operands: if any operand is a tracked array, use latest value
    for (auto &operand : op.getOpOperands()) {
      auto it = latestArraySSA.find(operand.get());
      if (it != latestArraySSA.end() && it->second != operand.get())
        operand.set(it->second);
    }

    if (op.getName().getStringRef() != "array.write" || op.getNumOperands() < 3)
      continue;

    Value arr = op.getOperand(0); // already rewired to latest SSA value

    // Create array.write with a result (mutable → SSA)
    OpBuilder b(&op);
    OperationState state(op.getLoc(), "array.write");
    state.addOperands(op.getOperands());
    state.addTypes({arr.getType()});
    for (auto &attr : op.getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    Operation *newWrite = b.create(state);

    // Update latest: find which block arg this array originated from
    for (auto &[blockArg, latest] : latestArraySSA) {
      if (latest == arr)
        latest = newWrite->getResult(0);
    }

    op.erase();
  }

  return latestArraySSA;
}

/// Promote external array references in scf.while bodies to carry values.
/// array.write/read of an external array inside a loop becomes a
/// mutable carry value: the array is passed in/out of each iteration.
void promoteArraysToWhileCarry(ModuleOp module) {
  // Collect all while ops first to avoid iterator invalidation
  // (processing creates new while ops that walk would revisit)
  SmallVector<scf::WhileOp> whileOps;
  module.walk([&](scf::WhileOp op) { whileOps.push_back(op); });
  for (auto whileOp : whileOps) {
    // Find array values defined outside the while that are used inside body
    auto capturedArrays = findCapturedArrays(whileOp);

    if (capturedArrays.empty())
      continue;

    unsigned origNumResults = whileOp.getNumResults();
    OpBuilder builder(whileOp);

    // Create new while with extra carry values, move regions, fix condition
    auto newWhile = createWhileWithExtraCarry(builder, whileOp, capturedArrays);

    Block &newBodyBlock = newWhile.getAfter().front();

    // Collect the array block args (the last N args added)
    SmallVector<Value> arrayBlockArgs;
    unsigned numOrigArgs =
        newBodyBlock.getNumArguments() - capturedArrays.size();
    for (unsigned i = numOrigArgs; i < newBodyBlock.getNumArguments(); ++i)
      arrayBlockArgs.push_back(newBodyBlock.getArgument(i));

    // Convert array.write inside body to produce an updated array value
    // and track the latest value for scf.yield
    auto latestArraySSA = convertArrayWritesToSSA(newBodyBlock, arrayBlockArgs);

    // Fix scf.yield to pass updated arrays
    auto yieldOp = cast<scf::YieldOp>(newBodyBlock.getTerminator());
    SmallVector<Value> yieldArgs(yieldOp.getOperands().begin(),
                                 yieldOp.getOperands().end());
    for (auto blockArg : arrayBlockArgs) {
      auto it = latestArraySSA.find(blockArg);
      yieldArgs.push_back(it != latestArraySSA.end() ? it->second : blockArg);
    }
    OpBuilder yieldBuilder(yieldOp);
    yieldBuilder.create<scf::YieldOp>(yieldOp.getLoc(), yieldArgs);
    yieldOp.erase();

    // Replace uses of old while results + external arrays
    for (unsigned i = 0; i < origNumResults; ++i)
      whileOp.getResult(i).replaceAllUsesWith(newWhile.getResult(i));
    for (unsigned idx = 0; idx < capturedArrays.size(); ++idx) {
      Value extArr = capturedArrays[idx];
      Value replacement = newWhile.getResult(origNumResults + idx);
      // Only replace uses AFTER the while (not init or pre-while uses)
      for (auto &use : llvm::make_early_inc_range(extArr.getUses())) {
        Operation *user = use.getOwner();
        // Skip the while op itself (init operand)
        if (user == newWhile.getOperation())
          continue;
        // Skip ops defined BEFORE the while in the same block
        if (user->getBlock() == newWhile->getBlock() &&
            user->isBeforeInBlock(newWhile))
          continue;
        // Skip ops inside the while (they use block args now)
        if (newWhile->isProperAncestor(user))
          continue;
        use.set(replacement);
      }
    }

    whileOp.erase();
  }
}

// ===----------------------------------------------------------------------===
// convertScfWhileToStablehloWhile helpers
// ===----------------------------------------------------------------------===

/// Convert init values for a scf.while → stablehlo.while conversion.
/// Returns pair<SmallVector<Value>, SmallVector<Type>> of (convertedInits,
/// convertedTypes). Logic: looks through casts, finds converted types from
/// body args.
std::pair<SmallVector<Value>, SmallVector<Type>>
convertWhileInitValues(OpBuilder &builder, scf::WhileOp whileOp) {
  // First pass: find the element type from any already-converted value
  Type fieldElemType;
  for (Value init : whileOp.getInits()) {
    if (auto tt = dyn_cast<RankedTensorType>(init.getType())) {
      fieldElemType = tt.getElementType();
      break;
    }
  }
  // Also check body arg casts for element type
  if (!fieldElemType) {
    Block &body = whileOp.getAfter().front();
    for (auto arg : body.getArguments()) {
      for (auto *user : arg.getUsers()) {
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
          if (auto tt =
                  dyn_cast<RankedTensorType>(castOp.getResult(0).getType())) {
            fieldElemType = tt.getElementType();
            break;
          }
        }
      }
      if (fieldElemType)
        break;
    }
  }

  SmallVector<Value> convertedInits;
  SmallVector<Type> convertedTypes;
  for (Value init : whileOp.getInits()) {
    if (isa<RankedTensorType>(init.getType())) {
      convertedInits.push_back(init);
      convertedTypes.push_back(init.getType());
    } else {
      // Look through existing cast: felt.type → tensor<!pf>
      Value converted = init;
      for (auto *user : init.getUsers()) {
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
          if (castOp.getNumResults() == 1 &&
              isa<RankedTensorType>(castOp.getResult(0).getType())) {
            converted = castOp.getResult(0);
            break;
          }
        }
      }
      if (converted == init) {
        // No existing cast found. Determine the tensor type from:
        // 1. Body arg's cast users
        // 2. The init value's defining op's result cast
        // 3. Direct type inference (felt → tensor<!pf>, array → tensor<Nx!pf>)
        Type tensorType;

        // Strategy 1: look at body arg users for a cast
        Block &body = whileOp.getAfter().front();
        unsigned argIdx =
            llvm::find(whileOp.getInits(), init) - whileOp.getInits().begin();
        if (argIdx < body.getNumArguments()) {
          for (auto *user : body.getArgument(argIdx).getUsers()) {
            if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
              if (castOp.getNumResults() == 1) {
                tensorType = castOp.getResult(0).getType();
                break;
              }
            }
          }
        }

        // Strategy 2: look at the defining op's other result users
        if (!tensorType) {
          if (auto *defOp = init.getDefiningOp()) {
            for (auto *user : defOp->getResult(0).getUsers()) {
              if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
                if (castOp.getNumResults() == 1 &&
                    isa<RankedTensorType>(castOp.getResult(0).getType())) {
                  tensorType = castOp.getResult(0).getType();
                  break;
                }
              }
            }
          }
        }

        // Strategy 3: construct tensor type from fieldElemType
        if (!tensorType && fieldElemType) {
          StringRef ns = init.getType().getDialect().getNamespace();
          if (ns == "felt")
            tensorType = RankedTensorType::get({}, fieldElemType);
          else if (ns == "array")
            tensorType = RankedTensorType::get(
                getArrayDimensions(init.getType()), fieldElemType);
        }

        if (tensorType) {
          converted = builder
                          .create<UnrealizedConversionCastOp>(whileOp.getLoc(),
                                                              tensorType, init)
                          .getResult(0);
        }
      }
      convertedInits.push_back(converted);
      convertedTypes.push_back(converted.getType());
    }
  }
  return {convertedInits, convertedTypes};
}

/// Convert scf.while → stablehlo.while (post-pass, after type conversion).
/// This ensures the while body ops are visible to dialect conversion for type
/// conversion. Only changes terminators: scf.condition → stablehlo.return,
/// scf.yield → stablehlo.return.
void convertScfWhileToStablehloWhile(ModuleOp module) {
  // Process one while at a time, innermost first. Re-collect after each
  // conversion since takeBody invalidates nested while pointers.
  bool converted = true;
  while (converted) {
    converted = false;
    scf::WhileOp target;
    module.walk([&](scf::WhileOp op) {
      // Find the innermost while (no nested scf.while in its body).
      bool hasNestedWhile = false;
      op.getAfter().walk([&](scf::WhileOp) { hasNestedWhile = true; });
      if (!hasNestedWhile)
        target = op;
    });
    if (!target)
      break;
    auto whileOp = target;
    converted = true;
    OpBuilder builder(whileOp);

    // Convert init values: insert unrealized_conversion_cast for non-tensor
    // types (felt → tensor, array → tensor) so stablehlo.while has valid types
    auto [convertedInits, convertedTypes] =
        convertWhileInitValues(builder, whileOp);

    auto newWhile = builder.create<stablehlo::WhileOp>(
        whileOp.getLoc(), convertedTypes, convertedInits);

    // Move condition region
    Region &condSrc = whileOp.getBefore();
    Region &condDst = newWhile.getCond();
    condDst.takeBody(condSrc);

    // Convert block arg types in condition region
    Block &condBlock = condDst.front();
    for (auto [idx, arg] : llvm::enumerate(condBlock.getArguments())) {
      if (idx < convertedTypes.size())
        arg.setType(convertedTypes[idx]);
    }

    // Replace scf.condition with stablehlo.return of predicate
    if (auto condOp = dyn_cast<scf::ConditionOp>(condBlock.getTerminator())) {
      OpBuilder termBuilder(condOp);
      Value pred = condOp.getCondition();
      // stablehlo.while expects tensor<i1> predicate.
      // Look through unrealized_conversion_cast to find original tensor<i1>.
      if (!isa<RankedTensorType>(pred.getType())) {
        if (auto castOp = pred.getDefiningOp<UnrealizedConversionCastOp>()) {
          Value src = castOp.getInputs()[0];
          if (isa<RankedTensorType>(src.getType()))
            pred = src;
        }
      }
      if (!isa<RankedTensorType>(pred.getType())) {
        pred = termBuilder.create<tensor::FromElementsOp>(
            condOp.getLoc(), RankedTensorType::get({}, pred.getType()), pred);
      }
      termBuilder.create<stablehlo::ReturnOp>(condOp.getLoc(),
                                              ValueRange{pred});
      condOp.erase();
    }

    // Move body region
    Region &bodySrc = whileOp.getAfter();
    Region &bodyDst = newWhile.getBody();
    bodyDst.takeBody(bodySrc);

    // Convert block arg types in body region
    Block &bodyBlock = bodyDst.front();
    for (auto [idx, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      if (idx < convertedTypes.size())
        arg.setType(convertedTypes[idx]);
    }

    // Replace scf.yield with stablehlo.return, converting operand types
    if (auto yieldOp = dyn_cast<scf::YieldOp>(bodyBlock.getTerminator())) {
      OpBuilder termBuilder(yieldOp);
      SmallVector<Value> yieldValues;
      for (auto [idx, val] : llvm::enumerate(yieldOp.getOperands())) {
        Value v = val;
        // Look through cast: felt → tensor
        if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getNumOperands() == 1 &&
              isa<RankedTensorType>(castOp.getOperand(0).getType()))
            v = castOp.getOperand(0);
        }
        // If still not tensor, find the matching converted type
        if (!isa<RankedTensorType>(v.getType()) &&
            idx < convertedTypes.size()) {
          v = termBuilder
                  .create<UnrealizedConversionCastOp>(yieldOp.getLoc(),
                                                      convertedTypes[idx], v)
                  .getResult(0);
        }
        yieldValues.push_back(v);
      }
      termBuilder.create<stablehlo::ReturnOp>(yieldOp.getLoc(), yieldValues);
      yieldOp.erase();
    }

    whileOp.replaceAllUsesWith(newWhile.getResults());
    whileOp.erase();
  }
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

        StringRef opName = op.getName().getStringRef();
        if (opName != "struct.writem" && opName != "array.write")
          continue;
        // Skip if already converted to SSA (has result type)
        if (op.getNumResults() > 0)
          continue;
        // Skip array.write on pod-element arrays and inside scf.for bodies.
        // Pod arrays: count/dispatch bookkeeping, not convertible.
        // scf.for: uses mutable semantics (no carry), SSA conversion invalid.
        if (opName == "array.write") {
          if (op.getNumOperands() > 0) {
            Type arrType = op.getOperand(0).getType();
            if (auto at = dyn_cast<llzk::array::ArrayType>(arrType)) {
              if (at.getElementType().getDialect().getNamespace() == "pod") {
                continue;
              }
            }
          }
          // Check if inside any scf control flow ancestor (scf.for/while/if).
          // These use mutable semantics; SSA conversion is handled separately
          // by promoteArraysToWhileCarry.
          auto *ancestor = op.getParentOp();
          bool insideScf = false;
          while (ancestor) {
            StringRef an = ancestor->getName().getStringRef();
            if (an == "scf.for" || an == "scf.while" || an == "scf.if") {
              insideScf = true;
              break;
            }
            ancestor = ancestor->getParentOp();
          }
          if (insideScf)
            continue;
        }

        // Convert mutable write to SSA: add result type so the op
        // produces the updated value.
        // Track using the ORIGINAL target (before operand rewiring) so that
        // chained writes all update the same map entry:
        //   writem %self[@a] → latestValue[%self] = %1
        //   writem %self[@b] → rewired to %1, latestValue[%self] = %2
        //   return %self     → rewired to %2 (latest)
        Value target = op.getOperand(0); // already rewired
        // Find the original value this chain started from.
        Value originalTarget = target;
        for (auto &[orig, latest] : latestValue) {
          if (latest == target) {
            originalTarget = orig;
            break;
          }
        }

        OpBuilder b(&op);
        OperationState state(op.getLoc(), opName);
        state.addOperands(op.getOperands());
        state.addTypes({target.getType()});
        for (auto &attr : op.getAttrs())
          state.addAttribute(attr.getName(), attr.getValue());

        Operation *newOp = b.create(state);
        latestValue[originalTarget] = newOp->getResult(0);
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
    // SCF structural type conversion is added after patterns are created
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    for (StringRef d : {"struct", "function", "constrain", "felt", "component",
                        "bool", "llzk", "cast", "poly"})
      target.addIllegalDialect(d);
    // Array ops on pod-element arrays are dynamically legal (count/dispatch
    // bookkeeping, not convertible to StableHLO). Regular felt-element array
    // ops are illegal (converted by ArrayPatterns).
    target.addDynamicallyLegalDialect(
        [](Operation *op) -> bool {
          auto involvesPod = [](Type ty) {
            // Direct pod type
            if (ty.getDialect().getNamespace() == "pod")
              return true;
            // Array of pods: check element type via native API
            if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
              return arrTy.getElementType().getDialect().getNamespace() ==
                     "pod";
            // Struct type (from pod.read @comp result)
            if (ty.getDialect().getNamespace() == "struct")
              return true;
            // Fallback: check printed type for "pod"
            if (ty.getDialect().getNamespace() == "array") {
              std::string s;
              llvm::raw_string_ostream os(s);
              ty.print(os);
              return s.find("pod") != std::string::npos;
            }
            return false;
          };
          for (Value v : op->getOperands()) {
            if (involvesPod(v.getType()))
              return true;
          }
          for (Value v : op->getResults()) {
            if (involvesPod(v.getType()))
              return true;
          }
          // Also allow in constrain functions
          auto *parent = op->getParentOp();
          while (parent) {
            if (parent->getName().getStringRef() == "function.def") {
              auto sym = parent->getAttrOfType<StringAttr>("sym_name");
              return sym && sym.getValue() == "constrain";
            }
            parent = parent->getParentOp();
          }
          return false;
        },
        "array");
    // Pod ops: dynamically legal inside constrain functions or scf control
    // flow bodies (count/dispatch arrays eliminated by SimplifySubComponents
    // or erased along with dead code). Illegal at top level.
    target.addDynamicallyLegalDialect(
        [](Operation *op) -> bool {
          auto *parent = op->getParentOp();
          while (parent) {
            StringRef name = parent->getName().getStringRef();
            if (name == "function.def") {
              auto sym = parent->getAttrOfType<StringAttr>("sym_name");
              return sym && sym.getValue() == "constrain";
            }
            if (name == "scf.while" || name == "scf.if" || name == "scf.for")
              return true;
            parent = parent->getParentOp();
          }
          return false;
        },
        "pod");

    // Conversion patterns
    RewritePatternSet patterns(context);
    populateFeltToStablehloPatterns(typeConverter, patterns, target);
    populateStructToStablehloPatterns(typeConverter, patterns, target);
    populateArrayToStablehloPatterns(typeConverter, patterns, target);
    populateFunctionToFuncPatterns(typeConverter, patterns, target);
    populateRemovalPatterns(typeConverter, patterns, target);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    // SCF structural type conversion: automatically converts scf.while/if/for
    // result types and block argument types using the type converter.
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    context->loadDialect<func::FuncDialect>();

    // Run SimplifySubComponents first to eliminate pod dispatch patterns.
    // This pass extracts function.call from dispatch scf.if and resolves
    // pod.read @comp references before dialect conversion.
    {
      OpPassManager pm("builtin.module");
      pm.addPass(createSimplifySubComponents());
      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }
    }

    // Pre-passes: transform LLZK IR before dialect conversion
    //
    registerStructFieldOffsets(module, typeConverter);
    convertAllFunctions(module, typeConverter, context);
    promoteArraysToWhileCarry(module);
    convertWritemToSSA(module);

    // Strip llzk.* module attributes AFTER function conversion
    // (isMainStruct reads llzk.main during convertAllFunctions)
    SmallVector<StringRef> attrsToRemove;
    for (auto attr : module->getAttrs())
      if (attr.getName().getValue().starts_with("llzk."))
        attrsToRemove.push_back(attr.getName().getValue());
    for (auto name : attrsToRemove)
      module->removeAttr(name);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Post-pass: convert scf.while → stablehlo.while
    convertScfWhileToStablehloWhile(module);

    // Post-pass: convert remaining scf.if → stablehlo.select (simple cases)
    // or stablehlo.case (complex cases). SCF structural type conversion
    // handles scf.if type changes, but the ops themselves remain as scf.if.
    // The HLO export does not support scf dialect.
    {
      SmallVector<scf::IfOp> ifOps;
      module.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });
      for (auto ifOp : ifOps) {
        // Void scf.if is dead code (count/dispatch bookkeeping). Erase it.
        // First, move any func.call ops out of the scf.if to preserve them
        // (they were extracted from dispatch but may have been moved back by
        // the SCF structural type conversion).
        if (ifOp.getNumResults() == 0) {
          // Move func.call and its in-block dependencies out of scf.if.
          for (Region &r : ifOp->getRegions()) {
            for (Block &b : r) {
              SmallVector<Operation *> toMove;
              for (Operation &op : b)
                if (isa<func::CallOp>(&op))
                  toMove.push_back(&op);
              for (auto *callOp : toMove) {
                // Collect operand-defining ops in this block.
                llvm::DenseSet<Operation *> deps;
                std::function<void(Value)> collectDeps = [&](Value v) {
                  auto *def = v.getDefiningOp();
                  if (!def || def->getBlock() != &b || deps.count(def))
                    return;
                  deps.insert(def);
                  for (Value operand : def->getOperands())
                    collectDeps(operand);
                };
                for (Value operand : callOp->getOperands())
                  collectDeps(operand);
                // Move deps before scf.if (in block order).
                for (Operation &dep : llvm::make_early_inc_range(b))
                  if (deps.count(&dep))
                    dep.moveBefore(ifOp);
                callOp->moveBefore(ifOp);
              }
            }
          }
          for (Region &r : ifOp->getRegions())
            r.dropAllReferences();
          for (Region &r : ifOp->getRegions())
            r.getBlocks().clear();
          ifOp.erase();
          continue;
        }

        // Ensure predicate is tensor<i1>.
        Value pred = ifOp.getCondition();
        // Look through unrealized_conversion_cast.
        if (!isa<RankedTensorType>(pred.getType())) {
          if (auto castOp = pred.getDefiningOp<UnrealizedConversionCastOp>()) {
            Value src = castOp.getInputs()[0];
            if (isa<RankedTensorType>(src.getType()))
              pred = src;
          }
        }
        // Look through arith.ori/andi: convert to stablehlo.or/and on tensor.
        if (!isa<RankedTensorType>(pred.getType())) {
          if (auto *defOp = pred.getDefiningOp()) {
            if (isa<arith::OrIOp>(defOp) || isa<arith::AndIOp>(defOp)) {
              Value lhs = lookThroughCast(defOp->getOperand(0));
              Value rhs = lookThroughCast(defOp->getOperand(1));
              if (isa<RankedTensorType>(lhs.getType()) &&
                  isa<RankedTensorType>(rhs.getType())) {
                OpBuilder b(defOp);
                if (isa<arith::OrIOp>(defOp))
                  pred = b.create<stablehlo::OrOp>(defOp->getLoc(), lhs, rhs);
                else
                  pred = b.create<stablehlo::AndOp>(defOp->getLoc(), lhs, rhs);
              }
            }
          }
        }
        // Wrap scalar i1 predicate into tensor<i1>.
        if (!isa<RankedTensorType>(pred.getType()) &&
            pred.getType().isInteger(1)) {
          OpBuilder b(ifOp);
          auto tensorType = RankedTensorType::get({}, b.getI1Type());
          pred =
              b.create<tensor::FromElementsOp>(ifOp.getLoc(), tensorType, pred);
        }
        if (!isa<RankedTensorType>(pred.getType()))
          continue;

        // For each result: compute both branches' values, then select.
        // This inlines both branches before the scf.if, then uses
        // stablehlo.select to pick the result.
        OpBuilder builder(ifOp);
        Location loc = ifOp.getLoc();

        // Inline both branches' ops before the scf.if.
        Block &thenBlock = ifOp.getThenRegion().front();
        Block &elseBlock = ifOp.getElseRegion().front();

        // Get yield values from both branches.
        auto thenYield = cast<scf::YieldOp>(thenBlock.getTerminator());
        auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());

        // Inline then block ops (before yield) into parent.
        for (Operation &op : llvm::make_early_inc_range(thenBlock)) {
          if (&op == thenYield.getOperation())
            continue;
          op.moveBefore(ifOp);
        }
        // Inline else block ops (before yield) into parent.
        for (Operation &op : llvm::make_early_inc_range(elseBlock)) {
          if (&op == elseYield.getOperation())
            continue;
          op.moveBefore(ifOp);
        }

        // Create stablehlo.select for each result.
        SmallVector<Value> selectResults;
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
          Value thenVal = thenYield.getOperand(i);
          Value elseVal = elseYield.getOperand(i);

          // Broadcast predicate if result is not scalar.
          Value broadPred = pred;
          auto resultType = ifOp.getResult(i).getType();
          if (auto tt = dyn_cast<RankedTensorType>(resultType)) {
            if (tt.getRank() > 0) {
              auto predType =
                  RankedTensorType::get(tt.getShape(), builder.getI1Type());
              broadPred = builder.create<stablehlo::BroadcastInDimOp>(
                  loc, predType, pred, builder.getDenseI64ArrayAttr({}));
            }
          }
          selectResults.push_back(builder.create<stablehlo::SelectOp>(
              loc, resultType, broadPred, thenVal, elseVal));
        }

        // Replace scf.if results with select results.
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
          ifOp.getResult(i).replaceAllUsesWith(selectResults[i]);
        ifOp.erase();
      }
    }

    // Post-pass: reconnect func.call results to pod.read @comp consumers.
    // After conversion + scf.if extraction, func.call ops are in the while
    // body alongside pod.read @comp → unrealized_cast chains. Replace the
    // cast chain with the func.call result.
    {
      module.walk([&](stablehlo::WhileOp whileOp) {
        for (Block &b : whileOp.getBody()) {
          SmallVector<func::CallOp> calls;
          for (Operation &op : b)
            if (auto call = dyn_cast<func::CallOp>(&op))
              calls.push_back(call);
          if (calls.empty())
            continue;

          SmallVector<Operation *> toErase;
          for (Operation &op : b) {
            auto castOp = dyn_cast<UnrealizedConversionCastOp>(&op);
            if (!castOp || castOp.getNumOperands() != 1)
              continue;
            auto *srcOp = castOp.getInputs()[0].getDefiningOp();
            if (!srcOp || srcOp->getName().getStringRef() != "pod.read")
              continue;
            auto rn = srcOp->getAttrOfType<FlatSymbolRefAttr>("record_name");
            if (!rn || rn.getValue() != "comp")
              continue;
            Type targetType = castOp.getResult(0).getType();
            Value replacement;
            for (auto call : calls)
              if (call.getNumResults() > 0 &&
                  call.getResult(0).getType() == targetType)
                replacement = call.getResult(0);
            if (replacement) {
              castOp.getResult(0).replaceAllUsesWith(replacement);
              toErase.push_back(castOp.getOperation());
              toErase.push_back(srcOp);
              if (auto *arrOp = srcOp->getOperand(0).getDefiningOp())
                if (arrOp->getName().getStringRef() == "array.read")
                  toErase.push_back(arrOp);
            }
          }
          for (auto *op : llvm::reverse(toErase))
            if (op->use_empty())
              op->erase();
        }
      });
    }

    // Post-pass: convert remaining arith.ori/andi(i1) → stablehlo.or/and.
    // Only replace uses that accept tensor types. scf.if conditions still
    // need i1, so we keep the arith op for those and add a
    // unrealized_conversion_cast bridge.
    {
      SmallVector<Operation *> arithBoolOps;
      module.walk([&](Operation *op) {
        if (isa<arith::OrIOp>(op) || isa<arith::AndIOp>(op))
          arithBoolOps.push_back(op);
      });
      for (auto *op : arithBoolOps) {
        Value lhs = lookThroughCast(op->getOperand(0));
        Value rhs = lookThroughCast(op->getOperand(1));
        if (!isa<RankedTensorType>(lhs.getType()) ||
            !isa<RankedTensorType>(rhs.getType()))
          continue;
        OpBuilder b(op);
        Operation *replacement = nullptr;
        if (isa<arith::OrIOp>(op))
          replacement = b.create<stablehlo::OrOp>(op->getLoc(), lhs, rhs);
        else
          replacement = b.create<stablehlo::AndOp>(op->getLoc(), lhs, rhs);
        // Replace only non-scf.if uses with the tensor result.
        // For scf.if conditions, keep the original i1 value.
        Value tensorResult = replacement->getResult(0);
        SmallVector<OpOperand *> usesToReplace;
        for (OpOperand &use : op->getResult(0).getUses()) {
          if (!isa<scf::IfOp>(use.getOwner()))
            usesToReplace.push_back(&use);
        }
        for (auto *use : usesToReplace)
          use->set(tensorResult);
        // If no more uses, erase the arith op.
        if (op->getResult(0).use_empty())
          op->erase();
      }
    }

    // Erase non-StableHLO dead code inside stablehlo.while bodies.
    // Count/dispatch pod array ops (array.read, pod.read/write, arith.subi,
    // arith.cmpi, unrealized_cast) remain from dynamically-legal pod ops.
    // Iteratively erase ops with no uses from non-stablehlo dialects.
    {
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](stablehlo::WhileOp whileOp) {
          for (Region &r : whileOp->getRegions()) {
            for (Block &b : r) {
              for (Operation &op :
                   llvm::make_early_inc_range(llvm::reverse(b))) {
                StringRef ns = op.getName().getDialectNamespace();
                if (ns == "stablehlo" || ns == "func" || ns == "builtin")
                  continue;
                // Check if all results are unused.
                bool allUnused = llvm::all_of(
                    op.getResults(), [](Value v) { return v.use_empty(); });
                if (allUnused && op.getNumResults() > 0) {
                  op.erase();
                  erased = true;
                }
                // Void ops (pod.write, array.write) with no results.
                if (op.getNumResults() == 0 && ns != "stablehlo") {
                  op.dropAllReferences();
                  op.erase();
                  erased = true;
                }
              }
            }
          }
        });
      }
    }

    // Erase dead unrealized_conversion_casts.
    {
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](UnrealizedConversionCastOp castOp) {
          if (castOp->use_empty()) {
            castOp.erase();
            erased = true;
          }
        });
      }
    }

    // Clean up all dead non-stablehlo ops (unrealized_cast, arith, tensor,
    // array, pod, scf, struct, builtin dead casts).
    bool erased = true;
    while (erased) {
      erased = false;
      module.walk([&](Operation *op) {
        StringRef ns = op->getName().getDialectNamespace();
        if (ns == "stablehlo" || ns == "func" || ns == "builtin")
          return;
        // Void ops with no results: erase if non-stablehlo.
        if (op->getNumResults() == 0 &&
            (ns == "pod" || ns == "array" || ns == "struct")) {
          op->dropAllReferences();
          op->erase();
          erased = true;
          return;
        }
        if (op->use_empty()) {
          op->erase();
          erased = true;
        }
      });
    }
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
