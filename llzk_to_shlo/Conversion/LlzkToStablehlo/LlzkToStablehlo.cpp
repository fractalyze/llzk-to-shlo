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
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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
      StringRef ns = operand.getType().getDialect().getNamespace();
      if (ns == "array")
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
  llvm::DenseMap<Value, Value> latestArraySSA;
  for (auto [idx, blockArg] : llvm::enumerate(arrayBlockArgs))
    latestArraySSA[blockArg] = blockArg;

  for (Operation &op : llvm::make_early_inc_range(bodyBlock.getOperations())) {
    if (op.getName().getStringRef() != "array.write")
      continue;
    if (op.getNumOperands() < 3)
      continue;

    Value arr = op.getOperand(0);
    // Update to latest array value
    auto it = latestArraySSA.find(arr);
    if (it != latestArraySSA.end())
      arr = it->second;

    // Create array.write with a result (mutable → SSA)
    OpBuilder b(&op);
    OperationState state(op.getLoc(), "array.write");
    SmallVector<Value> writeOperands = {arr};
    for (unsigned i = 1; i < op.getNumOperands(); ++i)
      writeOperands.push_back(op.getOperand(i));
    state.addOperands(writeOperands);
    state.addTypes({arr.getType()});
    for (auto &attr : op.getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    Operation *newWrite = b.create(state);

    // Track latest array value
    latestArraySSA[op.getOperand(0)] = newWrite->getResult(0);
    // Also update the blockArg mapping
    for (auto &[k, v] : latestArraySSA) {
      if (v == arr && k != op.getOperand(0))
        v = newWrite->getResult(0);
    }

    // Replace uses of old array with new result (for subsequent reads)
    for (auto &use : llvm::make_early_inc_range(arr.getUses())) {
      if (use.getOwner() != newWrite &&
          use.getOwner()->getBlock() == &bodyBlock)
        use.set(newWrite->getResult(0));
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
      return;

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

/// Fix scf.if result types to match their yield values after dialect
/// conversion. The body ops may have been type-converted but scf.if result
/// types stay as the original LLZK types. This rebuilds the scf.if with correct
/// types.
/// Convert scf.while → stablehlo.while at LLZK level (before type conversion).
/// This ensures the while body ops are visible to dialect conversion for type
/// conversion. Only changes terminators: scf.condition → stablehlo.return,
/// scf.yield → stablehlo.return.
void convertScfWhileToStablehloWhile(ModuleOp module) {
  SmallVector<scf::WhileOp> whileOps;
  module.walk([&](scf::WhileOp op) { whileOps.push_back(op); });
  for (auto whileOp : whileOps) {
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
      // stablehlo.while expects tensor<i1> predicate
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
    // SCF structural type conversion is added after patterns are created
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
    // SCF structural type conversion: automatically converts scf.while/if/for
    // result types and block argument types using the type converter.
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    context->loadDialect<func::FuncDialect>();

    // Pre-passes: transform LLZK IR before dialect conversion
    //
    // Erase constrain functions early: drop all references in their regions
    // first to break cycles, then erase. This prevents complex constrain
    // function bodies (with scf.while, arrays, etc.) from interfering
    // with type conversion of the compute function.
    {
      SmallVector<Operation *> constrainFuncs;
      module.walk([&](Operation *op) {
        if (op->getName().getStringRef() != "function.def")
          return;
        auto sn = op->getAttrOfType<StringAttr>("sym_name");
        if (sn && sn.getValue() == "constrain")
          constrainFuncs.push_back(op);
      });
      for (auto *op : constrainFuncs) {
        // Drop all references in nested regions to avoid dangling pointers
        for (Region &region : op->getRegions())
          region.dropAllReferences();
        op->erase();
      }
    }

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
    // (scf.if is handled by populateSCFStructuralTypeConversions above)
    convertScfWhileToStablehloWhile(module);
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
