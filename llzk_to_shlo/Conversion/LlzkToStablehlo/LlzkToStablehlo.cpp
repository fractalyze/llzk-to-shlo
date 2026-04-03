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
    // NOLINTNEXTLINE(whitespace/line_length)
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
      // Skip values defined anywhere inside this while — body block args,
      // condition block args, nested region values (e.g. inner while
      // block args) are all internal and must not be promoted to carry.
      auto *parentOp = operand.getParentBlock()->getParentOp();
      if (parentOp == whileOp.getOperation() ||
          whileOp->isProperAncestor(parentOp))
        continue;
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

/// Convert void array.write and array.insert ops inside while bodies to
/// SSA form. Tracks all array values (including extract results) so that
/// extract → write → insert chains are properly rewired.
void convertWhileBodyArgsToSSA(ModuleOp module) {
  SmallVector<scf::WhileOp> whileOps;
  module.walk([&](scf::WhileOp op) { whileOps.push_back(op); });

  for (auto whileOp : whileOps) {
    Block &body = whileOp.getAfter().front();

    auto isTrackedArray = [](Type ty) {
      if (ty.getDialect().getNamespace() != "array")
        return false;
      if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
        if (arrTy.getElementType().getDialect().getNamespace() == "pod")
          return false;
      return true;
    };

    // Seed tracking with array-typed body args
    llvm::DenseMap<Value, Value> latestSSA;
    for (auto arg : body.getArguments()) {
      if (isTrackedArray(arg.getType()))
        latestSSA[arg] = arg;
    }
    if (latestSSA.empty())
      continue;

    bool changed = false;
    for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
      StringRef name = op.getName().getStringRef();

      // Track scf.while: update latest to the while result, but do NOT
      // rewire the while's init operands (they must reference pre-while
      // values).
      if (name == "scf.while") {
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          if (i < op.getNumOperands()) {
            Value init = op.getOperand(i);
            auto it = latestSSA.find(init);
            if (it != latestSSA.end()) {
              it->second = op.getResult(i);
              changed = true;
            }
          }
        }
        continue;
      }

      // Rewire tracked operands to latest SSA values (skip scf.while above)
      for (auto &operand : op.getOpOperands()) {
        auto it = latestSSA.find(operand.get());
        if (it != latestSSA.end() && it->second != operand.get())
          operand.set(it->second);
      }

      // Track array.extract results so subsequent writes are connected
      if (name == "array.extract" && op.getNumResults() > 0 &&
          isTrackedArray(op.getResult(0).getType())) {
        latestSSA[op.getResult(0)] = op.getResult(0);
        continue;
      }

      // Convert void array.write/insert to SSA (add result type)
      if ((name == "array.write" || name == "array.insert") &&
          op.getNumResults() == 0 && op.getNumOperands() >= 3) {
        Value arr = op.getOperand(0);
        OpBuilder b(&op);
        OperationState state(op.getLoc(), name);
        state.addOperands(op.getOperands());
        state.addTypes({arr.getType()});
        for (auto &attr : op.getAttrs())
          state.addAttribute(attr.getName(), attr.getValue());
        Operation *newOp = b.create(state);
        for (auto &[key, latest] : latestSSA) {
          if (latest == arr) {
            latest = newOp->getResult(0);
            changed = true;
          }
        }
        op.erase();
        continue;
      }
    }

    if (!changed)
      continue;

    // Update scf.yield to use latest SSA values for carry args
    auto yieldOp = cast<scf::YieldOp>(body.getTerminator());
    SmallVector<Value> newYieldArgs;
    for (Value val : yieldOp.getOperands()) {
      auto it = latestSSA.find(val);
      newYieldArgs.push_back(it != latestSSA.end() ? it->second : val);
    }
    OpBuilder yb(yieldOp);
    yb.create<scf::YieldOp>(yieldOp.getLoc(), newYieldArgs);
    yieldOp.erase();
  }
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
      // Replace uses AFTER the while in the SAME block only.
      // Do NOT replace uses in sibling or outer blocks — those must
      // reference the parent while's result, not this inner while's.
      Block *whileBlock = newWhile->getBlock();
      for (auto &use : llvm::make_early_inc_range(extArr.getUses())) {
        Operation *user = use.getOwner();
        // Only replace in the same block as the while
        if (user->getBlock() != whileBlock)
          continue;
        // Skip the while op itself (init operand)
        if (user == newWhile.getOperation())
          continue;
        // Skip ops defined BEFORE the while
        if (user->isBeforeInBlock(newWhile))
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

        // Strategy 4: wrap bare scalar types (i1, index) in tensor<>.
        if (!tensorType && init.getType().isIntOrIndexOrFloat())
          tensorType = RankedTensorType::get({}, init.getType());

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

/// Vectorize stablehlo.while loops whose iterations are independent.
/// Pattern: while(i < N) { out[i] = f(in[i]); i++ } → out = f(in)
/// Detects element-wise computation chains (dynamic_slice → compute →
/// dynamic_update_slice) and replaces with vectorized element-wise ops.
void vectorizeIndependentWhileLoops(ModuleOp module) {
  SmallVector<stablehlo::WhileOp> whileOps;
  module.walk([&](stablehlo::WhileOp op) { whileOps.push_back(op); });

  for (auto whileOp : whileOps) {
    // --- 1. Check condition: compare(counter, constant_N) ---
    Block &condBlock = whileOp.getCond().front();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(condBlock.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      continue;
    auto cmpOp = returnOp.getOperand(0).getDefiningOp<stablehlo::CompareOp>();
    if (!cmpOp ||
        cmpOp.getComparisonDirection() != stablehlo::ComparisonDirection::LT)
      continue;

    // --- 2. Check body: find the element-wise computation pattern ---
    Block &bodyBlock = whileOp.getBody().front();

    // Find dynamic_slice ops that read from outer (func) args.
    SmallVector<stablehlo::DynamicSliceOp> sliceOps;
    SmallVector<stablehlo::DynamicUpdateSliceOp> updateOps;
    SmallVector<Operation *> computeOps;

    for (Operation &op : bodyBlock) {
      if (auto slice = dyn_cast<stablehlo::DynamicSliceOp>(&op)) {
        // Check: slicing from an outer value (not a body arg carry)
        Value src = slice.getOperand();
        if (src.getParentBlock() != &bodyBlock)
          sliceOps.push_back(slice);
      } else if (auto update = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op)) {
        updateOps.push_back(update);
      }
    }

    if (sliceOps.empty() || updateOps.empty())
      continue;

    // --- 3. Check independence: accumulator carry is only written, not read
    // --- Find which body arg is the accumulator (array carry).
    int accArgIdx = -1;
    for (auto [i, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      auto tt = dyn_cast<RankedTensorType>(arg.getType());
      if (!tt || tt.getRank() == 0)
        continue;
      // Check: only used by dynamic_update_slice as the first operand.
      bool onlyUpdateUse = true;
      for (auto *user : arg.getUsers()) {
        if (!isa<stablehlo::DynamicUpdateSliceOp>(user) &&
            !isa<stablehlo::ReturnOp>(user)) {
          onlyUpdateUse = false;
          break;
        }
      }
      if (onlyUpdateUse) {
        accArgIdx = i;
        break;
      }
    }
    if (accArgIdx < 0)
      continue;

    // --- 4. Extract the element-wise computation chain ---
    // Walk from each update_slice backward to find: reshape → compute → reshape
    // → dynamic_slice. Collect the element-wise ops.
    if (updateOps.size() != 1)
      continue; // PoC: single output array
    auto updateOp = updateOps[0];
    Value updateVal = updateOp.getUpdate(); // tensor<1x!pf>

    // Trace backward: reshape → element_op → reshape → dynamic_slice
    auto reshapeToUpdate = updateVal.getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeToUpdate)
      continue;
    Value scalarResult = reshapeToUpdate.getOperand(); // tensor<!pf>

    // Find the element-wise op (multiply, add, etc.)
    Operation *elemOp = scalarResult.getDefiningOp();
    if (!elemOp || elemOp->getNumOperands() < 1 || elemOp->getNumResults() != 1)
      continue;
    if (elemOp->getName().getDialectNamespace() != "stablehlo")
      continue;

    // Trace each operand of the element-wise op back to a dynamic_slice
    SmallVector<Value> outerArrays; // the full input arrays
    bool allFromSlice = true;
    for (Value operand : elemOp->getOperands()) {
      auto reshapeFromSlice = operand.getDefiningOp<stablehlo::ReshapeOp>();
      if (!reshapeFromSlice) {
        allFromSlice = false;
        break;
      }
      auto slice = reshapeFromSlice.getOperand()
                       .getDefiningOp<stablehlo::DynamicSliceOp>();
      if (!slice) {
        allFromSlice = false;
        break;
      }
      outerArrays.push_back(slice.getOperand());
    }
    if (!allFromSlice || outerArrays.empty())
      continue;

    // --- 5. Verify all source arrays have the same shape ---
    auto accType =
        dyn_cast<RankedTensorType>(bodyBlock.getArgument(accArgIdx).getType());
    if (!accType)
      continue;
    bool shapesMatch = true;
    for (Value arr : outerArrays) {
      auto arrType = dyn_cast<RankedTensorType>(arr.getType());
      if (!arrType || arrType.getShape() != accType.getShape()) {
        shapesMatch = false;
        break;
      }
    }
    if (!shapesMatch)
      continue;

    // --- 6. Build vectorized replacement ---
    OpBuilder builder(whileOp);
    Location loc = whileOp.getLoc();

    // Create the vectorized element-wise op on full arrays.
    OperationState state(loc, elemOp->getName());
    for (Value arr : outerArrays)
      state.addOperands(arr);
    state.addTypes({accType});
    // Copy attributes (e.g., comparison_direction)
    for (auto attr : elemOp->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    Operation *vectorizedOp = builder.create(state);

    // Replace uses: the while result at accArgIdx → vectorized result
    whileOp.getResult(accArgIdx).replaceAllUsesWith(vectorizedOp->getResult(0));

    // Replace other while results (counter etc.) with their init values
    auto inits = whileOp.getOperands();
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      if (i != static_cast<unsigned>(accArgIdx) &&
          !whileOp.getResult(i).use_empty())
        whileOp.getResult(i).replaceAllUsesWith(inits[i]);
    }

    whileOp->erase();
  }

  // --- Phase 1.5 + 2: 2D vectorization (experimental, opt-in) ---
  // Enable with LLZK_VECTORIZE_2D=1 environment variable.
  // Known issue: correctness bug — outer while(j) loop may be incorrectly
  // erased, losing K iterations of the chain computation.
  if (!std::getenv("LLZK_VECTORIZE_2D"))
    return;

  // --- Phase 1.5: Vectorize while loops with 2D carry writing one column ---
  // Pattern: while(i < N) { acc[i, const_col] = f(a[i], b[i]); out[i] = acc[i,
  // const_col2]; i++ } Replace with vectorized element-wise ops + column write.
  SmallVector<stablehlo::WhileOp> whileOps2;
  module.walk([&](stablehlo::WhileOp op) { whileOps2.push_back(op); });

  for (auto whileOp : whileOps2) {
    // Skip if body contains nested while (Phase 2 handles those)
    bool hasNestedWhile = false;
    whileOp.getBody().walk([&](stablehlo::WhileOp) { hasNestedWhile = true; });
    if (hasNestedWhile)
      continue;

    Block &condBlock = whileOp.getCond().front();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(condBlock.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      continue;
    auto cmpOp = returnOp.getOperand(0).getDefiningOp<stablehlo::CompareOp>();
    if (!cmpOp ||
        cmpOp.getComparisonDirection() != stablehlo::ComparisonDirection::LT)
      continue;

    Block &bodyBlock = whileOp.getBody().front();

    // Find 2D carry (rank-2 tensor, used by DUS or DS)
    int acc2dIdx = -1;
    RankedTensorType acc2dType;
    for (auto [i, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      auto tt = dyn_cast<RankedTensorType>(arg.getType());
      if (!tt || tt.getRank() != 2)
        continue;
      bool hasSliceOrUpdate = false;
      for (auto *user : arg.getUsers()) {
        if (isa<stablehlo::DynamicUpdateSliceOp>(user) ||
            isa<stablehlo::DynamicSliceOp>(user))
          hasSliceOrUpdate = true;
      }
      if (hasSliceOrUpdate) {
        acc2dIdx = i;
        acc2dType = tt;
        break;
      }
    }
    if (acc2dIdx < 0)
      continue;

    int64_t N = acc2dType.getDimSize(0);

    // Find all slice → compute → DUS chains in the body
    SmallVector<stablehlo::DynamicUpdateSliceOp> updateOps;
    for (Operation &op : bodyBlock)
      if (auto u = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op))
        updateOps.push_back(u);

    // Check: all DUS write to 2D accumulator or 1D carry
    // Find element-wise ops that read from 1D outer arrays
    SmallVector<stablehlo::DynamicSliceOp> outerSlices;
    for (Operation &op : bodyBlock) {
      if (auto slice = dyn_cast<stablehlo::DynamicSliceOp>(&op)) {
        Value src = slice.getOperand();
        if (src.getParentBlock() != &bodyBlock)
          outerSlices.push_back(slice);
      }
    }
    // Try to find: slice(outer1D) → reshape → elemOp → reshape → DUS(acc2D)
    bool vectorized = false;
    OpBuilder builder(whileOp);
    Location loc = whileOp.getLoc();
    auto inits = whileOp.getOperands();
    Type colType = RankedTensorType::get({N}, acc2dType.getElementType());

    for (auto update : updateOps) {
      // Only handle updates to the 2D accumulator
      if (update.getOperand().getType() != acc2dType)
        continue;

      Value updateVal = update.getUpdate();
      auto reshapeUp = updateVal.getDefiningOp<stablehlo::ReshapeOp>();
      if (!reshapeUp)
        continue;
      Operation *elemOp = reshapeUp.getOperand().getDefiningOp();
      if (!elemOp || elemOp->getNumResults() != 1 ||
          elemOp->getName().getDialectNamespace() != "stablehlo")
        continue;

      // All operands of elemOp must come from 1D outer array slices
      SmallVector<Value> vecOperands;
      bool allOuter = true;
      for (Value operand : elemOp->getOperands()) {
        auto reshapeDown = operand.getDefiningOp<stablehlo::ReshapeOp>();
        if (!reshapeDown) {
          allOuter = false;
          break;
        }
        auto slice =
            reshapeDown.getOperand().getDefiningOp<stablehlo::DynamicSliceOp>();
        if (!slice) {
          allOuter = false;
          break;
        }
        Value src = slice.getOperand();
        auto srcType = dyn_cast<RankedTensorType>(src.getType());
        if (!srcType || srcType.getRank() != 1 || srcType.getDimSize(0) != N) {
          allOuter = false;
          break;
        }
        vecOperands.push_back(src);
      }
      if (!allOuter || vecOperands.empty())
        continue;

      // Build vectorized element-wise op
      OperationState vecState(loc, elemOp->getName());
      for (Value v : vecOperands)
        vecState.addOperands(v);
      vecState.addTypes({colType});
      for (auto attr : elemOp->getAttrs())
        vecState.addAttribute(attr.getName(), attr.getValue());
      Operation *vecOp = builder.create(vecState);

      // Write column into 2D acc: reshape to [N,1] then DUS at column idx
      auto reshapedCol = builder.create<stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({N, 1}, acc2dType.getElementType()),
          vecOp->getResult(0));
      // Column index: use the constant from the original DUS
      // (second start index)
      Value colIdx = update.getStartIndices()[1];
      // Clone colIdx if it's defined inside the body
      if (colIdx.getParentBlock() == &bodyBlock) {
        if (auto *defOp = colIdx.getDefiningOp())
          colIdx = builder.clone(*defOp)->getResult(0);
      }
      auto zeroIdx = createIndexConstant(builder, loc, 0);
      auto updatedAcc = builder.create<stablehlo::DynamicUpdateSliceOp>(
          loc, acc2dType, inits[acc2dIdx], reshapedCol,
          SmallVector<Value>{zeroIdx, colIdx});

      whileOp.getResult(acc2dIdx).replaceAllUsesWith(updatedAcc);
      vectorized = true;
    }

    // Also handle "copy column from 2D to 1D" pattern:
    // DS(acc2D[i, const]) → reshape → DUS(out1D[i])
    // Replace with: dynamic_slice acc2D[:, const] → reshape → out1D
    if (!vectorized) {
      // Find 1D output carry
      int out1dIdx = -1;
      RankedTensorType out1dType;
      for (auto [i, arg] : llvm::enumerate(bodyBlock.getArguments())) {
        if (i == static_cast<unsigned>(acc2dIdx))
          continue;
        auto tt = dyn_cast<RankedTensorType>(arg.getType());
        if (!tt || tt.getRank() != 1 || tt.getDimSize(0) != N)
          continue;
        out1dIdx = i;
        out1dType = tt;
        break;
      }

      if (out1dIdx >= 0) {
        // Find: DS(acc2D) → reshape → DUS(out1D)
        for (auto update : updateOps) {
          if (update.getOperand().getType() != out1dType)
            continue;
          Value updVal = update.getUpdate();
          auto reshUp = updVal.getDefiningOp<stablehlo::ReshapeOp>();
          if (!reshUp)
            continue;
          Value scalar = reshUp.getOperand();
          auto reshDown = scalar.getDefiningOp<stablehlo::ReshapeOp>();
          if (!reshDown)
            continue;
          auto slice =
              reshDown.getOperand().getDefiningOp<stablehlo::DynamicSliceOp>();
          if (!slice)
            continue;
          // Verify source is the 2D accumulator
          Value sliceSrc = slice.getOperand();
          if (auto ba = dyn_cast<BlockArgument>(sliceSrc))
            if (ba.getArgNumber() != static_cast<unsigned>(acc2dIdx))
              continue;

          // Get column index (constant in the body)
          Value colIdx = slice.getStartIndices()[1];
          if (auto *defOp = colIdx.getDefiningOp()) {
            // Clone constant to outer scope
            IRMapping cloneMap;
            colIdx = builder.clone(*defOp, cloneMap)->getResult(0);
          }

          // Build: dynamic_slice acc[:, col], sizes=[N, 1] → reshape → out
          auto zeroIdx = createIndexConstant(builder, loc, 0);
          auto colSlice = builder.create<stablehlo::DynamicSliceOp>(
              loc, RankedTensorType::get({N, 1}, acc2dType.getElementType()),
              inits[acc2dIdx], SmallVector<Value>{zeroIdx, colIdx},
              builder.getDenseI64ArrayAttr({N, 1}));
          Value result =
              builder.create<stablehlo::ReshapeOp>(loc, colType, colSlice);

          whileOp.getResult(out1dIdx).replaceAllUsesWith(result);
          vectorized = true;
          break;
        }
      }
    }

    if (!vectorized)
      continue;

    // Replace remaining results with init values
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      if (!whileOp.getResult(i).use_empty())
        whileOp.getResult(i).replaceAllUsesWith(inits[i]);
    }
    whileOp->erase();
  }

  // --- Phase 2: Vectorize inner while loops inside outer while loops ---
  // Pattern: outer while(j) { inner while(i) { chain[i,j] = f(chain[i,j-1],
  // a[i]) } } The inner while iterates over the first dimension (i) of a 2D
  // accumulator while the outer while iterates over the second dimension (j).
  // Replace inner while with column-wise vectorized ops.
  SmallVector<stablehlo::WhileOp> outerWhileOps;
  module.walk([&](stablehlo::WhileOp op) { outerWhileOps.push_back(op); });

  for (auto outerWhile : outerWhileOps) {
    Block &outerBody = outerWhile.getBody().front();

    // Find the single inner while in the outer body.
    stablehlo::WhileOp innerWhile;
    int innerCount = 0;
    for (Operation &op : outerBody) {
      if (auto w = dyn_cast<stablehlo::WhileOp>(&op)) {
        innerWhile = w;
        innerCount++;
      }
    }
    if (innerCount != 1 || !innerWhile)
      continue;

    Block &innerBody = innerWhile.getBody().front();

    // Find 2D accumulator carry: tensor<N x M> that is sliced and updated.
    int accArgIdx = -1;
    RankedTensorType accType;
    for (auto [i, arg] : llvm::enumerate(innerBody.getArguments())) {
      auto tt = dyn_cast<RankedTensorType>(arg.getType());
      if (!tt || tt.getRank() != 2)
        continue;
      // Check: used by dynamic_slice AND dynamic_update_slice
      bool hasSlice = false, hasUpdate = false;
      for (auto *user : arg.getUsers()) {
        if (isa<stablehlo::DynamicSliceOp>(user))
          hasSlice = true;
        if (isa<stablehlo::DynamicUpdateSliceOp>(user))
          hasUpdate = true;
      }
      if (hasSlice && hasUpdate) {
        accArgIdx = i;
        accType = tt;
        break;
      }
    }
    if (accArgIdx < 0)
      continue;

    int64_t N = accType.getDimSize(0); // rows = independent elements
    int64_t M = accType.getDimSize(1); // cols = sequential steps

    // Find the element-wise op and its source patterns.
    SmallVector<stablehlo::DynamicUpdateSliceOp> innerUpdates;
    for (Operation &op : innerBody)
      if (auto u = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op))
        innerUpdates.push_back(u);
    if (innerUpdates.size() != 1)
      continue;
    auto innerUpdate = innerUpdates[0];

    // Trace: update ← reshape ← elemOp ← reshape ← dynamic_slice
    Value updateVal = innerUpdate.getUpdate();
    auto reshapeToUpdate = updateVal.getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeToUpdate)
      continue;
    Operation *elemOp = reshapeToUpdate.getOperand().getDefiningOp();
    if (!elemOp || elemOp->getNumResults() != 1 ||
        elemOp->getName().getDialectNamespace() != "stablehlo")
      continue;

    // Trace operands: each should come from a dynamic_slice
    SmallVector<stablehlo::DynamicSliceOp> sourceSlices;
    bool allFromSlice = true;
    for (Value operand : elemOp->getOperands()) {
      auto reshapeFrom = operand.getDefiningOp<stablehlo::ReshapeOp>();
      if (!reshapeFrom) {
        allFromSlice = false;
        break;
      }
      auto slice =
          reshapeFrom.getOperand().getDefiningOp<stablehlo::DynamicSliceOp>();
      if (!slice) {
        allFromSlice = false;
        break;
      }
      sourceSlices.push_back(slice);
    }
    if (!allFromSlice || sourceSlices.empty())
      continue;

    // --- 1D carry optimization for witness generation ---
    // Replace outer while(j) { inner while(i) { acc[i,j]=f(acc[i,j-1],a[i]) }}
    // with:   col = init_col; while(j) { col = f(col, a) }
    // This avoids 2D tensor carry overhead (N*M per iteration → N only).
    // Safe because witness generation only needs the final column.

    // Identify which operand of elemOp reads from the 2D accumulator
    // (self-referential: the "previous column" read) vs external 1D arrays.
    auto innerInits = innerWhile.getOperands();
    bool hasSelfRef = false;
    int selfRefIdx = -1;
    SmallVector<Value> externalArrays;
    SmallVector<int> externalOpIdx;

    for (auto [opIdx, slice] : llvm::enumerate(sourceSlices)) {
      Value src = slice.getOperand();
      if (auto ba = dyn_cast<BlockArgument>(src))
        if (ba.getOwner() == &innerBody)
          src = innerInits[ba.getArgNumber()];
      auto srcType = dyn_cast<RankedTensorType>(src.getType());
      if (!srcType) {
        allFromSlice = false;
        break;
      }
      if (srcType.getRank() == 2 && srcType == accType) {
        hasSelfRef = true;
        selfRefIdx = opIdx;
      } else if (srcType.getRank() == 1 && srcType.getDimSize(0) == N) {
        externalArrays.push_back(src);
        externalOpIdx.push_back(opIdx);
      } else {
        allFromSlice = false;
        break;
      }
    }
    if (!allFromSlice || !hasSelfRef)
      continue;

    Location loc = outerWhile.getLoc();
    Type colType = RankedTensorType::get({N}, accType.getElementType());

    // --- Build initial column (column 0) before the outer while ---
    // Find While 0 (the loop that fills chain[:,0] = a * b).
    // For now, use the first column of the 2D init as initial value.
    // The first while (While 0) already computed this.
    OpBuilder preBuilder(outerWhile);

    // Extract column 0 from outer while's init accumulator
    Value outerAccInit = outerWhile.getOperands()[1]; // 2D acc init
    auto zeroC = createIndexConstant(preBuilder, loc, 0);
    auto initColSlice = preBuilder.create<stablehlo::DynamicSliceOp>(
        loc, RankedTensorType::get({N, 1}, accType.getElementType()),
        outerAccInit, SmallVector<Value>{zeroC, zeroC},
        preBuilder.getDenseI64ArrayAttr({N, 1}));
    Value initCol =
        preBuilder.create<stablehlo::ReshapeOp>(loc, colType, initColSlice);

    // --- Build new outer while with 1D carry ---
    // New inits: [counter, col(tensor<N>)]
    Value counterInit = outerWhile.getOperands()[0];
    SmallVector<Value> newInits = {counterInit, initCol};
    SmallVector<Type> newTypes = {counterInit.getType(), colType};

    auto newOuterWhile =
        preBuilder.create<stablehlo::WhileOp>(loc, newTypes, newInits);

    // Condition: clone from original, remap block args
    {
      Region &cond = newOuterWhile.getCond();
      Block *condBlock = new Block();
      cond.push_back(condBlock);
      condBlock->addArgument(counterInit.getType(), loc);
      condBlock->addArgument(colType, loc);
      Block &origCond = outerWhile.getCond().front();
      IRMapping mapping;
      mapping.map(origCond.getArgument(0), condBlock->getArgument(0));
      if (origCond.getNumArguments() > 1)
        mapping.map(origCond.getArgument(1), condBlock->getArgument(1));
      OpBuilder cb(condBlock, condBlock->end());
      for (Operation &op : origCond)
        cb.clone(op, mapping);
    }

    // Body: vectorized element-wise op
    {
      Region &body = newOuterWhile.getBody();
      Block *bodyBlock = new Block();
      body.push_back(bodyBlock);
      Value jArg = bodyBlock->addArgument(counterInit.getType(), loc);
      Value colArg = bodyBlock->addArgument(colType, loc);
      OpBuilder bb(bodyBlock, bodyBlock->end());

      // Build vectorized operands in correct order
      SmallVector<Value> vecOperands(elemOp->getNumOperands());
      vecOperands[selfRefIdx] = colArg; // previous column
      for (auto [i, extIdx] : llvm::enumerate(externalOpIdx))
        vecOperands[extIdx] = externalArrays[i];

      // Create vectorized element-wise op
      OperationState vecState(loc, elemOp->getName());
      for (Value v : vecOperands)
        vecState.addOperands(v);
      vecState.addTypes({colType});
      for (auto attr : elemOp->getAttrs())
        vecState.addAttribute(attr.getName(), attr.getValue());
      Operation *vecOp = bb.create(vecState);

      // Increment counter: clone from original outer body
      // Find add(counter, 1) and constant(1) in original outer body
      IRMapping bodyMapping;
      bodyMapping.map(outerBody.getArgument(0), jArg);
      Value nextJ;
      auto origBodyRet = cast<stablehlo::ReturnOp>(outerBody.getTerminator());
      // The counter increment is the first return operand's def chain
      Value origNextJ = origBodyRet.getOperand(0);
      if (auto *defOp = origNextJ.getDefiningOp()) {
        // Clone the add and its constant operand
        for (Value operand : defOp->getOperands()) {
          if (auto *constOp = operand.getDefiningOp()) {
            if (!bodyMapping.contains(operand))
              bodyMapping.map(operand,
                              bb.clone(*constOp, bodyMapping)->getResult(0));
          }
        }
        nextJ = bb.clone(*defOp, bodyMapping)->getResult(0);
      }

      bb.create<stablehlo::ReturnOp>(loc,
                                     ValueRange{nextJ, vecOp->getResult(0)});
    }

    // Replace uses: outer while result #1 (2D acc) users need the final
    // column. For struct.writem of the 2D chain, replace with a zero tensor
    // (witness generation doesn't need intermediate values).
    // For the output array (1D), the final column is the result.
    Value finalCol = newOuterWhile.getResult(1);

    // Replace all uses of the original outer while results
    outerWhile.getResult(0).replaceAllUsesWith(newOuterWhile.getResult(0));

    // The 2D accumulator result — replace with a dummy or find uses
    if (!outerWhile.getResult(1).use_empty()) {
      // Build a 2D tensor with the final column in the last position
      // (for struct.writem that stores the chain). For witness gen,
      // we can just use a zero 2D tensor since constrain is separate.
      // Use the original 2D init (already a zero constant)
      outerWhile.getResult(1).replaceAllUsesWith(outerAccInit);
    }

    outerWhile->erase();
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
// Pre-pass: eliminate input pods ($inputs struct members)
// ===----------------------------------------------------------------------===

/// Remove `$inputs` pod fields from compute functions. These pods store
/// sub-component input parameters for the constrain function but are not
/// needed for witness generation. After SimplifySubComponents extracts
/// function calls, the input pods become dead code.
///
/// Removes: struct.member @xxx$inputs, struct.writem @xxx$inputs,
///          pod.new/pod.write that feed into $inputs fields.
void eliminateInputPods(ModuleOp module) {
  SmallVector<Operation *> toErase;

  module->walk([&](Operation *op) {
    // Erase struct.member with $inputs suffix.
    if (op->getName().getStringRef() == "struct.member") {
      auto sym = op->getAttrOfType<StringAttr>("sym_name");
      if (sym && sym.getValue().ends_with("$inputs"))
        toErase.push_back(op);
      return;
    }

    // Erase struct.writem to $inputs fields.
    if (op->getName().getStringRef() == "struct.writem") {
      auto member = op->getAttrOfType<FlatSymbolRefAttr>("member_name");
      if (member && member.getValue().ends_with("$inputs"))
        toErase.push_back(op);
      return;
    }
  });

  for (auto *op : toErase)
    op->erase();

  // Now erase dead pod.new/pod.write ops that have no remaining uses.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> deadOps;
    module->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "pod.new" && name != "pod.write")
        return;
      // pod.new: erase if all results are unused.
      if (name == "pod.new") {
        bool allDead = true;
        for (auto result : op->getResults()) {
          if (!result.use_empty()) {
            allDead = false;
            break;
          }
        }
        if (allDead)
          deadOps.push_back(op);
      }
      // pod.write: erase if its result (updated pod) is unused.
      if (name == "pod.write") {
        if (op->getNumResults() == 0 ||
            (op->getNumResults() > 0 && op->getResult(0).use_empty()))
          deadOps.push_back(op);
      }
    });
    for (auto *op : deadOps) {
      op->dropAllUses();
      op->erase();
      changed = true;
    }
  }

  // Also erase struct.readm of $inputs in constrain functions.
  SmallVector<Operation *> constrainReads;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "struct.readm") {
      auto member = op->getAttrOfType<FlatSymbolRefAttr>("member_name");
      if (member && member.getValue().ends_with("$inputs"))
        constrainReads.push_back(op);
    }
  });

  // Replace struct.readm @xxx$inputs results with zero/undef, then erase.
  for (auto *op : constrainReads) {
    // These are in constrain functions which get erased anyway.
    // Just drop uses (constrain body is cleared later).
    op->dropAllUses();
    op->erase();
  }
}

// ===----------------------------------------------------------------------===
// Pre-pass: inline single-field input pods to their field type
// ===----------------------------------------------------------------------===

/// Check if a pod type has no @count/@comp/@params fields (i.e., it's an
/// input pod, not a dispatch pod).
static bool isInputPodType(Type type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s.find("@count") == std::string::npos;
}

/// Replace input pods used as while carry with their flattened field types.
///
/// Pattern:
///   %pod = pod.new : <[@field: T]>
///   scf.while(%arg = %pod) : (!pod.type<[...]>) -> ... {
///     %val = pod.read %arg[@field]   // → just use %arg directly
///     pod.write %arg[@field] = %new  // → yield %new
///   }
///
/// For single-field pods, replaces the pod type with T everywhere.
/// For multi-field pods, creates a tuple of fields (not yet implemented).
void inlineInputPodCarries(ModuleOp module) {
  // Find all pod.new that are input pods (no @count/@comp/@params).
  SmallVector<Operation *> podNews;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "pod.new" && op->getNumResults() == 1 &&
        isInputPodType(op->getResult(0).getType()))
      podNews.push_back(op);
  });

  for (auto *podNew : podNews) {
    Value podVal = podNew->getResult(0);
    Type podType = podVal.getType();

    // Get the field names and types from the pod type string.
    // Parse "!pod.type<[@field: T]>" or "!pod.type<[@f1: T1, @f2: T2]>"
    std::string typeStr;
    {
      llvm::raw_string_ostream os(typeStr);
      podType.print(os);
    }

    // Only handle single-field pods for now.
    // Count '@' in the type (excluding nested types).
    size_t atCount = 0;
    for (char c : typeStr)
      if (c == '@')
        atCount++;
    if (atCount != 1)
      continue; // Multi-field pod — skip for now

    // Extract the field name from the pod type.
    auto atPos = typeStr.find('@');
    auto colonPos = typeStr.find(':', atPos);
    if (atPos == std::string::npos || colonPos == std::string::npos)
      continue;
    std::string fieldName = typeStr.substr(atPos + 1, colonPos - atPos - 1);

    // Find all pod.read and pod.write for this pod and its SSA aliases.
    // The pod flows through scf.while carry, so we need to handle block args.

    // Collect all Values that are this pod (including block args of while).
    SmallVector<Value> podValues;
    SmallVector<Value> worklist;
    DenseSet<Value> visited;
    worklist.push_back(podVal);

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;
      if (v.getType() != podType)
        continue;
      podValues.push_back(v);

      // Follow uses to find block args that receive this pod.
      for (auto &use : v.getUses()) {
        Operation *user = use.getOwner();
        // scf.yield passes values to while results and back to body args
        if (user->getName().getStringRef() == "scf.yield" ||
            user->getName().getStringRef() == "scf.condition") {
          // The corresponding while op's results and body args get this type.
          if (auto *whileOp = user->getParentOp()) {
            for (auto result : whileOp->getResults())
              if (result.getType() == podType)
                worklist.push_back(result);
            for (auto &region : whileOp->getRegions())
              for (auto arg : region.getArguments())
                if (arg.getType() == podType)
                  worklist.push_back(arg);
          }
        }
      }
    }

    // Find the inner type from pod.read operations.
    Type innerType;
    for (Value v : podValues) {
      for (auto &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.read" &&
            user->getNumResults() > 0) {
          innerType = user->getResult(0).getType();
          break;
        }
      }
      if (innerType)
        break;
    }
    if (!innerType)
      continue;

    // Now replace:
    // 1. Change all pod-typed Values to innerType
    // 2. Replace pod.read with identity (just use the value)
    // 3. Replace pod.write with its value operand
    // 4. Replace pod.new with the appropriate zero/init value

    // Step 1: Update types of all pod values.
    for (Value v : podValues)
      v.setType(innerType);

    // Step 2: Replace pod.read → forward the pod value directly.
    SmallVector<Operation *> toErase;
    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.read") {
          // pod.read %pod[@field] : T → replace result with %pod
          user->getResult(0).replaceAllUsesWith(v);
          toErase.push_back(user);
        }
      }
    }

    // Step 3: Replace pod.write → forward the written value.
    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.write") {
          // pod.write %pod[@field] = %val → replace pod result with %val
          // pod.write has no result in LLZK (it's a side-effecting op).
          // But the pod SSA value is reused after the write via scf.yield.
          // Actually in LLZK, pod.write modifies in place (no result).
          // The scf.yield just yields the same %arg. So we need to make
          // the scf.yield use %val instead of %arg.
          //
          // Find the value being written (second operand after the pod).
          if (user->getNumOperands() >= 2) {
            Value writtenVal = user->getOperand(1);
            // Replace all uses of the pod AFTER this write with writtenVal.
            // Since LLZK pod.write is in-place, the "updated" pod is the
            // same SSA value. We need to rewire the yield.
            // For now, just erase pod.write and let the array.write
            // that precedes it handle the update.
            toErase.push_back(user);
          }
        }
      }
    }

    // Step 4: Replace pod.new with a zero-initialized value of innerType.
    {
      OpBuilder builder(podNew);
      // Create: llzk.nondet : innerType (placeholder zero init)
      OperationState state(podNew->getLoc(), "llzk.nondet");
      state.addTypes({innerType});
      Operation *initOp = builder.create(state);
      podNew->getResult(0).replaceAllUsesWith(initOp->getResult(0));
      toErase.push_back(podNew);
    }

    for (auto *op : toErase)
      op->erase();
  }

  // Also update scf.while/scf.condition/scf.yield operand types that
  // were changed from pod to inner type. The block args were already
  // updated by setType, but the while op's result types need updating too.
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "scf.while") {
      // Update result types to match the body's yield types.
      auto &bodyRegion = op->getRegion(1); // body
      if (bodyRegion.empty())
        return;
      auto *terminator = bodyRegion.front().getTerminator();
      if (!terminator)
        return;
      for (auto [result, yielded] :
           llvm::zip(op->getResults(), terminator->getOperands())) {
        if (result.getType() != yielded.getType())
          result.setType(yielded.getType());
      }
    }
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
    // Pod ops: dynamically legal inside constrain functions, scf control
    // flow, or when they are input pods (no @count/@comp/@params — these
    // are dead carries after SimplifySubComponents and will be cleaned up
    // in post-passes).
    target.addDynamicallyLegalDialect(
        [](Operation *op) -> bool {
          // Input pods (no dispatch fields) are always legal — they survive
          // as dead code and get cleaned up after conversion.
          if (op->getName().getStringRef() == "pod.new" ||
              op->getName().getStringRef() == "pod.read" ||
              op->getName().getStringRef() == "pod.write") {
            // Check if this pod type is an input pod (no @count).
            Type podType;
            if (op->getNumResults() > 0)
              podType = op->getResult(0).getType();
            else if (op->getNumOperands() > 0)
              podType = op->getOperand(0).getType();
            if (podType) {
              std::string s;
              llvm::raw_string_ostream os(s);
              podType.print(os);
              if (s.find("@count") == std::string::npos)
                return true; // Input pod — legal
            }
          }

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

    // Pre-pass: hoist dispatch function.call ops from scf.if to parent.
    // In array-of-dispatch-pods pattern, function.call is inside scf.if
    // (count==0 guard), but pod.read @comp is outside. This causes
    // domination errors during conversion. Since circom dispatch always
    // executes the call (count always reaches 0), hoisting is safe.
    // Hoist both the call and its operands (pod.read for inputs).
    {
      SmallVector<scf::IfOp> dispatchIfs;
      module->walk([&](scf::IfOp ifOp) {
        // Check: does this scf.if contain a function.call that writes to
        // pod @comp? That's the dispatch pattern.
        bool isDispatch = false;
        ifOp.getThenRegion().walk([&](Operation *inner) {
          if (inner->getName().getStringRef() == "function.call") {
            for (auto *user : inner->getResult(0).getUsers()) {
              if (user->getName().getStringRef() == "pod.write") {
                auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
                if (rn && rn.getValue() == "comp")
                  isDispatch = true;
              }
            }
          }
        });
        if (isDispatch)
          dispatchIfs.push_back(ifOp);
      });

      for (auto ifOp : dispatchIfs) {
        // Hoist all ops from the then block to before the scf.if.
        // This includes: pod.read (input), function.call, pod.write @comp,
        // array.write (dispatch array update).
        Block &thenBlock = ifOp.getThenRegion().front();
        SmallVector<Operation *> toHoist;
        for (Operation &op : thenBlock) {
          if (!op.hasTrait<OpTrait::IsTerminator>())
            toHoist.push_back(&op);
        }
        for (auto *op : toHoist)
          op->moveBefore(ifOp);
      }
    }

    // Pre-pass: replace llzk.nondet : i1 with arith.constant false.
    // Keeps the i1 type intact so bool.not, scf.if, etc. still work.
    // The LlzkNonDetPattern in RemovalPatterns handles the tensor<i1>
    // wrapping during dialect conversion if needed.
    {
      SmallVector<Operation *> i1Nondets;
      module->walk([&](Operation *op) {
        if (op->getName().getStringRef() == "llzk.nondet" &&
            op->getNumResults() == 1 && op->getResult(0).getType().isInteger(1))
          i1Nondets.push_back(op);
      });
      for (auto *op : i1Nondets) {
        OpBuilder builder(op);
        auto falseVal = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getBoolAttr(false));
        op->getResult(0).replaceAllUsesWith(falseVal);
        op->erase();
      }

      // Also mark arith dialect as legal so arith.constant i1 survives.
      // The scf-to-stablehlo post-pass handles the i1→tensor<i1> wrapping.
    }

    // Pre-pass: remove $inputs pod fields (only needed by constrain).
    eliminateInputPods(module);

    // Pre-pass: inline single-field input pods to their field types.
    // This handles pods used as while carry (e.g., @claim: !array<8 x felt>).
    inlineInputPodCarries(module);

    // Pre-passes: transform LLZK IR before dialect conversion
    registerStructFieldOffsets(module, typeConverter);
    convertAllFunctions(module, typeConverter, context);
    promoteArraysToWhileCarry(module);
    convertWhileBodyArgsToSSA(module);

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

          // Wrap bare scalars (i1) in tensor<> for stablehlo.select.
          auto resultType = ifOp.getResult(i).getType();
          if (!isa<RankedTensorType>(resultType) &&
              resultType.isIntOrIndexOrFloat()) {
            auto tensorType = RankedTensorType::get({}, resultType);
            thenVal = builder.create<tensor::FromElementsOp>(loc, tensorType,
                                                             thenVal);
            elseVal = builder.create<tensor::FromElementsOp>(loc, tensorType,
                                                             elseVal);
            resultType = tensorType;
          }

          // Broadcast predicate if result is not scalar.
          Value broadPred = pred;
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
    // After conversion + scf.if extraction, func.call ops may be in the same
    // block or in nested while bodies. Replace the pod.read @comp →
    // unrealized_cast chain with the matching func.call result.
    {
      module.walk([&](stablehlo::WhileOp whileOp) {
        for (Block &b : whileOp.getBody()) {
          // Collect func.call ops from this block only. Calls in nested
          // regions can't be used directly (domination). Those cases are
          // handled by the LLZK cleanup pass below.
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
          // Erase array.new if all its uses are gone.
          for (Operation &op : llvm::make_early_inc_range(b))
            if (op.getName().getStringRef() == "array.new" && op.use_empty())
              op.erase();
        }
      });
    }

    // Post-pass: clean up remaining LLZK dialect ops (array.read, pod.read,
    // array.new). These are remnants of pod dispatch bookkeeping. Replace
    // unrealized_conversion_cast chains that source from pod.read with zero
    // tensors, then erase dead LLZK ops bottom-up.
    {
      SmallVector<Operation *> toErase;
      module.walk([&](UnrealizedConversionCastOp castOp) {
        if (castOp.getNumOperands() != 1)
          return;
        auto *srcOp = castOp.getInputs()[0].getDefiningOp();
        if (!srcOp)
          return;
        StringRef srcName = srcOp->getName().getStringRef();
        if (srcName != "pod.read" && srcName != "array.read")
          return;
        auto resultType =
            dyn_cast<RankedTensorType>(castOp.getResult(0).getType());
        if (!resultType)
          return;
        // Replace with zero constant
        OpBuilder b(castOp);
        auto zeroAttr = b.getZeroAttr(resultType);
        auto zero = b.create<stablehlo::ConstantOp>(castOp.getLoc(), resultType,
                                                    zeroAttr);
        castOp.getResult(0).replaceAllUsesWith(zero);
        toErase.push_back(castOp);
      });
      for (auto *op : toErase)
        op->erase();
      // Bottom-up erase dead LLZK ops
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](Operation *op) {
          StringRef ns = op->getName().getDialectNamespace();
          if ((ns == "pod" || ns == "array" || ns == "llzk") &&
              op->use_empty()) {
            op->erase();
            erased = true;
          }
        });
      }
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

    // Post-pass: vectorize independent while loops.
    // Detects while loops where each iteration independently computes
    // output[i] = f(input[i]) and replaces with vectorized element-wise
    // ops. This enables GPU-parallel witness generation.
    vectorizeIndependentWhileLoops(module);

    // Dead code elimination: remove unused constants and ops left over
    // from vectorization (e.g., dead constants from erased while loops).
    {
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](Operation *op) {
          if (isa<func::FuncOp>(op) || isa<func::ReturnOp>(op) ||
              isa<ModuleOp>(op))
            return;
          if (op->getNumResults() > 0 &&
              llvm::all_of(op->getResults(),
                           [](Value v) { return v.use_empty(); })) {
            op->erase();
            erased = true;
          }
        });
      }
    }
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
