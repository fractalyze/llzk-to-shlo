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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FeltPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FunctionPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/RemovalPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "llzk_to_shlo/Dialect/WLA/WLA.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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
    // or "Module_Struct_compute" when the struct sits inside a named
    // sub-module (LLZK v2 template-removal leaves a `module @X` shell around
    // each `struct.def`, so circom-emitted calls are `@X::@X::@compute`).
    std::string name = "main";
    if (auto *structParent = op->getParentOp()) {
      if (structParent->getName().getStringRef() == "struct.def") {
        if (auto sn = structParent->getAttrOfType<StringAttr>("sym_name")) {
          if (!isMainStruct(module, sn.getValue())) {
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

// ===----------------------------------------------------------------------===
// promoteArraysToWhileCarry helpers
// ===----------------------------------------------------------------------===

/// True for values that should be promoted as scf.while carries: felt-element
/// arrays (the original case) AND structs whose fields are mutated via
/// `struct.writem` inside the loop body (added so MiMC7-style scalar field
/// writes buried in scf.if/scf.while thread out as a carry instead of being
/// orphaned by the per-block convertWritemToSSA — see Bug 1 in
/// memory/maci-3-blocked-lowering-bugs-followup.md).
static bool isPromotableCarryType(Type ty) {
  StringRef ns = ty.getDialect().getNamespace();
  if (ns == "array") {
    if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
      if (arrTy.getElementType().getDialect().getNamespace() == "pod")
        return false;
    return true;
  }
  return ns == "struct";
}

/// True for op names whose first operand is a value-typed carry being mutated
/// in place (and thus a candidate for SSA-ification through scf.while/scf.if).
/// `array.insert` is gated by the caller's `includeInsertExtract` flag.
static bool isCarryMutationOp(StringRef name, bool includeInsertExtract) {
  return name == "array.write" || name == "struct.writem" ||
         (includeInsertExtract && name == "array.insert");
}

/// Find values defined outside the while body but used inside whose type is
/// promotable to a carry (see `isPromotableCarryType`).
///
/// Struct-typed captures are filtered down to those actually mutated by a
/// `struct.writem` inside the body. A read-only struct (e.g. the materialized
/// `pod.read %pod[@comp]` substruct that a reader-while only slices into) does
/// not need a carry — promoting it would just shuttle an unchanged tensor
/// through every iteration and break call-site CHECK patterns in lit fixtures
/// like `scalar_pod_comp_materialize.mlir`. Array captures are kept
/// unconditional to preserve existing keccak/AES carry behavior — a read-only
/// array carry is similarly redundant but the existing pipeline depends on it
/// for cross-while data flow consistency.
llvm::SmallSetVector<Value, 4> findCapturedArrays(scf::WhileOp whileOp) {
  Block &body = whileOp.getAfter().front();

  // Pre-collect struct values that are written inside the body so the main
  // walker can drop read-only struct captures.
  //
  // ASSUMPTION (verified across all 25 example LLZK fixtures 2026-05-03):
  // every `struct.writem` in circom-emitted LLZK targets `%self` directly,
  // never an scf.while iter-arg block-arg pass-through of an outer struct.
  // If a future frontend change starts emitting structs as scf.while iter
  // args, this single-operand insert would mis-classify the outer struct as
  // read-only — that case would need to walk the writem operand back through
  // the scf.while carry chain. The new chip's gate would fail byte-equal vs
  // circom and surface the regression loudly.
  llvm::DenseSet<Value> mutatedStructs;
  body.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "struct.writem" ||
        op->getNumOperands() < 1)
      return;
    mutatedStructs.insert(op->getOperand(0));
  });

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
      if (!isPromotableCarryType(operand.getType()))
        continue;
      if (operand.getType().getDialect().getNamespace() == "struct" &&
          !mutatedStructs.contains(operand))
        continue;
      // Values defined in the parent while's body region (e.g. results of
      // an already-promoted earlier sibling scf.while) are valid captures:
      // they sit before this while in the parent body, so SSA dominance
      // gives us their value at this while's init position. Skipping them
      // breaks the inter-sibling carrier bridge — sibling array mutations
      // land on untracked SSA values that downstream DCE silently erases,
      // along with the function.call producing the inserted value.
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

// Forward declarations: the lift recurses into branch bodies via the
// per-block walker, which itself dispatches on scf.if via the lift.
static bool
liftScfIfWithArrayWrites(scf::IfOp ifOp,
                         llvm::MapVector<Value, Value> &parentLatest,
                         bool includeInsertExtract, bool &changed);
static bool
extendResultBearingScfIfArrayChain(scf::IfOp ifOp,
                                   llvm::MapVector<Value, Value> &parentLatest,
                                   bool includeInsertExtract, bool &changed);
static bool
extendExecuteRegionArrayChain(scf::ExecuteRegionOp erOp,
                              llvm::MapVector<Value, Value> &parentLatest,
                              bool includeInsertExtract, bool &changed);

/// Walk a block, threading tracked array values through ops:
///   - Rewire any tracked operand to its latest SSA value.
///   - Convert void array.write (and, when `includeInsertExtract`,
///   array.insert)
///     into SSA form, updating the latest map.
///   - Track array.extract results as new tracked arrays (when
///     `includeInsertExtract`).
///   - For nested scf.while: rewire init operands to the latest tracked
///     SSA value (so a write earlier in this block threads into the
///     while's iter-arg inits) and update latest from the while's results
///     for tracked carries.
///   - For void scf.if that writes to a tracked array (directly or through
///     a nested scf.if): lift into result-bearing form so the update threads
///     through scf.yield. Without this the post-pass at
///     LlzkToStablehlo.cpp:1779 erases the void scf.if as dead code and the
///     write is silently dropped — the keccak_squeeze / chi / round / theta
///     bug fixed by this change.
///
/// `changed` is set to true iff any latest entry was rebound.
static void processBlockForArrayMutations(Block &block,
                                          llvm::MapVector<Value, Value> &latest,
                                          bool includeInsertExtract,
                                          bool &changed) {
  for (Operation &op : llvm::make_early_inc_range(block.getOperations())) {
    StringRef name = op.getName().getStringRef();

    // scf.while: rewire init operands to the latest tracked SSA value, so
    // a write earlier in this block (e.g. an array.insert that updated
    // `latest[%key]`) threads into the while's iter-arg inits. Without
    // this, the next inner stablehlo.while initializes from the pre-update
    // carrier, breaking the chain at the scf.if/scf.while boundary.
    // After rewire, rebind any tracked carry whose init now matches a
    // tracked value to the while's matching result so downstream ops in
    // this block see the post-while value.
    if (name == "scf.while") {
      for (auto &operand : op.getOpOperands()) {
        auto it = latest.find(operand.get());
        if (it != latest.end() && it->second != operand.get())
          operand.set(it->second);
      }
      for (unsigned i = 0; i < op.getNumResults(); ++i) {
        if (i < op.getNumOperands()) {
          Value init = op.getOperand(i);
          for (auto &[key, val] : latest) {
            if (val == init && val != op.getResult(i)) {
              val = op.getResult(i);
              changed = true;
            }
          }
        }
      }
      continue;
    }

    // scf.if with no results that writes to a tracked array: lift to
    // result-bearing form. The lift takes ownership of the body, recurses
    // through nested scf.ifs, and rebinds latest to the new scf.if results.
    // Already result-bearing scf.ifs whose branches modify tracked arrays
    // (typically via inner scf.whiles initialized from the parent carry —
    // LLZK's `<--` (compute-only) emits `%nondet_*` placeholder yields at
    // !array result slots and stuffs the real writes into the inner whiles)
    // get extended in place: new tail result slots carrying the modified
    // arrays so the chain reaches the outer carry.
    if (name == "scf.if") {
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (liftScfIfWithArrayWrites(ifOp, latest, includeInsertExtract,
                                     changed))
          continue;
        if (extendResultBearingScfIfArrayChain(ifOp, latest,
                                               includeInsertExtract, changed))
          continue;
      }
    }

    // scf.execute_region: SSC's struct-of-pods cascade materialiser
    // (`materializeStructOfPodsCompField`) wraps deep K-dispatch cascades
    // (e.g. iden3 Poseidon3's 56-deep MixS_* cascade) inside an
    // execute_region whose yield carries the pod-typed dispatch array, not
    // the felt-typed carrier the cascade arms actually mutate. Without
    // walking into the region, the inner cascade scf.ifs are invisible to
    // `extendResultBearingScfIfArrayChain` and every `array.insert
    // %carrier[%cK]` is left behind.
    if (name == "scf.execute_region") {
      if (auto erOp = dyn_cast<scf::ExecuteRegionOp>(&op)) {
        if (extendExecuteRegionArrayChain(erOp, latest, includeInsertExtract,
                                          changed))
          continue;
      }
    }

    // Rewire any tracked operands to the latest SSA value.
    for (auto &operand : op.getOpOperands()) {
      auto it = latest.find(operand.get());
      if (it != latest.end() && it->second != operand.get())
        operand.set(it->second);
    }

    // array.extract may produce a new tracked array (subarray slice). Track
    // only non-pod arrays — pod-element arrays are dispatch bookkeeping that
    // can't lower to tensor; `isPromotableCarryType` enforces the same rule.
    if (includeInsertExtract && name == "array.extract" &&
        op.getNumResults() > 0) {
      Type ty = op.getResult(0).getType();
      if (isPromotableCarryType(ty) &&
          ty.getDialect().getNamespace() == "array")
        latest.insert({op.getResult(0), op.getResult(0)});
      continue;
    }

    // Mirror SSA-fy state for result-bearing chain links left by an earlier
    // walker pass — keeps `latest` at the chain tip so the by-slot-index
    // yield rewrite in `convertWhileBodyArgsToSSA` is byte-equivalent to
    // walker-1's promote-yield output. Without this,
    // `while_paired_carrier_no_false_collapse` regresses.
    if (isCarryMutationOp(name, includeInsertExtract) &&
        op.getNumResults() == 1) {
      Value arr = op.getOperand(0);
      for (auto &[key, l] : latest) {
        if (l == arr) {
          l = op.getResult(0);
          changed = true;
        }
      }
      continue;
    }

    // struct.writem has 2 operands (struct, value); array.write/insert have
    // 3+ (array, indices..., value). The mutation-classifier predicate is
    // shared with `liftScfIfWithArrayWrites` via `isCarryMutationOp`.
    unsigned minOperands = name == "struct.writem" ? 2 : 3;
    if (isCarryMutationOp(name, includeInsertExtract) &&
        op.getNumResults() == 0 && op.getNumOperands() >= minOperands) {
      Value arr = op.getOperand(0);
      // Only convert writes whose target is in `latest` (directly as a key
      // or transitively via a previously-converted result). Eagerly
      // converting an untracked write would leave a result-bearing op with
      // no consumers — `latest` wouldn't be updated, the yield wouldn't be
      // re-routed, and downstream DCE would erase the new op, silently
      // dropping the write. Leaving it void here lets a later caller
      // (`convertWhileBodyArgsToSSA`) handle it once `arr` is tracked.
      bool isTracked = false;
      for (auto &[key, l] : latest)
        if (l == arr) {
          isTracked = true;
          break;
        }
      if (!isTracked)
        continue;
      OpBuilder b(&op);
      OperationState state(op.getLoc(), name);
      state.addOperands(op.getOperands());
      state.addTypes({arr.getType()});
      for (auto &attr : op.getAttrs())
        state.addAttribute(attr.getName(), attr.getValue());
      Operation *newOp = b.create(state);
      for (auto &[key, l] : latest) {
        if (l == arr) {
          l = newOp->getResult(0);
          changed = true;
        }
      }
      op.erase();
      // Chain subsequent same-block uses of `arr` onto the new op. An earlier
      // walker pass (`includeInsertExtract=false` mode) pins every in-branch
      // mutation operand to the same pre-chain rebind value; without this
      // rewrite, the next SSA-fy down the block sees a stale operand, fails
      // `isTracked`, and silently drops. Canonical case:
      // webb_poseidon_vanchor's @Poseidon_137 else branch (3 sequential
      // array.inserts on the same carrier). The position filter preserves
      // SSA dominance — uses before newOp still refer to the pre-chain value.
      arr.replaceUsesWithIf(newOp->getResult(0), [&](OpOperand &use) {
        // Walk to `block` level so uses nested inside a sibling scf.if /
        // scf.while AFTER newOp are also rebound — without this, deeper
        // recursive walks would still hit a stale operand.
        Operation *anc = block.findAncestorOpInBlock(*use.getOwner());
        if (!anc || anc == newOp)
          return false;
        return newOp->isBeforeInBlock(anc);
      });
    }
  }
}

static bool
liftScfIfWithArrayWrites(scf::IfOp ifOp,
                         llvm::MapVector<Value, Value> &parentLatest,
                         bool includeInsertExtract, bool &changed) {
  // Already result-bearing scf.ifs are handled by the post-pass at
  // LlzkToStablehlo.cpp:1820 (inline both branches + stablehlo.select).
  if (ifOp.getNumResults() != 0)
    return false;

  auto isMutation = [&](StringRef name) {
    return isCarryMutationOp(name, includeInsertExtract);
  };

  // Find which tracked arrays are written inside either region (recursively
  // through nested scf.ifs). The write's first operand is the current SSA
  // value of the array; match it against either a key (block-arg form) or a
  // value (already-updated form) of `parentLatest`. Tracking by key keeps the
  // map unique across nested writes to the same logical array.
  llvm::SmallSetVector<Value, 4> liveArrays;
  auto recordWrite = [&](Operation *op) {
    if (!isMutation(op->getName().getStringRef()) || op->getNumOperands() < 1)
      return;
    Value arr = op->getOperand(0);
    for (auto &[key, latest] : parentLatest) {
      if (latest == arr || key == arr) {
        liveArrays.insert(key);
        return;
      }
    }
  };
  ifOp.getThenRegion().walk(recordWrite);
  ifOp.getElseRegion().walk(recordWrite);

  if (liveArrays.empty())
    return false;

  // Build the new result-bearing scf.if. Result types mirror the array types;
  // the dialect-conversion type converter rewrites them to RankedTensorType
  // during applyPartialConversion.
  SmallVector<Type> resultTypes;
  for (Value arr : liveArrays)
    resultTypes.push_back(arr.getType());

  OpBuilder builder(ifOp);
  auto newIf =
      builder.create<scf::IfOp>(ifOp.getLoc(), resultTypes, ifOp.getCondition(),
                                /*hasElse=*/true);

  // Move each region's body across, then process it in branch-local context.
  // scf::IfOp::build with hasElse=true creates default blocks with yields
  // that we drop before recursing.
  auto migrate = [&](Region &dstRegion, Region &srcRegion) {
    if (!srcRegion.empty()) {
      // Preserve the original block — `takeBody` clears the auto-created one.
      dstRegion.takeBody(srcRegion);
    }
    Block &block = dstRegion.front();
    if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>())
      block.back().erase();

    llvm::MapVector<Value, Value> branchLatest;
    for (Value key : liveArrays)
      branchLatest[key] = parentLatest.lookup(key);

    bool branchChanged = false;
    processBlockForArrayMutations(block, branchLatest, includeInsertExtract,
                                  branchChanged);

    SmallVector<Value> yieldArgs;
    for (Value key : liveArrays)
      yieldArgs.push_back(branchLatest.lookup(key));
    OpBuilder yb(&block, block.end());
    yb.create<scf::YieldOp>(ifOp.getLoc(), yieldArgs);
  };
  migrate(newIf.getThenRegion(), ifOp.getThenRegion());
  migrate(newIf.getElseRegion(), ifOp.getElseRegion());

  for (auto [i, key] : llvm::enumerate(liveArrays)) {
    parentLatest[key] = newIf.getResult(i);
    changed = true;
  }
  ifOp.erase();
  return true;
}

// Extends an already-result-bearing scf.if whose branches modify tracked
// arrays without yielding them at any existing slot (LLZK's `<--` shape:
// the !array result slots get `llzk.nondet` placeholder yields while the
// actual writes happen in inner whiles initialized from the parent carry).
// For each tracked key whose branches mutate, we either reuse an existing
// tail slot (matched by yield-op identity) or append a new one. The append
// path is what threads the chain through the if; the reuse path keeps the
// rewrite idempotent across multiple walker invocations
// (promoteArraysToWhileCarry's convertArrayWritesToSSA seeds with captured
// arrays only, while convertWhileBodyArgsToSSA later seeds with all body
// args — the second walk discovers extra liveKeys not covered by the first
// extension and appends slots for them).
static bool
extendResultBearingScfIfArrayChain(scf::IfOp oldIf,
                                   llvm::MapVector<Value, Value> &parentLatest,
                                   bool includeInsertExtract, bool &changed) {
  if (oldIf.getNumResults() == 0)
    return false;
  if (oldIf.getThenRegion().empty() || oldIf.getElseRegion().empty())
    return false;

  // Process each branch with branchLatest seeded from parentLatest. The
  // recursion lets line-424 logic rebind tracked carries through inner
  // scf.whiles inside each branch.
  llvm::MapVector<Value, Value> thenLatest, elseLatest;
  for (auto &[k, v] : parentLatest) {
    thenLatest[k] = v;
    elseLatest[k] = v;
  }
  bool thenChanged = false, elseChanged = false;
  Block &thenBlock = oldIf.getThenRegion().front();
  Block &elseBlock = oldIf.getElseRegion().front();
  processBlockForArrayMutations(thenBlock, thenLatest, includeInsertExtract,
                                thenChanged);
  processBlockForArrayMutations(elseBlock, elseLatest, includeInsertExtract,
                                elseChanged);
  // Propagate branch-walk changes to the outer fixed-point flag so the
  // main pass loop does not terminate prematurely while branch IR is
  // still settling.
  changed |= thenChanged || elseChanged;

  // Find tracked keys whose branchLatest differs from parentLatest in either
  // branch — those are the keys the if body actually mutates. Iteration
  // order is deterministic because parentLatest is a MapVector seeded by
  // callers in stable order (block-arg order, etc.) — the resulting
  // liveKeys ordering controls the order of newly appended scf.if result
  // slots, which must be reproducible across runs.
  SmallVector<Value> liveKeys;
  for (auto &[k, parentVal] : parentLatest) {
    Value tNew = thenLatest.lookup(k);
    Value eNew = elseLatest.lookup(k);
    if (tNew != parentVal || eNew != parentVal)
      liveKeys.push_back(k);
  }
  if (liveKeys.empty())
    return false;

  // Classify each liveKey: reuse an existing slot whose yields already match
  // branchLatest, or queue for a new tail slot. Update parentLatest only
  // after we know whether oldIf will be replaced — if it is, the reuse
  // result-value must reference newIf, since oldIf gets erased.
  auto thenYield = cast<scf::YieldOp>(thenBlock.getTerminator());
  auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());
  SmallVector<std::pair<Value, unsigned>> reuseMappings;
  SmallVector<Value> keysToAppend;
  for (Value key : liveKeys) {
    Value tVal = thenLatest.lookup(key);
    Value eVal = elseLatest.lookup(key);
    bool found = false;
    for (unsigned i = 0; i < oldIf.getNumResults(); ++i) {
      if (thenYield.getOperand(i) == tVal && elseYield.getOperand(i) == eVal) {
        reuseMappings.push_back({key, i});
        found = true;
        break;
      }
    }
    if (!found)
      keysToAppend.push_back(key);
  }

  if (keysToAppend.empty()) {
    // Pure reuse, no IR change.
    for (auto &[key, i] : reuseMappings) {
      if (parentLatest[key] != oldIf.getResult(i)) {
        parentLatest[key] = oldIf.getResult(i);
        changed = true;
      }
    }
    return !reuseMappings.empty();
  }

  // Append path: build a new scf.if with extra tail slots for keysToAppend.
  unsigned origNumResults = oldIf.getNumResults();
  SmallVector<Type> newResultTypes(oldIf->getResultTypes().begin(),
                                   oldIf->getResultTypes().end());
  for (Value k : keysToAppend)
    newResultTypes.push_back(k.getType());

  OpBuilder builder(oldIf);
  auto newIf = builder.create<scf::IfOp>(
      oldIf.getLoc(), newResultTypes, oldIf.getCondition(), /*hasElse=*/true);
  newIf.getThenRegion().takeBody(oldIf.getThenRegion());
  newIf.getElseRegion().takeBody(oldIf.getElseRegion());

  auto extendYield = [&](Block &block, llvm::MapVector<Value, Value> &latest) {
    auto yield = cast<scf::YieldOp>(block.getTerminator());
    SmallVector<Value> newArgs(yield.getOperands().begin(),
                               yield.getOperands().end());
    for (Value k : keysToAppend)
      newArgs.push_back(latest.lookup(k));
    OpBuilder yb(yield);
    yb.create<scf::YieldOp>(yield.getLoc(), newArgs);
    yield.erase();
  };
  extendYield(newIf.getThenRegion().front(), thenLatest);
  extendYield(newIf.getElseRegion().front(), elseLatest);

  for (unsigned i = 0; i < origNumResults; ++i)
    oldIf.getResult(i).replaceAllUsesWith(newIf.getResult(i));

  for (auto &[key, i] : reuseMappings)
    parentLatest[key] = newIf.getResult(i);
  for (auto [i, k] : llvm::enumerate(keysToAppend))
    parentLatest[k] = newIf.getResult(origNumResults + i);

  oldIf.erase();
  changed = true;
  return true;
}

// Extend a result-bearing scf.execute_region by appending NEW tail result
// slots for every tracked array its body mutates. Same shape as
// `extendResultBearingScfIfArrayChain` but with a single body region (no
// then/else split). The classic trigger is SSC's
// `materializeStructOfPodsCompField`, which wraps a K-dispatch cascade in
// `scf.execute_region -> (!array<K x !pod>)`. The execute_region's existing
// yield slot is for the pod-typed dispatch array; the felt-typed carrier the
// cascade arms write to is invisible at this op's boundary. Without this
// extension, the walker cannot reach the inner cascade scf.ifs and the
// 56-deep `array.insert %carrier[%cK]` chain (iden3 Poseidon3 MixS_*) gets
// dropped wholesale.
static bool
extendExecuteRegionArrayChain(scf::ExecuteRegionOp oldEr,
                              llvm::MapVector<Value, Value> &parentLatest,
                              bool includeInsertExtract, bool &changed) {
  if (oldEr.getRegion().empty())
    return false;
  Block &body = oldEr.getRegion().front();

  llvm::MapVector<Value, Value> bodyLatest;
  for (auto &[k, v] : parentLatest)
    bodyLatest[k] = v;

  bool bodyChanged = false;
  processBlockForArrayMutations(body, bodyLatest, includeInsertExtract,
                                bodyChanged);
  changed |= bodyChanged;

  SmallVector<Value> liveKeys;
  for (auto &[k, parentVal] : parentLatest) {
    Value bNew = bodyLatest.lookup(k);
    if (bNew != parentVal)
      liveKeys.push_back(k);
  }
  if (liveKeys.empty())
    return false;

  // Idempotency: a second walker pass over the same execute_region (e.g.
  // promoteArraysToWhileCarry → convertWhileBodyArgsToSSA) must NOT re-append
  // slots already covered. Match by yield-op identity, like the sibling
  // extender does.
  auto yieldOp = cast<scf::YieldOp>(body.getTerminator());
  SmallVector<std::pair<Value, unsigned>> reuseMappings;
  SmallVector<Value> keysToAppend;
  for (Value key : liveKeys) {
    Value bVal = bodyLatest.lookup(key);
    bool found = false;
    for (unsigned i = 0; i < oldEr.getNumResults(); ++i) {
      if (yieldOp.getOperand(i) == bVal) {
        reuseMappings.push_back({key, i});
        found = true;
        break;
      }
    }
    if (!found)
      keysToAppend.push_back(key);
  }

  if (keysToAppend.empty()) {
    for (auto &[key, i] : reuseMappings) {
      if (parentLatest[key] != oldEr.getResult(i)) {
        parentLatest[key] = oldEr.getResult(i);
        changed = true;
      }
    }
    return !reuseMappings.empty();
  }

  unsigned origNumResults = oldEr.getNumResults();
  SmallVector<Type> newResultTypes(oldEr->getResultTypes().begin(),
                                   oldEr->getResultTypes().end());
  for (Value k : keysToAppend)
    newResultTypes.push_back(k.getType());

  OpBuilder builder(oldEr);
  auto newEr =
      builder.create<scf::ExecuteRegionOp>(oldEr.getLoc(), newResultTypes);
  newEr.getRegion().takeBody(oldEr.getRegion());

  Block &newBody = newEr.getRegion().front();
  auto newYield = cast<scf::YieldOp>(newBody.getTerminator());
  SmallVector<Value> newArgs(newYield.getOperands().begin(),
                             newYield.getOperands().end());
  for (Value k : keysToAppend)
    newArgs.push_back(bodyLatest.lookup(k));
  OpBuilder yb(newYield);
  yb.create<scf::YieldOp>(newYield.getLoc(), newArgs);
  newYield.erase();

  for (unsigned i = 0; i < origNumResults; ++i)
    oldEr.getResult(i).replaceAllUsesWith(newEr.getResult(i));

  for (auto &[key, i] : reuseMappings)
    parentLatest[key] = newEr.getResult(i);
  for (auto [i, k] : llvm::enumerate(keysToAppend))
    parentLatest[k] = newEr.getResult(origNumResults + i);

  oldEr.erase();
  changed = true;
  return true;
}

/// Walk the body block, convert array.write to produce SSA result, track
/// latest value. Returns a MapVector<Value, Value> of latestArraySSA so
/// downstream consumers iterate keys in deterministic insertion order.
llvm::MapVector<Value, Value>
convertArrayWritesToSSA(Block &bodyBlock, ArrayRef<Value> arrayBlockArgs) {
  llvm::MapVector<Value, Value> latestArraySSA;
  for (auto blockArg : arrayBlockArgs)
    latestArraySSA[blockArg] = blockArg;

  bool changed = false;
  processBlockForArrayMutations(bodyBlock, latestArraySSA,
                                /*includeInsertExtract=*/false, changed);
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

    llvm::MapVector<Value, Value> latestSSA;
    for (auto arg : body.getArguments()) {
      if (isPromotableCarryType(arg.getType()))
        latestSSA[arg] = arg;
    }
    if (latestSSA.empty())
      continue;

    bool changed = false;
    processBlockForArrayMutations(body, latestSSA,
                                  /*includeInsertExtract=*/true, changed);

    if (!changed)
      continue;

    auto yieldOp = cast<scf::YieldOp>(body.getTerminator());
    SmallVector<Value> newYieldArgs;
    // Rewrite by slot index, not by yield-operand identity: the line 491-509
    // scf.while rebind can shift the yield operand off the body arg key, so
    // `latestSSA.find(operand)` would miss a chain tip built by
    // `extendResultBearingScfIfArrayChain`. Canonical bug:
    // webb_poseidon_vanchor's `@Poseidon_137` outer-while slot 5 (`@mix`
    // carrier). See CLAUDE.md "phantom rebind via read-only inner-while
    // capture" for the forensic trace.
    for (auto [i, val] : llvm::enumerate(yieldOp.getOperands())) {
      // Defensive bounds check: scf.while invariants guarantee
      // yield.size() == body.numArgs(), but ill-formed IR from upstream
      // passes shouldn't crash us here.
      Value bodyArg =
          i < body.getNumArguments() ? body.getArgument(i) : nullptr;
      auto it = bodyArg ? latestSSA.find(bodyArg) : latestSSA.end();
      if (it != latestSSA.end() && it->second != bodyArg)
        newYieldArgs.push_back(it->second);
      else
        newYieldArgs.push_back(val);
    }
    OpBuilder yb(yieldOp);
    yb.create<scf::YieldOp>(yieldOp.getLoc(), newYieldArgs);
    yieldOp.erase();
  }
}

/// Rewire each `struct.readm %V[@out] : <@Ark_*>` op whose operand is a
/// hoisted `function.call` (defined in an ancestor scf.* region, not the
/// readm's immediate scope) to consume a freshly-emitted call on the same
/// row at the readm site.
///
/// Webb's circom-lowered Poseidon emits the per-row Ark inputs by hoisting
/// `Ark_{N+68}(array.extract %arg7[N])` to the inner-while body top — the
/// hoist runs BEFORE the per-iter cascade-arm column write. At inner iter
/// j the cascade arm for outer-K=N writes column j to `%arg7[N]`, then
/// commits the hoisted Ark result to `%carrier[N]`. By iter 4 (the last
/// inner iter, where column 4 is finally written), the hoist value was
/// computed against `%arg7[N]` with column 4 still stale — so the surviving
/// `%carrier[N]` write equals `Ark_{N+68}([@mix[N-1][0..3], stale])` rather
/// than the canonical `Ark_{N+68}(@mix[N-1])`. iden3 avoids the staleness
/// by splitting input-load from cascade into separate scf.while loops; we
/// match that semantically by re-emitting the call against the post-write
/// row at each consumer site.
///
/// Filter: only rewires when `function.call`'s parent region is an ANCESTOR
/// of the `struct.readm`'s parent region (i.e., the call is hoisted out).
/// iden3's `struct.readm @Ark_X` calls all live in symbol-getter function
/// bodies (operand = function arg, same region as readm) and are unaffected.
/// iden3's `struct.readm @Sigma_5` cascade-arm callers use inline
/// `function.call` defined in the same scf.if region — also unaffected.
void replaceHoistedArkReadmWithFreshCall(ModuleOp module) {
  SmallVector<Operation *> targets;
  module.walk([&](Operation *readm) {
    if (readm->getName().getStringRef() != "struct.readm")
      return;
    if (readm->getNumOperands() != 1 || readm->getNumResults() != 1)
      return;
    auto member = readm->getAttrOfType<FlatSymbolRefAttr>("member_name");
    if (!member || member.getValue() != "out")
      return;

    Operation *callOp = readm->getOperand(0).getDefiningOp();
    if (!callOp || callOp->getName().getStringRef() != "function.call")
      return;
    if (callOp->getNumOperands() != 1)
      return;

    auto callee = callOp->getAttrOfType<SymbolRefAttr>("callee");
    if (!callee)
      return;
    if (!callee.getRootReference().getValue().starts_with("Ark_"))
      return;

    Region *callRegion = callOp->getParentRegion();
    Region *readmRegion = readm->getParentRegion();
    if (!callRegion || !readmRegion || callRegion == readmRegion)
      return;
    if (!callRegion->isAncestor(readmRegion))
      return;

    Operation *origExtract = callOp->getOperand(0).getDefiningOp();
    if (!origExtract ||
        origExtract->getName().getStringRef() != "array.extract" ||
        origExtract->getNumOperands() != 2)
      return;
    Operation *idxDef = origExtract->getOperand(1).getDefiningOp();
    if (!idxDef || idxDef->getName().getStringRef() != "arith.constant")
      return;
    if (!idxDef->getAttrOfType<IntegerAttr>("value"))
      return;

    targets.push_back(readm);
  });

  for (Operation *readm : targets) {
    Operation *origCall = readm->getOperand(0).getDefiningOp();
    Operation *origExtract = origCall->getOperand(0).getDefiningOp();
    Value arrCarrier = origExtract->getOperand(0);
    Operation *idxDef = origExtract->getOperand(1).getDefiningOp();
    auto idxAttr = idxDef->getAttrOfType<IntegerAttr>("value");

    OpBuilder b(readm);
    Location loc = readm->getLoc();

    Value freshIdx =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(idxAttr.getInt()))
            .getResult();

    Operation *freshExtract = b.clone(*origExtract);
    freshExtract->setOperand(0, arrCarrier);
    freshExtract->setOperand(1, freshIdx);

    Operation *freshCall = b.clone(*origCall);
    freshCall->setOperand(0, freshExtract->getResult(0));

    readm->setOperand(0, freshCall->getResult(0));

    // The hoisted call+extract become dead when this was their last consumer
    // (the common case at the cascade arms we target). Erase eagerly so the
    // orphans don't trickle through every post-pass walk.
    if (origCall->use_empty()) {
      origCall->erase();
      if (origExtract->use_empty())
        origExtract->erase();
    }
  }
}

/// Clone the dead K=0 cascade-arm `(struct.readm + array.insert)` pair into
/// the OUTERMOST eq-counter-zero `scf.if`'s THEN branch.
///
/// SSC's `materializeStructOfPodsCompField` emits
/// `array.insert %carrier[%cK] = struct.readm(%hoisted_call[@out])` for each
/// dispatched call. Its writer-skip drops calls nested inside `scf.if`
/// ancestors that lack a wrapping `scf.execute_region` (see
/// `findDispatchCountGuardHoistAncestor`). Webb's Poseidon variants nest the
/// round-0 Ark trigger inside `scf.if(eq outer_counter, 0)` THEN with no
/// `scf.execute_region` wrapping — so the materializer drops the OUTERMOST
/// THEN writer. The OUTERMOST ELSE branch (Mix cascade) IS wrapped in
/// `scf.execute_region`, so the materializer still emits a writer into the
/// K=0 cascade arm there — even though that arm is structurally dead
/// (predicate `cmpi eq(cast_to_index counter, 0)` is false because the ELSE
/// branch is reached only when `counter != 0`). Net effect at outer iter 0:
/// OUTERMOST takes THEN -> `%carrier[0]` is never written -> downstream
/// reads return dense<0> instead of `Ark_K(input)` -> the input is erased.
///
/// This pre-pass restores the missing OUTERMOST THEN writer by cloning the
/// dead K=0 cascade arm's `struct.readm + array.insert` pair before THEN's
/// yield. The cloned `struct.readm`'s operand is a hoisted call value at the
/// enclosing inner-while body's top, which dominates both THEN and ELSE — so
/// the cloned op references it directly.
void injectDeadCascadeArmIntoOutermostThen(ModuleOp module) {
  // OUTERMOST predicate: `bool.cmp eq(%felt_counter, %felt_const_0)`.
  // The cascade-arm predicate is `arith.cmpi eq` on an index value, so the
  // felt-typed comparison distinguishes the outer scf.if from any enclosing
  // cascade arm K=0 we might walk past.
  auto isEqFeltCounterZero = [](scf::IfOp ifOp) -> bool {
    Operation *def = ifOp.getCondition().getDefiningOp();
    if (!def || def->getName().getStringRef() != "bool.cmp")
      return false;
    auto pred = parseBoolCmpPredicate(def->getAttr("predicate"));
    if (!pred || *pred != /*eq=*/0)
      return false;
    if (def->getNumOperands() != 2)
      return false;
    Operation *rhsDef = def->getOperand(1).getDefiningOp();
    if (!rhsDef || rhsDef->getName().getStringRef() != "felt.const")
      return false;
    auto feltAttr =
        dyn_cast_or_null<llzk::felt::FeltConstAttr>(rhsDef->getAttr("value"));
    return feltAttr && feltAttr.getValue().isZero();
  };

  SmallVector<Operation *> funcOps;
  module.walk([&](Operation *op) {
    StringRef n = op->getName().getStringRef();
    if (n == "func.func" || n == "function.def")
      funcOps.push_back(op);
  });

  for (Operation *funcOp : funcOps) {
    Region &funcRegion = funcOp->getRegion(0);
    if (funcRegion.empty())
      continue;
    Block &funcBlock = funcRegion.front();

    // mSoPCF.carrier-for tags the function-level `array.new` ops the
    // materializer created. There is at most one per dispatched field
    // (typically `@out`) per Poseidon-style class.
    SmallVector<Operation *> carriers;
    for (Operation &op : funcBlock) {
      if (op.getName().getStringRef() != "array.new")
        continue;
      if (op.hasAttr("mSoPCF.carrier-for"))
        carriers.push_back(&op);
    }

    for (Operation *carrierOp : carriers) {
      if (carrierOp->getNumResults() != 1)
        continue;
      Value carrier = carrierOp->getResult(0);

      // Find an `array.insert %carrier[%c0] = struct.readm(...)` pair that
      // sits in the ELSE region of an OUTERMOST eq-counter-zero scf.if.
      Operation *deadInsert = nullptr;
      Operation *deadReadm = nullptr;
      scf::IfOp outermostIf = nullptr;

      for (OpOperand &use : carrier.getUses()) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() != "array.insert" ||
            use.getOperandNumber() != 0 || user->getNumOperands() != 3)
          continue;
        Operation *idxDef = user->getOperand(1).getDefiningOp();
        if (!idxDef || idxDef->getName().getStringRef() != "arith.constant")
          continue;
        auto idxAttr = idxDef->getAttrOfType<IntegerAttr>("value");
        if (!idxAttr || !idxAttr.getValue().isZero())
          continue;
        Operation *valDef = user->getOperand(2).getDefiningOp();
        if (!valDef || valDef->getName().getStringRef() != "struct.readm")
          continue;

        // Walk ancestors to find the OUTERMOST eq-counter-zero scf.if; stop
        // at the first match. `user` must reside in its ELSE region — a
        // THEN-region match means the materializer already covered the
        // round-0 trigger and no fix is needed.
        Operation *cur = user;
        scf::IfOp found = nullptr;
        while (Operation *parent = cur->getParentOp()) {
          if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            if (isEqFeltCounterZero(ifOp)) {
              Region *region = cur->getParentRegion();
              while (region && region->getParentOp() != ifOp) {
                Operation *outer = region->getParentOp();
                region = outer ? outer->getParentRegion() : nullptr;
              }
              if (region == &ifOp.getElseRegion())
                found = ifOp;
              break;
            }
          }
          cur = parent;
        }
        if (found) {
          outermostIf = found;
          deadInsert = user;
          deadReadm = valDef;
          break; // The materializer emits at most one K=0 writer per carrier.
        }
      }

      if (!outermostIf)
        continue;

      // Skip when THEN already covers K=0 — implies the materializer found a
      // valid emit site (or some upstream pass already patched THEN) and we
      // would double-write the slot.
      bool thenHasInsert = false;
      outermostIf.getThenRegion().walk([&](Operation *op) {
        if (op->getName().getStringRef() != "array.insert" ||
            op->getNumOperands() != 3 || op->getOperand(0) != carrier)
          return WalkResult::advance();
        Operation *idxDef = op->getOperand(1).getDefiningOp();
        if (!idxDef || idxDef->getName().getStringRef() != "arith.constant")
          return WalkResult::advance();
        auto idxAttr = idxDef->getAttrOfType<IntegerAttr>("value");
        if (idxAttr && idxAttr.getValue().isZero()) {
          thenHasInsert = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (thenHasInsert)
        continue;

      // Inject a FRESH `(array.extract + function.call + struct.readm +
      // array.insert)` chain inside THEN before its yield. Cloning the dead
      // K=0 cascade arm's struct.readm directly would reuse the hoisted Ark
      // call result, which is computed at the enclosing inner-while body's
      // top — BEFORE THEN's input-load `array.insert %ark_inputs[%c0]`. That
      // value reflects the iter-start row 0, missing the last column (only
      // populated by the inner iter that runs the input write at column
      // `n-1`). Building a fresh extract+call+readm here lets the call see
      // the row 0 produced by THEN's input writes; at the inner iter where
      // column `n-1` is loaded, the carrier ends up with the full-state Ark
      // output. Subsequent SSA-fication threads `array.extract` to the
      // latest (post-insert) row 0 within the iter, so the new call uses
      // the just-loaded value.
      Operation *origCall = deadReadm->getOperand(0).getDefiningOp();
      if (!origCall || origCall->getName().getStringRef() != "function.call" ||
          origCall->getNumOperands() != 1)
        continue;
      Operation *origExtract = origCall->getOperand(0).getDefiningOp();
      if (!origExtract ||
          origExtract->getName().getStringRef() != "array.extract" ||
          origExtract->getNumOperands() != 2)
        continue;
      Value arkInputsCarrier = origExtract->getOperand(0);

      Block &thenBlock = outermostIf.getThenRegion().front();
      Operation *thenTerm = thenBlock.getTerminator();
      OpBuilder b(thenTerm);
      Location loc = deadInsert->getLoc();

      Value c0 =
          b.create<arith::ConstantOp>(loc, b.getIndexAttr(0)).getResult();

      // Clone origExtract / origCall / deadReadm so MLIR-property-stored
      // inherent attributes (e.g. `function.call.callee`) ride along — copying
      // only `op->getAttrs()` would miss those. Then rewire each clone's
      // input-operand to the previous op's fresh result.
      Operation *newExtract = b.clone(*origExtract);
      newExtract->setOperand(0, arkInputsCarrier);
      newExtract->setOperand(1, c0);

      Operation *newCall = b.clone(*origCall);
      newCall->setOperand(0, newExtract->getResult(0));

      Operation *newReadm = b.clone(*deadReadm);
      newReadm->setOperand(0, newCall->getResult(0));

      OperationState insertState(loc, "array.insert");
      insertState.addOperands({carrier, c0, newReadm->getResult(0)});
      b.create(insertState);
    }
  }
}

/// Inject `array.write(%mix_input_row, %c0, %sigma_value)` between the
/// partial-round Mix's `array.extract` and `function.call @Mix_X` so the
/// Mix input row's col 0 is the partial-round Sigma value, not the
/// dense<0> init.
///
/// Webb's Poseidon partial-round LLZK lowering reaches the post-loop Mix
/// via a 4-iter `scf.while` whose Mix-input-row carrier init is the OUTER
/// body's dense<0> slot (full rounds pass it through untouched). The loop
/// body writes Ark@out cols 1..4 into cols 1..4 of the current-round row
/// but never touches col 0, so col 0 stays at the dense<0> init value.
/// The Sigma scalar IS computed and written to the STATE carrier before
/// the loop, but never copied into the Mix-input-row carrier.
///
/// Canonical partial-round Mix input row = `[pow5(Ark[K][0]),
/// Ark[K][1..4]]`. Lowered IR produces `[0, Ark[K][1..4]]`. This pass
/// patches col 0 by walking backward from each Mix call to the most
/// recent Sigma scalar (= `array.read` of `array.write` of
/// `struct.readm` of `function.call @Sigma_X::@compute`).
///
/// Pattern guard: only matches Mix calls whose operand is
/// `array.extract` reading from a 3-result `scf.while` carrier (result
/// index 1). Iden3 / circomlib Poseidon use a separate input-load + main
/// + partial-round while structure whose Mix operand is a function block
/// argument, so they are rejected by the guard.
void injectSigmaIntoPartialRoundMixInput(ModuleOp module) {
  SmallVector<Operation *> targets;
  module.walk([&](Operation *mixCall) {
    if (mixCall->getName().getStringRef() != "function.call")
      return;
    auto callee = mixCall->getAttrOfType<SymbolRefAttr>("callee");
    if (!callee || !callee.getRootReference().getValue().starts_with("Mix_"))
      return;
    if (mixCall->getNumOperands() != 1)
      return;

    Operation *extractOp = mixCall->getOperand(0).getDefiningOp();
    if (!extractOp || extractOp->getName().getStringRef() != "array.extract" ||
        extractOp->getNumOperands() != 2)
      return;

    auto whileResult = dyn_cast<OpResult>(extractOp->getOperand(0));
    if (!whileResult)
      return;
    Operation *whileOp = whileResult.getOwner();
    if (!isa<scf::WhileOp>(whileOp))
      return;
    if (whileOp->getNumResults() != 3 || whileResult.getResultNumber() != 1)
      return;

    targets.push_back(mixCall);
  });

  for (Operation *mixCall : targets) {
    Operation *extractOp = mixCall->getOperand(0).getDefiningOp();

    // Walk backward in the same block for the most recent Sigma scalar:
    //   %sigma = array.read(%scratchpad_after_sigma_write, %pos)
    //   where %scratchpad_after_sigma_write = array.write(_, _, %readm)
    //         %readm = struct.readm(%sigma_call)[@out]
    //         %sigma_call = function.call @Sigma_*::@compute
    Operation *sigmaScalar = nullptr;
    for (Operation *cur = mixCall->getPrevNode(); cur;
         cur = cur->getPrevNode()) {
      if (cur->getName().getStringRef() != "array.read")
        continue;
      if (cur->getNumOperands() < 1)
        continue;
      Operation *writeOp = cur->getOperand(0).getDefiningOp();
      if (!writeOp || writeOp->getName().getStringRef() != "array.write" ||
          writeOp->getNumOperands() < 3)
        continue;
      // LLZK array.write op layout: (array, idx0, idx1, ..., value). The
      // value is always the last operand; using .back() instead of
      // getOperand(2) keeps the match correct when the scratchpad happens
      // to be multi-dimensional (would have additional index operands).
      Operation *readmOp = writeOp->getOperands().back().getDefiningOp();
      if (!readmOp || readmOp->getName().getStringRef() != "struct.readm" ||
          readmOp->getNumOperands() != 1)
        continue;
      Operation *sigmaCall = readmOp->getOperand(0).getDefiningOp();
      if (!sigmaCall || sigmaCall->getName().getStringRef() != "function.call")
        continue;
      auto sigCallee = sigmaCall->getAttrOfType<SymbolRefAttr>("callee");
      if (!sigCallee ||
          !sigCallee.getRootReference().getValue().starts_with("Sigma_"))
        continue;
      sigmaScalar = cur;
      break;
    }
    if (!sigmaScalar)
      continue;

    OpBuilder b(mixCall);
    Location loc = mixCall->getLoc();

    Value c0 = b.create<arith::ConstantOp>(loc, b.getIndexAttr(0)).getResult();

    OperationState writeState(loc, "array.write");
    writeState.addOperands(
        {extractOp->getResult(0), c0, sigmaScalar->getResult(0)});
    b.create(writeState);
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
    Block *whileBlock = newWhile->getBlock();
    for (unsigned idx = 0; idx < capturedArrays.size(); ++idx) {
      Value extArr = capturedArrays[idx];
      Value replacement = newWhile.getResult(origNumResults + idx);
      // Rewrite every use whose enclosing block is `whileBlock`, including
      // uses nested inside subsequent sibling whiles' bodies. Without the
      // nested case, a sibling while's `findCapturedArrays` would still see
      // the original llzk.nondet (zero-init) value and the inter-while data
      // flow would break.
      for (auto &use : llvm::make_early_inc_range(extArr.getUses())) {
        Operation *user = use.getOwner();
        if (user == newWhile.getOperation())
          continue;
        Operation *anc = whileBlock->findAncestorOpInBlock(*user);
        if (!anc)
          continue;
        if (anc->isBeforeInBlock(newWhile))
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

/// Collapse redundant carrier pairs in stablehlo.while.
///
/// Two upstream passes each create felt-typed carriers for the same logical
/// signal but don't communicate:
///   - SimplifySubComponents.flattenPodArrayWhileCarry flattens pod-array
///     iter-args into per-field felt iter-args. These inits inherit from the
///     original pod-array carrier's `%nondet` (post-eliminatePodDispatch),
///     which lowers to a zero constant. The body is never written: the
///     yield is the literal body argument.
///   - LlzkToStablehlo.promoteArraysToWhileCarry independently captures
///     module-level felt arrays (`array.new` produced by
///     SimplifySubComponents materializing substruct outputs). These end
///     up at separate iter-arg slots with `array.new`-zero init AND a
///     properly threaded yield (writes flow through inner whiles).
///
/// When both fire on the same logical signal — AES `@xor_3$inputs[@a]/[@b]`
/// is the canonical case — we get a pair of iter-args of identical type:
/// one DEAD (yield = body arg unchanged), one LIVE (yield = computed). A
/// downstream reader that bound to the DEAD slot before the captured-array
/// pair existed never gets re-bound, dropping all data flow on that path.
///
/// The fix: at every nesting level of `stablehlo.while`, find slots whose
/// yield is a literal pass-through of the body argument AND whose init is
/// a zero-splat constant. Pair each with a sibling slot of identical type
/// AND identical zero-splat init whose yield is computed. Redirect the
/// DEAD result's external uses to the LIVE result.
///
/// Process bottom-up so inner-level RAUW propagates: an outer slot's yield
/// that referenced an inner DEAD result (e.g., `%57:21` slot 11 yielding
/// `%61#3`) becomes `%61#17` after the inner pass, and the chain repairs
/// itself one level at a time.
///
/// Pairing is slot-order: the first DEAD pairs with the first LIVE of
/// matching type, second with second, etc. This preserves @a/@b ordering
/// when both fields have flatten/capture duplicates.
///
/// Both inits MUST be zero-splat constants. This is the load-bearing safety
/// constraint: it ensures round 0 starts identical for both slots, so RAUW
/// preserves observable semantics for ALL downstream readers regardless of
/// what they expected the DEAD slot to carry. A DEAD slot that yields its
/// body arg is just "constant-init plumbed through" — if init is zero,
/// every iteration yields zero. The LIVE slot also starts at zero.
/// Differing init values would mean the DEAD slot was carrying a
/// meaningful non-zero value (rare, but real for some keccak_pad shapes).
///
/// Idempotency: after RAUW, the DEAD slot has no external uses; a re-run
/// finds no new pairs to collapse.
void collapseRedundantWhileCarrierPairs(ModuleOp module) {
  // Trace transitively through enclosing-while body-arg chains so inner
  // whiles whose inits are outer body args still resolve to the root
  // constant. Without this the inner-level RAUW never fires.
  //
  // Critical safety check: at every enclosing-while level visited, require
  // the slot to be passthrough (yield operand at idx == body arg at idx).
  // Otherwise the inner slot's "every iteration body arg = init = zero"
  // claim (which the RAUW depends on) does not hold — an intermediate while
  // mutating the carrier means later iterations of THIS while see non-zero
  // body args and the dead slot is not actually always-zero. AES xor_2 .b
  // is the canonical violator: zero-init at the main while, but the main
  // while's body computes a non-zero .b yield, so any inner-level
  // passthrough is only momentarily zero on the very first main-while
  // iteration.
  auto isZeroSplatTransitively = [](Value v) -> bool {
    // Bound deeper than any real circuit (AES caps at depth 5 per CLAUDE.md);
    // the cap defends against malformed IR with a body-arg cycle.
    constexpr int kMaxWhileNestingHops = 16;
    for (int hops = 0; hops < kMaxWhileNestingHops; ++hops) {
      if (v.getDefiningOp())
        return isZeroSplatConstant(v);
      auto blockArg = dyn_cast<BlockArgument>(v);
      if (!blockArg)
        return false;
      Operation *parent = blockArg.getOwner()->getParentOp();
      auto parentWhile = dyn_cast_or_null<stablehlo::WhileOp>(parent);
      if (!parentWhile)
        return false;
      unsigned idx = blockArg.getArgNumber();
      if (idx >= parentWhile->getNumOperands())
        return false;
      // Reject if this enclosing while mutates the slot. Body arg comes from
      // the AFTER region; the yield must be the same body arg at the same
      // index for the slot to remain pinned to the init across iterations.
      Block &parentBody = parentWhile.getBody().front();
      auto parentReturn =
          dyn_cast<stablehlo::ReturnOp>(parentBody.getTerminator());
      if (!parentReturn || idx >= parentReturn.getNumOperands() ||
          idx >= parentBody.getNumArguments())
        return false;
      if (parentReturn.getOperand(idx) != parentBody.getArgument(idx))
        return false;
      v = parentWhile->getOperand(idx);
    }
    return false;
  };

  SmallVector<stablehlo::WhileOp> whileOps;
  module.walk([&](stablehlo::WhileOp op) { whileOps.push_back(op); });
  // Process bottom-up: pre-order walk visits parents first, so reversing
  // gives us innermost-first.
  for (auto whileOp : llvm::reverse(whileOps)) {
    Block &body = whileOp.getBody().front();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(body.getTerminator());
    if (!returnOp)
      continue;

    unsigned n = whileOp.getNumResults();
    if (returnOp.getNumOperands() != n || body.getNumArguments() != n)
      continue;

    auto inits = whileOp.getOperands();

    SmallVector<unsigned> deadSlots;
    SmallVector<unsigned> liveSlots;
    for (unsigned i = 0; i < n; ++i) {
      // Only consider tensor-typed slots with zero-splat init — same-type
      // comparison of scalars or non-zero-init slots pairs too aggressively
      // and breaks circuits that intentionally carry a constant value
      // through a while loop (keccak_pad's private-half sentinels).
      if (!isa<RankedTensorType>(whileOp.getResult(i).getType()))
        continue;
      if (!isZeroSplatTransitively(inits[i]))
        continue;
      if (returnOp.getOperand(i) == body.getArgument(i))
        deadSlots.push_back(i);
      else
        liveSlots.push_back(i);
    }

    // Pair a DEAD slot with a LIVE slot only when the LIVE slot's yield
    // transitively references the DEAD slot's body argument. Without that
    // link, the two slots carry semantically unrelated values that merely
    // happen to share the same type and zero-init — RAUW would corrupt the
    // DEAD reader. Canonical false-positive (without the guard): a
    // Poseidon3-style `@compute` whose enclosing scf.while threads a counter
    // (LIVE, init=0, increments) alongside an immutable capacity-init
    // (DEAD passthrough, init=0). Both surface as zero-init `tensor<!pf>`
    // scalars; the redirect would replace the call's capacity operand with
    // the post-loop counter (=N), miscompiling the call.
    //
    // The AES `@xor_3$inputs[@a]/[@b]` case stays linked because the LIVE
    // slot's yield is `xor(.a body arg, .b body arg)` — it consumes the
    // DEAD body arg directly, so the worklist finds the link.
    //
    // Buffer storage is hoisted out of the lambda so the (#dead × #live)
    // pair scan reuses the same SmallVector/SmallPtrSet allocation across
    // calls instead of re-initializing inline storage each time. The
    // `findAncestorOpInBlock` guard skips operand walks for defining ops
    // outside the body block (e.g. function arguments, pre-while
    // constants): SSA dominance prevents those values from reaching
    // `targetArg`, so traversing their operand chains can only waste work.
    Block *bodyBlock = &body;
    llvm::SmallPtrSet<Value, 16> visited;
    SmallVector<Value, 8> worklist;
    auto yieldReferencesArg = [&](Value yieldVal,
                                  BlockArgument targetArg) -> bool {
      visited.clear();
      worklist.clear();
      worklist.push_back(yieldVal);
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        if (!visited.insert(v).second)
          continue;
        if (v == targetArg)
          return true;
        if (Operation *defOp = v.getDefiningOp()) {
          if (bodyBlock->findAncestorOpInBlock(*defOp))
            for (Value operand : defOp->getOperands())
              worklist.push_back(operand);
        }
      }
      return false;
    };

    llvm::SmallSet<unsigned, 8> usedLive;
    for (unsigned dead : deadSlots) {
      if (whileOp.getResult(dead).use_empty())
        continue;
      Type deadTy = whileOp.getResult(dead).getType();
      BlockArgument deadBodyArg = body.getArgument(dead);
      for (unsigned live : liveSlots) {
        if (usedLive.contains(live))
          continue;
        if (whileOp.getResult(live).getType() != deadTy)
          continue;
        if (!yieldReferencesArg(returnOp.getOperand(live), deadBodyArg))
          continue;
        whileOp.getResult(dead).replaceAllUsesWith(whileOp.getResult(live));
        usedLive.insert(live);
        break;
      }
    }
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
}

/// Convert struct.writem, array.write, and array.insert from mutable to SSA
/// form at function-body scope. Each mutation gets a result, and subsequent
/// uses of the target value are updated to use the latest write result.
///
/// `array.insert` coverage is what threads function-body 2D-constant matrix
/// initialization (Poseidon MDS, AES SBox-style tables, any inline 2D literal
/// emitted by circom's `Expression::ArrayInLine` at gen_context.rs:2566 —
/// `array.new <Empty>` + per-row `array.insert`). Without it the inserts are
/// silently dropped: `ArrayInsertPattern::replaceWithDUS` creates a fresh
/// `stablehlo.dynamic_update_slice` for void inserts, but downstream uses of
/// the dest array still reference the original `array.new` result (an empty
/// `dense<0>`), the new DUS has no consumers, and DCE drops it. The
/// canonical victim was Webb `@Mix_69_compute` returning all-zero for any
/// input.
void convertWritemToSSA(ModuleOp module) {
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Block *block) {
      llvm::MapVector<Value, Value> latestValue;

      for (Operation &op : llvm::make_early_inc_range(block->getOperations())) {
        // Rewire operands to latest SSA values
        for (auto &operand : op.getOpOperands()) {
          auto it = latestValue.find(operand.get());
          if (it != latestValue.end())
            operand.set(it->second);
        }

        StringRef opName = op.getName().getStringRef();
        if (opName != "struct.writem" && opName != "array.write" &&
            opName != "array.insert")
          continue;
        // Skip if already converted to SSA (has result type)
        if (op.getNumResults() > 0)
          continue;
        // Skip writes whose target type can't lower to a tensor carry —
        // pod-element arrays are dispatch bookkeeping (handled by `pod.*`
        // patterns). `isPromotableCarryType` is the canonical predicate.
        if (op.getNumOperands() > 0 &&
            !isPromotableCarryType(op.getOperand(0).getType()))
          continue;
        // Skip writes inside scf control flow (for/while/if) — these use
        // mutable LLZK semantics that the per-block walker here cannot
        // honor (the new SSA value would be local to the inner block and
        // never thread out as an scf.yield, leaving the write orphaned and
        // silently dropped during DCE). promoteArraysToWhileCarry +
        // convertWhileBodyArgsToSSA handle the SSA-ification through carries
        // for array.write, struct.writem, and array.insert (the latter via
        // the `includeInsertExtract=true` flag on convertWhileBodyArgsToSSA).
        // struct.writem coverage is what fixes the MiMC7 `@out`-buried-in-
        // scf.if bug — Bug 1 in
        // memory/maci-3-blocked-lowering-bugs-followup.md.
        {
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
                           func::FuncDialect, wla::WLADialect>();
    // SCF structural type conversion is added after patterns are created
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    for (StringRef d : {"struct", "function", "constrain", "felt", "component",
                        "bool", "llzk", "cast", "poly"})
      target.addIllegalDialect(d);
    // Array ops on pod-element arrays are dynamically legal (count/dispatch
    // bookkeeping, not convertible to StableHLO). Regular felt-element array
    // ops are illegal (converted by ArrayPatterns). Exception: array.new
    // with a pod element type is illegal so ArrayNewPattern fires — the type
    // converter erases the pod element to felt, producing a zero-init tensor
    // of the converted shape. SSC has already cleaned any pod-side uses, so
    // the array.new's only consumer is an unrealized_conversion_cast that
    // becomes redundant once the new is rewritten to stablehlo.constant.
    target.addDynamicallyLegalDialect(
        [](Operation *op) -> bool {
          // array.new outside @constrain is always handled by the pattern;
          // pod element type doesn't disqualify it from conversion.
          if (op->getName().getStringRef() == "array.new") {
            auto *parent = op->getParentOp();
            while (parent) {
              if (parent->getName().getStringRef() == "function.def") {
                auto sym = parent->getAttrOfType<StringAttr>("sym_name");
                return sym && sym.getValue() == "constrain";
              }
              parent = parent->getParentOp();
            }
            return false;
          }
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
          // All pod ops are dynamically legal. After SimplifySubComponents
          // extracts function.call and resolves pod.read @comp, remaining
          // pod ops (both input and dispatch) are dead code that gets
          // cleaned up in post-passes.
          if (op->getName().getStringRef() == "pod.new" ||
              op->getName().getStringRef() == "pod.read" ||
              op->getName().getStringRef() == "pod.write") {
            return true;
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
    populateStructToStablehloPatterns(typeConverter, patterns, target,
                                      flagOrphanZeroWrites);
    populateArrayToStablehloPatterns(typeConverter, patterns, target);
    populateFunctionToFuncPatterns(typeConverter, patterns, target);
    populateRemovalPatterns(typeConverter, patterns, target);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    // SCF structural type conversion: automatically converts scf.while/if/for
    // result types and block argument types using the type converter.
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    // SSC owns the LLZK v2 prerequisites (input-pod cleanup, while-carry
    // inlining, template shell removal) so external CLI runs and this
    // internal run produce identical post-template IR.
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

    // Pre-passes: transform LLZK IR before dialect conversion.
    //
    // Three Webb-Poseidon structural patches run before the carrier-promotion
    // and SSA-ification passes. Each filters narrowly on the Webb hoisted-Ark
    // / dead-K=0 / partial-round-Mix IR shape; iden3 + circomlib Poseidon and
    // every other circuit family are no-ops by construction (filter rejects).
    //   1. replaceHoistedArkReadmWithFreshCall — cascade arm K>=1 hoist
    //      staleness: rewires each cascade `struct.readm @hoisted_Ark[@out]`
    //      to a freshly emitted call on the post-col-write row at the readm
    //      site, so Ark sees a fully-populated row at the LAST inner iter.
    //   2. injectDeadCascadeArmIntoOutermostThen — round-0 carrier population:
    //      clones the dead K=0 cascade arm into the OUTERMOST scf.if THEN so
    //      the K=0 Ark output reaches its carrier (SSC drops this writer for
    //      lack of an scf.execute_region ancestor in the THEN branch).
    //   3. injectSigmaIntoPartialRoundMixInput — partial-round Mix col 0:
    //      injects the Sigma scalar into col 0 of the partial-round Mix input
    //      row, fixing a per-round zeroing that came from the partial-round
    //      scf.while's slot 1 init being a dense<0> carrier.
    // Steps 1+2 run before promoteArraysToWhileCarry. Step 3 must run AFTER
    // it because the matched 3-result scf.while only exists post-promotion;
    // the injected array.write is SSA-ified by the following
    // convertWhileBodyArgsToSSA.
    registerStructFieldOffsets(module, typeConverter);
    convertAllFunctions(module, typeConverter, context);
    replaceHoistedArkReadmWithFreshCall(module);
    injectDeadCascadeArmIntoOutermostThen(module);
    promoteArraysToWhileCarry(module);
    injectSigmaIntoPartialRoundMixInput(module);
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

    // Pre-pass: fold `arith.cmpi` on two integer/index constants and the
    // `scf.if` op that consumes the result. Circom emits a constant
    // `min(N1, N2)` for trip-count guards as
    //     %c1 = arith.constant N1 : index
    //     %c2 = arith.constant N2 : index
    //     %p  = arith.cmpi ult, %c1, %c2 : index
    //     %r  = scf.if %p -> (index) { yield %c1 } else { yield %c2 }
    // (PointCompress's `long_div_*` uses this for `min(100, 200)`.)
    // Without folding, the scf.if reaches the stablehlo.select post-pass
    // below with `tensor<index>` operands — a type SelectOp does not
    // accept — and verification fails.
    {
      SmallVector<arith::CmpIOp> cmpiToFold;
      module.walk([&](arith::CmpIOp op) {
        if (op.getLhs().getDefiningOp<arith::ConstantOp>() &&
            op.getRhs().getDefiningOp<arith::ConstantOp>())
          cmpiToFold.push_back(op);
      });
      for (auto cmpi : cmpiToFold) {
        auto lhsAttr = dyn_cast<IntegerAttr>(
            cmpi.getLhs().getDefiningOp<arith::ConstantOp>().getValue());
        auto rhsAttr = dyn_cast<IntegerAttr>(
            cmpi.getRhs().getDefiningOp<arith::ConstantOp>().getValue());
        if (!lhsAttr || !rhsAttr)
          continue;
        const APInt &l = lhsAttr.getValue();
        const APInt &r = rhsAttr.getValue();
        // arith.cmpi verifier requires LHS/RHS operand types to match, and
        // IntegerAttr verifier requires APInt width to equal the integer
        // type's width — so widths are always equal here. Asserting the
        // invariant documents it without silently miscomputing on the
        // verifier-impossible mismatch case (sign-extending an i1 `true`
        // to wider yields UINT_MAX, which would flip unsigned predicates).
        assert(l.getBitWidth() == r.getBitWidth() &&
               "arith.cmpi operands have mismatched APInt widths");
        bool result;
        switch (cmpi.getPredicate()) {
        case arith::CmpIPredicate::eq:
          result = l == r;
          break;
        case arith::CmpIPredicate::ne:
          result = l != r;
          break;
        case arith::CmpIPredicate::slt:
          result = l.slt(r);
          break;
        case arith::CmpIPredicate::sle:
          result = l.sle(r);
          break;
        case arith::CmpIPredicate::sgt:
          result = l.sgt(r);
          break;
        case arith::CmpIPredicate::sge:
          result = l.sge(r);
          break;
        case arith::CmpIPredicate::ult:
          result = l.ult(r);
          break;
        case arith::CmpIPredicate::ule:
          result = l.ule(r);
          break;
        case arith::CmpIPredicate::ugt:
          result = l.ugt(r);
          break;
        case arith::CmpIPredicate::uge:
          result = l.uge(r);
          break;
        default:
          llvm_unreachable("unhandled CmpIPredicate");
        }
        OpBuilder b(cmpi);
        auto cstBool =
            b.create<arith::ConstantOp>(cmpi.getLoc(), b.getBoolAttr(result));
        cmpi.getResult().replaceAllUsesWith(cstBool);
        cmpi.erase();
      }

      SmallVector<scf::IfOp> ifToFold;
      module.walk([&](scf::IfOp ifOp) {
        // Void scf.if is handled by the existing post-pass below (it hoists
        // func.calls then erases). Folding it here is both redundant and
        // unsafe: a void scf.if may legally omit the else region, and the
        // chosen-block lookup below would dereference an empty Region.
        if (ifOp.getNumResults() == 0)
          return;
        auto cstOp = ifOp.getCondition().getDefiningOp<arith::ConstantOp>();
        if (!cstOp)
          return;
        auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue());
        if (!intAttr || !intAttr.getType().isInteger(1))
          return;
        ifToFold.push_back(ifOp);
      });
      for (auto ifOp : ifToFold) {
        auto cstOp = ifOp.getCondition().getDefiningOp<arith::ConstantOp>();
        auto intAttr = cast<IntegerAttr>(cstOp.getValue());
        bool condVal = intAttr.getValue().getBoolValue();
        Block &chosen = condVal ? ifOp.getThenRegion().front()
                                : ifOp.getElseRegion().front();
        auto yield = cast<scf::YieldOp>(chosen.getTerminator());
        // The un-chosen branch's ops are unreachable under scf.if semantics,
        // so dropping them with `ifOp.erase()` below is correct even if they
        // contain side effects.
        SmallVector<Value> yielded(yield.getOperands());
        for (Operation &op : llvm::make_early_inc_range(chosen)) {
          if (&op == yield.getOperation())
            continue;
          op.moveBefore(ifOp);
        }
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
          ifOp.getResult(i).replaceAllUsesWith(yielded[i]);
        ifOp.erase();
      }
    }

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

    // Post-pass: scrub residual `arith.*` fan-out emitted by `array.write
    // %carrier[%dyn_idx] = %v` scaffolding inside scf.while bodies when
    // the dynamic index is felt-typed. After convertScfWhile..., the
    // fan-out sits inside stablehlo.while bodies and looks like
    //   %idx = unrealized_conversion_cast %iter_i32 : tensor<i32> to index
    //   %true = arith.constant true
    //   %true_t = unrealized_conversion_cast %true : i1 to tensor<i1>
    //   %cN = arith.constant N : index
    //   %cmp = arith.cmpi eq, %idx, %cN : index
    //   %cmp_t = unrealized_conversion_cast %cmp : i1 to tensor<i1>
    //   %flag = arith.andi %cmp_t, %true_t : tensor<i1>
    //   ... %sel = stablehlo.select %flag, %v, %prev
    // The conversion target keeps arith::ArithDialect legal so partial
    // conversion intentionally leaves these behind, but the ZKX HLO
    // translator rejects every surviving arith.* op. Rewrite cmpi onto
    // the underlying tensor<i32> via stablehlo.compare EQ, and convert
    // the `arith.constant true|false : i1` carrier feeding the
    // `i1 → tensor<i1>` cast to a `stablehlo.constant`. The existing
    // arith.andi → stablehlo.and rewrite below then picks up the residue
    // (its operands now trace straight to tensor<i1> producers, satisfying
    // the lookThroughCast tensor guard at the next pass).
    {
      auto i32TensorType =
          RankedTensorType::get({}, IntegerType::get(context, 32));
      auto i1TensorType =
          RankedTensorType::get({}, IntegerType::get(context, 1));
      auto getI32Source = [](Value v) -> Value {
        auto cast = v.getDefiningOp<UnrealizedConversionCastOp>();
        if (!cast || cast.getInputs().size() != 1)
          return nullptr;
        Value src = cast.getInputs()[0];
        auto tty = dyn_cast<RankedTensorType>(src.getType());
        if (tty && tty.getElementType().isInteger(32) && tty.getRank() == 0)
          return src;
        return nullptr;
      };

      // Step A: arith.cmpi eq : index → stablehlo.compare EQ on tensor<i32>.
      SmallVector<arith::CmpIOp> cmpiOps;
      module.walk([&](arith::CmpIOp op) {
        if (op.getPredicate() != arith::CmpIPredicate::eq)
          return;
        if (!op.getLhs().getType().isIndex() ||
            !op.getRhs().getType().isIndex())
          return;
        cmpiOps.push_back(op);
      });
      for (auto cmpi : cmpiOps) {
        OpBuilder b(cmpi);
        auto materializeOperand = [&](Value v) -> Value {
          if (Value src = getI32Source(v))
            return src;
          if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
            auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
            if (!intAttr)
              return nullptr;
            return b.create<stablehlo::ConstantOp>(
                cmpi.getLoc(), i32TensorType,
                DenseElementsAttr::get(i32TensorType,
                                       b.getI32IntegerAttr(static_cast<int32_t>(
                                           intAttr.getInt()))));
          }
          return nullptr;
        };
        Value lhs = materializeOperand(cmpi.getLhs());
        Value rhs = materializeOperand(cmpi.getRhs());
        if (!lhs || !rhs)
          continue;
        auto cmpResult = b.create<stablehlo::CompareOp>(
            cmpi.getLoc(), i1TensorType, lhs, rhs,
            stablehlo::ComparisonDirectionAttr::get(
                cmpi.getContext(), stablehlo::ComparisonDirection::EQ));
        // Re-route consumers that cast the i1 scalar to tensor<i1>.
        SmallVector<UnrealizedConversionCastOp> dyingCasts;
        for (auto *user : cmpi.getResult().getUsers())
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
            if (cast.getNumResults() == 1 &&
                cast.getResult(0).getType() == i1TensorType)
              dyingCasts.push_back(cast);
        for (auto cast : dyingCasts) {
          cast.getResult(0).replaceAllUsesWith(cmpResult);
          cast.erase();
        }
        if (cmpi.getResult().use_empty())
          cmpi.erase();
      }

      // Step B: arith.constant true|false : i1 → stablehlo.constant
      // tensor<i1>, RAUW'ing its `i1 → tensor<i1>` cast consumers. The
      // arith.constant itself is left in place if other (e.g. scf.if)
      // consumers remain; otherwise the trailing DCE sweep clears it.
      // Single replacement constant per source: inserted directly after
      // `c` so it dominates every cast consumer regardless of use-list
      // ordering.
      SmallVector<arith::ConstantOp> i1Constants;
      module.walk([&](arith::ConstantOp op) {
        if (op.getType().isInteger(1))
          i1Constants.push_back(op);
      });
      for (auto c : i1Constants) {
        auto intAttr = dyn_cast<IntegerAttr>(c.getValue());
        if (!intAttr)
          continue;
        bool boolVal = intAttr.getInt() != 0;
        SmallVector<UnrealizedConversionCastOp> dyingCasts;
        for (auto *user : c.getResult().getUsers()) {
          auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
          if (!cast || cast.getNumResults() != 1)
            continue;
          auto rty = dyn_cast<RankedTensorType>(cast.getResult(0).getType());
          if (rty && rty.getElementType().isInteger(1) && rty.getRank() == 0)
            dyingCasts.push_back(cast);
        }
        if (dyingCasts.empty())
          continue;
        OpBuilder b(c);
        b.setInsertionPointAfter(c);
        auto cst = b.create<stablehlo::ConstantOp>(
            c.getLoc(), i1TensorType,
            DenseElementsAttr::get(i1TensorType, boolVal));
        for (auto cast : dyingCasts) {
          cast.getResult(0).replaceAllUsesWith(cst);
          cast.erase();
        }
        if (c.getResult().use_empty())
          c.erase();
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

    // Post-pass: inline survivor scf.execute_region wrappers whose body
    // is a single block ending in scf.yield. LLZK's `<--` returning a
    // felt (e.g. the 67-way Ark-constant lookup in Poseidon-using chips
    // like webb's vAnchor) emits a felt-array dispatch cascade wrapped
    // in scf.execute_region. Main conversion lowers the inner ops to
    // stablehlo but no pattern dissolves the wrapper; survivors then
    // block stablehlo_runner's module load with "Dialect `scf' not
    // found for custom op 'scf.execute_region'". Inlining a single-
    // block single-yield body is semantically a no-op: the region
    // executes once unconditionally, and after splicing the inner ops
    // run in the parent block at the same position. The resulting
    // felt→tensor round-trip cast pair (yield-side cast + consumer-side
    // cast) is cleaned up by the ReconcileUnrealizedCasts pass below.
    //
    // PostOrder walk so nested wrappers inline inner-first. The
    // single-block guard is load-bearing: scf.execute_region's region
    // is `AnyRegion` and multi-block bodies are legal (branching to
    // distinct terminators via cf.br), but inlining them would require
    // hoisting cf.* control flow into the parent block, which is out
    // of scope here.
    module.walk<WalkOrder::PostOrder>([&](scf::ExecuteRegionOp er) {
      if (!er.getRegion().hasOneBlock())
        return;
      Block &body = er.getRegion().front();
      auto yield = cast<scf::YieldOp>(body.getTerminator());
      for (auto [res, val] : llvm::zip(er.getResults(), yield.getOperands()))
        res.replaceAllUsesWith(val);
      Block *parent = er->getBlock();
      parent->getOperations().splice(er->getIterator(), body.getOperations(),
                                     body.begin(), yield->getIterator());
      er.erase();
    });

    // Reconcile leftover round-trip unrealized_conversion_cast pairs
    // (A → X → A). These appear when a pattern reads an
    // scf-structurally-converted block arg through cast.toindex; without
    // reconciliation the felt-typed cast intermediates survive and
    // downstream tools that don't load the felt dialect cannot parse the
    // result. Skip when no live cast remains — most circuits without
    // while-bodied iter-args fall through here.
    {
      bool hasLiveCast = false;
      module.walk([&](UnrealizedConversionCastOp) {
        hasLiveCast = true;
        return WalkResult::interrupt();
      });
      if (hasLiveCast) {
        OpPassManager pm("builtin.module");
        pm.addPass(createReconcileUnrealizedCastsPass());
        if (failed(runPipeline(pm, module))) {
          signalPassFailure();
          return;
        }
      }
    }

    // Clean up all dead non-stablehlo ops (unrealized_cast, arith, tensor,
    // array, pod, scf, struct, builtin dead casts). `wla.layout` (no
    // results, `Pure`) flows through this loop too. The anchor pass
    // (`--witness-layout-anchor`, TRACK 3) currently over-emits internal
    // entries that the lowering elides from `@main`'s DUS chain (struct-
    // typed sub-component members in particular), so connecting the
    // anchor → verify chain (preserving `wla.layout` past this DCE +
    // erasing in `--verify-witness-layout`) is gated on a filter
    // refinement that aligns the anchor's emit with the lowering's
    // actual per-chunk emission.
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
        // `use_empty` is trivially true for zero-result ops (terminators
        // like scf.yield / scf.condition / func.return). Skipping the
        // terminator trait keeps the block well-formed — without the
        // guard, a survivor scf.execute_region whose body ends in
        // scf.yield (e.g. the bn-typed felt-array dispatch cascade in
        // chips with `<--` returning a felt array) loses its terminator
        // here and the next verifier fires "empty block: expect at
        // least a terminator". stablehlo-namespace terminators were
        // already filtered above, so this guard only fires for
        // surviving non-stablehlo terminators.
        if (op->use_empty() && !op->hasTrait<OpTrait::IsTerminator>()) {
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

    // Post-pass: collapse redundant carrier pairs left when
    // flattenPodArrayWhileCarry and promoteArraysToWhileCarry independently
    // create per-field carriers for the same logical signal.
    collapseRedundantWhileCarrierPairs(module);

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
