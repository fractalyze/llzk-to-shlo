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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayMaterialize.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayWhileCarry.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodDispatchPhases.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructOfPodsConversion.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_SIMPLIFYSUBCOMPONENTS
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h.inc"

// ===----------------------------------------------------------------------===
// Shared helper functions (external linkage — declared in
// SimplifySubComponentsInternal.h, used by both this file and
// PodDispatchPhases.cpp).
// ===----------------------------------------------------------------------===

/// Check if all results of an operation are unused.
bool isAllResultsUnused(Operation &op) {
  for (auto result : op.getResults())
    if (!result.use_empty())
      return false;
  return true;
}

/// True iff `v` is defined inside one of `op`'s regions (either as an
/// OpResult of a nested op, or as a BlockArgument of a nested block).
/// `Value::getParentRegion()` returns the enclosing region for both
/// flavors; `Region::isAncestor` handles the rest.
bool isValueDefinedInside(Value v, Operation &op) {
  Region *def = v.getParentRegion();
  if (!def)
    return false;
  for (Region &r : op.getRegions())
    if (r.isAncestor(def))
      return true;
  return false;
}

/// Create an llzk.nondet operation producing an uninitialized value.
Value createNondet(OpBuilder &builder, Location loc, Type type) {
  OperationState state(loc, "llzk.nondet");
  state.addTypes({type});
  return builder.create(state)->getResult(0);
}

// `cloneDefiningOpBefore` has external linkage — it is declared in
// SimplifySubComponentsInternal.h and called by PodDispatchPhases.cpp.
// `isSafeToCloneBefore` is its file-private helper (`static`): no other
// translation unit needs it, so it keeps internal linkage.

/// True iff cloning `def` before `guardOp` is safe — i.e. the clone reads
/// the same value its original location would read. Two categories qualify:
/// (1) ops without memory effects (`Pure` trait or empty
/// `MemoryEffectOpInterface`), and (2) LLZK's read-only array/pod access ops
/// (`array.read` / `array.extract` / `array.len` / `pod.read`). These read
/// mutable LLZK records so they are NOT `Pure`, but hoisting them out of
/// `guardOp` is nevertheless safe because the clone reads the record operand
/// BEFORE `guardOp` executes — any writes that `guardOp`'s body would perform
/// haven't happened yet at the clone's new position. `pod.read` matters for
/// the webb Poseidon shape: writer-side inputs come from a struct-of-pods
/// dispatch chain (`pod.read %cell[@in]` ← `array.read %carrier[%cK]`) that
/// `materializeStructOfPodsCompField` must clone past the count-guard scf.if
/// to materialize 68-arm Ark cascade writers — without the pod.read entry
/// every writer's hoist clone fails and the carrier stays unwritten.
static bool isSafeToCloneBefore(Operation *def) {
  if (mlir::isMemoryEffectFree(def))
    return true;
  StringRef name = def->getName().getStringRef();
  return name == "array.read" || name == "array.extract" ||
         name == "array.len" || name == "pod.read";
}

/// Recursively clone the defining-op chain of `v` BEFORE `insertBefore` so
/// the clone's result dominates `insertBefore`. Returns the cloned value,
/// or a null `Value()` on failure (chain reaches a block argument inside
/// `guardOp`, an unsafe op, or `depth` exhausted).
///
/// `guardOp` is the scope we are hoisting OUT of (typically the enclosing
/// `scf.if`). Values defined outside `guardOp` are returned as-is and
/// terminate the recursion.
///
/// `cloneCache` dedupes when multiple args share a sub-chain. The depth
/// cap is a convergence guard against adversarial chains.
Value cloneDefiningOpBefore(Value v, Operation *insertBefore,
                            Operation &guardOp,
                            llvm::DenseMap<Value, Value> &cloneCache,
                            unsigned depth) {
  if (!isValueDefinedInside(v, guardOp))
    return v;
  if (depth == 0)
    return Value();
  if (auto it = cloneCache.find(v); it != cloneCache.end())
    return it->second;
  Operation *def = v.getDefiningOp();
  if (!def)
    return Value(); // block-argument inside guardOp — not clonable
  if (!isSafeToCloneBefore(def))
    return Value();

  SmallVector<Value> newOperands;
  newOperands.reserve(def->getNumOperands());
  for (Value o : def->getOperands()) {
    Value cloned =
        cloneDefiningOpBefore(o, insertBefore, guardOp, cloneCache, depth - 1);
    if (!cloned)
      return Value();
    newOperands.push_back(cloned);
  }

  OpBuilder builder(insertBefore);
  Operation *cloned = def->clone();
  for (auto [i, o] : llvm::enumerate(newOperands))
    cloned->setOperand(i, o);
  builder.insert(cloned);

  unsigned resultIdx = 0;
  for (unsigned i = 0, ni = def->getNumResults(); i < ni; ++i) {
    if (def->getResult(i) == v) {
      resultIdx = i;
      break;
    }
  }
  Value result = cloned->getResult(resultIdx);
  cloneCache[v] = result;
  return result;
}

/// Walk up from `funcBlock` past any nested `builtin.module` wrappers to
/// the top-level module. LLZK v2's `createEmptyTemplateRemoval` wraps each
/// component in its own `builtin.module`, so a SymbolTable lookup that
/// must reach a sibling component must start at the outermost module.
ModuleOp getTopLevelModule(Block &funcBlock) {
  ModuleOp moduleOp = funcBlock.getParentOp()->getParentOfType<ModuleOp>();
  while (moduleOp) {
    ModuleOp outer = moduleOp->getParentOfType<ModuleOp>();
    if (!outer)
      break;
    moduleOp = outer;
  }
  return moduleOp;
}

/// Build `array<destDims + innerDims x leafFelt>` when `innerFeltTy` is a
/// felt array (`!array<K x !felt>`), or `array<destDims x innerFeltTy>` when
/// it is a scalar `!felt`. Used by writers + readers materializing a
/// dispatch-sized parallel destination for a sub-component's `@out` member.
llzk::array::ArrayType
combineDispatchAndInnerFeltDims(Type innerFeltTy, ArrayRef<int64_t> destDims) {
  if (auto innerArr = dyn_cast<llzk::array::ArrayType>(innerFeltTy)) {
    auto innerDims = getArrayDimensions(innerArr);
    SmallVector<int64_t> combined(destDims.begin(), destDims.end());
    combined.append(innerDims.begin(), innerDims.end());
    return llzk::array::ArrayType::get(innerArr.getElementType(), combined);
  }
  return llzk::array::ArrayType::get(innerFeltTy, destDims);
}

/// True for types that participate in pod-array per-field flattening:
/// `!felt.type` or `!array.type<... x !felt.type>`.
bool isFlattenableFelt(Type ty) {
  if (ty.getDialect().getNamespace() == "felt")
    return true;
  if (auto at = dyn_cast<llzk::array::ArrayType>(ty))
    return at.getElementType().getDialect().getNamespace() == "felt";
  return false;
}

/// Index operands of an LLZK `array.read` / `array.write` op. The first
/// operand is the array; everything after is the index list.
SmallVector<Value> arrayAccessIndices(Operation *arrayAccess) {
  return llvm::to_vector(llvm::drop_begin(arrayAccess->getOperands()));
}

namespace {

// ===----------------------------------------------------------------------===
// Helper functions (anonymous namespace — file-private)
// ===----------------------------------------------------------------------===

/// Resolve pod.read @comp from array-sourced pods.
/// After function.call extraction, the chain:
///   array.read %arr[%idx] → pod.read @comp → struct.readm @out
/// should use the extracted function.call result directly.
/// This handles the count/dispatch array pattern where @comp holds the
/// computed struct result.
/// Extract function.call from a void dispatch scf.if by moving the call
/// (and its in-block dependencies) before the scf.if. The call is executed
/// unconditionally; for iterations where the dispatch doesn't fire, the result
/// is unused. This is correct for witness generation since functions are pure.
///
/// Returns the extracted call result, or null if not a dispatch pattern.
Value extractCallFromDispatch(scf::IfOp ifOp) {
  // Only handle void scf.if (dispatch bookkeeping).
  if (ifOp.getNumResults() > 0)
    return {};

  // Find function.call in then-branch.
  Operation *callOp = nullptr;
  ifOp.getThenRegion().walk([&](Operation *op) {
    if (op->getName().getStringRef() == "function.call" &&
        op->getNumResults() > 0)
      callOp = op;
  });
  if (!callOp)
    return {};

  // Collect the call and all its dependencies defined inside the scf.if.
  Block &thenBlock = ifOp.getThenRegion().front();
  llvm::DenseSet<Operation *> needed;

  std::function<void(Value)> collectDeps = [&](Value v) {
    auto *def = v.getDefiningOp();
    if (!def || needed.count(def))
      return;
    // Only collect ops defined inside the scf.if's regions.
    if (!ifOp->isAncestor(def))
      return;
    needed.insert(def);
    for (Value operand : def->getOperands())
      collectDeps(operand);
  };
  for (Value operand : callOp->getOperands())
    collectDeps(operand);
  needed.insert(callOp);

  // Move needed ops before the scf.if (preserving block order).
  // Collect from the then-block only (nested ops are inside their own regions).
  for (Operation &op : llvm::make_early_inc_range(thenBlock)) {
    if (needed.count(&op))
      op.moveBefore(ifOp);
  }

  return callOp->getResult(0);
}

bool resolveArrayPodCompReads(Block &block) {
  bool changed = false;

  // Strategy: find the function.call results that provide pod @comp values.
  // 1. Top-level calls dominate directly.
  // 2. Calls inside void scf.if dispatch: extract and hoist before scf.if.
  // Build a type → call result map to match each pod.read @comp by type.
  llvm::DenseMap<Type, Value> callResultByType;
  for (Operation &op : block) {
    if (op.getName().getStringRef() == "function.call" &&
        op.getNumResults() > 0)
      callResultByType[op.getResult(0).getType()] = op.getResult(0);
  }

  // Extract calls from void dispatch scf.if blocks.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (!ifOp)
      continue;
    if (Value result = extractCallFromDispatch(ifOp)) {
      callResultByType[result.getType()] = result;
      changed = true;
    }
  }

  if (callResultByType.empty())
    return false;

  // Find pod.read @comp chains: array.read → pod.read @comp → uses
  // Match each pod.read @comp by its result type to the correct call.
  SmallVector<Operation *> toErase;
  for (Operation &op : block) {
    if (op.getName().getStringRef() != "array.read" || op.getNumResults() == 0)
      continue;
    // Check element type is pod
    if (op.getResult(0).getType().getDialect().getNamespace() != "pod")
      continue;

    // Check if this array.read result is used by pod.read @comp
    for (OpOperand &use : op.getResult(0).getUses()) {
      Operation *user = use.getOwner();
      if (user->getName().getStringRef() != "pod.read" ||
          user->getNumResults() == 0)
        continue;
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!rn || rn.getValue() != "comp")
        continue;
      // Find matching call result by type.
      Type compType = user->getResult(0).getType();
      auto it = callResultByType.find(compType);
      if (it == callResultByType.end())
        continue;
      user->getResult(0).replaceAllUsesWith(it->second);
      toErase.push_back(user);
      changed = true;
    }
    // If array.read has no more uses, mark for erase.
    if (op.use_empty())
      toErase.push_back(&op);
  }
  for (auto *op : toErase)
    op->erase();

  return changed;
}

/// Fold `array.read → pod.read @<count|comp|in>` chains the per-pod
/// tracker can't resolve when the pod source is an array-of-pods.
///
/// Witness-gen relies on `@constrain` re-deriving these values, so
/// substitution is sound:
///   `@count` → `arith.constant 0 : index` (the surrounding
///       cmpi/scf.if dispatch scaffold is structurally dead once
///       `resolveArrayPodCompReads` has hoisted the function.call;
///       the underflowed `subi` keeps cmpi false so DCE can collapse).
///   `@comp`, `@in` → `llzk.nondet` of the result type.
///
/// `@in` reads are skipped when the pod.read still has a live
/// `function.call` operand consumer (the materializer's hoisted
/// dispatch call). Splicer-style bodies hold multiple un-flattened
/// input pod-arrays in flight across outer-loop iterations; without
/// this gate the helper nondet's a load-bearing operand before
/// `flattenPodArrayWhileCarry` can rewire it to `array.extract` on
/// the per-field carry, and the lowered call receives const-zero
/// inputs.
///
/// Non-whitelisted fields are left alone — blanket nondet would risk
/// silently breaking a real value flow on a circuit shape we haven't
/// analyzed.
bool rewriteArrayPodCountCompInReads(Block &block) {
  bool changed = false;
  SmallVector<Operation *> toErase;
  for (Operation &op : block) {
    if (op.getName().getStringRef() != "pod.read" || op.getNumOperands() == 0 ||
        op.getNumResults() == 0)
      continue;
    auto rn = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!rn)
      continue;
    Value src = op.getOperand(0);
    Operation *def = src.getDefiningOp();
    if (!def || def->getName().getStringRef() != "array.read")
      continue;
    if (src.getType().getDialect().getNamespace() != "pod")
      continue;

    StringRef field = rn.getValue();
    if (field != "count" && field != "comp" && field != "in")
      continue;

    if (field == "in") {
      bool hasCallUse = false;
      for (Operation *user : op.getResult(0).getUsers()) {
        if (user->getName().getStringRef() == "function.call") {
          hasCallUse = true;
          break;
        }
      }
      if (hasCallUse)
        continue;
    }

    OpBuilder builder(&op);
    Type rty = op.getResult(0).getType();
    Value replacement;
    if (field == "count" && rty.isIndex()) {
      OperationState state(op.getLoc(), "arith.constant");
      state.addAttribute("value", builder.getIndexAttr(0));
      state.addTypes({rty});
      replacement = builder.create(state)->getResult(0);
    } else {
      replacement = createNondet(builder, op.getLoc(), rty);
    }
    op.getResult(0).replaceAllUsesWith(replacement);
    toErase.push_back(&op);
    changed = true;
  }
  for (Operation *o : toErase)
    o->erase();
  return changed;
}

/// Rewrite `!struct.type<@X<[]>>` (empty params) to `!struct.type<@X>`
/// (no params), recursing into array element types.
Type stripEmptyStructParamsFromType(Type ty) {
  if (auto structTy = dyn_cast<llzk::component::StructType>(ty)) {
    auto params = structTy.getParams();
    if (params && params.empty())
      return llzk::component::StructType::get(structTy.getNameRef(),
                                              mlir::ArrayAttr());
    return ty;
  }
  if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty)) {
    Type innerStripped = stripEmptyStructParamsFromType(arrTy.getElementType());
    if (innerStripped == arrTy.getElementType())
      return ty;
    return llzk::array::ArrayType::get(innerStripped,
                                       getArrayDimensions(arrTy));
  }
  return ty;
}

/// Hoist a same-named child out of `builtin.module @X { function.def @X }`
/// (or `struct.def @X`) shells, erase the wrapper, then rewrite all
/// `@X::@X[::@method]` symbol refs to `@X[::@method]`.
///
/// project-llzk/circom PR #378 wraps every emitted `function.def` /
/// `struct.def` in a same-named `poly.template @X { function.def @X }`
/// to carry polymorphic typing upstream. After we run
/// `createEmptyTemplateRemoval`, those wrappers become
/// `builtin.module @X { function.def @X }`. The inner symbol now lives
/// as a sibling of the wrapping module under the same name in the
/// outer module's symbol table; any subsequent pass that walks the
/// parent's `SymbolTable` (LlzkToStablehlo conversion in particular)
/// trips with `redefinition of symbol named '<X>'`.
///
/// Refs in the IR look like `@X::@X` (struct/function self-ref) or
/// `@X::@X::@compute` (inner method ref). After hoist, the qualified
/// path collapses one level: the outer `@X` is gone, so what used to
/// resolve as `@X::@X::@compute` now resolves as `@X::@compute`.
void flattenSingleEntityWrapperModules(ModuleOp module) {
  // Collect wrappers safe to unwrap: `module @X { single child @X }` where
  // the child is a function.def or struct.def with the same name. Don't
  // touch non-wrapping submodules (e.g. nested IR in unit tests).
  llvm::DenseSet<StringRef> unwrapped;
  SmallVector<ModuleOp> wrappers;
  for (auto inner : module.getOps<ModuleOp>()) {
    StringRef innerName = inner.getSymName().value_or("");
    if (innerName.empty())
      continue;
    Block &body = inner.getBodyRegion().front();
    if (!llvm::hasSingleElement(body))
      continue;
    Operation &child = *body.begin();
    auto childSym = dyn_cast<SymbolOpInterface>(child);
    if (!childSym || childSym.getName() != innerName)
      continue;
    wrappers.push_back(inner);
    unwrapped.insert(innerName);
  }

  for (ModuleOp inner : wrappers) {
    Block &innerBlock = inner.getBodyRegion().front();
    Block &outerBlock = module.getBodyRegion().front();
    outerBlock.getOperations().splice(Block::iterator(inner),
                                      innerBlock.getOperations());
    inner.erase();
  }

  if (unwrapped.empty())
    return;

  // Rewrite `@X::@X[::@...]` → `@X[::@...]` wherever the leading two
  // SymbolRef components match an unwrapped name. References live in op
  // attributes AND in types (e.g. `!struct.type<@X::@X>`), so use
  // `AttrTypeReplacer` to recurse into both.
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolRefAttr ref) -> SymbolRefAttr {
    auto nested = ref.getNestedReferences();
    if (nested.empty())
      return ref;
    StringRef root = ref.getRootReference().getValue();
    if (!unwrapped.contains(root))
      return ref;
    StringRef firstNested = nested.front().getValue();
    if (firstNested != root)
      return ref;
    return SymbolRefAttr::get(ref.getContext(), firstNested,
                              nested.drop_front());
  });
  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

void stripEmptyStructParams(ModuleOp module) {
  // Scoped to `llzk.nondet` only: the upstream template-removal type
  // converter handles its own `OpClassesWithStructTypes` tuple; widening
  // the strip to those ops desyncs that converter and crashes its
  // legality walk.
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "llzk.nondet")
      return;
    for (Value result : op->getResults()) {
      Type t = result.getType();
      Type stripped = stripEmptyStructParamsFromType(t);
      if (stripped != t)
        result.setType(stripped);
    }
  });
}

/// Remove `@X$inputs` pod struct members and their `struct.writem`/
/// `struct.readm`/`pod.read` traffic. The `$inputs` pod materializes the
/// sub-component input values from `@compute` to `@constrain` (LLZK v2
/// pattern); for witness generation the `@constrain` side receives the
/// sub-component's input through this channel but the value itself is
/// don't-care (the sub-component's own `@constrain` re-derives or
/// constrains it independently). Replacing each `pod.read %readm[@field]`
/// with an `llzk.nondet` of the field type erases the lowering blocker.
///
/// This pre-step must run BEFORE `createEmptyTemplateRemoval` because that
/// pass's internal `applyFullConversion` has no target pattern for
/// `pod.read` and would bail out.
/// True when `member_name` ends with the `$inputs` suffix circom v2 uses
/// for sub-component input-channel struct members.
bool hasInputsMemberName(Operation *op) {
  auto member = op->getAttrOfType<FlatSymbolRefAttr>("member_name");
  return member && member.getValue().ends_with("$inputs");
}

void eliminateInputPods(ModuleOp module) {
  // Single walk collects both directions of $inputs traffic:
  // `struct.writem` (compute-side stow, always erasable) and
  // `struct.readm` (constrain-side, kept until its `pod.read` consumers
  // are rewritten below). The matching `struct.member` declaration is
  // intentionally left alone — erasing it would leave dangling refs
  // when an array-of-pods readm we skip still references the member.
  SmallVector<Operation *> writemsToErase;
  SmallVector<Operation *> constrainReads;
  module->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "struct.writem" && hasInputsMemberName(op))
      writemsToErase.push_back(op);
    else if (name == "struct.readm" && hasInputsMemberName(op))
      constrainReads.push_back(op);
  });
  for (auto *op : writemsToErase)
    op->erase();

  // DCE orphaned `pod.new` + `pod.write` chains. `pod.write` has no
  // SSA result, so the only safe trigger is "every remaining user of
  // the pod is another `pod.write`" — anything else (a `pod.read`, a
  // `function.call` arg, a surviving `struct.writem`) means the chain
  // is still load-bearing.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> deadOps;
    module->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "pod.new" || op->getNumResults() == 0)
        return;
      Value podVal = op->getResult(0);
      for (OpOperand &use : podVal.getUses())
        if (use.getOwner()->getName().getStringRef() != "pod.write")
          return;
      for (OpOperand &use : podVal.getUses())
        deadOps.push_back(use.getOwner());
      deadOps.push_back(op);
    });
    for (auto *op : deadOps) {
      op->dropAllUses();
      op->erase();
      changed = true;
    }
  }

  // For each `struct.readm @X$inputs`, swap its scalar `pod.read`
  // consumers with `llzk.nondet` of the field type. Array-of-pods
  // readms keep non-pod.read users (`array.read`/`array.extract`),
  // so their `struct.readm` stays — dropping it would leave dangling
  // operands.
  SmallVector<Operation *> podReadsToErase;
  for (auto *readmOp : constrainReads) {
    if (readmOp->getNumResults() == 0)
      continue;
    Value readmResult = readmOp->getResult(0);
    for (OpOperand &use : llvm::make_early_inc_range(readmResult.getUses())) {
      Operation *user = use.getOwner();
      if (user->getName().getStringRef() != "pod.read" ||
          user->getNumResults() == 0)
        continue;
      OpBuilder b(user);
      Value nondet =
          createNondet(b, user->getLoc(), user->getResult(0).getType());
      user->getResult(0).replaceAllUsesWith(nondet);
      podReadsToErase.push_back(user);
    }
  }
  for (auto *op : podReadsToErase)
    op->erase();

  // Array-of-input-pods variant: `struct.readm @X$inputs` returning
  // `!array<… x !pod<…>>`, used by `array.read` whose user is a
  // `pod.read @<field>`. The scalar branch above leaves the readm
  // alone so its `array.read` consumers don't dangle; that means the
  // intervening `pod.read`s survive into dialect conversion. Replace
  // each one with `llzk.nondet` of the field type — same justification
  // as the scalar case (the sub-component's own `@constrain`
  // re-derives or constrains the input value independently).
  SmallVector<Operation *> arrayPodReadsToErase;
  for (auto *readmOp : constrainReads) {
    if (readmOp->getNumResults() == 0)
      continue;
    Value readmResult = readmOp->getResult(0);
    for (OpOperand &arrUse :
         llvm::make_early_inc_range(readmResult.getUses())) {
      Operation *arrayRead = arrUse.getOwner();
      if (arrayRead->getName().getStringRef() != "array.read" ||
          arrayRead->getNumResults() == 0)
        continue;
      Value podVal = arrayRead->getResult(0);
      for (OpOperand &podUse : llvm::make_early_inc_range(podVal.getUses())) {
        Operation *podRead = podUse.getOwner();
        if (podRead->getName().getStringRef() != "pod.read" ||
            podRead->getNumResults() == 0)
          continue;
        OpBuilder b(podRead);
        Value nondet =
            createNondet(b, podRead->getLoc(), podRead->getResult(0).getType());
        podRead->getResult(0).replaceAllUsesWith(nondet);
        arrayPodReadsToErase.push_back(podRead);
      }
    }
  }
  for (auto *op : arrayPodReadsToErase)
    op->erase();

  for (auto *op : constrainReads)
    if (op->getNumResults() > 0 && op->getResult(0).use_empty())
      op->erase();
}

/// True if `type` is a `!pod.type<…>` whose record list contains no
/// `@count` field — i.e. a sub-component input pod, not a dispatch pod.
bool isInputPodType(Type type) {
  auto podTy = dyn_cast<llzk::pod::PodType>(type);
  if (!podTy)
    return false;
  for (auto rec : podTy.getRecords())
    if (rec.getName() == "count")
      return false;
  return true;
}

/// Inline single-field input pods (no `@count`) used as scf.while carry
/// to their inner field type. Must run before `createEmptyTemplateRemoval`
/// so its `applyFullConversion` doesn't see residual `pod.*` ops.
void inlineInputPodCarries(ModuleOp module) {
  SmallVector<Operation *> podNews;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "pod.new" && op->getNumResults() == 1 &&
        isInputPodType(op->getResult(0).getType()))
      podNews.push_back(op);
  });

  for (auto *podNew : podNews) {
    Value podVal = podNew->getResult(0);
    Type podType = podVal.getType();

    // Single-field pods only — multi-field needs tuple lowering.
    auto podTy = cast<llzk::pod::PodType>(podType);
    if (podTy.getRecords().size() != 1)
      continue;

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

      for (auto &use : v.getUses()) {
        Operation *user = use.getOwner();
        // Pod flows in as an scf.while init operand → block arg at the
        // matching index (for both the before and after regions) plus the
        // matching while result.
        if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
          unsigned idx = use.getOperandNumber();
          if (idx < whileOp.getNumResults())
            worklist.push_back(whileOp.getResult(idx));
          for (auto &region : whileOp->getRegions())
            if (idx < region.getNumArguments())
              worklist.push_back(region.getArgument(idx));
          continue;
        }
        // Pod flows out via the body yield / condition, back to the while
        // results and the peer region's block args.
        if (user->getName().getStringRef() == "scf.yield" ||
            user->getName().getStringRef() == "scf.condition") {
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
    // Single-field pods threaded through scf.while as a dead carry have no
    // pod.read user (circom emits the carry slot even when the body never
    // accesses it; earlier passes such as `eliminatePodDispatch`'s Phase 4
    // also DCE the pod.read/pod.write chain when structurally dead). The
    // inner type is unambiguous from the pod's record list, so fall back to
    // it instead of leaving a `pod.type` carry the downstream conversion
    // cannot legalize.
    if (!innerType)
      innerType = podTy.getRecords()[0].getType();

    for (Value v : podValues)
      v.setType(innerType);

    // A single `pod.write %pod = %value` is a user of BOTH `%pod` and
    // `%value` — if both are in `podValues` it would be added to
    // `toErase` twice and double-erased. SetVector preserves insertion
    // order so the erase walk fires in observed-uses order, matching
    // the pre-dedupe behavior for the unique-user case.
    llvm::SetVector<Operation *> toErase;
    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.read") {
          user->getResult(0).replaceAllUsesWith(v);
          toErase.insert(user);
        }
      }
    }

    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.write")
          toErase.insert(user);
      }
    }

    {
      OpBuilder builder(podNew);
      Value init = createNondet(builder, podNew->getLoc(), innerType);
      podNew->getResult(0).replaceAllUsesWith(init);
      toErase.insert(podNew);
    }

    for (auto *op : toErase)
      op->erase();
  }

  // Update scf.while result types when the body yield types changed.
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "scf.while")
      return;
    auto &bodyRegion = op->getRegion(1);
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
  });
}

/// Late-stage: lift a `function.call` OUT of an `scf.while` body when
/// (a) all operands are loop-invariant after pod resolution, and (b) the
/// call's result is consumed exclusively by `struct.readm %call[@F] →
/// array.insert/write %arr[const] = %felt` chains targeting arrays
/// declared OUTSIDE the while.
///
/// Why a late pass and not part of `materializePodArrayCompField`:
/// the per-writer hoist there leaves the call's operands as `pod.read`
/// chains; only after `materializePodArrayInputPodField` +
/// `flattenPodArrayWhileCarry` + `unpackPodWhileCarry` settle do those
/// reduce to `array.extract %ba[const]` patterns the resolver below can
/// recognize. Running after the outer `while (changed)` fixed point
/// guarantees pod resolution has completed.
///
/// Why this is sound: under the same-cell single-instance dispatch
/// pattern (the precondition `materializePodArrayCompField` enforces
/// before erasing earlier writers), each loop iter writes to the same
/// destArr cell; only the last iter's value survives last-write-wins.
/// Lifting collapses N body-iter calls into ONE post-while call that
/// computes the same final-iter value over the post-while resolved
/// operands.
///
/// Empirical: aes_256_ctr's surviving AES256Encrypt_6 dispatch sits
/// inside a 1920-iter scf.while filling the key schedule. Without
/// this lift, JIT compile + GPU exec scale linearly with the iter
/// count; with it, exactly one call after the loop completes.
bool liftConstIndexPodArrayCallPostWhile(Operation *root) {
  bool changed = false;

  // Whitelist: cloning these post-while is semantics-preserving. Other
  // ops (`function.call`, `array.read` of a mutated array, etc.) could
  // change semantics if duplicated outside the loop body — bail there.
  auto isCloneable = [](StringRef n) {
    return n == "array.extract" || n == "cast.toindex" || n == "felt.const";
  };

  SmallVector<scf::WhileOp> whiles;
  root->walk([&](scf::WhileOp w) { whiles.push_back(w); });

  for (scf::WhileOp whileOp : whiles) {
    Block &body = whileOp.getAfter().front();

    SmallVector<Operation *> calls;
    for (Operation &op : body)
      if (op.getName().getStringRef() == "function.call")
        calls.push_back(&op);

    for (Operation *callOp : calls) {
      OpBuilder postBuilder(whileOp);
      postBuilder.setInsertionPointAfter(whileOp);
      llvm::DenseMap<Value, Value> resolved;
      SmallVector<Operation *> clonedOps;

      std::function<std::optional<Value>(Value)> resolve;
      resolve = [&](Value v) -> std::optional<Value> {
        auto it = resolved.find(v);
        if (it != resolved.end())
          return it->second;
        if (auto ba = dyn_cast<BlockArgument>(v)) {
          if (ba.getOwner() == &body) {
            Value r = whileOp.getResult(ba.getArgNumber());
            resolved[v] = r;
            return r;
          }
          // BA inside the whileOp but NOT in `body` — typically a
          // before-region (condition) arg — does not dominate post-
          // while. Bail. Body-args of enclosing blocks (funcBlock,
          // outer scf.while bodies) dominate post-while because
          // post-while is still inside the enclosing block.
          if (Operation *ownerOp = ba.getOwner()->getParentOp())
            if (whileOp->isAncestor(ownerOp))
              return std::nullopt;
          resolved[v] = v;
          return v;
        }
        Operation *def = v.getDefiningOp();
        if (!def) {
          resolved[v] = v;
          return v;
        }
        if (!whileOp.getOperation()->isAncestor(def)) {
          resolved[v] = v;
          return v;
        }
        if (!isCloneable(def->getName().getStringRef()))
          return std::nullopt;
        SmallVector<Value> ops;
        for (Value operand : def->getOperands()) {
          auto r = resolve(operand);
          if (!r)
            return std::nullopt;
          ops.push_back(*r);
        }
        Operation *cloned = postBuilder.clone(*def);
        for (size_t i = 0; i < ops.size(); ++i)
          cloned->setOperand(i, ops[i]);
        clonedOps.push_back(cloned);
        Value cr = cloned->getResult(0);
        resolved[v] = cr;
        return cr;
      };

      // Collect call users + verify the chain shape:
      // `struct.readm %call[@F] → array.insert/write %arr[const] = %felt`.
      // `%arr` must be defined outside the while AND have no other
      // writes inside the body (so the post-while array.insert is
      // structurally equivalent to the last in-body iter's effect).
      struct UserGroup {
        Operation *readm;
        Operation *write;
        Value destArr;
        SmallVector<Value> resolvedIndices;
      };
      SmallVector<UserGroup> userGroups;
      bool usersOk = true;
      llvm::SmallSetVector<Operation *, 4> writeSet;
      for (OpOperand &use : callOp->getResult(0).getUses()) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() != "struct.readm" ||
            user->getNumResults() == 0 || !user->getResult(0).hasOneUse()) {
          usersOk = false;
          break;
        }
        Operation *write = *user->getResult(0).getUsers().begin();
        StringRef wn = write->getName().getStringRef();
        if (wn != "array.insert" && wn != "array.write") {
          usersOk = false;
          break;
        }
        Value destArr = write->getOperand(0);
        // destArr must dominate post-while. Op-defined: bail when
        // defined inside whileOp. BA: bail when its owning block sits
        // inside whileOp (e.g., a loop-carried iter-arg whose value is
        // tied to in-body yields).
        if (auto ba = dyn_cast<BlockArgument>(destArr)) {
          Operation *ownerOp = ba.getOwner()->getParentOp();
          if (ownerOp && whileOp->isAncestor(ownerOp)) {
            usersOk = false;
            break;
          }
        } else if (Operation *destDef = destArr.getDefiningOp()) {
          if (whileOp.getOperation()->isAncestor(destDef)) {
            usersOk = false;
            break;
          }
        }
        SmallVector<Value> resolvedIndices;
        for (unsigned i = 1; i + 1 < write->getNumOperands(); ++i) {
          auto r = resolve(write->getOperand(i));
          if (!r) {
            usersOk = false;
            break;
          }
          resolvedIndices.push_back(*r);
        }
        if (!usersOk)
          break;
        userGroups.push_back({user, write, destArr, resolvedIndices});
        writeSet.insert(write);
      }

      if (!usersOk || userGroups.empty()) {
        for (Operation *c : llvm::reverse(clonedOps))
          c->erase();
        continue;
      }

      // Verify each destArr has NO other uses inside the writerWhile
      // body besides the array.insert/write we're lifting. Reads
      // (`array.extract`, `array.read`) inside the body would observe
      // the in-loop write's value; lifting would change semantics
      // because post-while the read fires before the post-while write.
      // Other writes (`array.write`, `array.insert`, `pod.write`) to
      // the same destArr would be silently dropped by the lift since
      // we only re-emit the lifted writes post-while.
      bool destArrSafe = true;
      llvm::DenseSet<Value> destArrSet;
      for (auto &g : userGroups)
        destArrSet.insert(g.destArr);
      whileOp->walk([&](Operation *op) {
        if (writeSet.contains(op))
          return;
        for (Value operand : op->getOperands()) {
          if (destArrSet.contains(operand)) {
            destArrSafe = false;
            return;
          }
        }
      });
      if (!destArrSafe) {
        for (Operation *c : llvm::reverse(clonedOps))
          c->erase();
        continue;
      }

      // Resolve call operands.
      SmallVector<Value> resolvedCallOperands;
      bool ok = true;
      for (Value op : callOp->getOperands()) {
        auto r = resolve(op);
        if (!r) {
          ok = false;
          break;
        }
        resolvedCallOperands.push_back(*r);
      }
      if (!ok) {
        for (Operation *c : llvm::reverse(clonedOps))
          c->erase();
        continue;
      }

      auto callee = callOp->getAttrOfType<SymbolRefAttr>("callee");
      Operation *postCall = postBuilder.create<llzk::function::CallOp>(
          callOp->getLoc(), callOp->getResultTypes(), callee,
          resolvedCallOperands);
      Value postCallResult = postCall->getResult(0);

      for (auto &g : userGroups) {
        Operation *clonedReadm = postBuilder.clone(*g.readm);
        clonedReadm->setOperand(0, postCallResult);
        Value clonedFelt = clonedReadm->getResult(0);
        OperationState ws(g.write->getLoc(), g.write->getName().getStringRef());
        SmallVector<Value> wsOperands{g.destArr};
        wsOperands.append(g.resolvedIndices.begin(), g.resolvedIndices.end());
        wsOperands.push_back(clonedFelt);
        ws.addOperands(wsOperands);
        for (auto attr : g.write->getAttrs())
          ws.addAttribute(attr.getName(), attr.getValue());
        postBuilder.create(ws);
      }

      // Erase in-while: array.insert/write, struct.readm, call.
      // The user-collection above only enumerated `getResult(0).getUses()`.
      // If the call has additional results with surviving uses, leave
      // the call op in place — the per-result safety check guards
      // against dangling refs that would crash the verifier.
      for (auto &g : userGroups) {
        g.write->erase();
        g.readm->erase();
      }
      if (isAllResultsUnused(*callOp))
        callOp->erase();
      changed = true;
    }
  }

  return changed;
}

/// Erase pod-typed iter slots from `scf.while` and DCE the `pod.new`
/// chain that fed them. Runs after `pod.read` / `pod.write` have been
/// nondet'd / erased; at that point any pod-typed Value still in the IR
/// is structural bookkeeping (an empty `pod.new : <[]>` orphan plus its
/// dispatch-pod cascade) with no real consumer. Left in, the orphan
/// trips `createEmptyTemplateRemoval`'s `applyFullConversion` because
/// `pod.new` is outside its `OpClassesWithStructTypes` target tuple.
///
/// Why use-trace and not cascade reshape: the `<--` cascade carries
/// values that `extendResultBearingScfIfArrayChain` / `convertArrayWritesToSSA`
/// match on tracked-array type equality during LlzkToStablehlo. Reshaping
/// the scf.if / scf.execute_region scaffolding to remove pod-typed slots
/// breaks those invariants and trips downstream "empty block: expect at
/// least a terminator" failures at adjacent non-pod scf.execute_regions
/// (CLAUDE.md "Don't reshape the `<--` cascade from SSC"). Use-trace
/// recognizes the bundle as structurally dead without touching the
/// surrounding scaffolding shape.
///
/// The (rebuild → defer erase) split matters: a single-pass rebuild-
/// then-erase loop trips `Cannot destroy a value that still has uses!`
/// at `Block::eraseArgument` because an inner carrier's dropped result
/// is still referenced by an enclosing carrier's terminator operand
/// when erase fires — the enclosing one is rebuilt later in post order,
/// so its terminator hasn't been trimmed yet. Deferring the erase plus
/// pre-severing dead-pod.new and per-rebuild OLD-carrier references via
/// `dropAllReferences` lets the post-order rebuild trim every claim
/// before any value is destroyed.
bool erasePodTypedCarrierSlots(ModuleOp module) {
  using SlotKey = std::pair<Operation *, unsigned>;

  // `scf.while`, `scf.if`, `scf.execute_region` all forward pod-typed
  // carrier slots, so the closure tracks and rebuilds all three. The
  // companion `IsTerminator` guard in LlzkToStablehlo's dead-op DCE
  // keeps non-pod `scf.execute_region` bodies from losing their
  // `scf.yield` after the inner `scf.if` chain converts to
  // stablehlo.select.
  auto isCarrierOp = [](Operation *op) -> bool {
    return isa<scf::WhileOp, scf::IfOp, scf::ExecuteRegionOp>(op);
  };
  auto isPodTyped = [](Type t) -> bool {
    return t.getDialect().getNamespace() == "pod";
  };

  llvm::SetVector<SlotKey> droppableSlots;
  llvm::SetVector<Operation *> deadPodNews;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "pod.new")
      deadPodNews.insert(op);
    if (!isCarrierOp(op))
      return;
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
      if (isPodTyped(op->getResult(i).getType()))
        droppableSlots.insert({op, i});
  });
  if (droppableSlots.empty() && deadPodNews.empty())
    return false;

  // A use is "clean" iff it forwards into another candidate slot or is
  // an operand of a still-candidate pod.new — i.e. removing all
  // candidates simultaneously would leave the value with zero uses.
  auto isCleanUse = [&](OpOperand &use) -> bool {
    Operation *u = use.getOwner();
    unsigned on = use.getOperandNumber();
    if (isa<scf::WhileOp>(u))
      return droppableSlots.contains({u, on});
    if (isa<scf::ConditionOp>(u)) {
      if (on == 0)
        return false;
      Operation *parent = u->getParentOp();
      return parent && isa<scf::WhileOp>(parent) &&
             droppableSlots.contains({parent, on - 1});
    }
    if (isa<scf::YieldOp>(u)) {
      Operation *parent = u->getParentOp();
      if (!parent)
        return false;
      if (isa<scf::WhileOp, scf::IfOp, scf::ExecuteRegionOp>(parent))
        return droppableSlots.contains({parent, on});
      return false;
    }
    if (u->getName().getStringRef() == "pod.new")
      return deadPodNews.contains(u);
    return false;
  };
  auto allUsesClean = [&](Value v) -> bool {
    return llvm::all_of(v.getUses(),
                        [&](OpOperand &use) { return isCleanUse(use); });
  };
  auto slotLocalValues = [&](Operation *op, unsigned slot,
                             SmallVectorImpl<Value> &out) {
    if (auto w = dyn_cast<scf::WhileOp>(op)) {
      if (!w.getBefore().empty())
        out.push_back(w.getBefore().front().getArgument(slot));
      if (!w.getAfter().empty())
        out.push_back(w.getAfter().front().getArgument(slot));
      out.push_back(w.getResult(slot));
    } else {
      out.push_back(op->getResult(slot));
    }
  };

  // Narrow until stable: drop any slot whose live values have a non-
  // clean use; drop any pod.new whose results have a non-clean use.
  bool stable = false;
  while (!stable) {
    stable = true;
    for (SlotKey key : llvm::to_vector(droppableSlots)) {
      if (!droppableSlots.contains(key))
        continue;
      SmallVector<Value> values;
      slotLocalValues(key.first, key.second, values);
      if (!llvm::all_of(values, allUsesClean)) {
        droppableSlots.remove(key);
        stable = false;
      }
    }
    for (Operation *pn : llvm::to_vector(deadPodNews)) {
      if (!deadPodNews.contains(pn))
        continue;
      if (!llvm::all_of(pn->getResults(), allUsesClean)) {
        deadPodNews.remove(pn);
        stable = false;
      }
    }
  }

  llvm::DenseMap<Operation *, SmallVector<unsigned>> dropByOp;
  for (SlotKey key : droppableSlots)
    dropByOp[key.first].push_back(key.second);
  for (auto &kv : dropByOp)
    llvm::sort(kv.second);
  if (dropByOp.empty() && deadPodNews.empty())
    return false;

  // Pre-sever dead pod.new operand-uses so the carrier rebuilds below
  // can erase block args without tripping on uses claimed by a pod.new
  // that hasn't been erased yet. `dropAllReferences` leaves the op in
  // place; it just removes its OpOperand entries from the referenced
  // Values' use lists.
  for (Operation *pn : deadPodNews)
    pn->dropAllReferences();

  // Post-order is load-bearing: an enclosing carrier's terminator
  // operands on an inner carrier's dropped result are only trimmed
  // when the enclosing rebuild fires, so the inner must be rebuilt +
  // dropAllReferences'd first. Default is `WalkOrder::PostOrder` but
  // making it explicit pins the contract at the call site.
  SmallVector<Operation *> postOrderOps;
  module.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (dropByOp.count(op))
      postOrderOps.push_back(op);
  });

  SmallVector<Operation *> oldCarriers;
  for (Operation *op : postOrderOps) {
    const SmallVector<unsigned> &dropped = dropByOp[op];
    SmallVector<Type> newResultTypes;
    SmallVector<unsigned> keep;
    unsigned di = 0;
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      if (di < dropped.size() && dropped[di] == i) {
        ++di;
        continue;
      }
      keep.push_back(i);
      newResultTypes.push_back(op->getResult(i).getType());
    }

    OpBuilder b(op);
    if (auto w = dyn_cast<scf::WhileOp>(op)) {
      SmallVector<Value> newOperands;
      for (unsigned k : keep)
        newOperands.push_back(w.getOperand(k));
      auto newWhile =
          b.create<scf::WhileOp>(w.getLoc(), newResultTypes, newOperands);
      newWhile.getBefore().takeBody(w.getBefore());
      newWhile.getAfter().takeBody(w.getAfter());
      for (Region *region : {&newWhile.getBefore(), &newWhile.getAfter()}) {
        Block &blk = region->front();
        Operation *term = blk.getTerminator();
        unsigned termOffset = isa<scf::ConditionOp>(term) ? 1 : 0;
        for (unsigned slot : llvm::reverse(dropped))
          term->eraseOperand(slot + termOffset);
        for (unsigned slot : llvm::reverse(dropped))
          blk.eraseArgument(slot);
      }
      for (unsigned ns = 0; ns < keep.size(); ++ns)
        w.getResult(keep[ns]).replaceAllUsesWith(newWhile.getResult(ns));
    } else if (auto ifo = dyn_cast<scf::IfOp>(op)) {
      bool hasElse = !ifo.getElseRegion().empty();
      auto newIf = b.create<scf::IfOp>(ifo.getLoc(), newResultTypes,
                                       ifo.getCondition(), hasElse);
      newIf.getThenRegion().takeBody(ifo.getThenRegion());
      if (hasElse)
        newIf.getElseRegion().takeBody(ifo.getElseRegion());
      for (Region *region : {&newIf.getThenRegion(), &newIf.getElseRegion()}) {
        if (region->empty())
          continue;
        Operation *term = region->front().getTerminator();
        if (!isa<scf::YieldOp>(term))
          continue;
        for (unsigned slot : llvm::reverse(dropped))
          term->eraseOperand(slot);
      }
      for (unsigned ns = 0; ns < keep.size(); ++ns)
        ifo.getResult(keep[ns]).replaceAllUsesWith(newIf.getResult(ns));
    } else {
      auto er = cast<scf::ExecuteRegionOp>(op);
      auto newEr = b.create<scf::ExecuteRegionOp>(er.getLoc(), newResultTypes);
      newEr.getRegion().takeBody(er.getRegion());
      for (Block &blk : newEr.getRegion()) {
        Operation *term = blk.getTerminator();
        if (!isa<scf::YieldOp>(term))
          continue;
        for (unsigned slot : llvm::reverse(dropped))
          term->eraseOperand(slot);
      }
      for (unsigned ns = 0; ns < keep.size(); ++ns)
        er.getResult(keep[ns]).replaceAllUsesWith(newEr.getResult(ns));
    }
    // Sever this OLD carrier's operand-uses on enclosing block args /
    // outer values. Post-order means an enclosing carrier hasn't been
    // rebuilt yet — if this OLD op kept claiming uses on the enclosing
    // block args, the enclosing rebuild's `eraseArgument` would assert.
    op->dropAllReferences();
    oldCarriers.push_back(op);
  }

  // Iterate to fixed point: an OLD carrier's dropped result may retain
  // a pod.new-operand use until that pod.new clears, and a pod.new in
  // a chain may retain a use until its consumer pod.new clears. Each
  // iteration erases anything whose results are now use_empty.
  llvm::SetVector<Operation *> pending;
  for (Operation *op : oldCarriers)
    pending.insert(op);
  for (Operation *pn : deadPodNews)
    pending.insert(pn);
  bool progress = true;
  while (progress) {
    progress = false;
    for (Operation *op : llvm::to_vector(pending)) {
      if (!pending.contains(op) || !isAllResultsUnused(*op))
        continue;
      for (Region &r : op->getRegions()) {
        r.dropAllReferences();
        r.getBlocks().clear();
      }
      op->erase();
      pending.remove(op);
      progress = true;
    }
  }
  return true;
}

struct SimplifySubComponents
    : impl::SimplifySubComponentsBase<SimplifySubComponents> {
  using SimplifySubComponentsBase::SimplifySubComponentsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Snapshot the set of struct members read outside @constrain — Phase 4
    // (`eraseDeadPodAndCountOps`) consults this to decide whether an
    // scf.if-wrapped `struct.writem %self[@F]` is preserving a live wire or
    // dead bookkeeping. Must be computed BEFORE any phase erases readm ops.
    populateExternallyLiveMembers(module);

    // Idempotence: SSC runs from both the CLI and from inside
    // `--llzk-to-stablehlo`. The second invocation finds a pod-free,
    // template-free module; probe once and skip the v2 prereqs if no
    // pod or `$inputs` member remains.
    bool needsV2Prereqs = false;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "pod" ||
          (op->getName().getStringRef() == "struct.member" &&
           op->getAttrOfType<StringAttr>("sym_name") &&
           op->getAttrOfType<StringAttr>("sym_name")
               .getValue()
               .ends_with("$inputs"))) {
        needsV2Prereqs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Pre-step: strip `@X$inputs` pod traffic from `@compute`/`@constrain`
    // so the later template-removal `applyFullConversion` does not trip on
    // residual `pod.read` ops from the LLZK v2 constrain-side channel.
    if (needsV2Prereqs)
      eliminateInputPods(module);

    // Run to fixed point for multi-level sub-component chains.
    bool changed = true;
    while (changed) {
      changed = false;
      module.walk([&](Operation *structDef) {
        if (structDef->getName().getStringRef() != "struct.def")
          return;

        structDef->walk([&](Operation *funcDef) {
          if (funcDef->getName().getStringRef() != "function.def")
            return;
          auto symName = funcDef->getAttrOfType<StringAttr>("sym_name");
          if (!symName || symName.getValue() != "compute")
            return;

          for (Region &region : funcDef->getRegions()) {
            for (Block &block : region) {
              bool hasPod = false;
              block.walk([&](Operation *op) {
                if (op->getName().getDialectNamespace() == "pod")
                  hasPod = true;
              });
              if (hasPod) {
                changed |= materializePodArrayCompField(block);
                // Run BEFORE `materializePodArrayInputPodField` so any
                // struct-of-pods carrier (`!pod<[@idx_N..]>` with uniform
                // inner type) is rewritten to array-of-pods first. The
                // `@in` read-modify-write pattern downstream then sees
                // `pod.read` on an `array.read`-produced cell — the exact
                // shape `materializePodArrayInputPodField` and
                // `flattenPodArrayWhileCarry` already handle. One call per
                // outer iter is sufficient because the rewrite is
                // idempotent — after success the carrier's type is array,
                // no longer matching the seed predicate.
                changed |= convertStructOfPodsToArrayOfPods(block);
                // Non-uniform-inner struct-of-pods carriers
                // (`convertStructOfPodsToArrayOfPods` no-op) leave the
                // dispatched calls hoisted by `extractCallsFromScfIf` with
                // no consumer for their `!struct<@Sub_K>` results — the
                // reader cascade in a sibling scf.while body reads
                // `struct.readm [@F]` from a pre-existing
                // `llzk.nondet : !struct<@Sub_K>` instead. Bridge the
                // writer↔reader link by materializing a parallel felt
                // carrier per `@F`. Idempotent.
                changed |= materializeStructOfPodsCompField(block);
                // Run BEFORE `flattenPodArrayWhileCarry` so the
                // writer-side `pod.write %cell[@in] = %src` and the
                // firing-site `pod.read %cell[@in]` are still SSA-paired
                // through `%cell`. After flatten, `%cell` is severed from
                // the per-field carry across nested scf.whiles and the
                // pairing is unrecoverable.
                changed |= materializePodArrayInputPodField(block);
                changed |= flattenPodArrayWhileCarry(block);
                // Drive `unpackPodWhileCarry` to its own fixed point before
                // materializing tail calls. The unpacker processes one
                // while per call (it erases chained `scf.while` users
                // inline, invalidating SmallVector pointers, so it returns
                // early); without this loop the outer fixed point would
                // run `eliminatePodDispatch` after only one writer-while's
                // pod-carry was unpacked, and Phase 5
                // (`replaceRemainingPodOps`) would `llzk.nondet` the
                // sibling pods' cross-block readers before
                // `materializeScalarPodCompField` could see them.
                while (unpackPodWhileCarry(block))
                  changed = true;
                changed |= materializeScalarPodCompField(block);
                changed |= eliminatePodDispatch(block);
                // Recursively process nested while body blocks.
                std::function<void(Block &)> processNested;
                processNested = [&](Block &parent) {
                  for (Operation &op : parent) {
                    if (op.getName().getStringRef() != "scf.while")
                      continue;
                    for (Region &r : op.getRegions()) {
                      for (Block &b : r) {
                        changed |= flattenPodArrayWhileCarry(b);
                        changed |= unpackPodWhileCarry(b);
                        bool hasArrayOfPods = false;
                        for (Operation &bop : b)
                          if (bop.getName().getStringRef() == "array.read" &&
                              bop.getNumResults() > 0 &&
                              bop.getResult(0)
                                      .getType()
                                      .getDialect()
                                      .getNamespace() == "pod")
                            hasArrayOfPods = true;
                        // Skip eliminatePodDispatch when this scf.while
                        // body still has pod-typed block args. Phase 5
                        // (`replaceRemainingPodOps`) would nondet every
                        // `pod.read` in the block — including
                        // `pod.read %arg[@field]` reads that
                        // `unpackPodWhileCarry` needs to discover field
                        // types on the next outer fixed-point iteration.
                        // Once the carry is unpacked, the block args
                        // become non-pod and dispatch elimination can
                        // proceed normally.
                        bool hasPodBlockArg = false;
                        for (BlockArgument arg : b.getArguments())
                          if (arg.getType().getDialect().getNamespace() ==
                              "pod")
                            hasPodBlockArg = true;
                        if (!hasArrayOfPods && !hasPodBlockArg) {
                          changed |= eliminatePodDispatch(b);
                        } else if (hasArrayOfPods && !hasPodBlockArg) {
                          // Post-Option-B carrier: block has
                          // `array.read %carrier[%i] : !pod` at top
                          // level but no pod-typed block args. Phase 5's
                          // nondet would still clobber the `array.read`
                          // chain's pod field discovery, so skip the
                          // full eliminatePodDispatch. But Phase 1
                          // (`extractCallsFromScfIf`) is benign — it
                          // hoists buried `function.call`s out of
                          // dispatch-firing scf.if's via clone-hoist.
                          // Without this targeted call, deep dispatch
                          // chains (canonical case: webb Poseidon's
                          // 68-round Ark→Mix→Sigma) sit in
                          // statically-false scf.if's that
                          // `--llzk-to-stablehlo` later DCEs.
                          //
                          // Phase 2 (`replacePodReads`) is also safe
                          // here: the carrier is now array-of-pods, so
                          // pod-typed block args have already been
                          // unpacked by an earlier outer iter; no live
                          // pod-read-back patterns through block args
                          // remain. Phase 2 only RAUWs pod.reads whose
                          // source is tracked via the local Phase 1
                          // scan — narrow enough not to clobber
                          // unrelated pod traffic.
                          //
                          // Phase 2 is intentionally NOT run when
                          // `hasPodBlockArg` is true: pod block args
                          // host `pod.read %arg[@field]` read-back
                          // patterns that the broader RAUW would tear
                          // (canonical regression:
                          // `unpack_pod_while_carry_block_arg_*.mlir`).
                          llvm::DenseMap<Value, llvm::StringMap<Value>>
                              localTrackedPodValues;
                          changed |=
                              extractCallsFromScfIf(b, localTrackedPodValues);
                          changed |= replacePodReads(b, localTrackedPodValues);
                        } else if (hasPodBlockArg) {
                          // Surviving struct-of-pods carrier on a pod
                          // block-arg with NON-uniform inner types (each
                          // `@idx_K` resolves to a different `@comp:
                          // !struct<@Sub_K>`).
                          // `convertStructOfPodsToArrayOfPods` cannot rewrite
                          // the carrier — there's no single array element type
                          // that admits the K distinct struct classes. But the
                          // buried `function.call @Sub_K::@compute(%input)` is
                          // already concrete in the IR (circom emits
                          // the literal `@Sub_K` symbol per cascade arm).
                          // Phase 1 (`extractCallsFromScfIf`) can still
                          // clone-hoist these calls when their operand
                          // chain is pure / non-pod (e.g.
                          // `array.extract %outer_iter_arg[%c_K]` on
                          // a felt-array sibling carrier). Phase 2's
                          // broad RAUW is intentionally skipped: it
                          // would tear `pod.read %arg[@field]` read-back
                          // patterns that `unpackPodWhileCarry` later
                          // depends on (canonical regression:
                          // `unpack_pod_while_carry_block_arg_*.mlir`).
                          // Canonical case: webb Poseidon's 68-round
                          // Ark cascade where each `@idx_K` resolves to
                          // a distinct `!struct<@Ark_K>` round constant
                          // — there's no uniform-inner shape Option B
                          // can target.
                          llvm::DenseMap<Value, llvm::StringMap<Value>>
                              localTrackedPodValues;
                          changed |=
                              extractCallsFromScfIf(b, localTrackedPodValues);
                        }
                        // No remaining branch — every (hasArrayOfPods,
                        // hasPodBlockArg) combination is dispatched
                        // above.
                        changed |= resolveArrayPodCompReads(b);
                        // Fold residual `array.read → pod.read
                        // @count/@comp/@in` chains that
                        // `resolveArrayPodCompReads` can't redirect
                        // (the dispatched call's SSA value is local to
                        // the dispatch-firing block; circom v2 emits
                        // post-loop read-back loops that re-walk the
                        // dispatch-pod array).
                        if (hasArrayOfPods) {
                          changed |= rewriteArrayPodCountCompInReads(b);
                          changed |= eraseDeadPodAndCountOps(b);
                        }
                        processNested(b);
                      }
                    }
                  }
                };
                processNested(block);
              }
            }
          }
        });
      });
    }

    // Straggler flatten: `processNested` only recurses into scf.while bodies,
    // so a pod-array-carrying scf.while buried inside an scf.if branch is
    // never visited. AES `@AES256Encrypt_6::compute` has its xor_2 / xor_3
    // input-pod carries threaded through `scf.if` branches at depth 5, and
    // the main loop above leaves them as raw `<D x !pod<[@a, @b]>>`. Here we
    // walk the module looking for any remaining pod-array iter-arg and
    // re-invoke `flattenPodArrayWhileCarry` on the containing block.
    // Combined with the field-type fallback above, this closes the chain
    // regardless of nesting.
    {
      bool stragglerChanged = true;
      while (stragglerChanged) {
        stragglerChanged = false;
        llvm::SmallSetVector<Block *, 8> blocksToFlatten;
        module.walk([&](scf::WhileOp w) {
          for (unsigned i = 0; i < w.getNumResults(); ++i) {
            Type ty = w.getResult(i).getType();
            // NOLINTNEXTLINE(readability/braces)
            if (auto at = dyn_cast<llzk::array::ArrayType>(ty))
              if (at.getElementType().getDialect().getNamespace() == "pod") {
                blocksToFlatten.insert(w->getBlock());
                break;
              }
          }
        });
        for (Block *b : blocksToFlatten)
          stragglerChanged |= flattenPodArrayWhileCarry(*b);
      }
    }

    // Straggler scf.if + scf.while pod-array flattening, run as a single
    // fixed-point. scf.ifs whose result type list still contains
    // `<D x !pod<[@a, @b, ...]>>` survive the main while-carry flatten because
    // `flattenPodArrayWhileCarry` only walks scf.while iter-args. AES rounds-
    // loop has scf.ifs nested inside an already-flattened outer scf.while
    // whose pod-array result slots blocked the per-field carry from threading
    // through to outer scf.yields. After the scf.if rewrite, the new per-field
    // felt-array result slots become promotable carries
    // (`isPromotableCarryType` excludes pod-element arrays but accepts felt-
    // element arrays); LlzkToStablehlo's `extendResultBearingScfIfArrayChain`
    // then walks each branch, finds the inner scf.while's per-field SSA
    // carries, and rewrites the branch yields to use the latest values.
    //
    // The two ops are folded into one walk so a single iteration also catches
    // scf.while iter-args newly exposed when an scf.if's pod-array result fed
    // an outer scf.yield → outer scf.while iter-arg chain (and vice versa,
    // for completeness).
    {
      bool changed = true;
      while (changed) {
        changed = false;
        llvm::SmallSetVector<Block *, 8> blocksToFlattenIf;
        llvm::SmallSetVector<Block *, 8> blocksToFlattenWhile;
        module.walk([&](Operation *op) {
          if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
              auto at =
                  dyn_cast<llzk::array::ArrayType>(ifOp.getResult(i).getType());
              if (at && isa<llzk::pod::PodType>(at.getElementType())) {
                blocksToFlattenIf.insert(ifOp->getBlock());
                break;
              }
            }
          } else if (auto w = dyn_cast<scf::WhileOp>(op)) {
            for (unsigned i = 0; i < w.getNumResults(); ++i) {
              auto at =
                  dyn_cast<llzk::array::ArrayType>(w.getResult(i).getType());
              if (at && isa<llzk::pod::PodType>(at.getElementType())) {
                blocksToFlattenWhile.insert(w->getBlock());
                break;
              }
            }
          }
        });
        for (Block *b : blocksToFlattenIf)
          changed |= flattenPodArrayScfIfResults(*b);
        for (Block *b : blocksToFlattenWhile)
          changed |= flattenPodArrayWhileCarry(*b);
      }
    }

    // Post-flatten rewire: connect each scf.while's per-field block args to
    // nested whiles' per-field nondet inits. The inner rewire inside
    // `expandPodArrayWhile` (`rewritePodArrayUsesInBlock`) only fires when
    // the nested while has *already* been flattened by a prior
    // outer-fixed-point iteration; when the parent is flattened first, the
    // nested still carries the OLD pod-array type at that moment, so the
    // inner rewire's "contiguous run of llzk.nondet of per-field types"
    // match fails. Once the outer fixed point has settled and `processNested`
    // has flattened the nested whiles too, both sides have per-field
    // felt-typed carries, but the nested's inits are LOCAL nondets (created
    // by the nested's own `expandPodArrayWhile`) and never rewired back to
    // the parent's per-field block args. This module-level pass closes that
    // chain.
    //
    // Algorithm: for every scf.while parent and every nested scf.while
    // inside parent's body region, find each contiguous run of `llzk.nondet`
    // operands of flattenable felt / felt-array types in the nested's
    // init list, then find an unclaimed contiguous run of parent block args
    // with the same type sequence and rewire `nested.setOperand` to point
    // at parent block args. Same start+N pattern as the inner rewire; the
    // claimed-position set disambiguates when parent has multiple per-field
    // groups of identical types (e.g. xor_a (@a, @b) followed by xor_b
    // (@a, @b) of matching shape).
    //
    // Convergence: this only `setOperand`s on existing scf.whiles — no pod
    // ops added or removed. `eliminatePodDispatch` would not reach a
    // different fixed point even if re-run, so we run AFTER the main loop
    // has settled and don't need to revisit dispatch elimination.
    // Visit each scf.while as a `parent` and rewire ONLY its immediate child
    // scf.whiles' nondet inits to parent's per-field block args. Iterate the
    // block non-recursively — a deeper-nested while is the immediate child
    // of its own parent in module.walk's later visit, not of an outer
    // ancestor. No fixed point needed: this only setOperands existing ops,
    // and rewiring a nested to its parent's block args doesn't expose new
    // nondet operands elsewhere.
    module.walk([&](scf::WhileOp parent) {
      for (Region &r : parent->getRegions()) {
        if (r.empty())
          continue;
        Block &blk = r.front();
        // Collect direct-child scf.whiles plus those reachable through
        // scf.if/scf.for branches without crossing a deeper scf.while
        // boundary. AES rounds-loop has per-field carrier-bearing
        // scf.whiles at depth 4 (rounds-loop body → scf.if branch → scf.if
        // branch → scf.while), introduced by `flattenPodArrayScfIfResults`
        // after each enclosing scf.if's pod-array result slot was
        // rewritten to per-field. Without recursion the rewire would
        // never see the depth-4 scf.while's per-field nondet inits,
        // leaving them disconnected from `parent`'s per-field block args.
        // Each inner scf.while is itself a parent in its own
        // module.walk visit, so we stop descent at scf.while boundaries
        // to avoid double-rewire (the nondet-operand check at the matcher
        // also defends against this). `Block::walk` over `blk` never visits
        // `parent` itself — parent contains blk's region rather than living
        // inside it — so no equality guard is needed.
        SmallVector<scf::WhileOp, 8> nestedWhiles;
        blk.walk([&](scf::WhileOp w) {
          nestedWhiles.push_back(w);
          return WalkResult::skip();
        });
        for (scf::WhileOp nested : nestedWhiles) {
          llvm::DenseSet<unsigned> claimedBaPositions;

          auto isFlattenableNondet = [](Value v) {
            Operation *def = v.getDefiningOp();
            return def && def->getName().getStringRef() == "llzk.nondet" &&
                   isFlattenableFelt(v.getType());
          };

          // Try to commit a contiguous run of `nested` operands at
          // `[start, start+runLen)` to a same-typed unclaimed run of
          // parent block args. Returns true on commit.
          auto tryClaimRun = [&](unsigned start, unsigned runLen) -> bool {
            for (unsigned baStart = 0;
                 baStart + runLen <= blk.getNumArguments(); ++baStart) {
              bool ok = true;
              for (unsigned k = 0; k < runLen && ok; ++k) {
                ok = !claimedBaPositions.count(baStart + k) &&
                     blk.getArgument(baStart + k).getType() ==
                         nested->getOperand(start + k).getType();
              }
              if (!ok)
                continue;
              for (unsigned k = 0; k < runLen; ++k) {
                nested->setOperand(start + k, blk.getArgument(baStart + k));
                claimedBaPositions.insert(baStart + k);
              }
              return true;
            }
            return false;
          };

          // First pass — full contiguous mixed-type match. A single
          // per-pod-array group whose per-field carries occupy adjacent
          // positions in BOTH inner's operand list AND parent's block
          // args (after `flattenPodArrayWhileCarry`'s record-order
          // resort) commits as one run. Canonical case: maci_splicer's
          // QuinSelector_6 @in fill loop nested in @main — inner's
          // `[5x5 felt, 5 felt]` for `[@in, @index]` matches parent's
          // adjacent same sequence. Without this pass the homogeneous
          // fallback below picks the first unclaimed same-type arg,
          // which for chips with multiple 5-felt per-field carries
          // (mux's @s) is the wrong group.
          for (unsigned start = 0; start < nested->getNumOperands();) {
            unsigned end = start;
            while (end < nested->getNumOperands() &&
                   isFlattenableNondet(nested->getOperand(end)))
              ++end;
            if (end - start < 2) {
              start = end == start ? start + 1 : end;
              continue;
            }
            start = tryClaimRun(start, end - start) ? end : start + 1;
          }

          // Second pass — homogeneous-type sub-runs. AES rounds-loop is
          // the canonical case: 4 adjacent nondets typed
          // <13,4,3,32>×2 + <13,4,32>×2 must be matched as two
          // independent type-homogeneous runs against parent body args
          // at non-contiguous positions [0,1] + [8,9].
          for (unsigned start = 0; start < nested->getNumOperands();) {
            unsigned end = start;
            Type runType;
            while (end < nested->getNumOperands() &&
                   isFlattenableNondet(nested->getOperand(end))) {
              Type ty = nested->getOperand(end).getType();
              if (end == start)
                runType = ty;
              else if (ty != runType)
                break;
              ++end;
            }
            if (end == start) {
              ++start;
              continue;
            }
            tryClaimRun(start, end - start);
            start = end;
          }
        }
      }
    });

    // Post-step: inline single-field input pods used as `scf.while` carry
    // (e.g. `!pod.type<[@claim: !array<8 x felt>]>`) to the raw inner
    // type. These are orthogonal to dispatch pods (no `@count` / `@comp`
    // fields) and are not caught by the dispatch rewrite, but they still
    // need to be gone before template removal runs `applyFullConversion`.
    if (needsV2Prereqs)
      inlineInputPodCarries(module);

    // Post-step: scrub residual pod ops module-wide. `eliminatePodDispatch`
    // tracks pod-field values per-block and per-pod-SSA-value; circom v2
    // emits scalar dispatch pods (`pod.new { @count = N }`) at the
    // `@compute` body level whose `pod.read`s live many regions deep
    // (inside `scf.while` / `scf.for` bodies that iterate the dispatch),
    // so the per-block tracker never sees them and Phase 5's
    // non-recursive walk leaves them behind. After
    // `inlineInputPodCarries` has already inlined the input-pod carries
    // it can match (single-field, while-threaded), anything still
    // referencing a `pod.*` op is structural bookkeeping whose witness
    // value is don't-care for our lowering — replace `pod.read` with
    // `llzk.nondet` of the result type, then DCE `pod.write`/`pod.new`
    // chains. Runs AFTER `inlineInputPodCarries` so that pass can still
    // discover the inner type from a live `pod.read`.
    if (needsV2Prereqs) {
      SmallVector<Operation *> podReadsToErase;
      SmallVector<Operation *> podWritesToErase;
      module.walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        if (name == "pod.write") {
          podWritesToErase.push_back(op);
          return;
        }
        if (name != "pod.read" || op->getNumResults() == 0)
          return;
        OpBuilder b(op);
        Type rty = op->getResult(0).getType();
        Value replacement;
        if (rty.isIndex()) {
          // `llzk.nondet : index` is illegal in the dialect-conversion
          // target. Substitute `arith.constant 0` for index-typed
          // dispatch-pod `@count` reads (same justification as
          // `rewriteArrayPodCountCompInReads` for the scf.while case).
          OperationState state(op->getLoc(), "arith.constant");
          state.addAttribute("value", b.getIndexAttr(0));
          state.addTypes({rty});
          replacement = b.create(state)->getResult(0);
        } else {
          replacement = createNondet(b, op->getLoc(), rty);
        }
        op->getResult(0).replaceAllUsesWith(replacement);
        podReadsToErase.push_back(op);
      });
      for (Operation *op : podReadsToErase)
        op->erase();
      for (Operation *op : podWritesToErase)
        op->erase();

      // `erasePodTypedCarrierSlots` rebuilds carrier ops to drop pod-typed
      // iter/result slots whose only consumers are other pod.new chains
      // or surviving carrier-forwarding terminators. The standalone
      // pod.new DCE that follows catches any orphan it missed (e.g. a
      // pod.new whose pre-cleanup result use was a now-erased pod.read).
      // The pod-typed `llzk.nondet` DCE clears orphan placeholders the
      // pod.read substitution above left behind once the carrier slot
      // drop severed all their downstream consumers — `llzk.nondet : !pod`
      // is dialect-conversion-illegal at LlzkToStablehlo so any survivor
      // would trip the next pass. Iterate the trio to a fixed point:
      // dropping a carrier slot may unlock a new pod.new for DCE, and
      // erasing a pod.new chain may unlock new carrier slots or new
      // orphan nondets for the next round.
      bool changedCleanup = true;
      while (changedCleanup) {
        changedCleanup = false;
        changedCleanup |= erasePodTypedCarrierSlots(module);
        bool dcePodNew = true;
        while (dcePodNew) {
          dcePodNew = false;
          SmallVector<Operation *> deadOrphans;
          module.walk([&](Operation *op) {
            StringRef name = op->getName().getStringRef();
            bool isCandidate =
                name == "pod.new" ||
                (name == "llzk.nondet" && op->getNumResults() == 1 &&
                 op->getResult(0).getType().getDialect().getNamespace() ==
                     "pod");
            if (isCandidate && isAllResultsUnused(*op))
              deadOrphans.push_back(op);
          });
          for (Operation *op : deadOrphans) {
            op->erase();
            dcePodNew = true;
            changedCleanup = true;
          }
        }
      }
    }

    // Late lift: hoist single-instance dispatch calls out of their
    // writerWhile bodies. By this point all pod.read chains have
    // settled into `array.extract %ba[const]` patterns the lift's
    // operand resolver can recognize — the same call inside an N-iter
    // scf.while runs N times structurally, but writes its result to a
    // const-index destArr cell where last-write-wins makes the first
    // N-1 iters dead. The lift collapses to one post-while call.
    // Drives a small fixed point because lifting one chain may expose
    // another in an outer while (rare but correct under conservative
    // gating).
    {
      bool liftChanged = true;
      while (liftChanged)
        liftChanged = liftConstIndexPodArrayCallPostWhile(module);
    }

    // Post-step: unwrap LLZK v2 `poly.template` shells and collapse empty
    // parameter lists on struct type refs (`@X<[]>` → `@X`). Runs AFTER
    // dispatch cleanup and while-carry inlining so the `pod.new`/
    // `pod.read`/`pod.write` ops that circom v2 emits have already been
    // DCE'd — the upstream pass's `applyFullConversion` has no target
    // pattern for pod ops and would bail out otherwise. Skip when no
    // `poly.template` is present (hand-written LIT fixtures bypass the
    // template wrapping).
    {
      bool hasTemplate = false;
      module.walk([&](llzk::polymorphic::TemplateOp) {
        hasTemplate = true;
        return WalkResult::interrupt();
      });
      if (hasTemplate) {
        // Pre-strip `<[]>` from `llzk.nondet` result types before template
        // removal. The upstream pass walks only the ops in its
        // `OpClassesWithStructTypes` tuple; `llzk.nondet` is not in that
        // list, so any SSA value SSC's dispatch cleanup synthesized above
        // with a `!struct.type<@X::@X<[]>>` result would keep its
        // pre-strip form, produce an unrealized `conversion_cast` against
        // the stripped form expected by downstream `struct.readm`, and
        // bail the pass's `applyFullConversion`.
        stripEmptyStructParams(module);

        OpPassManager pm("builtin.module");
        pm.addPass(llzk::polymorphic::createEmptyTemplateRemoval());
        if (failed(runPipeline(pm, module))) {
          signalPassFailure();
          return;
        }

        // Post-step: project-llzk/circom PR #378 wraps every emitted
        // function.def / struct.def in a same-named `poly.template` so
        // upstream passes can track polymorphic typing. After
        // EmptyTemplateRemoval converts those to `builtin.module @X
        // { function.def @X }` (or `struct.def @X`), the inner symbol
        // collides with the wrapping module on the next pass that walks
        // the parent's symbol table (LlzkToStablehlo trips with
        // "redefinition of symbol named '<X>'"). Hoist each child out of
        // its single-purpose wrapper, then erase the wrapper. Symbol
        // refs still resolve because the inner @X kept its name and the
        // wrapper had no semantically-load-bearing identity.
        flattenSingleEntityWrapperModules(module);
      }
    }

    // NOTE: constrain function body clearing causes crashes in circuits
    // with sub-component function.call chains (multimimc7, mimcsponge_wrap).
    // Left as future work — constrain clearing needs careful handling of
    // cross-function references and verification constraints.
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
