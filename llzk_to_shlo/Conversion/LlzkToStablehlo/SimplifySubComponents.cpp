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
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_SIMPLIFYSUBCOMPONENTS
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h.inc"

namespace {

// ===----------------------------------------------------------------------===
// Helper functions
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

/// Stronger variant of `isAllResultsUnused` for region-bearing candidates of
/// Phase 4 erase. Also rejects when any op inside `op`'s regions has a result
/// consumed by a user OUTSIDE `op`. Phase 1 (`extractCallsFromScfIf`) hoists a
/// `function.call` before an `scf.if` using `trackedPodValues` operands that
/// may still be defined inside the scf.if body (e.g. an `array.extract` whose
/// result the hoisted call consumes). The scf.if's own pod-array result is
/// then unused, but its body contains a live producer; region-clearing it
/// destroys an op that still has external users.
bool isOpAndNestedResultsExternallyUnused(Operation &op) {
  if (!isAllResultsUnused(op))
    return false;
  // Fast path: ops with no regions (the common case in Phase 4 — pod.read,
  // pod.write, arith.constant, dispatch counters) can't have inner external
  // users by construction.
  if (op.getNumRegions() == 0)
    return true;
  for (Region &r : op.getRegions()) {
    WalkResult walk = r.walk([&](Operation *inner) {
      for (Value v : inner->getResults())
        for (Operation *user : v.getUsers())
          if (!op.isAncestor(user))
            return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return false;
  }
  return true;
}

/// True iff cloning `def` before `guardOp` is safe — i.e. the clone reads
/// the same value its original location would read. Two categories qualify:
/// (1) ops without memory effects (`Pure` trait or empty
/// `MemoryEffectOpInterface`), and (2) LLZK's read-only array access ops
/// (`array.read` / `array.extract` / `array.len`). These read mutable LLZK
/// arrays so they are NOT `Pure`, but hoisting them out of `guardOp` is
/// nevertheless safe because the clone reads the array operand BEFORE
/// `guardOp` executes — any writes that `guardOp`'s body would perform
/// haven't happened yet at the clone's new position.
bool isSafeToCloneBefore(Operation *def) {
  if (mlir::isMemoryEffectFree(def))
    return true;
  StringRef name = def->getName().getStringRef();
  return name == "array.read" || name == "array.extract" || name == "array.len";
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
                            unsigned depth = 8) {
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

/// Create an llzk.nondet operation producing an uninitialized value.
Value createNondet(OpBuilder &builder, Location loc, Type type) {
  OperationState state(loc, "llzk.nondet");
  state.addTypes({type});
  return builder.create(state)->getResult(0);
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

/// True for types that participate in pod-array per-field flattening:
/// `!felt.type` or `!array.type<... x !felt.type>`.
bool isFlattenableFelt(Type ty) {
  if (ty.getDialect().getNamespace() == "felt")
    return true;
  if (auto at = dyn_cast<llzk::array::ArrayType>(ty))
    return at.getElementType().getDialect().getNamespace() == "felt";
  return false;
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

/// Discover pod field names and types from pod.read/pod.write ops that
/// reference `podValue`. Walks ops via `walkFn` and populates `fieldOrder`
/// and `fieldTypes` in discovery order.
///
/// If `feltOnly` is true, only fields with felt-dialect types are recorded.
void discoverPodFields(
    function_ref<void(function_ref<void(Operation *)>)> walkFn, Value podValue,
    SmallVector<StringRef> &fieldOrder, llvm::StringMap<Type> &fieldTypes,
    bool feltOnly = false) {
  auto discover = [&](StringRef name, Type type) {
    if (fieldTypes.count(name))
      return;
    if (feltOnly && type.getDialect().getNamespace() != "felt")
      return;
    fieldTypes[name] = type;
    fieldOrder.push_back(name);
  };

  walkFn([&](Operation *op) {
    if (op->getName().getStringRef() == "pod.read" &&
        op->getNumOperands() > 0 && op->getOperand(0) == podValue &&
        op->getNumResults() > 0) {
      auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn)
        discover(rn.getValue(), op->getResult(0).getType());
    }
    if (op->getName().getStringRef() == "pod.write" &&
        op->getNumOperands() >= 2 && op->getOperand(0) == podValue) {
      auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn)
        discover(rn.getValue(), op->getOperand(1).getType());
    }
  });
}

/// Discover pod fields from direct users of a Value (not walking nested ops).
void discoverPodFieldsFromUsers(Value podValue,
                                SmallVector<StringRef> &fieldOrder,
                                llvm::StringMap<Type> &fieldTypes) {
  auto walkUsers = [&](function_ref<void(Operation *)> callback) {
    for (OpOperand &use : podValue.getUses())
      callback(use.getOwner());
  };
  discoverPodFields(walkUsers, podValue, fieldOrder, fieldTypes);
}

/// Replace pod.read/pod.write on `podValue` within `blk`.
/// - pod.read: replace result with current value from `fieldValues`
/// - pod.write: update `fieldValues` (only if in `parentRegion`)
/// Erases replaced ops.
void replacePodOpsOnValue(Block &blk, Value podValue, Region *parentRegion,
                          llvm::StringMap<Value> &fieldValues) {
  SmallVector<Operation *> toErase;
  blk.walk([&](Operation *op) {
    StringRef opName = op->getName().getStringRef();
    if (opName == "pod.read" && op->getNumOperands() > 0 &&
        op->getOperand(0) == podValue && op->getNumResults() > 0) {
      auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue())) {
        op->getResult(0).replaceAllUsesWith(fieldValues[rn.getValue()]);
        toErase.push_back(op);
      }
    } else if (opName == "pod.write" && op->getNumOperands() >= 2 &&
               op->getOperand(0) == podValue) {
      auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue())) {
        if (op->getParentRegion() == parentRegion)
          fieldValues[rn.getValue()] = op->getOperand(1);
        toErase.push_back(op);
      }
    }
  });
  for (auto *op : toErase)
    op->erase();
}

/// Expand terminator operand at carry position `podArrIdx`.
/// For scf.condition: operands are [pred, arg0, arg1, ...] → index + 1.
/// For scf.yield: operands are [val0, val1, ...] → index directly.
void expandTerminatorArg(Block &blk, unsigned podArrIdx,
                         ArrayRef<StringRef> fieldOrder,
                         llvm::StringMap<Value> &fieldValues) {
  Operation *term = blk.getTerminator();
  unsigned offset = isa<scf::ConditionOp>(term) ? 1 : 0;
  unsigned expandIdx = podArrIdx + offset;
  SmallVector<Value> newArgs;
  for (unsigned i = 0; i < term->getNumOperands(); ++i) {
    if (i == expandIdx) {
      for (StringRef fn : fieldOrder)
        newArgs.push_back(fieldValues[fn]);
    } else {
      newArgs.push_back(term->getOperand(i));
    }
  }
  term->setOperands(newArgs);
}

/// Process post-while pod users: replace pod.read with expanded field values,
/// track pod.write updates, erase struct.writem. Processes top-level users
/// (same block as `whileBlock`) in program order, then nested users.
void replacePostWhilePodUsers(Value oldPodResult, Block *whileBlock,
                              ArrayRef<StringRef> fieldOrder,
                              llvm::StringMap<Value> &fieldValues) {
  SmallVector<Operation *> topLevelUsers, nestedUsers;
  for (OpOperand &use : oldPodResult.getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() == whileBlock)
      topLevelUsers.push_back(user);
    else
      nestedUsers.push_back(user);
  }
  llvm::sort(topLevelUsers,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  SmallVector<Operation *> toErase;
  for (Operation *user : topLevelUsers) {
    StringRef userName = user->getName().getStringRef();
    if (userName == "pod.write") {
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && user->getNumOperands() >= 2)
        fieldValues[rn.getValue()] = user->getOperand(1);
      toErase.push_back(user);
    } else if (userName == "pod.read") {
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue()))
        user->getResult(0).replaceAllUsesWith(fieldValues[rn.getValue()]);
      toErase.push_back(user);
    } else if (userName == "struct.writem") {
      toErase.push_back(user);
    }
  }
  for (Operation *user : nestedUsers) {
    StringRef userName = user->getName().getStringRef();
    if (userName == "pod.read") {
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue()))
        user->getResult(0).replaceAllUsesWith(fieldValues[rn.getValue()]);
      toErase.push_back(user);
    } else if (userName == "pod.write") {
      toErase.push_back(user);
    }
  }
  for (auto *op : toErase)
    op->erase();
}

/// Expand block args for a while region: insert per-field args at `podIdx`,
/// replace pod.read/pod.write, fix terminator, erase old pod arg.
void expandWhileRegionArgs(Region &region, unsigned podIdx,
                           ArrayRef<StringRef> fieldOrder,
                           llvm::StringMap<Type> &fieldTypes, Location loc) {
  Block &blk = region.front();
  Value oldPodArg = blk.getArgument(podIdx);

  // Insert new field args after podIdx.
  llvm::StringMap<Value> fieldValues;
  for (size_t f = 0; f < fieldOrder.size(); ++f) {
    auto arg =
        blk.insertArgument(podIdx + 1 + f, fieldTypes[fieldOrder[f]], loc);
    fieldValues[fieldOrder[f]] = arg;
  }

  replacePodOpsOnValue(blk, oldPodArg, &region, fieldValues);
  expandTerminatorArg(blk, podIdx, fieldOrder, fieldValues);
  blk.eraseArgument(podIdx);
}

/// Replace non-pod results of `oldWhile` with corresponding results from
/// `newWhile`, skipping `podIdx` (which was expanded into `numFields` results).
void replaceNonPodWhileResults(scf::WhileOp oldWhile, scf::WhileOp newWhile,
                               unsigned podIdx, unsigned numFields) {
  for (unsigned i = 0, ni = 0; i < oldWhile.getNumResults(); ++i) {
    if (i == podIdx) {
      ni += numFields;
    } else {
      oldWhile.getResult(i).replaceAllUsesWith(newWhile.getResult(ni));
      ni++;
    }
  }
}

// ===----------------------------------------------------------------------===
// Phase functions
// ===----------------------------------------------------------------------===

/// Phase 1: Scan block, track pod field values and extract function.call
/// from scf.if into the parent block.
bool extractCallsFromScfIf(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues) {
  bool changed = false;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    StringRef name = op.getName().getStringRef();

    if (name == "pod.new") {
      if (op.getNumResults() > 0) {
        trackedPodValues[op.getResult(0)] = {};
        auto fieldNames = getPodInitializedRecords(&op);
        for (auto [idx, fn] : llvm::enumerate(fieldNames)) {
          if (idx < op.getNumOperands())
            trackedPodValues[op.getResult(0)][fn] = op.getOperand(idx);
        }
      }
    } else if (name == "array.read") {
      // Track pods from array.read (array-of-pods dispatch pattern).
      if (op.getNumResults() > 0 &&
          op.getResult(0).getType().getDialect().getNamespace() == "pod")
        trackedPodValues[op.getResult(0)] = {};
    } else if (name == "pod.write") {
      auto field = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
      // Skip @count writes: count has circular dependency
      // (count = subi(count, 1)). Keep the initial value from pod.new.
      if (field && op.getNumOperands() >= 2 && field.getValue() != "count") {
        Value val = op.getOperand(1);
        Value pod = op.getOperand(0);
        // Detect read-modify-write: pod.write %pod[@f] = (pod.read %pod[@f]).
        // LLZK arrays are mutable, so the read and write refer to the same
        // object. When we already have a tracked value, skip the redundant
        // write-back to keep the earliest (dominating) definition.
        bool isReadBack = false;
        if (auto *def = val.getDefiningOp()) {
          if (def->getName().getStringRef() == "pod.read") {
            auto rf = def->getAttrOfType<FlatSymbolRefAttr>("record_name");
            if (rf && rf.getValue() == field.getValue() &&
                def->getNumOperands() > 0 && def->getOperand(0) == pod) {
              auto pit = trackedPodValues.find(pod);
              if (pit != trackedPodValues.end() &&
                  pit->second.count(field.getValue()))
                isReadBack = true;
            }
          }
        }
        if (!isReadBack)
          trackedPodValues[pod][field.getValue()] = val;
      }
    } else if (name == "scf.if") {
      // Extract function.call @compute from inside scf.if.
      // Build a new call BEFORE the scf.if using pod-tracked inputs.
      SymbolRefAttr calleeRef = nullptr;
      SmallVector<Type> resultTypes;
      // Track call args: either from pod.read or direct values
      SmallVector<std::pair<Value, StringRef>> inputPodFields;
      SmallVector<Value> directArgs; // non-pod.read args (already resolved)
      bool hasDirectArgs = false;
      Value compPod;

      op.walk([&](Operation *nested) {
        StringRef nn = nested->getName().getStringRef();
        if (nn == "function.call" && !calleeRef) {
          calleeRef = nested->getAttrOfType<SymbolRefAttr>("callee");
          for (Type t : nested->getResultTypes())
            resultTypes.push_back(t);
          for (Value arg : nested->getOperands()) {
            if (auto *def = arg.getDefiningOp()) {
              if (def->getName().getStringRef() == "pod.read") {
                auto rn = def->getAttrOfType<FlatSymbolRefAttr>("record_name");
                // Get the specific pod this reads from
                Value srcPod =
                    def->getNumOperands() > 0 ? def->getOperand(0) : Value();
                if (rn && srcPod) {
                  inputPodFields.push_back({srcPod, rn.getValue()});
                  continue;
                }
              }
            }
            // Non-pod.read arg: use directly (already resolved by unpack)
            directArgs.push_back(arg);
            hasDirectArgs = true;
          }
        }
        if (nn == "pod.write") {
          auto fn = nested->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (fn && fn.getValue() == "comp" && nested->getNumOperands() >= 1)
            compPod = nested->getOperand(0);
        }
      });

      if (calleeRef && !resultTypes.empty()) {
        // Collect arguments: resolve pod.read args from tracked values,
        // use direct args as-is.
        SmallVector<Value> args;
        if (hasDirectArgs && inputPodFields.empty()) {
          // All args are direct (already resolved by pod unpack)
          args = directArgs;
        } else {
          for (auto &[srcPod, fieldName] : inputPodFields) {
            auto pit = trackedPodValues.find(srcPod);
            if (pit != trackedPodValues.end()) {
              auto fit = pit->second.find(fieldName);
              if (fit != pit->second.end()) {
                args.push_back(fit->second);
                continue;
              }
            }
            break; // couldn't resolve
          }
        }

        // directArgs path: when args are taken directly from the inner
        // call's operands (no pod.read resolution through trackedPodValues),
        // an arg defined INSIDE op's regions does not dominate the hoisted
        // new call. Two options for that case:
        //
        // (1) Skip the hoist (pre-existing fallback): keeps the pass
        //     idempotent — the inner call survives statically-false
        //     scf.if dispatch and is later DCE'd by `--llzk-to-stablehlo`,
        //     dropping the sub-component compute. Necessary fallback for
        //     un-hoistable chains.
        //
        // (2) Clone-hoist the directArg's defining-op chain (pure /
        //     LLZK read-only ops only) BEFORE op so the hoisted call is
        //     valid. Required for `convertStructOfPodsToArrayOfPods`'s
        //     output shape, where the dispatched call sees
        //     `array.extract %arg7[%c0]` defined inside op — both
        //     operands are outside, so the extract is hoist-safe.
        //     Idempotence is preserved by erasing the inner function.call
        //     after a successful clone-hoist (RAUWing its result with the
        //     hoisted call's result); subsequent fixed-point iterations
        //     find no inner `function.call` to hoist.
        bool needsCloneHoist = hasDirectArgs && inputPodFields.empty() &&
                               llvm::any_of(args, [&](Value v) {
                                 return isValueDefinedInside(v, op);
                               });
        bool cloneHoisted = false;
        if (needsCloneHoist) {
          llvm::DenseMap<Value, Value> cloneCache;
          SmallVector<Value> hoistedArgs;
          hoistedArgs.reserve(args.size());
          bool allHoisted = true;
          for (Value a : args) {
            Value cloned = cloneDefiningOpBefore(a, &op, op, cloneCache);
            if (!cloned) {
              allHoisted = false;
              break;
            }
            hoistedArgs.push_back(cloned);
          }
          if (allHoisted) {
            args = std::move(hoistedArgs);
            cloneHoisted = true;
          }
        }
        bool dominatesScfIf = !needsCloneHoist || cloneHoisted;
        if (dominatesScfIf && !args.empty() &&
            (hasDirectArgs || args.size() == inputPodFields.size())) {
          // Create function.call using LLZK's CallOp builder API.
          OpBuilder builder(&op);
          Operation *newCall = builder.create<llzk::function::CallOp>(
              op.getLoc(), resultTypes, calleeRef, args);

          // Track: pod[@comp] = newCall result
          if (newCall->getNumResults() > 0 && compPod)
            trackedPodValues[compPod]["comp"] = newCall->getResult(0);

          // Idempotence: after a clone-hoist, erase the inner function.call
          // and RAUW its result with the hoisted call's result. The next
          // outer fixed-point iter then sees no function.call inside op and
          // skips re-hoisting. The surrounding dispatch scf.if becomes
          // externally-unused (its other body ops only feed the dead inner
          // call) and Phase 4 erases it (unless body has non-pod array.write,
          // in which case op is preserved — still safe, no inner call to
          // re-hoist).
          if (cloneHoisted) {
            op.walk([&](Operation *nested) {
              if (nested->getName().getStringRef() != "function.call")
                return WalkResult::advance();
              if (nested->getAttrOfType<SymbolRefAttr>("callee") != calleeRef)
                return WalkResult::advance();
              if (nested->getNumResults() == newCall->getNumResults()) {
                for (auto [oldR, newR] :
                     llvm::zip(nested->getResults(), newCall->getResults()))
                  oldR.replaceAllUsesWith(newR);
              }
              nested->erase();
              return WalkResult::interrupt();
            });
          }

          changed = true;
        }
      }
    }
  }

  return changed;
}

/// Resolve `trackedPodValues[startPod][startField]` transitively to a value
/// that is NOT another tracker-resolvable `pod.read` result. The tracker
/// stores raw `pod.write %P[@f] = %v` values (Phase 1) which may themselves
/// be `pod.read` results — Phase 2 RAUWs (and erases) those pod.reads in
/// the same walk, so resolving to a single-step target risks rebinding uses
/// onto an op that's about to be destroyed. Chain-walking to the terminal
/// keeps `replaceAllUsesWith` honoring values that survive the erase loop.
/// Returns a null `Value()` on cycle detection so the caller can skip the
/// RAUW outright instead of resolving to a value still inside the cycle.
Value resolveTrackedPodValueTransitive(
    Value initial,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues) {
  Value terminal = initial;
  llvm::SmallDenseSet<Value> seen;
  seen.insert(terminal);
  while (auto *def = terminal.getDefiningOp()) {
    if (def->getName().getStringRef() != "pod.read" ||
        def->getNumOperands() == 0 || def->getNumResults() == 0)
      break;
    auto rf = def->getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!rf)
      break;
    auto pit = trackedPodValues.find(def->getOperand(0));
    if (pit == trackedPodValues.end())
      break;
    auto fit = pit->second.find(rf.getValue());
    if (fit == pit->second.end())
      break;
    if (fit->second == terminal)
      break; // self-reference — Phase 1's read-back guard normally
             // prevents this, but bail conservatively if it slips through.
    if (!seen.insert(fit->second).second)
      return Value(); // cycle — caller skips RAUW rather than rebind onto
                      // a value still inside the cycle.
    terminal = fit->second;
  }
  return terminal;
}

/// Phase 2: Replace pod.read results with tracked values.
/// Walks ALL ops including nested regions (scf.if body) to ensure all
/// pod.read references are replaced, making the scf.if erasable.
bool replacePodReads(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues) {
  bool changed = false;

  SmallVector<Operation *> toErase;
  block.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "pod.read" || op->getNumResults() == 0)
      return;
    auto field = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!field || op->getNumOperands() == 0)
      return;
    auto pit = trackedPodValues.find(op->getOperand(0));
    if (pit == trackedPodValues.end())
      return;
    auto fit = pit->second.find(field.getValue());
    if (fit == pit->second.end())
      return;
    // Chain-resolve: tracker target may itself be a pod.read this walk
    // will erase. See `resolveTrackedPodValueTransitive` for the
    // use-after-erase ordering argument.
    Value terminal =
        resolveTrackedPodValueTransitive(fit->second, trackedPodValues);
    // Skip if the chain hit a cycle (terminal is null) or cycles back to
    // this pod.read's own result.
    if (!terminal || terminal == op->getResult(0))
      return;
    op->getResult(0).replaceAllUsesWith(terminal);
    toErase.push_back(op);
    changed = true;
  });
  for (auto *op : toErase)
    op->erase();

  return changed;
}

/// Phase 3: Erase struct.writem that writes pod/struct-typed values
/// (sub-component bookkeeping, not needed for witness generation).
bool eraseStructWritemForPodValues(Block &block) {
  bool changed = false;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.getName().getStringRef() != "struct.writem" ||
        op.getNumOperands() < 2)
      continue;
    Type valType = op.getOperand(1).getType();
    StringRef ns = valType.getDialect().getNamespace();
    if (ns == "pod" || ns == "struct") {
      op.erase();
      changed = true;
    }
  }

  return changed;
}

/// Returns true if `op` has any nested `array.write` whose target array's
/// element type is not a `!pod.type<...>`. Such writes are real
/// side-effecting witness emission (the struct-element drain pattern that
/// circom emits for sub-component aggregation fields, e.g. AES
/// `@bits2num_1` / `@xor_2`); 0-iter-arg `scf.for` fill loops produce no
/// SSA results, so without this guard `isAllResultsUnused` would drop the
/// loop and silently erase the writes. Covers both `array.write` (felt-element
/// fields) and `array.insert` (array-element fields, e.g. maci's `@bits :
/// !array<104 x !struct>`), since `flattenPodArrayWhileCarry` emits whichever
/// matches the field type.
bool hasNonPodArrayWriteInBody(Operation &op) {
  return op
      .walk([](Operation *inner) {
        StringRef name = inner->getName().getStringRef();
        if ((name != "array.write" && name != "array.insert") ||
            inner->getNumOperands() == 0)
          return WalkResult::advance();
        auto arrTy =
            dyn_cast<llzk::array::ArrayType>(inner->getOperand(0).getType());
        if (!arrTy || isa<llzk::pod::PodType>(arrTy.getElementType()))
          return WalkResult::advance();
        return WalkResult::interrupt();
      })
      .wasInterrupted();
}

/// Index operands of an LLZK `array.read` / `array.write` op. The first
/// operand is the array; everything after is the index list.
SmallVector<Value> arrayAccessIndices(Operation *arrayAccess) {
  return llvm::to_vector(llvm::drop_begin(arrayAccess->getOperands()));
}

/// Phase 4: Iteratively erase dead ops that are not core computation.
/// Core computation: felt.*, struct.*, array.*, function.*, scf.while/yield.
/// Everything else (pod.*, arith.*, bool.*, scf.if) is erased if unused.
bool eraseDeadPodAndCountOps(Block &block) {
  bool changed = false;

  bool erasing = true;
  while (erasing) {
    erasing = false;
    for (Operation &op : llvm::make_early_inc_range(block)) {
      // Keep core computation ops
      StringRef ns = op.getName().getDialectNamespace();
      StringRef name = op.getName().getStringRef();
      bool isCore =
          (ns == "felt" || ns == "struct" || ns == "array" ||
           ns == "function" || name == "func.call" || name == "func.return");
      // scf.while/yield/condition are computation; scf.if/for are dispatch
      if (ns == "scf" && name != "scf.if" && name != "scf.for")
        isCore = true;
      if (isCore)
        continue;

      // scf.for / scf.if whose body writes to a non-pod-element array is
      // doing real work, not dispatch bookkeeping — preserve it so the
      // side-effecting array.write reaches the rest of the pipeline. The
      // scf.if case covers `flattenPodArrayWhileCarry`'s rewrite for
      // multi-record input pods whose per-iteration value source is itself
      // wrapped in an scf.if (e.g. webb's @ManyMerkleProof_275 @switcher
      // input @L: parent's input on iter 0, Poseidon[i-1].@out on iter
      // i>0). The rewrite turns `pod.write %cell[@L] = %src` into
      // `array.write %perFieldArr[%i] = %src` inside a now-void-result
      // scf.if. `isOpAndNestedResultsExternallyUnused` only inspects
      // inner ops' SSA *results*, so the side-effecting array.write into
      // an outer-defined `%perFieldArr` is invisible to it.
      if ((name == "scf.for" || name == "scf.if") &&
          hasNonPodArrayWriteInBody(op))
        continue;

      // Preserve `pod.write %arr_elem[@user_field] = %src` rewrite-back
      // chains until `flattenPodArrayWhileCarry` converts them to
      // `array.insert`. Otherwise the pod.write — which has zero SSA
      // results and thus is "vacuously unused" — gets erased between
      // `materializePodArrayInputPodField`'s RAUW of the firing-site
      // pod.read and flatten's per-iter-arg pass, severing the chain.
      // Dispatch protocol fields (@count/@comp/@params) are still erased.
      if (name == "pod.write" && op.getNumOperands() >= 2) {
        auto rn = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
        Operation *cellDef = op.getOperand(0).getDefiningOp();
        bool isUserInputArrayWrite =
            rn && cellDef &&
            cellDef->getName().getStringRef() == "array.read" &&
            rn.getValue() != "count" && rn.getValue() != "comp" &&
            rn.getValue() != "params";
        if (isUserInputArrayWrite)
          continue;
      }

      if (isOpAndNestedResultsExternallyUnused(op)) {
        // Clear nested regions to avoid LLZK verifier crashes during erase.
        for (Region &r : op.getRegions())
          r.dropAllReferences();
        for (Region &r : op.getRegions())
          r.getBlocks().clear();
        op.erase();
        erasing = true;
        changed = true;
      }
    }
  }

  return changed;
}

/// Phase 5: Replace remaining pod.read (self-referential sources) with
/// llzk.nondet, then erase orphaned pod.new ops.
/// After phases 1-4, the only surviving pod ops are pod.new + pod.read pairs
/// where the pod.read provides the initial mutable value (e.g., an array that
/// is later modified in-place by array.write). Since the array is fully
/// overwritten before meaningful use, the initial value is don't-care.
bool replaceRemainingPodOps(Block &block) {
  bool changed = false;

  // Replace pod.read with llzk.nondet (uninitialized value).
  SmallVector<Operation *> toErase;
  for (Operation &op : block) {
    if (op.getName().getStringRef() != "pod.read" || op.getNumResults() == 0)
      continue;
    OpBuilder builder(&op);
    Value nondet =
        createNondet(builder, op.getLoc(), op.getResult(0).getType());
    op.getResult(0).replaceAllUsesWith(nondet);
    toErase.push_back(&op);
    changed = true;
  }
  for (auto *op : toErase)
    op->erase();

  // Erase pod.new whose results are now unused.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.getName().getStringRef() != "pod.new")
      continue;
    if (isAllResultsUnused(op)) {
      op.erase();
      changed = true;
    }
  }

  return changed;
}

/// Materialize a per-struct-field felt array for every function-scope
/// `array.new : <N x !pod<[..., @comp: !struct, ...]>>` whose `@comp`
/// slots are written by a hoisted `function.call` result in one block and
/// read back in a *different* block via `pod.read [@comp]; struct.readm
/// [@F] : !felt`.
///
/// Circomlib's `gates.circom` `XorArray*` / `AndArray*` family (used in
/// keccak chi/iota/rhopi/round/squeeze/theta + AES + iden3 + SHA-256)
/// emits the writer/reader split as two sibling `scf.while` ops over the
/// same dispatch-pod array. `extractCallsFromScfIf`'s tracker is keyed by
/// pod SSA value, and the writer's `%dp = array.read %arr[%i]` produces a
/// different SSA value than the reader's `%dp2 = array.read %arr[%j]`,
/// so the call result is not forwarded across the array slot. Without
/// intervention `rewriteArrayPodCountCompInReads` (and Phase 5 as a
/// backstop) substitutes the reader's `pod.read %dp2[@comp]` with
/// `llzk.nondet`, severing the data flow and silently emitting zeros.
///
/// This pass allocates a per-struct-field `%arr_F = array.new : <N x
/// !felt>` parallel to `%arr` for each field `F` consumed by a
/// `struct.readm` reader, hoists the dispatch `function.call` out of the
/// writer's `scf.if`, inserts `array.write %arr_F[%i] = struct.readm
/// %callResult[@F]` after the scf.if at body level, and rewrites every
/// reader-side `struct.readm [@F]` into `array.read %arr_F[%j]`. The
/// original pod array's `pod.write @comp` / `pod.read @comp` / `array.write`
/// traffic is left in place — Phase 4 DCEs the dispatch-firing scf.if (its
/// only effect was the dead pod.write/array.write pair) along with the
/// reader-side pod.read once its struct.readm consumer is gone.
///
/// We materialize at the felt level (not at the !struct level) because
/// `convertWhileBodyArgsToSSA` (in `LlzkToStablehlo.cpp`) promotes any
/// captured non-pod-element array used inside `scf.while` to an iter-arg
/// and converts each `array.write` to a result-bearing form. `array.write`
/// is `ZeroResults`, so the result-bearing rewrite only verifies once
/// `ArrayWritePattern` rewrites it to `stablehlo.dynamic_update_slice`.
/// `ArrayWritePattern` is gated on `involvesPod`, which treats `!struct`
/// element arrays as pod-involved (legal, *not* converted) — so a
/// `<N x !struct>` array.write inside an scf.while body is left
/// result-bearing and trips the verifier. The felt element type sidesteps
/// that path entirely.
///
/// The transform fires only when at least one writer (call-result-paired
/// `pod.write [@comp]` directly preceding `array.write %arr[*]`) and at
/// least one cross-block reader whose `pod.read [@comp]` is consumed by a
/// `struct.readm [@F]` exist. Same-block readers are gated out — those
/// are already forwarded by `resolveArrayPodCompReads` via the
/// per-block call-result-by-type map. Readers whose `pod.read [@comp]`
/// is consumed by something other than `struct.readm` (e.g. an
/// `array.write %array_3[%i] = %comp` extracting the struct into a
/// `struct.member @aux` array) are also skipped — those represent
/// don't-care bookkeeping that the existing nondet path handles
/// correctly. Idempotent: subsequent invocations find no readers and
/// skip.
bool materializePodArrayCompField(Block &funcBlock) {
  bool changed = false;

  // Collect candidate function-scope pod-arrays whose `@comp` field is a
  // struct type (the only case where the cross-while bug currently bites
  // — `gates.circom` helpers all dispatch a struct-returning component).
  struct Candidate {
    Operation *arrNew;
    Type compType;
  };
  SmallVector<Candidate> candidates;
  for (Operation &op : funcBlock) {
    if (op.getName().getStringRef() != "array.new" || op.getNumResults() == 0)
      continue;
    auto arrTy = dyn_cast<llzk::array::ArrayType>(op.getResult(0).getType());
    if (!arrTy)
      continue;
    auto podTy = dyn_cast<llzk::pod::PodType>(arrTy.getElementType());
    if (!podTy)
      continue;
    Type compType;
    for (auto rec : podTy.getRecords())
      if (rec.getName() == "comp") {
        compType = rec.getType();
        break;
      }
    if (compType && compType.getDialect().getNamespace() == "struct")
      candidates.push_back({&op, compType});
  }

  for (auto &cand : candidates) {
    Value arr = cand.arrNew->getResult(0);
    auto arrTy = cast<llzk::array::ArrayType>(arr.getType());

    struct Writer {
      SmallVector<Value> outerIndices; // body-scope indices used to read %arr.
      Operation *insertAfter;          // body-block ancestor of the writer.
      Value callResult;    // function.call result fed into pod.write[@comp].
      Operation *podWrite; // pod.write %cell[@comp] = %callResult, sited
                           // in writer's scf.if body. Captured here so the
                           // single-instance fold below can erase it
                           // directly without re-scanning the call's uses.
    };
    SmallVector<Writer> writers;

    // Cross-block reader: `%dp2 = array.read %arr[%j]; %comp = pod.read
    // %dp2[@comp]; struct.readm %comp[@F] : !felt`. We materialize one
    // per-field felt array per distinct F.
    struct Reader {
      Operation *arrayRead;   // %arr[%j] — its block + operand 1 (index)
                              // are the same-block guard key + the
                              // body-scope index for the rewritten read.
      Operation *structReadm; // the felt-yielding consumer to rewrite.
      StringRef field;        // struct field name, key into per-field arrays.
    };
    SmallVector<Reader> readers;
    llvm::StringMap<Type> fieldFeltTypes;

    // Cross-block drain reader: `%dp2 = array.read %arr[%j]; %comp =
    // pod.read %dp2[@comp]; array.write %destArr[%j] = %comp` where
    // `%destArr : array<D x !struct>` flows into `struct.writem
    // %self[@F'] = %destArr` on the parent struct. The parent member's
    // witness slot is sized by `getMemberFlatSize` as the array's dim
    // count (one felt per cell), so we materialize a parallel felt
    // array, populate it from the writer-side `function.call` results
    // by extracting the inner struct's single felt-typed member (e.g.
    // `@out`), and redirect the parent `struct.writem` to consume the
    // felt array. The original struct-array drain stays in place; its
    // `pod.read [@comp]` is nondet'd by Phase 5 as before, and the now-
    // unused `%destArr = array.new` flows into a writem that no longer
    // uses it (DCE'd later).
    struct DrainReader {
      Operation *arrayRead;     // %arr[%j] (the dispatch-pod array read).
      Operation *arrayWriteDst; // array.write %destArr[*] = %comp.
      Value destArr;            // %destArr (struct-element array).
      Operation *writem;        // struct.writem %self[@F'] = %destArr.
      StringRef parentField;    // @F' name (key into parent struct.def).
    };
    SmallVector<DrainReader> drainReaders;

    for (OpOperand &use : arr.getUses()) {
      Operation *user = use.getOwner();
      if (use.getOperandNumber() != 0)
        continue;
      StringRef name = user->getName().getStringRef();

      if (name == "array.write" && user->getNumOperands() >= 3) {
        // For multi-dim arrays the value is always the last operand,
        // preceded by 1+ index operands. Hard-coding `getOperand(2)`
        // would silently grab an index for any rank > 1.
        Value writtenPod = user->getOperand(user->getNumOperands() - 1);
        Operation *podWrite = nullptr;
        for (Operation *prev = user->getPrevNode(); prev;
             prev = prev->getPrevNode()) {
          if (prev->getName().getStringRef() != "pod.write" ||
              prev->getNumOperands() < 2 || prev->getOperand(0) != writtenPod)
            continue;
          auto rn = prev->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (rn && rn.getValue() == "comp") {
            podWrite = prev;
            break;
          }
        }
        if (!podWrite)
          continue;
        Operation *callDef = podWrite->getOperand(1).getDefiningOp();
        if (!callDef || callDef->getName().getStringRef() != "function.call")
          continue;

        // %writtenPod must be defined by an outer `array.read %arr[idx...]`.
        Operation *podDef = writtenPod.getDefiningOp();
        if (!podDef || podDef->getName().getStringRef() != "array.read" ||
            podDef->getNumOperands() < 2 || podDef->getOperand(0) != arr)
          continue;
        Block *bodyBlock = podDef->getBlock();
        SmallVector<Value> outerIndices = arrayAccessIndices(podDef);

        // Walk up from `user` to the immediate ancestor that lives in
        // bodyBlock; that's where the writer materialization will go.
        Operation *ancestor = user;
        while (ancestor && ancestor->getBlock() != bodyBlock)
          ancestor = ancestor->getParentOp();
        if (!ancestor)
          continue;

        writers.push_back({std::move(outerIndices), ancestor,
                           podWrite->getOperand(1), podWrite});
        continue;
      }

      if (name == "array.read" && user->getNumResults() > 0 &&
          user->getNumOperands() >= 2) {
        Value podVal = user->getResult(0);
        for (OpOperand &subUse : podVal.getUses()) {
          Operation *subUser = subUse.getOwner();
          if (subUser->getName().getStringRef() != "pod.read" ||
              subUser->getNumResults() == 0)
            continue;
          auto rn = subUser->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (!rn || rn.getValue() != "comp")
            continue;
          // Collect every felt-yielding `struct.readm [@F]` consumer of
          // this `pod.read [@comp]`. Multi-field structs (e.g. a gate
          // with both @out and @aux fields read in loop2) need every
          // field materialized; non-`struct.readm` consumers (e.g. the
          // `array.write %array_3[%i] = %comp` pattern that drains the
          // struct into a `struct.member @aux` array) are @constrain
          // bookkeeping and stay don't-care — the existing
          // `rewriteArrayPodCountCompInReads` nondets the surviving
          // pod.read correctly for them.
          Value compVal = subUser->getResult(0);
          for (OpOperand &compUse : compVal.getUses()) {
            Operation *consumer = compUse.getOwner();
            StringRef consumerName = consumer->getName().getStringRef();

            if (consumerName == "struct.readm" &&
                consumer->getNumResults() > 0) {
              auto memberAttr =
                  consumer->getAttrOfType<FlatSymbolRefAttr>("member_name");
              if (!memberAttr)
                continue;
              Type feltTy = consumer->getResult(0).getType();
              // Accept scalar `!felt` OR `!array<... x !felt>`. The array
              // case (e.g. `@Num2Bits_2.@out : !array<32 x !felt>`) needs
              // the per-field array's dims to combine the dispatch dims
              // with the field's inner dims, and writer/reader emit
              // `array.insert` / `array.extract` to move whole sub-arrays
              // — see writer/reader sites below for the dispatch logic.
              bool isFelt = feltTy.getDialect().getNamespace() == "felt";
              bool isFeltArr = false;
              if (auto arrFieldTy = dyn_cast<llzk::array::ArrayType>(feltTy))
                isFeltArr =
                    arrFieldTy.getElementType().getDialect().getNamespace() ==
                    "felt";
              if (!isFelt && !isFeltArr)
                continue;
              StringRef fieldName = memberAttr.getValue();
              // All readers of the same field must agree on the field type.
              auto it = fieldFeltTypes.find(fieldName);
              if (it == fieldFeltTypes.end())
                fieldFeltTypes[fieldName] = feltTy;
              else if (it->second != feltTy)
                continue;
              readers.push_back({user, consumer, fieldName});
              continue;
            }

            // Drain consumer: `array.write %destArr[*] = %comp` where the
            // value is `%comp` (last operand) and the destination's
            // element type is a struct. The destination must flow into
            // a single `struct.writem %self[@F'] = %destArr` on the
            // parent struct — only then is there a witness slot to
            // populate.
            if (consumerName == "array.write" &&
                consumer->getNumOperands() >= 3 &&
                consumer->getOperand(consumer->getNumOperands() - 1) ==
                    compVal) {
              Value destArr = consumer->getOperand(0);
              auto destArrTy =
                  dyn_cast<llzk::array::ArrayType>(destArr.getType());
              if (!destArrTy)
                continue;
              if (!isa<llzk::component::StructType>(destArrTy.getElementType()))
                continue;
              Operation *writem = nullptr;
              StringRef parentField;
              for (OpOperand &dstUse : destArr.getUses()) {
                Operation *dstUser = dstUse.getOwner();
                if (dstUser->getName().getStringRef() != "struct.writem" ||
                    dstUse.getOperandNumber() != 1)
                  continue;
                auto memberAttr =
                    dstUser->getAttrOfType<FlatSymbolRefAttr>("member_name");
                if (!memberAttr)
                  continue;
                if (writem) {
                  // Multiple struct.writems on the same destArr — bail
                  // (we'd need a single redirect target).
                  writem = nullptr;
                  break;
                }
                writem = dstUser;
                parentField = memberAttr.getValue();
              }
              if (!writem)
                continue;
              drainReaders.push_back(
                  {user, consumer, destArr, writem, parentField});
            }
          }
        }
      }
    }

    // Materialize struct.readm consumers in any block, including the
    // writer's body. The previous same-block prune assumed
    // `resolveArrayPodCompReads`'s call-result-by-type-per-block fallback
    // would forward same-block reads safely; that fallback misforwards
    // chained-call patterns where every chain link returns the same
    // struct type (chained-XOR @b-input is the canonical case — the
    // fallback routes every prior-link reader to the LAST call's @out).
    // The per-field array path indexes by the reader's own array.read
    // indices, so a prior-link read at index `i-1` correctly consumes
    // the iter-`i-1` writer's @out, not the latest writer's.
    //
    // Drain readers (struct.writem destinations) do not exhibit the
    // chained-call shape — each iteration drains its own dispatch-pod
    // cell into its own destArr slot — so the same-block prune still
    // applies to them.
    llvm::DenseSet<Block *> writerBodyBlocks;
    for (auto &w : writers)
      writerBodyBlocks.insert(w.insertAfter->getBlock());
    drainReaders.erase(llvm::remove_if(drainReaders,
                                       [&](const DrainReader &r) {
                                         return writerBodyBlocks.count(
                                             r.arrayRead->getBlock());
                                       }),
                       drainReaders.end());

    if (writers.empty() || (readers.empty() && drainReaders.empty()))
      continue;

    // Materialize one `%arr_F = array.new : <N x feltType>` per distinct
    // field name seen in cross-block readers, just after %arr. When the
    // field type is itself a felt array (e.g. `@out : !array<32 x !felt>`),
    // combine the dispatch dims with the field's inner dims so a single
    // `array.insert`/`array.extract` at outer indices stores/loads the
    // whole sub-array slice — same shape as `flattenPodArrayWhileCarry`'s
    // per-field array construction at line ~1843.
    OpBuilder builder(cand.arrNew);
    builder.setInsertionPointAfter(cand.arrNew);
    auto dims = getArrayDimensions(arrTy);
    llvm::StringMap<Value> perFieldArrays;
    for (auto &entry : fieldFeltTypes) {
      // Skip fields no reader actually targets after same-block pruning.
      bool stillUsed = false;
      for (auto &r : readers)
        if (r.field == entry.first()) {
          stillUsed = true;
          break;
        }
      if (!stillUsed)
        continue;
      auto perFieldArrTy = combineDispatchAndInnerFeltDims(entry.second, dims);
      Value arrField = builder.create<llzk::array::CreateArrayOp>(
          cand.arrNew->getLoc(), perFieldArrTy);
      perFieldArrays[entry.first()] = arrField;
    }

    // For each unique drain destination, plan to populate a parallel
    // felt array at the writer sites (extracting the inner struct's
    // single felt member from each call result), then redirect the
    // parent `struct.writem` to consume the felt array and flip the
    // `struct.member @F'` TypeAttr from `array<D x !struct>` to
    // `array<D x !felt>`. Direct struct-element `array.write` inside
    // `scf.while` bodies hits a known capture-then-lift hazard
    // (`processBlockForArrayMutations` lifts the write to SSA form,
    // but `ArrayWritePattern` is gated to skip pod/struct-element
    // arrays — leaving an SSA-form `array.write` that fails the LLZK
    // op def's zero-results constraint). Felt writes don't trip that
    // path: they get lifted to SSA, then `ArrayWritePattern` rewrites
    // to `stablehlo.dynamic_update_slice` cleanly. Flipping `@F'`'s
    // type cascades to `@constrain` whose `struct.readm @F' →
    // array.read → struct.readm @out` chain must be repaired in the
    // same pass; a downstream `function.call @Sub::@constrain(...)`
    // whose operand type would now mismatch the callee signature is
    // erased (`@constrain` itself is unreachable at witness-gen time
    // — `ConstrainFunctionErasePattern` deletes it during the main
    // conversion).
    //
    // The inner struct must expose at least one `{llzk.pub}` member,
    // each typed as scalar `!felt` (`@XOR_0`, `@Bits2Num_1`,
    // `@Switcher_206.@outL/@outR`) or `!array<M x !felt>` (AES
    // `@Num2Bits_5.@out`, `@BitElementMulAny_6.@dblOut/@addOut`). When
    // K=1, the parent's flat witness slot stays `<D x innerTy>` (sister
    // K=1 chips are byte-equal post-flip). When K>1, the K pub felt
    // members must share a single inner type (uniform-shape constraint
    // — both Switcher's all-scalar and BitElementMulAny's all-`<2 x
    // !felt>` shapes satisfy this); the parent member widens to
    // `<D, K x innerTy>` with one slot per pub field in declaration
    // order, matching circom's `.wtns` flat felt layout.
    struct PubFelt {
      StringRef field;
      Type ty;
    };
    // K=0 path: inner struct has zero pub felt members but at least one
    // writem-targeted, non-pod, non-zero-flat member. The inner contribution
    // is the concat of each writem-targeted member's recursive flat felts
    // (e.g. webb's `@ManyMerkleProof_275 → @hasher : <30 x !felt>` (30 felts)
    // + `@switcher : <30, 2 x !felt>` (60 felts) = 90 per instance). Member
    // shapes are *heterogeneous*, so each member's natural per-element
    // layout (row-major across its declared dims) is unrolled at writer
    // sites into per-element `array.write` into the flat destFelt at offset
    // `offsetWithinInstance + linear_idx`.
    struct WritemMember {
      StringRef field;
      Type ty;
      int64_t flatSize;
      int64_t offsetWithinInstance; // cumulative offset of this member
                                    // within one inner-struct instance
    };
    struct DrainPlan {
      Value destFelt; // Parallel destination — type follows
                      // `combineDispatchAndInnerFeltDims(combinedInnerTy)`
                      // for K=1/K>1 paths, or `<destDims..., totalFlat x
                      // !felt>` for the K=0 recursive flatten path.
      Type combinedInnerTy; // K=1: `pubFelts[0].ty` (preserves single-pub
                            // byte-layout). K>1: `!array<K x ...inner>`
                            // (one extra K dim prepended). K=0:
                            // `!array<totalFlat x !felt>` (flat per-instance
                            // concat of writem-targeted member contents).
      SmallVector<PubFelt, 2> pubFelts; // Pub members in declaration order.
                                        // Empty for the K=0 recursive path.
      SmallVector<Value, 2> kIndices;   // K=1: empty. K>1: K shared
                                        // `arith.constant <j> : index` Values
                                        // emitted once at destFelt allocation
                                        // (function-block-dominant) and reused
                                        // across every writer site + @compute
                                        // reader. Single-instance fold keeps
                                        // the surviving writer's `insertAfter`
                                        // dominated by these.
      SmallVector<WritemMember, 2> recursiveMembers; // K=0 path:
                                                     // writem-targeted members.
                                                     // Empty otherwise.
      int64_t totalFlat = 0; // K=0 path: sum of `recursiveMembers[*].flatSize`.
                             // Zero on K>=1 paths (unused).
      Type structArrTy;      // Original array<D x !struct.type<@Sub>>.
      // True when destFelt was reused from `perFieldArrays[innerField]`
      // (Loop A's reader-side allocation). In that case Loop A already
      // emits the `array.write %perFieldArrays[innerField][i] = struct.readm
      // %callResult[@innerField]` per writer site, and Loop B (drain
      // emission) must skip its `array.write` to avoid duplicating into the
      // same array at the same indices. The duplicate-write pair was the
      // smoking gun behind the AES `aes_256_encrypt` [0,128) ciphertext
      // residual — see memory/aes-encrypt-mod16-residual-followup.md.
      // Reuse only fires for K=1 — K>1's combined `<D, K x ...>` shape
      // never matches a per-field `<D x ...>` allocation by Loop A.
      bool reusedFromPerField = false;
    };
    llvm::DenseMap<Value, DrainPlan> drainPlans;
    auto findInnerFeltMembers = [&](llzk::component::StructType structTy,
                                    SmallVectorImpl<PubFelt> &out) -> bool {
      // Locate the struct.def for `structTy` by walking the enclosing
      // module. Returns true and populates `out` with the pub felt
      // members in declaration order when the struct.def has at least
      // one `{llzk.pub}` member whose type is `!felt` or `!array<... x
      // !felt>`. When K>1, all members must share the same type
      // (uniform-shape constraint — see DrainPlan comment).
      out.clear();
      ModuleOp moduleOp = getTopLevelModule(funcBlock);
      if (!moduleOp)
        return false;
      // Match the struct.def by its leaf symbol name. AES sub-component
      // structs have unique leaf names (`@XOR_0`, `@Bits2Num_1`, …) so
      // leaf matching is sufficient — no need to track the enclosing
      // `poly.template` / `builtin.module` chain that LLZK v2 wraps
      // around each component.
      StringRef leaf = structTy.getNameRef().getLeafReference().getValue();
      Operation *foundDef = nullptr;
      moduleOp->walk([&](Operation *op) {
        if (op->getName().getStringRef() != "struct.def")
          return WalkResult::advance();
        auto sym = op->getAttrOfType<StringAttr>("sym_name");
        if (!sym || sym.getValue() != leaf)
          return WalkResult::advance();
        foundDef = op;
        return WalkResult::interrupt();
      });
      if (!foundDef)
        return false;
      // Public is the canonical "this is the witness output" marker —
      // structs like `MiMC7_0` expose felt-or-felt-array internals
      // (`@t2/@t4/@t6/@t7`) alongside the pub `@out`; without the pub
      // filter, widening acceptance to felt-array would make the count
      // ambiguous and silently drop the case.
      foundDef->walk([&](Operation *m) {
        if (m->getName().getStringRef() != "struct.member")
          return;
        auto memTy = m->getAttrOfType<TypeAttr>("type");
        if (!memTy || !isFlattenableFelt(memTy.getValue()))
          return;
        if (!m->hasAttr(llzk::PublicAttr::name))
          return;
        auto sym = m->getAttrOfType<StringAttr>("sym_name");
        if (!sym)
          return;
        out.push_back({sym.getValue(), memTy.getValue()});
      });
      if (out.empty())
        return false;
      // Mixed-shape multi-pub would need a flat-felt concat path
      // (`<D, totalFlat x !felt>`). No bucket-1 chip exhibits that
      // shape today — Switcher is all-scalar and BitElementMulAny is
      // all-`<2 x !felt>` — so reject and let the caller fall back to
      // the existing nondet path until evidence demands otherwise.
      if (out.size() > 1) {
        Type first = out.front().ty;
        for (const PubFelt &pf : out)
          if (pf.ty != first) {
            out.clear();
            return false;
          }
      }
      return true;
    };
    // K=0 path helper. Collects writem-targeted, non-pod members of
    // `structTy` in declaration order with their recursive flat sizes,
    // and (if `promote` is true) promotes those members to `{llzk.pub}`
    // on the inner struct.def so subsequent `struct.readm` from outside
    // the inner struct is legal under LLZK's MemberReadOp verifier
    // (which rejects external reads of private members).
    //
    // Returns true when at least one writem-targeted member contributes
    // non-zero recursive flat (i.e. the K=0 path has real content to
    // drain). The "no pub" filter is intentionally *omitted* — the
    // caller decides which path to take based on `findInnerFeltMembers`
    // first; this fallback applies only when that path rejected.
    //
    // For Phase 3 single-level support, struct-typed writem-targeted
    // members with non-zero recursive flat are *rejected* — multi-level
    // recursion (emitting per-level struct.readm chains at writer sites)
    // is not implemented today. Zero-flat struct-typed members (e.g.
    // webb's `@indexBits` / `@set` whose inner Num2Bits_205 /
    // ForceSetMembershipIfEnabled_274 contribute zero felts in MMP's
    // writem-target set) are silently skipped.
    auto collectAndPromoteRecursiveWritemMembers =
        [&](llzk::component::StructType structTy,
            SmallVectorImpl<WritemMember> &out, bool promote) -> bool {
      out.clear();
      ModuleOp moduleOp = getTopLevelModule(funcBlock);
      if (!moduleOp)
        return false;
      StringRef leaf = structTy.getNameRef().getLeafReference().getValue();
      Operation *foundDef = nullptr;
      moduleOp->walk([&](Operation *op) {
        if (op->getName().getStringRef() != "struct.def")
          return WalkResult::advance();
        auto sym = op->getAttrOfType<StringAttr>("sym_name");
        if (!sym || sym.getValue() != leaf)
          return WalkResult::advance();
        foundDef = op;
        return WalkResult::interrupt();
      });
      if (!foundDef)
        return false;
      llvm::DenseSet<StringAttr> writemSet = collectWritemTargets(foundDef);
      int64_t offset = 0;
      bool rejected = false;
      SmallVector<Operation *, 2> memberOps;
      foundDef->walk([&](Operation *m) {
        if (rejected || m->getName().getStringRef() != "struct.member")
          return;
        auto sym = m->getAttrOfType<StringAttr>("sym_name");
        if (!sym || !writemSet.count(sym))
          return;
        auto memTy = m->getAttrOfType<TypeAttr>("type");
        if (!memTy)
          return;
        Type ty = memTy.getValue();
        // Skip pod-typed members at any ShapedType depth (mirrors
        // `isPodMemberType` in TypeConversion.cpp).
        Type leafTy = ty;
        while (auto shaped = dyn_cast<ShapedType>(leafTy))
          leafTy = shaped.getElementType();
        if (leafTy.getDialect().getNamespace() == "pod")
          return;
        // Struct-typed writem-targets with non-zero recursive flat require
        // multi-level recursion — Phase 3 single-level scope rejects.
        if (LlzkToStablehloTypeConverter::isStructType(ty)) {
          int64_t flat = getMemberFlatSize(ty, moduleOp);
          if (flat > 0) {
            rejected = true;
            return;
          }
          // Zero-flat struct member contributes nothing — skip.
          return;
        }
        int64_t flat = getMemberFlatSize(ty, moduleOp);
        if (flat == 0)
          return;
        out.push_back({sym.getValue(), ty, flat, offset});
        offset += flat;
        memberOps.push_back(m);
      });
      if (rejected) {
        out.clear();
        return false;
      }
      if (out.empty())
        return false;
      // LLZK's `MemberReadOp::verifySymbolUses` rejects external reads of
      // private members (see llzk struct dialect Ops.cpp). The writer-emit
      // step below issues `struct.readm %callResult[@<field>]` from inside
      // the *parent* struct's @compute, which is "outside" by that
      // verifier's notion. Promote each collected member to `{llzk.pub}`
      // on the inner struct.def so the read is legal.
      //
      // Semantic argument: WLA already counts these slots in the parent's
      // witness-layout footprint via `getMemberFlatSize`'s recursive walk
      // (PR #99). Promoting them to pub aligns the LLZK struct visibility
      // contract with the WLA exposure already in effect — the witness
      // layer treats them as observable; pub makes the LLZK verifier
      // agree.
      if (promote) {
        for (Operation *m : memberOps) {
          if (!m->hasAttr(llzk::PublicAttr::name))
            m->setAttr(llzk::PublicAttr::name, UnitAttr::get(m->getContext()));
        }
      }
      return true;
    };
    for (auto &dr : drainReaders) {
      if (drainPlans.count(dr.destArr))
        continue;
      auto destArrTy = dyn_cast<llzk::array::ArrayType>(dr.destArr.getType());
      if (!destArrTy)
        continue;
      auto innerStruct =
          dyn_cast<llzk::component::StructType>(destArrTy.getElementType());
      if (!innerStruct)
        continue;
      SmallVector<PubFelt, 2> pubFelts;
      SmallVector<WritemMember, 2> recursiveMembers;
      bool useRecursive = false;
      if (!findInnerFeltMembers(innerStruct, pubFelts)) {
        // K=0 (or mixed-shape K>1 rejected by uniform-shape gate)
        // fall-through: try recursive writem-target flatten. Promote
        // collected members to `{llzk.pub}` so the writer-side
        // `struct.readm %callResult[@<field>]` is legal under LLZK's
        // MemberReadOp visibility verifier.
        if (!collectAndPromoteRecursiveWritemMembers(
                innerStruct, recursiveMembers, /*promote=*/true))
          continue;
        useRecursive = true;
      }

      // K=1 keeps the existing combined shape (`<D x innerTy>`) so AES
      // sister chips stay byte-identical. K>1 prepends a K dim so each
      // pub field gets its own slot at outer index `j`; declaration
      // order is the canonical pub-field ordering and matches circom's
      // .wtns flat felt layout per chip iteration. `combineDispatchAnd
      // InnerFeltDims` already does the scalar-vs-array branch with the
      // exact "prepend leading dim(s)" semantics — reuse it with `{K}`
      // as the leading-dim list.
      //
      // K=0 (recursive) widens the parent slot to `<D × totalFlat × !felt>`
      // — `totalFlat` is the sum of writem-targeted member flats per
      // instance (e.g. MMP_275 → 30 hasher + 60 switcher = 90), with
      // per-instance row-major layout matching circom's `.wtns` for the
      // chained sub-component output (writem-targeted members in
      // declaration order, each member's elements walked in declared
      // dim order).
      int64_t totalFlat = 0;
      Type combinedInnerTy;
      if (useRecursive) {
        for (const WritemMember &wm : recursiveMembers)
          totalFlat += wm.flatSize;
        // Inner felt-type for the recursive flatten path. All members
        // contribute to the *same* `!felt` element type today (we reject
        // anything that's not felt-or-felt-array via the writem walk's
        // pod / struct gates). Pick the leaf felt type from the first
        // member — for `<N x !felt>` it's the array's element type, for
        // scalar `!felt` it's the member type itself.
        Type leafFelt = recursiveMembers.front().ty;
        while (auto shaped = dyn_cast<ShapedType>(leafFelt))
          leafFelt = shaped.getElementType();
        combinedInnerTy = llzk::array::ArrayType::get(leafFelt, {totalFlat});
      } else {
        combinedInnerTy =
            pubFelts.size() == 1
                ? pubFelts[0].ty
                : Type(combineDispatchAndInnerFeltDims(
                      pubFelts[0].ty, {static_cast<int64_t>(pubFelts.size())}));
      }

      // Allocate the parallel felt array right after the dispatch pod
      // array `%arr` so it dominates every writer site. The drain
      // destination is typically declared near the bottom of the
      // @compute body (after every writer); felt writes will go into
      // the parallel array at writer sites and the original destArr
      // will be replaced by an `unrealized_conversion_cast` of the
      // felt array on the consumer side.
      //
      // Reuse `perFieldArrays[innerField]` when Loop A already allocated
      // a felt array of the same element type and shape for the same
      // field name. Without this guard, Loop A and Loop B both emit one
      // `array.write` per writer site at identical outer indices —
      // ciphertext = same XOR result splattered into both per-field
      // carriers (`@a` reader-driven AND `@b` drain-driven), clobbering
      // a sibling `pod.write [@b] = round-key-bit` from
      // `rewritePodArrayUsesInBlock`. AES `aes_256_encrypt` is the
      // canonical case (60/128 ciphertext bytes wrong on round-0 XOR);
      // the post-conversion `dynamic_update_slice` pair at offsets
      // matching `iterArg_192`/`iterArg_193` collapses on identical RHS
      // and the legitimate round-key-bit emission is dropped. K>1
      // never reuses — its `<D, K x ...>` combined shape never matches
      // a Loop A per-field `<D x ...>` allocation.
      auto destDims = getArrayDimensions(destArrTy);
      auto feltArrTy =
          combineDispatchAndInnerFeltDims(combinedInnerTy, destDims);
      Value destFelt;
      bool reused = false;
      // K=0 path never reuses a `perFieldArrays` allocation — its
      // `<D × totalFlat × !felt>` shape never matches a Loop A
      // per-field `<D × ...>` carrier (no pub felts means no Loop A
      // reader-side allocation for the inner struct's members).
      if (!useRecursive && pubFelts.size() == 1) {
        auto reuseIt = perFieldArrays.find(pubFelts[0].field);
        if (reuseIt != perFieldArrays.end() &&
            reuseIt->second.getType() == feltArrTy) {
          destFelt = reuseIt->second;
          reused = true;
        }
      }
      if (!destFelt) {
        OpBuilder db(cand.arrNew);
        db.setInsertionPointAfter(cand.arrNew);
        destFelt = db.create<llzk::array::CreateArrayOp>(cand.arrNew->getLoc(),
                                                         feltArrTy);
      }
      // K>1: emit the K shared index constants once near `cand.arrNew`
      // (function-block scope, dominates every writer site). Without
      // this hoist, the writer loop would emit one fresh constant per
      // (writer, j) = W·K constants per chip — D=30 K=2 → 60 surplus
      // `arith.constant <0..1>` ops that CSE folds downstream but
      // bloat the IR every intermediate pass walks.
      SmallVector<Value, 2> kIndices;
      if (pubFelts.size() > 1) {
        OpBuilder ib(cand.arrNew);
        ib.setInsertionPointAfter(cand.arrNew);
        for (size_t j = 0; j < pubFelts.size(); ++j) {
          OperationState idxState(cand.arrNew->getLoc(), "arith.constant");
          idxState.addAttribute("value", ib.getIndexAttr(j));
          idxState.addTypes({ib.getIndexType()});
          kIndices.push_back(ib.create(idxState)->getResult(0));
        }
      }
      drainPlans[dr.destArr] = {destFelt,
                                combinedInnerTy,
                                SmallVector<PubFelt, 2>(pubFelts),
                                std::move(kIndices),
                                std::move(recursiveMembers),
                                totalFlat,
                                dr.destArr.getType(),
                                reused};
    }

    if (perFieldArrays.empty() && drainPlans.empty())
      continue;

    // Single-instance pod-array dispatch fast path: when every writer
    // targets the same destArr cell (outerIndex tuples compare equal as
    // `felt.const` values, looked through `cast.toindex`) AND every cross-
    // block reader/drainReader follows the last writer in funcBlock
    // source order, drop all but the last writer. Earlier writers'
    // results are last-write-wins-clobbered on the destArr cell, so
    // emitting their `function.call`s + `array.insert`s only inflates
    // GPU work (each redundant call lowers to its own kernel that
    // produces a value the next writer immediately overwrites).
    //
    // Empirical: `aes_256_ctr` has 2 writers sharing `cast.toindex
    // %felt_const_0` over a `!array<1 x !pod>` dispatch pod, so
    // pre-fold it emits 2 hoisted calls × {128, 1920} body iters =
    // 2048 calls at runtime; post-fold, 1 call × 128 iters = 128.
    // The `materializeScalarPodCompField` sister already handles the
    // analogous `pod.new` shape; this is the array-of-pods variant.
    // Normalize to `index`-typed bit width at construction. Felt APInts
    // come out of `FeltConstAttr` at the literal's minimum-bits-needed
    // width (e.g. `felt.const 1` → 4-bit); arith.constant-index APInts
    // are 64-bit. Comparing them directly trips
    // `APInt::operator==`'s "equal bit widths" assertion. Felt indices
    // that don't fit in 64 bits cannot be valid array offsets anyway,
    // so `zextOrTrunc(kIndexBitWidth)` is loss-free for this fold's
    // purpose. Same precedent as `TypeConversion.cpp` for felt
    // constants under `APInt::zextOrTrunc(storageWidth)`.
    constexpr unsigned kIndexBitWidth = 64;
    auto outerIndexConstValues =
        [](ArrayRef<Value> indices) -> std::optional<SmallVector<llvm::APInt>> {
      SmallVector<llvm::APInt> out;
      for (Value idx : indices) {
        Operation *def = idx.getDefiningOp();
        while (def && def->getName().getStringRef() == "cast.toindex" &&
               def->getNumOperands() >= 1)
          def = def->getOperand(0).getDefiningOp();
        if (!def)
          return std::nullopt;
        StringRef opName = def->getName().getStringRef();
        // `felt.const` carries a custom `FeltConstAttr` (APInt + FeltType),
        // not a plain `IntegerAttr`. Standard `arith.constant` paths use
        // `IntegerAttr` on integer-typed results — accept either so the
        // fold keeps working if someone canonicalizes a `cast.toindex`
        // chain into a plain `arith.constant index` upstream.
        if (opName == "felt.const") {
          auto attr = def->getAttr("value");
          if (auto feltConst =
                  dyn_cast_or_null<llzk::felt::FeltConstAttr>(attr)) {
            out.push_back(feltConst.getValue().zextOrTrunc(kIndexBitWidth));
            continue;
          }
          return std::nullopt;
        }
        if (opName == "arith.constant") {
          auto intAttr = def->getAttrOfType<IntegerAttr>("value");
          if (!intAttr)
            return std::nullopt;
          out.push_back(intAttr.getValue().zextOrTrunc(kIndexBitWidth));
          continue;
        }
        return std::nullopt;
      }
      return out;
    };
    // True iff `afterOp` strictly follows `beforeOp` in their smallest
    // common enclosing block. Returns false when no common block exists
    // (different scf.if regions etc.) — caller treats that as "ordering
    // unprovable" and bails out of the fold.
    auto strictlyAfter = [&](Operation *afterOp, Operation *beforeOp) -> bool {
      if (!afterOp || !beforeOp)
        return false;
      for (Block *blk = afterOp->getBlock(); blk;
           blk = blk->getParentOp() ? blk->getParentOp()->getBlock()
                                    : nullptr) {
        Operation *afterAnc = blk->findAncestorOpInBlock(*afterOp);
        Operation *beforeAnc = blk->findAncestorOpInBlock(*beforeOp);
        if (afterAnc && beforeAnc)
          return beforeAnc->isBeforeInBlock(afterAnc);
      }
      return false;
    };

    if (writers.size() > 1) {
      bool allSameOuterIndex = true;
      auto firstVals = outerIndexConstValues(writers.front().outerIndices);
      if (!firstVals) {
        allSameOuterIndex = false;
      } else {
        for (size_t i = 1; i < writers.size() && allSameOuterIndex; ++i) {
          auto v = outerIndexConstValues(writers[i].outerIndices);
          if (!v || v->size() != firstVals->size()) {
            allSameOuterIndex = false;
            break;
          }
          for (size_t j = 0; j < v->size(); ++j)
            if ((*v)[j] != (*firstVals)[j]) {
              allSameOuterIndex = false;
              break;
            }
        }
      }

      if (allSameOuterIndex) {
        // Pick the source-order-latest writer.
        size_t lastIdx = 0;
        for (size_t i = 1; i < writers.size(); ++i) {
          Operation *cur = writers[i].callResult.getDefiningOp();
          Operation *best = writers[lastIdx].callResult.getDefiningOp();
          if (cur && best && strictlyAfter(cur, best))
            lastIdx = i;
        }

        Operation *lastCall = writers[lastIdx].callResult.getDefiningOp();
        bool consumersAllAfter = lastCall != nullptr;
        for (auto &r : readers)
          if (!strictlyAfter(r.structReadm, lastCall)) {
            consumersAllAfter = false;
            break;
          }
        // NOLINTNEXTLINE(readability/braces)
        if (consumersAllAfter)
          for (auto &dr : drainReaders)
            if (!strictlyAfter(dr.writem, lastCall)) {
              consumersAllAfter = false;
              break;
            }

        if (consumersAllAfter) {
          // Erase the dropped writers' `function.call` + `pod.write
          // [@comp] = %callResult` so `extractCallsFromScfIf` (Phase 1)
          // doesn't re-emit them as orphan calls before the enclosing
          // scf.if. Pod.write [@comp] is the ONLY consumer of the
          // call's struct result (verified at writer-collection time:
          // we walked back from the array.write site to find this
          // pair). The post-while lift to ONE total call (vs. one per
          // surviving writerWhile body iter) happens later in
          // `liftConstIndexPodArrayCallPostWhile`, after all pod.read
          // operands have been resolved to their staged array values.
          for (size_t i = 0; i < writers.size(); ++i) {
            if (i == lastIdx)
              continue;
            Operation *callOp = writers[i].callResult.getDefiningOp();
            if (writers[i].podWrite)
              writers[i].podWrite->erase();
            if (callOp && isAllResultsUnused(*callOp))
              callOp->erase();
          }

          Writer keep = std::move(writers[lastIdx]);
          writers.clear();
          writers.push_back(std::move(keep));
        }
      }
    }

    // For each writer: hoist the dispatch `function.call` (and its
    // transitive operand defs) out of `w.insertAfter`'s scf.if so its
    // result is visible at body level, then for each per-field array
    // emit `%felt = struct.readm %callResult[@F]; array.write
    // %arr_F[%outerIndex] = %felt` after `w.insertAfter`.
    //
    // Per-writer hoisting is required because Phase 1's tracker is
    // overwritten on each scf.if extraction (count-countdown helpers
    // emit one fire per pending input — keccak XorArray fires twice,
    // arity-3 gates would fire three times). Forwarding via Phase 2
    // would resolve all hoisted reads to the *last* call, breaking
    // dominance for earlier writes. The single-instance fold above
    // sidesteps this concern by collapsing same-cell writers up front
    // when no consumer is sandwiched between them.
    for (auto &w : writers) {
      Operation *callOp = w.callResult.getDefiningOp();
      if (!callOp)
        continue;

      // Collect transitive operand defs that live under `w.insertAfter`
      // in operands-before-uses order, then move each before
      // `w.insertAfter`. Building the order during traversal (recurse
      // first, then append) avoids a second pass walking every op in
      // the ancestor's regions just to find which are `needed` —
      // ~6× fewer op visits across keccak's 48 dispatch fires.
      llvm::DenseSet<Operation *> seen;
      SmallVector<Operation *> ordered;
      auto collect = [&](auto &self, Value v) -> void {
        Operation *def = v.getDefiningOp();
        if (!def || !seen.insert(def).second)
          return;
        // Leave external defs in `seen` so subsequent visits early-return
        // at the `insert` check instead of re-running `isAncestor` and
        // recursing into their (possibly large) operand chains.
        if (!w.insertAfter->isAncestor(def))
          return;
        for (Value operand : def->getOperands())
          self(self, operand);
        ordered.push_back(def);
      };
      for (Value operand : callOp->getOperands())
        collect(collect, operand);
      ordered.push_back(callOp);

      for (Operation *op : ordered)
        op->moveBefore(w.insertAfter);

      OpBuilder b(w.insertAfter);
      b.setInsertionPointAfter(w.insertAfter);
      for (auto &kv : perFieldArrays) {
        StringRef fieldName = kv.first();
        Value arrField = kv.second;
        Type feltTy = fieldFeltTypes[fieldName];
        // MemberReadOp carries `AttrSizedOperandSegments`; use the
        // typed builder so the segment-sizes attribute is populated.
        Value feltVal = b.create<llzk::component::MemberReadOp>(
            w.insertAfter->getLoc(), feltTy, w.callResult,
            b.getStringAttr(fieldName));

        // For scalar `!felt` fields use `array.write` (single element);
        // for `!array<K x !felt>` fields use `array.insert` to store the
        // whole sub-array slice at outer indices. Mirrors
        // `rewritePodArrayUsesInBlock` line ~1619.
        StringRef writeOpName = isa<llzk::array::ArrayType>(feltTy)
                                    ? "array.insert"
                                    : "array.write";
        OperationState writeState(w.insertAfter->getLoc(), writeOpName);
        SmallVector<Value> writeOperands;
        writeOperands.push_back(arrField);
        writeOperands.append(w.outerIndices.begin(), w.outerIndices.end());
        writeOperands.push_back(feltVal);
        writeState.addOperands(writeOperands);
        b.create(writeState);
      }
      // Drain side: extract each pub felt member from %callResult and
      // write it into the parallel destFelt array. K=1 emits one write
      // per writer at outerIndices (existing single-pub shape). K>1
      // emits K writes per writer with a constant K-dim index `j`
      // appended to outerIndices, one per pub field in declaration
      // order. Either way, the destArr's struct-element layout is
      // discarded — the lowered tensor for the parent's `@F'` member
      // becomes `tensor<D[*K[*M]] x F>`, exactly the felt-flat witness
      // contribution the canonical circom `.wtns` expects per cell.
      for (auto &kv : drainPlans) {
        const DrainPlan &plan = kv.second;
        // K=0 recursive flatten path: emit one struct.readm per
        // writem-targeted member, then walk each member's natural
        // dim shape row-major and emit a per-element `array.write` into
        // `destFelt[outerIndices..., %off]`. The flat offset
        // `offsetWithinInstance + linear_idx` is materialized as
        // `arith.constant : index`. Each writer site contributes
        // `sum(memberFlats)` writes (e.g. 90 for MMP_275: 30 hasher +
        // 60 switcher). Sister K=1/K>1 chips never reach this branch
        // because they take the pub-felt path above.
        if (!plan.recursiveMembers.empty()) {
          for (const WritemMember &wm : plan.recursiveMembers) {
            Value memVal = b.create<llzk::component::MemberReadOp>(
                w.insertAfter->getLoc(), wm.ty, w.callResult,
                b.getStringAttr(wm.field));
            SmallVector<int64_t> memDims;
            if (auto arrTy = dyn_cast<llzk::array::ArrayType>(wm.ty))
              memDims = getArrayDimensions(arrTy);
            // Scalar `!felt` member: single write at offsetWithinInstance.
            // Multi-dim array member: row-major walk of all elements.
            int64_t numElements = wm.flatSize;
            for (int64_t linear = 0; linear < numElements; ++linear) {
              // Decompose `linear` into multi-dim coords via row-major
              // (last dim varies fastest). For scalar members memDims is
              // empty and the loop reduces to a no-op (no array.read /
              // memVal IS the felt directly).
              SmallVector<Value> coordVals;
              if (!memDims.empty()) {
                int64_t rem = linear;
                SmallVector<int64_t> coords(memDims.size(), 0);
                for (int dim = static_cast<int>(memDims.size()) - 1; dim >= 0;
                     --dim) {
                  coords[dim] = rem % memDims[dim];
                  rem /= memDims[dim];
                }
                for (int64_t c : coords) {
                  OperationState idxState(w.insertAfter->getLoc(),
                                          "arith.constant");
                  idxState.addAttribute("value", b.getIndexAttr(c));
                  idxState.addTypes({b.getIndexType()});
                  coordVals.push_back(b.create(idxState)->getResult(0));
                }
              }
              Value scalar;
              if (memDims.empty()) {
                scalar = memVal;
              } else {
                // Scalar leaf type: peel any nested ShapedType wrappers
                // off `wm.ty` to land on the felt type itself.
                Type leafTy = wm.ty;
                while (auto shaped = dyn_cast<ShapedType>(leafTy))
                  leafTy = shaped.getElementType();
                OperationState readState(w.insertAfter->getLoc(), "array.read");
                SmallVector<Value> readOperands;
                readOperands.push_back(memVal);
                readOperands.append(coordVals.begin(), coordVals.end());
                readState.addOperands(readOperands);
                readState.addTypes({leafTy});
                scalar = b.create(readState)->getResult(0);
              }
              // Flat inner offset constant for destFelt's last dim.
              OperationState offState(w.insertAfter->getLoc(),
                                      "arith.constant");
              offState.addAttribute(
                  "value", b.getIndexAttr(wm.offsetWithinInstance + linear));
              offState.addTypes({b.getIndexType()});
              Value offVal = b.create(offState)->getResult(0);
              OperationState writeState(w.insertAfter->getLoc(), "array.write");
              SmallVector<Value> writeOperands;
              writeOperands.push_back(plan.destFelt);
              writeOperands.append(w.outerIndices.begin(),
                                   w.outerIndices.end());
              writeOperands.push_back(offVal);
              writeOperands.push_back(scalar);
              writeState.addOperands(writeOperands);
              b.create(writeState);
            }
          }
          continue;
        }
        // When the drain target shares storage with Loop A's
        // `perFieldArrays[plan.pubFelts[0].field]`, Loop A above
        // already emitted `array.write %perFieldArrays[innerField][i]
        // = struct.readm %callResult[@innerField]` for this writer.
        // Re-emitting the same write here would produce a duplicate
        // pair at identical indices — the post-conversion
        // `dynamic_update_slice` collapse on identical RHS clobbers
        // sibling per-field writes elsewhere (the round-key-bit
        // emission for AES `aes_256_encrypt`).
        if (plan.reusedFromPerField)
          continue;
        for (size_t j = 0; j < plan.pubFelts.size(); ++j) {
          Value feltVal = b.create<llzk::component::MemberReadOp>(
              w.insertAfter->getLoc(), plan.pubFelts[j].ty, w.callResult,
              b.getStringAttr(plan.pubFelts[j].field));
          // Array-typed pub field stores a slice via `array.insert`;
          // scalar uses `array.write`. K>1 appends the shared K-dim
          // index emitted at destFelt-allocation time, keeping K=1's
          // IR shape byte-identical to the AES byte-stable single-pub
          // path.
          StringRef writeOpName =
              isa<llzk::array::ArrayType>(plan.pubFelts[j].ty) ? "array.insert"
                                                               : "array.write";
          OperationState writeState(w.insertAfter->getLoc(), writeOpName);
          SmallVector<Value> writeOperands;
          writeOperands.push_back(plan.destFelt);
          writeOperands.append(w.outerIndices.begin(), w.outerIndices.end());
          if (!plan.kIndices.empty())
            writeOperands.push_back(plan.kIndices[j]);
          writeOperands.push_back(feltVal);
          writeState.addOperands(writeOperands);
          b.create(writeState);
        }
      }
    }

    // The intervening `pod.read %dp2[@comp]` becomes use-empty; Phase 5
    // / `rewriteArrayPodCountCompInReads` nondets it on the next driver
    // iteration, then Phase 4 DCEs the dead substitute.
    SmallVector<Operation *> toErase;
    for (auto &r : readers) {
      auto fieldIt = perFieldArrays.find(r.field);
      if (fieldIt == perFieldArrays.end())
        continue;
      Value arrField = fieldIt->second;
      Type feltTy = fieldFeltTypes[r.field];
      OpBuilder b(r.structReadm);
      // For scalar `!felt` use `array.read` (full indices return single
      // element); for `!array<K x !felt>` use `array.extract` (partial
      // indices return a sub-array slice). Mirrors
      // `rewritePodArrayUsesInBlock` line ~1648.
      StringRef readOpName =
          isa<llzk::array::ArrayType>(feltTy) ? "array.extract" : "array.read";
      OperationState readState(r.structReadm->getLoc(), readOpName);
      SmallVector<Value> readOperands{arrField};
      llvm::append_range(readOperands, arrayAccessIndices(r.arrayRead));
      readState.addOperands(readOperands);
      readState.addTypes({feltTy});
      Value newReadVal = b.create(readState)->getResult(0);
      r.structReadm->getResult(0).replaceAllUsesWith(newReadVal);
      toErase.push_back(r.structReadm);
    }
    for (Operation *op : toErase)
      op->erase();

    // For each unique drain destination: redirect the parent
    // `struct.writem` operand from `%destArr` (struct array, populated
    // with `Phase 5`-nondet'd zeros) to the parallel felt array. This
    // requires also flipping the parent `struct.member @F'`'s declared
    // TypeAttr from `array<D x !struct>` to `array<D x !felt>` so the
    // writem's operand type matches the member type. The
    // `@constrain` chain that reads `@F'` is repaired to match —
    // struct.readm result type is retyped, the inner `array.read` is
    // retyped, and the inner `struct.readm @out` is erased
    // (replaced by the array.read result). Function calls in
    // `@constrain` whose operand types now mismatch (e.g.
    // `function.call @Sub::@constrain(%cell, ...)`) are erased
    // wholesale — `@constrain` is unreachable from witness generation
    // (`ConstrainFunctionErasePattern` deletes it during the main
    // conversion) so this is safe.
    llvm::DenseSet<Value> processedDest;
    for (auto &dr : drainReaders) {
      auto planIt = drainPlans.find(dr.destArr);
      if (planIt == drainPlans.end())
        continue;

      // Erase the drain reader's `array.write %destArr[idx] = %comp`
      // (Phase 5 already nondets %comp to a zero struct, which would
      // otherwise overwrite the felt-array slot if the writem still
      // pointed at %destArr).
      dr.arrayWriteDst->erase();

      if (!processedDest.insert(dr.destArr).second)
        continue;

      const DrainPlan &plan = planIt->second;
      bool isRecursive = !plan.recursiveMembers.empty();
      auto destDims =
          getArrayDimensions(cast<llzk::array::ArrayType>(plan.structArrTy));
      // Sister chips with scalar inner @out (K=1) are byte-equal post-flip;
      // chips with `!array<M x !felt>` inner members inflate the parent's
      // flat witness slot from D to D*M; multi-pub (K>1) inflates by an
      // additional factor of K (matches `getMemberFlatSize` over the
      // lowered `tensor<D[*K[*M]]>` shape). K=0 recursive flatten
      // produces `<D × totalFlat × !felt>` directly from `combinedInnerTy
      // = <totalFlat × !felt>`.
      auto newMemberArrTy =
          combineDispatchAndInnerFeltDims(plan.combinedInnerTy, destDims);

      // Redirect the parent struct.writem.
      dr.writem->setOperand(1, plan.destFelt);

      // Flip the parent struct.member @F' TypeAttr.
      Operation *parentStructDef = dr.writem->getParentOp();
      while (parentStructDef &&
             parentStructDef->getName().getStringRef() != "struct.def")
        parentStructDef = parentStructDef->getParentOp();
      if (!parentStructDef)
        continue;
      Operation *memberOp = nullptr;
      for (Region &region : parentStructDef->getRegions())
        for (Block &block : region)
          for (Operation &nested : block) {
            if (nested.getName().getStringRef() != "struct.member")
              continue;
            auto sym = nested.getAttrOfType<StringAttr>("sym_name");
            if (sym && sym.getValue() == dr.parentField) {
              memberOp = &nested;
              break;
            }
          }
      if (!memberOp)
        continue;
      memberOp->setAttr("type", TypeAttr::get(newMemberArrTy));

      // Repair @constrain: retype struct.readm @F' result and the
      // immediate `array.read %readm[..]` result; erase the inner
      // `struct.readm @<f_j>` (K=1: replace with array.read result;
      // K>1: replace with `array.read|extract %slice[%c_j]` per pub
      // field's index in declaration order; K=0 recursive: replace
      // each inner struct.readm with an `llzk.nondet` of the original
      // member type — its only consumer is the now-dead sibling
      // `function.call ::@constrain`, so the placeholder DCEs in
      // Phase 4); erase any function.call consumer whose operand
      // type now mismatches.
      bool isMultiPub = plan.pubFelts.size() > 1;
      bool innerIsArray =
          !isRecursive && isa<llzk::array::ArrayType>(plan.pubFelts[0].ty);
      llvm::DenseMap<StringRef, size_t> fieldIdx;
      for (size_t j = 0; j < plan.pubFelts.size(); ++j)
        fieldIdx[plan.pubFelts[j].field] = j;
      // K=0: map inner member name to original type so each
      // `struct.readm @<member>` in @constrain can be replaced by a
      // shape-matched llzk.nondet placeholder (the slice's true value is
      // not available as a felt-typed projection because heterogeneous
      // member shapes can't be re-extracted from a flat `<totalFlat ×
      // !felt>` row).
      llvm::DenseMap<StringRef, Type> recursiveMemberTys;
      if (isRecursive) {
        for (const WritemMember &wm : plan.recursiveMembers)
          recursiveMemberTys[wm.field] = wm.ty;
      }
      parentStructDef->walk([&](Operation *funcOp) {
        if (funcOp->getName().getStringRef() != "function.def")
          return;
        auto sym = funcOp->getAttrOfType<StringAttr>("sym_name");
        if (!sym || sym.getValue() != "constrain")
          return;
        SmallVector<Operation *> readms;
        funcOp->walk([&](Operation *rm) {
          if (rm->getName().getStringRef() != "struct.readm" ||
              rm->getNumResults() == 0)
            return;
          auto memberAttr = rm->getAttrOfType<FlatSymbolRefAttr>("member_name");
          if (!memberAttr || memberAttr.getValue() != dr.parentField)
            return;
          readms.push_back(rm);
        });
        // K=1 scalar inner: in-place retype keeps the array.read
        // intact (one fewer op + matches the byte-stable single-pub
        // shape). All other shapes (K=1 array, K>1 anything, K=0
        // recursive) shave one dim off the parent and need an
        // `array.extract` to produce the per-cell slice.
        bool sliceViaExtract = isRecursive || isMultiPub || innerIsArray;
        for (Operation *rm : readms) {
          rm->getResult(0).setType(newMemberArrTy);
          SmallVector<Operation *> toErase;
          // @compute's `kIndices` live in a different function — emit
          // a parallel set lazily inside @constrain so K>1 readers in
          // this function have an SSA value to index into the K-dim
          // slice with.
          SmallVector<Value, 2> constrainKIndices;
          for (OpOperand &use : rm->getResult(0).getUses()) {
            Operation *user = use.getOwner();
            if (user->getName().getStringRef() != "array.read" ||
                user->getNumResults() == 0)
              continue;
            Value newReadResult;
            if (sliceViaExtract) {
              OpBuilder rb(user);
              OperationState extractState(user->getLoc(), "array.extract");
              extractState.addOperands(user->getOperands());
              extractState.addTypes({plan.combinedInnerTy});
              Operation *extractOp = rb.create(extractState);
              newReadResult = extractOp->getResult(0);
              user->getResult(0).replaceAllUsesWith(newReadResult);
              toErase.push_back(user);
            } else {
              user->getResult(0).setType(plan.combinedInnerTy);
              newReadResult = user->getResult(0);
            }
            for (OpOperand &arUse :
                 llvm::make_early_inc_range(newReadResult.getUses())) {
              Operation *arUser = arUse.getOwner();
              StringRef arUserName = arUser->getName().getStringRef();
              // function.call to sibling `Sub::@constrain` now has a
              // felt[-array] operand where it expected `!struct<@Sub>`
              // — erase. @constrain is dead from a witness perspective
              // (`ConstrainFunctionErasePattern`) so unreferenced
              // operands DCE in Phase 4.
              if (arUserName == "function.call") {
                toErase.push_back(arUser);
                continue;
              }
              if (arUserName != "struct.readm" || arUser->getNumResults() == 0)
                continue;
              auto innerMember =
                  arUser->getAttrOfType<FlatSymbolRefAttr>("member_name");
              if (!innerMember)
                continue;
              // K=0 recursive: heterogeneous member shapes can't be
              // re-projected from a flat felt slice. Replace the inner
              // `struct.readm @<member>` with a typed `llzk.nondet`
              // placeholder — its only downstream consumer is the
              // sibling `function.call ::@constrain` (queued for erase
              // above), so the placeholder DCEs cleanly in Phase 4.
              if (isRecursive) {
                auto rmIt = recursiveMemberTys.find(innerMember.getValue());
                if (rmIt == recursiveMemberTys.end())
                  continue;
                OpBuilder ib(arUser);
                Value placeholder =
                    createNondet(ib, arUser->getLoc(), rmIt->second);
                arUser->getResult(0).replaceAllUsesWith(placeholder);
                toErase.push_back(arUser);
                continue;
              }
              auto fIt = fieldIdx.find(innerMember.getValue());
              if (fIt == fieldIdx.end())
                continue;
              if (!isMultiPub) {
                // K=1: the slice IS the field value.
                arUser->getResult(0).replaceAllUsesWith(newReadResult);
                toErase.push_back(arUser);
                continue;
              }
              // K>1: index into the K-dim slice with the field's
              // declaration-order position. Array pub field produces
              // a sub-slice via `array.extract`; scalar pub field
              // produces a scalar via `array.read`.
              if (constrainKIndices.empty()) {
                OpBuilder ib(rm);
                ib.setInsertionPointAfter(rm);
                for (size_t k = 0; k < plan.pubFelts.size(); ++k) {
                  OperationState idxState(rm->getLoc(), "arith.constant");
                  idxState.addAttribute("value", ib.getIndexAttr(k));
                  idxState.addTypes({ib.getIndexType()});
                  constrainKIndices.push_back(
                      ib.create(idxState)->getResult(0));
                }
              }
              OpBuilder ib(arUser);
              StringRef readOpName =
                  innerIsArray ? "array.extract" : "array.read";
              OperationState readState(arUser->getLoc(), readOpName);
              readState.addOperands(
                  {newReadResult, constrainKIndices[fIt->second]});
              readState.addTypes({plan.pubFelts[fIt->second].ty});
              Operation *readOp = ib.create(readState);
              arUser->getResult(0).replaceAllUsesWith(readOp->getResult(0));
              toErase.push_back(arUser);
            }
          }
          for (Operation *op : toErase)
            op->erase();
        }
      });
    }

    if (!perFieldArrays.empty() || !drainPlans.empty())
      changed = true;
  }

  return changed;
}

/// Close the input-pod data-flow gap that survives `flattenPodArrayWhileCarry`.
///
/// Circom emits a per-instance input-pod array `array.new : <D x !pod<[@in:
/// !felt]>>` to stage the operand for a deferred sub-component dispatch. The
/// writer body inside an scf.while iteration writes `%src` into the cell's
/// `@in` field; the firing scf.if (count countdown == 0) reads it back via
/// `pod.read %cell[@in]` and feeds the dispatched call. SSA-wise, `%src` and
/// the firing-site read share the same pod cell.
///
/// `flattenPodArrayWhileCarry` is supposed to rebuild that data flow at the
/// felt-array level when the carry crosses scf.while levels, but at 3+ levels
/// of nesting (AES `@AES256Encrypt_6::@compute` is the canonical case) the
/// per-level rewire leaves the firing-site read disconnected. Phase 5
/// (`rewriteArrayPodCountCompInReads`) then nondets it, the dispatched call
/// gets fed const-zero, and the parent witness sees the sub-component's
/// `Bits(0) = zeros` output.
///
/// This pass runs BEFORE `flattenPodArrayWhileCarry` while writer and reader
/// are still SSA-paired through `%cell`: it replaces every sibling `pod.read
/// %cell[@in]` result with `%src` directly and erases the pod.read. The
/// pod.write becomes use-empty and is DCE'd by Phase 4. Dominance holds
/// because `%src` and `%cell` are defined at the writer body block level
/// and the firing-site read lives in a child region of that same block.
///
/// Convergence with `eliminatePodDispatch`: this pass only does
/// `replaceAllUsesWith` + `erase` on existing pod.read ops, so Phase 5
/// finds nothing extra and Phase 1 routes the call's now-felt operand
/// through its `directArgs` branch. Idempotent: subsequent invocations
/// find no `pod.read [@in]` left.

// ===----------------------------------------------------------------------===
// Struct-of-pods carrier rewrite (pre-flatten)
// ===----------------------------------------------------------------------===

/// Match a `!pod<[@idx_0..@idx_K-1: T]>` shape with sequential idx-names
/// and uniform inner type T. Returns K and T on match; nullopt otherwise.
struct StructOfPodsShape {
  int64_t k;
  Type innerType;
};

std::optional<StructOfPodsShape> matchStructOfPodsShape(Type type) {
  auto podTy = dyn_cast<llzk::pod::PodType>(type);
  if (!podTy)
    return std::nullopt;
  auto records = podTy.getRecords();
  if (records.empty())
    return std::nullopt;
  Type uniformInner;
  for (size_t i = 0; i < records.size(); ++i) {
    StringRef name = records[i].getName().getValue();
    std::string expected = ("idx_" + Twine(i)).str();
    if (name != expected)
      return std::nullopt;
    if (i == 0)
      uniformInner = records[i].getType();
    else if (records[i].getType() != uniformInner)
      return std::nullopt;
  }
  return StructOfPodsShape{static_cast<int64_t>(records.size()), uniformInner};
}

int parseIdxFieldName(StringRef name) {
  if (!name.consume_front("idx_"))
    return -1;
  int64_t idx;
  if (name.getAsInteger(10, idx) || idx < 0)
    return -1;
  return static_cast<int>(idx);
}

Value buildConstIndex(OpBuilder &builder, Location loc, int64_t v) {
  OperationState state(loc, "arith.constant");
  state.addAttribute("value", builder.getIndexAttr(v));
  state.addTypes({builder.getIndexType()});
  return builder.create(state)->getResult(0);
}

/// Rewriter for struct-of-pods carriers: rewrites
/// `!pod<[@idx_0..@idx_K-1: T]>` → `!array<K x T>` for one seed at a time,
/// cascading through SSA users (pod.read[@idx_N] → array.read[%c_N],
/// pod.write[@idx_N] → array.write[%c_N], scf.while iter-arg slot retype,
/// scf.if result slot retype, scf.yield/scf.condition operand retype).
/// Idempotent: after rewrite the carrier's type is `!array<K x T>` and
/// the seed-match predicate finds no further candidates.
class StructOfPodsRewriter {
public:
  // Returns true iff `seed`'s chain was fully rewritten. On false, IR is
  // returned to its original state via the rollback (caller must check the
  // worklist's bail signal).
  bool run(Operation *seed) {
    auto shapeOpt = matchStructOfPodsShape(seed->getResult(0).getType());
    if (!shapeOpt)
      return false;
    shape = *shapeOpt;
    arrType = llzk::array::ArrayType::get(shape.innerType,
                                          ArrayRef<int64_t>{shape.k});

    if (!validateChain(seed))
      return false;

    if (!rewriteSeed(seed))
      return false;
    while (!worklist.empty()) {
      auto pair = worklist.pop_back_val();
      processUsers(pair.first, pair.second);
    }

    // Drop operand references in two passes so cross-references between
    // to-be-erased ops (e.g. the old scf.while's operand pointing at the old
    // pod.new's result; an old pod.read/write whose pod_ref is a now-orphan
    // block arg) don't keep block args alive when we erase below.
    for (Operation *op : toErase)
      op->dropAllReferences();
    for (BlockArgument oldArg : oldArgsToErase) {
      if (!oldArg.use_empty())
        return false;
      oldArg.getOwner()->eraseArgument(oldArg.getArgNumber());
    }
    for (Operation *op : toErase)
      op->erase();
    return true;
  }

private:
  StructOfPodsShape shape;
  llzk::array::ArrayType arrType;
  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<std::pair<Value, Value>> worklist;
  llvm::SmallSetVector<Operation *, 16> toErase;
  SmallVector<BlockArgument> oldArgsToErase;
  // Ops we've already rebuilt (the NEW op produced by a rebuild). Probed by
  // rewriteYieldOperand to avoid re-rebuilding the same
  // scf.if/scf.execute_region when both yields (then/else, or multi-block) flow
  // our value through.
  llvm::DenseSet<Operation *> rebuiltOps;

  // Walk SSA users transitively; return false if any user shape isn't
  // handleable. Pure analysis — no IR mutation.
  bool validateChain(Operation *seed) {
    llvm::SmallSetVector<Value, 16> visited;
    SmallVector<Value> stack;
    auto enqueue = [&](Value v) {
      if (visited.insert(v))
        stack.push_back(v);
    };
    enqueue(seed->getResult(0));

    // Validate the seed op itself supports the rewrite.
    StringRef seedName = seed->getName().getStringRef();
    if (seedName == "pod.new") {
      auto initAttr = seed->getAttrOfType<ArrayAttr>("initializedRecords");
      unsigned numInits = initAttr ? initAttr.size() : 0;
      if (numInits != 0 && numInits != (unsigned)shape.k)
        return false;
      // No map operands (pod.new with affine-mapped types).
      if (seed->getNumOperands() != numInits)
        return false;
    } else if (seedName != "llzk.nondet") {
      return false;
    }

    while (!stack.empty()) {
      Value v = stack.pop_back_val();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        unsigned opIdx = use.getOperandNumber();
        StringRef name = user->getName().getStringRef();

        if (name == "pod.read") {
          auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (!rn)
            return false;
          int idx = parseIdxFieldName(rn.getValue());
          if (idx < 0 || idx >= shape.k)
            return false;
          continue;
        }
        if (name == "pod.write") {
          if (opIdx != 0)
            return false;
          auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (!rn)
            return false;
          int idx = parseIdxFieldName(rn.getValue());
          if (idx < 0 || idx >= shape.k)
            return false;
          continue;
        }
        if (auto w = dyn_cast<scf::WhileOp>(user)) {
          enqueue(w.getBefore().front().getArgument(opIdx));
          enqueue(w.getAfter().front().getArgument(opIdx));
          enqueue(w.getResult(opIdx));
          continue;
        }
        if (auto y = dyn_cast<scf::YieldOp>(user)) {
          Operation *parent = y->getParentOp();
          if (auto pw = dyn_cast<scf::WhileOp>(parent)) {
            enqueue(pw.getResult(opIdx));
            continue;
          }
          if (auto pIf = dyn_cast<scf::IfOp>(parent)) {
            // scf.if rebuild needs both then and else yield operands at the
            // same slot in the rewrite chain. Enqueue the sibling branch's
            // yield operand so the cascade reaches it.
            enqueue(pIf.getResult(opIdx));
            for (Region *region :
                 {&pIf.getThenRegion(), &pIf.getElseRegion()}) {
              if (region->empty())
                continue;
              auto sib =
                  dyn_cast<scf::YieldOp>(region->front().getTerminator());
              if (!sib || sib == y)
                continue;
              if (opIdx >= sib.getNumOperands())
                return false;
              enqueue(sib.getOperand(opIdx));
            }
            continue;
          }
          if (auto pEr = dyn_cast<scf::ExecuteRegionOp>(parent)) {
            enqueue(pEr.getResult(opIdx));
            continue;
          }
          return false;
        }
        if (auto c = dyn_cast<scf::ConditionOp>(user)) {
          if (opIdx == 0)
            return false;
          auto pw = dyn_cast<scf::WhileOp>(c->getParentOp());
          if (!pw)
            return false;
          unsigned slot = opIdx - 1;
          enqueue(pw.getAfter().front().getArgument(slot));
          enqueue(pw.getResult(slot));
          continue;
        }
        // struct.writem, function.call, function.return etc. unsupported.
        return false;
      }
    }
    return true;
  }

  bool rewriteSeed(Operation *seed) {
    OpBuilder builder(seed);
    Location loc = seed->getLoc();
    Value newVal;
    StringRef opName = seed->getName().getStringRef();
    if (opName == "pod.new") {
      auto initAttr = seed->getAttrOfType<ArrayAttr>("initializedRecords");
      unsigned numInits = initAttr ? initAttr.size() : 0;
      SmallVector<Value> elements;
      if (numInits == (unsigned)shape.k) {
        elements.assign(shape.k, Value());
        for (unsigned i = 0; i < numInits; ++i) {
          auto nameAttr = dyn_cast<StringAttr>(initAttr[i]);
          if (!nameAttr)
            return false;
          int idx = parseIdxFieldName(nameAttr.getValue());
          if (idx < 0 || idx >= shape.k)
            return false;
          elements[idx] = seed->getOperand(i);
        }
        for (Value el : elements)
          if (!el)
            return false;
      }
      auto newCreate = builder.create<llzk::array::CreateArrayOp>(
          loc, arrType, ValueRange(elements));
      newVal = newCreate.getResult();
    } else if (opName == "llzk.nondet") {
      newVal = createNondet(builder, loc, arrType);
    } else {
      return false;
    }
    valueMap[seed->getResult(0)] = newVal;
    worklist.push_back({seed->getResult(0), newVal});
    toErase.insert(seed);
    return true;
  }

  void processUsers(Value oldVal, Value newVal) {
    // Snapshot uses since rewriting will mutate the use-list.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : oldVal.getUses())
      uses.push_back(&use);

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      unsigned opIdx = use->getOperandNumber();
      StringRef name = user->getName().getStringRef();

      if (name == "pod.read") {
        rewritePodRead(user, newVal);
      } else if (name == "pod.write") {
        rewritePodWrite(user, newVal);
      } else if (auto w = dyn_cast<scf::WhileOp>(user)) {
        rewriteWhileOperand(w, opIdx, newVal);
      } else if (auto y = dyn_cast<scf::YieldOp>(user)) {
        rewriteYieldOperand(y, opIdx, newVal);
      } else if (auto c = dyn_cast<scf::ConditionOp>(user)) {
        rewriteConditionOperand(c, opIdx, newVal);
      } else {
        llvm_unreachable("validateChain admitted an unsupported user");
      }
    }
  }

  void rewritePodRead(Operation *user, Value newCarrier) {
    auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
    int idx = parseIdxFieldName(rn.getValue());
    OpBuilder builder(user);
    Value cstIdx = buildConstIndex(builder, user->getLoc(), idx);
    auto readOp = builder.create<llzk::array::ReadArrayOp>(
        user->getLoc(), shape.innerType, newCarrier, ValueRange{cstIdx});
    user->getResult(0).replaceAllUsesWith(readOp.getResult());
    toErase.insert(user);
  }

  void rewritePodWrite(Operation *user, Value newCarrier) {
    auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
    int idx = parseIdxFieldName(rn.getValue());
    OpBuilder builder(user);
    Value cstIdx = buildConstIndex(builder, user->getLoc(), idx);
    Value writeVal = user->getOperand(1);
    builder.create<llzk::array::WriteArrayOp>(user->getLoc(), newCarrier,
                                              ValueRange{cstIdx}, writeVal);
    toErase.insert(user);
  }

  void rewriteWhileOperand(scf::WhileOp w, unsigned slot, Value newOperand) {
    if (rebuiltOps.count(w.getOperation()))
      return;

    OpBuilder builder(w);
    SmallVector<Value> newOperands(w.getOperands().begin(),
                                   w.getOperands().end());
    newOperands[slot] = newOperand;
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(w.getNumResults());
    for (unsigned i = 0; i < w.getNumResults(); ++i)
      newResultTypes.push_back(i == slot ? Type(arrType)
                                         : w.getResult(i).getType());

    auto newWhile =
        builder.create<scf::WhileOp>(w.getLoc(), newResultTypes, newOperands);
    newWhile.getBefore().takeBody(w.getBefore());
    newWhile.getAfter().takeBody(w.getAfter());
    rebuiltOps.insert(newWhile.getOperation());

    for (int regionIdx = 0; regionIdx < 2; ++regionIdx) {
      Block &blk = regionIdx == 0 ? newWhile.getBefore().front()
                                  : newWhile.getAfter().front();
      BlockArgument oldArg = blk.getArgument(slot);
      // Insert NEW arg at slot+1 so old keeps its index until cleanup.
      BlockArgument newArg = blk.insertArgument(slot + 1, arrType, w.getLoc());
      valueMap[oldArg] = newArg;
      worklist.push_back({oldArg, newArg});
      // After cascade migrates oldArg's uses, eraseArgument removes slot.
      // Re-fetch via getArgument(slot) because insertArgument shifts indices.
      oldArgsToErase.push_back(blk.getArgument(slot));
    }

    // Rewire non-pod result slots immediately; mapping + worklist for the
    // converted slot so downstream users cascade.
    for (unsigned i = 0; i < w.getNumResults(); ++i) {
      Value oldRes = w.getResult(i);
      Value newRes = newWhile.getResult(i);
      if (i == slot) {
        valueMap[oldRes] = newRes;
        worklist.push_back({oldRes, newRes});
      } else {
        oldRes.replaceAllUsesWith(newRes);
      }
    }

    toErase.insert(w.getOperation());
  }

  void rewriteYieldOperand(scf::YieldOp y, unsigned opIdx, Value newVal) {
    Operation *parent = y->getParentOp();
    if (auto er = dyn_cast<scf::ExecuteRegionOp>(parent)) {
      if (!rebuiltOps.count(er.getOperation()))
        rebuildExecuteRegion(er, opIdx);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      if (!rebuiltOps.count(ifOp.getOperation()))
        rebuildIfResultSlot(ifOp, opIdx);
    }
    // After rebuildExecuteRegion / rebuildIfResultSlot, the yield op `y`
    // remains alive (it's now in the new parent's region via takeBody),
    // so this setOperand updates the same yield in its new parent.
    y->setOperand(opIdx, newVal);
  }

  void rebuildExecuteRegion(scf::ExecuteRegionOp er, unsigned slot) {
    OpBuilder builder(er);
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(er.getNumResults());
    for (unsigned i = 0; i < er.getNumResults(); ++i)
      newResultTypes.push_back(i == slot ? Type(arrType)
                                         : er.getResult(i).getType());
    auto newEr =
        builder.create<scf::ExecuteRegionOp>(er.getLoc(), newResultTypes);
    newEr.getRegion().takeBody(er.getRegion());
    rebuiltOps.insert(newEr.getOperation());
    for (unsigned i = 0; i < er.getNumResults(); ++i) {
      Value oldRes = er.getResult(i);
      Value newRes = newEr.getResult(i);
      if (i == slot) {
        valueMap[oldRes] = newRes;
        worklist.push_back({oldRes, newRes});
      } else {
        oldRes.replaceAllUsesWith(newRes);
      }
    }
    toErase.insert(er.getOperation());
  }

  void rebuildIfResultSlot(scf::IfOp ifOp, unsigned slot) {
    OpBuilder builder(ifOp);
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(ifOp.getNumResults());
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      newResultTypes.push_back(i == slot ? Type(arrType)
                                         : ifOp.getResult(i).getType());
    bool withElse = !ifOp.getElseRegion().empty();
    auto newIf = builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                           ifOp.getCondition(), withElse);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (withElse)
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    rebuiltOps.insert(newIf.getOperation());
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Value oldRes = ifOp.getResult(i);
      Value newRes = newIf.getResult(i);
      if (i == slot) {
        valueMap[oldRes] = newRes;
        worklist.push_back({oldRes, newRes});
      } else {
        oldRes.replaceAllUsesWith(newRes);
      }
    }
    toErase.insert(ifOp.getOperation());
  }

  void rewriteConditionOperand(scf::ConditionOp c, unsigned opIdx,
                               Value newVal) {
    c->setOperand(opIdx, newVal);
  }
};

/// Rewrite struct-of-pods carriers (`!pod<[@idx_0..@idx_K-1: T]>`) to
/// array-of-pods (`!array<K x T>`) so the existing `flattenPodArrayWhileCarry`
/// infrastructure handles them. Driver placement: BEFORE
/// `materializePodArrayInputPodField` so the read-modify-write guard there
/// still sees pod-typed cells with the carrier already converted.
///
/// Canonical case: webb_poseidon_vanchor_2_2's `@Poseidon_137_compute` has a
/// 7-iter-arg outer scf.while carrying both `%arg2: !pod<[@idx_0..@idx_67: T]>`
/// (load-bearing read-modify-write carrier) and `%arg5: !array<68 x T>`
/// (parallel array form). `flattenPodArrayWhileCarry` handles `%arg5`; this
/// pass converts `%arg2` so its sibling phases follow.
bool convertStructOfPodsToArrayOfPods(Block &funcBlock) {
  // Deep walk for seeds (the carrier seed may be nested inside scf.while/
  // scf.if regions). Only `pod.new` seeds are processed; `llzk.nondet` of
  // struct-of-pods shape is a downstream-Phase-4 artifact whose chain is
  // already on its way to elimination — rewriting it forces an extra
  // outer fixed-point iter for no benefit.
  SmallVector<Operation *> seeds;
  funcBlock.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "pod.new")
      return;
    if (op->getNumResults() != 1)
      return;
    if (matchStructOfPodsShape(op->getResult(0).getType()))
      seeds.push_back(op);
  });

  for (Operation *seed : seeds) {
    StructOfPodsRewriter rewriter;
    if (rewriter.run(seed))
      return true;
  }
  return false;
}

/// Completion pass for NON-uniform-inner struct-of-pods carriers.
///
/// `convertStructOfPodsToArrayOfPods` only rewrites carriers whose K
/// `@idx_*` records share a single inner type. The canonical case it
/// CANNOT handle is webb Poseidon's 68-round Ark cascade, where each
/// `@idx_K` resolves to a distinct `!struct<@Ark_K>` round-constant class
/// (`matchStructOfPodsShape`'s uniform-inner check rejects). The carrier
/// stays alive on the scf.while iter-arg; `processNested`'s
/// `hasPodBlockArg=true` branch runs `extractCallsFromScfIf` to clone-hoist
/// every dispatched `function.call @Ark_K::@compute(%array.extract
/// %carrier[%cK]) : !struct<@Ark_K>` out of its statically-false dispatch
/// scf.if. After that, the writer-side cascade has N hoisted calls inside
/// an inner scf.while body, but the *reader*-side cascade — in a SIBLING
/// scf.while body — still reads `struct.readm [@F]` from a pre-existing
/// `llzk.nondet : !struct<@Ark_K>` (NOT from the hoisted call results;
/// the writer↔reader dispatch link was severed by an earlier phase). The
/// hoisted call's result has no consumer to RAUW onto and
/// `--llzk-to-stablehlo` DCEs it.
///
/// This pass closes the link by materializing one parallel felt array per
/// `@F` member referenced by readers: `<N x ...innerFeltDims>` allocated
/// at function-body scope (so it dominates both writer and reader
/// scf.while bodies). At each hoisted call site, emit `array.write
/// %carrier_F[%cK] = struct.readm %call[@F]` (or `array.insert` for
/// array-typed F) right after the call. Each reader-side
/// `struct.readm %nondet[@F]` is rewritten to `array.extract|array.read
/// %carrier_F[%cK]` using the writer-side class→K map (lookup by the
/// nondet's struct class leaf name). The original `llzk.nondet` becomes
/// use-empty once all its `struct.readm` consumers are rewritten and is
/// DCE'd by `eraseDeadPodAndCountOps` (Phase 4); the dispatch scf.if
/// cascade scaffolding is preserved per the "Don't reshape the `<--`
/// cascade" invariant — downstream `extendResultBearingScfIfArrayChain`
/// / `convertArrayWritesToSSA` need its result-slot shape intact.
///
/// SSA threading of `%carrier_F` through the writer + reader scf.while
/// iter-args is handled downstream by
/// `LlzkToStablehlo.cpp::convertWhileBodyArgsToSSA` and
/// `flattenPodArrayWhileCarry` — we don't manually rewire iter-args here.
/// LLZK is mutable at this stage so a function-scope `array.new` is
/// visible to writer/reader scf.while bodies via capture.
///
/// Idempotent: subsequent invocations find no `struct.readm` consumers on
/// the nondet ops (they were erased) and the readers list is empty, so
/// the pass returns false. Convergence with `eliminatePodDispatch`: emits
/// only `array.new` / `struct.readm` / `array.write`(insert) /
/// `array.extract`(read) — all are core (felt/struct/array) ops Phase 4
/// leaves alone, and the writer/reader scf.while bodies remain
/// computation-tagged scf.while (not scf.if/for) so the
/// `hasNonPodArrayWriteInBody` carve-out is not exercised.
///
/// Matching contract (single-operand calls, single-pub @F only —
/// matches Poseidon's Ark cascade today; multi-pub / multi-operand
/// shapes can extend the matcher when a chip demands it):
/// - Writer: `%call = function.call @<C>::@compute(%v) : !struct<@<C>>`
///   where `%v = array.extract %arr[%cK]` and `%cK = arith.constant <k>
///   : index`. Class leaf name `<C>` is the join key.
/// - Reader: `%nondet = llzk.nondet : !struct<@<C>>` followed by
///   `%readm = struct.readm %nondet[@F]`. Matching is by class leaf name;
///   no scf.if cascade decoding required.
/// - `@F` must be `{llzk.pub}` on the inner struct.def (LLZK's
///   `MemberReadOp::verifySymbolUses` rejects external reads of non-pub
///   members). Poseidon's `@Ark_K::@out` is pub by Circom contract.
bool materializeStructOfPodsCompField(Block &funcBlock) {
  struct Writer {
    Operation *call;
    StringRef className;
    int64_t kConst;
  };
  struct Reader {
    Operation *readm;
    StringRef className;
    StringRef field;
    Type fieldTy;
  };

  SmallVector<Writer> writers;
  SmallVector<Reader> readers;

  // Pre-scan: count ALL potential writers across the entire function body,
  // including ones still nested inside dispatch-firing scf.if cascade arms
  // that `eliminatePodDispatch` Phase 1 hasn't hoisted yet. This is needed
  // because Phase 1 hoists incrementally across outer fixed-point iters
  // (one or a few per iter), so the first invocation where my pass fires
  // would see only a partial set of writers and miscompute N. Without the
  // pre-scan, the allocated carrier shape `<N_partial x ...>` is too small
  // for cascade K values surfaced in later iters — the IR ends up with
  // out-of-bounds `array.insert %carrier[%c67] : <1,5>` and lowering fails.
  // The pre-scan still gates on the `array.extract %arr[%cK]` operand
  // pattern with K an arith.constant, so unrelated function.calls (e.g.
  // top-level @Poseidon::@compute calls, Mix calls with non-constant K)
  // are correctly excluded.
  int64_t preScanMaxK = -1;
  llvm::StringMap<int64_t> preScanClassToK;
  funcBlock.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "function.call" ||
        op->getNumResults() != 1 || op->getNumOperands() != 1)
      return;
    auto structTy =
        dyn_cast<llzk::component::StructType>(op->getResult(0).getType());
    if (!structTy)
      return;
    Operation *def = op->getOperand(0).getDefiningOp();
    if (!def || def->getName().getStringRef() != "array.extract" ||
        def->getNumOperands() != 2)
      return;
    Operation *idxDef = def->getOperand(1).getDefiningOp();
    if (!idxDef || idxDef->getName().getStringRef() != "arith.constant")
      return;
    auto intAttr = idxDef->getAttrOfType<IntegerAttr>("value");
    if (!intAttr)
      return;
    llvm::APInt apInt = intAttr.getValue();
    if (apInt.getBitWidth() > 64 || apInt.isNegative())
      return;
    int64_t k = apInt.getSExtValue();
    StringRef cls = structTy.getNameRef().getLeafReference().getValue();
    auto it = preScanClassToK.find(cls);
    if (it == preScanClassToK.end())
      preScanClassToK[cls] = k;
    else if (it->second != k)
      return; // conflicting K for same class — leave undecided
    preScanMaxK = std::max(preScanMaxK, k);
  });

  // Marker attribute on the writer's function.call op signaling that this
  // pass has ALREADY emitted its `struct.readm + array.insert` shadow chain
  // for this call. The outer fixed-point loop re-invokes this pass after
  // every successful emission; without the marker we would re-emit every
  // iter, allocating a fresh carrier each time and accumulating O(N²)
  // dead ops + carrier-rename churn before `--llzk-to-stablehlo` DCEs them.
  StringRef kMaterializedAttr = "mSoPCF.materialized";
  funcBlock.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();

    if (name == "function.call" && op->getNumResults() == 1 &&
        op->getNumOperands() == 1) {
      auto structTy =
          dyn_cast<llzk::component::StructType>(op->getResult(0).getType());
      if (!structTy)
        return;
      // Idempotence: skip writers already processed by a previous outer iter.
      if (op->hasAttr(kMaterializedAttr))
        return;
      // Skip writers whose enclosing block is an scf.if —
      // `eliminatePodDispatch` Phase 1 (`extractCallsFromScfIf`) runs LATER in
      // the driver loop, so on the first outer fixed-point iter the dispatched
      // calls are still INSIDE the dispatch-firing scf.if cascade arms.
      // Emitting inserts at that position lands them in branches that Phase 4
      // will then make statically false (the `arith.subi 0, 1` predicate trap),
      // and `--llzk-to-stablehlo` DCEs the entire scf.if — silently dropping
      // the carrier writes for all K != 0 cascade arms. Wait until Phase 1
      // hoists them to the scf.while body level (parent != scf.if), then this
      // matcher fires on iter 2+.
      Block *callBlock = op->getBlock();
      if (callBlock && callBlock->getParentOp() &&
          callBlock->getParentOp()->getName().getStringRef() == "scf.if")
        return;
      Operation *def = op->getOperand(0).getDefiningOp();
      if (!def || def->getName().getStringRef() != "array.extract" ||
          def->getNumOperands() != 2)
        return;
      Operation *idxDef = def->getOperand(1).getDefiningOp();
      if (!idxDef || idxDef->getName().getStringRef() != "arith.constant")
        return;
      auto intAttr = idxDef->getAttrOfType<IntegerAttr>("value");
      if (!intAttr)
        return;
      llvm::APInt apInt = intAttr.getValue();
      // index-typed constants are 64-bit; struct-of-pods indices are
      // always small non-negative — reject anything weird defensively.
      if (apInt.getBitWidth() > 64 || apInt.isNegative())
        return;
      int64_t k = apInt.getSExtValue();
      StringRef cls = structTy.getNameRef().getLeafReference().getValue();
      writers.push_back({op, cls, k});
      return;
    }

    // Reader source can be either `llzk.nondet : !struct<@<C>>` (post
    // post-loop pod.read→nondet conversion at line ~5915) OR
    // `pod.read %something[@comp] : !struct<@<C>>` (BEFORE that conversion,
    // i.e. during the outer fixed point — which is when this pass runs).
    // Both produce the same downstream `struct.readm [@F]` consumer.
    // Match against both shapes so the materializer fires inside the
    // outer fixed point and the cleanup phase finds nothing left to
    // nondet-replace for the cascade arms.
    bool isStructReaderSrc =
        (name == "llzk.nondet" && op->getNumResults() == 1) ||
        (name == "pod.read" && op->getNumResults() == 1);
    if (isStructReaderSrc) {
      auto structTy =
          dyn_cast<llzk::component::StructType>(op->getResult(0).getType());
      if (!structTy)
        return;
      // pod.read must be on the dispatch pod's @comp field — otherwise
      // it's reading some unrelated pod member that happens to be struct-
      // typed (no such case exists in current chips, but the explicit
      // gate keeps the matcher narrow).
      if (name == "pod.read") {
        auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (!rn || rn.getValue() != "comp")
          return;
      }
      StringRef cls = structTy.getNameRef().getLeafReference().getValue();
      for (OpOperand &use : op->getResult(0).getUses()) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() != "struct.readm" ||
            user->getNumResults() != 1)
          continue;
        auto memAttr = user->getAttrOfType<FlatSymbolRefAttr>("member_name");
        if (!memAttr)
          continue;
        readers.push_back(
            {user, cls, memAttr.getValue(), user->getResult(0).getType()});
      }
    }
  });

  if (writers.empty() || readers.empty())
    return false;

  // Build class→K map. Multiple writers with the same (class, K) are fine
  // (duplicate cascade arms emit identical results — last-write-wins).
  // Conflicting K for the same class is a malformed cascade — bail
  // defensively rather than silently corrupting the carrier.
  llvm::StringMap<int64_t> classToK;
  for (const Writer &w : writers) {
    auto it = classToK.find(w.className);
    if (it == classToK.end())
      classToK[w.className] = w.kConst;
    else if (it->second != w.kConst)
      return false;
  }

  // Cap dispatch dim N at max K + 1. Use the pre-scan's max K (covering
  // both hoisted and not-yet-hoisted cascade arms) so the carrier shape
  // is stable across outer iters — see pre-scan comment above for the
  // partial-hoist trap this avoids.
  int64_t N = 0;
  for (const auto &kv : classToK)
    N = std::max(N, kv.getValue() + 1);
  N = std::max(N, preScanMaxK + 1);

  // Group readers by @F. Drop readers whose class has no writer (orphan
  // nondets from unrelated dispatch chains). Drop fields whose readers
  // disagree on the inner type (heterogeneous-inner @F can't share one
  // carrier).
  struct FieldPlan {
    Type fieldTy;
    Value carrier;
    SmallVector<Reader *> targetReaders;
    bool fieldTyConflict = false;
  };
  llvm::StringMap<FieldPlan> fieldPlans;
  for (Reader &r : readers) {
    if (!classToK.count(r.className))
      continue;
    auto it = fieldPlans.find(r.field);
    if (it == fieldPlans.end()) {
      fieldPlans[r.field] = {r.fieldTy, Value(), {&r}, false};
      continue;
    }
    if (it->second.fieldTyConflict)
      continue;
    if (it->second.fieldTy != r.fieldTy) {
      it->second.fieldTyConflict = true;
      it->second.targetReaders.clear();
      continue;
    }
    it->second.targetReaders.push_back(&r);
  }
  SmallVector<StringRef> toEraseFields;
  for (auto &kv : fieldPlans)
    if (kv.second.fieldTyConflict || kv.second.targetReaders.empty())
      toEraseFields.push_back(kv.getKey());
  for (StringRef k : toEraseFields)
    fieldPlans.erase(k);
  if (fieldPlans.empty())
    return false;

  // Verify each `@F` is `{llzk.pub}` on every writer class's struct.def.
  // The LLZK verifier rejects `struct.readm` of a non-pub member from
  // outside the member's parent struct (see CLAUDE.md "Load-Bearing
  // Invariants"). Poseidon's `@out` is pub by Circom contract, but a
  // future cascade with private members would silently fail the LIT /
  // verifier — drop those fields rather than emit illegal IR.
  ModuleOp moduleOp = getTopLevelModule(funcBlock);
  if (moduleOp) {
    llvm::StringSet<> classNames;
    for (const auto &kv : classToK)
      classNames.insert(kv.getKey());
    SmallVector<StringRef> nonPubFields;
    for (auto &kv : fieldPlans) {
      StringRef fieldName = kv.getKey();
      bool allPub = true;
      for (StringRef cls : classNames.keys()) {
        bool foundPub = false;
        moduleOp->walk([&](Operation *defOp) {
          if (defOp->getName().getStringRef() != "struct.def")
            return WalkResult::advance();
          auto sym = defOp->getAttrOfType<StringAttr>("sym_name");
          if (!sym || sym.getValue() != cls)
            return WalkResult::advance();
          defOp->walk([&](Operation *m) {
            if (m->getName().getStringRef() != "struct.member")
              return;
            auto memSym = m->getAttrOfType<StringAttr>("sym_name");
            if (!memSym || memSym.getValue() != fieldName)
              return;
            if (m->hasAttr(llzk::PublicAttr::name))
              foundPub = true;
          });
          return WalkResult::interrupt();
        });
        if (!foundPub) {
          allPub = false;
          break;
        }
      }
      if (!allPub)
        nonPubFields.push_back(fieldName);
    }
    for (StringRef k : nonPubFields)
      fieldPlans.erase(k);
  }
  if (fieldPlans.empty())
    return false;

  // Allocate one carrier per surviving `@F` at funcBlock front. Front
  // placement guarantees dominance over every writer + reader scf.while
  // body in the function. Carriers are tagged with a `mSoPCF.carrier-for`
  // attribute so subsequent outer iters of this pass — which see
  // additional writers hoisted incrementally by Phase 1 (one or a few per
  // outer fixed-point iter, not all at once) — can REUSE the same carrier
  // instead of allocating fresh ones. Without reuse, the per-iter
  // allocations leak ~N orphan `array.new` ops, each carrying ~1 insert,
  // and `processBlockForArrayMutations` only threads ONE carrier as
  // iter-arg per scf.while body, leaving the others orphaned and DCE-bait.
  StringRef kCarrierAttr = "mSoPCF.carrier-for";
  Location loc = funcBlock.getParentOp()->getLoc();
  // First pass: locate existing carriers tagged for each surviving field.
  llvm::StringMap<Value> existingCarriers;
  for (Operation &op : funcBlock) {
    if (op.getName().getStringRef() != "array.new")
      continue;
    auto attr = op.getAttrOfType<StringAttr>(kCarrierAttr);
    if (attr && op.getNumResults() == 1)
      existingCarriers[attr.getValue()] = op.getResult(0);
  }
  OpBuilder builder(&funcBlock, funcBlock.begin());
  for (auto &kv : fieldPlans) {
    StringRef fieldName = kv.getKey();
    auto it = existingCarriers.find(fieldName);
    if (it != existingCarriers.end()) {
      kv.second.carrier = it->second;
      continue;
    }
    auto carrierTy = combineDispatchAndInnerFeltDims(kv.second.fieldTy, {N});
    auto newOp = builder.create<llzk::array::CreateArrayOp>(loc, carrierTy);
    newOp->setAttr(kCarrierAttr, builder.getStringAttr(fieldName));
    kv.second.carrier = newOp;
  }

  // Writer-side: after each hoisted call, emit `%v = struct.readm
  // %call[@F]; array.write|insert %carrier_F[%cK] = %v` for every
  // surviving `@F`. Duplicate (class, K) writers (e.g. Poseidon K=0
  // hit twice from cascade arm collapse) overwrite the same carrier
  // slot — semantically a no-op since both produce identical results.
  for (const Writer &w : writers) {
    OpBuilder wb(w.call);
    wb.setInsertionPointAfter(w.call);
    OperationState idxState(w.call->getLoc(), "arith.constant");
    idxState.addAttribute("value", wb.getIndexAttr(w.kConst));
    idxState.addTypes({wb.getIndexType()});
    Value kIdx = wb.create(idxState)->getResult(0);
    for (auto &kv : fieldPlans) {
      StringRef fieldName = kv.getKey();
      Type fieldTy = kv.second.fieldTy;
      Value feltVal = wb.create<llzk::component::MemberReadOp>(
          w.call->getLoc(), fieldTy, w.call->getResult(0),
          wb.getStringAttr(fieldName));
      StringRef writeOpName =
          isa<llzk::array::ArrayType>(fieldTy) ? "array.insert" : "array.write";
      OperationState writeState(w.call->getLoc(), writeOpName);
      writeState.addOperands({kv.second.carrier, kIdx, feltVal});
      wb.create(writeState);
    }
    // Mark the call so the next outer iter's walker skips it.
    w.call->setAttr(kMaterializedAttr, wb.getUnitAttr());
  }

  // Reader-side: replace every `struct.readm %nondet[@F]` with a read
  // from `%carrier_F[%cK_for_class]`. Scalar `@F` uses `array.read`
  // (full indices → single element); array-typed `@F` uses
  // `array.extract` (partial indices → sub-array slice). Mirrors
  // `materializePodArrayCompField`'s reader rewrite at line ~1962.
  SmallVector<Operation *> readmsToErase;
  for (auto &kv : fieldPlans) {
    Type fieldTy = kv.second.fieldTy;
    Value carrier = kv.second.carrier;
    for (Reader *r : kv.second.targetReaders) {
      int64_t k = classToK[r->className];
      OpBuilder rb(r->readm);
      OperationState idxState(r->readm->getLoc(), "arith.constant");
      idxState.addAttribute("value", rb.getIndexAttr(k));
      idxState.addTypes({rb.getIndexType()});
      Value kIdx = rb.create(idxState)->getResult(0);
      StringRef readOpName =
          isa<llzk::array::ArrayType>(fieldTy) ? "array.extract" : "array.read";
      OperationState readState(r->readm->getLoc(), readOpName);
      readState.addOperands({carrier, kIdx});
      readState.addTypes({fieldTy});
      Value extracted = rb.create(readState)->getResult(0);
      r->readm->getResult(0).replaceAllUsesWith(extracted);
      readmsToErase.push_back(r->readm);
    }
  }
  for (Operation *op : readmsToErase)
    op->erase();

  return true;
}

bool materializePodArrayInputPodField(Block &funcBlock) {
  SmallVector<Operation *> toErase;

  funcBlock.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "pod.write" || op->getNumOperands() < 2)
      return;
    auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!rn || rn.getValue() != "in")
      return;

    Value cell = op->getOperand(0);
    Value src = op->getOperand(1);

    // Array-element only. Scalar input pods (`pod.new : <[@in:...]>`) are
    // handled by `inlineInputPodCarries`; firing on a scalar risks
    // RAUW-ing a self-referential read-modify-write (`%v = pod.read
    // %p[@in]; pod.write %p[@in] = %v`) where erasing the pod.read leaves
    // dangling references in the function.call we just rewired.
    Operation *cellDef = cell.getDefiningOp();
    if (!cellDef || cellDef->getName().getStringRef() != "array.read")
      return;
    auto podTy = dyn_cast<llzk::pod::PodType>(cell.getType());
    if (!podTy)
      return;
    auto recs = podTy.getRecords();
    if (recs.size() != 1 || recs[0].getName() != "in")
      return;

    // Read-modify-write guard: an array-typed `@in: !array<...>` cell uses
    // the same pattern at the inner array (`%v = pod.read %cell[@in];
    // array.write %v[i] = %x; pod.write %cell[@in] = %v`). Defer to
    // `eliminatePodDispatch`'s tracker (mirrors `extractCallsFromScfIf`
    // line 287-303).
    if (auto *srcDef = src.getDefiningOp()) {
      if (srcDef->getName().getStringRef() == "pod.read" &&
          srcDef->getOperand(0) == cell) {
        auto srcRn = srcDef->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (srcRn && srcRn.getValue() == "in")
          return;
      }
    }

    for (OpOperand &use : llvm::make_early_inc_range(cell.getUses())) {
      Operation *user = use.getOwner();
      if (user == op || use.getOperandNumber() != 0)
        continue;
      if (user->getName().getStringRef() != "pod.read" ||
          user->getNumResults() == 0)
        continue;
      auto rn2 = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!rn2 || rn2.getValue() != "in")
        continue;
      user->getResult(0).replaceAllUsesWith(src);
      toErase.push_back(user);
    }
  });

  for (Operation *op : toErase)
    op->erase();
  return !toErase.empty();
}

/// Materialize a tail `function.call` after the writer-while for cross-block
/// readbacks of a function-scope SCALAR `pod.new : <[..., @comp: !struct,
/// ...]>`.
///
/// Distinct from `materializePodArrayCompField` (which handles `array.new : <N
/// x !pod<...>>`): this variant fires when the dispatch storage is a single
/// pod whose `@comp` is written under `scf.if` in one `scf.while` body and
/// read back from a different block. The per-block trackers in
/// `eliminatePodDispatch` cannot bridge this — Phase 5 nondets the cross-
/// block reader and silently zeroes the dispatch result. Symptom on circomlib
/// `gates.circom` users with a *scalar* outer dispatch (keccak iota3/iota10's
/// `XorArray_2` invocation): @main never calls `<gate>_compute` and the
/// reader-loop's lane reads `dense<0>` for the dispatch's output range.
///
/// The fix exploits the count-countdown invariant of circomlib's dispatch
/// pattern: the substantively-firing call's operands match the writer-while's
/// post-loop iter-arg state. After `unpackPodWhileCarry` runs, the writer-
/// while's body-block args carry one felt-array per dispatch input, mutated
/// in-place by `array.write` per iteration. The LAST writer's call (in source
/// order) names these block args directly — its post-while projection is
/// exactly `whileOp.getResult(i)` for each block-arg operand.
///
/// Multi-while: Mux2/Mux3 templates emit two sequential scf.whiles (one per
/// dispatch input field — e.g. an @c-loop and an @s-loop) sharing the same
/// dispatch pod. The count countdown fires only on the SECOND loop's last
/// iteration, so the LAST writer is in the LAST while in funcBlock source
/// order; its enclosing while is the projection target. The per-while body
/// tracking otherwise generalizes unchanged.
///
/// We emit `%postCall = function.call @F(<post-while operand values>)` after
/// the writer-while terminator and replace each cross-block `pod.read
/// %pod[@comp]` with `%postCall`. Existing Phase 4/5 cleanup DCEs the
/// (orphaned) writer scf.if + pod.write + pod.new traffic.
///
/// Driver order: AFTER `unpackPodWhileCarry` (so writer-while iter-args are
/// felt/array, not pod — call operands are direct block args), BEFORE
/// `eliminatePodDispatch` (so writer scf.ifs + pod.writes are still intact).
bool materializeScalarPodCompField(Block &funcBlock) {
  bool changed = false;

  // 1. Function-scope candidates: `pod.new` or `llzk.nondet` producing a
  //    dispatch pod whose @comp field is a struct. Post-circom-llzk PR #390
  //    (2026-04-30) the dispatch pod's pod-creation site is emitted as
  //    `llzk.nondet : !pod.type<[@count, @comp, @params]>` instead of
  //    `pod.new {@count = const_N}`; the cross-block @comp readback shape is
  //    otherwise identical, so the same materialization is sound.
  //    (Array-of-pods is `materializePodArrayCompField`'s concern.)
  struct Candidate {
    Operation *podDef;
    llzk::component::StructType compTy;
  };
  SmallVector<Candidate> candidates;
  for (Operation &op : funcBlock) {
    StringRef opName = op.getName().getStringRef();
    if ((opName != "pod.new" && opName != "llzk.nondet") ||
        op.getNumResults() == 0)
      continue;
    auto podTy = dyn_cast<llzk::pod::PodType>(op.getResult(0).getType());
    if (!podTy)
      continue;
    for (auto rec : podTy.getRecords()) {
      if (rec.getName() != "comp")
        continue;
      auto compTy = dyn_cast<llzk::component::StructType>(rec.getType());
      if (!compTy)
        break;
      candidates.push_back({&op, compTy});
      break;
    }
  }

  for (auto &cand : candidates) {
    Operation *podDef = cand.podDef;
    Value pod = podDef->getResult(0);

    struct Writer {
      Operation *callOp; // function.call result fed into pod.write[@comp].
      scf::WhileOp writerWhile; // enclosing scf.while, or null if writer is at
                                // function scope (e.g. inside an scf.if hanging
                                // directly off funcBlock — getValueByIndex's
                                // @main pattern).
    };
    SmallVector<Writer> writers;
    SmallVector<Operation *> readers;

    // 2. Walk pod uses, classify writers and readers.
    for (OpOperand &use : pod.getUses()) {
      Operation *user = use.getOwner();
      if (use.getOperandNumber() != 0)
        continue;
      auto field = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!field || field.getValue() != "comp")
        continue;
      StringRef name = user->getName().getStringRef();

      if (name == "pod.write" && user->getNumOperands() >= 2) {
        Operation *callOp = user->getOperand(1).getDefiningOp();
        if (!callOp || callOp->getName().getStringRef() != "function.call")
          continue;
        scf::WhileOp w = user->getParentOfType<scf::WhileOp>();
        if (w) {
          Block &writerBody = w.getAfter().front();
          if (!writerBody.findAncestorOpInBlock(*user))
            continue;
        }
        writers.push_back({callOp, w});
      } else if (name == "pod.read" && user->getNumResults() > 0) {
        readers.push_back(user);
      }
    }

    if (readers.empty())
      continue;

    // 2b. No-writer path: only readers exist (constant-table sub-component
    //     dispatch, e.g. keccak's RC_0). Synthesize a zero-arg call at
    //     function scope and rebind readers. Gate on the @compute method
    //     actually being zero-arg (looked up via the module SymbolTable)
    //     so we never invent operands for a sub-component that needs them.
    if (writers.empty()) {
      llzk::component::StructType compTy = cand.compTy;
      SymbolRefAttr structRef = compTy.getNameRef();
      MLIRContext *ctx = structRef.getContext();
      SmallVector<FlatSymbolRefAttr> nested(structRef.getNestedReferences());
      nested.push_back(FlatSymbolRefAttr::get(ctx, "compute"));
      auto callee = SymbolRefAttr::get(structRef.getRootReference(), nested);
      ModuleOp module = getTopLevelModule(funcBlock);
      auto fnDef = dyn_cast_or_null<llzk::function::FuncDefOp>(
          SymbolTable::lookupSymbolIn(module, callee));
      if (!fnDef || fnDef.getFunctionType().getNumInputs() != 0)
        continue;

      OpBuilder builder(podDef);
      builder.setInsertionPointAfter(podDef);
      Operation *postCall = builder.create<llzk::function::CallOp>(
          podDef->getLoc(), TypeRange{compTy}, callee, ValueRange{});
      Value postCallResult = postCall->getResult(0);
      for (Operation *r : readers) {
        r->getResult(0).replaceAllUsesWith(postCallResult);
        r->erase();
      }
      changed = true;
      continue;
    }

    // 3. Pick the LAST writer in funcBlock source order; its enclosing
    //    scf.while (or, for function-scope writers, its funcBlock-level
    //    anchor) is the projection target. Single-writer inputs collapse to
    //    the original behavior. Per-writer source-order is well-defined at
    //    funcBlock scope across all shapes (in-while + function-scope).
    auto funcAnchorOf = [&](Operation *op) -> Operation * {
      return funcBlock.findAncestorOpInBlock(*op);
    };
    Writer *lastWriter = &writers.front();
    Operation *lastFuncAnchor = funcAnchorOf(lastWriter->callOp);
    if (!lastFuncAnchor)
      continue;
    for (auto &w : writers) {
      Operation *fa = funcAnchorOf(w.callOp);
      if (!fa)
        continue;
      if (lastFuncAnchor->isBeforeInBlock(fa)) {
        lastWriter = &w;
        lastFuncAnchor = fa;
        continue;
      }
      // Same funcBlock anchor (e.g. two writers in one scf.while body, or
      // two writers in one funcBlock-level scf.if): isBeforeInBlock is
      // ill-defined at funcBlock level. Disambiguate at the smallest block
      // that contains both call ops by walking their parent chains until
      // they share a common block, then comparing direct-children-of-that-
      // block ancestors. Skip pairs whose call ops share no common block
      // (e.g. different scf.if regions) — source order between disjoint
      // execution regions has no meaningful answer; keep the earlier-seen
      // writer.
      if (fa != lastFuncAnchor)
        continue;
      for (Block *b = w.callOp->getBlock(); b;
           b = b->getParentOp() ? b->getParentOp()->getBlock() : nullptr) {
        Operation *aAnc = b->findAncestorOpInBlock(*lastWriter->callOp);
        Operation *bAnc = b->findAncestorOpInBlock(*w.callOp);
        if (aAnc && bAnc) {
          if (aAnc->isBeforeInBlock(bAnc))
            lastWriter = &w;
          break;
        }
      }
    }
    scf::WhileOp writerWhile = lastWriter->writerWhile;
    // The op `def` must dominate to be usable at funcBlock scope.
    Operation *dominanceScope =
        writerWhile ? writerWhile.getOperation() : lastFuncAnchor;

    // 4. Cross-block guard: drop readers nested under any writer-while body
    //    (those are forwarded same-block by `eliminatePodDispatch`'s tracker).
    //    Function-scope writers contribute no body — they're already at
    //    funcBlock level.
    llvm::DenseSet<Block *> writerBodies;
    for (auto &w : writers)
      if (w.writerWhile)
        writerBodies.insert(&w.writerWhile.getAfter().front());
    SmallVector<Operation *> crossBlockReaders;
    for (auto *r : readers) {
      bool inAnyWriterBody = false;
      for (Block *wb : writerBodies)
        if (wb->findAncestorOpInBlock(*r)) {
          inAnyWriterBody = true;
          break;
        }
      if (!inAnyWriterBody)
        crossBlockReaders.push_back(r);
    }
    if (crossBlockReaders.empty())
      continue;

    // 5. Dominance gate: every cross-block reader must follow lastFuncAnchor
    //    so RAUW to the tail call won't create use-before-def for readers
    //    sandwiched between writers.
    bool readersAfter = llvm::all_of(crossBlockReaders, [&](Operation *r) {
      Operation *rAnchor = funcAnchorOf(r);
      return rAnchor && lastFuncAnchor->isBeforeInBlock(rAnchor);
    });
    if (!readersAfter)
      continue;

    // 6. Resolve call operands at funcBlock scope. Body-block args of an
    //    in-while writer project to `whileOp.getResult(i)`; everything else
    //    must be defined outside `dominanceScope` (the writerWhile, or for
    //    function-scope writers, lastFuncAnchor) so it dominates the post-
    //    anchor insertion point. After `unpackPodWhileCarry` runs, function-
    //    scope `pod.read %carry[@field]` operands have already been replaced
    //    by unpacked while results — those satisfy the dominance check
    //    automatically.
    SmallVector<Value> resolvedOperands;
    bool resolveOK = true;
    for (Value operand : lastWriter->callOp->getOperands()) {
      if (auto ba = dyn_cast<BlockArgument>(operand)) {
        if (writerWhile && ba.getOwner() == &writerWhile.getAfter().front()) {
          resolvedOperands.push_back(writerWhile.getResult(ba.getArgNumber()));
          continue;
        }
        if (ba.getOwner() == &funcBlock) {
          resolvedOperands.push_back(operand);
          continue;
        }
        resolveOK = false;
        break;
      }
      Operation *def = operand.getDefiningOp();
      if (!def || dominanceScope->isAncestor(def)) {
        resolveOK = false;
        break;
      }
      resolvedOperands.push_back(operand);
    }
    if (!resolveOK)
      continue;

    // 7. Emit tail call after `dominanceScope` (the writerWhile, or the
    //    funcBlock-level scf.if/etc holding the last function-scope writer).
    Operation *insertAfter = dominanceScope;
    OpBuilder builder(insertAfter);
    builder.setInsertionPointAfter(insertAfter);
    auto callee = lastWriter->callOp->getAttrOfType<SymbolRefAttr>("callee");
    if (!callee)
      continue;
    Operation *postCall = builder.create<llzk::function::CallOp>(
        lastWriter->callOp->getLoc(), lastWriter->callOp->getResultTypes(),
        callee, resolvedOperands);
    Value postCallResult = postCall->getResult(0);

    // 8. Replace cross-block reader pod.read results with the tail-call
    //    result. The reader's chained `struct.readm [@F]` consumers now
    //    read from the materialized struct directly.
    for (Operation *r : crossBlockReaders) {
      r->getResult(0).replaceAllUsesWith(postCallResult);
      r->erase();
    }

    changed = true;
  }

  return changed;
}

/// Returns true if `arr` traces back to `podArrBlockArg` through any number of
/// nested scf.while iter-args (or is `podArrBlockArg` directly).
bool valueTracesToPodArr(Value arr, Value podArrBlockArg) {
  // Defensive depth cap — well-formed MLIR has acyclic SSA, but iter-arg
  // BlockArgument cycles are theoretically reachable through ill-formed IR.
  for (unsigned hops = 0; hops < 64; ++hops) {
    if (arr == podArrBlockArg)
      return true;
    auto ba = dyn_cast<BlockArgument>(arr);
    if (!ba)
      return false;
    auto pw = dyn_cast<scf::WhileOp>(ba.getOwner()->getParentOp());
    if (!pw)
      return false;
    unsigned idx = ba.getArgNumber();
    if (idx >= pw.getNumOperands())
      return false;
    arr = pw.getOperand(idx);
  }
  return false;
}

// Forward declaration: rewriting a body recurses through nested whiles via
// `expandPodArrayWhile`, which in turn recurses back via this helper.
void rewritePodArrayUsesInBlock(Block &blk, Value oldArrArg,
                                ArrayRef<Value> perFieldArrs,
                                ArrayRef<StringRef> fieldOrder,
                                const llvm::StringMap<Type> &fieldArrTypes);

/// Replace `podRead` (a `pod.read %x[@field]`) with an `array.read` /
/// `array.extract` on `fieldArr` at `indices`. RAUWs uses of `podRead`'s
/// result; caller owns erasure timing (deferred via `toErase` or immediate).
Operation *buildPerFieldRead(Operation *podRead, Value fieldArr,
                             ArrayRef<Value> indices) {
  OpBuilder b(podRead);
  Type resultType = podRead->getResult(0).getType();
  StringRef opName =
      isa<llzk::array::ArrayType>(resultType) ? "array.extract" : "array.read";
  OperationState state(podRead->getLoc(), opName);
  SmallVector<Value> ops{fieldArr};
  ops.append(indices.begin(), indices.end());
  state.addOperands(ops);
  state.addTypes({resultType});
  Operation *newRead = b.create(state);
  podRead->getResult(0).replaceAllUsesWith(newRead->getResult(0));
  return newRead;
}

/// Replace `whileOp`'s pod-array iter-arg at `podArrIdx` with N per-field
/// array iter-args. Per-field inits at the call site are `outerPerFieldInits`
/// (must dominate `whileOp`). Recursively rewrites the body. Erases `whileOp`.
/// Returns the new scf.while op.
scf::WhileOp expandPodArrayWhile(scf::WhileOp whileOp, unsigned podArrIdx,
                                 ArrayRef<Value> outerPerFieldInits,
                                 ArrayRef<StringRef> fieldOrder,
                                 const llvm::StringMap<Type> &fieldArrTypes) {
  OpBuilder builder(whileOp);
  Location loc = whileOp.getLoc();

  // Build expanded operand and type lists.
  SmallVector<Value> newInits;
  SmallVector<Type> newTypes;
  for (unsigned i = 0; i < whileOp.getNumOperands(); ++i) {
    if (i == podArrIdx) {
      for (size_t f = 0; f < fieldOrder.size(); ++f) {
        newInits.push_back(outerPerFieldInits[f]);
        newTypes.push_back(fieldArrTypes.lookup(fieldOrder[f]));
      }
    } else {
      newInits.push_back(whileOp.getOperand(i));
      newTypes.push_back(whileOp.getOperand(i).getType());
    }
  }

  auto newWhile = builder.create<scf::WhileOp>(loc, newTypes, newInits);
  newWhile.getBefore().takeBody(whileOp.getBefore());
  newWhile.getAfter().takeBody(whileOp.getAfter());

  for (int ri = 0; ri < 2; ++ri) {
    Region &region = ri == 0 ? newWhile.getBefore() : newWhile.getAfter();
    Block &blk = region.front();
    Value oldArrArg = blk.getArgument(podArrIdx);

    // Insert per-field block args after podArrIdx.
    SmallVector<Value> perFieldArgs;
    for (size_t f = 0; f < fieldOrder.size(); ++f) {
      auto arg = blk.insertArgument(podArrIdx + 1 + f,
                                    fieldArrTypes.lookup(fieldOrder[f]), loc);
      perFieldArgs.push_back(arg);
    }

    // Rewrite body uses of oldArrArg (recursively expands nested whiles that
    // take oldArrArg as init operand).
    rewritePodArrayUsesInBlock(blk, oldArrArg, perFieldArgs, fieldOrder,
                               fieldArrTypes);

    // Replace any remaining uses of oldArrArg with nondet (typically dead
    // captures left over after rewrite).
    if (!oldArrArg.use_empty()) {
      OpBuilder nb(newWhile);
      oldArrArg.replaceAllUsesWith(createNondet(nb, loc, oldArrArg.getType()));
    }

    // Expand the terminator's operand at podArrIdx with per-field block args.
    // Circom-emitted forwarder loops yield the iter-arg unchanged; LLZK array
    // mutability means per-field array.writes inside the body update the
    // arrays in place, so the per-field block args are the correct yield.
    Operation *term = blk.getTerminator();
    unsigned offset = isa<scf::ConditionOp>(term) ? 1 : 0;
    unsigned expandIdx = podArrIdx + offset;
    SmallVector<Value> newTermArgs;
    for (unsigned i = 0; i < term->getNumOperands(); ++i) {
      if (i == expandIdx) {
        for (Value pfa : perFieldArgs)
          newTermArgs.push_back(pfa);
      } else {
        newTermArgs.push_back(term->getOperand(i));
      }
    }
    term->setOperands(newTermArgs);

    blk.eraseArgument(podArrIdx);
  }

  // Replace non-pod-array results of old while with corresponding new results.
  replaceNonPodWhileResults(whileOp, newWhile, podArrIdx, fieldOrder.size());

  // Handle post-while users of the pod-array result: rewrite
  // `array.read %podArrResult[%i]; pod.read %x[@field]` chains to read
  // directly from the corresponding per-field result; erase struct.writem;
  // nondet anything else. Without the per-field rewrite, an outer scf.while
  // body that consumes a nested scf.while's flattened pod-array result
  // (canonical case: maci_splicer's outer Splicer body consuming the
  // inner QuinSelector @in fill loop's post-while result) loses the
  // per-field connection — the materializer's hoisted `function.call`
  // operand collapses to const-zero in the lowered StableHLO.
  llvm::StringMap<unsigned> fieldIdx;
  for (size_t f = 0; f < fieldOrder.size(); ++f)
    fieldIdx[fieldOrder[f]] = (unsigned)f;
  Value podArrResult = whileOp.getResult(podArrIdx);
  SmallVector<Operation *> postErase;
  for (OpOperand &use : llvm::make_early_inc_range(podArrResult.getUses())) {
    Operation *user = use.getOwner();
    if (user->getName().getStringRef() == "struct.writem") {
      postErase.push_back(user);
      continue;
    }
    if (user->getName().getStringRef() != "array.read" ||
        user->getNumOperands() < 2 || user->getNumResults() == 0)
      continue;
    Value podCell = user->getResult(0);
    SmallVector<Value> readIndices = arrayAccessIndices(user);
    for (OpOperand &subUse : llvm::make_early_inc_range(podCell.getUses())) {
      Operation *subUser = subUse.getOwner();
      if (subUser->getName().getStringRef() != "pod.read" ||
          subUser->getNumResults() == 0)
        continue;
      auto rn = subUser->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!rn)
        continue;
      auto fIt = fieldIdx.find(rn.getValue());
      if (fIt == fieldIdx.end())
        continue;
      buildPerFieldRead(subUser, newWhile.getResult(podArrIdx + fIt->second),
                        readIndices);
      subUser->erase();
    }
    if (podCell.use_empty())
      postErase.push_back(user);
  }
  for (auto *op : postErase)
    op->erase();
  if (!podArrResult.use_empty()) {
    OpBuilder nb(newWhile->getNextNode());
    podArrResult.replaceAllUsesWith(
        createNondet(nb, loc, podArrResult.getType()));
  }

  whileOp->dropAllReferences();
  whileOp->erase();
  return newWhile;
}

void rewritePodArrayUsesInBlock(Block &blk, Value oldArrArg,
                                ArrayRef<Value> perFieldArrs,
                                ArrayRef<StringRef> fieldOrder,
                                const llvm::StringMap<Type> &fieldArrTypes) {
  llvm::StringMap<unsigned> fieldIdx;
  for (size_t f = 0; f < fieldOrder.size(); ++f)
    fieldIdx[fieldOrder[f]] = f;

  // (Note: recursive expansion of nested scf.whiles using oldArrArg as init
  // is intentionally omitted. An earlier prototype recursed but caused a
  // non-converging fixed-point on circuits with multi-instance sub-component
  // input pods — `eliminatePodDispatch` kept reporting changes after our
  // expand. The recursion is replaced by relying on the outer fixed-point
  // loop and `processNested` to revisit nested whiles in subsequent
  // iterations, plus the rewire pass below to reconnect inits when a
  // nested while was already flattened in a prior iteration.)

  // Rewire nested whiles already flattened in a prior fixed-point iteration:
  // their inits are a contiguous run of `llzk.nondet` matching the per-field
  // types we're flattening now. Without this rewire, the outer carry into
  // those whiles would be lost (each outer iteration would reset the inner
  // arrays to nondet zeros and discard accumulated writes).
  blk.walk([&](scf::WhileOp nw) {
    if (nw.getNumOperands() < fieldOrder.size())
      return WalkResult::advance();
    for (unsigned start = 0;
         start + fieldOrder.size() <= nw.getNumOperands();) {
      bool match = true;
      for (size_t f = 0; f < fieldOrder.size(); ++f) {
        Value v = nw.getOperand(start + f);
        if (v.getType() != fieldArrTypes.lookup(fieldOrder[f])) {
          match = false;
          break;
        }
        Operation *def = v.getDefiningOp();
        if (!def || def->getName().getStringRef() != "llzk.nondet") {
          match = false;
          break;
        }
      }
      if (match) {
        for (size_t f = 0; f < fieldOrder.size(); ++f)
          nw.setOperand(start + f, perFieldArrs[f]);
        start += fieldOrder.size();
      } else {
        start += 1;
      }
    }
    return WalkResult::advance();
  });

  // Rewrite array.read/pod.read/pod.write/array.write on `oldArrArg` and the
  // pod values produced by reads of it.
  llvm::DenseMap<Value, SmallVector<Value>> localPodToIndices;
  SmallVector<Operation *> toErase;
  blk.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();

    // array.read on oldArrArg → track all indices, erase.
    if (name == "array.read" && op->getNumOperands() > 1 &&
        op->getOperand(0) == oldArrArg && op->getNumResults() > 0) {
      localPodToIndices[op->getResult(0)] = arrayAccessIndices(op);
      toErase.push_back(op);
      return;
    }

    // pod.write on tracked pod → array.write/insert on per-field array.
    if (name == "pod.write" && op->getNumOperands() >= 2) {
      auto it = localPodToIndices.find(op->getOperand(0));
      if (it != localPodToIndices.end()) {
        auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn) {
          auto fIt = fieldIdx.find(rn.getValue());
          if (fIt != fieldIdx.end()) {
            Value fieldArr = perFieldArrs[fIt->second];
            Value val = op->getOperand(1);
            OpBuilder b(op);
            StringRef writeOp = isa<llzk::array::ArrayType>(val.getType())
                                    ? "array.insert"
                                    : "array.write";
            OperationState state(op->getLoc(), writeOp);
            SmallVector<Value> ops;
            ops.push_back(fieldArr);
            for (Value idx : it->second)
              ops.push_back(idx);
            ops.push_back(val);
            state.addOperands(ops);
            b.create(state);
            toErase.push_back(op);
            return;
          }
        }
      }
    }

    // pod.read on tracked pod → array.read/extract on per-field array.
    if (name == "pod.read" && op->getNumOperands() > 0 &&
        op->getNumResults() > 0) {
      auto it = localPodToIndices.find(op->getOperand(0));
      if (it != localPodToIndices.end()) {
        auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn) {
          auto fIt = fieldIdx.find(rn.getValue());
          if (fIt != fieldIdx.end()) {
            buildPerFieldRead(op, perFieldArrs[fIt->second], it->second);
            toErase.push_back(op);
            return;
          }
        }
      }
    }

    // array.write of tracked pod back to oldArrArg → erase (no-op pass).
    if (name == "array.write" && op->getNumOperands() >= 2 &&
        op->getOperand(0) == oldArrArg) {
      Value written = op->getOperands().back();
      if (localPodToIndices.count(written))
        toErase.push_back(op);
    }

    // scf.if returning oldArrArg type → forward result to oldArrArg.
    if (name == "scf.if" && op->getNumResults() > 0) {
      bool hasPodResult = false;
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (op->getResult(i).getType() == oldArrArg.getType()) {
          op->getResult(i).replaceAllUsesWith(oldArrArg);
          hasPodResult = true;
        }
      }
      if (hasPodResult && op->getNumResults() == 1) {
        OpBuilder b(op);
        auto newIf = b.create<scf::IfOp>(op->getLoc(), TypeRange{},
                                         op->getOperand(0), true);
        newIf.getThenRegion().takeBody(op->getRegion(0));
        newIf.getElseRegion().takeBody(op->getRegion(1));
        for (Region *r : {&newIf.getThenRegion(), &newIf.getElseRegion()}) {
          if (r->empty())
            continue;
          Operation *yield = r->front().getTerminator();
          if (yield)
            yield->setOperands({});
        }
        toErase.push_back(op);
      }
    }
  });

  for (auto *op : llvm::reverse(toErase)) {
    if (op->use_empty() || op->getNumResults() == 0)
      op->erase();
  }
}

/// Phase -1: Flatten array-of-pods (all-felt fields) carried through scf.while
/// into per-field felt arrays.
///   scf.while (%i, %pod_arr) : (felt, !array.type<2 x !pod.type<[@x: felt, @k:
///   felt]>>) → scf.while (%i, %arr_x, %arr_k) : (felt, !array.type<2 x felt>,
///   !array.type<2 x felt>)
/// Rewrites array.read+pod.read/pod.write+array.write → direct
/// array.read/write.
///
/// Handles arbitrarily-deep nested while-carry chains: when the pod-array
/// iter-arg is threaded through N nested whiles before reaching the body
/// that does the actual array.read, all N+1 whiles in the chain are
/// flattened simultaneously. The per-field arrays preserve the full source
/// dims so multi-dim sources don't alias on inner indices.
bool flattenPodArrayWhileCarry(Block &block) {
  // Find scf.while ops that carry array-of-pods.
  SmallVector<scf::WhileOp> whileOps;
  for (Operation &op : block) {
    if (auto w = dyn_cast<scf::WhileOp>(&op))
      whileOps.push_back(w);
  }

  for (scf::WhileOp whileOp : whileOps) {
    // Collect all pod-array iter-arg slots. Try them last-to-first and
    // flatten the first one whose body walk discovers fields. A single
    // unflattenable slot (e.g. an `@in: felt` carry whose pod.read/write
    // lives in a sibling region the body walk can't reach) must not block
    // peer slots from being processed; the outer fixed-point loop revisits
    // this while for the remaining carries afterwards.
    SmallVector<unsigned> podArrCandidates;
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      Type ty = whileOp.getResult(i).getType();
      if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
        if (arrTy.getElementType().getDialect().getNamespace() == "pod")
          podArrCandidates.push_back(i);
    }
    if (podArrCandidates.empty())
      continue;

    for (unsigned podArrIdx : llvm::reverse(podArrCandidates)) {
      Block &bodyBlock = whileOp.getAfter().front();
      Value podArrBlockArg = bodyBlock.getArgument(podArrIdx);

      llvm::StringMap<Type> fieldTypes;
      SmallVector<StringRef> fieldOrder;

      // Field discovery: walk body and follow array.read sources back to
      // `podArrBlockArg` through arbitrary scf.while-iter-arg chains. The
      // pod values produced by such reads carry the fields we need to flatten.
      llvm::DenseSet<Value> trackedPods;
      bodyBlock.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "array.read" &&
            op->getNumOperands() > 1 && op->getNumResults() > 0) {
          if (valueTracesToPodArr(op->getOperand(0), podArrBlockArg))
            trackedPods.insert(op->getResult(0));
        }
        auto discover = [&](StringRef fn, Type ty) {
          if (fieldTypes.count(fn) || !isFlattenableFelt(ty))
            return;
          fieldTypes[fn] = ty;
          fieldOrder.push_back(fn);
        };
        if (op->getName().getStringRef() == "pod.read" &&
            op->getNumOperands() > 0 && op->getNumResults() > 0 &&
            trackedPods.count(op->getOperand(0))) {
          if (auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name"))
            discover(rn.getValue(), op->getResult(0).getType());
        }
        if (op->getName().getStringRef() == "pod.write" &&
            op->getNumOperands() >= 2 && trackedPods.count(op->getOperand(0))) {
          if (auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name"))
            discover(rn.getValue(), op->getOperand(1).getType());
        }
      });

      // Canonicalize `fieldOrder` to the pod type's record declaration
      // order. Body-walk discovery order depends on which pod.read/write
      // the walker hits first, which differs between an outer scf.while
      // and a nested forwarder while taking the same pod-array as iter-arg.
      // The post-`runOnOperation` rewire pass below matches inner's
      // per-field operands against parent's block args by type; if the
      // orders disagree it falls back to homogeneous-split and can wire
      // inner's carry to a wrong-group same-type arg (canonical case:
      // maci_splicer's QuinSelector_6 @in fill loop's @index carry
      // colliding with mux's @s carry).
      if (auto arrTy = dyn_cast<llzk::array::ArrayType>(
              whileOp.getResult(podArrIdx).getType())) {
        if (auto podTy = dyn_cast<llzk::pod::PodType>(arrTy.getElementType())) {
          SmallVector<StringRef> sorted;
          for (auto rec : podTy.getRecords())
            if (fieldTypes.count(rec.getName()))
              sorted.push_back(rec.getName());
          if (sorted.size() == fieldOrder.size())
            fieldOrder = std::move(sorted);
        }
      }

      // Fallback when body discovery comes up empty: extract @-fields from
      // the pod-array element type's record list directly. This handles the
      // common case where a nested scf.while was the only consumer of this
      // iter-arg and a prior outer-fixed-point iteration's `processNested`
      // already flattened the nested — its init operand has been rewritten
      // to local per-field nondets, severing the body-side `array.read`
      // chain that body discovery walks. Without this fallback the outer
      // pod-array stays threaded as `<D x !pod<[@a, @b]>>`, the AES xor_3
      // (4992 felts) / num2bits_1 chain stays disconnected, and the post-
      // flatten rewire below has no parent per-field block args to wire to.
      if (fieldOrder.empty()) {
        auto arrTy = cast<llzk::array::ArrayType>(
            whileOp.getResult(podArrIdx).getType());
        if (auto podTy = dyn_cast<llzk::pod::PodType>(arrTy.getElementType())) {
          bool allFlattenable = !podTy.getRecords().empty();
          for (auto rec : podTy.getRecords()) {
            if (!isFlattenableFelt(rec.getType())) {
              allFlattenable = false;
              break;
            }
          }
          if (allFlattenable) {
            for (auto rec : podTy.getRecords()) {
              StringRef fn = rec.getName();
              fieldTypes[fn] = rec.getType();
              fieldOrder.push_back(fn);
            }
          }
        }
      }

      if (fieldOrder.empty())
        continue;

      // Per-field array uses the full source dims (felt) or the source dims
      // followed by the inner array's dims (array-of-felt). Dropping inner
      // dims on a multi-dim source aliases all inner indices to slot 0.
      auto arrType =
          cast<llzk::array::ArrayType>(whileOp.getResult(podArrIdx).getType());
      auto dims = getArrayDimensions(arrType);
      if (dims.empty() || dims.front() <= 0)
        continue;

      OpBuilder builder(whileOp);
      Location loc = whileOp.getLoc();
      llvm::StringMap<Type> fieldArrTypes;
      for (StringRef fn : fieldOrder) {
        Type ft = fieldTypes[fn];
        if (auto innerArr = dyn_cast<llzk::array::ArrayType>(ft)) {
          auto innerDims = getArrayDimensions(innerArr);
          SmallVector<int64_t> combined(dims.begin(), dims.end());
          combined.append(innerDims.begin(), innerDims.end());
          fieldArrTypes[fn] =
              llzk::array::ArrayType::get(innerArr.getElementType(), combined);
        } else {
          fieldArrTypes[fn] = llzk::array::ArrayType::get(ft, dims);
        }
      }

      SmallVector<Value> outerPerFieldInits;
      for (StringRef fn : fieldOrder)
        outerPerFieldInits.push_back(
            createNondet(builder, loc, fieldArrTypes[fn]));

      expandPodArrayWhile(whileOp, (unsigned)podArrIdx, outerPerFieldInits,
                          fieldOrder, fieldArrTypes);

      return true; // Process one at a time.
    }
  }

  return false;
}

/// Replace each pod-array result slot of a result-bearing `scf.if` with N
/// per-field felt-array result slots. Both branches' `scf.yield` operands at
/// that slot are replaced with N fresh `llzk.nondet : <D x !felt>`
/// placeholders; downstream users of the old pod-array result are rewired to a
/// single fresh `llzk.nondet : <D x !pod>` so they remain well-typed but
/// orphan. The actual data flow through the new per-field slots is closed
/// later by `LlzkToStablehlo.cpp`'s `extendResultBearingScfIfArrayChain` once
/// `isPromotableCarryType` (which excludes pod-element arrays) starts tracking
/// the new felt-array slots — that pass walks each branch, finds the inner
/// scf.while's per-field SSA carries, and rewrites the branch yields to use
/// the latest values.
///
/// Mirrors `flattenPodArrayWhileCarry`'s record-list fallback for
/// field-discovery: only flattens slots whose pod element type's record list
/// is non-empty and uniformly felt-flattenable. Slot-index shifts are tracked
/// so non-pod result uses are rewired to the correct new index.
///
/// Returns true iff any rewrite fired. Idempotent: running twice on the same
/// block is a no-op since the second walk finds no pod-array result slots.
bool flattenPodArrayScfIfResults(Block &block) {
  SmallVector<scf::IfOp> ifOps;
  for (Operation &op : block)
    if (auto ifOp = dyn_cast<scf::IfOp>(&op))
      ifOps.push_back(ifOp);

  bool changed = false;

  for (scf::IfOp oldIf : ifOps) {
    if (oldIf.getNumResults() == 0)
      continue;
    if (oldIf.getThenRegion().empty() || oldIf.getElseRegion().empty())
      continue;

    // Collect pod-array result slots and the per-field flatten plan for each.
    struct SlotPlan {
      unsigned origIdx;
      SmallVector<StringRef> fieldOrder;
      llvm::StringMap<Type> fieldArrTypes;
    };
    SmallVector<SlotPlan> plans;
    for (unsigned i = 0; i < oldIf.getNumResults(); ++i) {
      auto arrTy =
          dyn_cast<llzk::array::ArrayType>(oldIf.getResult(i).getType());
      if (!arrTy)
        continue;
      auto podTy = dyn_cast<llzk::pod::PodType>(arrTy.getElementType());
      if (!podTy)
        continue;
      bool allFlattenable = !podTy.getRecords().empty();
      for (auto rec : podTy.getRecords()) {
        if (!isFlattenableFelt(rec.getType())) {
          allFlattenable = false;
          break;
        }
      }
      if (!allFlattenable)
        continue;
      auto dims = getArrayDimensions(arrTy);
      if (dims.empty() || dims.front() <= 0)
        continue;

      SlotPlan plan;
      plan.origIdx = i;
      for (auto rec : podTy.getRecords()) {
        StringRef fn = rec.getName();
        Type ft = rec.getType();
        plan.fieldOrder.push_back(fn);
        if (auto innerArr = dyn_cast<llzk::array::ArrayType>(ft)) {
          auto innerDims = getArrayDimensions(innerArr);
          SmallVector<int64_t> combined(dims.begin(), dims.end());
          combined.append(innerDims.begin(), innerDims.end());
          plan.fieldArrTypes[fn] =
              llzk::array::ArrayType::get(innerArr.getElementType(), combined);
        } else {
          plan.fieldArrTypes[fn] = llzk::array::ArrayType::get(ft, dims);
        }
      }
      plans.push_back(std::move(plan));
    }

    if (plans.empty())
      continue;

    // Build the new result-type list and remember where each old non-pod slot
    // lands. Flattened-slot positions are not needed: the slot rewire walks
    // the post-flatten new scf.if directly.
    SmallVector<Type> newResultTypes;
    SmallVector<unsigned> newIdxForOldNonFlattened(oldIf.getNumResults(), 0);
    unsigned planCursor = 0;
    unsigned newPos = 0;
    for (unsigned i = 0; i < oldIf.getNumResults(); ++i) {
      if (planCursor < plans.size() && plans[planCursor].origIdx == i) {
        for (StringRef fn : plans[planCursor].fieldOrder) {
          newResultTypes.push_back(plans[planCursor].fieldArrTypes.lookup(fn));
          ++newPos;
        }
        ++planCursor;
      } else {
        newIdxForOldNonFlattened[i] = newPos;
        newResultTypes.push_back(oldIf.getResult(i).getType());
        ++newPos;
      }
    }

    OpBuilder builder(oldIf);
    Location loc = oldIf.getLoc();
    auto newIf =
        builder.create<scf::IfOp>(loc, newResultTypes, oldIf.getCondition(),
                                  /*hasElse=*/true);
    newIf.getThenRegion().takeBody(oldIf.getThenRegion());
    newIf.getElseRegion().takeBody(oldIf.getElseRegion());

    // Rewrite each branch's yield: replace each pod-array operand with N
    // fresh per-field nondets. Non-flattened operands are forwarded.
    auto rewriteYield = [&](Block &branch) {
      auto yield = cast<scf::YieldOp>(branch.getTerminator());
      OpBuilder yb(yield);
      SmallVector<Value> newArgs;
      unsigned cursor = 0;
      for (unsigned i = 0; i < yield.getNumOperands(); ++i) {
        if (cursor < plans.size() && plans[cursor].origIdx == i) {
          for (StringRef fn : plans[cursor].fieldOrder) {
            Type ft = plans[cursor].fieldArrTypes.lookup(fn);
            newArgs.push_back(createNondet(yb, yield.getLoc(), ft));
          }
          ++cursor;
        } else {
          newArgs.push_back(yield.getOperand(i));
        }
      }
      yb.create<scf::YieldOp>(yield.getLoc(), newArgs);
      yield.erase();
    };
    rewriteYield(newIf.getThenRegion().front());
    rewriteYield(newIf.getElseRegion().front());

    // Rewire users of old scf.if results.
    planCursor = 0;
    for (unsigned i = 0; i < oldIf.getNumResults(); ++i) {
      Value oldRes = oldIf.getResult(i);
      if (planCursor < plans.size() && plans[planCursor].origIdx == i) {
        if (!oldRes.use_empty()) {
          OpBuilder nb(oldIf);
          oldRes.replaceAllUsesWith(createNondet(nb, loc, oldRes.getType()));
        }
        ++planCursor;
      } else {
        oldRes.replaceAllUsesWith(newIf.getResult(newIdxForOldNonFlattened[i]));
      }
    }

    oldIf.erase();
    changed = true;
  }

  return changed;
}

/// Phase 0: Unpack pod-typed scf.while carry values into individual fields.
/// Transforms:
///   scf.while (%i, %pod) : (felt, !pod.type<[@c: array, @s: felt]>)
/// Into:
///   scf.while (%i, %c, %s) : (felt, array, felt)
/// Uses takeBody to move regions, then modifies block args and pod ops
/// in-place.
bool unpackPodWhileCarry(Block &block) {
  bool changed = false;

  // Collect while ops to process (avoid walk invalidation).
  SmallVector<scf::WhileOp> whileOps;
  for (Operation &op : block)
    if (auto w = dyn_cast<scf::WhileOp>(&op))
      whileOps.push_back(w);

  for (scf::WhileOp whileOp : whileOps) {
    // Find pod-typed carry positions.
    SmallVector<unsigned> podCarryIndices;
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      Type ty = whileOp.getResult(i).getType();
      if (ty.getDialect().getNamespace() == "pod")
        podCarryIndices.push_back(i);
    }
    if (podCarryIndices.empty())
      continue;
    if (podCarryIndices.size() != 1)
      continue;
    unsigned podIdx = podCarryIndices[0];

    // Discover fields from pod.read/pod.write in the while body.
    Block &bodyBlock = whileOp.getAfter().front();
    Value podBlockArg = bodyBlock.getArgument(podIdx);

    // Collect field names → types from pod ops in body AND post-while.
    llvm::StringMap<Type> fieldTypes;
    SmallVector<StringRef> fieldOrder;

    // From body.
    auto walkBody = [&](function_ref<void(Operation *)> callback) {
      bodyBlock.walk(callback);
    };
    discoverPodFields(walkBody, podBlockArg, fieldOrder, fieldTypes);

    // From post-while users.
    Value whilePodResult = whileOp.getResult(podIdx);
    discoverPodFieldsFromUsers(whilePodResult, fieldOrder, fieldTypes);

    if (fieldOrder.empty())
      continue;

    // Use-shape gate: `expandWhileRegionArgs` only rewires pod.read /
    // pod.write uses of the pod block arg + the immediate body
    // terminator at the expand position; `replacePostWhilePodUsers`
    // handles pod.read / pod.write / struct.writem; the chained-while
    // branch below handles scf.while users. Any other use of the pod
    // block arg or the pod result (e.g. a nested `scf.yield` carrying
    // the result back to an enclosing while, a `function.call`
    // operand) survives the rewires and trips `use_empty()` at the
    // subsequent `eraseArgument` / `op.erase()`. Bail here and let
    // the outer fixed point retry after sibling phases simplify the
    // unhandled users.
    auto checkBlockArgUses = [](Block &body, unsigned argIdx) -> bool {
      Value podArg = body.getArgument(argIdx);
      Operation *term = body.getTerminator();
      unsigned offset = isa<scf::ConditionOp>(term) ? 1 : 0;
      unsigned expandIdx = argIdx + offset;
      for (OpOperand &use : podArg.getUses()) {
        Operation *user = use.getOwner();
        StringRef n = user->getName().getStringRef();
        if (n == "pod.read" || n == "pod.write")
          continue;
        if (user == term && use.getOperandNumber() == expandIdx)
          continue;
        return false;
      }
      return true;
    };
    // `allowChained` accepts `scf.while` users because the chained-
    // while branch below handles them by building a per-field
    // replacement and erasing the chained while. For chained whiles
    // themselves we set `allowChained = false` because the branch
    // doesn't recurse: a chained-of-chained `scf.while` would survive
    // and leave a dangling reference at the chained `erase` step.
    auto checkPostWhileUses = [](Value result, bool allowChained) -> bool {
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        StringRef n = user->getName().getStringRef();
        if (n == "pod.read" || n == "pod.write" || n == "struct.writem")
          continue;
        if (allowChained && isa<scf::WhileOp>(user))
          continue;
        return false;
      }
      return true;
    };

    if (!checkBlockArgUses(whileOp.getBefore().front(), podIdx) ||
        !checkBlockArgUses(whileOp.getAfter().front(), podIdx) ||
        !checkPostWhileUses(whilePodResult, /*allowChained=*/true))
      continue;

    // The chained-while branch below transitively runs the same
    // expansion on each chained `scf.while` that consumes
    // `whilePodResult` as init. Apply the same gate to those whiles
    // up front; partial expansion of one chained while followed by a
    // crash on another leaves the IR in a wedged state.
    bool chainedOk = true;
    for (OpOperand &use : whilePodResult.getUses()) {
      auto chained = dyn_cast<scf::WhileOp>(use.getOwner());
      if (!chained)
        continue;
      unsigned chainedIdx = use.getOperandNumber();
      if (!checkBlockArgUses(chained.getBefore().front(), chainedIdx) ||
          !checkBlockArgUses(chained.getAfter().front(), chainedIdx) ||
          !checkPostWhileUses(chained.getResult(chainedIdx),
                              /*allowChained=*/false)) {
        chainedOk = false;
        break;
      }
    }
    if (!chainedOk)
      continue;

    // Build new init values and types.
    OpBuilder builder(whileOp);
    Location loc = whileOp.getLoc();

    SmallVector<Value> newInits;
    SmallVector<Type> newTypes;
    for (unsigned i = 0; i < whileOp.getNumOperands(); ++i) {
      if (i == podIdx) {
        Value podInit = whileOp.getOperand(i);
        for (StringRef fn : fieldOrder) {
          Type ft = fieldTypes[fn];
          Value initVal;
          // Look for pre-while pod.write that initializes this field.
          for (Operation *user : podInit.getUsers()) {
            if (user->getName().getStringRef() == "pod.write" &&
                user->getParentRegion() == whileOp->getParentRegion()) {
              auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
              if (rn && rn.getValue() == fn && user->getNumOperands() >= 2 &&
                  user->isBeforeInBlock(whileOp))
                initVal = user->getOperand(1);
            }
          }
          if (!initVal)
            initVal = createNondet(builder, loc, ft);
          newInits.push_back(initVal);
          newTypes.push_back(ft);
        }
      } else {
        newInits.push_back(whileOp.getOperand(i));
        newTypes.push_back(whileOp.getOperand(i).getType());
      }
    }

    // Create new while and take over the old regions.
    auto newWhile = builder.create<scf::WhileOp>(loc, newTypes, newInits);
    newWhile.getBefore().takeBody(whileOp.getBefore());
    newWhile.getAfter().takeBody(whileOp.getAfter());

    // Expand block args in both regions: insert field args, fix pod ops.
    for (int regionIdx = 0; regionIdx < 2; ++regionIdx) {
      Region &region =
          regionIdx == 0 ? newWhile.getBefore() : newWhile.getAfter();
      expandWhileRegionArgs(region, podIdx, fieldOrder, fieldTypes, loc);
    }

    // Replace uses of old while results.
    replaceNonPodWhileResults(whileOp, newWhile, podIdx, fieldOrder.size());

    // Handle pod result: process users in program order.
    llvm::StringMap<Value> postWhileFieldValues;
    unsigned podBase = podIdx;
    for (size_t f = 0; f < fieldOrder.size(); ++f)
      postWhileFieldValues[fieldOrder[f]] = newWhile.getResult(podBase + f);

    replacePostWhilePodUsers(whileOp.getResult(podIdx), whileOp->getBlock(),
                             fieldOrder, postWhileFieldValues);

    // Handle chained while: if another scf.while uses the old pod result as
    // init, replace the pod init with individual field inits. This creates a
    // new while with expanded operands so the next unpack iteration can
    // process it.
    Value oldPodResult = whileOp.getResult(podIdx);
    for (OpOperand &use : llvm::make_early_inc_range(oldPodResult.getUses())) {
      Operation *user = use.getOwner();
      if (auto nextWhile = dyn_cast<scf::WhileOp>(user)) {
        // Find which operand index is the pod
        unsigned opIdx = use.getOperandNumber();
        // Build new inits with pod expanded to fields
        SmallVector<Value> nextInits;
        SmallVector<Type> nextTypes;
        for (unsigned i = 0; i < nextWhile.getNumOperands(); ++i) {
          if (i == opIdx) {
            for (StringRef fn : fieldOrder)
              nextInits.push_back(postWhileFieldValues[fn]);
            for (StringRef fn : fieldOrder)
              nextTypes.push_back(fieldTypes[fn]);
          } else {
            nextInits.push_back(nextWhile.getOperand(i));
            nextTypes.push_back(nextWhile.getOperand(i).getType());
          }
        }
        // Create replacement while, take body
        OpBuilder nextBuilder(nextWhile);
        auto replacementWhile = nextBuilder.create<scf::WhileOp>(
            nextWhile.getLoc(), nextTypes, nextInits);
        replacementWhile.getBefore().takeBody(nextWhile.getBefore());
        replacementWhile.getAfter().takeBody(nextWhile.getAfter());

        // Expand block args in both regions.
        unsigned nextPodIdx = opIdx;
        for (int ri = 0; ri < 2; ++ri) {
          Region &region = ri == 0 ? replacementWhile.getBefore()
                                   : replacementWhile.getAfter();
          expandWhileRegionArgs(region, nextPodIdx, fieldOrder, fieldTypes,
                                loc);
        }

        // Replace old while results
        replaceNonPodWhileResults(nextWhile, replacementWhile, nextPodIdx,
                                  fieldOrder.size());

        // Handle pod result of next while same way
        llvm::StringMap<Value> nextPostFields;
        unsigned nBase = nextPodIdx;
        for (size_t f = 0; f < fieldOrder.size(); ++f)
          nextPostFields[fieldOrder[f]] = replacementWhile.getResult(nBase + f);

        replacePostWhilePodUsers(nextWhile.getResult(nextPodIdx),
                                 nextWhile->getBlock(), fieldOrder,
                                 nextPostFields);

        nextWhile->dropAllReferences();
        nextWhile->erase();
      }
    }

    whileOp->dropAllReferences();
    whileOp->erase();
    // Process one while per call — the outer fixed-point loop will re-collect
    // for chained while ops (second while uses first while's result).
    return true;
  }

  return changed;
}

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

/// Simplify POD-based sub-component dispatch in a single function.def block.
bool eliminatePodDispatch(Block &block) {
  // Track: pod SSA value → {field_name → latest written SSA value}
  llvm::DenseMap<Value, llvm::StringMap<Value>> trackedPodValues;

  // Phase 1: Scan block, track pod field values and extract function.call
  // from scf.if into the parent block.
  bool changed = extractCallsFromScfIf(block, trackedPodValues);
  // Phase 2: Replace pod.read results with tracked values.
  changed |= replacePodReads(block, trackedPodValues);
  // Phase 3: Erase struct.writem that writes pod/struct-typed values
  // (sub-component bookkeeping, not needed for witness generation).
  changed |= eraseStructWritemForPodValues(block);
  // Phase 4: Iteratively erase dead pod/scf.if/count-tracking ops.
  changed |= eraseDeadPodAndCountOps(block);
  // Phase 5: Replace remaining self-referential pod.read with llzk.nondet.
  changed |= replaceRemainingPodOps(block);

  return changed;
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
