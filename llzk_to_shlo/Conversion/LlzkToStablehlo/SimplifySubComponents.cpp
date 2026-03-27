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

#include "llvm/ADT/StringMap.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

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

/// Create an llzk.nondet operation producing an uninitialized value.
Value createNondet(OpBuilder &builder, Location loc, Type type) {
  OperationState state(loc, "llzk.nondet");
  state.addTypes({type});
  return builder.create(state)->getResult(0);
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

/// Expand terminator operand: replace `oldArg` with per-field values from
/// `fieldValues` in `fieldOrder`.
void expandTerminatorArg(Block &blk, Value oldArg,
                         ArrayRef<StringRef> fieldOrder,
                         llvm::StringMap<Value> &fieldValues) {
  Operation *term = blk.getTerminator();
  SmallVector<Value> newArgs;
  for (unsigned i = 0; i < term->getNumOperands(); ++i) {
    if (term->getOperand(i) == oldArg) {
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
  expandTerminatorArg(blk, oldPodArg, fieldOrder, fieldValues);
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

        if (!args.empty() &&
            (hasDirectArgs || args.size() == inputPodFields.size())) {
          // Create function.call using LLZK's CallOp builder API.
          OpBuilder builder(&op);
          Operation *newCall = builder.create<llzk::function::CallOp>(
              op.getLoc(), resultTypes, calleeRef, args);

          // Track: pod[@comp] = newCall result
          if (newCall->getNumResults() > 0 && compPod)
            trackedPodValues[compPod]["comp"] = newCall->getResult(0);

          changed = true;
        }
      }
    }
  }

  return changed;
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
    // Skip if the tracked value IS this pod.read's result (self-reference).
    // Erasing would delete the definition that other ops depend on.
    if (fit->second == op->getResult(0))
      return;
    op->getResult(0).replaceAllUsesWith(fit->second);
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

      if (isAllResultsUnused(op)) {
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

/// Phase -1: Flatten array-of-pods (all-felt fields) carried through scf.while
/// into per-field felt arrays.
///   scf.while (%i, %pod_arr) : (felt, !array.type<2 x !pod.type<[@x: felt, @k:
///   felt]>>) → scf.while (%i, %arr_x, %arr_k) : (felt, !array.type<2 x felt>,
///   !array.type<2 x felt>)
/// Rewrites array.read+pod.read/pod.write+array.write → direct
/// array.read/write.
bool flattenPodArrayWhileCarry(Block &block) {
  // Find scf.while ops that carry array-of-pods.
  SmallVector<scf::WhileOp> whileOps;
  for (Operation &op : block) {
    if (auto w = dyn_cast<scf::WhileOp>(&op))
      whileOps.push_back(w);
  }

  for (scf::WhileOp whileOp : whileOps) {
    // Find array-of-pods carry position.
    int podArrIdx = -1;
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      Type ty = whileOp.getResult(i).getType();
      if (auto arrTy = dyn_cast<llzk::array::ArrayType>(ty))
        if (arrTy.getElementType().getDialect().getNamespace() == "pod")
          podArrIdx = i;
    }
    if (podArrIdx < 0)
      continue;

    // Discover pod fields from pod.read/pod.write in the while body.
    Block &bodyBlock = whileOp.getAfter().front();
    Value podArrBlockArg = bodyBlock.getArgument(podArrIdx);

    llvm::StringMap<Type> fieldTypes;
    SmallVector<StringRef> fieldOrder;

    // Map: pod SSA value (from array.read) → index used to read it.
    llvm::DenseMap<Value, Value> podToIndex;

    // Walk body to discover fields. Need custom walk because we also track
    // array.read → pod+index mapping.
    bodyBlock.walk([&](Operation *op) {
      // Track array.read from pod array → pod value + index
      if (op->getName().getStringRef() == "array.read" &&
          op->getNumOperands() > 1 && op->getNumResults() > 0 &&
          op->getOperand(0) == podArrBlockArg) {
        podToIndex[op->getResult(0)] = op->getOperand(1);
      }
      // Discover fields from pod.read/pod.write on tracked pod values.
      if (op->getName().getStringRef() == "pod.read" &&
          op->getNumOperands() > 0 && op->getNumResults() > 0) {
        if (podToIndex.count(op->getOperand(0))) {
          auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (rn && !fieldTypes.count(rn.getValue()) &&
              op->getResult(0).getType().getDialect().getNamespace() ==
                  "felt") {
            fieldTypes[rn.getValue()] = op->getResult(0).getType();
            fieldOrder.push_back(rn.getValue());
          }
        }
      }
      if (op->getName().getStringRef() == "pod.write" &&
          op->getNumOperands() >= 2) {
        if (podToIndex.count(op->getOperand(0))) {
          auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (rn && !fieldTypes.count(rn.getValue()) &&
              op->getOperand(1).getType().getDialect().getNamespace() ==
                  "felt") {
            fieldTypes[rn.getValue()] = op->getOperand(1).getType();
            fieldOrder.push_back(rn.getValue());
          }
        }
      }
    });

    if (fieldOrder.empty())
      continue;

    // Get array dimension.
    auto arrType =
        cast<llzk::array::ArrayType>(whileOp.getResult(podArrIdx).getType());
    auto dims = getArrayDimensions(arrType);
    if (dims.empty() || dims[0] <= 0)
      continue;
    int64_t arrSize = dims[0];

    // Create per-field felt arrays.
    OpBuilder builder(whileOp);
    Location loc = whileOp.getLoc();
    Type feltType = fieldTypes[fieldOrder[0]];
    auto fieldArrType = llzk::array::ArrayType::get(feltType, {arrSize});

    llvm::StringMap<Value> fieldArrayInits;
    for (StringRef fn : fieldOrder)
      fieldArrayInits[fn] = createNondet(builder, loc, fieldArrType);

    // Build new while with expanded carries.
    SmallVector<Value> newInits;
    SmallVector<Type> newTypes;
    for (unsigned i = 0; i < whileOp.getNumOperands(); ++i) {
      if (i == (unsigned)podArrIdx) {
        for (StringRef fn : fieldOrder) {
          newInits.push_back(fieldArrayInits[fn]);
          newTypes.push_back(fieldArrType);
        }
      } else {
        newInits.push_back(whileOp.getOperand(i));
        newTypes.push_back(whileOp.getOperand(i).getType());
      }
    }

    auto newWhile = builder.create<scf::WhileOp>(loc, newTypes, newInits);
    newWhile.getBefore().takeBody(whileOp.getBefore());
    newWhile.getAfter().takeBody(whileOp.getAfter());

    // Expand block args in both regions and rewrite pod ops.
    for (int ri = 0; ri < 2; ++ri) {
      Region &region = ri == 0 ? newWhile.getBefore() : newWhile.getAfter();
      Block &blk = region.front();
      Value oldArrArg = blk.getArgument(podArrIdx);

      // Insert per-field array args after podArrIdx.
      llvm::StringMap<Value> fieldBlockArgs;
      for (size_t f = 0; f < fieldOrder.size(); ++f) {
        auto arg = blk.insertArgument(podArrIdx + 1 + f, fieldArrType, loc);
        fieldBlockArgs[fieldOrder[f]] = arg;
      }

      // Track latest per-field arrays for yield.
      llvm::StringMap<Value> latestFieldArrs;
      for (StringRef fn : fieldOrder)
        latestFieldArrs[fn] = fieldBlockArgs[fn];

      // Rebuild podToIndex for this block (array.reads from oldArrArg).
      llvm::DenseMap<Value, Value> localPodToIndex;

      // Rewrite ops: walk and collect, then erase.
      SmallVector<Operation *> toErase;
      blk.walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();

        // array.read from pod array → track pod+index
        if (name == "array.read" && op->getNumOperands() > 1 &&
            op->getOperand(0) == oldArrArg && op->getNumResults() > 0) {
          localPodToIndex[op->getResult(0)] = op->getOperand(1);
          toErase.push_back(op);
          return;
        }

        // pod.write on tracked pod → array.write to per-field array
        if (name == "pod.write" && op->getNumOperands() >= 2) {
          auto it = localPodToIndex.find(op->getOperand(0));
          if (it != localPodToIndex.end()) {
            auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
            if (rn && latestFieldArrs.count(rn.getValue())) {
              Value fieldArr = latestFieldArrs[rn.getValue()];
              Value idx = it->second;
              OpBuilder b(op);
              OperationState state(op->getLoc(), "array.write");
              state.addOperands({fieldArr, idx, op->getOperand(1)});
              b.create(state);
              toErase.push_back(op);
              return;
            }
          }
        }

        // pod.read on tracked pod → array.read from per-field array
        if (name == "pod.read" && op->getNumOperands() > 0 &&
            op->getNumResults() > 0) {
          auto it = localPodToIndex.find(op->getOperand(0));
          if (it != localPodToIndex.end()) {
            auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
            if (rn && latestFieldArrs.count(rn.getValue())) {
              Value fieldArr = latestFieldArrs[rn.getValue()];
              Value idx = it->second;
              OpBuilder b(op);
              OperationState state(op->getLoc(), "array.read");
              state.addOperands({fieldArr, idx});
              state.addTypes({op->getResult(0).getType()});
              Operation *newRead = b.create(state);
              op->getResult(0).replaceAllUsesWith(newRead->getResult(0));
              toErase.push_back(op);
              return;
            }
          }
        }

        // array.write of pod back to pod array → erase
        if (name == "array.write" && op->getNumOperands() >= 2 &&
            op->getOperand(0) == oldArrArg) {
          // Check if the value being written is a tracked pod
          Value written =
              op->getNumOperands() > 2 ? op->getOperand(2) : op->getOperand(1);
          if (localPodToIndex.count(written))
            toErase.push_back(op);
        }
      });

      // Erase in reverse order (nested ops first).
      for (auto *op : llvm::reverse(toErase)) {
        if (op->use_empty() || op->getNumResults() == 0)
          op->erase();
      }

      expandTerminatorArg(blk, oldArrArg, fieldOrder, latestFieldArrs);
      blk.eraseArgument(podArrIdx);
    }

    // Replace non-pod-array results of old while.
    replaceNonPodWhileResults(whileOp, newWhile, podArrIdx, fieldOrder.size());

    // Erase post-while uses of the pod array result (struct.writem etc).
    SmallVector<Operation *> postErase;
    for (OpOperand &use :
         llvm::make_early_inc_range(whileOp.getResult(podArrIdx).getUses())) {
      Operation *user = use.getOwner();
      if (user->getName().getStringRef() == "struct.writem")
        postErase.push_back(user);
    }
    for (auto *op : postErase)
      op->erase();

    whileOp->dropAllReferences();
    whileOp->erase();
    return true; // Process one at a time.
  }

  return false;
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

struct SimplifySubComponents
    : impl::SimplifySubComponentsBase<SimplifySubComponents> {
  using SimplifySubComponentsBase::SimplifySubComponentsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

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
                changed |= flattenPodArrayWhileCarry(block);
                changed |= unpackPodWhileCarry(block);
                changed |= eliminatePodDispatch(block);
                // Process while body blocks for nested pod dispatch.
                // Skip bodies with array-of-pods (count array) — the
                // verifier crashes on CallOp created in that context.
                for (Operation &op : block) {
                  if (op.getName().getStringRef() != "scf.while")
                    continue;
                  for (Region &r : op.getRegions()) {
                    for (Block &b : r) {
                      bool hasArrayOfPods = false;
                      for (Operation &bop : b)
                        if (bop.getName().getStringRef() == "array.read" &&
                            bop.getNumResults() > 0 &&
                            bop.getResult(0)
                                    .getType()
                                    .getDialect()
                                    .getNamespace() == "pod")
                          hasArrayOfPods = true;
                      if (!hasArrayOfPods)
                        changed |= eliminatePodDispatch(b);
                    }
                  }
                }
              }
            }
          }
        });
      });
    }
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
