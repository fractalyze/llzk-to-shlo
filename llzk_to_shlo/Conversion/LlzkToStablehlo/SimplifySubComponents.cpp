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
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_SIMPLIFYSUBCOMPONENTS
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h.inc"

namespace {

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
          // Find the original function.call inside scf.if to clone
          // its properties (operandSegmentSizes, numDimsPerMap, etc.)
          Operation *origCall = nullptr;
          op.walk([&](Operation *n) {
            if (n->getName().getStringRef() == "function.call")
              origCall = n;
          });

          // Clone the original call op, replacing operands and callee
          OpBuilder builder(&op);
          OperationState state(op.getLoc(), origCall->getName());
          state.addOperands(args);
          state.addTypes(resultTypes);
          // Copy all attributes, updating callee
          for (auto &attr : origCall->getAttrs()) {
            if (attr.getName() == "callee")
              state.addAttribute("callee", calleeRef);
            else
              state.addAttribute(attr.getName(), attr.getValue());
          }
          // Copy properties (includes operandSegmentSizes)
          if (origCall->getPropertiesStorage())
            state.propertiesAttr = origCall->getPropertiesAsAttribute();
          Operation *newCall = builder.create(state);

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

      // Check all results are unused
      bool allUnused = true;
      for (auto result : op.getResults()) {
        if (!result.use_empty()) {
          allUnused = false;
          break;
        }
      }
      if (allUnused) {
        // Drop references in nested regions before erasing
        op.dropAllReferences();
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
    OperationState state(op.getLoc(), "llzk.nondet");
    state.addTypes({op.getResult(0).getType()});
    Operation *nondet = builder.create(state);
    op.getResult(0).replaceAllUsesWith(nondet->getResult(0));
    toErase.push_back(&op);
    changed = true;
  }
  for (auto *op : toErase)
    op->erase();

  // Erase pod.new whose results are now unused.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.getName().getStringRef() != "pod.new")
      continue;
    bool allUnused = true;
    for (auto result : op.getResults())
      if (!result.use_empty()) {
        allUnused = false;
        break;
      }
    if (allUnused) {
      op.erase();
      changed = true;
    }
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
    auto discoverField = [&](StringRef name, Type type) {
      if (!fieldTypes.count(name)) {
        fieldTypes[name] = type;
        fieldOrder.push_back(name);
      }
    };

    // From body.
    bodyBlock.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "pod.read" &&
          op->getNumOperands() > 0 && op->getOperand(0) == podBlockArg &&
          op->getNumResults() > 0) {
        auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn)
          discoverField(rn.getValue(), op->getResult(0).getType());
      }
      if (op->getName().getStringRef() == "pod.write" &&
          op->getNumOperands() >= 2 && op->getOperand(0) == podBlockArg) {
        auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn)
          discoverField(rn.getValue(), op->getOperand(1).getType());
      }
    });

    // From post-while users.
    Value whilePodResult = whileOp.getResult(podIdx);
    for (OpOperand &use : whilePodResult.getUses()) {
      Operation *user = use.getOwner();
      if (user->getName().getStringRef() == "pod.read") {
        auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn && user->getNumResults() > 0)
          discoverField(rn.getValue(), user->getResult(0).getType());
      }
      if (user->getName().getStringRef() == "pod.write") {
        auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn && user->getNumOperands() >= 2)
          discoverField(rn.getValue(), user->getOperand(1).getType());
      }
    }
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
          if (!initVal) {
            OperationState state(loc, "llzk.nondet");
            state.addTypes({ft});
            initVal = builder.create(state)->getResult(0);
          }
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
      Block &blk = region.front();
      Value oldPodArg = blk.getArgument(podIdx);

      // Insert new field args after podIdx, then erase the pod arg.
      SmallVector<Value> fieldArgs;
      for (size_t f = 0; f < fieldOrder.size(); ++f) {
        auto arg =
            blk.insertArgument(podIdx + 1 + f, fieldTypes[fieldOrder[f]], loc);
        fieldArgs.push_back(arg);
      }

      // Replace pod.read/pod.write referencing the old pod block arg.
      llvm::StringMap<Value> latestFieldValues;
      for (size_t f = 0; f < fieldOrder.size(); ++f)
        latestFieldValues[fieldOrder[f]] = fieldArgs[f];

      SmallVector<Operation *> toErase;
      blk.walk([&](Operation *op) {
        StringRef opName = op->getName().getStringRef();
        if (opName == "pod.read" && op->getNumOperands() > 0 &&
            op->getOperand(0) == oldPodArg && op->getNumResults() > 0) {
          auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (rn && latestFieldValues.count(rn.getValue())) {
            op->getResult(0).replaceAllUsesWith(
                latestFieldValues[rn.getValue()]);
            toErase.push_back(op);
          }
        } else if (opName == "pod.write" && op->getNumOperands() >= 2 &&
                   op->getOperand(0) == oldPodArg) {
          auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (rn && latestFieldValues.count(rn.getValue())) {
            if (op->getParentRegion() == &region)
              latestFieldValues[rn.getValue()] = op->getOperand(1);
            toErase.push_back(op);
          }
        }
      });
      for (auto *op : toErase)
        op->erase();

      // Fix scf.condition / scf.yield: expand pod operand into fields.
      Operation *terminator = blk.getTerminator();
      SmallVector<Value> newTermArgs;
      for (unsigned i = 0; i < terminator->getNumOperands(); ++i) {
        if (terminator->getOperand(i) == oldPodArg) {
          for (StringRef fn : fieldOrder)
            newTermArgs.push_back(latestFieldValues[fn]);
        } else {
          newTermArgs.push_back(terminator->getOperand(i));
        }
      }
      terminator->setOperands(newTermArgs);

      // Erase the old pod block arg (all uses should be gone).
      blk.eraseArgument(podIdx);
    }

    // Replace uses of old while results.
    unsigned podBase = podIdx;
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      if (i == podIdx)
        continue;
      unsigned ni = i < podIdx ? i : i + fieldOrder.size() - 1;
      whileOp.getResult(i).replaceAllUsesWith(newWhile.getResult(ni));
    }

    // Handle pod result: process users in program order.
    llvm::StringMap<Value> postWhileFieldValues;
    for (size_t f = 0; f < fieldOrder.size(); ++f)
      postWhileFieldValues[fieldOrder[f]] = newWhile.getResult(podBase + f);

    // Process top-level users first (in block order), then nested users.
    // Separate into top-level (same block as while) and nested.
    SmallVector<Operation *> topLevelUsers, nestedUsers;
    for (OpOperand &use : whileOp.getResult(podIdx).getUses()) {
      Operation *user = use.getOwner();
      if (user->getBlock() == whileOp->getBlock())
        topLevelUsers.push_back(user);
      else
        nestedUsers.push_back(user);
    }
    llvm::sort(topLevelUsers, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    // Process top-level first: pod.write tracks values, pod.read replaces.
    SmallVector<Operation *> toErase;
    for (Operation *user : topLevelUsers) {
      StringRef userName = user->getName().getStringRef();
      if (userName == "pod.write") {
        auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn && user->getNumOperands() >= 2)
          postWhileFieldValues[rn.getValue()] = user->getOperand(1);
        toErase.push_back(user);
      } else if (userName == "pod.read") {
        auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn && postWhileFieldValues.count(rn.getValue()))
          user->getResult(0).replaceAllUsesWith(
              postWhileFieldValues[rn.getValue()]);
        toErase.push_back(user);
      } else if (userName == "struct.writem") {
        toErase.push_back(user);
      }
    }
    // Process nested users (inside scf.if etc).
    for (Operation *user : nestedUsers) {
      StringRef userName = user->getName().getStringRef();
      if (userName == "pod.read") {
        auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn && postWhileFieldValues.count(rn.getValue()))
          user->getResult(0).replaceAllUsesWith(
              postWhileFieldValues[rn.getValue()]);
        toErase.push_back(user);
      } else if (userName == "pod.write") {
        toErase.push_back(user);
      }
    }
    for (auto *op : toErase)
      op->erase();

    whileOp->dropAllReferences();
    whileOp->erase();
    changed = true;
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
              for (Operation &op : block)
                if (op.getName().getDialectNamespace() == "pod") {
                  hasPod = true;
                  break;
                }
              if (hasPod) {
                changed |= unpackPodWhileCarry(block);
                changed |= eliminatePodDispatch(block);
                // Also process while body blocks for nested pod dispatch.
                for (Operation &op : block) {
                  if (op.getName().getStringRef() != "scf.while")
                    continue;
                  for (Region &r : op.getRegions())
                    for (Block &b : r)
                      changed |= eliminatePodDispatch(b);
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
