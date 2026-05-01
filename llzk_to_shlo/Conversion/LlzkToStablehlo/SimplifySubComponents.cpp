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
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
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

/// Returns true if `op` has any nested `array.write` whose target array's
/// element type is not a `!pod.type<...>`. Such writes are real
/// side-effecting witness emission (the struct-element drain pattern that
/// circom emits for sub-component aggregation fields, e.g. AES
/// `@bits2num_1` / `@xor_2`); 0-iter-arg `scf.for` fill loops produce no
/// SSA results, so without this guard `isAllResultsUnused` would drop the
/// loop and silently erase the writes.
bool hasNonPodArrayWriteInBody(Operation &op) {
  return op
      .walk([](Operation *inner) {
        if (inner->getName().getStringRef() != "array.write" ||
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

      // scf.for whose body writes to a non-pod-element array is a real
      // fill loop, not dispatch bookkeeping — preserve it so the
      // side-effecting array.write reaches the rest of the pipeline.
      if (name == "scf.for" && hasNonPodArrayWriteInBody(op))
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
      Value callResult; // function.call result fed into pod.write[@comp].
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

        writers.push_back(
            {std::move(outerIndices), ancestor, podWrite->getOperand(1)});
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
      llzk::array::ArrayType perFieldArrTy;
      if (auto innerArr = dyn_cast<llzk::array::ArrayType>(entry.second)) {
        auto innerDims = getArrayDimensions(innerArr);
        SmallVector<int64_t> combined(dims.begin(), dims.end());
        combined.append(innerDims.begin(), innerDims.end());
        perFieldArrTy =
            llzk::array::ArrayType::get(innerArr.getElementType(), combined);
      } else {
        perFieldArrTy = llzk::array::ArrayType::get(entry.second, dims);
      }
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
    // The inner struct must expose a single felt-typed member (e.g.
    // `@out` for circomlib's `@XOR_0` and `@Bits2Num_1`). Multi-felt
    // `@out` fields (e.g. `@Num2Bits_2.@out : array<32 x !felt>`) are
    // left for a follow-up since the witness layout itself would need
    // rework.
    struct DrainPlan {
      Value destFelt;       // %destArr_felt = array.new : array<D x !felt>.
      Type innerFeltTy;     // The inner struct's felt member type.
      StringRef innerField; // Field name (e.g. "out") of the felt member.
      Type structArrTy;     // Original array<D x !struct.type<@Sub>>.
    };
    llvm::DenseMap<Value, DrainPlan> drainPlans;
    auto findInnerFeltMember = [&](llzk::component::StructType structTy,
                                   StringRef &outField,
                                   Type &outFeltTy) -> bool {
      // Locate the struct.def for `structTy` by walking the enclosing
      // module. Returns true and populates outField/outFeltTy when the
      // struct.def has exactly one felt-typed `struct.member`.
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
      Operation *singleFeltMember = nullptr;
      foundDef->walk([&](Operation *m) {
        if (m->getName().getStringRef() != "struct.member")
          return;
        auto memTy = m->getAttrOfType<TypeAttr>("type");
        if (!memTy)
          return;
        if (memTy.getValue().getDialect().getNamespace() != "felt")
          return;
        if (singleFeltMember) {
          // More than one felt member; ambiguous.
          singleFeltMember = nullptr;
        } else {
          singleFeltMember = m;
        }
      });
      if (!singleFeltMember)
        return false;
      // Recheck: walk again counting felt members to ensure uniqueness.
      int feltCount = 0;
      foundDef->walk([&](Operation *m) {
        if (m->getName().getStringRef() != "struct.member")
          return;
        auto memTy = m->getAttrOfType<TypeAttr>("type");
        if (memTy && memTy.getValue().getDialect().getNamespace() == "felt")
          ++feltCount;
      });
      if (feltCount != 1)
        return false;
      auto sym = singleFeltMember->getAttrOfType<StringAttr>("sym_name");
      auto memTy = singleFeltMember->getAttrOfType<TypeAttr>("type");
      if (!sym || !memTy)
        return false;
      outField = sym.getValue();
      outFeltTy = memTy.getValue();
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
      StringRef innerField;
      Type innerFeltTy;
      if (!findInnerFeltMember(innerStruct, innerField, innerFeltTy))
        continue;
      // Allocate the parallel felt array right after the dispatch pod
      // array `%arr` so it dominates every writer site. The drain
      // destination is typically declared near the bottom of the
      // @compute body (after every writer); felt writes will go into
      // the parallel array at writer sites and the original destArr
      // will be replaced by an `unrealized_conversion_cast` of the
      // felt array on the consumer side.
      auto destDims = getArrayDimensions(destArrTy);
      auto feltArrTy = llzk::array::ArrayType::get(innerFeltTy, destDims);
      OpBuilder db(cand.arrNew);
      db.setInsertionPointAfter(cand.arrNew);
      Value destFelt = db.create<llzk::array::CreateArrayOp>(
          cand.arrNew->getLoc(), feltArrTy);
      drainPlans[dr.destArr] = {destFelt, innerFeltTy, innerField,
                                dr.destArr.getType()};
    }

    if (perFieldArrays.empty() && drainPlans.empty())
      continue;

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
    // dominance for earlier writes.
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
      // Drain side: extract the inner struct's felt member from
      // %callResult, build a fresh struct with that felt, and write it
      // directly into the destArr. This keeps the destArr's
      // struct-element type intact so the parent `struct.member @F'`
      // declaration and `@constrain` readback chain remain valid; the
      // converted struct flattens to a single felt per cell (the inner
      // sub-component's `@out`), exactly the witness contribution the
      // parent expects.
      for (auto &kv : drainPlans) {
        const DrainPlan &plan = kv.second;
        // Extract the inner struct's felt from the call result, then
        // write it into the parallel felt array at the writer's outer
        // indices. `array.write` to a felt array is recognized by
        // `processBlockForArrayMutations` + `ArrayWritePattern` and
        // lowers cleanly to `stablehlo.dynamic_update_slice` even
        // inside an `scf.while` body.
        Value feltVal = b.create<llzk::component::MemberReadOp>(
            w.insertAfter->getLoc(), plan.innerFeltTy, w.callResult,
            b.getStringAttr(plan.innerField));
        OperationState writeState(w.insertAfter->getLoc(), "array.write");
        SmallVector<Value> writeOperands;
        writeOperands.push_back(plan.destFelt);
        writeOperands.append(w.outerIndices.begin(), w.outerIndices.end());
        writeOperands.push_back(feltVal);
        writeState.addOperands(writeOperands);
        b.create(writeState);
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
      auto destDims =
          getArrayDimensions(cast<llzk::array::ArrayType>(plan.structArrTy));
      auto newMemberArrTy =
          llzk::array::ArrayType::get(plan.innerFeltTy, destDims);

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
      // `struct.readm @out` and any consumer whose operand type now
      // mismatches (function.call to sibling constrain).
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
        for (Operation *rm : readms) {
          rm->getResult(0).setType(newMemberArrTy);
          SmallVector<Operation *> toErase;
          for (OpOperand &use : rm->getResult(0).getUses()) {
            Operation *user = use.getOwner();
            if (user->getName().getStringRef() != "array.read" ||
                user->getNumResults() == 0)
              continue;
            user->getResult(0).setType(plan.innerFeltTy);
            for (OpOperand &arUse : user->getResult(0).getUses()) {
              Operation *arUser = arUse.getOwner();
              StringRef arUserName = arUser->getName().getStringRef();
              if (arUserName == "struct.readm" && arUser->getNumResults() > 0) {
                auto innerMember =
                    arUser->getAttrOfType<FlatSymbolRefAttr>("member_name");
                if (innerMember && innerMember.getValue() == plan.innerField) {
                  arUser->getResult(0).replaceAllUsesWith(user->getResult(0));
                  toErase.push_back(arUser);
                  continue;
                }
              }
              if (arUserName == "function.call") {
                // Operand type now mismatches the callee's signature
                // (the cell is a felt, not a struct). The call sits in
                // @constrain — dead from a witness perspective — so
                // erase it. Subsequent dead consumers DCE in Phase 4.
                toErase.push_back(arUser);
                continue;
              }
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

  // Handle post-while users of the pod-array result: erase struct.writem,
  // nondet anything else.
  Value podArrResult = whileOp.getResult(podArrIdx);
  SmallVector<Operation *> postErase;
  for (OpOperand &use : llvm::make_early_inc_range(podArrResult.getUses())) {
    Operation *user = use.getOwner();
    if (user->getName().getStringRef() == "struct.writem")
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
            Value fieldArr = perFieldArrs[fIt->second];
            OpBuilder b(op);
            Type resultType = op->getResult(0).getType();
            StringRef opName = isa<llzk::array::ArrayType>(resultType)
                                   ? "array.extract"
                                   : "array.read";
            OperationState state(op->getLoc(), opName);
            SmallVector<Value> ops;
            ops.push_back(fieldArr);
            for (Value idx : it->second)
              ops.push_back(idx);
            state.addOperands(ops);
            state.addTypes({resultType});
            Operation *newRead = b.create(state);
            op->getResult(0).replaceAllUsesWith(newRead->getResult(0));
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

    SmallVector<Operation *> toErase;
    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.read") {
          user->getResult(0).replaceAllUsesWith(v);
          toErase.push_back(user);
        }
      }
    }

    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (user->getName().getStringRef() == "pod.write")
          toErase.push_back(user);
      }
    }

    {
      OpBuilder builder(podNew);
      Value init = createNondet(builder, podNew->getLoc(), innerType);
      podNew->getResult(0).replaceAllUsesWith(init);
      toErase.push_back(podNew);
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
                        if (!hasArrayOfPods && !hasPodBlockArg)
                          changed |= eliminatePodDispatch(b);
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
        for (Operation &op : blk) {
          auto nested = dyn_cast<scf::WhileOp>(&op);
          if (!nested || nested == parent)
            continue;
          llvm::DenseSet<unsigned> claimedBaPositions;
          for (unsigned start = 0; start < nested->getNumOperands();) {
            unsigned end = start;
            while (end < nested->getNumOperands()) {
              Value v = nested->getOperand(end);
              Operation *def = v.getDefiningOp();
              if (!def || def->getName().getStringRef() != "llzk.nondet")
                break;
              if (!isFlattenableFelt(v.getType()))
                break;
              ++end;
            }
            if (end == start) {
              ++start;
              continue;
            }
            unsigned runLen = end - start;
            for (unsigned baStart = 0;
                 baStart + runLen <= blk.getNumArguments(); ++baStart) {
              bool overlap = false;
              for (unsigned k = 0; k < runLen; ++k) {
                if (claimedBaPositions.count(baStart + k)) {
                  overlap = true;
                  break;
                }
              }
              if (overlap)
                continue;
              bool typeMatch = true;
              for (unsigned k = 0; k < runLen; ++k) {
                if (blk.getArgument(baStart + k).getType() !=
                    nested->getOperand(start + k).getType()) {
                  typeMatch = false;
                  break;
                }
              }
              if (!typeMatch)
                continue;
              for (unsigned k = 0; k < runLen; ++k) {
                nested->setOperand(start + k, blk.getArgument(baStart + k));
                claimedBaPositions.insert(baStart + k);
              }
              break;
            }
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

      bool dcePodNew = true;
      while (dcePodNew) {
        dcePodNew = false;
        SmallVector<Operation *> deadPodNews;
        module.walk([&](Operation *op) {
          if (op->getName().getStringRef() != "pod.new")
            return;
          if (isAllResultsUnused(*op))
            deadPodNews.push_back(op);
        });
        for (Operation *op : deadPodNews) {
          op->erase();
          dcePodNew = true;
        }
      }
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
