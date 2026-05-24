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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodDispatchPhases.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir::llzk_to_shlo {

namespace {

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

/// Module-scope cache of `(struct_name, member_name)` pairs whose fields are
/// read by a `struct.readm` in some function OTHER than the owning struct's
/// `@constrain` (the @constrain side gets stripped by
/// `createEmptyTemplateRemoval` later in the pipeline, so its readers do not
/// represent live downstream consumers). Populated once at the start of
/// `runOnOperation` and consulted by `hasStructWritemInBody` to decide whether
/// a scf-wrapped writem warrants Phase 4 preservation. A static here is safe —
/// MLIR pass-manager runs `SimplifySubComponents::runOnOperation` to completion
/// sequentially per module.
static llvm::DenseSet<std::pair<StringRef, StringRef>> g_externallyLiveMembers;

/// Returns true if `op` has any nested `struct.writem` whose written value
/// is felt-typed (i.e. real witness emission, not sub-component
/// bookkeeping) AND whose target member is `externally live` — meaning a
/// `struct.readm` of that member exists in some function other than the
/// owning struct's `@constrain`. Pod/struct-typed writems are bookkeeping
/// and are stripped at body level by `eraseStructWritemForPodValues`
/// (Phase 3), so they are intentionally NOT preserved here.
///
/// Without this guard, Phase 4 erases an `scf.if` whose body's only
/// side effect is a conditional `struct.writem %self[@F] = %felt_const`
/// — the canonical `LessThanBounded_5::@compute` pattern in pointcompress.
/// The wrapping scf.if has an unused yielded result and no nested external
/// SSA consumers, so `isOpAndNestedResultsExternallyUnused` returns true
/// and the entire scf.if (writem included) is erased. With every writem on
/// `@F` gone, `collectWritemTargets` returns an empty set,
/// `registerStructFieldOffsets` allocates no slot for `@F`, and a
/// downstream `struct.readm %_[@F]` in a parent's `@compute` fails to
/// legalize with `member offset not found for: F`.
///
/// The cross-function liveness gate prevents the inverse over-preservation
/// regression: standalone `lessthan_bounded` has the same writem-in-scf.if
/// shape on `@LessThanBounded_2::@out`, but its only reader is the owning
/// struct's `@constrain` (no parent `@compute` reads it). Preserving the
/// scf.if there allocates a phantom witness slot that breaks the m3
/// correctness gate against circom-native (`witness_compare: literal has 2
/// elements but wtns_indices has 1`).
bool hasStructWritemInBody(Operation &op) {
  return op
      .walk([](Operation *inner) {
        if (inner->getName().getStringRef() != "struct.writem" ||
            inner->getNumOperands() < 2)
          return WalkResult::advance();
        Type valType = inner->getOperand(1).getType();
        // Type-system check (matches the sibling `hasNonPodArrayWriteInBody`
        // idiom and survives any future dialect-namespace renames).
        if (isa<llzk::pod::PodType, llzk::component::StructType>(valType))
          return WalkResult::advance();
        // Cross-function liveness: only preserve if some non-@constrain
        // function reads this member. Otherwise the writem is dead-after-
        // @constrain-erasure and allocating a witness slot for it would be
        // a phantom slot that breaks downstream witness-layout invariants.
        auto sTy = dyn_cast<llzk::component::StructType>(
            inner->getOperand(0).getType());
        auto memberRef = inner->getAttrOfType<FlatSymbolRefAttr>("member_name");
        if (!sTy || !memberRef)
          return WalkResult::advance();
        StringRef structName = sTy.getNameRef().getLeafReference().getValue();
        if (!g_externallyLiveMembers.contains(
                {structName, memberRef.getValue()}))
          return WalkResult::advance();
        return WalkResult::interrupt();
      })
      .wasInterrupted();
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

} // namespace

// ===----------------------------------------------------------------------===
// Phase functions
// ===----------------------------------------------------------------------===

/// Phase 1: Scan block, track pod field values and extract function.call
/// from scf.if into the parent block.
bool extractCallsFromScfIf(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues) {
  bool changed = false;

  // Lazily computed; only needed when a hoist candidate's args must be
  // dominance-checked. Newly inserted CallOps are always BEFORE the scf.if
  // they replace, so they cannot violate dominance for any pre-existing
  // (arg, op) pair we query — caching across scf.ifs in the same block
  // iteration stays valid. Anchored at the enclosing `function.def` so the
  // dominance tree covers Values defined in outer regions (function args,
  // ops before an enclosing scf.while) — `block.getParentOp()` alone is
  // too narrow when `extractCallsFromScfIf` is invoked recursively on an
  // inner scf.while/scf.if body.
  std::optional<DominanceInfo> domCache;
  auto dom = [&]() -> DominanceInfo & {
    if (!domCache) {
      Operation *root = block.getParentOp();
      while (root && root->getName().getStringRef() != "function.def")
        root = root->getParentOp();
      domCache.emplace(root ? root : block.getParentOp());
    }
    return *domCache;
  };

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
        // Positional dominance guard: when a carrier pod has `pod.write
        // %carrier[@<F>] = %v` writes interleaved between sibling scf.if
        // dispatch sites in the same block, the tracker entry for `@F` can
        // point at a `%v` defined LATER than `op` (the current scf.if).
        // Hoisting the call BEFORE `op` would then create an SSA dominance
        // violation. Reject the hoist when any resolved arg fails to
        // properly dominate `op`.
        if (dominatesScfIf && !args.empty() &&
            !llvm::all_of(
                args, [&](Value a) { return dom().properlyDominates(a, &op); }))
          dominatesScfIf = false;
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
          (hasNonPodArrayWriteInBody(op) || hasStructWritemInBody(op)))
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

void populateExternallyLiveMembers(ModuleOp module) {
  g_externallyLiveMembers.clear();
  module->walk([&](Operation *readm) {
    if (readm->getName().getStringRef() != "struct.readm" ||
        readm->getNumOperands() < 1)
      return;
    auto memberRef = readm->getAttrOfType<FlatSymbolRefAttr>("member_name");
    if (!memberRef)
      return;
    Operation *fn = readm->getParentOp();
    while (fn && fn->getName().getStringRef() != "function.def")
      fn = fn->getParentOp();
    if (!fn)
      return;
    auto fnName = fn->getAttrOfType<StringAttr>("sym_name");
    if (!fnName || fnName.getValue() == "constrain")
      return;
    auto sTy =
        dyn_cast<llzk::component::StructType>(readm->getOperand(0).getType());
    if (!sTy)
      return;
    StringRef structName = sTy.getNameRef().getLeafReference().getValue();
    g_externallyLiveMembers.insert({structName, memberRef.getValue()});
  });
}

} // namespace mlir::llzk_to_shlo
