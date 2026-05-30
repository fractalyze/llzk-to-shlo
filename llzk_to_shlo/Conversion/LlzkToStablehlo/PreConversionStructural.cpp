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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PreConversionStructural.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LoweringInfrastructure.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

namespace {

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
      if (isa<llzk::component::StructType>(operand.getType()) &&
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
      if (isPromotableCarryType(ty) && isa<llzk::array::ArrayType>(ty))
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
} // namespace

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

/// Rewire each `struct.readm %V[@out]` whose defining `function.call` is
/// hoisted out of the readm's region (call lives in a strict ancestor
/// region) to consume a freshly-emitted call on the same carrier row at
/// the readm site.
///
/// The structural problem: a single-arg `function.call` whose operand is
/// `array.extract(%carrier, %const_idx)` is hoisted above a loop body that
/// per-iter writes additional columns into `%carrier[%const_idx]` and
/// then commits the (now stale) hoisted call result. At the last iter
/// the row is fully populated, but the hoisted call value still reflects
/// the iter-start row — so the commit propagates wrong data and the
/// canonical "call on the fully-populated row" semantics are lost.
///
/// The rewrite re-emits the `(array.extract + function.call)` chain at
/// each `struct.readm @out` consumer site, so the call sees the current
/// (post-write) row. Frontends that don't hoist — where the call is
/// already inline at the consumer site or in a symbol-getter function
/// body — produce IR the filter below rejects, leaving them untouched.
///
/// Filter: only rewires when `function.call`'s parent region is an
/// ANCESTOR of the `struct.readm`'s parent region (i.e., the call is
/// hoisted out). Same-region cases — call defined in the readm's
/// immediate scope, or in a symbol-getter function body where the
/// call's operand is a function arg — are excluded by construction.
void replaceHoistedReadmWithFreshCall(ModuleOp module) {
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

    // The hoisted call+extract become dead when this was their last
    // consumer (the common case). Erase eagerly so the orphans don't
    // trickle through every post-pass walk.
    if (origCall->use_empty()) {
      origCall->erase();
      if (origExtract->use_empty())
        origExtract->erase();
    }
  }
}

/// Clone a `(struct.readm + array.insert)` zero-index carrier-writer from the
/// ELSE region of an OUTERMOST `bool.cmp eq(%felt_counter, 0)` `scf.if` into
/// its THEN region.
///
/// SSC's `materializeStructOfPodsCompField` emits
/// `array.insert %carrier[%cK] = struct.readm(%hoisted_call[@out])` for each
/// dispatched call. Its writer-skip drops calls nested inside `scf.if`
/// ancestors that lack a wrapping `scf.execute_region` (see
/// `findDispatchCountGuardHoistAncestor`). When the K=0 trigger sits inside
/// the THEN branch of an outermost `scf.if(eq %felt_counter, 0)` with no
/// `scf.execute_region` wrapping, the materializer drops the THEN writer.
/// The ELSE branch is typically wrapped in `scf.execute_region`, so the
/// materializer still emits a K=0 writer there — even though the ELSE branch
/// is structurally dead at outer iter 0 (the predicate guarantees `counter
/// != 0` there). Net effect at outer iter 0: OUTERMOST takes THEN ->
/// `%carrier[0]` is never written -> downstream reads return dense<0>
/// instead of the intended call result -> the input is erased.
///
/// This pre-pass restores the missing OUTERMOST THEN writer by cloning the
/// dead ELSE-region `struct.readm + array.insert` pair before THEN's yield.
/// The cloned `struct.readm`'s operand is a hoisted call value at the
/// enclosing inner-while body's top, which dominates both THEN and ELSE — so
/// the cloned op references it directly. The carrier is identified by the
/// `mSoPCF.carrier-for` attribute SSC tags on the function-level `array.new`.
void cloneZeroIdxCarrierWriteIntoOutermostFeltZeroThen(ModuleOp module) {
  // OUTERMOST predicate: `bool.cmp eq(%felt_counter, %felt_const_0)`.
  // Nested K=0 `scf.if`s the ancestor walk crosses use index-typed
  // `arith.cmpi eq` predicates, so the felt-typed `bool.cmp` discriminates
  // the outer scf.if from any inner K=0 scf.if we might walk past.
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
    // (typically `@out`) per dispatched class.
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
      // ELSE-region `struct.readm` directly would reuse the hoisted call
      // result, which is computed at the enclosing inner-while body's top —
      // BEFORE THEN's input-load `array.insert %input_row[%c0]`. That value
      // reflects the iter-start row 0, missing the last column (only
      // populated by the inner iter that runs the input write at column
      // `n-1`). Building a fresh extract+call+readm here lets the call see
      // the row 0 produced by THEN's input writes; at the inner iter where
      // column `n-1` is loaded, the carrier ends up with the full-state
      // call output. Subsequent SSA-fication threads `array.extract` to the
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

/// Inject `array.write(%row, %c0, %scalar)` between the `array.extract`
/// that produces `%row` and the consuming `function.call(%row)`, so col 0
/// of the call's input row is the most-recent in-block scratchpad scalar
/// instead of the row's iter-start init value.
///
/// The structural problem: a `function.call` operand is `array.extract`
/// of result index 1 of a 3-result `scf.while`. The while body per-iter
/// writes cols 1..n-1 of the per-round row carrier but never col 0;
/// col 0 inherits the outer-body init value (typically `dense<0>`). The
/// canonical col 0 value is computed by another call in the enclosing
/// block and written into a scratchpad just before the consumer call,
/// but never copied into the per-iter row carrier the while is threading.
///
/// The rewrite finds that most-recent scratchpad write —
/// `array.read(array.write(_, _, struct.readm(function.call X)[@out]))`
/// — in the same block, walking backward from the consumer call, and
/// emits `array.write(%row, %c0, %scalar)` between the row's
/// `array.extract` and the consumer call.
///
/// Pattern guard: only matches consumer calls whose operand row is an
/// `array.extract` from result index 1 of a 3-result `scf.while`.
/// Frontends that lower the same logical pattern through a different
/// loop structure — e.g. a separate input-load + main + partial-round
/// while triplet whose consumer's row is a function block argument —
/// fail the guard and are not rewritten.
void fillCallRowCol0FromInBlockScalar(ModuleOp module) {
  SmallVector<Operation *> targets;
  module.walk([&](Operation *consumerCall) {
    if (consumerCall->getName().getStringRef() != "function.call")
      return;
    if (consumerCall->getNumOperands() != 1)
      return;

    Operation *extractOp = consumerCall->getOperand(0).getDefiningOp();
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

    targets.push_back(consumerCall);
  });

  for (Operation *consumerCall : targets) {
    Operation *extractOp = consumerCall->getOperand(0).getDefiningOp();

    // Walk backward in the same block for the most recent in-block scalar:
    //   %scalar = array.read(%scratchpad_after_write, %pos)
    //   where %scratchpad_after_write = array.write(_, _, %readm)
    //         %readm = struct.readm(%call)[@out]
    //         %call = function.call ...
    Operation *scalarRead = nullptr;
    for (Operation *cur = consumerCall->getPrevNode(); cur;
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
      auto readmMember =
          readmOp->getAttrOfType<FlatSymbolRefAttr>("member_name");
      if (!readmMember || readmMember.getValue() != "out")
        continue;
      Operation *scalarCall = readmOp->getOperand(0).getDefiningOp();
      if (!scalarCall ||
          scalarCall->getName().getStringRef() != "function.call")
        continue;
      scalarRead = cur;
      break;
    }
    if (!scalarRead)
      continue;

    OpBuilder b(consumerCall);
    Location loc = consumerCall->getLoc();

    Value c0 = b.create<arith::ConstantOp>(loc, b.getIndexAttr(0)).getResult();

    OperationState writeState(loc, "array.write");
    writeState.addOperands(
        {extractOp->getResult(0), c0, scalarRead->getResult(0)});
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

} // namespace mlir::llzk_to_shlo
