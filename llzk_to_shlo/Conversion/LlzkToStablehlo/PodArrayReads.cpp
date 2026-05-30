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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayReads.h"

#include <functional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

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
    if (isa<llzk::function::CallOp>(op) && op->getNumResults() > 0)
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

} // namespace

bool resolveArrayPodCompReads(Block &block) {
  bool changed = false;

  // Strategy: find the function.call results that provide pod @comp values.
  // 1. Top-level calls dominate directly.
  // 2. Calls inside void scf.if dispatch: extract and hoist before scf.if.
  // Build a type → call result map to match each pod.read @comp by type.
  llvm::DenseMap<Type, Value> callResultByType;
  for (Operation &op : block) {
    if (isa<llzk::function::CallOp>(op) && op.getNumResults() > 0)
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
    if (!isa<llzk::array::ReadArrayOp>(op) || op.getNumResults() == 0)
      continue;
    // Check element type is pod
    if (!isa<llzk::pod::PodType>(op.getResult(0).getType()))
      continue;

    // Check if this array.read result is used by pod.read @comp
    for (OpOperand &use : op.getResult(0).getUses()) {
      Operation *user = use.getOwner();
      if (!isa<llzk::pod::ReadPodOp>(user) || user->getNumResults() == 0)
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
    if (!isa<llzk::pod::ReadPodOp>(op) || op.getNumOperands() == 0 ||
        op.getNumResults() == 0)
      continue;
    auto rn = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!rn)
      continue;
    Value src = op.getOperand(0);
    Operation *def = src.getDefiningOp();
    if (!def || !isa<llzk::array::ReadArrayOp>(def))
      continue;
    if (!isa<llzk::pod::PodType>(src.getType()))
      continue;

    StringRef field = rn.getValue();
    if (field != "count" && field != "comp" && field != "in")
      continue;

    if (field == "in") {
      bool hasCallUse = false;
      for (Operation *user : op.getResult(0).getUsers()) {
        if (isa<llzk::function::CallOp>(user)) {
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

} // namespace mlir::llzk_to_shlo
