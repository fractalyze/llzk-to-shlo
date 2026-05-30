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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/InputPodElimination.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
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

} // namespace

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
    if (isa<llzk::component::MemberWriteOp>(op) && hasInputsMemberName(op))
      writemsToErase.push_back(op);
    else if (isa<llzk::component::MemberReadOp>(op) && hasInputsMemberName(op))
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
      if (!isa<llzk::pod::NewPodOp>(op) || op->getNumResults() == 0)
        return;
      Value podVal = op->getResult(0);
      for (OpOperand &use : podVal.getUses())
        if (!isa<llzk::pod::WritePodOp>(use.getOwner()))
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
      if (!isa<llzk::pod::ReadPodOp>(user) || user->getNumResults() == 0)
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
      if (!isa<llzk::array::ReadArrayOp>(arrayRead) ||
          arrayRead->getNumResults() == 0)
        continue;
      Value podVal = arrayRead->getResult(0);
      for (OpOperand &podUse : llvm::make_early_inc_range(podVal.getUses())) {
        Operation *podRead = podUse.getOwner();
        if (!isa<llzk::pod::ReadPodOp>(podRead) ||
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

namespace {

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

} // namespace

/// Inline single-field input pods (no `@count`) used as scf.while carry
/// to their inner field type. Must run before `createEmptyTemplateRemoval`
/// so its `applyFullConversion` doesn't see residual `pod.*` ops.
void inlineInputPodCarries(ModuleOp module) {
  SmallVector<Operation *> podNews;
  module->walk([&](Operation *op) {
    if (isa<llzk::pod::NewPodOp>(op) && op->getNumResults() == 1 &&
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
        if (isa<scf::YieldOp, scf::ConditionOp>(user)) {
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
        if (isa<llzk::pod::ReadPodOp>(user) && user->getNumResults() > 0) {
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
        if (isa<llzk::pod::ReadPodOp>(user)) {
          user->getResult(0).replaceAllUsesWith(v);
          toErase.insert(user);
        }
      }
    }

    for (Value v : podValues) {
      for (auto &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (isa<llzk::pod::WritePodOp>(user))
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
    if (!isa<scf::WhileOp>(op))
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

} // namespace mlir::llzk_to_shlo
