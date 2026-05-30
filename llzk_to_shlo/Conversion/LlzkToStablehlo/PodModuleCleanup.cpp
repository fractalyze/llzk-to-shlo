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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodModuleCleanup.h"

#include <functional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
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
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::llzk_to_shlo {

namespace {

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

} // namespace

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
    if (!isa<llzk::NonDetOp>(op))
      return;
    for (Value result : op->getResults()) {
      Type t = result.getType();
      Type stripped = stripEmptyStructParamsFromType(t);
      if (stripped != t)
        result.setType(stripped);
    }
  });
}

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
  auto isCloneable = [](Operation *op) {
    return isa<llzk::array::ExtractArrayOp, llzk::cast::FeltToIndexOp,
               llzk::felt::FeltConstantOp>(op);
  };

  SmallVector<scf::WhileOp> whiles;
  root->walk([&](scf::WhileOp w) { whiles.push_back(w); });

  for (scf::WhileOp whileOp : whiles) {
    Block &body = whileOp.getAfter().front();

    SmallVector<Operation *> calls;
    for (Operation &op : body)
      if (isa<llzk::function::CallOp>(op))
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
        if (!isCloneable(def))
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
        if (!isa<llzk::component::MemberReadOp>(user) ||
            user->getNumResults() == 0 || !user->getResult(0).hasOneUse()) {
          usersOk = false;
          break;
        }
        Operation *write = *user->getResult(0).getUsers().begin();
        if (!isa<llzk::array::InsertArrayOp, llzk::array::WriteArrayOp>(
                write)) {
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
  auto isPodTyped = [](Type t) -> bool { return isa<llzk::pod::PodType>(t); };

  llvm::SetVector<SlotKey> droppableSlots;
  llvm::SetVector<Operation *> deadPodNews;
  module.walk([&](Operation *op) {
    if (isa<llzk::pod::NewPodOp>(op))
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
    if (isa<llzk::pod::NewPodOp>(u))
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

} // namespace mlir::llzk_to_shlo
