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
