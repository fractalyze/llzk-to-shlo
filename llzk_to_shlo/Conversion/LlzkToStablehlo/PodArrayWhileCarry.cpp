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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayWhileCarry.h"

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir::llzk_to_shlo {

namespace {

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
    if (isa<llzk::pod::ReadPodOp>(op) && op->getNumOperands() > 0 &&
        op->getOperand(0) == podValue && op->getNumResults() > 0) {
      auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn)
        discover(rn.getValue(), op->getResult(0).getType());
    }
    if (isa<llzk::pod::WritePodOp>(op) && op->getNumOperands() >= 2 &&
        op->getOperand(0) == podValue) {
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
    if (isa<llzk::pod::ReadPodOp>(op) && op->getNumOperands() > 0 &&
        op->getOperand(0) == podValue && op->getNumResults() > 0) {
      auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue())) {
        op->getResult(0).replaceAllUsesWith(fieldValues[rn.getValue()]);
        toErase.push_back(op);
      }
    } else if (isa<llzk::pod::WritePodOp>(op) && op->getNumOperands() >= 2 &&
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
    if (isa<llzk::pod::WritePodOp>(user)) {
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && user->getNumOperands() >= 2)
        fieldValues[rn.getValue()] = user->getOperand(1);
      toErase.push_back(user);
    } else if (isa<llzk::pod::ReadPodOp>(user)) {
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue()))
        user->getResult(0).replaceAllUsesWith(fieldValues[rn.getValue()]);
      toErase.push_back(user);
    } else if (isa<llzk::component::MemberWriteOp>(user)) {
      toErase.push_back(user);
    }
  }
  for (Operation *user : nestedUsers) {
    if (isa<llzk::pod::ReadPodOp>(user)) {
      auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (rn && fieldValues.count(rn.getValue()))
        user->getResult(0).replaceAllUsesWith(fieldValues[rn.getValue()]);
      toErase.push_back(user);
    } else if (isa<llzk::pod::WritePodOp>(user)) {
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
    if (isa<llzk::component::MemberWriteOp>(user)) {
      postErase.push_back(user);
      continue;
    }
    if (!isa<llzk::array::ReadArrayOp>(user) || user->getNumOperands() < 2 ||
        user->getNumResults() == 0)
      continue;
    Value podCell = user->getResult(0);
    SmallVector<Value> readIndices = arrayAccessIndices(user);
    for (OpOperand &subUse : llvm::make_early_inc_range(podCell.getUses())) {
      Operation *subUser = subUse.getOwner();
      if (!isa<llzk::pod::ReadPodOp>(subUser) || subUser->getNumResults() == 0)
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
        if (!def || !isa<llzk::NonDetOp>(def)) {
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
    // array.read on oldArrArg → track all indices, erase.
    if (isa<llzk::array::ReadArrayOp>(op) && op->getNumOperands() > 1 &&
        op->getOperand(0) == oldArrArg && op->getNumResults() > 0) {
      localPodToIndices[op->getResult(0)] = arrayAccessIndices(op);
      toErase.push_back(op);
      return;
    }

    // pod.write on tracked pod → array.write/insert on per-field array.
    if (isa<llzk::pod::WritePodOp>(op) && op->getNumOperands() >= 2) {
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
    if (isa<llzk::pod::ReadPodOp>(op) && op->getNumOperands() > 0 &&
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
    if (isa<llzk::array::WriteArrayOp>(op) && op->getNumOperands() >= 2 &&
        op->getOperand(0) == oldArrArg) {
      Value written = op->getOperands().back();
      if (localPodToIndices.count(written))
        toErase.push_back(op);
    }

    // scf.if returning oldArrArg type → forward result to oldArrArg.
    if (isa<scf::IfOp>(op) && op->getNumResults() > 0) {
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

} // namespace

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
        if (isa<llzk::array::ReadArrayOp>(op) && op->getNumOperands() > 1 &&
            op->getNumResults() > 0) {
          if (valueTracesToPodArr(op->getOperand(0), podArrBlockArg))
            trackedPods.insert(op->getResult(0));
        }
        auto discover = [&](StringRef fn, Type ty) {
          if (fieldTypes.count(fn) || !isFlattenableFelt(ty))
            return;
          fieldTypes[fn] = ty;
          fieldOrder.push_back(fn);
        };
        if (isa<llzk::pod::ReadPodOp>(op) && op->getNumOperands() > 0 &&
            op->getNumResults() > 0 && trackedPods.count(op->getOperand(0))) {
          if (auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name"))
            discover(rn.getValue(), op->getResult(0).getType());
        }
        if (isa<llzk::pod::WritePodOp>(op) && op->getNumOperands() >= 2 &&
            trackedPods.count(op->getOperand(0))) {
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
        if (isa<llzk::pod::ReadPodOp, llzk::pod::WritePodOp>(user))
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
        if (isa<llzk::pod::ReadPodOp, llzk::pod::WritePodOp,
                llzk::component::MemberWriteOp>(user))
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
            if (isa<llzk::pod::WritePodOp>(user) &&
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

} // namespace mlir::llzk_to_shlo
