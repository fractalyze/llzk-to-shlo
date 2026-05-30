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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructOfPodsConversion.h"

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir::llzk_to_shlo {

namespace {

// ===----------------------------------------------------------------------===
// Struct-of-pods carrier rewrite (pre-flatten)
// ===----------------------------------------------------------------------===

/// Match a `!pod<[@idx_0..@idx_K-1: T]>` shape with sequential idx-names
/// and uniform inner type T. Returns K and T on match; nullopt otherwise.
struct StructOfPodsShape {
  int64_t k;
  Type innerType;
};

std::optional<StructOfPodsShape> matchStructOfPodsShape(Type type) {
  auto podTy = dyn_cast<llzk::pod::PodType>(type);
  if (!podTy)
    return std::nullopt;
  auto records = podTy.getRecords();
  if (records.empty())
    return std::nullopt;
  Type uniformInner;
  for (size_t i = 0; i < records.size(); ++i) {
    StringRef name = records[i].getName().getValue();
    std::string expected = ("idx_" + Twine(i)).str();
    if (name != expected)
      return std::nullopt;
    if (i == 0)
      uniformInner = records[i].getType();
    else if (records[i].getType() != uniformInner)
      return std::nullopt;
  }
  return StructOfPodsShape{static_cast<int64_t>(records.size()), uniformInner};
}

int parseIdxFieldName(StringRef name) {
  if (!name.consume_front("idx_"))
    return -1;
  int64_t idx;
  if (name.getAsInteger(10, idx) || idx < 0)
    return -1;
  return static_cast<int>(idx);
}

Value buildConstIndex(OpBuilder &builder, Location loc, int64_t v) {
  OperationState state(loc, "arith.constant");
  state.addAttribute("value", builder.getIndexAttr(v));
  state.addTypes({builder.getIndexType()});
  return builder.create(state)->getResult(0);
}

/// Rewriter for struct-of-pods carriers: rewrites
/// `!pod<[@idx_0..@idx_K-1: T]>` → `!array<K x T>` for one seed at a time,
/// cascading through SSA users (pod.read[@idx_N] → array.read[%c_N],
/// pod.write[@idx_N] → array.write[%c_N], scf.while iter-arg slot retype,
/// scf.if result slot retype, scf.yield/scf.condition operand retype).
/// Idempotent: after rewrite the carrier's type is `!array<K x T>` and
/// the seed-match predicate finds no further candidates.
class StructOfPodsRewriter {
public:
  // Returns true iff `seed`'s chain was fully rewritten. On false, IR is
  // returned to its original state via the rollback (caller must check the
  // worklist's bail signal).
  bool run(Operation *seed) {
    auto shapeOpt = matchStructOfPodsShape(seed->getResult(0).getType());
    if (!shapeOpt)
      return false;
    shape = *shapeOpt;
    arrType = llzk::array::ArrayType::get(shape.innerType,
                                          ArrayRef<int64_t>{shape.k});

    if (!validateChain(seed))
      return false;

    if (!rewriteSeed(seed))
      return false;
    while (!worklist.empty()) {
      auto pair = worklist.pop_back_val();
      processUsers(pair.first, pair.second);
    }

    // Drop operand references in two passes so cross-references between
    // to-be-erased ops (e.g. the old scf.while's operand pointing at the old
    // pod.new's result; an old pod.read/write whose pod_ref is a now-orphan
    // block arg) don't keep block args alive when we erase below.
    for (Operation *op : toErase)
      op->dropAllReferences();
    for (BlockArgument oldArg : oldArgsToErase) {
      if (!oldArg.use_empty())
        return false;
      oldArg.getOwner()->eraseArgument(oldArg.getArgNumber());
    }
    for (Operation *op : toErase)
      op->erase();
    return true;
  }

private:
  StructOfPodsShape shape;
  llzk::array::ArrayType arrType;
  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<std::pair<Value, Value>> worklist;
  llvm::SmallSetVector<Operation *, 16> toErase;
  SmallVector<BlockArgument> oldArgsToErase;
  // Ops we've already rebuilt (the NEW op produced by a rebuild). Probed by
  // rewriteYieldOperand to avoid re-rebuilding the same
  // scf.if/scf.execute_region when both yields (then/else, or multi-block) flow
  // our value through.
  llvm::DenseSet<Operation *> rebuiltOps;

  // Walk SSA users transitively; return false if any user shape isn't
  // handleable. Pure analysis — no IR mutation.
  bool validateChain(Operation *seed) {
    llvm::SmallSetVector<Value, 16> visited;
    SmallVector<Value> stack;
    auto enqueue = [&](Value v) {
      if (visited.insert(v))
        stack.push_back(v);
    };
    enqueue(seed->getResult(0));

    // Reject chains that enter the same region-bearing op (scf.while /
    // scf.if / scf.execute_region) via two distinct slots. Each of
    // `rewriteWhileOperand` / `rebuildIfResultSlot` / `rebuildExecuteRegion`
    // rebuilds its target ONCE via `takeBody`; a second visit on the same
    // OLD op would re-`takeBody` from an already-emptied region and produce
    // a no-body scf.while/scf.if (verifier failure). Multi-slot single-seed
    // chains aren't a known production pattern today — bail cleanly so the
    // carrier survives this pass and is caught by downstream diagnostics
    // (CLAUDE.md "Pod-array iter-arg survival post-simplify ..." check)
    // rather than silently miscompiling.
    llvm::DenseMap<Operation *, llvm::SmallDenseSet<unsigned, 2>>
        rebuildOpSlots;
    auto recordRebuildSlot = [&](Operation *op, unsigned slot) {
      auto &slots = rebuildOpSlots[op];
      slots.insert(slot);
      return slots.size() <= 1;
    };

    // Validate the seed op itself supports the rewrite.
    if (isa<llzk::pod::NewPodOp>(seed)) {
      auto initAttr = seed->getAttrOfType<ArrayAttr>("initializedRecords");
      unsigned numInits = initAttr ? initAttr.size() : 0;
      if (numInits != 0 && numInits != (unsigned)shape.k)
        return false;
      // No map operands (pod.new with affine-mapped types).
      if (seed->getNumOperands() != numInits)
        return false;
    } else if (!isa<llzk::NonDetOp>(seed)) {
      return false;
    }

    while (!stack.empty()) {
      Value v = stack.pop_back_val();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        unsigned opIdx = use.getOperandNumber();

        if (isa<llzk::pod::ReadPodOp>(user)) {
          auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (!rn)
            return false;
          int idx = parseIdxFieldName(rn.getValue());
          if (idx < 0 || idx >= shape.k)
            return false;
          continue;
        }
        if (isa<llzk::pod::WritePodOp>(user)) {
          if (opIdx != 0)
            return false;
          auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (!rn)
            return false;
          int idx = parseIdxFieldName(rn.getValue());
          if (idx < 0 || idx >= shape.k)
            return false;
          continue;
        }
        if (auto w = dyn_cast<scf::WhileOp>(user)) {
          if (!recordRebuildSlot(w.getOperation(), opIdx))
            return false;
          enqueue(w.getBefore().front().getArgument(opIdx));
          enqueue(w.getAfter().front().getArgument(opIdx));
          enqueue(w.getResult(opIdx));
          continue;
        }
        if (auto y = dyn_cast<scf::YieldOp>(user)) {
          Operation *parent = y->getParentOp();
          if (auto pw = dyn_cast<scf::WhileOp>(parent)) {
            enqueue(pw.getResult(opIdx));
            continue;
          }
          if (auto pIf = dyn_cast<scf::IfOp>(parent)) {
            if (!recordRebuildSlot(pIf.getOperation(), opIdx))
              return false;
            // scf.if rebuild needs both then and else yield operands at the
            // same slot in the rewrite chain. Enqueue the sibling branch's
            // yield operand so the cascade reaches it.
            enqueue(pIf.getResult(opIdx));
            for (Region *region :
                 {&pIf.getThenRegion(), &pIf.getElseRegion()}) {
              if (region->empty())
                continue;
              auto sib =
                  dyn_cast<scf::YieldOp>(region->front().getTerminator());
              if (!sib || sib == y)
                continue;
              if (opIdx >= sib.getNumOperands())
                return false;
              enqueue(sib.getOperand(opIdx));
            }
            continue;
          }
          if (auto pEr = dyn_cast<scf::ExecuteRegionOp>(parent)) {
            if (!recordRebuildSlot(pEr.getOperation(), opIdx))
              return false;
            enqueue(pEr.getResult(opIdx));
            continue;
          }
          return false;
        }
        if (auto c = dyn_cast<scf::ConditionOp>(user)) {
          if (opIdx == 0)
            return false;
          auto pw = dyn_cast<scf::WhileOp>(c->getParentOp());
          if (!pw)
            return false;
          unsigned slot = opIdx - 1;
          enqueue(pw.getAfter().front().getArgument(slot));
          enqueue(pw.getResult(slot));
          continue;
        }
        // struct.writem, function.call, function.return etc. unsupported.
        return false;
      }
    }
    return true;
  }

  bool rewriteSeed(Operation *seed) {
    OpBuilder builder(seed);
    Location loc = seed->getLoc();
    Value newVal;
    if (isa<llzk::pod::NewPodOp>(seed)) {
      auto initAttr = seed->getAttrOfType<ArrayAttr>("initializedRecords");
      unsigned numInits = initAttr ? initAttr.size() : 0;
      SmallVector<Value> elements;
      if (numInits == (unsigned)shape.k) {
        elements.assign(shape.k, Value());
        for (unsigned i = 0; i < numInits; ++i) {
          auto nameAttr = dyn_cast<StringAttr>(initAttr[i]);
          if (!nameAttr)
            return false;
          int idx = parseIdxFieldName(nameAttr.getValue());
          if (idx < 0 || idx >= shape.k)
            return false;
          elements[idx] = seed->getOperand(i);
        }
        for (Value el : elements)
          if (!el)
            return false;
      }
      auto newCreate = builder.create<llzk::array::CreateArrayOp>(
          loc, arrType, ValueRange(elements));
      newVal = newCreate.getResult();
    } else if (isa<llzk::NonDetOp>(seed)) {
      newVal = createNondet(builder, loc, arrType);
    } else {
      return false;
    }
    valueMap[seed->getResult(0)] = newVal;
    worklist.push_back({seed->getResult(0), newVal});
    toErase.insert(seed);
    return true;
  }

  void processUsers(Value oldVal, Value newVal) {
    // Snapshot uses since rewriting will mutate the use-list.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : oldVal.getUses())
      uses.push_back(&use);

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      unsigned opIdx = use->getOperandNumber();

      if (isa<llzk::pod::ReadPodOp>(user)) {
        rewritePodRead(user, newVal);
      } else if (isa<llzk::pod::WritePodOp>(user)) {
        rewritePodWrite(user, newVal);
      } else if (auto w = dyn_cast<scf::WhileOp>(user)) {
        rewriteWhileOperand(w, opIdx, newVal);
      } else if (auto y = dyn_cast<scf::YieldOp>(user)) {
        rewriteYieldOperand(y, opIdx, newVal);
      } else if (auto c = dyn_cast<scf::ConditionOp>(user)) {
        rewriteConditionOperand(c, opIdx, newVal);
      } else {
        llvm_unreachable("validateChain admitted an unsupported user");
      }
    }
  }

  void rewritePodRead(Operation *user, Value newCarrier) {
    auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
    int idx = parseIdxFieldName(rn.getValue());
    OpBuilder builder(user);
    Value cstIdx = buildConstIndex(builder, user->getLoc(), idx);
    auto readOp = builder.create<llzk::array::ReadArrayOp>(
        user->getLoc(), shape.innerType, newCarrier, ValueRange{cstIdx});
    user->getResult(0).replaceAllUsesWith(readOp.getResult());
    toErase.insert(user);
  }

  void rewritePodWrite(Operation *user, Value newCarrier) {
    auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
    int idx = parseIdxFieldName(rn.getValue());
    OpBuilder builder(user);
    Value cstIdx = buildConstIndex(builder, user->getLoc(), idx);
    Value writeVal = user->getOperand(1);
    builder.create<llzk::array::WriteArrayOp>(user->getLoc(), newCarrier,
                                              ValueRange{cstIdx}, writeVal);
    toErase.insert(user);
  }

  void rewriteWhileOperand(scf::WhileOp w, unsigned slot, Value newOperand) {
    if (rebuiltOps.count(w.getOperation()))
      return;

    OpBuilder builder(w);
    SmallVector<Value> newOperands(w.getOperands().begin(),
                                   w.getOperands().end());
    newOperands[slot] = newOperand;
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(w.getNumResults());
    for (unsigned i = 0; i < w.getNumResults(); ++i)
      newResultTypes.push_back(i == slot ? Type(arrType)
                                         : w.getResult(i).getType());

    auto newWhile =
        builder.create<scf::WhileOp>(w.getLoc(), newResultTypes, newOperands);
    newWhile.getBefore().takeBody(w.getBefore());
    newWhile.getAfter().takeBody(w.getAfter());
    rebuiltOps.insert(newWhile.getOperation());

    for (int regionIdx = 0; regionIdx < 2; ++regionIdx) {
      Block &blk = regionIdx == 0 ? newWhile.getBefore().front()
                                  : newWhile.getAfter().front();
      BlockArgument oldArg = blk.getArgument(slot);
      // Insert NEW arg at slot+1 so old keeps its index until cleanup.
      BlockArgument newArg = blk.insertArgument(slot + 1, arrType, w.getLoc());
      valueMap[oldArg] = newArg;
      worklist.push_back({oldArg, newArg});
      // After cascade migrates oldArg's uses, eraseArgument removes slot.
      // Re-fetch via getArgument(slot) because insertArgument shifts indices.
      oldArgsToErase.push_back(blk.getArgument(slot));
    }

    // Rewire non-pod result slots immediately; mapping + worklist for the
    // converted slot so downstream users cascade.
    for (unsigned i = 0; i < w.getNumResults(); ++i) {
      Value oldRes = w.getResult(i);
      Value newRes = newWhile.getResult(i);
      if (i == slot) {
        valueMap[oldRes] = newRes;
        worklist.push_back({oldRes, newRes});
      } else {
        oldRes.replaceAllUsesWith(newRes);
      }
    }

    toErase.insert(w.getOperation());
  }

  void rewriteYieldOperand(scf::YieldOp y, unsigned opIdx, Value newVal) {
    Operation *parent = y->getParentOp();
    if (auto er = dyn_cast<scf::ExecuteRegionOp>(parent)) {
      if (!rebuiltOps.count(er.getOperation()))
        rebuildExecuteRegion(er, opIdx);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      if (!rebuiltOps.count(ifOp.getOperation()))
        rebuildIfResultSlot(ifOp, opIdx);
    }
    // After rebuildExecuteRegion / rebuildIfResultSlot, the yield op `y`
    // remains alive (it's now in the new parent's region via takeBody),
    // so this setOperand updates the same yield in its new parent.
    y->setOperand(opIdx, newVal);
  }

  void rebuildExecuteRegion(scf::ExecuteRegionOp er, unsigned slot) {
    OpBuilder builder(er);
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(er.getNumResults());
    for (unsigned i = 0; i < er.getNumResults(); ++i)
      newResultTypes.push_back(i == slot ? Type(arrType)
                                         : er.getResult(i).getType());
    auto newEr =
        builder.create<scf::ExecuteRegionOp>(er.getLoc(), newResultTypes);
    newEr.getRegion().takeBody(er.getRegion());
    rebuiltOps.insert(newEr.getOperation());
    for (unsigned i = 0; i < er.getNumResults(); ++i) {
      Value oldRes = er.getResult(i);
      Value newRes = newEr.getResult(i);
      if (i == slot) {
        valueMap[oldRes] = newRes;
        worklist.push_back({oldRes, newRes});
      } else {
        oldRes.replaceAllUsesWith(newRes);
      }
    }
    toErase.insert(er.getOperation());
  }

  void rebuildIfResultSlot(scf::IfOp ifOp, unsigned slot) {
    OpBuilder builder(ifOp);
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(ifOp.getNumResults());
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      newResultTypes.push_back(i == slot ? Type(arrType)
                                         : ifOp.getResult(i).getType());
    bool withElse = !ifOp.getElseRegion().empty();
    auto newIf = builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                           ifOp.getCondition(), withElse);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (withElse)
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    rebuiltOps.insert(newIf.getOperation());
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Value oldRes = ifOp.getResult(i);
      Value newRes = newIf.getResult(i);
      if (i == slot) {
        valueMap[oldRes] = newRes;
        worklist.push_back({oldRes, newRes});
      } else {
        oldRes.replaceAllUsesWith(newRes);
      }
    }
    toErase.insert(ifOp.getOperation());
  }

  void rewriteConditionOperand(scf::ConditionOp c, unsigned opIdx,
                               Value newVal) {
    c->setOperand(opIdx, newVal);
  }
};

} // namespace

/// Rewrite struct-of-pods carriers (`!pod<[@idx_0..@idx_K-1: T]>`) to
/// array-of-pods (`!array<K x T>`) so the existing `flattenPodArrayWhileCarry`
/// infrastructure handles them. Driver placement: BEFORE
/// `materializePodArrayInputPodField` so the read-modify-write guard there
/// still sees pod-typed cells with the carrier already converted.
///
/// Canonical case: webb_poseidon_vanchor_2_2's `@Poseidon_137_compute` has a
/// 7-iter-arg outer scf.while carrying both `%arg2: !pod<[@idx_0..@idx_67: T]>`
/// (load-bearing read-modify-write carrier) and `%arg5: !array<68 x T>`
/// (parallel array form). `flattenPodArrayWhileCarry` handles `%arg5`; this
/// pass converts `%arg2` so its sibling phases follow.
bool convertStructOfPodsToArrayOfPods(Block &funcBlock) {
  // Deep walk for seeds (the carrier seed may be nested inside scf.while/
  // scf.if regions). Only `pod.new` seeds are processed; `llzk.nondet` of
  // struct-of-pods shape is a downstream-Phase-4 artifact whose chain is
  // already on its way to elimination — rewriting it forces an extra
  // outer fixed-point iter for no benefit.
  SmallVector<Operation *> seeds;
  funcBlock.walk([&](Operation *op) {
    if (!isa<llzk::pod::NewPodOp>(op))
      return;
    if (op->getNumResults() != 1)
      return;
    if (matchStructOfPodsShape(op->getResult(0).getType()))
      seeds.push_back(op);
  });

  for (Operation *seed : seeds) {
    StructOfPodsRewriter rewriter;
    if (rewriter.run(seed))
      return true;
  }
  return false;
}

/// Completion pass for NON-uniform-inner struct-of-pods carriers.
///
/// `convertStructOfPodsToArrayOfPods` only rewrites carriers whose K
/// `@idx_*` records share a single inner type. The canonical case it
/// CANNOT handle is webb Poseidon's 68-round Ark cascade, where each
/// `@idx_K` resolves to a distinct `!struct<@Ark_K>` round-constant class
/// (`matchStructOfPodsShape`'s uniform-inner check rejects). The carrier
/// stays alive on the scf.while iter-arg; `processNested`'s
/// `hasPodBlockArg=true` branch runs `extractCallsFromScfIf` to clone-hoist
/// every dispatched `function.call @Ark_K::@compute(%array.extract
/// %carrier[%cK]) : !struct<@Ark_K>` out of its statically-false dispatch
/// scf.if. After that, the writer-side cascade has N hoisted calls inside
/// an inner scf.while body, but the *reader*-side cascade — in a SIBLING
/// scf.while body — still reads `struct.readm [@F]` from a pre-existing
/// `llzk.nondet : !struct<@Ark_K>` (NOT from the hoisted call results;
/// the writer↔reader dispatch link was severed by an earlier phase). The
/// hoisted call's result has no consumer to RAUW onto and
/// `--llzk-to-stablehlo` DCEs it.
///
/// This pass closes the link by materializing one parallel felt array per
/// `@F` member referenced by readers: `<N x ...innerFeltDims>` allocated
/// at function-body scope (so it dominates both writer and reader
/// scf.while bodies). At each hoisted call site, emit `array.write
/// %carrier_F[%cK] = struct.readm %call[@F]` (or `array.insert` for
/// array-typed F) right after the call. Each reader-side
/// `struct.readm %nondet[@F]` is rewritten to `array.extract|array.read
/// %carrier_F[%cK]` using the writer-side class→K map (lookup by the
/// nondet's struct class leaf name). The original `llzk.nondet` becomes
/// use-empty once all its `struct.readm` consumers are rewritten and is
/// DCE'd by `eraseDeadPodAndCountOps` (Phase 4); the dispatch scf.if
/// cascade scaffolding is preserved per the "Don't reshape the `<--`
/// cascade" invariant — downstream `extendResultBearingScfIfArrayChain`
/// / `convertArrayWritesToSSA` need its result-slot shape intact.
///
/// SSA threading of `%carrier_F` through the writer + reader scf.while
/// iter-args is handled downstream by
/// `LlzkToStablehlo.cpp::convertWhileBodyArgsToSSA` and
/// `flattenPodArrayWhileCarry` — we don't manually rewire iter-args here.
/// LLZK is mutable at this stage so a function-scope `array.new` is
/// visible to writer/reader scf.while bodies via capture.
///
/// Idempotent: subsequent invocations find no `struct.readm` consumers on
/// the nondet ops (they were erased) and the readers list is empty, so
/// the pass returns false. Convergence with `eliminatePodDispatch`: emits
/// only `array.new` / `struct.readm` / `array.write`(insert) /
/// `array.extract`(read) — all are core (felt/struct/array) ops Phase 4
/// leaves alone, and the writer/reader scf.while bodies remain
/// computation-tagged scf.while (not scf.if/for) so the
/// `hasNonPodArrayWriteInBody` carve-out is not exercised.
///
/// Matching contract (single-operand calls, single-pub @F only —
/// matches Poseidon's Ark cascade today; multi-pub / multi-operand
/// shapes can extend the matcher when a chip demands it):
/// - Writer: `%call = function.call @<C>::@compute(%v) : !struct<@<C>>`
///   where `%v = array.extract %arr[%cK]` and `%cK = arith.constant <k>
///   : index`. Class leaf name `<C>` is the join key.
/// - Reader: `%nondet = llzk.nondet : !struct<@<C>>` followed by
///   `%readm = struct.readm %nondet[@F]`. Matching is by class leaf name;
///   no scf.if cascade decoding required.
/// - `@F` must be `{llzk.pub}` on the inner struct.def (LLZK's
///   `MemberReadOp::verifySymbolUses` rejects external reads of non-pub
///   members). Poseidon's `@Ark_K::@out` is pub by Circom contract.
bool materializeStructOfPodsCompField(Block &funcBlock) {
  struct Writer {
    Operation *call;
    StringRef className;
    int64_t kConst;
    // When the call sits inside one or more statically-false (count-guard)
    // scf.ifs, `hoistAbove` is the outermost such scf.if — the materializer
    // re-emits the call and carrier writes immediately BEFORE it so they
    // live in the next non-dead enclosing scope (typically a runtime
    // cascade-arm scf.if). nullptr means the call is already at a safe
    // scope and no hoist is needed.
    Operation *hoistAbove;
  };
  struct Reader {
    Operation *readm;
    StringRef className;
    StringRef field;
    Type fieldTy;
    std::optional<int64_t> resolvedK;
  };

  SmallVector<Writer> writers;
  SmallVector<Reader> readers;
  auto funcAnchorOf = [&funcBlock](Operation *op) -> Operation * {
    return op ? funcBlock.findAncestorOpInBlock(*op) : nullptr;
  };
  auto getDominanceRoot = [&funcBlock]() -> Operation * {
    Operation *root = funcBlock.getParentOp();
    while (root && root->getName().getStringRef() != "function.def" &&
           root->getName().getStringRef() != "func.func")
      root = root->getParentOp();
    return root ? root : funcBlock.getParentOp();
  };
  std::optional<DominanceInfo> domCache;
  auto dom = [&]() -> DominanceInfo & {
    if (!domCache)
      domCache.emplace(getDominanceRoot());
    return *domCache;
  };
  auto canDirectBindHoistedWriter = [&](const Writer &w,
                                        Operation *reader) -> bool {
    if (!w.hoistAbove || w.call->getNumResults() != 1)
      return false;
    if (!canCloneDefiningOpBefore(w.call->getOperand(0), *w.hoistAbove))
      return false;
    Operation *writerAnchor = funcAnchorOf(w.call);
    Operation *readerAnchor = funcAnchorOf(reader);
    if (!writerAnchor || !readerAnchor)
      return false;
    if (writerAnchor != readerAnchor &&
        !writerAnchor->isBeforeInBlock(readerAnchor))
      return false;
    if (writerAnchor == readerAnchor && !w.hoistAbove->isBeforeInBlock(reader))
      return false;
    return !isValueDefinedInside(reader->getOperand(0), *w.hoistAbove);
  };

  // Recognize circom's dispatch count-guard scf.if structurally: a void
  // scf.if (zero results) wrapping a `function.call`. Such an scf.if's
  // predicate is `arith.cmpi eq, arith.subi(@count, 1), 0` — initially
  // runtime (gated on the dispatch pod's `@count` field), then folded to
  // statically-false (`arith.subi 0, 1 == 0` ⇒ false) by
  // `rewriteArrayPodCountCompInReads` after the call has been recognized
  // for dispatch. Either way downstream DCE erases the scf.if body — we
  // must materialize the call's result BEFORE the call rides into dead
  // control flow. Result-bearing scf.ifs are the runtime cascade arms
  // (predicate `bool.and %true, eq(arg-1, c<K>)`); their bodies must stay
  // in-scope so the carrier insert fires conditionally on the cascade arm
  // K matching the dispatch K — those are NEVER walked past.
  auto isDispatchCountGuard = [](scf::IfOp ifOp) -> bool {
    if (ifOp.getNumResults() != 0)
      return false; // result-bearing scf.if is a runtime cascade arm.
    bool sawCall = false;
    for (Operation &nested : ifOp.getThenRegion().front()) {
      if (isa<llzk::function::CallOp>(nested)) {
        sawCall = true;
        break;
      }
    }
    return sawCall;
  };

  // Walk up `op`'s ancestor chain and return the outermost void scf.if
  // (count-guard) ancestor; returns nullptr if no count-guard surrounds the
  // call OR if `eliminatePodDispatch` Phase 1 would naturally hoist the
  // call out via its `extractCallsFromScfIf` driver.
  //
  // Phase 1 operates on `scf.if` ops at the IMMEDIATE block level of an
  // `scf.while` body (or function body). It does not recurse through
  // `scf.execute_region` — calls buried inside an execute_region's body are
  // invisible to Phase 1 and need this pass to bridge them. The canonical
  // case is Poseidon's full-rounds `@mix` cascade where the cascade itself
  // lives inside an `scf.execute_region -> !felt` wrapping the multi-K
  // branch tree.
  auto findDispatchCountGuardHoistAncestor =
      [&isDispatchCountGuard](Operation *op) -> Operation * {
    Operation *hoistAbove = nullptr;
    Operation *cur = op;
    while (Operation *parent = cur->getParentOp()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (ifOp && isDispatchCountGuard(ifOp)) {
        hoistAbove = ifOp.getOperation();
        cur = parent;
        continue;
      }
      break;
    }
    if (!hoistAbove)
      return nullptr;
    // Hoisting past the count-guard is only worthwhile when the resulting
    // call position is unreachable to Phase 1 (i.e. an scf.execute_region
    // sits between the count-guard and the enclosing scf.while body /
    // function body). Otherwise Phase 1 will hoist the call itself on its
    // own driver iter, and double-hoisting collides with the existing
    // dispatch teardown — yielding duplicated function.calls and lost
    // carrier inserts. Walk the remaining ancestor chain from the
    // count-guard's enclosing scope upward; return the count-guard only
    // when an `scf.execute_region` is found before reaching the scf.while
    // body / function body.
    Operation *probe = hoistAbove->getParentOp();
    while (probe) {
      if (isa<scf::ExecuteRegionOp>(probe))
        return hoistAbove;
      if (isa<scf::WhileOp, llzk::function::FuncDefOp, func::FuncOp>(probe))
        return nullptr;
      probe = probe->getParentOp();
    }
    return nullptr;
  };

  // Pre-scan: count ALL potential writers across the entire function body,
  // including ones still nested inside dispatch-firing scf.if cascade arms
  // that `eliminatePodDispatch` Phase 1 hasn't hoisted yet. This is needed
  // because Phase 1 hoists incrementally across outer fixed-point iters
  // (one or a few per iter), so the first invocation where my pass fires
  // would see only a partial set of writers and miscompute N. Without the
  // pre-scan, the allocated carrier shape `<N_partial x ...>` is too small
  // for cascade K values surfaced in later iters — the IR ends up with
  // out-of-bounds `array.insert %carrier[%c67] : <1,5>` and lowering fails.
  // The pre-scan still gates on the `array.extract %arr[%cK]` operand
  // pattern with K an arith.constant, so unrelated function.calls (e.g.
  // top-level @Poseidon::@compute calls, Mix calls with non-constant K)
  // are correctly excluded.
  // Extract the dispatch index K from a value `input` that flows through a
  // pod-dispatch chain. The SSC pipeline can leave the chain in any of
  // three equivalent shapes depending on which conversions have fired in
  // the outer fixed-point loop. All three encode the same K — the slot at
  // which the dispatched component instance lives.
  //   (a) `array.extract %arr[%cK]` — fully converted form (writer side
  //       only; reader-side chains always go through `pod.read [@outer]`).
  //   (b) `pod.read %cell[@outer]` ← `array.read %arr[%cK]` — partial: the
  //       outer carrier has been rewritten by
  //       `convertStructOfPodsToArrayOfPods` but the per-cell `pod.read
  //       [@outer]` hasn't been folded yet.
  //   (c) `pod.read %cell[@outer]` ← `pod.read %struct[@idx_K]` — pre-
  //       conversion: the outer carrier is still a struct-of-pods. The K
  //       is encoded as the `@idx_K` field-name suffix.
  // `outerRecord` selects which pod cell the chain reads through:
  //   - "in" — the writer-side `function.call`'s input chain reads through
  //            the dispatch pod's `@in` cell.
  //   - "comp" — the reader-side `struct.readm` chain reads through the
  //              dispatch pod's `@comp` cell.
  // Returns std::nullopt if no pattern matches.
  auto extractDispatchK = [](Value input,
                             StringRef outerRecord) -> std::optional<int64_t> {
    Operation *def = input.getDefiningOp();
    if (!def)
      return std::nullopt;
    auto getConstIdx = [](Value v) -> std::optional<int64_t> {
      auto *d = v.getDefiningOp();
      if (!d || !isa<arith::ConstantOp>(d))
        return std::nullopt;
      auto attr = d->getAttrOfType<IntegerAttr>("value");
      if (!attr)
        return std::nullopt;
      llvm::APInt ap = attr.getValue();
      if (ap.getBitWidth() > 64 || ap.isNegative())
        return std::nullopt;
      return ap.getSExtValue();
    };
    // Case (a): direct `array.extract %arr[%cK]`.
    if (isa<llzk::array::ExtractArrayOp>(def) && def->getNumOperands() == 2)
      return getConstIdx(def->getOperand(1));
    // Cases (b) and (c) both arrive as `pod.read [@outerRecord]`.
    if (isa<llzk::pod::ReadPodOp>(def) && def->getNumOperands() >= 1) {
      auto rn = def->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!rn || rn.getValue() != outerRecord)
        return std::nullopt;
      Operation *inner = def->getOperand(0).getDefiningOp();
      if (!inner)
        return std::nullopt;
      // Case (b): inner is `array.read %arr[%cK]`.
      if (isa<llzk::array::ReadArrayOp>(inner) && inner->getNumOperands() == 2)
        return getConstIdx(inner->getOperand(1));
      // Case (c): inner is `pod.read %struct[@idx_K]`.
      if (isa<llzk::pod::ReadPodOp>(inner)) {
        auto rn2 = inner->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (!rn2)
          return std::nullopt;
        StringRef recName = rn2.getValue();
        if (!recName.starts_with("idx_"))
          return std::nullopt;
        int64_t k;
        if (recName.drop_front(4).getAsInteger(10, k))
          return std::nullopt;
        return k;
      }
    }
    return std::nullopt;
  };

  // Pre-scan tracks max K observed across ALL writer candidates, regardless
  // of whether their class is repeated at multiple K positions. The carrier
  // shape is sized off this max, never off a per-class K — so a class
  // instantiated at multiple K's (e.g. Poseidon's full-round `@Mix_81` at
  // K={0,1,2,4,5,6}) carries its slots through unchanged.
  int64_t preScanMaxK = -1;
  funcBlock.walk([&](Operation *op) {
    if (!isa<llzk::function::CallOp>(op) || op->getNumResults() != 1 ||
        op->getNumOperands() != 1)
      return;
    auto structTy =
        dyn_cast<llzk::component::StructType>(op->getResult(0).getType());
    if (!structTy)
      return;
    auto k = extractDispatchK(op->getOperand(0), "in");
    if (!k)
      return;
    preScanMaxK = std::max(preScanMaxK, *k);
  });

  // Marker attribute on the writer's function.call op signaling that this
  // pass has ALREADY emitted its `struct.readm + array.insert` shadow chain
  // for this call. The outer fixed-point loop re-invokes this pass after
  // every successful emission; without the marker we would re-emit every
  // iter, allocating a fresh carrier each time and accumulating O(N²)
  // dead ops + carrier-rename churn before `--llzk-to-stablehlo` DCEs them.
  StringRef kMaterializedAttr = "mSoPCF.materialized";
  funcBlock.walk([&](Operation *op) {
    if (isa<llzk::function::CallOp>(op) && op->getNumResults() == 1 &&
        op->getNumOperands() == 1) {
      auto structTy =
          dyn_cast<llzk::component::StructType>(op->getResult(0).getType());
      if (!structTy)
        return;
      // Idempotence: skip writers already processed by a previous outer iter.
      if (op->hasAttr(kMaterializedAttr))
        return;
      auto k = extractDispatchK(op->getOperand(0), "in");
      if (!k)
        return;
      StringRef cls = structTy.getNameRef().getLeafReference().getValue();
      // Skip writers inside a runtime (non-statically-false) scf.if that is
      // NOT a count-guard pattern — `eliminatePodDispatch` Phase 1 hoists
      // those out, and emitting now would race with that hoist. Statically-
      // false enclosing scf.ifs (the `arith.subi 0, 1` count-guard pattern)
      // ARE allowed: we hoist the materialized call past them ourselves
      // below, so they don't trap the emission inside dead control flow.
      Operation *hoistAbove = findDispatchCountGuardHoistAncestor(op);
      Block *callBlock = op->getBlock();
      // Skip writers nested inside scf.if that is NOT a dispatch count-guard
      // (i.e. inside a runtime scf.if cascade arm). `eliminatePodDispatch`
      // Phase 1 hoists those out on the next outer iter; emitting now races
      // with the hoist. The count-guard ancestor case ALWAYS allows emission
      // — we re-emit the call ourselves past the guard below.
      if (!hoistAbove && callBlock && callBlock->getParentOp() &&
          isa<scf::IfOp>(callBlock->getParentOp()))
        return;
      writers.push_back({op, cls, *k, hoistAbove});
      return;
    }

    // Reader source can be either `llzk.nondet : !struct<@<C>>` (post
    // post-loop pod.read→nondet conversion at line ~5915) OR
    // `pod.read %something[@comp] : !struct<@<C>>` (BEFORE that conversion,
    // i.e. during the outer fixed point — which is when this pass runs).
    // Both produce the same downstream `struct.readm [@F]` consumer.
    // Match against both shapes so the materializer fires inside the
    // outer fixed point and the cleanup phase finds nothing left to
    // nondet-replace for the cascade arms.
    bool isStructReaderSrc =
        (isa<llzk::NonDetOp>(op) && op->getNumResults() == 1) ||
        (isa<llzk::pod::ReadPodOp>(op) && op->getNumResults() == 1);
    if (isStructReaderSrc) {
      auto structTy =
          dyn_cast<llzk::component::StructType>(op->getResult(0).getType());
      if (!structTy)
        return;
      // pod.read must be on the dispatch pod's @comp field — otherwise
      // it's reading some unrelated pod member that happens to be struct-
      // typed (no such case exists in current chips, but the explicit
      // gate keeps the matcher narrow).
      if (isa<llzk::pod::ReadPodOp>(op)) {
        auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (!rn || rn.getValue() != "comp")
          return;
      }
      StringRef cls = structTy.getNameRef().getLeafReference().getValue();
      for (OpOperand &use : op->getResult(0).getUses()) {
        Operation *user = use.getOwner();
        if (!isa<llzk::component::MemberReadOp>(user) ||
            user->getNumResults() != 1)
          continue;
        auto memAttr = user->getAttrOfType<FlatSymbolRefAttr>("member_name");
        if (!memAttr)
          continue;
        readers.push_back(
            {user, cls, memAttr.getValue(), user->getResult(0).getType()});
      }
    }
  });

  if (writers.empty() || readers.empty())
    return false;

  // Build class→Ks map. Multiple writers with the same (class, K) are fine
  // (duplicate cascade arms emit identical results — last-write-wins).
  // A class instantiated at multiple distinct K values is also fine — circom
  // can reuse the same component class across distinct dispatch slots (e.g.
  // Poseidon's full-rounds `@mix` field where `@Mix_81` appears at K={0,1,2,
  // 4,5,6} alongside `@Mix_85` at K=3). Each (class, K) writer gets its own
  // carrier slot; reader-side disambiguation walks the surrounding scf.if
  // cascade predicate (`arith.cmpi eq %expr, %c<K>`) to recover K when the
  // class appears at more than one slot.
  llvm::StringMap<llvm::SmallVector<int64_t, 4>> classToKs;
  for (const Writer &w : writers) {
    auto &ks = classToKs[w.className];
    if (llvm::find(ks, w.kConst) == ks.end())
      ks.push_back(w.kConst);
  }

  // Cap dispatch dim N at max K + 1. Use the pre-scan's max K (covering
  // both hoisted and not-yet-hoisted cascade arms) so the carrier shape
  // is stable across outer iters — see pre-scan comment above for the
  // partial-hoist trap this avoids.
  int64_t N = 0;
  for (const auto &kv : classToKs)
    for (int64_t k : kv.getValue())
      N = std::max(N, k + 1);
  N = std::max(N, preScanMaxK + 1);

  // Group readers by (@F, fieldTy). Drop readers whose class has no writer
  // (orphan nondets from unrelated dispatch chains). Same field name with
  // different inner types is supported via per-(field,type) carriers — this
  // is necessary for chips like iden3's `Poseidon3` where `@out` is
  // `!array<4 x !felt>` on `Mix_81/Mix_85` and `!felt` on the partial-round
  // `MixS_*` sibling. Each (field, type) bucket gets its own
  // `array.new`-allocated carrier of the matching inner shape.
  struct DirectValueEntry {
    StringRef className;
    int64_t kConst;
    Value value;
  };
  struct FieldPlan {
    StringRef field;
    Type fieldTy;
    Value carrier;
    SmallVector<Reader *> targetReaders;
    bool useDirectBinding = false;
    SmallVector<DirectValueEntry> directValues;
  };
  SmallVector<FieldPlan> fieldPlans;
  for (Reader &r : readers) {
    if (!classToKs.count(r.className))
      continue;
    FieldPlan *plan = nullptr;
    for (auto &fp : fieldPlans) {
      if (fp.field == r.field && fp.fieldTy == r.fieldTy) {
        plan = &fp;
        break;
      }
    }
    if (!plan) {
      fieldPlans.push_back({r.field, r.fieldTy, Value(), {&r}});
    } else {
      plan->targetReaders.push_back(&r);
    }
  }
  if (fieldPlans.empty())
    return false;

  // Reader-side K disambiguation. When a class is instantiated at exactly one
  // K, every reader of that class targets that K — direct lookup.
  // When a class is instantiated at multiple K's, K is encoded in EITHER of:
  //   (B) The surrounding scf.if cascade predicate. Walk up enclosing scf.if
  //       regions whose `then` branch contains the reader, and pull K from
  //       the first predicate of form `arith.cmpi eq %expr, %c<K>`
  //       (optionally wrapped in `bool.and %true, %cmp`) whose K is one of
  //       the class's writer slots. This handles readers nested inside the
  //       cascade arm chain itself.
  //   (A) The reader's dispatch chain feeding the readm. Pre-Phase-5, the
  //       chain `struct.readm` ← `pod.read [@comp]` ← `array.read %arr[%cK]`
  //       (or `pod.read %struct[@idx_K]`) is intact and `extractDispatchK`
  //       on the readm's source recovers K. This handles post-cascade
  //       readers in sibling-while bodies with no surrounding scf.if
  //       predicate, e.g. iden3 Poseidon3's post-cascade Sigma_F loop
  //       where Mix_81 at K={0,1,2,4,5,6} carries through to a fresh
  //       `scf.while` body outside the cascade arm chain.
  // Try (B) first — it is the established mechanism for cascade-arm
  // readers; fall back to (A) for post-cascade readers. Readers that fail
  // both are left as nondet→struct.readm; the existing Phase 5 nondet
  // replacement and DCE retire them safely.
  auto deriveReaderK = [&classToKs, &extractDispatchK](
                           Operation *readm,
                           StringRef className) -> std::optional<int64_t> {
    auto it = classToKs.find(className);
    if (it == classToKs.end())
      return std::nullopt;
    const auto &ks = it->second;
    // (B) Walk up enclosing scf.if cascade arm predicates.
    auto isArithCmpiEq = [](Operation *cmpDef) -> bool {
      auto cmpOp = dyn_cast_or_null<arith::CmpIOp>(cmpDef);
      return cmpOp && cmpOp.getPredicate() == arith::CmpIPredicate::eq;
    };
    auto unwrapBoolAnd = [](Value cond) -> Value {
      Operation *def = cond.getDefiningOp();
      if (!def || def->getName().getStringRef() != "bool.and" ||
          def->getNumOperands() != 2)
        return cond;
      // Drop the `arith.constant true` operand; keep the other.
      for (Value operand : def->getOperands()) {
        Operation *opd = operand.getDefiningOp();
        if (opd && opd->getName().getStringRef() == "arith.constant") {
          auto attr = opd->getAttrOfType<IntegerAttr>("value");
          if (attr && attr.getValue().getBitWidth() == 1 &&
              attr.getValue().isOne())
            continue; // skip %true
        }
        return operand;
      }
      return cond;
    };
    Operation *cur = readm;
    while (cur) {
      Operation *parent = cur->getParentOp();
      if (!parent)
        break;
      if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
        Region *thenR = &ifOp.getThenRegion();
        bool inThen = false;
        for (Region *r = cur->getParentRegion(); r; r = r->getParentRegion()) {
          if (r == thenR) {
            inThen = true;
            break;
          }
          if (r == &ifOp.getElseRegion())
            break;
        }
        if (inThen) {
          Value cond = unwrapBoolAnd(ifOp.getCondition());
          Operation *cmpDef = cond.getDefiningOp();
          if (isArithCmpiEq(cmpDef)) {
            Operation *rhsDef = cmpDef->getOperand(1).getDefiningOp();
            if (rhsDef &&
                rhsDef->getName().getStringRef() == "arith.constant") {
              auto kAttr = rhsDef->getAttrOfType<IntegerAttr>("value");
              if (kAttr) {
                llvm::APInt apK = kAttr.getValue();
                if (apK.getBitWidth() <= 64 && !apK.isNegative()) {
                  int64_t k = apK.getSExtValue();
                  if (llvm::find(ks, k) != ks.end())
                    return k;
                }
              }
            }
          }
        }
      }
      cur = parent;
    }
    if (readm->getNumOperands() < 1)
      return std::nullopt;
    auto kOpt = extractDispatchK(readm->getOperand(0), "comp");
    if (kOpt && llvm::find(ks, *kOpt) != ks.end())
      return kOpt;
    return std::nullopt;
  };

  // Verify each `@F` is `{llzk.pub}` on every writer class's struct.def.
  // The LLZK verifier rejects `struct.readm` of a non-pub member from
  // outside the member's parent struct (see CLAUDE.md "Load-Bearing
  // Invariants"). Poseidon's `@out` is pub by Circom contract, but a
  // future cascade with private members would silently fail the LIT /
  // verifier — drop those fields rather than emit illegal IR.
  //
  // Single moduleOp walk builds a class → pub-fields map, then probe
  // in-memory: the naive nested form is O(fields × classes × moduleWalk)
  // per outer iter, which on the 68-class Ark cascade is ~70× redundant
  // moduleOp walks per iter.
  ModuleOp moduleOp = getTopLevelModule(funcBlock);
  if (moduleOp) {
    llvm::StringMap<llvm::StringSet<>> pubFieldsByClass;
    for (const auto &kv : classToKs)
      pubFieldsByClass[kv.getKey()];
    moduleOp->walk([&](Operation *defOp) {
      if (!isa<llzk::component::StructDefOp>(defOp))
        return;
      auto sym = defOp->getAttrOfType<StringAttr>("sym_name");
      if (!sym)
        return;
      auto it = pubFieldsByClass.find(sym.getValue());
      if (it == pubFieldsByClass.end())
        return;
      defOp->walk([&](Operation *m) {
        if (!isa<llzk::component::MemberDefOp>(m))
          return;
        if (!m->hasAttr(llzk::PublicAttr::name))
          return;
        auto memSym = m->getAttrOfType<StringAttr>("sym_name");
        if (memSym)
          it->second.insert(memSym.getValue());
      });
    });
    llvm::erase_if(fieldPlans, [&](const FieldPlan &fp) {
      for (const auto &cls : pubFieldsByClass)
        if (!cls.getValue().contains(fp.field))
          return true;
      return false;
    });
  }
  if (fieldPlans.empty())
    return false;

  for (FieldPlan &fp : fieldPlans) {
    bool allReadersStatic = true;
    for (Reader *r : fp.targetReaders) {
      const auto &ks = classToKs[r->className];
      if (ks.size() == 1)
        r->resolvedK = ks.front();
      else
        r->resolvedK = deriveReaderK(r->readm, r->className);
      if (!r->resolvedK) {
        allReadersStatic = false;
        break;
      }
    }

    if (!allReadersStatic)
      continue;

    bool allReadersDirectBindable = true;
    for (Reader *r : fp.targetReaders) {
      bool foundWriter = false;
      for (const Writer &w : writers) {
        if (w.className != r->className || w.kConst != *r->resolvedK ||
            w.call->getNumResults() != 1)
          continue;
        if (w.hoistAbove) {
          if (!canDirectBindHoistedWriter(w, r->readm))
            continue;
        } else {
          Operation *writerAnchor = funcAnchorOf(w.call);
          Operation *readerAnchor = funcAnchorOf(r->readm);
          if (!writerAnchor || !readerAnchor)
            continue;
          if (writerAnchor != readerAnchor &&
              !writerAnchor->isBeforeInBlock(readerAnchor))
            continue;
          if (!dom().properlyDominates(w.call->getResult(0), r->readm))
            continue;
        }
        foundWriter = true;
        break;
      }
      if (!foundWriter) {
        allReadersDirectBindable = false;
        break;
      }
    }

    if (!allReadersDirectBindable)
      continue;

    fp.useDirectBinding = true;
  }

  // Allocate one carrier per surviving `@F` at funcBlock front. Front
  // placement guarantees dominance over every writer + reader scf.while
  // body in the function. Carriers are tagged with a `mSoPCF.carrier-for`
  // attribute so subsequent outer iters of this pass — which see
  // additional writers hoisted incrementally by Phase 1 (one or a few per
  // outer fixed-point iter, not all at once) — can REUSE the same carrier
  // instead of allocating fresh ones. Without reuse, the per-iter
  // allocations leak ~N orphan `array.new` ops, each carrying ~1 insert,
  // and `processBlockForArrayMutations` only threads ONE carrier as
  // iter-arg per scf.while body, leaving the others orphaned and DCE-bait.
  // Carrier matching keys on (field name, carrier shape). The shape is the
  // inner dims appended with the dispatch dim — so two carriers for the same
  // @F but different inner types (e.g. `@out` as `!felt` for partial-round
  // `MixS_*` vs `!array<4 x !felt>` for full-round `Mix_81/85`) get distinct
  // `array.new`s. The attribute alone (which encodes only the field name)
  // would alias these two into one carrier; tie-break by shape via the
  // array.new's result type.
  StringRef kCarrierAttr = "mSoPCF.carrier-for";
  Location loc = funcBlock.getParentOp()->getLoc();
  OpBuilder builder(&funcBlock, funcBlock.begin());
  for (FieldPlan &fp : fieldPlans) {
    if (fp.useDirectBinding)
      continue;
    auto carrierTy = combineDispatchAndInnerFeltDims(fp.fieldTy, {N});
    // First pass: locate existing carrier with matching (field, type).
    Value existing;
    for (Operation &op : funcBlock) {
      if (!isa<llzk::array::CreateArrayOp>(op))
        continue;
      auto attr = op.getAttrOfType<StringAttr>(kCarrierAttr);
      if (!attr || attr.getValue() != fp.field)
        continue;
      if (op.getNumResults() != 1 || op.getResult(0).getType() != carrierTy)
        continue;
      existing = op.getResult(0);
      break;
    }
    if (existing) {
      fp.carrier = existing;
      continue;
    }
    auto newOp = builder.create<llzk::array::CreateArrayOp>(loc, carrierTy);
    newOp->setAttr(kCarrierAttr, builder.getStringAttr(fp.field));
    fp.carrier = newOp;
  }

  // Duplicate (class, K) writers (e.g. Poseidon K=0 hit twice from cascade
  // arm collapse) overwrite the same carrier slot — semantically a no-op
  // since both produce identical results.
  //
  // Writers with `hoistAbove != nullptr` live inside one or more
  // statically-false (count-guard) scf.ifs — emit a clone of the call
  // immediately before the outermost such scf.if so the carrier writes
  // execute under the next non-dead enclosing predicate (typically a
  // runtime cascade-arm scf.if). The original in-scf.if call becomes
  // dead and is DCE'd downstream; we mark BOTH the original and the
  // clone as materialized so the outer fixed point sees the work done.
  for (const Writer &w : writers) {
    Operation *callRef = w.call;
    Operation *emitAnchor = w.call;
    if (w.hoistAbove) {
      OpBuilder hb(w.hoistAbove);
      llvm::DenseMap<Value, Value> cloneCache;
      SmallVector<Value> newArgs;
      newArgs.reserve(w.call->getNumOperands());
      bool ok = true;
      for (Value operand : w.call->getOperands()) {
        // `cloneDefiningOpBefore` returns the value as-is if defined outside
        // `*w.hoistAbove`, and otherwise clones the safe-to-clone defining
        // chain (Pure ops + LLZK read-only `array.extract`/`array.read`/
        // `array.len`). For the Poseidon multi-K cascade the args are
        // `array.extract %arg7[%cK]` where %arg7 is the enclosing scf.while
        // body block-arg (outside hoistAbove) and %cK is an arith.constant
        // — both clone trivially.
        Value cloned = cloneDefiningOpBefore(operand, w.hoistAbove,
                                             *w.hoistAbove, cloneCache);
        if (!cloned) {
          ok = false;
          break;
        }
        newArgs.push_back(cloned);
      }
      if (!ok) {
        // Can't hoist this writer; skip it. Readers for this (class, K)
        // pair will fail to disambiguate (no carrier write) and stay as
        // nondet→readm — equivalent to the pre-fix silent miscompile for
        // this slot, but no NEW regression.
        continue;
      }
      OperationState callState(w.call->getLoc(), "function.call");
      callState.addOperands(newArgs);
      callState.addTypes(w.call->getResultTypes());
      for (auto &attr : w.call->getAttrs())
        callState.addAttribute(attr.getName(), attr.getValue());
      Operation *newCall = hb.create(callState);
      newCall->setAttr(kMaterializedAttr, hb.getUnitAttr());
      callRef = newCall;
      emitAnchor = newCall;
    }
    OpBuilder wb(emitAnchor);
    wb.setInsertionPointAfter(emitAnchor);
    Value kIdx = buildConstIndex(wb, emitAnchor->getLoc(), w.kConst);
    // Emit a carrier write for every surviving (field, type) plan. All
    // classes that share a dispatch carrier in current iden3 / circomlib
    // chips also declare each `@F`'s type identically (e.g. Mix_81,
    // Mix_85, MixS_100..141, MixLast, Ark_* all expose `@out : !array<4 x
    // !felt>`), so a single `@out` plan absorbs every writer. The plan
    // survives the upstream `pubFieldsByClass` gate only when every class
    // exposes the field as `{llzk.pub}` with the recorded type, so
    // emitting against `fp.fieldTy` will type-check on the resulting
    // `struct.readm` cast. If a future chip mixes types under one field
    // name, add a per-writer type filter here keyed on the class's
    // struct.def member type — until then the unconditional emit matches
    // the original pre-PR-#116 behavior and avoids re-introducing the
    // silent-miscompile pattern where reader-less writers (e.g. the
    // partial-rounds `@mixS` cascade in `PoseidonEx_146`) skip the
    // carrier write entirely.
    for (FieldPlan &fp : fieldPlans) {
      Value feltVal = wb.create<llzk::component::MemberReadOp>(
          emitAnchor->getLoc(), fp.fieldTy, callRef->getResult(0),
          wb.getStringAttr(fp.field));
      if (fp.useDirectBinding) {
        // Preserve every candidate value for this (class, K). Duplicate
        // cascade-arm writers can survive in the same function, and direct
        // binding must pick a candidate that actually dominates each reader.
        fp.directValues.push_back({w.className, w.kConst, feltVal});
        continue;
      }
      StringRef writeOpName = isa<llzk::array::ArrayType>(fp.fieldTy)
                                  ? "array.insert"
                                  : "array.write";
      OperationState writeState(emitAnchor->getLoc(), writeOpName);
      writeState.addOperands({fp.carrier, kIdx, feltVal});
      wb.create(writeState);
    }
    // Mark the original call so the next outer iter's walker skips it.
    w.call->setAttr(kMaterializedAttr, wb.getUnitAttr());
  }

  // Scalar `@F` uses `array.read` (full indices → single element);
  // array-typed `@F` uses `array.extract` (partial indices → sub-array slice).
  SmallVector<Operation *> readmsToErase;
  std::optional<DominanceInfo> directBindDomCache;
  auto directBindDom = [&]() -> DominanceInfo & {
    if (!directBindDomCache)
      directBindDomCache.emplace(getDominanceRoot());
    return *directBindDomCache;
  };
  for (FieldPlan &fp : fieldPlans) {
    if (fp.useDirectBinding) {
      for (Reader *r : fp.targetReaders) {
        if (!r->resolvedK)
          continue;
        Value directValue;
        for (const DirectValueEntry &entry : fp.directValues) {
          if (entry.className == r->className &&
              entry.kConst == *r->resolvedK &&
              directBindDom().properlyDominates(entry.value, r->readm)) {
            directValue = entry.value;
            break;
          }
        }
        if (!directValue)
          continue;
        r->readm->getResult(0).replaceAllUsesWith(directValue);
        readmsToErase.push_back(r->readm);
      }
      continue;
    }

    for (Reader *r : fp.targetReaders) {
      const auto &ks = classToKs[r->className];
      int64_t k;
      if (ks.size() == 1) {
        k = ks.front();
      } else {
        if (!r->resolvedK)
          continue; // can't disambiguate; leave nondet→struct.readm intact.
        k = *r->resolvedK;
      }
      OpBuilder rb(r->readm);
      Value kIdx = buildConstIndex(rb, r->readm->getLoc(), k);
      StringRef readOpName = isa<llzk::array::ArrayType>(fp.fieldTy)
                                 ? "array.extract"
                                 : "array.read";
      OperationState readState(r->readm->getLoc(), readOpName);
      readState.addOperands({fp.carrier, kIdx});
      readState.addTypes({fp.fieldTy});
      Value extracted = rb.create(readState)->getResult(0);
      r->readm->getResult(0).replaceAllUsesWith(extracted);
      readmsToErase.push_back(r->readm);
    }
  }
  for (Operation *op : readmsToErase)
    op->erase();

  return true;
}

} // namespace mlir::llzk_to_shlo
