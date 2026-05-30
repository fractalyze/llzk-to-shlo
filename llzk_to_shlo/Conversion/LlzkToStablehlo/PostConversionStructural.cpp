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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PostConversionStructural.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
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
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

namespace {

/// Convert init values for a scf.while → stablehlo.while conversion.
/// Returns pair<SmallVector<Value>, SmallVector<Type>> of (convertedInits,
/// convertedTypes). Logic: looks through casts, finds converted types from
/// body args.
std::pair<SmallVector<Value>, SmallVector<Type>>
convertWhileInitValues(OpBuilder &builder, scf::WhileOp whileOp) {
  // First pass: find the element type from any already-converted value
  Type fieldElemType;
  for (Value init : whileOp.getInits()) {
    if (auto tt = dyn_cast<RankedTensorType>(init.getType())) {
      fieldElemType = tt.getElementType();
      break;
    }
  }
  // Also check body arg casts for element type
  if (!fieldElemType) {
    Block &body = whileOp.getAfter().front();
    for (auto arg : body.getArguments()) {
      for (auto *user : arg.getUsers()) {
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
          if (auto tt =
                  dyn_cast<RankedTensorType>(castOp.getResult(0).getType())) {
            fieldElemType = tt.getElementType();
            break;
          }
        }
      }
      if (fieldElemType)
        break;
    }
  }

  SmallVector<Value> convertedInits;
  SmallVector<Type> convertedTypes;
  for (Value init : whileOp.getInits()) {
    if (isa<RankedTensorType>(init.getType())) {
      convertedInits.push_back(init);
      convertedTypes.push_back(init.getType());
    } else {
      // Look through existing cast: felt.type → tensor<!pf>
      Value converted = init;
      for (auto *user : init.getUsers()) {
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
          if (castOp.getNumResults() == 1 &&
              isa<RankedTensorType>(castOp.getResult(0).getType())) {
            converted = castOp.getResult(0);
            break;
          }
        }
      }
      if (converted == init) {
        // No existing cast found. Determine the tensor type from:
        // 1. Body arg's cast users
        // 2. The init value's defining op's result cast
        // 3. Direct type inference (felt → tensor<!pf>, array → tensor<Nx!pf>)
        Type tensorType;

        // Strategy 1: look at body arg users for a cast
        Block &body = whileOp.getAfter().front();
        unsigned argIdx =
            llvm::find(whileOp.getInits(), init) - whileOp.getInits().begin();
        if (argIdx < body.getNumArguments()) {
          for (auto *user : body.getArgument(argIdx).getUsers()) {
            if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
              if (castOp.getNumResults() == 1) {
                tensorType = castOp.getResult(0).getType();
                break;
              }
            }
          }
        }

        // Strategy 2: look at the defining op's other result users
        if (!tensorType) {
          if (auto *defOp = init.getDefiningOp()) {
            for (auto *user : defOp->getResult(0).getUsers()) {
              if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
                if (castOp.getNumResults() == 1 &&
                    isa<RankedTensorType>(castOp.getResult(0).getType())) {
                  tensorType = castOp.getResult(0).getType();
                  break;
                }
              }
            }
          }
        }

        // Strategy 3: construct tensor type from fieldElemType
        if (!tensorType && fieldElemType) {
          Type initTy = init.getType();
          if (isa<llzk::felt::FeltType>(initTy))
            tensorType = RankedTensorType::get({}, fieldElemType);
          else if (isa<llzk::array::ArrayType>(initTy))
            tensorType = RankedTensorType::get(getArrayDimensions(initTy),
                                               fieldElemType);
        }

        // Strategy 4: wrap bare scalar types (i1, index) in tensor<>.
        if (!tensorType && init.getType().isIntOrIndexOrFloat())
          tensorType = RankedTensorType::get({}, init.getType());

        if (tensorType) {
          converted = builder
                          .create<UnrealizedConversionCastOp>(whileOp.getLoc(),
                                                              tensorType, init)
                          .getResult(0);
        }
      }
      convertedInits.push_back(converted);
      convertedTypes.push_back(converted.getType());
    }
  }
  return {convertedInits, convertedTypes};
}
} // namespace

/// Convert scf.while → stablehlo.while (post-pass, after type conversion).
/// This ensures the while body ops are visible to dialect conversion for type
/// conversion. Only changes terminators: scf.condition → stablehlo.return,
/// scf.yield → stablehlo.return.
void convertScfWhileToStablehloWhile(ModuleOp module) {
  // Process one while at a time, innermost first. Re-collect after each
  // conversion since takeBody invalidates nested while pointers.
  bool converted = true;
  while (converted) {
    converted = false;
    scf::WhileOp target;
    module.walk([&](scf::WhileOp op) {
      // Find the innermost while (no nested scf.while in its body).
      bool hasNestedWhile = false;
      op.getAfter().walk([&](scf::WhileOp) { hasNestedWhile = true; });
      if (!hasNestedWhile)
        target = op;
    });
    if (!target)
      break;
    auto whileOp = target;
    converted = true;
    OpBuilder builder(whileOp);

    // Convert init values: insert unrealized_conversion_cast for non-tensor
    // types (felt → tensor, array → tensor) so stablehlo.while has valid types
    auto [convertedInits, convertedTypes] =
        convertWhileInitValues(builder, whileOp);

    auto newWhile = builder.create<stablehlo::WhileOp>(
        whileOp.getLoc(), convertedTypes, convertedInits);

    // Move condition region
    Region &condSrc = whileOp.getBefore();
    Region &condDst = newWhile.getCond();
    condDst.takeBody(condSrc);

    // Convert block arg types in condition region
    Block &condBlock = condDst.front();
    for (auto [idx, arg] : llvm::enumerate(condBlock.getArguments())) {
      if (idx < convertedTypes.size())
        arg.setType(convertedTypes[idx]);
    }

    // Replace scf.condition with stablehlo.return of predicate
    if (auto condOp = dyn_cast<scf::ConditionOp>(condBlock.getTerminator())) {
      OpBuilder termBuilder(condOp);
      Value pred = condOp.getCondition();
      // stablehlo.while expects tensor<i1> predicate.
      // Look through unrealized_conversion_cast to find original tensor<i1>.
      if (!isa<RankedTensorType>(pred.getType())) {
        if (auto castOp = pred.getDefiningOp<UnrealizedConversionCastOp>()) {
          Value src = castOp.getInputs()[0];
          if (isa<RankedTensorType>(src.getType()))
            pred = src;
        }
      }
      if (!isa<RankedTensorType>(pred.getType())) {
        pred = termBuilder.create<tensor::FromElementsOp>(
            condOp.getLoc(), RankedTensorType::get({}, pred.getType()), pred);
      }
      termBuilder.create<stablehlo::ReturnOp>(condOp.getLoc(),
                                              ValueRange{pred});
      condOp.erase();
    }

    // Move body region
    Region &bodySrc = whileOp.getAfter();
    Region &bodyDst = newWhile.getBody();
    bodyDst.takeBody(bodySrc);

    // Convert block arg types in body region
    Block &bodyBlock = bodyDst.front();
    for (auto [idx, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      if (idx < convertedTypes.size())
        arg.setType(convertedTypes[idx]);
    }

    // Replace scf.yield with stablehlo.return, converting operand types
    if (auto yieldOp = dyn_cast<scf::YieldOp>(bodyBlock.getTerminator())) {
      OpBuilder termBuilder(yieldOp);
      SmallVector<Value> yieldValues;
      for (auto [idx, val] : llvm::enumerate(yieldOp.getOperands())) {
        Value v = val;
        // Look through cast: felt → tensor
        if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getNumOperands() == 1 &&
              isa<RankedTensorType>(castOp.getOperand(0).getType()))
            v = castOp.getOperand(0);
        }
        // If still not tensor, find the matching converted type
        if (!isa<RankedTensorType>(v.getType()) &&
            idx < convertedTypes.size()) {
          v = termBuilder
                  .create<UnrealizedConversionCastOp>(yieldOp.getLoc(),
                                                      convertedTypes[idx], v)
                  .getResult(0);
        }
        yieldValues.push_back(v);
      }
      termBuilder.create<stablehlo::ReturnOp>(yieldOp.getLoc(), yieldValues);
      yieldOp.erase();
    }

    whileOp.replaceAllUsesWith(newWhile.getResults());
    whileOp.erase();
  }
}

/// Collapse redundant carrier pairs in stablehlo.while.
///
/// Two upstream passes each create felt-typed carriers for the same logical
/// signal but don't communicate:
///   - SimplifySubComponents.flattenPodArrayWhileCarry flattens pod-array
///     iter-args into per-field felt iter-args. These inits inherit from the
///     original pod-array carrier's `%nondet` (post-eliminatePodDispatch),
///     which lowers to a zero constant. The body is never written: the
///     yield is the literal body argument.
///   - LlzkToStablehlo.promoteArraysToWhileCarry independently captures
///     module-level felt arrays (`array.new` produced by
///     SimplifySubComponents materializing substruct outputs). These end
///     up at separate iter-arg slots with `array.new`-zero init AND a
///     properly threaded yield (writes flow through inner whiles).
///
/// When both fire on the same logical signal — AES `@xor_3$inputs[@a]/[@b]`
/// is the canonical case — we get a pair of iter-args of identical type:
/// one DEAD (yield = body arg unchanged), one LIVE (yield = computed). A
/// downstream reader that bound to the DEAD slot before the captured-array
/// pair existed never gets re-bound, dropping all data flow on that path.
///
/// The fix: at every nesting level of `stablehlo.while`, find slots whose
/// yield is a literal pass-through of the body argument AND whose init is
/// a zero-splat constant. Pair each with a sibling slot of identical type
/// AND identical zero-splat init whose yield is computed. Redirect the
/// DEAD result's external uses to the LIVE result.
///
/// Process bottom-up so inner-level RAUW propagates: an outer slot's yield
/// that referenced an inner DEAD result (e.g., `%57:21` slot 11 yielding
/// `%61#3`) becomes `%61#17` after the inner pass, and the chain repairs
/// itself one level at a time.
///
/// Pairing is slot-order: the first DEAD pairs with the first LIVE of
/// matching type, second with second, etc. This preserves @a/@b ordering
/// when both fields have flatten/capture duplicates.
///
/// Both inits MUST be zero-splat constants. This is the load-bearing safety
/// constraint: it ensures round 0 starts identical for both slots, so RAUW
/// preserves observable semantics for ALL downstream readers regardless of
/// what they expected the DEAD slot to carry. A DEAD slot that yields its
/// body arg is just "constant-init plumbed through" — if init is zero,
/// every iteration yields zero. The LIVE slot also starts at zero.
/// Differing init values would mean the DEAD slot was carrying a
/// meaningful non-zero value (rare, but real for some keccak_pad shapes).
///
/// Idempotency: after RAUW, the DEAD slot has no external uses; a re-run
/// finds no new pairs to collapse.
void collapseRedundantWhileCarrierPairs(ModuleOp module) {
  // Trace transitively through enclosing-while body-arg chains so inner
  // whiles whose inits are outer body args still resolve to the root
  // constant. Without this the inner-level RAUW never fires.
  //
  // Critical safety check: at every enclosing-while level visited, require
  // the slot to be passthrough (yield operand at idx == body arg at idx).
  // Otherwise the inner slot's "every iteration body arg = init = zero"
  // claim (which the RAUW depends on) does not hold — an intermediate while
  // mutating the carrier means later iterations of THIS while see non-zero
  // body args and the dead slot is not actually always-zero. AES xor_2 .b
  // is the canonical violator: zero-init at the main while, but the main
  // while's body computes a non-zero .b yield, so any inner-level
  // passthrough is only momentarily zero on the very first main-while
  // iteration.
  auto isZeroSplatTransitively = [](Value v) -> bool {
    // Bound deeper than any real circuit (AES caps at depth 5 per CLAUDE.md);
    // the cap defends against malformed IR with a body-arg cycle.
    constexpr int kMaxWhileNestingHops = 16;
    for (int hops = 0; hops < kMaxWhileNestingHops; ++hops) {
      if (v.getDefiningOp())
        return isZeroSplatConstant(v);
      auto blockArg = dyn_cast<BlockArgument>(v);
      if (!blockArg)
        return false;
      Operation *parent = blockArg.getOwner()->getParentOp();
      auto parentWhile = dyn_cast_or_null<stablehlo::WhileOp>(parent);
      if (!parentWhile)
        return false;
      unsigned idx = blockArg.getArgNumber();
      if (idx >= parentWhile->getNumOperands())
        return false;
      // Reject if this enclosing while mutates the slot. Body arg comes from
      // the AFTER region; the yield must be the same body arg at the same
      // index for the slot to remain pinned to the init across iterations.
      Block &parentBody = parentWhile.getBody().front();
      auto parentReturn =
          dyn_cast<stablehlo::ReturnOp>(parentBody.getTerminator());
      if (!parentReturn || idx >= parentReturn.getNumOperands() ||
          idx >= parentBody.getNumArguments())
        return false;
      if (parentReturn.getOperand(idx) != parentBody.getArgument(idx))
        return false;
      v = parentWhile->getOperand(idx);
    }
    return false;
  };

  SmallVector<stablehlo::WhileOp> whileOps;
  module.walk([&](stablehlo::WhileOp op) { whileOps.push_back(op); });
  // Process bottom-up: pre-order walk visits parents first, so reversing
  // gives us innermost-first.
  for (auto whileOp : llvm::reverse(whileOps)) {
    Block &body = whileOp.getBody().front();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(body.getTerminator());
    if (!returnOp)
      continue;

    unsigned n = whileOp.getNumResults();
    if (returnOp.getNumOperands() != n || body.getNumArguments() != n)
      continue;

    auto inits = whileOp.getOperands();

    SmallVector<unsigned> deadSlots;
    SmallVector<unsigned> liveSlots;
    for (unsigned i = 0; i < n; ++i) {
      // Only consider tensor-typed slots with zero-splat init — same-type
      // comparison of scalars or non-zero-init slots pairs too aggressively
      // and breaks circuits that intentionally carry a constant value
      // through a while loop (keccak_pad's private-half sentinels).
      if (!isa<RankedTensorType>(whileOp.getResult(i).getType()))
        continue;
      if (!isZeroSplatTransitively(inits[i]))
        continue;
      if (returnOp.getOperand(i) == body.getArgument(i))
        deadSlots.push_back(i);
      else
        liveSlots.push_back(i);
    }

    // Pair a DEAD slot with a LIVE slot only when the LIVE slot's yield
    // transitively references the DEAD slot's body argument. Without that
    // link, the two slots carry semantically unrelated values that merely
    // happen to share the same type and zero-init — RAUW would corrupt the
    // DEAD reader. Canonical false-positive (without the guard): a
    // Poseidon3-style `@compute` whose enclosing scf.while threads a counter
    // (LIVE, init=0, increments) alongside an immutable capacity-init
    // (DEAD passthrough, init=0). Both surface as zero-init `tensor<!pf>`
    // scalars; the redirect would replace the call's capacity operand with
    // the post-loop counter (=N), miscompiling the call.
    //
    // The AES `@xor_3$inputs[@a]/[@b]` case stays linked because the LIVE
    // slot's yield is `xor(.a body arg, .b body arg)` — it consumes the
    // DEAD body arg directly, so the worklist finds the link.
    //
    // Buffer storage is hoisted out of the lambda so the (#dead × #live)
    // pair scan reuses the same SmallVector/SmallPtrSet allocation across
    // calls instead of re-initializing inline storage each time. The
    // `findAncestorOpInBlock` guard skips operand walks for defining ops
    // outside the body block (e.g. function arguments, pre-while
    // constants): SSA dominance prevents those values from reaching
    // `targetArg`, so traversing their operand chains can only waste work.
    Block *bodyBlock = &body;
    llvm::SmallPtrSet<Value, 16> visited;
    SmallVector<Value, 8> worklist;
    auto yieldReferencesArg = [&](Value yieldVal,
                                  BlockArgument targetArg) -> bool {
      visited.clear();
      worklist.clear();
      worklist.push_back(yieldVal);
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        if (!visited.insert(v).second)
          continue;
        if (v == targetArg)
          return true;
        if (Operation *defOp = v.getDefiningOp()) {
          if (bodyBlock->findAncestorOpInBlock(*defOp))
            for (Value operand : defOp->getOperands())
              worklist.push_back(operand);
        }
      }
      return false;
    };

    llvm::SmallSet<unsigned, 8> usedLive;
    for (unsigned dead : deadSlots) {
      if (whileOp.getResult(dead).use_empty())
        continue;
      Type deadTy = whileOp.getResult(dead).getType();
      BlockArgument deadBodyArg = body.getArgument(dead);
      for (unsigned live : liveSlots) {
        if (usedLive.contains(live))
          continue;
        if (whileOp.getResult(live).getType() != deadTy)
          continue;
        if (!yieldReferencesArg(returnOp.getOperand(live), deadBodyArg))
          continue;
        whileOp.getResult(dead).replaceAllUsesWith(whileOp.getResult(live));
        usedLive.insert(live);
        break;
      }
    }
  }
}

/// Vectorize stablehlo.while loops whose iterations are independent.
/// Pattern: while(i < N) { out[i] = f(in[i]); i++ } → out = f(in)
/// Detects element-wise computation chains (dynamic_slice → compute →
/// dynamic_update_slice) and replaces with vectorized element-wise ops.
void vectorizeIndependentWhileLoops(ModuleOp module) {
  SmallVector<stablehlo::WhileOp> whileOps;
  module.walk([&](stablehlo::WhileOp op) { whileOps.push_back(op); });

  for (auto whileOp : whileOps) {
    // --- 1. Check condition: compare(counter, constant_N) ---
    Block &condBlock = whileOp.getCond().front();
    auto returnOp = dyn_cast<stablehlo::ReturnOp>(condBlock.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1)
      continue;
    auto cmpOp = returnOp.getOperand(0).getDefiningOp<stablehlo::CompareOp>();
    if (!cmpOp ||
        cmpOp.getComparisonDirection() != stablehlo::ComparisonDirection::LT)
      continue;

    // --- 2. Check body: find the element-wise computation pattern ---
    Block &bodyBlock = whileOp.getBody().front();

    // Find dynamic_slice ops that read from outer (func) args.
    SmallVector<stablehlo::DynamicSliceOp> sliceOps;
    SmallVector<stablehlo::DynamicUpdateSliceOp> updateOps;
    SmallVector<Operation *> computeOps;

    for (Operation &op : bodyBlock) {
      if (auto slice = dyn_cast<stablehlo::DynamicSliceOp>(&op)) {
        // Check: slicing from an outer value (not a body arg carry)
        Value src = slice.getOperand();
        if (src.getParentBlock() != &bodyBlock)
          sliceOps.push_back(slice);
      } else if (auto update = dyn_cast<stablehlo::DynamicUpdateSliceOp>(&op)) {
        updateOps.push_back(update);
      }
    }

    if (sliceOps.empty() || updateOps.empty())
      continue;

    // --- 3. Check independence: accumulator carry is only written, not read
    // --- Find which body arg is the accumulator (array carry).
    int accArgIdx = -1;
    for (auto [i, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      auto tt = dyn_cast<RankedTensorType>(arg.getType());
      if (!tt || tt.getRank() == 0)
        continue;
      // Check: only used by dynamic_update_slice as the first operand.
      bool onlyUpdateUse = true;
      for (auto *user : arg.getUsers()) {
        if (!isa<stablehlo::DynamicUpdateSliceOp>(user) &&
            !isa<stablehlo::ReturnOp>(user)) {
          onlyUpdateUse = false;
          break;
        }
      }
      if (onlyUpdateUse) {
        accArgIdx = i;
        break;
      }
    }
    if (accArgIdx < 0)
      continue;

    // --- 4. Extract the element-wise computation chain ---
    // Walk from each update_slice backward to find: reshape → compute → reshape
    // → dynamic_slice. Collect the element-wise ops.
    if (updateOps.size() != 1)
      continue; // PoC: single output array
    auto updateOp = updateOps[0];
    Value updateVal = updateOp.getUpdate(); // tensor<1x!pf>

    // Trace backward: reshape → element_op → reshape → dynamic_slice
    auto reshapeToUpdate = updateVal.getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeToUpdate)
      continue;
    Value scalarResult = reshapeToUpdate.getOperand(); // tensor<!pf>

    // Find the element-wise op (multiply, add, etc.)
    Operation *elemOp = scalarResult.getDefiningOp();
    if (!elemOp || elemOp->getNumOperands() < 1 || elemOp->getNumResults() != 1)
      continue;
    if (elemOp->getName().getDialectNamespace() != "stablehlo")
      continue;

    // Trace each operand of the element-wise op back to a dynamic_slice
    SmallVector<Value> outerArrays; // the full input arrays
    bool allFromSlice = true;
    for (Value operand : elemOp->getOperands()) {
      auto reshapeFromSlice = operand.getDefiningOp<stablehlo::ReshapeOp>();
      if (!reshapeFromSlice) {
        allFromSlice = false;
        break;
      }
      auto slice = reshapeFromSlice.getOperand()
                       .getDefiningOp<stablehlo::DynamicSliceOp>();
      if (!slice) {
        allFromSlice = false;
        break;
      }
      outerArrays.push_back(slice.getOperand());
    }
    if (!allFromSlice || outerArrays.empty())
      continue;

    // --- 5. Verify all source arrays have the same shape ---
    auto accType =
        dyn_cast<RankedTensorType>(bodyBlock.getArgument(accArgIdx).getType());
    if (!accType)
      continue;
    bool shapesMatch = true;
    for (Value arr : outerArrays) {
      auto arrType = dyn_cast<RankedTensorType>(arr.getType());
      if (!arrType || arrType.getShape() != accType.getShape()) {
        shapesMatch = false;
        break;
      }
    }
    if (!shapesMatch)
      continue;

    // --- 6. Build vectorized replacement ---
    OpBuilder builder(whileOp);
    Location loc = whileOp.getLoc();

    // Create the vectorized element-wise op on full arrays.
    OperationState state(loc, elemOp->getName());
    for (Value arr : outerArrays)
      state.addOperands(arr);
    state.addTypes({accType});
    // Copy attributes (e.g., comparison_direction)
    for (auto attr : elemOp->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    Operation *vectorizedOp = builder.create(state);

    // Replace uses: the while result at accArgIdx → vectorized result
    whileOp.getResult(accArgIdx).replaceAllUsesWith(vectorizedOp->getResult(0));

    // Replace other while results (counter etc.) with their init values
    auto inits = whileOp.getOperands();
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      if (i != static_cast<unsigned>(accArgIdx) &&
          !whileOp.getResult(i).use_empty())
        whileOp.getResult(i).replaceAllUsesWith(inits[i]);
    }

    whileOp->erase();
  }
}

/// Convert struct.writem, array.write, and array.insert from mutable to SSA
/// form at function-body scope. Each mutation gets a result, and subsequent
/// uses of the target value are updated to use the latest write result.
///
/// `array.insert` coverage is what threads function-body 2D-constant matrix
/// initialization (Poseidon MDS, AES SBox-style tables, any inline 2D literal
/// emitted by circom's `Expression::ArrayInLine` at gen_context.rs:2566 —
/// `array.new <Empty>` + per-row `array.insert`). Without it the inserts are
/// silently dropped: `ArrayInsertPattern::replaceWithDUS` creates a fresh
/// `stablehlo.dynamic_update_slice` for void inserts, but downstream uses of
/// the dest array still reference the original `array.new` result (an empty
/// `dense<0>`), the new DUS has no consumers, and DCE drops it. The
/// canonical victim was Webb `@Mix_69_compute` returning all-zero for any
/// input.
void convertWritemToSSA(ModuleOp module) {
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Block *block) {
      llvm::MapVector<Value, Value> latestValue;

      for (Operation &op : llvm::make_early_inc_range(block->getOperations())) {
        // Rewire operands to latest SSA values
        for (auto &operand : op.getOpOperands()) {
          auto it = latestValue.find(operand.get());
          if (it != latestValue.end())
            operand.set(it->second);
        }

        StringRef opName = op.getName().getStringRef();
        if (opName != "struct.writem" && opName != "array.write" &&
            opName != "array.insert")
          continue;
        // Skip if already converted to SSA (has result type)
        if (op.getNumResults() > 0)
          continue;
        // Skip writes whose target type can't lower to a tensor carry —
        // pod-element arrays are dispatch bookkeeping (handled by `pod.*`
        // patterns). `isPromotableCarryType` is the canonical predicate.
        if (op.getNumOperands() > 0 &&
            !isPromotableCarryType(op.getOperand(0).getType()))
          continue;
        // Skip writes inside scf control flow (for/while/if) — these use
        // mutable LLZK semantics that the per-block walker here cannot
        // honor (the new SSA value would be local to the inner block and
        // never thread out as an scf.yield, leaving the write orphaned and
        // silently dropped during DCE). promoteArraysToWhileCarry +
        // convertWhileBodyArgsToSSA handle the SSA-ification through carries
        // for array.write, struct.writem, and array.insert (the latter via
        // the `includeInsertExtract=true` flag on convertWhileBodyArgsToSSA).
        // struct.writem coverage is what fixes the MiMC7 `@out`-buried-in-
        // scf.if bug — Bug 1 in
        // memory/maci-3-blocked-lowering-bugs-followup.md.
        {
          auto *ancestor = op.getParentOp();
          bool insideScf = false;
          while (ancestor) {
            StringRef an = ancestor->getName().getStringRef();
            if (an == "scf.for" || an == "scf.while" || an == "scf.if") {
              insideScf = true;
              break;
            }
            ancestor = ancestor->getParentOp();
          }
          if (insideScf)
            continue;
        }

        // Convert mutable write to SSA: add result type so the op
        // produces the updated value.
        // Track using the ORIGINAL target (before operand rewiring) so that
        // chained writes all update the same map entry:
        //   writem %self[@a] → latestValue[%self] = %1
        //   writem %self[@b] → rewired to %1, latestValue[%self] = %2
        //   return %self     → rewired to %2 (latest)
        Value target = op.getOperand(0); // already rewired
        // Find the original value this chain started from.
        Value originalTarget = target;
        for (auto &[orig, latest] : latestValue) {
          if (latest == target) {
            originalTarget = orig;
            break;
          }
        }

        OpBuilder b(&op);
        OperationState state(op.getLoc(), opName);
        state.addOperands(op.getOperands());
        state.addTypes({target.getType()});
        for (auto &attr : op.getAttrs())
          state.addAttribute(attr.getName(), attr.getValue());

        Operation *newOp = b.create(state);
        latestValue[originalTarget] = newOp->getResult(0);
        op.erase();
      }
    });
  });
}

} // namespace mlir::llzk_to_shlo
