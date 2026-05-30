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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkToStablehlo.h"

#include <cstdlib>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/ArrayPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FeltPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/FunctionPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LoweringInfrastructure.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PostConversionStructural.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PreConversionStructural.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/RemovalPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructPatterns.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "llzk_to_shlo/Dialect/WLA/WLA.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_LLZKTOSTABLEHLO
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkToStablehlo.h.inc"

namespace {

// ===----------------------------------------------------------------------===
// Prime field parsing
// ===----------------------------------------------------------------------===

struct PrimeAlias {
  const char *name;
  const char *modulus;
  unsigned bitWidth;
};

const PrimeAlias kPrimeAliases[] = {
    // clang-format off
    // NOLINTNEXTLINE(whitespace/line_length)
    {"bn254", "21888242871839275222246405745257275088548364400416034343698204186575808495617", 256},
    // clang-format on
};

std::string resolvePrimeAlias(llvm::StringRef primeStr) {
  for (const auto &alias : kPrimeAliases) {
    if (primeStr.equals_insensitive(alias.name))
      return std::string(alias.modulus) + ":i" + std::to_string(alias.bitWidth);
  }
  return primeStr.str();
}

std::pair<llvm::APInt, unsigned> parsePrimeString(llvm::StringRef primeStr) {
  std::string resolved = resolvePrimeAlias(primeStr);
  auto [valueStr, typeStr] = llvm::StringRef(resolved).split(':');
  unsigned bitWidth = 64;
  if (!typeStr.empty() && typeStr.starts_with("i"))
    bitWidth = std::strtoul(typeStr.drop_front(1).str().c_str(), nullptr, 10);
  return {llvm::APInt(bitWidth, valueStr, 10), bitWidth};
}

// ===----------------------------------------------------------------------===
// Main pass
// ===----------------------------------------------------------------------===

struct LlzkToStablehlo : impl::LlzkToStablehloBase<LlzkToStablehlo> {
  using LlzkToStablehloBase::LlzkToStablehloBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    auto [primeValue, storageBitWidth] = parsePrimeString(prime);
    LlzkToStablehloTypeConverter typeConverter(
        context, primeValue, storageBitWidth, usePrimeFieldType);

    // Conversion target: legal and illegal dialects
    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect, arith::ArithDialect,
                           tensor::TensorDialect, prime_ir::field::FieldDialect,
                           func::FuncDialect, wla::WLADialect>();
    // SCF structural type conversion is added after patterns are created
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    for (StringRef d : {"struct", "function", "constrain", "felt", "component",
                        "bool", "llzk", "cast", "poly"})
      target.addIllegalDialect(d);
    // Array ops on pod-element arrays are dynamically legal (count/dispatch
    // bookkeeping, not convertible to StableHLO). Regular felt-element array
    // ops are illegal (converted by ArrayPatterns). Exception: array.new
    // with a pod element type is illegal so ArrayNewPattern fires — the type
    // converter erases the pod element to felt, producing a zero-init tensor
    // of the converted shape. SSC has already cleaned any pod-side uses, so
    // the array.new's only consumer is an unrealized_conversion_cast that
    // becomes redundant once the new is rewritten to stablehlo.constant.
    target.addDynamicallyLegalDialect(
        [](Operation *op) -> bool {
          // array.new outside @constrain is always handled by the pattern;
          // pod element type doesn't disqualify it from conversion.
          if (op->getName().getStringRef() == "array.new") {
            auto *parent = op->getParentOp();
            while (parent) {
              if (parent->getName().getStringRef() == "function.def") {
                auto sym = parent->getAttrOfType<StringAttr>("sym_name");
                return sym && sym.getValue() == "constrain";
              }
              parent = parent->getParentOp();
            }
            return false;
          }
          for (Value v : op->getOperands()) {
            if (involvesPodType(v.getType()))
              return true;
          }
          for (Value v : op->getResults()) {
            if (involvesPodType(v.getType()))
              return true;
          }
          // Also allow in constrain functions
          auto *parent = op->getParentOp();
          while (parent) {
            if (parent->getName().getStringRef() == "function.def") {
              auto sym = parent->getAttrOfType<StringAttr>("sym_name");
              return sym && sym.getValue() == "constrain";
            }
            parent = parent->getParentOp();
          }
          return false;
        },
        "array");
    // Pod ops: dynamically legal inside constrain functions, scf control
    // flow, or when they are input pods (no @count/@comp/@params — these
    // are dead carries after SimplifySubComponents and will be cleaned up
    // in post-passes).
    target.addDynamicallyLegalDialect(
        [](Operation *op) -> bool {
          // All pod ops are dynamically legal. After SimplifySubComponents
          // extracts function.call and resolves pod.read @comp, remaining
          // pod ops (both input and dispatch) are dead code that gets
          // cleaned up in post-passes.
          if (op->getName().getStringRef() == "pod.new" ||
              op->getName().getStringRef() == "pod.read" ||
              op->getName().getStringRef() == "pod.write") {
            return true;
          }

          auto *parent = op->getParentOp();
          while (parent) {
            StringRef name = parent->getName().getStringRef();
            if (name == "function.def") {
              auto sym = parent->getAttrOfType<StringAttr>("sym_name");
              return sym && sym.getValue() == "constrain";
            }
            if (name == "scf.while" || name == "scf.if" || name == "scf.for")
              return true;
            parent = parent->getParentOp();
          }
          return false;
        },
        "pod");

    // Conversion patterns
    RewritePatternSet patterns(context);
    populateFeltToStablehloPatterns(typeConverter, patterns, target);
    populateStructToStablehloPatterns(typeConverter, patterns, target,
                                      flagOrphanZeroWrites);
    populateArrayToStablehloPatterns(typeConverter, patterns, target);
    populateFunctionToFuncPatterns(typeConverter, patterns, target);
    populateRemovalPatterns(typeConverter, patterns, target);
    addStructuralConversionPatterns(typeConverter, patterns, target);
    // SCF structural type conversion: automatically converts scf.while/if/for
    // result types and block argument types using the type converter.
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    // SSC owns the LLZK v2 prerequisites (input-pod cleanup, while-carry
    // inlining, template shell removal) so external CLI runs and this
    // internal run produce identical post-template IR.
    {
      OpPassManager pm("builtin.module");
      pm.addPass(createSimplifySubComponents());
      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }
    }

    // Pre-pass: hoist dispatch function.call ops from scf.if to parent.
    // In array-of-dispatch-pods pattern, function.call is inside scf.if
    // (count==0 guard), but pod.read @comp is outside. This causes
    // domination errors during conversion. Since circom dispatch always
    // executes the call (count always reaches 0), hoisting is safe.
    // Hoist both the call and its operands (pod.read for inputs).
    {
      SmallVector<scf::IfOp> dispatchIfs;
      module->walk([&](scf::IfOp ifOp) {
        // Check: does this scf.if contain a function.call that writes to
        // pod @comp? That's the dispatch pattern.
        bool isDispatch = false;
        ifOp.getThenRegion().walk([&](Operation *inner) {
          if (inner->getName().getStringRef() == "function.call") {
            for (auto *user : inner->getResult(0).getUsers()) {
              if (user->getName().getStringRef() == "pod.write") {
                auto rn = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
                if (rn && rn.getValue() == "comp")
                  isDispatch = true;
              }
            }
          }
        });
        if (isDispatch)
          dispatchIfs.push_back(ifOp);
      });

      for (auto ifOp : dispatchIfs) {
        // Hoist all ops from the then block to before the scf.if.
        // This includes: pod.read (input), function.call, pod.write @comp,
        // array.write (dispatch array update).
        Block &thenBlock = ifOp.getThenRegion().front();
        SmallVector<Operation *> toHoist;
        for (Operation &op : thenBlock) {
          if (!op.hasTrait<OpTrait::IsTerminator>())
            toHoist.push_back(&op);
        }
        for (auto *op : toHoist)
          op->moveBefore(ifOp);
      }
    }

    // Pre-pass: replace llzk.nondet : i1 with arith.constant false.
    // Keeps the i1 type intact so bool.not, scf.if, etc. still work.
    // The LlzkNonDetPattern in RemovalPatterns handles the tensor<i1>
    // wrapping during dialect conversion if needed.
    {
      SmallVector<Operation *> i1Nondets;
      module->walk([&](Operation *op) {
        if (op->getName().getStringRef() == "llzk.nondet" &&
            op->getNumResults() == 1 && op->getResult(0).getType().isInteger(1))
          i1Nondets.push_back(op);
      });
      for (auto *op : i1Nondets) {
        OpBuilder builder(op);
        auto falseVal = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getBoolAttr(false));
        op->getResult(0).replaceAllUsesWith(falseVal);
        op->erase();
      }

      // Also mark arith dialect as legal so arith.constant i1 survives.
      // The scf-to-stablehlo post-pass handles the i1→tensor<i1> wrapping.
    }

    // Pre-passes: transform LLZK IR before dialect conversion.
    //
    // Three structural patches run before the carrier-promotion and
    // SSA-ification passes. Each filters narrowly on a specific LLZK IR
    // shape; circuits whose IR doesn't match are no-ops by construction.
    //   1. replaceHoistedReadmWithFreshCall — hoisted-call staleness
    //      rescue: rewires each `struct.readm @out` whose `function.call`
    //      was hoisted to an ancestor region of the readm to a freshly
    //      emitted call on the current (post-write) row at the readm site,
    //      so the call sees a fully-populated row at the LAST inner iter.
    //   2. cloneZeroIdxCarrierWriteIntoOutermostFeltZeroThen — restores the
    //      OUTERMOST THEN K=0 carrier-writer that SSC drops when the K=0
    //      trigger sits in an `scf.if(eq %felt_counter, 0)` THEN without an
    //      `scf.execute_region` wrapper, by cloning the structurally-dead
    //      ELSE-region writer for the same `mSoPCF.carrier-for` carrier.
    //   3. fillCallRowCol0FromInBlockScalar — fills col 0 of a
    //      `function.call`'s input row when the row comes from result
    //      index 1 of a 3-result `scf.while` whose body never wrote col 0,
    //      using the most-recent in-block scratchpad scalar read.
    // Steps 1+2 run before promoteArraysToWhileCarry. Step 3 must run AFTER
    // it because the matched 3-result scf.while only exists post-promotion;
    // the injected array.write is SSA-ified by the following
    // convertWhileBodyArgsToSSA.
    registerStructFieldOffsets(module, typeConverter);
    convertAllFunctions(module, typeConverter, context);
    replaceHoistedReadmWithFreshCall(module);
    cloneZeroIdxCarrierWriteIntoOutermostFeltZeroThen(module);
    promoteArraysToWhileCarry(module);
    fillCallRowCol0FromInBlockScalar(module);
    convertWhileBodyArgsToSSA(module);

    convertWritemToSSA(module);

    // Strip llzk.* module attributes AFTER function conversion
    // (isMainStruct reads llzk.main during convertAllFunctions)
    SmallVector<StringRef> attrsToRemove;
    for (auto attr : module->getAttrs())
      if (attr.getName().getValue().starts_with("llzk."))
        attrsToRemove.push_back(attr.getName().getValue());
    for (auto name : attrsToRemove)
      module->removeAttr(name);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Post-pass: convert scf.while → stablehlo.while
    convertScfWhileToStablehloWhile(module);

    // Pre-pass: fold `arith.cmpi` on two integer/index constants and the
    // `scf.if` op that consumes the result. Circom emits a constant
    // `min(N1, N2)` for trip-count guards as
    //     %c1 = arith.constant N1 : index
    //     %c2 = arith.constant N2 : index
    //     %p  = arith.cmpi ult, %c1, %c2 : index
    //     %r  = scf.if %p -> (index) { yield %c1 } else { yield %c2 }
    // (PointCompress's `long_div_*` uses this for `min(100, 200)`.)
    // Without folding, the scf.if reaches the stablehlo.select post-pass
    // below with `tensor<index>` operands — a type SelectOp does not
    // accept — and verification fails.
    {
      SmallVector<arith::CmpIOp> cmpiToFold;
      module.walk([&](arith::CmpIOp op) {
        if (op.getLhs().getDefiningOp<arith::ConstantOp>() &&
            op.getRhs().getDefiningOp<arith::ConstantOp>())
          cmpiToFold.push_back(op);
      });
      for (auto cmpi : cmpiToFold) {
        auto lhsAttr = dyn_cast<IntegerAttr>(
            cmpi.getLhs().getDefiningOp<arith::ConstantOp>().getValue());
        auto rhsAttr = dyn_cast<IntegerAttr>(
            cmpi.getRhs().getDefiningOp<arith::ConstantOp>().getValue());
        if (!lhsAttr || !rhsAttr)
          continue;
        const APInt &l = lhsAttr.getValue();
        const APInt &r = rhsAttr.getValue();
        // arith.cmpi verifier requires LHS/RHS operand types to match, and
        // IntegerAttr verifier requires APInt width to equal the integer
        // type's width — so widths are always equal here. Asserting the
        // invariant documents it without silently miscomputing on the
        // verifier-impossible mismatch case (sign-extending an i1 `true`
        // to wider yields UINT_MAX, which would flip unsigned predicates).
        assert(l.getBitWidth() == r.getBitWidth() &&
               "arith.cmpi operands have mismatched APInt widths");
        bool result;
        switch (cmpi.getPredicate()) {
        case arith::CmpIPredicate::eq:
          result = l == r;
          break;
        case arith::CmpIPredicate::ne:
          result = l != r;
          break;
        case arith::CmpIPredicate::slt:
          result = l.slt(r);
          break;
        case arith::CmpIPredicate::sle:
          result = l.sle(r);
          break;
        case arith::CmpIPredicate::sgt:
          result = l.sgt(r);
          break;
        case arith::CmpIPredicate::sge:
          result = l.sge(r);
          break;
        case arith::CmpIPredicate::ult:
          result = l.ult(r);
          break;
        case arith::CmpIPredicate::ule:
          result = l.ule(r);
          break;
        case arith::CmpIPredicate::ugt:
          result = l.ugt(r);
          break;
        case arith::CmpIPredicate::uge:
          result = l.uge(r);
          break;
        default:
          llvm_unreachable("unhandled CmpIPredicate");
        }
        OpBuilder b(cmpi);
        auto cstBool =
            b.create<arith::ConstantOp>(cmpi.getLoc(), b.getBoolAttr(result));
        cmpi.getResult().replaceAllUsesWith(cstBool);
        cmpi.erase();
      }

      SmallVector<scf::IfOp> ifToFold;
      module.walk([&](scf::IfOp ifOp) {
        // Void scf.if is handled by the existing post-pass below (it hoists
        // func.calls then erases). Folding it here is both redundant and
        // unsafe: a void scf.if may legally omit the else region, and the
        // chosen-block lookup below would dereference an empty Region.
        if (ifOp.getNumResults() == 0)
          return;
        auto cstOp = ifOp.getCondition().getDefiningOp<arith::ConstantOp>();
        if (!cstOp)
          return;
        auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue());
        if (!intAttr || !intAttr.getType().isInteger(1))
          return;
        ifToFold.push_back(ifOp);
      });
      for (auto ifOp : ifToFold) {
        auto cstOp = ifOp.getCondition().getDefiningOp<arith::ConstantOp>();
        auto intAttr = cast<IntegerAttr>(cstOp.getValue());
        bool condVal = intAttr.getValue().getBoolValue();
        Block &chosen = condVal ? ifOp.getThenRegion().front()
                                : ifOp.getElseRegion().front();
        auto yield = cast<scf::YieldOp>(chosen.getTerminator());
        // The un-chosen branch's ops are unreachable under scf.if semantics,
        // so dropping them with `ifOp.erase()` below is correct even if they
        // contain side effects.
        SmallVector<Value> yielded(yield.getOperands());
        for (Operation &op : llvm::make_early_inc_range(chosen)) {
          if (&op == yield.getOperation())
            continue;
          op.moveBefore(ifOp);
        }
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
          ifOp.getResult(i).replaceAllUsesWith(yielded[i]);
        ifOp.erase();
      }
    }

    // Post-pass: convert remaining scf.if → stablehlo.select (simple cases)
    // or stablehlo.case (complex cases). SCF structural type conversion
    // handles scf.if type changes, but the ops themselves remain as scf.if.
    // The HLO export does not support scf dialect.
    {
      SmallVector<scf::IfOp> ifOps;
      module.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });
      for (auto ifOp : ifOps) {
        // Void scf.if is dead code (count/dispatch bookkeeping). Erase it.
        // First, move any func.call ops out of the scf.if to preserve them
        // (they were extracted from dispatch but may have been moved back by
        // the SCF structural type conversion).
        if (ifOp.getNumResults() == 0) {
          // Move func.call and its in-block dependencies out of scf.if.
          for (Region &r : ifOp->getRegions()) {
            for (Block &b : r) {
              SmallVector<Operation *> toMove;
              for (Operation &op : b)
                if (isa<func::CallOp>(&op))
                  toMove.push_back(&op);
              for (auto *callOp : toMove) {
                // Collect operand-defining ops in this block.
                llvm::DenseSet<Operation *> deps;
                std::function<void(Value)> collectDeps = [&](Value v) {
                  auto *def = v.getDefiningOp();
                  if (!def || def->getBlock() != &b || deps.count(def))
                    return;
                  deps.insert(def);
                  for (Value operand : def->getOperands())
                    collectDeps(operand);
                };
                for (Value operand : callOp->getOperands())
                  collectDeps(operand);
                // Move deps before scf.if (in block order).
                for (Operation &dep : llvm::make_early_inc_range(b))
                  if (deps.count(&dep))
                    dep.moveBefore(ifOp);
                callOp->moveBefore(ifOp);
              }
            }
          }
          for (Region &r : ifOp->getRegions())
            r.dropAllReferences();
          for (Region &r : ifOp->getRegions())
            r.getBlocks().clear();
          ifOp.erase();
          continue;
        }

        // Ensure predicate is tensor<i1>.
        Value pred = ifOp.getCondition();
        // Look through unrealized_conversion_cast.
        if (!isa<RankedTensorType>(pred.getType())) {
          if (auto castOp = pred.getDefiningOp<UnrealizedConversionCastOp>()) {
            Value src = castOp.getInputs()[0];
            if (isa<RankedTensorType>(src.getType()))
              pred = src;
          }
        }
        // Look through arith.ori/andi: convert to stablehlo.or/and on tensor.
        if (!isa<RankedTensorType>(pred.getType())) {
          if (auto *defOp = pred.getDefiningOp()) {
            if (isa<arith::OrIOp>(defOp) || isa<arith::AndIOp>(defOp)) {
              Value lhs = lookThroughCast(defOp->getOperand(0));
              Value rhs = lookThroughCast(defOp->getOperand(1));
              if (isa<RankedTensorType>(lhs.getType()) &&
                  isa<RankedTensorType>(rhs.getType())) {
                OpBuilder b(defOp);
                if (isa<arith::OrIOp>(defOp))
                  pred = b.create<stablehlo::OrOp>(defOp->getLoc(), lhs, rhs);
                else
                  pred = b.create<stablehlo::AndOp>(defOp->getLoc(), lhs, rhs);
              }
            }
          }
        }
        // Wrap scalar i1 predicate into tensor<i1>.
        if (!isa<RankedTensorType>(pred.getType()) &&
            pred.getType().isInteger(1)) {
          OpBuilder b(ifOp);
          auto tensorType = RankedTensorType::get({}, b.getI1Type());
          pred =
              b.create<tensor::FromElementsOp>(ifOp.getLoc(), tensorType, pred);
        }
        if (!isa<RankedTensorType>(pred.getType()))
          continue;

        // For each result: compute both branches' values, then select.
        // This inlines both branches before the scf.if, then uses
        // stablehlo.select to pick the result.
        OpBuilder builder(ifOp);
        Location loc = ifOp.getLoc();

        // Inline both branches' ops before the scf.if.
        Block &thenBlock = ifOp.getThenRegion().front();
        Block &elseBlock = ifOp.getElseRegion().front();

        // Get yield values from both branches.
        auto thenYield = cast<scf::YieldOp>(thenBlock.getTerminator());
        auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());

        // Inline then block ops (before yield) into parent.
        for (Operation &op : llvm::make_early_inc_range(thenBlock)) {
          if (&op == thenYield.getOperation())
            continue;
          op.moveBefore(ifOp);
        }
        // Inline else block ops (before yield) into parent.
        for (Operation &op : llvm::make_early_inc_range(elseBlock)) {
          if (&op == elseYield.getOperation())
            continue;
          op.moveBefore(ifOp);
        }

        // Create stablehlo.select for each result.
        SmallVector<Value> selectResults;
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
          Value thenVal = thenYield.getOperand(i);
          Value elseVal = elseYield.getOperand(i);

          // Wrap bare scalars (i1) in tensor<> for stablehlo.select.
          auto resultType = ifOp.getResult(i).getType();
          if (!isa<RankedTensorType>(resultType) &&
              resultType.isIntOrIndexOrFloat()) {
            auto tensorType = RankedTensorType::get({}, resultType);
            thenVal = builder.create<tensor::FromElementsOp>(loc, tensorType,
                                                             thenVal);
            elseVal = builder.create<tensor::FromElementsOp>(loc, tensorType,
                                                             elseVal);
            resultType = tensorType;
          }

          // Broadcast predicate if result is not scalar.
          Value broadPred = pred;
          if (auto tt = dyn_cast<RankedTensorType>(resultType)) {
            if (tt.getRank() > 0) {
              auto predType =
                  RankedTensorType::get(tt.getShape(), builder.getI1Type());
              broadPred = builder.create<stablehlo::BroadcastInDimOp>(
                  loc, predType, pred, builder.getDenseI64ArrayAttr({}));
            }
          }
          selectResults.push_back(builder.create<stablehlo::SelectOp>(
              loc, resultType, broadPred, thenVal, elseVal));
        }

        // Replace scf.if results with select results.
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
          ifOp.getResult(i).replaceAllUsesWith(selectResults[i]);
        ifOp.erase();
      }
    }

    // Post-pass: reconnect func.call results to pod.read @comp consumers.
    // After conversion + scf.if extraction, func.call ops may be in the same
    // block or in nested while bodies. Replace the pod.read @comp →
    // unrealized_cast chain with the matching func.call result.
    {
      module.walk([&](stablehlo::WhileOp whileOp) {
        for (Block &b : whileOp.getBody()) {
          // Collect func.call ops from this block only. Calls in nested
          // regions can't be used directly (domination). Those cases are
          // handled by the LLZK cleanup pass below.
          SmallVector<func::CallOp> calls;
          for (Operation &op : b)
            if (auto call = dyn_cast<func::CallOp>(&op))
              calls.push_back(call);
          if (calls.empty())
            continue;

          SmallVector<Operation *> toErase;
          for (Operation &op : b) {
            auto castOp = dyn_cast<UnrealizedConversionCastOp>(&op);
            if (!castOp || castOp.getNumOperands() != 1)
              continue;
            auto *srcOp = castOp.getInputs()[0].getDefiningOp();
            if (!srcOp || srcOp->getName().getStringRef() != "pod.read")
              continue;
            auto rn = srcOp->getAttrOfType<FlatSymbolRefAttr>("record_name");
            if (!rn || rn.getValue() != "comp")
              continue;
            Type targetType = castOp.getResult(0).getType();
            Value replacement;
            for (auto call : calls)
              if (call.getNumResults() > 0 &&
                  call.getResult(0).getType() == targetType)
                replacement = call.getResult(0);
            if (replacement) {
              castOp.getResult(0).replaceAllUsesWith(replacement);
              toErase.push_back(castOp.getOperation());
              toErase.push_back(srcOp);
              if (auto *arrOp = srcOp->getOperand(0).getDefiningOp())
                if (arrOp->getName().getStringRef() == "array.read")
                  toErase.push_back(arrOp);
            }
          }
          for (auto *op : llvm::reverse(toErase))
            if (op->use_empty())
              op->erase();
          // Erase array.new if all its uses are gone.
          for (Operation &op : llvm::make_early_inc_range(b))
            if (op.getName().getStringRef() == "array.new" && op.use_empty())
              op.erase();
        }
      });
    }

    // Post-pass: clean up remaining LLZK dialect ops (array.read, pod.read,
    // array.new). These are remnants of pod dispatch bookkeeping. Replace
    // unrealized_conversion_cast chains that source from pod.read with zero
    // tensors, then erase dead LLZK ops bottom-up.
    {
      SmallVector<Operation *> toErase;
      module.walk([&](UnrealizedConversionCastOp castOp) {
        if (castOp.getNumOperands() != 1)
          return;
        auto *srcOp = castOp.getInputs()[0].getDefiningOp();
        if (!srcOp)
          return;
        StringRef srcName = srcOp->getName().getStringRef();
        if (srcName != "pod.read" && srcName != "array.read")
          return;
        auto resultType =
            dyn_cast<RankedTensorType>(castOp.getResult(0).getType());
        if (!resultType)
          return;
        // Replace with zero constant
        OpBuilder b(castOp);
        auto zeroAttr = b.getZeroAttr(resultType);
        auto zero = b.create<stablehlo::ConstantOp>(castOp.getLoc(), resultType,
                                                    zeroAttr);
        castOp.getResult(0).replaceAllUsesWith(zero);
        toErase.push_back(castOp);
      });
      for (auto *op : toErase)
        op->erase();
      // Bottom-up erase dead LLZK ops
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](Operation *op) {
          StringRef ns = op->getName().getDialectNamespace();
          if ((ns == "pod" || ns == "array" || ns == "llzk") &&
              op->use_empty()) {
            op->erase();
            erased = true;
          }
        });
      }
    }

    // Post-pass: scrub residual `arith.*` fan-out emitted by `array.write
    // %carrier[%dyn_idx] = %v` scaffolding inside scf.while bodies when
    // the dynamic index is felt-typed. After convertScfWhile..., the
    // fan-out sits inside stablehlo.while bodies and looks like
    //   %idx = unrealized_conversion_cast %iter_i32 : tensor<i32> to index
    //   %true = arith.constant true
    //   %true_t = unrealized_conversion_cast %true : i1 to tensor<i1>
    //   %cN = arith.constant N : index
    //   %cmp = arith.cmpi eq, %idx, %cN : index
    //   %cmp_t = unrealized_conversion_cast %cmp : i1 to tensor<i1>
    //   %flag = arith.andi %cmp_t, %true_t : tensor<i1>
    //   ... %sel = stablehlo.select %flag, %v, %prev
    // The conversion target keeps arith::ArithDialect legal so partial
    // conversion intentionally leaves these behind, but the ZKX HLO
    // translator rejects every surviving arith.* op. Rewrite cmpi onto
    // the underlying tensor<i32> via stablehlo.compare EQ, and convert
    // the `arith.constant true|false : i1` carrier feeding the
    // `i1 → tensor<i1>` cast to a `stablehlo.constant`. The existing
    // arith.andi → stablehlo.and rewrite below then picks up the residue
    // (its operands now trace straight to tensor<i1> producers, satisfying
    // the lookThroughCast tensor guard at the next pass).
    {
      auto i32TensorType =
          RankedTensorType::get({}, IntegerType::get(context, 32));
      auto i1TensorType =
          RankedTensorType::get({}, IntegerType::get(context, 1));
      auto getI32Source = [](Value v) -> Value {
        auto cast = v.getDefiningOp<UnrealizedConversionCastOp>();
        if (!cast || cast.getInputs().size() != 1)
          return nullptr;
        Value src = cast.getInputs()[0];
        auto tty = dyn_cast<RankedTensorType>(src.getType());
        if (tty && tty.getElementType().isInteger(32) && tty.getRank() == 0)
          return src;
        return nullptr;
      };

      // Step A: arith.cmpi eq : index → stablehlo.compare EQ on tensor<i32>.
      SmallVector<arith::CmpIOp> cmpiOps;
      module.walk([&](arith::CmpIOp op) {
        if (op.getPredicate() != arith::CmpIPredicate::eq)
          return;
        if (!op.getLhs().getType().isIndex() ||
            !op.getRhs().getType().isIndex())
          return;
        cmpiOps.push_back(op);
      });
      for (auto cmpi : cmpiOps) {
        OpBuilder b(cmpi);
        auto materializeOperand = [&](Value v) -> Value {
          if (Value src = getI32Source(v))
            return src;
          if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
            auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
            if (!intAttr)
              return nullptr;
            return b.create<stablehlo::ConstantOp>(
                cmpi.getLoc(), i32TensorType,
                DenseElementsAttr::get(i32TensorType,
                                       b.getI32IntegerAttr(static_cast<int32_t>(
                                           intAttr.getInt()))));
          }
          return nullptr;
        };
        Value lhs = materializeOperand(cmpi.getLhs());
        Value rhs = materializeOperand(cmpi.getRhs());
        if (!lhs || !rhs)
          continue;
        auto cmpResult = b.create<stablehlo::CompareOp>(
            cmpi.getLoc(), i1TensorType, lhs, rhs,
            stablehlo::ComparisonDirectionAttr::get(
                cmpi.getContext(), stablehlo::ComparisonDirection::EQ));
        // Re-route consumers that cast the i1 scalar to tensor<i1>.
        SmallVector<UnrealizedConversionCastOp> dyingCasts;
        for (auto *user : cmpi.getResult().getUsers())
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
            if (cast.getNumResults() == 1 &&
                cast.getResult(0).getType() == i1TensorType)
              dyingCasts.push_back(cast);
        for (auto cast : dyingCasts) {
          cast.getResult(0).replaceAllUsesWith(cmpResult);
          cast.erase();
        }
        if (cmpi.getResult().use_empty())
          cmpi.erase();
      }

      // Step B: arith.constant true|false : i1 → stablehlo.constant
      // tensor<i1>, RAUW'ing its `i1 → tensor<i1>` cast consumers. The
      // arith.constant itself is left in place if other (e.g. scf.if)
      // consumers remain; otherwise the trailing DCE sweep clears it.
      // Single replacement constant per source: inserted directly after
      // `c` so it dominates every cast consumer regardless of use-list
      // ordering.
      SmallVector<arith::ConstantOp> i1Constants;
      module.walk([&](arith::ConstantOp op) {
        if (op.getType().isInteger(1))
          i1Constants.push_back(op);
      });
      for (auto c : i1Constants) {
        auto intAttr = dyn_cast<IntegerAttr>(c.getValue());
        if (!intAttr)
          continue;
        bool boolVal = intAttr.getInt() != 0;
        SmallVector<UnrealizedConversionCastOp> dyingCasts;
        for (auto *user : c.getResult().getUsers()) {
          auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
          if (!cast || cast.getNumResults() != 1)
            continue;
          auto rty = dyn_cast<RankedTensorType>(cast.getResult(0).getType());
          if (rty && rty.getElementType().isInteger(1) && rty.getRank() == 0)
            dyingCasts.push_back(cast);
        }
        if (dyingCasts.empty())
          continue;
        OpBuilder b(c);
        b.setInsertionPointAfter(c);
        auto cst = b.create<stablehlo::ConstantOp>(
            c.getLoc(), i1TensorType,
            DenseElementsAttr::get(i1TensorType, boolVal));
        for (auto cast : dyingCasts) {
          cast.getResult(0).replaceAllUsesWith(cst);
          cast.erase();
        }
        if (c.getResult().use_empty())
          c.erase();
      }
    }

    // Post-pass: convert remaining arith.ori/andi(i1) → stablehlo.or/and.
    // Only replace uses that accept tensor types. scf.if conditions still
    // need i1, so we keep the arith op for those and add a
    // unrealized_conversion_cast bridge.
    {
      SmallVector<Operation *> arithBoolOps;
      module.walk([&](Operation *op) {
        if (isa<arith::OrIOp>(op) || isa<arith::AndIOp>(op))
          arithBoolOps.push_back(op);
      });
      for (auto *op : arithBoolOps) {
        Value lhs = lookThroughCast(op->getOperand(0));
        Value rhs = lookThroughCast(op->getOperand(1));
        if (!isa<RankedTensorType>(lhs.getType()) ||
            !isa<RankedTensorType>(rhs.getType()))
          continue;
        OpBuilder b(op);
        Operation *replacement = nullptr;
        if (isa<arith::OrIOp>(op))
          replacement = b.create<stablehlo::OrOp>(op->getLoc(), lhs, rhs);
        else
          replacement = b.create<stablehlo::AndOp>(op->getLoc(), lhs, rhs);
        // Replace only non-scf.if uses with the tensor result.
        // For scf.if conditions, keep the original i1 value.
        Value tensorResult = replacement->getResult(0);
        SmallVector<OpOperand *> usesToReplace;
        for (OpOperand &use : op->getResult(0).getUses()) {
          if (!isa<scf::IfOp>(use.getOwner()))
            usesToReplace.push_back(&use);
        }
        for (auto *use : usesToReplace)
          use->set(tensorResult);
        // If no more uses, erase the arith op.
        if (op->getResult(0).use_empty())
          op->erase();
      }
    }

    // Erase non-StableHLO dead code inside stablehlo.while bodies.
    // Count/dispatch pod array ops (array.read, pod.read/write, arith.subi,
    // arith.cmpi, unrealized_cast) remain from dynamically-legal pod ops.
    // Iteratively erase ops with no uses from non-stablehlo dialects.
    {
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](stablehlo::WhileOp whileOp) {
          for (Region &r : whileOp->getRegions()) {
            for (Block &b : r) {
              for (Operation &op :
                   llvm::make_early_inc_range(llvm::reverse(b))) {
                StringRef ns = op.getName().getDialectNamespace();
                if (ns == "stablehlo" || ns == "func" || ns == "builtin")
                  continue;
                // Check if all results are unused.
                bool allUnused = llvm::all_of(
                    op.getResults(), [](Value v) { return v.use_empty(); });
                if (allUnused && op.getNumResults() > 0) {
                  op.erase();
                  erased = true;
                }
                // Void ops (pod.write, array.write) with no results.
                if (op.getNumResults() == 0 && ns != "stablehlo") {
                  op.dropAllReferences();
                  op.erase();
                  erased = true;
                }
              }
            }
          }
        });
      }
    }

    // Erase dead unrealized_conversion_casts.
    {
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](UnrealizedConversionCastOp castOp) {
          if (castOp->use_empty()) {
            castOp.erase();
            erased = true;
          }
        });
      }
    }

    // Post-pass: inline survivor scf.execute_region wrappers whose body
    // is a single block ending in scf.yield. LLZK's `<--` returning a
    // felt (e.g. the 67-way Ark-constant lookup in Poseidon-using chips
    // like webb's vAnchor) emits a felt-array dispatch cascade wrapped
    // in scf.execute_region. Main conversion lowers the inner ops to
    // stablehlo but no pattern dissolves the wrapper; survivors then
    // block stablehlo_runner's module load with "Dialect `scf' not
    // found for custom op 'scf.execute_region'". Inlining a single-
    // block single-yield body is semantically a no-op: the region
    // executes once unconditionally, and after splicing the inner ops
    // run in the parent block at the same position. The resulting
    // felt→tensor round-trip cast pair (yield-side cast + consumer-side
    // cast) is cleaned up by the ReconcileUnrealizedCasts pass below.
    //
    // PostOrder walk so nested wrappers inline inner-first. The
    // single-block guard is load-bearing: scf.execute_region's region
    // is `AnyRegion` and multi-block bodies are legal (branching to
    // distinct terminators via cf.br), but inlining them would require
    // hoisting cf.* control flow into the parent block, which is out
    // of scope here.
    module.walk<WalkOrder::PostOrder>([&](scf::ExecuteRegionOp er) {
      if (!er.getRegion().hasOneBlock())
        return;
      Block &body = er.getRegion().front();
      auto yield = cast<scf::YieldOp>(body.getTerminator());
      for (auto [res, val] : llvm::zip(er.getResults(), yield.getOperands()))
        res.replaceAllUsesWith(val);
      Block *parent = er->getBlock();
      parent->getOperations().splice(er->getIterator(), body.getOperations(),
                                     body.begin(), yield->getIterator());
      er.erase();
    });

    // Reconcile leftover round-trip unrealized_conversion_cast pairs
    // (A → X → A). These appear when a pattern reads an
    // scf-structurally-converted block arg through cast.toindex; without
    // reconciliation the felt-typed cast intermediates survive and
    // downstream tools that don't load the felt dialect cannot parse the
    // result. Skip when no live cast remains — most circuits without
    // while-bodied iter-args fall through here.
    {
      bool hasLiveCast = false;
      module.walk([&](UnrealizedConversionCastOp) {
        hasLiveCast = true;
        return WalkResult::interrupt();
      });
      if (hasLiveCast) {
        OpPassManager pm("builtin.module");
        pm.addPass(createReconcileUnrealizedCastsPass());
        if (failed(runPipeline(pm, module))) {
          signalPassFailure();
          return;
        }
      }
    }

    // Clean up all dead non-stablehlo ops (unrealized_cast, arith, tensor,
    // array, pod, scf, struct, builtin dead casts). `wla.layout` (no
    // results, `Pure`) flows through this loop too. The anchor pass
    // (`--witness-layout-anchor`, TRACK 3) currently over-emits internal
    // entries that the lowering elides from `@main`'s DUS chain (struct-
    // typed sub-component members in particular), so connecting the
    // anchor → verify chain (preserving `wla.layout` past this DCE +
    // erasing in `--verify-witness-layout`) is gated on a filter
    // refinement that aligns the anchor's emit with the lowering's
    // actual per-chunk emission.
    bool erased = true;
    while (erased) {
      erased = false;
      module.walk([&](Operation *op) {
        StringRef ns = op->getName().getDialectNamespace();
        if (ns == "stablehlo" || ns == "func" || ns == "builtin")
          return;
        // Void ops with no results: erase if non-stablehlo.
        if (op->getNumResults() == 0 &&
            (ns == "pod" || ns == "array" || ns == "struct")) {
          op->dropAllReferences();
          op->erase();
          erased = true;
          return;
        }
        // `use_empty` is trivially true for zero-result ops (terminators
        // like scf.yield / scf.condition / func.return). Skipping the
        // terminator trait keeps the block well-formed — without the
        // guard, a survivor scf.execute_region whose body ends in
        // scf.yield (e.g. the bn-typed felt-array dispatch cascade in
        // chips with `<--` returning a felt array) loses its terminator
        // here and the next verifier fires "empty block: expect at
        // least a terminator". stablehlo-namespace terminators were
        // already filtered above, so this guard only fires for
        // surviving non-stablehlo terminators.
        if (op->use_empty() && !op->hasTrait<OpTrait::IsTerminator>()) {
          op->erase();
          erased = true;
        }
      });
    }

    // Post-pass: vectorize independent while loops.
    // Detects while loops where each iteration independently computes
    // output[i] = f(input[i]) and replaces with vectorized element-wise
    // ops. This enables GPU-parallel witness generation.
    vectorizeIndependentWhileLoops(module);

    // Post-pass: collapse redundant carrier pairs left when
    // flattenPodArrayWhileCarry and promoteArraysToWhileCarry independently
    // create per-field carriers for the same logical signal.
    collapseRedundantWhileCarrierPairs(module);

    // Dead code elimination: remove unused constants and ops left over
    // from vectorization (e.g., dead constants from erased while loops).
    {
      bool erased = true;
      while (erased) {
        erased = false;
        module.walk([&](Operation *op) {
          if (isa<func::FuncOp>(op) || isa<func::ReturnOp>(op) ||
              isa<ModuleOp>(op))
            return;
          if (op->getNumResults() > 0 &&
              llvm::all_of(op->getResults(),
                           [](Value v) { return v.use_empty(); })) {
            op->erase();
            erased = true;
          }
        });
      }
    }
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
