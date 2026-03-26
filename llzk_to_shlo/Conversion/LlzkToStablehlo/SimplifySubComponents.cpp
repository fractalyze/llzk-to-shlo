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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"

#include "llvm/ADT/StringMap.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_SIMPLIFYSUBCOMPONENTS
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h.inc"

namespace {

/// Phase 1: Scan block, track pod field values and extract function.call
/// from scf.if into the parent block.
/// Returns true if any changes were made.
bool extractCallsFromScfIf(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues) {
  bool changed = false;

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
    } else if (name == "pod.write") {
      auto field = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (field && op.getNumOperands() >= 2)
        trackedPodValues[op.getOperand(0)][field.getValue()] = op.getOperand(1);
    } else if (name == "scf.if") {
      // Extract function.call @compute from inside scf.if.
      // Build a new call BEFORE the scf.if using pod-tracked inputs.
      SymbolRefAttr calleeRef = nullptr;
      SmallVector<Type> resultTypes;
      // Track (pod_value, field_name) pairs for each call arg
      SmallVector<std::pair<Value, StringRef>> inputPodFields;
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
                // Get the SPECIFIC pod this reads from
                Value srcPod =
                    def->getNumOperands() > 0 ? def->getOperand(0) : Value();
                if (rn && srcPod)
                  inputPodFields.push_back({srcPod, rn.getValue()});
              }
            }
          }
        }
        if (nn == "pod.write") {
          auto fn = nested->getAttrOfType<FlatSymbolRefAttr>("record_name");
          if (fn && fn.getValue() == "comp" && nested->getNumOperands() >= 1)
            compPod = nested->getOperand(0);
        }
      });

      if (calleeRef && !resultTypes.empty()) {
        // Collect arguments from the SPECIFIC pods tracked
        SmallVector<Value> args;
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

        if (args.size() == inputPodFields.size()) {
          // Create the direct function.call before scf.if
          // Find the original function.call inside scf.if to clone
          // its properties (operandSegmentSizes, numDimsPerMap, etc.)
          Operation *origCall = nullptr;
          op.walk([&](Operation *n) {
            if (n->getName().getStringRef() == "function.call")
              origCall = n;
          });

          OpBuilder builder(&op);
          // Clone the original call op, replacing operands and callee
          OperationState state(op.getLoc(), origCall->getName());
          state.addOperands(args);
          state.addTypes(resultTypes);
          // Copy all attributes, updating callee
          for (auto &attr : origCall->getAttrs()) {
            if (attr.getName() == "callee")
              state.addAttribute("callee", calleeRef);
            else
              state.addAttribute(attr.getName(), attr.getValue());
          }
          // Copy properties (includes operandSegmentSizes)
          if (origCall->getPropertiesStorage()) {
            state.propertiesAttr = origCall->getPropertiesAsAttribute();
          }
          Operation *newCall = builder.create(state);

          // Track: pod[@comp] = newCall result
          if (newCall->getNumResults() > 0 && compPod)
            trackedPodValues[compPod]["comp"] = newCall->getResult(0);

          changed = true;
        }
      }
    }
  }

  return changed;
}

/// Phase 2: Replace pod.read results with tracked values.
/// Skip @count field — it's count-tracking dead code that will be removed
/// in Phase 4 when scf.if and its users are erased.
/// Returns true if any changes were made.
bool replacePodReads(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues) {
  bool changed = false;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.getName().getStringRef() != "pod.read" || op.getNumResults() == 0)
      continue;
    auto field = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!field || op.getNumOperands() == 0)
      continue;
    // Skip count field — it's count-tracking, will be DCE'd
    if (field.getValue() == "count")
      continue;
    auto pit = trackedPodValues.find(op.getOperand(0));
    if (pit == trackedPodValues.end())
      continue;
    auto fit = pit->second.find(field.getValue());
    if (fit == pit->second.end())
      continue;
    op.getResult(0).replaceAllUsesWith(fit->second);
    op.erase();
    changed = true;
  }

  return changed;
}

/// Phase 3: Erase struct.writem that writes pod/struct-typed values
/// (sub-component bookkeeping, not needed for witness generation).
/// Returns true if any changes were made.
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

/// Phase 4: Iteratively erase dead pod/scf.if/count-tracking ops.
/// Returns true if any changes were made.
bool eraseDeadPodAndCountOps(Block &block) {
  bool changed = false;

  bool erasing = true;
  while (erasing) {
    erasing = false;
    for (Operation &op : llvm::make_early_inc_range(block)) {
      StringRef name = op.getName().getStringRef();
      bool isDead =
          (name == "pod.new" || name == "pod.read" || name == "pod.write" ||
           name == "scf.if" || name == "arith.subi" || name == "arith.cmpi");
      if (!isDead)
        continue;

      // Check all results are unused
      bool allUnused = true;
      for (auto result : op.getResults()) {
        if (!result.use_empty()) {
          allUnused = false;
          break;
        }
      }
      if (allUnused) {
        op.erase();
        erasing = true;
        changed = true;
      }
    }
  }

  return changed;
}

/// Simplify POD-based sub-component dispatch in a single function.def block.
/// Returns true if any changes were made.
bool eliminatePodDispatch(Block &block) {
  // Track: pod SSA value → {field_name → latest written SSA value}
  llvm::DenseMap<Value, llvm::StringMap<Value>> trackedPodValues;

  // Phase 1: Scan block, track pod field values and extract function.call
  // from scf.if into the parent block.
  bool changed = extractCallsFromScfIf(block, trackedPodValues);

  // Phase 2: Replace pod.read results with tracked values.
  // Skip @count field — it's count-tracking dead code that will be removed
  // in Phase 4 when scf.if and its users are erased.
  changed |= replacePodReads(block, trackedPodValues);

  // Phase 3: Erase struct.writem that writes pod/struct-typed values
  // (sub-component bookkeeping, not needed for witness generation).
  changed |= eraseStructWritemForPodValues(block);

  // Phase 4: Iteratively erase dead pod/scf.if/count-tracking ops.
  changed |= eraseDeadPodAndCountOps(block);

  return changed;
}

struct SimplifySubComponents
    : impl::SimplifySubComponentsBase<SimplifySubComponents> {
  using SimplifySubComponentsBase::SimplifySubComponentsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([](Operation *structDef) {
      if (structDef->getName().getStringRef() != "struct.def")
        return;

      structDef->walk([](Operation *funcDef) {
        if (funcDef->getName().getStringRef() != "function.def")
          return;
        auto symName = funcDef->getAttrOfType<StringAttr>("sym_name");
        if (!symName || symName.getValue() != "compute")
          return;

        for (Region &region : funcDef->getRegions())
          for (Block &block : region)
            eliminatePodDispatch(block);
      });
    });
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
