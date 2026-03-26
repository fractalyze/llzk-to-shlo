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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_SIMPLIFYSUBCOMPONENTS
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h.inc"

namespace {

/// Simplify POD-based sub-component dispatch in a single function.def block.
/// Returns true if any changes were made.
bool simplifyBlock(Block &block) {
  // Track: pod SSA value → {field_name → latest written SSA value}
  llvm::DenseMap<Value, llvm::StringMap<Value>> podFields;
  bool changed = false;

  // Phase 1: Scan block, track pod field values and extract function.call
  // from scf.if into the parent block.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    StringRef name = op.getName().getStringRef();

    if (name == "pod.new") {
      if (op.getNumResults() > 0) {
        podFields[op.getResult(0)] = {};
        // Track initialization values. For pod.new { @count = %c2 },
        // the operand %c2 maps to field @count. Extract field names
        // from the properties.
        if (op.getNumOperands() > 0) {
          // Get field names from printed properties
          auto propsAttr = op.getPropertiesAsAttribute();
          if (propsAttr) {
            std::string s;
            llvm::raw_string_ostream os(s);
            propsAttr.print(os);
            // Find initializedRecords = ["count", ...]
            size_t p = s.find("initializedRecords");
            if (p != std::string::npos) {
              size_t lb = s.find('[', p);
              size_t rb = s.find(']', lb);
              if (lb != std::string::npos && rb != std::string::npos) {
                StringRef list = StringRef(s).slice(lb + 1, rb);
                unsigned argIdx = 0;
                while (!list.empty() && argIdx < op.getNumOperands()) {
                  auto [tok, rest] = list.split(',');
                  tok = tok.trim().trim('"');
                  if (!tok.empty()) {
                    podFields[op.getResult(0)][tok] = op.getOperand(argIdx++);
                  }
                  list = rest;
                }
              }
            }
          }
        }
      }
    } else if (name == "pod.write") {
      auto field = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (field && op.getNumOperands() >= 2)
        podFields[op.getOperand(0)][field.getValue()] = op.getOperand(1);
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
          auto pit = podFields.find(srcPod);
          if (pit != podFields.end()) {
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
            podFields[compPod]["comp"] = newCall->getResult(0);

          changed = true;
        }
      }
    }
  }

  // Phase 2: Replace pod.read results with tracked values.
  // Skip @count field — it's count-tracking dead code that will be removed
  // in Phase 4 when scf.if and its users are erased.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.getName().getStringRef() != "pod.read" || op.getNumResults() == 0)
      continue;
    auto field = op.getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!field || op.getNumOperands() == 0)
      continue;
    // Skip count field — it's count-tracking, will be DCE'd
    if (field.getValue() == "count")
      continue;
    auto pit = podFields.find(op.getOperand(0));
    if (pit == podFields.end())
      continue;
    auto fit = pit->second.find(field.getValue());
    if (fit == pit->second.end())
      continue;
    op.getResult(0).replaceAllUsesWith(fit->second);
    op.erase();
    changed = true;
  }

  // Phase 3: Erase struct.writem that writes pod/struct-typed values
  // (sub-component bookkeeping, not needed for witness generation).
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

  // Phase 4: Iteratively erase dead pod/scf.if/count-tracking ops.
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
            simplifyBlock(block);
      });
    });
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
