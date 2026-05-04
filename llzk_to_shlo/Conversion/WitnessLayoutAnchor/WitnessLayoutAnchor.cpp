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

#include "llzk_to_shlo/Conversion/WitnessLayoutAnchor/WitnessLayoutAnchor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "llzk_to_shlo/Dialect/WLA/WLA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_WITNESSLAYOUTANCHOR
#include "llzk_to_shlo/Conversion/WitnessLayoutAnchor/WitnessLayoutAnchor.h.inc"

namespace {

// Skip pod-typed members. These are LLZK lowering scaffolding (the
// `*$inputs` dispatch records circom emits for input collection of every
// sub-component call) and are erased by `--simplify-sub-components`
// before the lowering registers per-member offsets. Including them in
// the layout would make T4's per-chunk match diverge for every chip with
// sub-components. Unwraps any depth of `ShapedType` so multi-rank arrays
// like `!array<... x !pod.type>` and any future nested-array shapes are
// caught equally.
bool isPodMember(Type ty) {
  while (auto shaped = dyn_cast<ShapedType>(ty))
    ty = shaped.getElementType();
  return ty.getDialect().getNamespace() == "pod";
}

// Build the canonical signal list for `mainStruct` and emit one
// `wla.layout` op at the top of `moduleOp`'s body.
void emitLayoutForMainStruct(ModuleOp moduleOp,
                             llzk::component::StructDefOp mainStruct) {
  MLIRContext *ctx = moduleOp.getContext();
  SmallVector<Attribute> signals;
  int64_t offset = 0;

  auto pushSignal = [&](StringRef name, wla::SignalKind kind, int64_t length) {
    signals.push_back(wla::SignalAttr::get(ctx, name, kind, offset, length));
    offset += length;
  };

  // const_one head (universal in circom-generated witnesses).
  pushSignal("const_one", wla::SignalKind::Internal, /*length=*/1);

  // {llzk.pub} struct.member declarations → output signals.
  std::vector<llzk::component::MemberDefOp> members =
      mainStruct.getMemberDefs();
  for (auto m : members) {
    if (!m.hasPublicAttr() || isPodMember(m.getType()))
      continue;
    pushSignal(("@" + m.getSymName()).str(), wla::SignalKind::Output,
               getMemberFlatSize(m.getType()));
  }

  // @compute function arguments → input signals.
  if (auto computeFn = mainStruct.getComputeFuncOp()) {
    for (auto [idx, argTy] : llvm::enumerate(computeFn.getArgumentTypes())) {
      pushSignal(("%arg" + Twine(idx)).str(), wla::SignalKind::Input,
                 getMemberFlatSize(argTy));
    }
  }

  // Non-pub, writem-targeted struct.member declarations → internals.
  // The writem filter drops pod-typed `*$inputs` bookkeeping (written via
  // `pod.write`, not `struct.writem`) so the layout matches the
  // post-simplify member set the lowering registers offsets for.
  DenseSet<StringAttr> writemTargets = collectWritemTargets(mainStruct);
  for (auto m : members) {
    if (m.hasPublicAttr() || isPodMember(m.getType()))
      continue;
    if (!writemTargets.contains(m.getSymNameAttr()))
      continue;
    pushSignal(("@" + m.getSymName()).str(), wla::SignalKind::Internal,
               getMemberFlatSize(m.getType()));
  }

  OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
  builder.create<wla::LayoutOp>(moduleOp.getLoc(),
                                ArrayAttr::get(ctx, signals));
}

class WitnessLayoutAnchorPass
    : public impl::WitnessLayoutAnchorBase<WitnessLayoutAnchorPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTableCollection symbolTable;
    auto mainDef = llzk::getMainInstanceDef(symbolTable, moduleOp);
    if (failed(mainDef) || !mainDef->get())
      return; // Non-chip module (helper / fixture); silent no-op.
    emitLayoutForMainStruct(moduleOp, mainDef->get());
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
