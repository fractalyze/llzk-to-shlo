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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkUpstreamArtifacts.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
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

} // namespace mlir::llzk_to_shlo
