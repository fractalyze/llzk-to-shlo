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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_SIMPLIFYSUBCOMPONENTSINTERNAL_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_SIMPLIFYSUBCOMPONENTSINTERNAL_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace mlir::llzk_to_shlo {

/// Walk up from `funcBlock` past any nested `builtin.module` wrappers to the
/// top-level module (LLZK v2's `createEmptyTemplateRemoval` wraps each
/// component in its own `builtin.module`).
ModuleOp getTopLevelModule(Block &funcBlock);

/// Build `array<destDims + innerDims x leafFelt>` when `innerFeltTy` is a felt
/// array, or `array<destDims x innerFeltTy>` when it is a scalar `!felt`.
llzk::array::ArrayType
combineDispatchAndInnerFeltDims(Type innerFeltTy, ArrayRef<int64_t> destDims);

/// Check if all results of an operation are unused.
bool isAllResultsUnused(Operation &op);

/// True iff `v` is defined inside one of `op`'s regions.
bool isValueDefinedInside(Value v, Operation &op);

/// Recursively clone the defining-op chain of `v` BEFORE `insertBefore`.
/// Returns cloned value or null Value() on failure.
Value cloneDefiningOpBefore(Value v, Operation *insertBefore,
                            Operation &guardOp,
                            llvm::DenseMap<Value, Value> &cloneCache,
                            unsigned depth = 8);

/// True iff `cloneDefiningOpBefore` would be able to clone `v` out of
/// `guardOp` using the same safety rules and recursion depth.
bool canCloneDefiningOpBefore(Value v, Operation &guardOp, unsigned depth = 8);

/// Create an llzk.nondet operation producing an uninitialized value.
Value createNondet(OpBuilder &builder, Location loc, Type type);

/// True for types that participate in pod-array per-field flattening:
/// `!felt.type` or `!array.type<... x !felt.type>`.
bool isFlattenableFelt(Type ty);

/// Index operands of an LLZK `array.read` / `array.write` op. The first
/// operand is the array; everything after is the index list.
SmallVector<Value> arrayAccessIndices(Operation *arrayAccess);

/// Populate the module-scope cache of struct members read outside @constrain.
/// Must be called before any phase erases readm ops.
void populateExternallyLiveMembers(ModuleOp module);

} // namespace mlir::llzk_to_shlo

#endif // NOLINT(build/header_guard): guard exceeds the 80-col line limit
