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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODMODULECLEANUP_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODMODULECLEANUP_H_

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

/// Hoist a same-named child out of `builtin.module @X { function.def @X }`
/// (or `struct.def @X`) wrapper shells, erase the wrapper, then rewrite all
/// `@X::@X[::@method]` symbol refs to `@X[::@method]`. Untangles the
/// post-template-removal residue of circom PR #378's same-named wrappers,
/// which otherwise trip `SymbolTable` redefinition errors downstream.
void flattenSingleEntityWrapperModules(ModuleOp module);

/// Rewrite `!struct.type<@X<[]>>` (empty params) to `!struct.type<@X>` on
/// `llzk.nondet` results only. Scoped narrowly because the upstream
/// template-removal type converter manages its own struct-typed ops.
void stripEmptyStructParams(ModuleOp module);

/// Remove `@X$inputs` pod struct members and their `struct.writem`/
/// `struct.readm`/`pod.read` traffic, replacing `pod.read` consumers with
/// `llzk.nondet` of the field type. The `$inputs` channel is don't-care for
/// witness generation; this must run before `createEmptyTemplateRemoval`,
/// whose `applyFullConversion` has no `pod.read` target pattern.
void eliminateInputPods(ModuleOp module);

/// Inline single-field input pods (no `@count`) used as `scf.while` carry to
/// their inner field type. Must run before `createEmptyTemplateRemoval` so
/// its `applyFullConversion` doesn't see residual `pod.*` ops.
void inlineInputPodCarries(ModuleOp module);

/// Lift a `function.call` out of an `scf.while` body when its operands are
/// loop-invariant after pod resolution and its result feeds only
/// `struct.readm → array.insert/write` chains on arrays declared outside the
/// loop. Collapses N body-iter calls into one post-while call (last-write-wins
/// under the same-cell single-instance dispatch precondition). Runs after the
/// outer pod-resolution fixed point so the operand patterns are recognizable.
bool liftConstIndexPodArrayCallPostWhile(Operation *root);

/// Erase pod-typed iter slots from `scf.while` and DCE the `pod.new` chain
/// that fed them, once `pod.read`/`pod.write` have been nondet'd/erased. Uses
/// a use-trace (not cascade reshape) to avoid breaking the `<--` cascade type
/// invariants LlzkToStablehlo matches on; rebuilds in post order and defers
/// the erase so no value is destroyed while still claimed.
bool erasePodTypedCarrierSlots(ModuleOp module);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODMODULECLEANUP_H_
