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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODINVARIANTS_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODINVARIANTS_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

/// Debug-only structural post-condition for `SimplifySubComponents`. Asserts
/// that no `scf.while` op carries an `llzk::pod::PodType` in any iter-arg
/// (block argument of either region) or result. The while-carry flatten +
/// unpack phases exist precisely to remove these; a surviving pod-typed carry
/// means one of those phases was reordered or skipped, which silently
/// miscompiles (caught only by the GPU gate today).
///
/// MUST be called only at end-of-pass (after the outer fixed point and all
/// straggler flatten passes converge) — intermediate IR legitimately contains
/// pod-typed while carries during iterative dispatch elimination.
///
/// Strict no-op under NDEBUG. See docs/passes/simplify-sub-components.md.
void assertNoPodTypedWhileCarry(ModuleOp module);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODINVARIANTS_H_
