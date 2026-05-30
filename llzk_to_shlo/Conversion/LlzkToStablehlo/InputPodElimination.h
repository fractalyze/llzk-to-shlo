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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_INPUTPODELIMINATION_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_INPUTPODELIMINATION_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

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

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_INPUTPODELIMINATION_H_
