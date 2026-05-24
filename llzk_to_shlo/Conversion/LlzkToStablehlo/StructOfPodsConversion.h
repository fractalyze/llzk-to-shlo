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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_STRUCTOFPODSCONVERSION_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_STRUCTOFPODSCONVERSION_H_

#include "mlir/IR/Block.h"

namespace mlir::llzk_to_shlo {

/// Rewrite struct-of-pods carriers (`!pod<[@idx_0..@idx_K-1: T]>`) to
/// array-of-pods (`!array<K x T>`) so the existing `flattenPodArrayWhileCarry`
/// infrastructure handles them. Runs BEFORE `materializePodArrayInputPodField`.
bool convertStructOfPodsToArrayOfPods(Block &funcBlock);

/// Materialize a struct-of-pods component field read into per-field array
/// reads, mirroring `materializePodArrayCompField` for the struct-of-pods
/// shape.
bool materializeStructOfPodsCompField(Block &funcBlock);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_STRUCTOFPODSCONVERSION_H_
