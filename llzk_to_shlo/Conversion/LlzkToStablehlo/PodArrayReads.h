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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYREADS_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYREADS_H_

#include "mlir/IR/Block.h"

namespace mlir::llzk_to_shlo {

/// Resolve `array.read %arr[%idx] → pod.read @comp → struct.readm @out`
/// chains, sourced from array-of-pods, to the extracted `function.call`
/// result directly. Handles the count/dispatch array pattern where `@comp`
/// holds the computed struct result.
bool resolveArrayPodCompReads(Block &block);

/// Fold `array.read → pod.read @<count|comp|in>` chains the per-pod tracker
/// can't resolve when the pod source is an array-of-pods: `@count` collapses
/// to a const-0 index, `@comp`/`@in` to an `llzk.nondet` of the result type.
/// Witness-gen relies on `@constrain` re-deriving these values, so the
/// substitution is sound; non-whitelisted fields are left untouched.
bool rewriteArrayPodCountCompInReads(Block &block);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYREADS_H_
