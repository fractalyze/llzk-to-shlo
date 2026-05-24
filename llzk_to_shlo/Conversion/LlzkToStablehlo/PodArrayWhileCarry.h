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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYWHILECARRY_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYWHILECARRY_H_

#include "mlir/IR/Block.h"

namespace mlir::llzk_to_shlo {

/// Flatten array-of-pods (all-felt fields) carried through `scf.while` into
/// per-field felt arrays, rewriting `array.read`+`pod.read`/`pod.write`+
/// `array.write` chains into direct `array.read`/`array.write`. Handles
/// arbitrarily-deep nested while-carry chains: all whiles threading the
/// pod-array iter-arg are flattened simultaneously, with per-field arrays
/// preserving the full source dims so multi-dim sources don't alias.
bool flattenPodArrayWhileCarry(Block &block);

/// Replace each pod-array result slot of a result-bearing `scf.if` with N
/// per-field felt-array result slots. The actual data flow through the new
/// slots is closed later by `LlzkToStablehlo.cpp`. Only flattens slots whose
/// pod element type's record list is non-empty and uniformly felt-flattenable.
/// Returns true iff any rewrite fired; idempotent.
bool flattenPodArrayScfIfResults(Block &block);

/// Unpack pod-typed `scf.while` carry values into individual fields, e.g.
/// `scf.while (%i, %pod)` carrying `!pod.type<[@c: array, @s: felt]>` becomes
/// `scf.while (%i, %c, %s)`. Uses `takeBody` to move regions, then modifies
/// block args and pod ops in-place.
bool unpackPodWhileCarry(Block &block);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYWHILECARRY_H_
