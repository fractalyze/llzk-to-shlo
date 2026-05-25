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
///
/// Precondition: `block` may contain `scf.while` ops carrying
///   `array<N x !pod<[@f: felt|felt-array, ...]>>` iter-arg slots whose pod
///   element type has a non-empty, uniformly felt-flattenable record list.
/// Postcondition: one such slot (processed per call, last candidate first) is
///   split into per-field `array<N x T>` carries on the while and all nested
///   whiles threading it; body access chains are rewritten to direct
///   `array.read`/`array.write`. Slots with no discoverable fields are left
///   for a later fixed-point iteration.
/// Returns: true iff a slot was flattened (one per call) — drives the driver's
///   fixed-point loop.
bool flattenPodArrayWhileCarry(Block &block);

/// Replace each pod-array result slot of a result-bearing `scf.if` with N
/// per-field felt-array result slots. The actual data flow through the new
/// slots is closed later by `LlzkToStablehlo.cpp`. Only flattens slots whose
/// pod element type's record list is non-empty and uniformly felt-flattenable.
///
/// Precondition: `block` may contain result-bearing `scf.if` ops (both
///   branches non-empty) whose result list includes `array<N x !pod<...>>`
///   slots. Runs in the straggler fixed-point after the main while-carry
///   flatten, on scf.ifs whose pod-array result slots blocked per-field
///   carries from threading to outer `scf.yield`s.
/// Postcondition: matching slots are rewired to N per-field `array<N x felt>`
///   result slots (both branch yields fed fresh nondets); old pod-array result
///   users are rewired to an orphan `array<N x !pod>` nondet so the IR stays
///   well-typed until `extendResultBearingScfIfArrayChain` closes the flow.
/// Returns: true iff any rewrite fired; idempotent — a second run finds no
///   pod-array result slots.
bool flattenPodArrayScfIfResults(Block &block);

/// Unpack pod-typed `scf.while` carry values into individual fields, e.g.
/// `scf.while (%i, %pod)` carrying `!pod.type<[@c: array, @s: felt]>` becomes
/// `scf.while (%i, %c, %s)`. Uses `takeBody` to move regions, then modifies
/// block args and pod ops in-place.
///
/// Precondition: `block` may contain `scf.while` ops carrying exactly one
///   pod-typed (`!pod<...>`) iter-arg slot whose fields are discoverable from
///   `pod.read`/`pod.write` in the body or post-while users, and whose only
///   uses of the carry are pod.read/pod.write/struct.writem/chained-while
///   (the use-shape gate bails otherwise and lets the outer loop retry).
/// Postcondition: the matched while's pod carry is replaced by per-field
///   carries; any chained `scf.while` consuming the old pod result as init is
///   expanded and erased inline in the same call.
/// Returns: true after unpacking ONE while (then returns to caller — erasing
///   chained-while users inline invalidates the local SmallVector, so the
///   call site drives it to its own fixed point). false iff nothing matched.
bool unpackPodWhileCarry(Block &block);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYWHILECARRY_H_
