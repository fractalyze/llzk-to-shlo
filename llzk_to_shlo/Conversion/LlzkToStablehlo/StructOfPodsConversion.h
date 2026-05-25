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
///
/// Precondition: `funcBlock` is a `@compute` function-body block that may
///   contain `pod.new` seeds of struct-of-pods shape — a `!pod<...>` whose K
///   records are named `@idx_0..@idx_K-1` and share a single uniform inner
///   type T. (`llzk.nondet` of that shape is intentionally NOT a seed: it is a
///   Phase-4 artifact already headed for elimination.)
/// Postcondition: one such carrier (per call) and its pod traffic are rewritten
///   to `array<K x T>` form, so `flattenPodArrayWhileCarry` and the `@in`
///   read-modify-write materializer then see the array-of-pods shape they
///   already handle. Non-uniform-inner carriers are left untouched (handled by
///   `materializeStructOfPodsCompField`).
/// Returns: true iff a carrier was converted (one per call); idempotent — after
///   success the carrier's type is array and no longer matches the seed.
bool convertStructOfPodsToArrayOfPods(Block &funcBlock);

/// Materialize a struct-of-pods component field read into per-field array
/// reads, mirroring `materializePodArrayCompField` for the struct-of-pods
/// shape.
///
/// Precondition: `funcBlock` is a `@compute` function-body block holding a
///   NON-uniform-inner struct-of-pods cascade that `convertStructOfPodsToArray-
///   OfPods` cannot rewrite (each `@idx_K` resolves to a distinct
///   `!struct<@Sub_K>` class). The dispatched writers must already be hoisted
///   (`function.call @<C>::@compute(array.extract %carrier[%cK]) :
///   !struct<@C>`) and the readers must be `struct.readm %nondet[@F]` on a pub
///   member `@F`.
/// Postcondition: one parallel felt array per read `@F` is allocated at
///   function-body scope; each hoisted call site writes its `struct.readm @F`
///   into it and each reader is rewired to read from it, bridging the severed
///   writer→reader dispatch link. The dispatch scf.if cascade scaffolding is
///   preserved; the orphaned reader nondets become DCE-able by Phase 4.
/// Returns: true iff a link was materialized; idempotent — a re-run finds no
///   `struct.readm` consumers left on the nondet ops.
bool materializeStructOfPodsCompField(Block &funcBlock);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_STRUCTOFPODSCONVERSION_H_
