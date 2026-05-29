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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODDISPATCHPHASES_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODDISPATCHPHASES_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

namespace mlir::llzk_to_shlo {

// NOTE: this header exposes phases 1, 2, 4, 5. Phase 3
// (`eraseStructWritemForPodValues`) lives entirely inside
// PodDispatchPhases.cpp because no out-of-TU caller needs it — the
// sequence runs through `eliminatePodDispatch` below, which executes
// all five phases sequentially.

/// Phase 1: Scan block, track pod field values and extract a dispatch-fired
/// `function.call` out of its (statically-false) `scf.if` into the parent
/// block via clone-hoist of its operand chain.
///
/// Precondition: `block` may contain `pod.new` dispatch records plus
///   `function.call`s buried in dispatch-firing `scf.if`s whose call operands
///   are hoistable (pure / read-only). `trackedPodValues` is the per-block
///   pod-SSA-value → {field → written value} map this phase populates.
/// Postcondition: hoistable buried calls are cloned before their `scf.if`
///   (the clone dominates it) and `trackedPodValues` records each pod's
///   field writes for Phase 2.
/// Returns: true iff any call was hoisted or tracking advanced the IR.
bool extractCallsFromScfIf(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues);

/// Phase 2: Replace `pod.read` results with the value tracked for that field,
/// then erase the now-dead reads (walks nested regions too).
///
/// Precondition: `trackedPodValues` was populated by `extractCallsFromScfIf`
///   on the same `block`. Must NOT be run on blocks with surviving pod-typed
///   block args (the broad RAUW would tear `pod.read %arg[@field]` read-back
///   patterns `unpackPodWhileCarry` still depends on).
/// Postcondition: every `pod.read` whose source+field is tracked is RAUW'd to
///   the (transitively resolved) terminal value and erased; cyclic/self chains
///   are skipped.
/// Returns: true iff any read was replaced.
bool replacePodReads(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues);

/// Phase 4: Iteratively erase dead dispatch/bookkeeping ops that are not core
/// computation (core = felt/struct/array/function + scf.while/yield/condition).
///
/// Precondition: Phases 1–3 have RAUW'd live `pod.read` consumers, so the only
///   pod/`scf.if`/count ops left are dead bookkeeping — except side-effecting
///   `scf.for`/`scf.if` bodies with a non-pod array.write or struct.writem,
///   and user-input `pod.write %arr_elem[@field]` chains, which are preserved.
/// Postcondition: dead non-core ops are erased to a fixed point; the preserved
///   carve-outs survive for `flattenPodArrayWhileCarry`/downstream passes.
/// Returns: true iff any op was erased.
bool eraseDeadPodAndCountOps(Block &block);

/// Phase 5: Replace remaining (self-referential) `pod.read` with `llzk.nondet`
/// of the result type, then erase orphaned `pod.new` ops.
///
/// Precondition: after Phases 1–4 the only surviving pod ops are `pod.new` +
///   `pod.read` pairs supplying a don't-care initial mutable value (the array
///   is fully overwritten before meaningful use). Must NOT be run on a block
///   still holding pod-typed block args (it would clobber field-discovery
///   reads `unpackPodWhileCarry` needs).
/// Postcondition: those reads become `llzk.nondet`; `pod.new` ops with no
///   remaining users are erased.
/// Returns: true iff any read was substituted or `pod.new` erased.
bool replaceRemainingPodOps(Block &block);

/// Simplify POD-based sub-component dispatch in a single function.def block.
/// Runs Phases 1–5 in sequence.
///
/// Precondition: `block` is a `function.def`/`@compute` body that may contain
///   the pod-dispatch state machine (`pod.new` records + counter + delayed
///   `function.call`). Must NOT be called on blocks with pod-typed block args
///   or top-level array-of-pods reads (Phase 5 would clobber field discovery —
///   the driver gates on this and runs only the safe phases there instead).
/// Postcondition: dispatch boilerplate is removed and the direct
///   `function.call` is left, ready for the type-aware conversion pass.
/// Returns: true iff any phase changed the IR — drives the driver's fixed
///   point.
bool eliminatePodDispatch(Block &block);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODDISPATCHPHASES_H_
