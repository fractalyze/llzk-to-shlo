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

/// Phase 1: Scan block, track pod field values and extract function.call
/// from scf.if into the parent block.
bool extractCallsFromScfIf(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues);

/// Phase 2: Replace pod.read results with tracked values.
bool replacePodReads(
    Block &block,
    llvm::DenseMap<Value, llvm::StringMap<Value>> &trackedPodValues);

/// Phase 4: Iteratively erase dead ops that are not core computation.
bool eraseDeadPodAndCountOps(Block &block);

/// Phase 5: Replace remaining pod.read with llzk.nondet, then erase orphaned
/// pod.new ops.
bool replaceRemainingPodOps(Block &block);

/// Simplify POD-based sub-component dispatch in a single function.def block.
/// Runs Phases 1–5 in sequence.
bool eliminatePodDispatch(Block &block);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODDISPATCHPHASES_H_
