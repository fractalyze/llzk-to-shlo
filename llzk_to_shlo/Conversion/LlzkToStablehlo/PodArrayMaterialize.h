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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYMATERIALIZE_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYMATERIALIZE_H_

#include "mlir/IR/Block.h"

namespace mlir::llzk_to_shlo {

/// Materialize a per-struct-field felt array for every function-scope pod-array
/// whose `@comp` slot is written by a hoisted `function.call` result in one
/// block and read back in a *different* block via `pod.read [@comp];
/// struct.readm [@F]`. Without it the cross-block reader is nondet'd and the
/// data flow is severed (circomlib `gates.circom` `XorArray*`/`AndArray*`
/// family). Materializes at the felt level to sidestep the `<N x !struct>`
/// while-body verifier path. Idempotent.
bool materializePodArrayCompField(Block &funcBlock);

/// Forward a dispatch *input* pod's writer value directly to its sibling
/// firing-site `pod.read [@in]`, replacing the read and erasing it, before
/// `flattenPodArrayWhileCarry` while writer and reader are still SSA-paired
/// through the pod cell. Prevents deeply-nested dispatch inputs (AES
/// `@AES256Encrypt_6::@compute`) from being disconnected and fed const-zero.
/// Idempotent.
bool materializePodArrayInputPodField(Block &funcBlock);

/// Project the post-while `function.call` result of a scalar dispatch-pod
/// `@comp` field onto its cross-block `pod.read [@comp]` readers, exploiting
/// the count-countdown invariant (the firing call's operands match the writer-
/// while's post-loop iter-arg state). Runs AFTER `unpackPodWhileCarry` and
/// BEFORE `eliminatePodDispatch`. Idempotent.
bool materializeScalarPodCompField(Block &funcBlock);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_PODARRAYMATERIALIZE_H_
