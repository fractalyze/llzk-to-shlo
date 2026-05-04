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

#ifndef LLZK_TO_SHLO_VERIFY_WITNESS_LAYOUT_TD_
#define LLZK_TO_SHLO_VERIFY_WITNESS_LAYOUT_TD_

include "mlir/Pass/PassBase.td"

def VerifyWitnessLayout : Pass<"verify-witness-layout", "ModuleOp"> {
  let summary =
    "Assert the lowered StableHLO `@main`'s dynamic_update_slice chain "
    "matches the module's `wla.layout` spec.";
  let description = [{
    Runs LAST in the llzk-to-shlo pipeline. For each module, looks up the
    optional `wla.layout` op emitted by `--witness-layout-anchor`, walks
    `@main`'s terminal `stablehlo.dynamic_update_slice` chain, and asserts
    that every signal in the layout maps to a covering chunk whose source
    is not a splat-zero constant (for `output` and `input` kinds). For
    `internal` signals, splat-zero is permitted: a legitimately zero
    internal member is indistinguishable from an orphan at this stage.

    Pure read-only on IR. Silently no-ops on modules without `wla.layout`,
    so the pass can be wired into the pipeline before the anchor pass is
    universally enabled.

    Replaces (eventually) the `flag-orphan-zero-writes` heuristic from
    PR #67/#71 with an exact per-signal check. See
    `docs/WITNESS_LAYOUT_ANCHOR.md` for the contract.
  }];
}

#endif  // LLZK_TO_SHLO_VERIFY_WITNESS_LAYOUT_TD_
