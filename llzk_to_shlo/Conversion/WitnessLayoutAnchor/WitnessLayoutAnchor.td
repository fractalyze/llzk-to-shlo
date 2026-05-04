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

#ifndef LLZK_TO_SHLO_CONVERSION_WITNESS_LAYOUT_ANCHOR_TD_
#define LLZK_TO_SHLO_CONVERSION_WITNESS_LAYOUT_ANCHOR_TD_

include "mlir/Pass/PassBase.td"

def WitnessLayoutAnchor
    : Pass<"witness-layout-anchor", "::mlir::ModuleOp"> {
  let summary =
      "Emit one wla.layout op per module from the LLZK input's main struct.";
  let description = [{
    Walks the top-level `ModuleOp` and, for each chip whose `llzk.main`
    attribute names a struct, emits one `wla.layout` op at module top
    carrying the canonical circom witness layout: a `const_one` head, the
    `{llzk.pub}` outputs of the main struct, the `@compute` function
    parameters, and the writem-targeted internal members — sized by
    `getMemberFlatSize` so the spec agrees with the lowering's per-member
    flat size (registered later by `registerStructFieldOffsets` in
    `--llzk-to-stablehlo`).

    Pure-additive: the pass only adds a new op at module scope and never
    modifies or erases existing ops. It runs AFTER
    `--simplify-sub-components` and BEFORE `--llzk-to-stablehlo` so the
    `struct.def` + `struct.member` + `struct.writem` shape it reads is
    exactly the shape `registerStructFieldOffsets` (in the lowering)
    consumes — making T3's emit and the lowering's offset table
    consistent by construction. Running before SSC would conflict with
    `createEmptyTemplateRemoval`'s `applyFullConversion`, which rejects
    any op outside the upstream LLZK target's legal set. The
    `--verify-witness-layout` pass at pipeline tail consumes the spec.
  }];
  let dependentDialects = ["::mlir::llzk_to_shlo::wla::WLADialect"];
}

#endif  // LLZK_TO_SHLO_CONVERSION_WITNESS_LAYOUT_ANCHOR_TD_
