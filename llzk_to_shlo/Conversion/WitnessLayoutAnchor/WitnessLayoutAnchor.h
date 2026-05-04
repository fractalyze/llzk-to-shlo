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

#ifndef LLZK_TO_SHLO_CONVERSION_WITNESSLAYOUTANCHOR_WITNESSLAYOUTANCHOR_H_
#define LLZK_TO_SHLO_CONVERSION_WITNESSLAYOUTANCHOR_WITNESSLAYOUTANCHOR_H_

#include "mlir/Pass/Pass.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DECL_WITNESSLAYOUTANCHOR
#define GEN_PASS_REGISTRATION
#include "llzk_to_shlo/Conversion/WitnessLayoutAnchor/WitnessLayoutAnchor.h.inc"

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_WITNESSLAYOUTANCHOR_WITNESSLAYOUTANCHOR_H_
