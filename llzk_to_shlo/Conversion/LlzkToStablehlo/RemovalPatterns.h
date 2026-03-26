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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_REMOVALPATTERNS_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_REMOVALPATTERNS_H_

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

/// Populate patterns to remove LLZK structural operations that have no
/// equivalent in StableHLO:
/// - struct.def: Erased (struct definitions become tensors)
/// - struct.member: Erased (member definitions absorbed into tensor layout)
/// - function.def @constrain: Erased (constraint functions not needed at
/// runtime)
/// - constrain.eq: Erased (constraints not needed at runtime)
void populateRemovalPatterns(LlzkToStablehloTypeConverter &converter,
                             RewritePatternSet &patterns,
                             ConversionTarget &target);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_REMOVALPATTERNS_H_
