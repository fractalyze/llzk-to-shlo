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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_ARRAYPATTERNS_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_ARRAYPATTERNS_H_

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

/// Populate conversion patterns for array.* operations to StableHLO.
void populateArrayToStablehloPatterns(LlzkToStablehloTypeConverter &converter,
                                      RewritePatternSet &patterns,
                                      ConversionTarget &target);

/// Replace `op` with `stablehlo::DynamicUpdateSliceOp(dest, update,
/// startIndices)`. Handles both void (0-result) and result-bearing input ops:
/// for void inputs the DUS is created standalone and `op` is erased; for
/// result-bearing inputs the standard `replaceOpWithNewOp` path runs.
/// Used by `array.write` / `array.insert` / `struct.writem` lowering.
void replaceWithDUS(ConversionPatternRewriter &rewriter, Operation *op,
                    Location loc, Value dest, Value update,
                    ValueRange startIndices);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_ARRAYPATTERNS_H_
