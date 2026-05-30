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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_LOWERINGINFRASTRUCTURE_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_LOWERINGINFRASTRUCTURE_H_

// The type-shape predicates live here (rather than in PreConversionStructural)
// so PostConversionStructural can call isPromotableCarryType without taking a
// dep on PreConversionStructural.

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::llzk_to_shlo {

void registerStructFieldOffsets(ModuleOp module,
                                LlzkToStablehloTypeConverter &typeConverter);

void convertAllFunctions(ModuleOp module,
                         LlzkToStablehloTypeConverter &typeConverter,
                         MLIRContext *context);

void addStructuralConversionPatterns(
    LlzkToStablehloTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target);

bool involvesPodType(Type ty);
bool isPromotableCarryType(Type ty);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_LOWERINGINFRASTRUCTURE_H_
