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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkToStablehlo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/Register.h"

// LLZK dialects and transforms
#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk_to_shlo/Conversion/BatchStablehlo/BatchStablehlo.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"
#include "llzk_to_shlo/Conversion/VerifyWitnessLayout/VerifyWitnessLayout.h"
#include "llzk_to_shlo/Conversion/WitnessLayoutAnchor/WitnessLayoutAnchor.h"
#include "llzk_to_shlo/Dialect/WLA/WLA.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR core dialects
  mlir::registerAllDialects(registry);

  // Register StableHLO dialects
  mlir::stablehlo::registerAllDialects(registry);

  // Register prime-ir field dialect
  registry.insert<mlir::prime_ir::field::FieldDialect>();

  // Register LLZK dialects
  registry.insert<llzk::array::ArrayDialect>();
  registry.insert<llzk::boolean::BoolDialect>();
  registry.insert<llzk::cast::CastDialect>();
  registry.insert<llzk::constrain::ConstrainDialect>();
  registry.insert<llzk::felt::FeltDialect>();
  registry.insert<llzk::function::FunctionDialect>();
  registry.insert<llzk::global::GlobalDialect>();
  registry.insert<llzk::include::IncludeDialect>();
  registry.insert<llzk::LLZKDialect>();
  registry.insert<llzk::pod::PODDialect>();
  registry.insert<llzk::polymorphic::PolymorphicDialect>();
  registry.insert<llzk::string::StringDialect>();
  registry.insert<llzk::component::StructDialect>();
  registry.insert<mlir::llzk_to_shlo::wla::WLADialect>();

  // Register passes
  mlir::llzk_to_shlo::registerBatchStablehloPass();
  mlir::llzk_to_shlo::registerLlzkToStablehloPass();
  mlir::llzk_to_shlo::registerSimplifySubComponentsPass();
  mlir::llzk_to_shlo::registerVerifyWitnessLayoutPass();
  mlir::llzk_to_shlo::registerWitnessLayoutAnchorPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LLZK to StableHLO optimizer\n", registry));
}
