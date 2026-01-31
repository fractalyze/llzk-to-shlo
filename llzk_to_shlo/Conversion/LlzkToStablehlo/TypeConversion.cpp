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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::llzk_to_shlo {

namespace {
/// Get the dialect namespace from a type.
/// Handles both registered and unregistered (opaque) types.
StringRef getDialectNamespace(Type type) {
  // For opaque types (unregistered dialects), get namespace from the type
  if (auto opaqueType = dyn_cast<OpaqueType>(type)) {
    return opaqueType.getDialectNamespace();
  }
  // For registered types, get from dialect
  return type.getDialect().getNamespace();
}
} // namespace

bool LlzkToStablehloTypeConverter::isFeltType(Type type) {
  return getDialectNamespace(type) == "felt";
}

bool LlzkToStablehloTypeConverter::isArrayType(Type type) {
  return getDialectNamespace(type) == "array";
}

bool LlzkToStablehloTypeConverter::isStructType(Type type) {
  StringRef ns = getDialectNamespace(type);
  // The llzk struct dialect is called "component"
  return ns == "struct" || ns == "component";
}

LlzkToStablehloTypeConverter::LlzkToStablehloTypeConverter(
    MLIRContext *ctx, const llvm::APInt &prime, unsigned storageBitWidth,
    bool usePrimeFieldType) {
  // Create the prime field type with the given modulus and storage bit width
  auto intType = IntegerType::get(ctx, storageBitWidth);
  auto modulusAttr = IntegerAttr::get(intType, prime);
  primeFieldType = prime_ir::field::PrimeFieldType::get(ctx, modulusAttr);

  // Set element type based on flag
  this->usePrimeFieldType = usePrimeFieldType;
  if (usePrimeFieldType) {
    this->elementType = primeFieldType;
  } else {
    this->elementType = intType;
  }

  // Identity conversion for types we don't need to convert
  addConversion([](Type type) { return type; });

  // Convert felt.type to tensor<element_type>
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isFeltType(type))
      return std::nullopt;
    return RankedTensorType::get({}, elementType);
  });

  // Convert array types
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isArrayType(type))
      return std::nullopt;
    // For now, return a placeholder tensor type
    // Actual dimension extraction requires LLZK type interface
    return RankedTensorType::get({ShapedType::kDynamic}, elementType);
  });

  // Convert struct types
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isStructType(type))
      return std::nullopt;
    auto flatSize = getStructFlattenedSize(type);
    if (!flatSize) {
      // Default to dynamic if not registered
      return RankedTensorType::get({ShapedType::kDynamic}, elementType);
    }
    return RankedTensorType::get({*flatSize}, elementType);
  });

  // Handle function types
  addConversion([this](FunctionType type) -> Type {
    SmallVector<Type> inputs;
    SmallVector<Type> results;

    for (Type input : type.getInputs()) {
      inputs.push_back(convertType(input));
    }
    for (Type result : type.getResults()) {
      results.push_back(convertType(result));
    }

    return FunctionType::get(type.getContext(), inputs, results);
  });

  // Add source materialization (convert from source type to target type)
  addSourceMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  // Add target materialization (convert from target type back to source type)
  addTargetMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

std::optional<int64_t>
LlzkToStablehloTypeConverter::getStructFlattenedSize(Type structType) const {
  auto it = structFlattenedSizes.find(structType);
  if (it == structFlattenedSizes.end()) {
    return std::nullopt;
  }
  return it->second;
}

void LlzkToStablehloTypeConverter::registerStructFlattenedSize(Type structType,
                                                               int64_t size) {
  structFlattenedSizes[structType] = size;
}

std::optional<int64_t>
LlzkToStablehloTypeConverter::getFieldOffset(Type structType,
                                             StringRef fieldName) const {
  auto *ctx = structType.getContext();
  auto key = std::make_pair(structType, StringAttr::get(ctx, fieldName));
  auto it = fieldOffsets.find(key);
  if (it == fieldOffsets.end()) {
    return std::nullopt;
  }
  return it->second;
}

void LlzkToStablehloTypeConverter::registerFieldOffset(Type structType,
                                                       StringRef fieldName,
                                                       int64_t offset) {
  auto *ctx = structType.getContext();
  auto key = std::make_pair(structType, StringAttr::get(ctx, fieldName));
  fieldOffsets[key] = offset;
}

IntegerType LlzkToStablehloTypeConverter::getStorageType() const {
  if (usePrimeFieldType) {
    return primeFieldType.getStorageType();
  }
  return cast<IntegerType>(elementType);
}

DenseElementsAttr LlzkToStablehloTypeConverter::createConstantAttr(
    RankedTensorType tensorType, int64_t value, OpBuilder &rewriter) const {
  auto storageType = getStorageType();
  auto storageTensorType =
      RankedTensorType::get(tensorType.getShape(), storageType);
  return DenseElementsAttr::get(storageTensorType,
                                rewriter.getIntegerAttr(storageType, value));
}

} // namespace mlir::llzk_to_shlo
