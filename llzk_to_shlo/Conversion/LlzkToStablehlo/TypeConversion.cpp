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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

// ===----------------------------------------------------------------------===
// Type checking helpers
// ===----------------------------------------------------------------------===

namespace {
StringRef getDialectNamespace(Type type) {
  if (auto opaqueType = dyn_cast<OpaqueType>(type))
    return opaqueType.getDialectNamespace();
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
  // The LLZK struct dialect is registered as "component" in the namespace
  return ns == "struct" || ns == "component";
}

// ===----------------------------------------------------------------------===
// Shared utility functions
// ===----------------------------------------------------------------------===

SmallVector<int64_t> getArrayDimensions(Type arrayType) {
  // LLZK ArrayType implements ShapedType::Trait and provides getShape().
  if (auto shaped = dyn_cast<ShapedType>(arrayType)) {
    auto shape = shaped.getShape();
    return SmallVector<int64_t>(shape.begin(), shape.end());
  }
  return {ShapedType::kDynamic};
}

int64_t getMemberFlatSize(Type memberType) {
  if (LlzkToStablehloTypeConverter::isArrayType(memberType)) {
    auto dims = getArrayDimensions(memberType);
    int64_t size = 1;
    for (int64_t d : dims)
      size *= (d == ShapedType::kDynamic) ? 1 : d;
    return size;
  }
  return 1;
}

Value lookThroughCast(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    if (castOp.getNumOperands() == 1)
      return castOp.getOperand(0);
  return v;
}

Value ensureTensorType(OpBuilder &b, Value v, Type originalType,
                       const LlzkToStablehloTypeConverter &tc, Location loc) {
  v = lookThroughCast(v);
  if (isa<RankedTensorType>(v.getType()))
    return v;
  Type converted = tc.convertType(originalType);
  if (!converted)
    return v;
  return b.create<UnrealizedConversionCastOp>(loc, converted, v).getResult(0);
}

Value indexToI64Tensor(OpBuilder &b, Value idx, Location loc) {
  // If already a 0-d i64 tensor, return as-is.
  if (auto tt = dyn_cast<RankedTensorType>(idx.getType()))
    if (tt.getRank() == 0 && tt.getElementType().isInteger(64))
      return idx;
  // Look through unrealized_conversion_cast to find original tensor<i64>.
  if (auto castOp = idx.getDefiningOp<UnrealizedConversionCastOp>()) {
    Value src = castOp.getInputs()[0];
    if (auto tt = dyn_cast<RankedTensorType>(src.getType()))
      if (tt.getRank() == 0 && tt.getElementType().isInteger(64))
        return src;
  }
  // Convert index/integer scalar to i64 scalar, then wrap in tensor.
  Value i64Val = idx;
  if (idx.getType().isIndex())
    i64Val = b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
  return b.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({}, b.getI64Type()), i64Val);
}

Value createI64ScalarConstant(OpBuilder &b, Location loc, int64_t value) {
  auto type = RankedTensorType::get({}, b.getI64Type());
  auto attr = DenseElementsAttr::get(type, b.getI64IntegerAttr(value));
  return b.create<stablehlo::ConstantOp>(loc, attr);
}

std::optional<int64_t> parseBoolCmpPredicate(Attribute predicateAttr) {
  // Try direct integer value first
  if (auto intAttr = dyn_cast<IntegerAttr>(predicateAttr))
    return intAttr.getInt();

  // Fall back to string parsing: enum printed as #bool<cmp lt> etc.
  std::string s;
  llvm::raw_string_ostream os(s);
  predicateAttr.print(os);

  // Predicate enum: eq=0, ne=1, lt=2, le=3, gt=4, ge=5
  if (s.find("lt") != std::string::npos)
    return 2;
  if (s.find("le") != std::string::npos)
    return 3;
  if (s.find("gt") != std::string::npos)
    return 4;
  if (s.find("ge") != std::string::npos)
    return 5;
  if (s.find("ne") != std::string::npos)
    return 1;
  if (s.find("eq") != std::string::npos)
    return 0;
  return std::nullopt;
}

SmallVector<StringRef> getPodInitializedRecords(Operation *podNewOp) {
  SmallVector<StringRef> fieldNames;
  auto propsAttr = podNewOp->getPropertiesAsAttribute();
  if (!propsAttr)
    return fieldNames;

  std::string s;
  llvm::raw_string_ostream os(s);
  propsAttr.print(os);

  size_t p = s.find("initializedRecords");
  if (p == std::string::npos)
    return fieldNames;

  size_t lb = s.find('[', p);
  size_t rb = s.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos)
    return fieldNames;

  StringRef list = StringRef(s).slice(lb + 1, rb);
  while (!list.empty()) {
    auto [tok, rest] = list.split(',');
    tok = tok.trim().trim('"');
    if (!tok.empty())
      fieldNames.push_back(tok);
    list = rest;
  }
  return fieldNames;
}

// ===----------------------------------------------------------------------===
// Type converter implementation
// ===----------------------------------------------------------------------===

LlzkToStablehloTypeConverter::LlzkToStablehloTypeConverter(
    MLIRContext *ctx, const llvm::APInt &prime, unsigned storageBitWidth,
    bool usePrimeFieldType) {
  auto intType = IntegerType::get(ctx, storageBitWidth);
  auto modulusAttr = IntegerAttr::get(intType, prime);
  primeFieldType = prime_ir::field::PrimeFieldType::get(ctx, modulusAttr);

  this->usePrimeFieldType = usePrimeFieldType;
  this->elementType = usePrimeFieldType ? Type(primeFieldType) : Type(intType);

  // Identity conversion for types we don't need to convert
  addConversion([](Type type) { return type; });

  // Convert felt.type → tensor<!pf>
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isFeltType(type))
      return std::nullopt;
    return RankedTensorType::get({}, elementType);
  });

  // Convert !array.type<N x felt> → tensor<N x !pf>
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isArrayType(type))
      return std::nullopt;
    return RankedTensorType::get(getArrayDimensions(type), elementType);
  });

  // Convert !struct.type<@Name> → tensor<M x !pf> (flattened)
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isStructType(type))
      return std::nullopt;
    auto flatSize = getStructFlattenedSize(type);
    if (!flatSize)
      // Default to dynamic if struct not registered
      return RankedTensorType::get({ShapedType::kDynamic}, elementType);
    return RankedTensorType::get({*flatSize}, elementType);
  });

  // Convert function types recursively
  addConversion([this](FunctionType type) -> Type {
    SmallVector<Type> inputs, results;
    for (Type t : type.getInputs())
      inputs.push_back(convertType(t));
    for (Type t : type.getResults())
      results.push_back(convertType(t));
    return FunctionType::get(type.getContext(), inputs, results);
  });

  // Source materialization: convert from target type back to source type
  addSourceMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  // Target materialization: convert from source type to target type
  addTargetMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

std::optional<int64_t>
LlzkToStablehloTypeConverter::getStructFlattenedSize(Type structType) const {
  auto it = structFlattenedSizes.find(structType);
  if (it == structFlattenedSizes.end())
    return std::nullopt;
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
  if (it == fieldOffsets.end())
    return std::nullopt;
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
  if (usePrimeFieldType)
    return primeFieldType.getStorageType();
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
