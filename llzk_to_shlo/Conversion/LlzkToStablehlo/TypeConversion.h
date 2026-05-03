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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_TYPECONVERSION_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_TYPECONVERSION_H_

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::llzk_to_shlo {

/// Type converter for LLZK to StableHLO conversion.
///
/// Type mappings:
/// - `!felt.type` -> `tensor<!field.pf<prime>>`
/// - `!array.type<N x felt>` -> `tensor<N x !field.pf<prime>>`
/// - `!struct.type<@Name>` -> `tensor<M x !field.pf<prime>>` (flattened)
class LlzkToStablehloTypeConverter : public TypeConverter {
public:
  explicit LlzkToStablehloTypeConverter(MLIRContext *ctx,
                                        const llvm::APInt &prime,
                                        unsigned storageBitWidth,
                                        bool usePrimeFieldType = true);

  /// Get the prime field type used for conversion.
  prime_ir::field::PrimeFieldType getPrimeFieldType() const {
    return primeFieldType;
  }

  /// Get the element type used for tensor representation.
  Type getElementType() const { return elementType; }

  /// Get the storage type for creating constant attributes.
  /// Returns the underlying integer type used for storing field elements.
  IntegerType getStorageType() const;

  /// Create a DenseElementsAttr for a constant tensor with the appropriate
  /// storage type.
  DenseElementsAttr createConstantAttr(RankedTensorType tensorType,
                                       int64_t value,
                                       OpBuilder &rewriter) const;

  /// APInt overload — preserves field-element values that exceed int64_t
  /// (e.g. LessThan's `1 << 252` offset). The input APInt is zero-extended or
  /// truncated to the storage type's bitwidth before being stored.
  DenseElementsAttr createConstantAttr(RankedTensorType tensorType,
                                       const llvm::APInt &value,
                                       OpBuilder &rewriter) const;

  /// Get the flattened size for a struct type.
  std::optional<int64_t> getStructFlattenedSize(Type structType) const;

  /// Register a struct type with its flattened size.
  void registerStructFlattenedSize(Type structType, int64_t size);

  /// Get the field offset within a flattened struct for a field name.
  std::optional<int64_t> getFieldOffset(Type structType,
                                        StringRef fieldName) const;

  /// Register field offset within a flattened struct.
  void registerFieldOffset(Type structType, StringRef fieldName,
                           int64_t offset);

  /// Check if a type is a felt type (by dialect namespace)
  static bool isFeltType(Type type);

  /// Check if a type is an array type (by dialect namespace)
  static bool isArrayType(Type type);

  /// Check if a type is a struct type (by dialect namespace)
  static bool isStructType(Type type);

private:
  prime_ir::field::PrimeFieldType primeFieldType;
  Type elementType; // Element type used in tensors (i64 or PrimeFieldType)
  bool usePrimeFieldType;
  DenseMap<Type, int64_t> structFlattenedSizes;
  DenseMap<std::pair<Type, StringAttr>, int64_t> fieldOffsets;
};

/// Helper to downcast TypeConverter to LlzkToStablehloTypeConverter.
inline const LlzkToStablehloTypeConverter &
getConverter(const TypeConverter *tc) {
  return *static_cast<const LlzkToStablehloTypeConverter *>(tc);
}

// ===----------------------------------------------------------------------===
// Shared utilities for conversion patterns
// ===----------------------------------------------------------------------===

/// Extract static dimensions from an LLZK array type by parsing its printed
/// representation. Returns {ShapedType::kDynamic} if parsing fails.
/// Example: !array.type<8 x !felt.type> → {8}
SmallVector<int64_t> getArrayDimensions(Type arrayType);

/// Compute the flattened element count for a struct member type.
/// Felt types count as 1, array types count as their product of dimensions.
int64_t getMemberFlatSize(Type memberType);

/// Product of `t`'s static dimensions. Dynamic dimensions are treated as 1.
int64_t getStaticShapeProduct(RankedTensorType t);

/// True if `v` is defined by a `stablehlo.constant` whose attribute is an
/// integer-typed splat with all-zero value.
bool isZeroSplatConstant(Value v);

/// Look through an unrealized_conversion_cast to get the underlying value.
/// If the value is produced by a single-input cast, returns the input.
/// Otherwise returns the original value.
Value lookThroughCast(Value v);

/// Ensure a value has tensor type. If the value is an unconverted LLZK type
/// (e.g., block arg with !array.type or !felt.type), inserts an
/// unrealized_conversion_cast to the converted tensor type.
Value ensureTensorType(OpBuilder &b, Value v, Type originalType,
                       const LlzkToStablehloTypeConverter &tc, Location loc);

/// Convert a value to a 0-d tensor<i32> for use as a StableHLO
/// dynamic_slice/dynamic_update_slice index. Looks through
/// unrealized_conversion_cast and cast.toindex to find or create the
/// appropriate tensor<i32> value. May emit `arith.index_cast` +
/// `tensor.from_elements` when the index is a bare integer/index scalar
/// (e.g. `arith.subi` from `array.len - 1`).
/// `tc` materializes any felt operand that hasn't yet been converted to
/// a tensor — required when `cast.toindex` consumes an scf.while iter-arg
/// whose SCF structural conversion runs after the calling pattern.
/// Returns a null Value when no path matched, signalling the caller to
/// fail its `matchAndRewrite` and let the dialect-conversion driver retry
/// once operand dependencies settle.
Value convertToIndexTensor(OpBuilder &b, Value idx,
                           const LlzkToStablehloTypeConverter &tc,
                           Location loc);

/// Create a scalar i32 constant tensor for StableHLO slice indices.
Value createIndexConstant(OpBuilder &b, Location loc, int64_t value);

/// Parse a bool.cmp predicate attribute into a stablehlo ComparisonDirection.
/// Returns std::nullopt if the predicate cannot be parsed.
std::optional<int64_t> parseBoolCmpPredicate(Attribute predicateAttr);

/// Parse initialized record names from a pod.new operation's properties.
/// Returns the field names listed in the initializedRecords property.
SmallVector<std::string> getPodInitializedRecords(Operation *podNewOp);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_TYPECONVERSION_H_
