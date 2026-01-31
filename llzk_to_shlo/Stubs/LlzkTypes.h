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

// Stub type definitions for LLZK dialects.
// These provide minimal interfaces needed for the conversion pass.
// Once llzk-lib supports Bazel, replace with actual llzk headers.

#ifndef LLZK_TO_SHLO_STUBS_LLZKTYPES_H_
#define LLZK_TO_SHLO_STUBS_LLZKTYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir::llzk {

namespace felt {

// Stub for !felt.type
class FeltType : public Type::TypeBase<FeltType, Type, TypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "felt.type";

  static FeltType get(MLIRContext *context) { return Base::get(context); }

  static bool classof(Type type) {
    return type.getTypeID() == TypeID::get<FeltType>();
  }
};

// Stub dialect
class FeltDialect : public Dialect {
public:
  explicit FeltDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "felt"; }
};

} // namespace felt

namespace array {

// Stub for !array.type<dims x element>
class ArrayType : public Type::TypeBase<ArrayType, Type, TypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "array.type";

  // Get element type
  Type getElementType() const;

  // Get dimension sizes
  ArrayRef<int64_t> getDimensionSizes() const;

  static bool classof(Type type) {
    return type.getTypeID() == TypeID::get<ArrayType>();
  }
};

// Stub dialect
class ArrayDialect : public Dialect {
public:
  explicit ArrayDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "array"; }
};

} // namespace array

namespace component {

// Stub for !struct.type<@Name>
class StructType : public Type::TypeBase<StructType, Type, TypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "struct.type";

  // Get the struct name
  StringRef getName() const;

  static bool classof(Type type) {
    return type.getTypeID() == TypeID::get<StructType>();
  }
};

// Stub dialect
class StructDialect : public Dialect {
public:
  explicit StructDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "struct"; }
};

} // namespace component

} // namespace mlir::llzk

#endif // LLZK_TO_SHLO_STUBS_LLZKTYPES_H_
