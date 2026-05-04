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

#ifndef LLZK_TO_SHLO_DIALECT_WLA_WLA_H_
#define LLZK_TO_SHLO_DIALECT_WLA_WLA_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Enum decls.
#include "llzk_to_shlo/Dialect/WLA/WLAEnums.h.inc"

// Dialect class.
#include "llzk_to_shlo/Dialect/WLA/WLADialect.h.inc"

// Attribute classes.
#define GET_ATTRDEF_CLASSES
#include "llzk_to_shlo/Dialect/WLA/WLAAttrs.h.inc"

// Op classes.
#define GET_OP_CLASSES
#include "llzk_to_shlo/Dialect/WLA/WLAOps.h.inc"

#endif // LLZK_TO_SHLO_DIALECT_WLA_WLA_H_
