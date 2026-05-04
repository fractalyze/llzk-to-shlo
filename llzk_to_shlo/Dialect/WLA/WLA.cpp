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

#include "llzk_to_shlo/Dialect/WLA/WLA.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llzk_to_shlo/Dialect/WLA/WLADialect.cpp.inc"
#include "llzk_to_shlo/Dialect/WLA/WLAEnums.cpp.inc"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "llzk_to_shlo/Dialect/WLA/WLAAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "llzk_to_shlo/Dialect/WLA/WLAOps.cpp.inc"

namespace mlir::llzk_to_shlo::wla {

void WLADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "llzk_to_shlo/Dialect/WLA/WLAOps.cpp.inc" // NOLINT(build/include)
      >();
  registerAttributes();
}

void WLADialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "llzk_to_shlo/Dialect/WLA/WLAAttrs.cpp.inc" // NOLINT(build/include)
      >();
}

mlir::LogicalResult LayoutOp::verify() {
  for (auto attr : getSignals()) {
    auto signal = mlir::dyn_cast<SignalAttr>(attr);
    if (!signal) {
      return emitOpError(
          "`signals` array must contain only `#wla.signal` entries");
    }
    if (signal.getLength() <= 0) {
      return emitOpError("signal `")
             << signal.getName() << "` has non-positive length "
             << signal.getLength();
    }
    if (signal.getOffset() < 0) {
      return emitOpError("signal `")
             << signal.getName() << "` has negative offset "
             << signal.getOffset();
    }
  }
  return mlir::success();
}

} // namespace mlir::llzk_to_shlo::wla
