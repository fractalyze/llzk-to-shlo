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

#include "llzk_to_shlo/Util/WitnessChunkWalker.h"

#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Twine.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

std::optional<int64_t> extractScalarConstant(Value v) {
  auto cst = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst)
    return std::nullopt;
  auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!attr || !attr.getElementType().isInteger())
    return std::nullopt;
  llvm::APInt value = attr.getSplatValue<llvm::APInt>();
  // `getSExtValue` asserts on bitwidth > 64; bail loudly on malformed input.
  // (Same trap the codebase has documented for `FeltConstPattern`.)
  if (value.getSignificantBits() > 64)
    return std::nullopt;
  return value.getSExtValue();
}

Value lookThroughReshapes(Value v) {
  while (v) {
    auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      break;
    v = reshape.getOperand();
  }
  return v;
}

OpDescription describeSourceOp(Value canonical) {
  OpDescription out;
  if (!canonical) {
    out.kind = "<null>";
    return out;
  }
  Operation *def = canonical.getDefiningOp();
  if (!def) {
    out.kind = "<block-arg>";
    if (auto blockArg = dyn_cast<BlockArgument>(canonical))
      out.details =
          (llvm::Twine("arg#") + llvm::Twine(blockArg.getArgNumber())).str();
    return out;
  }
  out.kind = def->getName().getStringRef().str();
  if (auto callOp = dyn_cast<func::CallOp>(def)) {
    out.details = (llvm::Twine("callee=@") + callOp.getCallee()).str();
  } else if (auto cstOp = dyn_cast<stablehlo::ConstantOp>(def)) {
    auto attr = dyn_cast<DenseElementsAttr>(cstOp.getValue());
    if (attr && attr.isSplat() && attr.getElementType().isInteger()) {
      llvm::raw_string_ostream os(out.details);
      os << "splat=" << attr.getSplatValue<llvm::APInt>();
    } else {
      out.details = "non-splat-or-non-int-constant";
    }
  }
  return out;
}

std::optional<llvm::SmallVector<ChunkInfo, 16>>
collectChunks(func::FuncOp fn, raw_ostream &errs) {
  if (fn.empty()) {
    errs << "error: function @" << fn.getName() << " has no body\n";
    return std::nullopt;
  }
  auto returnOp = dyn_cast<func::ReturnOp>(fn.front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) {
    errs << "error: function @" << fn.getName()
         << " must terminate with a single-operand func.return; got "
         << (returnOp ? returnOp.getNumOperands() : 0u) << " operands\n";
    return std::nullopt;
  }
  llvm::SmallVector<ChunkInfo, 16> chunks;
  Value cur = returnOp.getOperand(0);
  while (cur) {
    auto dus = cur.getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
    if (!dus)
      break;
    ChunkInfo info;
    info.startIndices.reserve(dus.getStartIndices().size());
    for (Value idx : dus.getStartIndices()) {
      auto v = extractScalarConstant(idx);
      info.startIndices.push_back(v.value_or(-1));
    }
    Value upd = dus.getUpdate();
    if (auto rt = dyn_cast<RankedTensorType>(upd.getType())) {
      info.updateShape.assign(rt.getShape().begin(), rt.getShape().end());
      info.length = getStaticShapeProduct(rt);
    }
    Value canonical = lookThroughReshapes(upd);
    info.isSplatZero = canonical && isZeroSplatConstant(canonical);
    OpDescription desc = describeSourceOp(canonical);
    info.sourceOpKind = std::move(desc.kind);
    info.sourceOpDetails = std::move(desc.details);
    chunks.push_back(std::move(info));
    cur = dus.getOperand();
  }
  std::reverse(chunks.begin(), chunks.end());
  return chunks;
}

} // namespace mlir::llzk_to_shlo
