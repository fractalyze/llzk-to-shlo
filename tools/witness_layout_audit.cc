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

// witness_layout_audit — per-chunk inspection of a lowered StableHLO module's
// witness-output assembly.
//
// Walks the `stablehlo.dynamic_update_slice` chain feeding the target
// function's `func.return` and reports each chunk's start indices, length,
// source op kind, and whether the value is a silent splat-zero. The
// splat-zero flag is the diagnostic signal for the AES-class debug-drift
// bug: an upstream pass orphans a struct member's wire, the witness-output
// assembly silently emits `stablehlo.constant dense<0>` in that chunk, and
// the gate fails downstream with no signal pointing at the missing wire.
// Companion to PR #67's loud-failure assertion in `StructWriteMPattern` —
// the assertion fails the build at a single offender; this tool inspects
// the lowered output post-hoc and lists every chunk on a chip-wide map.
//
// Two output formats:
//   --format=human  (default) human-readable table
//   --format=json   single-line JSON object for differential testing

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

struct OpDescription {
  std::string kind;
  std::string details;
};

struct ChunkInfo {
  llvm::SmallVector<int64_t, 4> startIndices;
  llvm::SmallVector<int64_t, 4> updateShape;
  int64_t length = 0;
  std::string sourceOpKind;
  std::string sourceOpDetails;
  bool isSplatZero = false;
};

std::optional<int64_t> extractScalarConstant(mlir::Value v) {
  auto cst = v.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst)
    return std::nullopt;
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!attr || !attr.getElementType().isInteger())
    return std::nullopt;
  llvm::APInt value = attr.getSplatValue<llvm::APInt>();
  // `getSExtValue` asserts on bitwidth > 64; bail loudly on malformed input.
  // (Same trap the codebase has documented for `FeltConstPattern`.)
  if (value.getSignificantBits() > 64)
    return std::nullopt;
  return value.getSExtValue();
}

mlir::Value lookThroughReshapes(mlir::Value v) {
  while (v) {
    auto reshape = v.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!reshape)
      break;
    v = reshape.getOperand();
  }
  return v;
}

OpDescription describeSourceOp(mlir::Value canonical) {
  OpDescription out;
  if (!canonical) {
    out.kind = "<null>";
    return out;
  }
  mlir::Operation *def = canonical.getDefiningOp();
  if (!def) {
    out.kind = "<block-arg>";
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(canonical))
      out.details =
          (llvm::Twine("arg#") + llvm::Twine(blockArg.getArgNumber())).str();
    return out;
  }
  out.kind = def->getName().getStringRef().str();
  if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(def)) {
    out.details = (llvm::Twine("callee=@") + callOp.getCallee()).str();
  } else if (auto cstOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(cstOp.getValue());
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
collectChunks(mlir::func::FuncOp fn, llvm::raw_ostream &errs) {
  if (fn.empty()) {
    errs << "error: function @" << fn.getName() << " has no body\n";
    return std::nullopt;
  }
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(fn.front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) {
    errs << "error: function @" << fn.getName()
         << " must terminate with a single-operand func.return; got "
         << (returnOp ? returnOp.getNumOperands() : 0u) << " operands\n";
    return std::nullopt;
  }
  llvm::SmallVector<ChunkInfo, 16> chunks;
  mlir::Value cur = returnOp.getOperand(0);
  while (cur) {
    auto dus = cur.getDefiningOp<mlir::stablehlo::DynamicUpdateSliceOp>();
    if (!dus)
      break;
    ChunkInfo info;
    info.startIndices.reserve(dus.getStartIndices().size());
    for (mlir::Value idx : dus.getStartIndices()) {
      auto v = extractScalarConstant(idx);
      info.startIndices.push_back(v.value_or(-1));
    }
    mlir::Value upd = dus.getUpdate();
    if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(upd.getType())) {
      info.updateShape.assign(rt.getShape().begin(), rt.getShape().end());
      info.length = mlir::llzk_to_shlo::getStaticShapeProduct(rt);
    }
    mlir::Value canonical = lookThroughReshapes(upd);
    info.isSplatZero =
        canonical && mlir::llzk_to_shlo::isZeroSplatConstant(canonical);
    OpDescription desc = describeSourceOp(canonical);
    info.sourceOpKind = std::move(desc.kind);
    info.sourceOpDetails = std::move(desc.details);
    chunks.push_back(std::move(info));
    cur = dus.getOperand();
  }
  std::reverse(chunks.begin(), chunks.end());
  return chunks;
}

size_t countSplatZeros(llvm::ArrayRef<ChunkInfo> chunks) {
  size_t count = 0;
  for (const auto &c : chunks)
    if (c.isSplatZero)
      ++count;
  return count;
}

void appendIndexList(llvm::raw_ostream &os, llvm::ArrayRef<int64_t> values) {
  os << "[";
  for (size_t k = 0; k < values.size(); ++k) {
    if (k)
      os << ",";
    os << values[k];
  }
  os << "]";
}

void printHuman(llvm::StringRef inputPath, llvm::StringRef funcName,
                int64_t totalLength, llvm::ArrayRef<ChunkInfo> chunks,
                size_t splatZeroCount, llvm::raw_ostream &os) {
  os << "witness_layout_audit  file=" << inputPath << "  func=@" << funcName
     << "  chunks=" << chunks.size() << "  total_length=" << totalLength
     << "  splat_zero=" << splatZeroCount << "\n";
  os << "  idx | start_indices  | length | splat?   | source\n";
  os << "  ----+----------------+--------+----------+-----------------------"
        "----------\n";
  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto &c = chunks[i];
    std::string sis;
    {
      llvm::raw_string_ostream s(sis);
      appendIndexList(s, c.startIndices);
    }
    os << "  " << llvm::format("%3zu", i) << " | "
       << llvm::format("%-14s", sis.c_str()) << " | "
       << llvm::format_decimal(c.length, 6) << " | "
       << (c.isSplatZero ? "ZERO!!!  " : "non-zero ") << " | "
       << c.sourceOpKind;
    if (!c.sourceOpDetails.empty())
      os << " (" << c.sourceOpDetails << ")";
    os << "\n";
  }
}

void escapeJsonString(llvm::StringRef s, llvm::raw_ostream &os) {
  for (char ch : s) {
    switch (ch) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << ch;
      break;
    }
  }
}

void printJson(llvm::StringRef inputPath, llvm::StringRef funcName,
               int64_t totalLength, llvm::ArrayRef<ChunkInfo> chunks,
               size_t splatZeroCount, llvm::raw_ostream &os) {
  os << "{\"file\":\"";
  escapeJsonString(inputPath, os);
  os << "\",\"func\":\"";
  escapeJsonString(funcName, os);
  os << "\",\"total_length\":" << totalLength
     << ",\"splat_zero_count\":" << splatZeroCount << ",\"chunks\":[";
  for (size_t i = 0; i < chunks.size(); ++i) {
    if (i)
      os << ",";
    const auto &c = chunks[i];
    os << "{\"index\":" << i << ",\"start_indices\":";
    appendIndexList(os, c.startIndices);
    os << ",\"update_shape\":";
    appendIndexList(os, c.updateShape);
    os << ",\"length\":" << c.length
       << ",\"is_splat_zero\":" << (c.isSplatZero ? "true" : "false")
       << ",\"source_op_kind\":\"";
    escapeJsonString(c.sourceOpKind, os);
    os << "\",\"source_op_details\":\"";
    escapeJsonString(c.sourceOpDetails, os);
    os << "\"}";
  }
  os << "]}\n";
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::OptionCategory cat("witness_layout_audit options");
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(cat));
  llvm::cl::opt<std::string> format(
      "format", llvm::cl::desc("Output format: human|json"),
      llvm::cl::init("human"), llvm::cl::cat(cat));
  llvm::cl::opt<std::string> funcName(
      "func", llvm::cl::desc("Function to audit (default: main)"),
      llvm::cl::init("main"), llvm::cl::cat(cat));

  llvm::cl::HideUnrelatedOptions(cat);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "witness_layout_audit — inspect the dynamic_update_slice chain that\n"
      "feeds a function's func.return and report each chunk's offset, length,\n"
      "splat-zero state, and source op kind. Use to triage silent-zero\n"
      "fallbacks in lowered StableHLO output (cf. AES debug-drift).\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::prime_ir::field::FieldDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string errorMsg;
  auto file = mlir::openInputFile(inputFilename, &errorMsg);
  if (!file) {
    llvm::errs() << "error: " << errorMsg << "\n";
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!module) {
    llvm::errs() << "error: failed to parse " << inputFilename << "\n";
    return 1;
  }

  auto targetFn = module->lookupSymbol<mlir::func::FuncOp>(funcName);
  if (!targetFn) {
    llvm::errs() << "error: function @" << funcName << " not found in "
                 << inputFilename << "\n";
    return 1;
  }

  auto chunks = collectChunks(targetFn, llvm::errs());
  if (!chunks)
    return 1;

  int64_t totalLength = 0;
  if (targetFn.getNumResults() == 1)
    if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
            targetFn.getFunctionType().getResult(0)))
      totalLength = mlir::llzk_to_shlo::getStaticShapeProduct(rt);

  size_t splatZeroCount = countSplatZeros(*chunks);
  if (format == "json") {
    printJson(inputFilename, funcName, totalLength, *chunks, splatZeroCount,
              llvm::outs());
  } else if (format == "human") {
    printHuman(inputFilename, funcName, totalLength, *chunks, splatZeroCount,
               llvm::outs());
  } else {
    llvm::errs() << "error: unknown --format=" << format
                 << " (must be 'human' or 'json')\n";
    return 2;
  }
  return 0;
}
