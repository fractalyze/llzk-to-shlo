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

#include "llzk_to_shlo/Conversion/VerifyWitnessLayout/VerifyWitnessLayout.h"

#include <string>

#include "llvm/Support/raw_ostream.h"
#include "llzk_to_shlo/Dialect/WLA/WLA.h"
#include "llzk_to_shlo/Util/WitnessChunkWalker.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_VERIFYWITNESSLAYOUT
#include "llzk_to_shlo/Conversion/VerifyWitnessLayout/VerifyWitnessLayout.h.inc"

namespace {

// Walks `chunks` from latest to earliest emission and returns the first
// (i.e. last-emitted) chunk that fully contains the half-open signal range
// `[sigStart, sigEnd)`. Last-emitted matters: a `dynamic_update_slice` later
// in the chain overwrites earlier chunks at the same offset, so the value
// the witness ends up holding is whatever the most recent covering chunk
// stored. Skips multi-dimensional or dynamic-index chunks (offset == -1)
// — they cannot be statically matched against a flat-tensor offset.
const ChunkInfo *findCoveringChunk(llvm::ArrayRef<ChunkInfo> chunks,
                                   int64_t sigStart, int64_t sigEnd) {
  for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
    if (it->startIndices.size() != 1)
      continue;
    int64_t chunkStart = it->startIndices.front();
    if (chunkStart < 0)
      continue;
    int64_t chunkEnd = chunkStart + it->length;
    if (chunkStart <= sigStart && sigEnd <= chunkEnd)
      return &*it;
  }
  return nullptr;
}

struct VerifyWitnessLayoutPass
    : public impl::VerifyWitnessLayoutBase<VerifyWitnessLayoutPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    wla::LayoutOp layoutOp;
    for (auto op : module.getOps<wla::LayoutOp>()) {
      if (layoutOp) {
        op.emitOpError()
            << "module has multiple `wla.layout` ops; the verify pass "
               "expects at most one";
        return signalPassFailure();
      }
      layoutOp = op;
    }
    // No anchor output to verify against — silent no-op so this pass can
    // ship before `--witness-layout-anchor` is universally wired.
    if (!layoutOp)
      return;

    auto mainFn = module.lookupSymbol<func::FuncOp>("main");
    if (!mainFn) {
      module.emitOpError()
          << "module has `wla.layout` but no `@main` function to verify "
             "against";
      return signalPassFailure();
    }

    std::string chunkErr;
    llvm::raw_string_ostream chunkErrStream(chunkErr);
    auto chunksOpt = collectChunks(mainFn, chunkErrStream);
    if (!chunksOpt) {
      mainFn.emitOpError() << "witness-output `func.return` is malformed: "
                           << chunkErrStream.str();
      return signalPassFailure();
    }

    bool failed = false;
    for (Attribute attr : layoutOp.getSignals()) {
      auto signal = dyn_cast<wla::SignalAttr>(attr);
      if (!signal)
        continue; // op verifier already enforces; defense in depth

      int64_t sigStart = signal.getOffset();
      int64_t sigEnd = sigStart + signal.getLength();

      const ChunkInfo *covering =
          findCoveringChunk(*chunksOpt, sigStart, sigEnd);
      if (!covering) {
        layoutOp.emitOpError()
            << "signal `" << signal.getName()
            << "` (kind=" << wla::stringifySignalKind(signal.getKind())
            << ", offset=" << sigStart << ", length=" << signal.getLength()
            << ") has no covering `dynamic_update_slice` chunk in @main";
        failed = true;
        continue;
      }

      // Splat-zero on `internal` signals is permissible: a legitimately zero
      // internal wire is indistinguishable from an orphan at this stage.
      if (covering->isSplatZero &&
          signal.getKind() != wla::SignalKind::Internal) {
        layoutOp.emitOpError()
            << "signal `" << signal.getName()
            << "` (kind=" << wla::stringifySignalKind(signal.getKind())
            << ", offset=" << sigStart << ", length=" << signal.getLength()
            << ") is sourced by a splat-zero constant — upstream pass "
               "orphaned this wire";
        failed = true;
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
