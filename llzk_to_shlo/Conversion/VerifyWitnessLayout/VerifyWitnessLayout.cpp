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

// Canonical block-order rank: circom's flat witness is laid out
// `[const, outputs, inputs, internals]`, so outputs sort before inputs
// before internals.
int blockOrderRank(wla::SignalKind kind) {
  switch (kind) {
  case wla::SignalKind::Output:
    return 0;
  case wla::SignalKind::Input:
    return 1;
  case wla::SignalKind::Internal:
    return 2;
  }
  llvm_unreachable("unhandled wla::SignalKind");
}

// Enforces the cross-entry invariants the anchor pass emits and this pass
// owns (see docs/contracts/witness-layout-anchor.md): signals sorted
// ascending by offset, non-overlapping, and in canonical
// output*/input*/internal* block order after an optional `const_one`
// internal head. These are properties of the spec alone, independent of
// @main, so they run before the per-chunk match. Emits a diagnostic and
// returns false on the first violation.
bool verifyCrossEntryInvariants(wla::LayoutOp layoutOp) {
  wla::SignalAttr prev;
  int prevRank = -1;
  bool first = true;
  bool sawConstOne = false;
  for (Attribute attr : layoutOp.getSignals()) {
    auto sig = dyn_cast<wla::SignalAttr>(attr);
    if (!sig)
      continue; // op verifier enforces element type; defense in depth.

    // Invariant 4: the reserved constant-1 wire is unique and, when present,
    // heads the layout as `internal` at offset 0, length 1. The block-order
    // check below exempts the head by name+kind, so a malformed `const_one`
    // would otherwise be a silent-accept.
    bool isConstOne = sig.getName() == "const_one";
    if (isConstOne) {
      if (sawConstOne) {
        layoutOp.emitOpError() << "layout has more than one `const_one` "
                                  "signal; the reserved constant-1 wire is "
                                  "unique";
        return false;
      }
      sawConstOne = true;
      if (!first) {
        layoutOp.emitOpError() << "`const_one` must be the first layout entry";
        return false;
      }
      if (sig.getKind() != wla::SignalKind::Internal) {
        layoutOp.emitOpError()
            << "`const_one` head must be `internal`-kind, got kind="
            << wla::stringifySignalKind(sig.getKind());
        return false;
      }
      if (sig.getOffset() != 0) {
        layoutOp.emitOpError()
            << "`const_one` head must be at offset 0, got offset "
            << sig.getOffset();
        return false;
      }
      if (sig.getLength() != 1) {
        layoutOp.emitOpError()
            << "`const_one` head must have length 1, got length "
            << sig.getLength();
        return false;
      }
    }

    // The reserved constant-1 wire is `internal`-kind but heads the layout
    // by design; exempt it from block order (sort/overlap still apply).
    // Invariant 4 above guarantees a leading `const_one` is a valid head.
    bool isConstOneHead = first && isConstOne;
    first = false;

    if (prev) {
      int64_t prevEndOffset = prev.getOffset() + prev.getLength();
      if (sig.getOffset() < prev.getOffset()) {
        layoutOp.emitOpError()
            << "signals must be sorted by ascending offset: signal `"
            << sig.getName() << "` at offset " << sig.getOffset()
            << " follows offset " << prev.getOffset();
        return false;
      }
      if (prevEndOffset > sig.getOffset()) {
        layoutOp.emitOpError()
            << "signals overlap: signal `" << prev.getName() << "` occupies ["
            << prev.getOffset() << ", " << prevEndOffset << ") but signal `"
            << sig.getName() << "` starts at offset " << sig.getOffset();
        return false;
      }
    }

    if (!isConstOneHead) {
      int rank = blockOrderRank(sig.getKind());
      if (rank < prevRank) {
        layoutOp.emitOpError()
            << "signal `" << sig.getName()
            << "` (kind=" << wla::stringifySignalKind(sig.getKind())
            << ") breaks canonical block order; entries must run output*, "
               "input*, internal* after the optional const_one head";
        return false;
      }
      prevRank = rank;
    }

    prev = sig;
  }
  return true;
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

    if (!verifyCrossEntryInvariants(layoutOp))
      return signalPassFailure();

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

      // 5th cross-entry invariant: an `input` chunk must copy a function
      // parameter straight in (a block arg, through reshapes), not a
      // constant/compute. The `!isSplatZero` guard defers a splat-zero input to
      // the more precise orphaned-wire diagnostic above.
      if (signal.getKind() == wla::SignalKind::Input &&
          !covering->isSplatZero && !covering->sourceIsBlockArg) {
        layoutOp.emitOpError()
            << "input signal `" << signal.getName() << "` (offset=" << sigStart
            << ", length=" << signal.getLength()
            << ") does not source from a function parameter; its covering "
               "chunk's value is `"
            << covering->sourceOpKind << "`";
        failed = true;
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
