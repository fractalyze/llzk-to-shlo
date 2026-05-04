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

#ifndef LLZK_TO_SHLO_UTIL_WITNESSCHUNKWALKER_H_
#define LLZK_TO_SHLO_UTIL_WITNESSCHUNKWALKER_H_

// Walker for the terminal `stablehlo.dynamic_update_slice` chain feeding the
// `func.return` of a witness-output function. Lifted from
// `tools/witness_layout_audit.cc` so the audit tool and the
// `--verify-witness-layout` pass share a single inspection routine. Both
// consumers reduce a lowered StableHLO module to the per-signal chunk map and
// flag silent splat-zero fallbacks (the AES-class debug-drift signature
// documented in `docs/WITNESS_LAYOUT_ANCHOR.md`).

#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

namespace mlir::llzk_to_shlo {

// One slice of a `stablehlo.dynamic_update_slice` chain. `startIndices` carry
// each statically known start position (or -1 if the index op was not a
// `stablehlo.constant`); `updateShape` and `length` describe the inserted
// tensor's shape and flat element count; `sourceOpKind`/`sourceOpDetails`
// describe the op that produced the inserted value (after looking through
// `stablehlo.reshape` chains); `isSplatZero` is true iff the canonical source
// is a splat-zero constant — the silent-fallback signal.
struct ChunkInfo {
  llvm::SmallVector<int64_t, 4> startIndices;
  llvm::SmallVector<int64_t, 4> updateShape;
  int64_t length = 0;
  std::string sourceOpKind;
  std::string sourceOpDetails;
  bool isSplatZero = false;
};

struct OpDescription {
  std::string kind;
  std::string details;
};

// Returns the scalar integer value of `v` if `v` is a `stablehlo.constant`
// whose payload is a splat integer fitting in 64 bits; std::nullopt otherwise.
std::optional<int64_t> extractScalarConstant(Value v);

// Walks back through any number of `stablehlo.reshape` ops starting at `v`
// and returns the underlying value.
Value lookThroughReshapes(Value v);

// Summarizes the op that defines `canonical` (block-arg if none).
OpDescription describeSourceOp(Value canonical);

// Walks the `stablehlo.dynamic_update_slice` chain feeding `fn`'s
// single-operand `func.return`. Returns chunks ordered from the deepest base
// (offset 0) toward the return value. Returns std::nullopt and writes a
// diagnostic to `errs` if the function has no body or the terminator is
// malformed (no `func.return`, or with a result count other than 1).
std::optional<llvm::SmallVector<ChunkInfo, 16>>
collectChunks(func::FuncOp fn, raw_ostream &errs);

} // namespace mlir::llzk_to_shlo

#endif // LLZK_TO_SHLO_UTIL_WITNESSCHUNKWALKER_H_
