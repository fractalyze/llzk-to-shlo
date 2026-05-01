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

#ifndef BENCH_M3_WITNESS_COMPARE_H_
#define BENCH_M3_WITNESS_COMPARE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "circom/wtns/wtns.h"
#include "zkx/literal.h"

namespace llzk_to_shlo::bench_m3 {

// Compares the GPU `output` Literal element-by-element to the witnesses at
// `wtns_indices` in `wtns`. Element i of `output` is expected to byte-match
// `wtns.Witness(wtns_indices[i])` exactly (no field-arithmetic conversion).
// circom's .wtns stores standard-form little-endian field elements, and zkx's
// prime-field Literals use the same in-memory representation — that is the
// load-bearing invariant `batch[i] == single[i]` (CLAUDE.md) builds on.
//
// PR-C scope: single-tensor non-tuple outputs at N=1. Tuple outputs and
// batched outputs (N>1) are explicitly out of scope and rejected here.
//
// Constraint-only circuits (no `signal output`) lower to tensor<0>; for those
// the comparator returns OkStatus vacuously — nothing to byte-compare, but the
// chip can still be gated to anchor shape stability against a future lowering
// regression that turns the output into tensor<N>=N>0 (caught by the
// element-count != index-count branch).
//
// Errors:
//   - InvalidArgument: tuple shape, element-count != index-count, per-element
//     byte size != wtns.field_size_bytes, or any index out of [0,
//     num_witnesses).
//   - DataLossError: first byte mismatch. Status message includes the failing
//     literal element index, the corresponding wire index, and a side-by-side
//     hex dump of the field-size bytes.
absl::Status CompareLiteralToWtns(const zkx::Literal &output,
                                  const llzk_to_shlo::circom::WitnessFile &wtns,
                                  absl::Span<const int64_t> wtns_indices);

} // namespace llzk_to_shlo::bench_m3

#endif // BENCH_M3_WITNESS_COMPARE_H_
