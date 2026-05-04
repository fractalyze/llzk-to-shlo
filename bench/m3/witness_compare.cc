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

#include "bench/m3/witness_compare.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "circom/wtns/wtns.h"
#include "zkx/literal.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace llzk_to_shlo::bench_m3 {
namespace {

// 16-bytes-per-row hex like xxd's default, joined with spaces every two bytes
// so a 32-byte field element fits on one terminal line for diff-by-eye.
std::string HexLine(absl::Span<const uint8_t> bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2 + bytes.size() / 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i > 0 && i % 2 == 0) {
      out.push_back(' ');
    }
    out.push_back(kHex[bytes[i] >> 4]);
    out.push_back(kHex[bytes[i] & 0x0f]);
  }
  return out;
}

} // namespace

absl::Status CompareLiteralToWtns(const zkx::Literal &output,
                                  const llzk_to_shlo::circom::WitnessFile &wtns,
                                  absl::Span<const int64_t> wtns_indices,
                                  int64_t prefix_size) {
  const zkx::Shape &shape = output.shape();
  if (!shape.IsArray()) {
    return absl::InvalidArgumentError(
        "witness_compare: tuple/non-array Literals are out of scope");
  }
  const int64_t num_elements = zkx::ShapeUtil::ElementsIn(shape);
  if (prefix_size < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "witness_compare: prefix_size=", prefix_size, " is negative"));
  }
  if (prefix_size > num_elements) {
    return absl::InvalidArgumentError(
        absl::StrCat("witness_compare: prefix_size=", prefix_size,
                     " exceeds literal num_elements=", num_elements));
  }
  // prefix_size == 0 ⇒ strict full-literal semantics for the 25 sister chips.
  // prefix_size > 0 ⇒ output-only / partial gating; only [0, prefix_size) is
  // byte-compared and the index list must match the prefix length exactly.
  const int64_t compare_count = (prefix_size == 0) ? num_elements : prefix_size;
  if (static_cast<int64_t>(wtns_indices.size()) != compare_count) {
    if (prefix_size == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("witness_compare: literal has ", compare_count,
                       " elements but wtns_indices has ", wtns_indices.size()));
    }
    return absl::InvalidArgumentError(
        absl::StrCat("witness_compare: prefix_size=", compare_count,
                     " but wtns_indices has ", wtns_indices.size()));
  }
  // Constraint-only circuits (e.g. iden3_verify_credential_subject) lower to
  // tensor<0> because the template has no `signal output` — only `===`
  // assertions. There is nothing to byte-compare, so vacuously pass; the chip
  // still anchors shape stability via the size-mismatch branch above (a future
  // lowering change to tensor<N>=N>0 would diverge from the empty wtns_indices
  // sentinel and surface there).
  if (compare_count == 0) {
    return absl::OkStatus();
  }
  // Derive per-element byte size from the literal's total footprint instead of
  // ShapeUtil::ByteSizeOfPrimitiveType — the latter would need a special-case
  // for prime-field PrimitiveTypes that this comparison doesn't otherwise care
  // about. The literal owns N field elements packed back-to-back, so the
  // division always rounds exactly.
  const int64_t total_bytes = output.size_bytes();
  if (total_bytes % num_elements != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("witness_compare: literal size_bytes=", total_bytes,
                     " not divisible by num_elements=", num_elements));
  }
  const int64_t per_elem_bytes = total_bytes / num_elements;
  if (per_elem_bytes != static_cast<int64_t>(wtns.field_size_bytes)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "witness_compare: literal element size ", per_elem_bytes,
        " B != wtns.field_size_bytes ", wtns.field_size_bytes, " B"));
  }
  for (int64_t i = 0; i < compare_count; ++i) {
    const int64_t wire = wtns_indices[i];
    if (wire < 0 || static_cast<uint64_t>(wire) >=
                        static_cast<uint64_t>(wtns.num_witnesses)) {
      return absl::InvalidArgumentError(
          absl::StrCat("witness_compare: index ", wire, " (literal element ", i,
                       ") out of range [0, ", wtns.num_witnesses, ")"));
    }
  }
  const uint8_t *base = static_cast<const uint8_t *>(output.untyped_data());
  for (int64_t i = 0; i < compare_count; ++i) {
    absl::Span<const uint8_t> gpu_bytes(base + i * per_elem_bytes,
                                        per_elem_bytes);
    absl::Span<const uint8_t> wtns_bytes = wtns.Witness(wtns_indices[i]);
    if (std::memcmp(gpu_bytes.data(), wtns_bytes.data(), per_elem_bytes) != 0) {
      return absl::DataLossError(
          absl::StrCat("witness_compare: literal[", i, "] != wtns[",
                       wtns_indices[i], "]\n  gpu : ", HexLine(gpu_bytes),
                       "\n  wtns: ", HexLine(wtns_bytes)));
    }
  }
  return absl::OkStatus();
}

} // namespace llzk_to_shlo::bench_m3
