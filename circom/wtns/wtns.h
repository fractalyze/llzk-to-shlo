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

#ifndef CIRCOM_WTNS_WTNS_H_
#define CIRCOM_WTNS_WTNS_H_

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace llzk_to_shlo::circom {

// In-memory representation of an iden3 binfileutils v2 .wtns file — the
// witness format emitted by `circom -c` and consumed by snarkjs.
//
// File layout (little-endian throughout):
//   "wtns" (4 B) + version u32 (=2) + num_sections u32
//   per section: type u32 + size u64 + payload[size]
//     type 1 (header): field_size u32 + modulus[field_size] + num_witnesses u32
//     type 2 (data):   num_witnesses * field_size bytes (LE field elements)
//
// The parser is self-contained: no dependency on field-arithmetic types.
// Downstream consumers (e.g. M3 harness's `witness_compare`) compare GPU
// witness output to `data` at the same byte offset — see CLAUDE.md
// "Load-Bearing Invariants" (`batch[i] == single[i]` against circom).
struct WitnessFile {
  uint32_t field_size_bytes = 0; // 32 for bn254 / bls12-381 scalar fields.
  uint32_t num_witnesses = 0;
  std::vector<uint8_t> modulus;
  std::vector<uint8_t> data;

  // Returns the i-th witness as a span of `field_size_bytes` bytes. The
  // caller is responsible for ensuring `i < num_witnesses`.
  absl::Span<const uint8_t> Witness(size_t i) const;
};

// Parses a .wtns file from disk. Returns an error if the file is missing,
// has the wrong magic, declares a version other than 2, has section sizes
// that don't fit the file length, or has a data section whose size doesn't
// match `num_witnesses * field_size_bytes`.
absl::StatusOr<WitnessFile> ParseWtns(std::string_view path);

} // namespace llzk_to_shlo::circom

#endif // CIRCOM_WTNS_WTNS_H_
