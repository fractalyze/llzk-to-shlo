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

#ifndef CIRCOM_WTNS_WTNS_TEST_UTILS_H_
#define CIRCOM_WTNS_WTNS_TEST_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/base/internal/endian.h"

namespace llzk_to_shlo::circom::testing_internal {

// Little-endian append of a 4- or 8-byte integer to `out`. Synthetic .wtns
// blobs (the iden3 v2 format) encode every numeric field as LE u32 / u64;
// hand-rolling the byte order in each test was producing line-for-line
// duplicates between `wtns_test.cc` and downstream consumers like
// `bench/m3/witness_compare_test.cc`.
template <typename T>
void AppendLE(std::vector<uint8_t> &out, T v) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "AppendLE supports u32 and u64 only");
  uint8_t buf[sizeof(T)];
  if constexpr (sizeof(T) == 4) {
    absl::little_endian::Store32(buf, static_cast<uint32_t>(v));
  } else {
    absl::little_endian::Store64(buf, static_cast<uint64_t>(v));
  }
  out.insert(out.end(), buf, buf + sizeof(T));
}

} // namespace llzk_to_shlo::circom::testing_internal

#endif // CIRCOM_WTNS_WTNS_TEST_UTILS_H_
