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

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "bench/m3/witness_compare.h"
#include "circom/wtns/wtns.h"
#include "gtest/gtest.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/literal.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace llzk_to_shlo::bench_m3 {
namespace {

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

// Builds a synthetic 8-byte-field .wtns containing `witnesses` as low-byte
// little-endian values and writes it to TempDir. Uses the same blob shape as
// circom/wtns/wtns_test.cc's `ParsesSyntheticBlob` so any drift between that
// fixture and this one stays obvious.
std::string WriteSyntheticWtns8B(const std::vector<uint64_t> &witnesses,
                                 absl::string_view name) {
  constexpr uint32_t kFieldSize = 8;
  std::vector<uint8_t> blob = {'w', 't', 'n', 's'};
  AppendLE<uint32_t>(blob, /*version=*/2);
  AppendLE<uint32_t>(blob, /*num_sections=*/2);
  AppendLE<uint32_t>(blob, /*type=*/1);
  AppendLE<uint64_t>(blob, /*size=*/4 + kFieldSize + 4);
  AppendLE<uint32_t>(blob, kFieldSize);
  // 8-byte modulus payload — content unused by witness_compare.
  for (int i = 0; i < kFieldSize; ++i) {
    blob.push_back(static_cast<uint8_t>(0xa0 | i));
  }
  AppendLE<uint32_t>(blob, static_cast<uint32_t>(witnesses.size()));
  AppendLE<uint32_t>(blob, /*type=*/2);
  AppendLE<uint64_t>(blob, witnesses.size() * kFieldSize);
  for (uint64_t w : witnesses) {
    AppendLE<uint64_t>(blob, w);
  }
  std::string path = std::string(::testing::TempDir()) + "/wcmp_" +
                     std::string(name) + ".wtns";
  absl::string_view payload(reinterpret_cast<const char *>(blob.data()),
                            blob.size());
  ABSL_CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), path, payload));
  return path;
}

zkx::Literal MakeU64Literal(const std::vector<uint64_t> &values) {
  zkx::Shape shape = zkx::ShapeUtil::MakeShape(
      zkx::U64, {static_cast<int64_t>(values.size())});
  zkx::Literal lit(shape);
  auto data = lit.data<uint64_t>();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  return lit;
}

// Pins the load-bearing case the gate exercises end-to-end on MontgomeryDouble:
// the GPU output Literal's elements map to NON-CONTIGUOUS .wtns wire indices.
// (MontgomeryDouble's actual fixture decodes to indices [1, 2, 5, 6] —
// witnesses [3, 4] are echoed inputs, so the recipe's "outputs at [1..1+N)"
// shortcut would silently match a wrong slot. Keep this case as the canonical
// regression for the indices API.)
TEST(WitnessCompareTest, MatchesScatteredIndices) {
  std::string path = WriteSyntheticWtns8B(
      {/*0*/ 1, /*1*/ 60, /*2*/ 3, /*3*/ 4, /*4*/ 5, /*5*/ 12}, "scattered");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60, 3, 12});
  std::vector<int64_t> idx = {1, 2, 5};
  EXPECT_TRUE(CompareLiteralToWtns(lit, wtns, idx).ok());
}

TEST(WitnessCompareTest, MatchesContiguousFromIndex1) {
  std::string path = WriteSyntheticWtns8B({1, 60, 3, 4, 5, 12}, "contiguous");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60});
  std::vector<int64_t> idx = {1};
  EXPECT_TRUE(CompareLiteralToWtns(lit, wtns, idx).ok());
}

TEST(WitnessCompareTest, ByteMismatchIsDataLoss) {
  std::string path = WriteSyntheticWtns8B({1, 60}, "bytemismatch");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({99}); // wtns wire 1 = 60.
  std::vector<int64_t> idx = {1};
  absl::Status s = CompareLiteralToWtns(lit, wtns, idx);
  EXPECT_EQ(s.code(), absl::StatusCode::kDataLoss);
  EXPECT_NE(s.message().find("literal[0]"), std::string::npos) << s.message();
  EXPECT_NE(s.message().find("wtns[1]"), std::string::npos) << s.message();
}

TEST(WitnessCompareTest, IndexCountMismatchIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60, 3}, "idxcount");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60, 3}); // 2 elements
  std::vector<int64_t> idx = {1};             // 1 index
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(WitnessCompareTest, IndexOutOfRangeIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60}, "oob");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({0});
  std::vector<int64_t> idx = {99}; // only 2 wires
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(WitnessCompareTest, NegativeIndexIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60}, "neg");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({0});
  std::vector<int64_t> idx = {-1};
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(WitnessCompareTest, FieldSizeMismatchIsInvalidArgument) {
  // 8-byte wtns vs 4-byte literal element.
  std::string path = WriteSyntheticWtns8B({1, 60}, "fieldsize");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Shape shape = zkx::ShapeUtil::MakeShape(zkx::U32, {1});
  zkx::Literal lit(shape);
  lit.data<uint32_t>()[0] = 60;
  std::vector<int64_t> idx = {1};
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(WitnessCompareTest, TupleShapeIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60}, "tuple");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  std::vector<zkx::Literal> elems;
  elems.push_back(MakeU64Literal({60}));
  elems.push_back(MakeU64Literal({60}));
  zkx::Literal tuple = zkx::Literal::MoveIntoTuple(absl::MakeSpan(elems));
  std::vector<int64_t> idx = {1, 1};
  EXPECT_EQ(CompareLiteralToWtns(tuple, wtns, idx).code(),
            absl::StatusCode::kInvalidArgument);
}

} // namespace
} // namespace llzk_to_shlo::bench_m3
