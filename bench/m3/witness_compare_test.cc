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

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "bench/m3/witness_compare.h"
#include "circom/wtns/wtns.h"
#include "circom/wtns/wtns_test_utils.h"
#include "gtest/gtest.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/literal.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace llzk_to_shlo::bench_m3 {
namespace {

using ::llzk_to_shlo::circom::testing_internal::AppendLE;

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

// Constraint-only circuits (no `signal output`, e.g.
// iden3_verify_credential_subject) lower to tensor<0>. The comparator must
// vacuously pass — and crucially must not divide by zero in the per-element
// byte-size derivation.
TEST(WitnessCompareTest, VacuousGateOnZeroElementOutput) {
  std::string path = WriteSyntheticWtns8B({1, 60}, "vacuous");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal empty(zkx::ShapeUtil::MakeShape(zkx::U64, {0}));
  std::vector<int64_t> idx; // empty — must match num_elements=0.
  EXPECT_TRUE(CompareLiteralToWtns(empty, wtns, idx).ok());
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

// `prefix_size` opts into output-only / partial gating: only [0, prefix_size)
// of the literal is byte-compared, and the index list must match prefix_size
// exactly. Models the AES `aes_256_encrypt` shape: tensor<14048> literal whose
// first 128 elements (`@out`) are byte-equal to the circom witness.
TEST(WitnessCompareTest, PrefixSizeMatchesPartialPrefix) {
  std::string path =
      WriteSyntheticWtns8B({1, 60, 3, 4, 5, 12, 99, 99}, "prefix_partial");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  // 6-element literal; only first 2 elements are expected to byte-match.
  // Trailing 4 positions intentionally hold garbage that would fail under the
  // strict full-literal compare — the prefix gate must not look at them.
  zkx::Literal lit = MakeU64Literal({60, 3, 7, 7, 7, 7});
  std::vector<int64_t> idx = {1, 2};
  EXPECT_TRUE(CompareLiteralToWtns(lit, wtns, idx, /*prefix_size=*/2).ok());
}

TEST(WitnessCompareTest, PrefixSizeEqualsNumElementsBehavesAsStrict) {
  std::string path = WriteSyntheticWtns8B({1, 60, 3, 4}, "prefix_full");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60, 3});
  std::vector<int64_t> idx = {1, 2};
  EXPECT_TRUE(CompareLiteralToWtns(lit, wtns, idx, /*prefix_size=*/2).ok());
}

TEST(WitnessCompareTest, PrefixSizeGreaterThanNumElementsIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60, 3}, "prefix_overflow");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60}); // 1 element only
  std::vector<int64_t> idx = {1, 2};
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx, /*prefix_size=*/2).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(WitnessCompareTest, PrefixSizeIndexCountMismatchIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60, 3, 4}, "prefix_idxcount");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60, 3, 4, 4});
  std::vector<int64_t> idx = {1, 2, 3}; // 3 indices vs prefix_size=2
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx, /*prefix_size=*/2).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(WitnessCompareTest, PrefixSizeMismatchInPrefixIsDataLoss) {
  std::string path = WriteSyntheticWtns8B({1, 60, 3, 4}, "prefix_mismatch");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit =
      MakeU64Literal({60, 99, 7, 7}); // prefix[1]=99 != wtns[2]=3
  std::vector<int64_t> idx = {1, 2};
  absl::Status s = CompareLiteralToWtns(lit, wtns, idx, /*prefix_size=*/2);
  EXPECT_EQ(s.code(), absl::StatusCode::kDataLoss);
  EXPECT_NE(s.message().find("literal[1]"), std::string::npos) << s.message();
}

TEST(WitnessCompareTest, PrefixSizeNegativeIsInvalidArgument) {
  std::string path = WriteSyntheticWtns8B({1, 60}, "prefix_neg");
  TF_ASSERT_OK_AND_ASSIGN(circom::WitnessFile wtns, circom::ParseWtns(path));
  zkx::Literal lit = MakeU64Literal({60});
  std::vector<int64_t> idx = {1};
  EXPECT_EQ(CompareLiteralToWtns(lit, wtns, idx, /*prefix_size=*/-1).code(),
            absl::StatusCode::kInvalidArgument);
}

} // namespace
} // namespace llzk_to_shlo::bench_m3
