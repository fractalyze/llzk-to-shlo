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
#include <fstream>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "bench/m3/json_input.h"
#include "gtest/gtest.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/literal.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace llzk_to_shlo::bench_m3 {
namespace {

// Builds a tiny single-computation HloModule whose entry takes the given
// parameter shapes and returns the first parameter unchanged. We only need
// the entry's parameter list for ParseInputLiteralsFromJson; the body shape
// doesn't matter.
std::unique_ptr<zkx::HloModule>
BuildModuleWithParams(absl::Span<const zkx::Shape> shapes) {
  auto builder = zkx::HloComputation::Builder("entry");
  zkx::HloInstruction *first = nullptr;
  for (int64_t i = 0; i < static_cast<int64_t>(shapes.size()); ++i) {
    auto *p = builder.AddInstruction(zkx::HloInstruction::CreateParameter(
        i, shapes[i], absl::StrCat("p", i)));
    if (i == 0) {
      first = p;
    }
  }
  auto module =
      std::make_unique<zkx::HloModule>("test_mod", zkx::HloModuleConfig());
  module->AddEntryComputation(builder.Build(first));
  return module;
}

std::string WriteTempJson(const std::string &name, const std::string &content) {
  std::string path =
      tsl::io::JoinPath(testing::TempDir(), absl::StrCat(name, ".json"));
  std::ofstream ofs(path);
  ofs << content;
  ofs.close();
  return path;
}

TEST(LiteralFromDecStringsTest, ScalarU32) {
  zkx::Shape shape = zkx::ShapeUtil::MakeScalarShape(zkx::U32);
  TF_ASSERT_OK_AND_ASSIGN(zkx::Literal lit,
                          LiteralFromDecStrings(shape, {"42"}));
  EXPECT_EQ(lit.data<uint32_t>()[0], 42u);
}

TEST(LiteralFromDecStringsTest, OneDimBitArray) {
  zkx::Shape shape = zkx::ShapeUtil::MakeShape(zkx::PRED, {4});
  TF_ASSERT_OK_AND_ASSIGN(zkx::Literal lit,
                          LiteralFromDecStrings(shape, {"0", "1", "1", "0"}));
  auto data = lit.data<bool>();
  EXPECT_EQ(data[0], false);
  EXPECT_EQ(data[1], true);
  EXPECT_EQ(data[2], true);
  EXPECT_EQ(data[3], false);
}

TEST(LiteralFromDecStringsTest, OneDimU32Array) {
  zkx::Shape shape = zkx::ShapeUtil::MakeShape(zkx::U32, {3});
  TF_ASSERT_OK_AND_ASSIGN(zkx::Literal lit,
                          LiteralFromDecStrings(shape, {"7", "11", "13"}));
  auto data = lit.data<uint32_t>();
  EXPECT_EQ(data[0], 7u);
  EXPECT_EQ(data[1], 11u);
  EXPECT_EQ(data[2], 13u);
}

TEST(LiteralFromDecStringsTest, ElementCountMismatchIsError) {
  zkx::Shape shape = zkx::ShapeUtil::MakeShape(zkx::U32, {3});
  EXPECT_EQ(LiteralFromDecStrings(shape, {"7", "11"}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// JSON-string element path: prime field values that don't fit in an IEEE 754
// double round-trip as JSON strings rather than numbers, so the parser must
// honor `is_string()` elements and pass them straight to
// NativeTypeFromDecString.
TEST(ParseInputLiteralsFromJsonTest, StringElementForLargeIntegers) {
  zkx::Shape p0 = zkx::ShapeUtil::MakeScalarShape(zkx::U64);
  auto module = BuildModuleWithParams({p0});

  std::string path =
      WriteTempJson("string_elem", R"({"big": "12345678901234"})");
  TF_ASSERT_OK_AND_ASSIGN(auto lits, ParseInputLiteralsFromJson(*module, path));
  ASSERT_EQ(lits.size(), 1u);
  EXPECT_EQ(lits[0].data<uint64_t>()[0], 12345678901234ull);
}

TEST(ParseInputLiteralsFromJsonTest, KeyedMappedPositionally) {
  // 2-param module: p0=tensor<4xPRED> (bit array), p1=scalar U32. Fixture
  // keys are "in" then "key" — must map to params 0 and 1 in insertion order.
  zkx::Shape p0 = zkx::ShapeUtil::MakeShape(zkx::PRED, {4});
  zkx::Shape p1 = zkx::ShapeUtil::MakeScalarShape(zkx::U32);
  auto module = BuildModuleWithParams({p0, p1});

  std::string path =
      WriteTempJson("keyed_positional", R"({"in": [1, 0, 1, 1], "key": 42})");
  TF_ASSERT_OK_AND_ASSIGN(auto lits, ParseInputLiteralsFromJson(*module, path));
  ASSERT_EQ(lits.size(), 2u);
  EXPECT_EQ(lits[0].data<bool>()[0], true);
  EXPECT_EQ(lits[0].data<bool>()[1], false);
  EXPECT_EQ(lits[0].data<bool>()[2], true);
  EXPECT_EQ(lits[0].data<bool>()[3], true);
  EXPECT_EQ(lits[1].data<uint32_t>()[0], 42u);
}

TEST(ParseInputLiteralsFromJsonTest, KeyOrderMatters) {
  // Same module shape as above, but JSON keys swapped. p0 expects 4 elements
  // — the scalar value (1 element) won't fit, so we get InvalidArgument.
  zkx::Shape p0 = zkx::ShapeUtil::MakeShape(zkx::PRED, {4});
  zkx::Shape p1 = zkx::ShapeUtil::MakeScalarShape(zkx::U32);
  auto module = BuildModuleWithParams({p0, p1});

  std::string path =
      WriteTempJson("keys_swapped", R"({"key": 42, "in": [1, 0, 1, 1]})");
  EXPECT_EQ(ParseInputLiteralsFromJson(*module, path).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ParseInputLiteralsFromJsonTest, ParamCountMismatch) {
  zkx::Shape p0 = zkx::ShapeUtil::MakeShape(zkx::U32, {2});
  auto module = BuildModuleWithParams({p0});

  std::string path = WriteTempJson("too_many_keys", R"({"a": [1, 2], "b": 3})");
  EXPECT_EQ(ParseInputLiteralsFromJson(*module, path).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ParseInputLiteralsFromJsonTest, MissingFileReturnsNotFound) {
  zkx::Shape p0 = zkx::ShapeUtil::MakeScalarShape(zkx::U32);
  auto module = BuildModuleWithParams({p0});
  EXPECT_EQ(
      ParseInputLiteralsFromJson(*module, "/tmp/_does_not_exist_m3_test.json")
          .status()
          .code(),
      absl::StatusCode::kNotFound);
}

TEST(ParseInputLiteralsFromJsonTest, NestedArrayRejected) {
  zkx::Shape p0 = zkx::ShapeUtil::MakeShape(zkx::U32, {4});
  auto module = BuildModuleWithParams({p0});

  std::string path = WriteTempJson("nested", R"({"in": [[1, 2], [3, 4]]})");
  EXPECT_EQ(ParseInputLiteralsFromJson(*module, path).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ParseInputLiteralsFromJsonTest, MalformedJsonReturnsInvalidArgument) {
  zkx::Shape p0 = zkx::ShapeUtil::MakeScalarShape(zkx::U32);
  auto module = BuildModuleWithParams({p0});

  std::string path = WriteTempJson("malformed", R"({"in": )");
  EXPECT_EQ(ParseInputLiteralsFromJson(*module, path).status().code(),
            absl::StatusCode::kInvalidArgument);
}

} // namespace
} // namespace llzk_to_shlo::bench_m3
