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
#include "absl/strings/string_view.h"
#include "circom/wtns/wtns.h"
#include "gtest/gtest.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"

namespace llzk_to_shlo::circom {
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

std::string WriteTempWtns(const std::vector<uint8_t> &bytes,
                          absl::string_view name) {
  std::string path =
      std::string(::testing::TempDir()) + "/" + std::string(name) + ".wtns";
  absl::string_view payload(reinterpret_cast<const char *>(bytes.data()),
                            bytes.size());
  ABSL_CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), path, payload));
  return path;
}

TEST(WtnsTest, ParsesSyntheticBlob) {
  // 8-byte field, 3 witnesses (0x11, 0x22, 0x33 in the low byte; rest zero).
  constexpr uint32_t kFieldSize = 8;
  std::vector<uint8_t> modulus = {0xa1, 0xa2, 0xa3, 0xa4,
                                  0xa5, 0xa6, 0xa7, 0xa8};
  std::vector<std::vector<uint8_t>> witnesses = {
      {0x11, 0, 0, 0, 0, 0, 0, 0},
      {0x22, 0, 0, 0, 0, 0, 0, 0},
      {0x33, 0, 0, 0, 0, 0, 0, 0},
  };

  std::vector<uint8_t> blob = {'w', 't', 'n', 's'};
  AppendLE<uint32_t>(blob, /*version=*/2);
  AppendLE<uint32_t>(blob, /*num_sections=*/2);
  // Section 1: header.
  AppendLE<uint32_t>(blob, /*type=*/1);
  AppendLE<uint64_t>(blob, /*size=*/4 + kFieldSize + 4);
  AppendLE<uint32_t>(blob, kFieldSize);
  blob.insert(blob.end(), modulus.begin(), modulus.end());
  AppendLE<uint32_t>(blob, /*num_witnesses=*/3);
  // Section 2: data.
  AppendLE<uint32_t>(blob, /*type=*/2);
  AppendLE<uint64_t>(blob, /*size=*/witnesses.size() * kFieldSize);
  for (const auto &w : witnesses)
    blob.insert(blob.end(), w.begin(), w.end());

  std::string path = WriteTempWtns(blob, "synthetic");
  TF_ASSERT_OK_AND_ASSIGN(WitnessFile wtns, ParseWtns(path));
  EXPECT_EQ(wtns.field_size_bytes, kFieldSize);
  EXPECT_EQ(wtns.num_witnesses, 3u);
  EXPECT_EQ(wtns.modulus, modulus);
  ASSERT_EQ(wtns.data.size(), witnesses.size() * kFieldSize);
  for (size_t i = 0; i < witnesses.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "witness[" << i << "]");
    auto span = wtns.Witness(i);
    EXPECT_EQ(span.size(), kFieldSize);
    EXPECT_EQ(0, std::memcmp(span.data(), witnesses[i].data(), kFieldSize));
  }
}

TEST(WtnsTest, ParsesCircomMultiplier3Fixture) {
  // Generated upstream by `circom -c` on a 3-input multiplier with input
  // {"in":["3","4","5"]}. Expected witnesses: [1, 60, 3, 4, 5, 12]
  // (1 = constant, 60 = output, 3/4/5 = inputs, 12 = 3*4 intermediate).
  // Field is bn254 Fr (32 bytes); each value's 8 low LE bytes encode the
  // small integer with the upper 24 bytes zero. See README.md for fixture
  // attribution.
  TF_ASSERT_OK_AND_ASSIGN(WitnessFile wtns,
                          ParseWtns("circom/wtns/multiplier_3.wtns"));
  EXPECT_EQ(wtns.field_size_bytes, 32u);
  EXPECT_EQ(wtns.num_witnesses, 6u);
  ASSERT_EQ(wtns.modulus.size(), 32u);

  const std::vector<uint64_t> expected_lowbits = {1, 60, 3, 4, 5, 12};
  ASSERT_EQ(wtns.data.size(), expected_lowbits.size() * 32u);
  for (size_t i = 0; i < expected_lowbits.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "witness[" << i << "]");
    auto span = wtns.Witness(i);
    uint64_t low;
    std::memcpy(&low, span.data(), sizeof(low));
    EXPECT_EQ(low, expected_lowbits[i]);
    for (size_t j = 8; j < 32; ++j) {
      EXPECT_EQ(span[j], 0u) << "high byte " << j << " should be zero";
    }
  }
}

TEST(WtnsTest, RejectsBadMagic) {
  std::vector<uint8_t> blob = {'X', 'X', 'X', 'X'};
  AppendLE<uint32_t>(blob, /*version=*/2);
  AppendLE<uint32_t>(blob, /*num_sections=*/0);
  std::string path = WriteTempWtns(blob, "bad_magic");
  EXPECT_FALSE(ParseWtns(path).ok());
}

TEST(WtnsTest, RejectsUnsupportedVersion) {
  std::vector<uint8_t> blob = {'w', 't', 'n', 's'};
  AppendLE<uint32_t>(blob, /*version=*/1);
  AppendLE<uint32_t>(blob, /*num_sections=*/0);
  std::string path = WriteTempWtns(blob, "bad_version");
  EXPECT_FALSE(ParseWtns(path).ok());
}

TEST(WtnsTest, RejectsTruncatedSection) {
  // Section claims 1024 bytes of payload but file ends right after the
  // section header.
  std::vector<uint8_t> blob = {'w', 't', 'n', 's'};
  AppendLE<uint32_t>(blob, /*version=*/2);
  AppendLE<uint32_t>(blob, /*num_sections=*/1);
  AppendLE<uint32_t>(blob, /*type=*/1);
  AppendLE<uint64_t>(blob, /*size=*/1024);
  std::string path = WriteTempWtns(blob, "truncated");
  EXPECT_FALSE(ParseWtns(path).ok());
}

TEST(WtnsTest, RejectsDataSizeMismatch) {
  // Header declares num_witnesses=2 with field_size=8 (so 16 bytes of data),
  // but the data section carries 24 bytes.
  std::vector<uint8_t> blob = {'w', 't', 'n', 's'};
  AppendLE<uint32_t>(blob, /*version=*/2);
  AppendLE<uint32_t>(blob, /*num_sections=*/2);
  AppendLE<uint32_t>(blob, /*type=*/1);
  AppendLE<uint64_t>(blob, /*size=*/4 + 8 + 4);
  AppendLE<uint32_t>(blob, /*field_size=*/8);
  for (int i = 0; i < 8; ++i)
    blob.push_back(static_cast<uint8_t>(i));
  AppendLE<uint32_t>(blob, /*num_witnesses=*/2);
  AppendLE<uint32_t>(blob, /*type=*/2);
  AppendLE<uint64_t>(blob, /*size=*/24);
  blob.insert(blob.end(), 24u, static_cast<uint8_t>(0));
  std::string path = WriteTempWtns(blob, "size_mismatch");
  auto status = ParseWtns(path);
  EXPECT_FALSE(status.ok());
}

} // namespace
} // namespace llzk_to_shlo::circom
