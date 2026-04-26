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

#include "circom/wtns/wtns.h"

#include <cstring>
#include <string>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"

namespace llzk_to_shlo::circom {
namespace {

constexpr char kMagic[4] = {'w', 't', 'n', 's'};
constexpr uint32_t kSupportedVersion = 2;
constexpr uint32_t kHeaderSectionId = 1;
constexpr uint32_t kDataSectionId = 2;
constexpr size_t kPreambleBytes = 12;      // magic + version + num_sections
constexpr size_t kSectionHeaderBytes = 12; // type u32 + size u64
// Sanity ceiling on `field_size_bytes`. Real fields are 32 (bn254, bls12-381
// scalar) or 48 (bls12-381 base); 256 is a generous bound that catches
// corrupt headers without imposing on legitimate cryptographic fields.
constexpr uint32_t kMaxFieldSizeBytes = 256;
// OOM guard against a malformed (or hostile) `path` pointing at a huge file.
// Realistic .wtns sit well under 1 GiB even for million-signal circuits
// (32 MiB at 1M signals × 32 B); 4 GiB leaves headroom for outliers.
constexpr uint64_t kMaxFileSizeBytes = uint64_t{4} << 30;

// Phrased to avoid `offset + need` arithmetic on attacker-controlled `need`
// values that could otherwise wrap a 64-bit `size_t`.
absl::Status BoundsCheck(size_t offset, size_t need, size_t total,
                         std::string_view what) {
  if (offset > total || need > total - offset) {
    return absl::DataLossError(absl::Substitute(
        "wtns: $0: need $1 bytes at offset $2, only $3 available", what, need,
        offset, total));
  }
  return absl::OkStatus();
}

} // namespace

absl::Span<const uint8_t> WitnessFile::Witness(size_t i) const {
  return absl::MakeConstSpan(data.data() + i * field_size_bytes,
                             field_size_bytes);
}

absl::StatusOr<WitnessFile> ParseWtns(std::string_view path) {
  std::string path_owned(path);
  uint64_t file_size = 0;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSize(path_owned, &file_size));
  if (file_size > kMaxFileSizeBytes) {
    return absl::ResourceExhaustedError(absl::Substitute(
        "wtns: file size $0 exceeds limit $1", file_size, kMaxFileSizeBytes));
  }
  std::string contents;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path_owned, &contents));
  const uint8_t *base = reinterpret_cast<const uint8_t *>(contents.data());
  const size_t total = contents.size();

  TF_RETURN_IF_ERROR(BoundsCheck(0, kPreambleBytes, total, "file preamble"));
  if (std::memcmp(base, kMagic, sizeof(kMagic)) != 0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "wtns: bad magic; expected 'wtns', got '$0$1$2$3'",
        static_cast<char>(base[0]), static_cast<char>(base[1]),
        static_cast<char>(base[2]), static_cast<char>(base[3])));
  }
  uint32_t version = absl::little_endian::Load32(base + 4);
  if (version != kSupportedVersion) {
    return absl::InvalidArgumentError(absl::Substitute(
        "wtns: unsupported version $0 (only v2 is supported)", version));
  }
  uint32_t num_sections = absl::little_endian::Load32(base + 8);

  // Discover all section ranges first; resolve type-1/type-2 afterward so we
  // don't require the header to precede data on disk.
  struct Range {
    uint32_t type;
    size_t from;
    size_t size;
  };
  std::vector<Range> ranges;
  ranges.reserve(num_sections);
  size_t offset = kPreambleBytes;
  for (uint32_t i = 0; i < num_sections; ++i) {
    TF_RETURN_IF_ERROR(
        BoundsCheck(offset, kSectionHeaderBytes, total, "section header"));
    uint32_t type = absl::little_endian::Load32(base + offset);
    uint64_t size = absl::little_endian::Load64(base + offset + 4);
    offset += kSectionHeaderBytes;
    TF_RETURN_IF_ERROR(BoundsCheck(offset, size, total, "section payload"));
    ranges.push_back({type, offset, static_cast<size_t>(size)});
    offset += size;
  }

  WitnessFile out;
  bool seen_header = false;
  bool seen_data = false;
  for (const Range &r : ranges) {
    if (r.type == kHeaderSectionId) {
      if (seen_header) {
        return absl::InvalidArgumentError("wtns: duplicate header section");
      }
      seen_header = true;
      if (r.size < 8) {
        return absl::DataLossError("wtns: header section too small");
      }
      const uint8_t *p = base + r.from;
      out.field_size_bytes = absl::little_endian::Load32(p);
      if (out.field_size_bytes == 0 ||
          out.field_size_bytes > kMaxFieldSizeBytes) {
        return absl::InvalidArgumentError(absl::Substitute(
            "wtns: implausible field_size_bytes $0", out.field_size_bytes));
      }
      const size_t expected = 4u + out.field_size_bytes + 4u;
      if (r.size != expected) {
        return absl::DataLossError(
            absl::Substitute("wtns: header section size $0 != expected $1 "
                             "(4 + field_size + 4)",
                             r.size, expected));
      }
      p += 4;
      out.modulus.assign(p, p + out.field_size_bytes);
      p += out.field_size_bytes;
      out.num_witnesses = absl::little_endian::Load32(p);
    } else if (r.type == kDataSectionId) {
      if (seen_data) {
        return absl::InvalidArgumentError("wtns: duplicate data section");
      }
      seen_data = true;
      out.data.assign(base + r.from, base + r.from + r.size);
    }
    // Unknown section types are ignored — iden3 readers are forward-compatible.
  }

  if (!seen_header) {
    return absl::InvalidArgumentError("wtns: missing header section");
  }
  if (!seen_data) {
    return absl::InvalidArgumentError("wtns: missing data section");
  }
  const size_t expected_data =
      static_cast<size_t>(out.num_witnesses) * out.field_size_bytes;
  if (out.data.size() != expected_data) {
    return absl::DataLossError(
        absl::Substitute("wtns: data section size $0 != num_witnesses($1) * "
                         "field_size_bytes($2) = $3",
                         out.data.size(), out.num_witnesses,
                         out.field_size_bytes, expected_data));
  }

  return out;
}

} // namespace llzk_to_shlo::circom
