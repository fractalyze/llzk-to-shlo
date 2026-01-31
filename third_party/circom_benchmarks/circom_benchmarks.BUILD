# Copyright 2026 The llzk-to-shlo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

package(default_visibility = ["//visibility:public"])

# Export all circom files for use in E2E tests
exports_files(glob(["**/*.circom"]))

# Filegroups for specific benchmark categories
filegroup(
    name = "all_circom",
    srcs = glob(["**/*.circom"]),
)

filegroup(
    name = "applications",
    srcs = glob(["applications/**/*.circom"]),
)

filegroup(
    name = "libs",
    srcs = glob(["libs/**/*.circom"]),
)

# Individual application benchmarks
filegroup(
    name = "num2bits_srcs",
    srcs = glob(["applications/Num2Bits/src/*.circom"]),
)

filegroup(
    name = "binsum_srcs",
    srcs = glob(["applications/BinSum/src/*.circom"]),
)

filegroup(
    name = "decoder_srcs",
    srcs = glob(["applications/Decoder/src/*.circom"]),
)

filegroup(
    name = "fulladder_srcs",
    srcs = glob(["applications/fulladder/src/*.circom"]),
)

filegroup(
    name = "multiplier_srcs",
    srcs = glob(["applications/WindowMulFix/src/*.circom"]),
)
