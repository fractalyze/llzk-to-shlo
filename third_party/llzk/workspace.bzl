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

"""Provides the repo macro to import llzk-lib."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

LLZK_COMMIT = "adbea14b423fd4bf63ca983e13c18214caa65830"
LLZK_SHA256 = "20f37ac7a900ff81f25da1a850697a24ce895b7fe6c191326f8bf0c092181932"

def repo():
    http_archive(
        name = "llzk",
        sha256 = LLZK_SHA256,
        strip_prefix = "llzk-lib-{commit}".format(commit = LLZK_COMMIT),
        urls = ["https://github.com/project-llzk/llzk-lib/archive/{commit}.tar.gz".format(commit = LLZK_COMMIT)],
        build_file = "@llzk_to_shlo//third_party/llzk:llzk.BUILD",
        patch_args = ["-p1"],
        patches = [
            "@llzk_to_shlo//third_party/llzk:llvm20_call_interface.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_compat.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_symbol_lookup.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_symbol_cache.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_symbol_lookup_code.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_builders_include.patch",
        ],
    )

    # For local development, uncomment this:
    # native.new_local_repository(
    #     name = "llzk",
    #     path = "../llzk-lib",
    #     build_file = "@llzk_to_shlo//third_party/llzk:llzk.BUILD",
    # )
