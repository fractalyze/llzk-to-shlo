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

# LLZK v2.0.0, released 2026-04-10. Bumped from adbea14b to pick up
# PRs #378 (poly.template) and #397 (FeltConstAttr holds FeltType).
LLZK_COMMIT = "9ea1976b10d045e55d440067de244c547cecb397"
LLZK_SHA256 = "bd34cf50358f8efae44a203f11c7c60b8142fbdac620154800d9f289e64a23d3"

def repo():
    http_archive(
        name = "llzk",
        sha256 = LLZK_SHA256,
        strip_prefix = "llzk-lib-{commit}".format(commit = LLZK_COMMIT),
        urls = ["https://github.com/project-llzk/llzk-lib/archive/{commit}.tar.gz".format(commit = LLZK_COMMIT)],
        build_file = "@llzk_to_shlo//third_party/llzk:llzk.BUILD",
        # Use GNU patch so hunks with offsets from upstream changes still apply.
        patch_tool = "patch",
        patch_args = ["-p1"],
        patches = [
            "@llzk_to_shlo//third_party/llzk:llvm20_call_interface.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_compat.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_symbol_lookup.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_symbol_cache.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_symbol_lookup_code.patch",
            "@llzk_to_shlo//third_party/llzk:llvm20_builders_include.patch",
            # LLZK v2 introduced Shared/TypeConversionPatterns.h, which uses
            # the pre-LLVM-20 split `match`/`rewrite` override pair.
            "@llzk_to_shlo//third_party/llzk:llvm20_type_conversion_patterns.patch",
            # Same split override pattern lingers in EmptyTemplateRemovalPass
            # (llzk-drop-empty-templates, required for LLZK v2 input).
            "@llzk_to_shlo//third_party/llzk:llvm20_empty_template_match_rewrite.patch",
        ],
    )

    # For local development, uncomment this:
    # native.new_local_repository(
    #     name = "llzk",
    #     path = "../llzk-lib",
    #     build_file = "@llzk_to_shlo//third_party/llzk:llzk.BUILD",
    # )
