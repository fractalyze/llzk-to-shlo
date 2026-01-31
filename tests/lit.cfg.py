"""Lit configuration for llzk-to-shlo tests."""

# Copyright 2026 The llzk-to-shlo Authors.
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

# -*- Python -*-
# pylint: disable=undefined-variable

import os

import lit.formats
from lit.llvm import llvm_config

# Populate Lit configuration with the minimal required metadata.
config.name = "LLZK_TO_SHLO_TESTS"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

# test_source_root is set by lit.site.cfg.py

# Disallow reusing variables across CHECK-LABEL matches.
config.environment["FILECHECK_OPTS"] = "-enable-var-scope"

# Make LLVM and llzk-to-shlo tools available in RUN directives
tools = [
    "FileCheck",
    "llzk-to-shlo-opt",
    "not",
]
tool_dirs = [
    config.llvm_tools_dir,
    config.llzk_to_shlo_tools_dir,
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
