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

load("@rules_cc//cc:defs.bzl", "cc_library")

"""BUILD file for llzk-lib (LLZK MLIR dialects)."""

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(default_visibility = ["//visibility:public"])

LLZK_COPTS = [
    "-std=c++20",
]

# =============================================================================
# Config Header Generation
# =============================================================================

genrule(
    name = "llzk_config_h",
    srcs = ["include/llzk/Config/Config.h.in"],
    outs = ["include/llzk/Config/Config.h"],
    cmd = """
        sed -e 's/$${CMAKE_PROJECT_VERSION}/1.1.5/g' \
            -e 's/$${CMAKE_PROJECT_VERSION_MAJOR}/1/g' \
            -e 's/$${CMAKE_PROJECT_VERSION_MINOR}/1/g' \
            -e 's/$${CMAKE_PROJECT_VERSION_PATCH}/5/g' \
            -e 's/$${CMAKE_PROJECT_HOMEPAGE_URL}/https:\\/\\/github.com\\/project-llzk\\/llzk-lib/g' \
            -e 's/#cmakedefine01 LLZK_WITH_PCL_BOOL/#define LLZK_WITH_PCL_BOOL 0/g' \
            $< > $@
    """,
)

# =============================================================================
# TableGen file libraries
# =============================================================================

td_library(
    name = "LLZKIncludeTdFiles",
    srcs = glob(["include/llzk/**/*.td"]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:MemorySlotInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

# =============================================================================
# LLZK Header-Only Library (all headers for cross-dialect dependencies)
# Note: This needs to be defined after all TableGen rules so it can depend on them
# =============================================================================

# This will be defined later after all TableGen rules

# =============================================================================
# LLZK Core Dialect
# =============================================================================

gentbl_cc_library(
    name = "LLZKDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=llzk",
            ],
            "include/llzk/Dialect/LLZK/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=llzk",
            ],
            "include/llzk/Dialect/LLZK/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/LLZK/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "LLZKAttrsIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/llzk/Dialect/LLZK/IR/Attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/llzk/Dialect/LLZK/IR/Attrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/LLZK/IR/Attrs.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "LLZKOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/LLZK/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/LLZK/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/LLZK/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "LLZKDialect",
    srcs = glob(["lib/Dialect/LLZK/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKAttrsIncGen",
        ":LLZKDialectIncGen",
        ":LLZKHeaders",
        ":LLZKOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# =============================================================================
# Felt Dialect
# =============================================================================

gentbl_cc_library(
    name = "FeltDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=felt",
            ],
            "include/llzk/Dialect/Felt/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=felt",
            ],
            "include/llzk/Dialect/Felt/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Felt/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "FeltOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Felt/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Felt/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Felt/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "FeltTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=felt",
            ],
            "include/llzk/Dialect/Felt/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=felt",
            ],
            "include/llzk/Dialect/Felt/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Felt/IR/Types.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "FeltAttrsIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/llzk/Dialect/Felt/IR/Attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/llzk/Dialect/Felt/IR/Attrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Felt/IR/Attrs.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "FeltOpInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/llzk/Dialect/Felt/IR/OpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/llzk/Dialect/Felt/IR/OpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Felt/IR/OpInterfaces.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "FeltDialect",
    srcs = glob(["lib/Dialect/Felt/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Array Dialect
# =============================================================================

gentbl_cc_library(
    name = "ArrayDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=array",
            ],
            "include/llzk/Dialect/Array/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=array",
            ],
            "include/llzk/Dialect/Array/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Array/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "ArrayOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Array/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Array/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Array/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "ArrayTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=array",
            ],
            "include/llzk/Dialect/Array/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=array",
            ],
            "include/llzk/Dialect/Array/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Array/IR/Types.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "ArrayOpInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/llzk/Dialect/Array/IR/OpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/llzk/Dialect/Array/IR/OpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Array/IR/OpInterfaces.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "ArrayDialect",
    srcs = glob(["lib/Dialect/Array/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Struct Dialect
# =============================================================================

gentbl_cc_library(
    name = "StructDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=struct",
            ],
            "include/llzk/Dialect/Struct/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=struct",
            ],
            "include/llzk/Dialect/Struct/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Struct/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "StructOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Struct/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Struct/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Struct/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "StructTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=struct",
            ],
            "include/llzk/Dialect/Struct/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=struct",
            ],
            "include/llzk/Dialect/Struct/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Struct/IR/Types.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "StructOpInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/llzk/Dialect/Struct/IR/OpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/llzk/Dialect/Struct/IR/OpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Struct/IR/OpInterfaces.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "StructDialect",
    srcs = glob(["lib/Dialect/Struct/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Function Dialect
# =============================================================================

gentbl_cc_library(
    name = "FunctionDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=function",
            ],
            "include/llzk/Dialect/Function/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=function",
            ],
            "include/llzk/Dialect/Function/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Function/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "FunctionOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Function/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Function/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Function/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "FunctionAttrsIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/llzk/Dialect/Function/IR/Attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/llzk/Dialect/Function/IR/Attrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Function/IR/Attrs.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "FunctionDialect",
    srcs = glob(["lib/Dialect/Function/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Constrain Dialect
# =============================================================================

gentbl_cc_library(
    name = "ConstrainDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=constrain",
            ],
            "include/llzk/Dialect/Constrain/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=constrain",
            ],
            "include/llzk/Dialect/Constrain/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Constrain/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "ConstrainOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Constrain/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Constrain/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Constrain/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "ConstrainOpInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/llzk/Dialect/Constrain/IR/OpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/llzk/Dialect/Constrain/IR/OpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Constrain/IR/OpInterfaces.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "ConstrainDialect",
    srcs = glob(["lib/Dialect/Constrain/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Global Dialect
# =============================================================================

gentbl_cc_library(
    name = "GlobalDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=global",
            ],
            "include/llzk/Dialect/Global/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=global",
            ],
            "include/llzk/Dialect/Global/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Global/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "GlobalOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Global/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Global/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Global/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "GlobalOpInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/llzk/Dialect/Global/IR/OpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/llzk/Dialect/Global/IR/OpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Global/IR/OpInterfaces.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "GlobalDialect",
    srcs = glob(["lib/Dialect/Global/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Bool Dialect
# =============================================================================

gentbl_cc_library(
    name = "BoolDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=bool",
            ],
            "include/llzk/Dialect/Bool/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=bool",
            ],
            "include/llzk/Dialect/Bool/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Bool/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "BoolOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Bool/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Bool/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Bool/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "BoolAttrsIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/llzk/Dialect/Bool/IR/Attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/llzk/Dialect/Bool/IR/Attrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Bool/IR/Attrs.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "BoolEnumsIncGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/llzk/Dialect/Bool/IR/Enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/llzk/Dialect/Bool/IR/Enums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Bool/IR/Enums.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "BoolDialect",
    srcs = glob(["lib/Dialect/Bool/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Cast Dialect
# =============================================================================

gentbl_cc_library(
    name = "CastDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=cast",
            ],
            "include/llzk/Dialect/Cast/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=cast",
            ],
            "include/llzk/Dialect/Cast/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Cast/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "CastOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Cast/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Cast/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Cast/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "CastDialect",
    srcs = glob(["lib/Dialect/Cast/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Include Dialect
# =============================================================================

gentbl_cc_library(
    name = "IncludeDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=include",
            ],
            "include/llzk/Dialect/Include/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=include",
            ],
            "include/llzk/Dialect/Include/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Include/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "IncludeOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Include/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Include/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Include/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "IncludeDialect",
    srcs = glob(["lib/Dialect/Include/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# POD Dialect
# =============================================================================

gentbl_cc_library(
    name = "PODDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=pod",
            ],
            "include/llzk/Dialect/POD/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=pod",
            ],
            "include/llzk/Dialect/POD/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/POD/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "PODOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/POD/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/POD/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/POD/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "PODTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=pod",
            ],
            "include/llzk/Dialect/POD/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=pod",
            ],
            "include/llzk/Dialect/POD/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/POD/IR/Types.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "PODAttrsIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/llzk/Dialect/POD/IR/Attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/llzk/Dialect/POD/IR/Attrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/POD/IR/Attrs.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "PODDialect",
    srcs = glob(["lib/Dialect/POD/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Polymorphic Dialect
# =============================================================================

gentbl_cc_library(
    name = "PolymorphicDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=poly",
            ],
            "include/llzk/Dialect/Polymorphic/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=poly",
            ],
            "include/llzk/Dialect/Polymorphic/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Polymorphic/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "PolymorphicOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/Polymorphic/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/Polymorphic/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Polymorphic/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "PolymorphicTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=poly",
            ],
            "include/llzk/Dialect/Polymorphic/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=poly",
            ],
            "include/llzk/Dialect/Polymorphic/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/Polymorphic/IR/Types.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "PolymorphicDialect",
    srcs = glob(["lib/Dialect/Polymorphic/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# String Dialect
# =============================================================================

gentbl_cc_library(
    name = "StringDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=string",
            ],
            "include/llzk/Dialect/String/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=string",
            ],
            "include/llzk/Dialect/String/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/String/IR/Dialect.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "StringOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/llzk/Dialect/String/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/llzk/Dialect/String/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/String/IR/Ops.td",
    deps = [":LLZKIncludeTdFiles"],
)

gentbl_cc_library(
    name = "StringTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=string",
            ],
            "include/llzk/Dialect/String/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=string",
            ],
            "include/llzk/Dialect/String/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Dialect/String/IR/Types.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "StringDialect",
    srcs = glob(["lib/Dialect/String/IR/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# LLZK Headers Library (all headers + TableGen outputs)
# =============================================================================

cc_library(
    name = "LLZKHeaders",
    hdrs = glob([
        "include/llzk/**/*.h",
    ]) + [":llzk_config_h"],
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        # TableGen generated files for all dialects
        ":ArrayDialectIncGen",
        ":ArrayOpInterfacesIncGen",
        ":ArrayOpsIncGen",
        ":ArrayTypesIncGen",
        ":BoolAttrsIncGen",
        ":BoolDialectIncGen",
        ":BoolEnumsIncGen",
        ":BoolOpsIncGen",
        ":CastDialectIncGen",
        ":CastOpsIncGen",
        ":ConstrainDialectIncGen",
        ":ConstrainOpInterfacesIncGen",
        ":ConstrainOpsIncGen",
        ":FeltAttrsIncGen",
        ":FeltDialectIncGen",
        ":FeltOpInterfacesIncGen",
        ":FeltOpsIncGen",
        ":FeltTypesIncGen",
        ":FunctionAttrsIncGen",
        ":FunctionDialectIncGen",
        ":FunctionOpsIncGen",
        ":GlobalDialectIncGen",
        ":GlobalOpInterfacesIncGen",
        ":GlobalOpsIncGen",
        ":IncludeDialectIncGen",
        ":IncludeOpsIncGen",
        ":LLZKAttrsIncGen",
        ":LLZKDialectIncGen",
        ":LLZKOpsIncGen",
        ":PODAttrsIncGen",
        ":PODDialectIncGen",
        ":PODOpsIncGen",
        ":PODTypesIncGen",
        ":PolymorphicDialectIncGen",
        ":PolymorphicOpsIncGen",
        ":PolymorphicTypesIncGen",
        ":StringDialectIncGen",
        ":StringOpsIncGen",
        ":StringTypesIncGen",
        ":StructDialectIncGen",
        ":StructOpInterfacesIncGen",
        ":StructOpsIncGen",
        ":StructTypesIncGen",
        # MLIR dependencies
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemorySlotInterfaces",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

# =============================================================================
# LLZK Utility Library
# =============================================================================

cc_library(
    name = "LLZKUtil",
    srcs = glob(["lib/Util/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
    ],
)

# =============================================================================
# Combined LLZK Dialects (Monolithic library with all sources)
# =============================================================================

cc_library(
    name = "LLZKDialects",
    srcs = glob([
        "lib/Dialect/*/IR/*.cpp",
        "lib/Dialect/*/Util/*.cpp",
        "lib/Dialect/Shared/*.cpp",
    ]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKHeaders",
        ":LLZKUtil",
        "@llvm-project//mlir:Parser",
    ],
    alwayslink = True,
)

# =============================================================================
# LLZK Transforms (inline-structs, etc.)
# =============================================================================

gentbl_cc_library(
    name = "LLZKTransformsIncGen",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "include/llzk/Transforms/LLZKTransformationPasses.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/llzk/Transforms/LLZKTransformationPasses.td",
    deps = [":LLZKIncludeTdFiles"],
)

cc_library(
    name = "LLZKTransforms",
    srcs = glob(["lib/Transforms/*.cpp"]),
    copts = LLZK_COPTS,
    includes = ["include"],
    deps = [
        ":LLZKDialects",
        ":LLZKHeaders",
        ":LLZKTransformsIncGen",
        ":LLZKUtil",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = True,
)
