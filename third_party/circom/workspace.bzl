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

"""Workspace setup for circom (project-llzk/circom).

Assumes circom is pre-installed locally. Set the CIRCOM_PATH environment
variable to specify the circom binary location, or ensure circom is in PATH.

LLZK v2.0.0 requires project-llzk/circom from the `llzk` branch at commit
8420d0b (2026-04-24) or newer, which carries:
  - PR #362 (LLZK v2.0.0 upgrade) — new `poly.template` IR shape
  - PR #360 (LLZK bytecode default, `--llzk_plaintext` for textual IR)

Building from source:
  git clone -b llzk https://github.com/project-llzk/circom.git
  cd circom
  MLIR_SYS_200_PREFIX=/usr/lib/llvm-20 \\
  TABLEGEN_200_PREFIX=/usr/lib/llvm-20 \\
  LLVM_SYS_200_PREFIX=/usr/lib/llvm-20 \\
  cargo build --release
  export CIRCOM_PATH="$PWD/target/release/circom"

System prerequisites (Debian/Ubuntu): llvm-20-dev, libmlir-20-dev.
"""

def _circom_repository_impl(ctx):
    """Sets up circom from a pre-installed binary."""

    # Check CIRCOM_PATH environment variable first, then fall back to PATH
    circom_path = ctx.os.environ.get("CIRCOM_PATH", "")

    if circom_path:
        # Use the path from environment variable
        ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "circom",
    srcs = ["circom_wrapper.sh"],
)
""")
        ctx.file("circom_wrapper.sh", """#!/bin/bash
exec "{}" "$@"
""".format(circom_path), executable = True)
        return

    # Try to find circom in PATH
    result = ctx.execute(["which", "circom"])
    if result.return_code == 0:
        circom_path = result.stdout.strip()
        ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "circom",
    srcs = ["circom_wrapper.sh"],
)
""")
        ctx.file("circom_wrapper.sh", """#!/bin/bash
exec "{}" "$@"
""".format(circom_path), executable = True)
        return

    # circom not found
    fail("""
circom not found. Please install circom and either:
1. Add it to PATH, or
2. Set CIRCOM_PATH environment variable to the circom binary path

Installation (LLZK v2 compatible — requires the `llzk` branch):
  git clone -b llzk https://github.com/project-llzk/circom.git
  cd circom
  MLIR_SYS_200_PREFIX=/usr/lib/llvm-20 \\
  TABLEGEN_200_PREFIX=/usr/lib/llvm-20 \\
  LLVM_SYS_200_PREFIX=/usr/lib/llvm-20 \\
  cargo build --release
  export CIRCOM_PATH="$PWD/target/release/circom"

System prerequisites (Debian/Ubuntu): llvm-20-dev, libmlir-20-dev.
""")

circom_repository = repository_rule(
    implementation = _circom_repository_impl,
    environ = ["CIRCOM_PATH"],
    doc = "Sets up circom from a pre-installed binary.",
)

def circom_deps():
    """Set up circom as a Bazel dependency."""
    circom_repository(name = "circom")
