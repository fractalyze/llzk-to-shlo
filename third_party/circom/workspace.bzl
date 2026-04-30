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

def _content_hash(ctx, path):
    """Returns the sha256 of `path`'s file content, or "unknown" on failure.

    Tries GNU coreutils `sha256sum` first (Linux self-hosted CI), then BSD/macOS
    `shasum -a 256` (local dev on macOS). Both emit `<hash>  <path>`; we take
    the first whitespace-delimited token. The wrapper script embeds this hash
    so a binary content swap perturbs the wrapper text by one hex string and
    Bazel invalidates downstream actions; without it the resolved path string
    is the only cache key and an in-place binary update is silently cached.
    """
    for cmd in [["sha256sum", path], ["shasum", "-a", "256", path]]:
        result = ctx.execute(cmd)
        if result.return_code == 0:
            return result.stdout.split(" ")[0]
    return "unknown"

def _emit_circom_wrapper(ctx, circom_path):
    """Writes the @circom repo's BUILD.bazel and content-hashed wrapper script."""
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "circom",
    srcs = ["circom_wrapper.sh"],
)
""")
    ctx.file("circom_wrapper.sh", """#!/bin/bash
# binary-sha256: {content_hash}
exec "{path}" "$@"
""".format(path = circom_path, content_hash = _content_hash(ctx, circom_path)), executable = True)

def _circom_repository_impl(ctx):
    """Sets up circom from a pre-installed binary."""

    # Check CIRCOM_PATH environment variable first, then fall back to PATH
    circom_path = ctx.os.environ.get("CIRCOM_PATH", "")

    if circom_path:
        _emit_circom_wrapper(ctx, circom_path)
        return

    # Try to find circom in PATH
    result = ctx.execute(["which", "circom"])
    if result.return_code == 0:
        _emit_circom_wrapper(ctx, result.stdout.strip())
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
