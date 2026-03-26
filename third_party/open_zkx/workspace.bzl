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

"""Workspace setup for open-zkx stablehlo_runner.

Assumes stablehlo_runner_main is pre-built. Set STABLEHLO_RUNNER_PATH to the
binary location, or ensure it's in PATH.
"""

def _stablehlo_runner_repository_impl(ctx):
    """Sets up stablehlo_runner from a pre-built binary."""

    runner_path = ctx.os.environ.get("STABLEHLO_RUNNER_PATH", "")

    if runner_path:
        ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "stablehlo_runner",
    srcs = ["stablehlo_runner_wrapper.sh"],
)
""")
        ctx.file("stablehlo_runner_wrapper.sh", """#!/bin/bash
exec "{}" "$@"
""".format(runner_path), executable = True)
        return

    # Try to find in PATH
    result = ctx.execute(["which", "stablehlo_runner_main"])
    if result.return_code == 0:
        runner_path = result.stdout.strip()
        ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "stablehlo_runner",
    srcs = ["stablehlo_runner_wrapper.sh"],
)
""")
        ctx.file("stablehlo_runner_wrapper.sh", """#!/bin/bash
exec "{}" "$@"
""".format(runner_path), executable = True)
        return

    # Not found — create a stub that prints instructions
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "stablehlo_runner",
    srcs = ["stablehlo_runner_wrapper.sh"],
)
""")
    ctx.file("stablehlo_runner_wrapper.sh", """#!/bin/bash
echo "ERROR: stablehlo_runner not found." >&2
echo "Build it from open-zkx:" >&2
echo "  cd open-zkx && bazel build //zkx/tools/stablehlo_runner:stablehlo_runner_main" >&2
echo "Then set STABLEHLO_RUNNER_PATH to the binary path." >&2
exit 1
""", executable = True)

stablehlo_runner_repository = repository_rule(
    implementation = _stablehlo_runner_repository_impl,
    environ = ["STABLEHLO_RUNNER_PATH"],
    doc = "Sets up stablehlo_runner from a pre-built binary.",
)

def stablehlo_runner_deps():
    """Set up stablehlo_runner as a Bazel dependency."""
    stablehlo_runner_repository(name = "stablehlo_runner")
