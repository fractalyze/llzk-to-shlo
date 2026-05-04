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

"""Bazel rules for Circom to StableHLO E2E conversion."""

def _circom_to_llzk_impl(ctx):
    """Compiles Circom to LLZK IR using the circom compiler."""
    output = ctx.actions.declare_file(ctx.label.name + ".llzk")

    # Find main.circom or use first .circom file
    main_src = None
    for src in ctx.files.srcs:
        if src.basename == "main.circom":
            main_src = src
            break
    if main_src == None:
        main_src = ctx.files.srcs[0]

    # Build include path arguments
    include_args = []
    seen_dirs = {}
    for inc in ctx.files.includes:
        if inc.dirname not in seen_dirs:
            include_args.extend(["-l", inc.dirname])
            seen_dirs[inc.dirname] = True

    # Also add source file directory as include path
    if main_src.dirname not in seen_dirs:
        include_args.extend(["-l", main_src.dirname])

    # Create a wrapper script to handle circom's directory output structure
    # circom outputs to: output_dir/<circuit>_llzk/<circuit>.llzk
    wrapper = ctx.actions.declare_file(ctx.label.name + "_wrapper.sh")
    ctx.actions.write(
        output = wrapper,
        content = """#!/bin/bash
set -e
CIRCOM="$1"
shift
OUTPUT="$1"
shift
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
"$CIRCOM" "$@" -o "$TMPDIR"
LLZK_FILE=$(find "$TMPDIR" -name "*.llzk" | head -1)
if [ -z "$LLZK_FILE" ]; then
    echo "ERROR: No .llzk file generated"
    exit 1
fi
cp "$LLZK_FILE" "$OUTPUT"
""",
        is_executable = True,
    )

    # `--llzk_plaintext` keeps the textual IR format; without it, LLZK v2 circom
    # (PR project-llzk/circom#360) emits bytecode by default. `llzk-to-shlo-opt`
    # reads either via MlirOptMain, but textual output is easier to inspect.
    args = [
        ctx.executable._circom.path,
        output.path,
        main_src.path,
        "--llzk",
        "concrete",
        "--llzk_plaintext",
    ] + include_args

    ctx.actions.run(
        outputs = [output],
        inputs = ctx.files.srcs + ctx.files.includes,
        tools = [ctx.executable._circom, wrapper],
        executable = wrapper,
        arguments = args,
        mnemonic = "CircomToLLZK",
        progress_message = "Compiling %s to LLZK" % main_src.short_path,
    )

    return [DefaultInfo(files = depset([output]))]

circom_to_llzk = rule(
    implementation = _circom_to_llzk_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".circom"],
            mandatory = True,
            doc = "Circom source files",
        ),
        "includes": attr.label_list(
            allow_files = True,
            doc = "Include directories for circom imports",
        ),
        "_circom": attr.label(
            default = "@circom//:circom",
            executable = True,
            cfg = "exec",
            doc = "The circom compiler",
        ),
    },
    doc = "Compiles Circom to LLZK IR.",
)

def _llzk_to_stablehlo_impl(ctx):
    """Converts LLZK IR to StableHLO using llzk-to-shlo-opt."""
    output = ctx.actions.declare_file(ctx.label.name + ".stablehlo.mlir")

    args = ctx.actions.args()
    args.add("--allow-unregistered-dialect")
    args.add("--simplify-sub-components")
    args.add("--witness-layout-anchor")
    args.add("--llzk-to-stablehlo=prime=" + ctx.attr.prime)
    args.add("--verify-witness-layout")
    args.add(ctx.files.srcs[0])
    args.add("-o", output)

    ctx.actions.run(
        outputs = [output],
        inputs = ctx.files.srcs,
        executable = ctx.executable._opt,
        arguments = [args],
        mnemonic = "LLZKToStableHLO",
        progress_message = "Converting %s to StableHLO" % ctx.files.srcs[0].short_path,
    )

    return [DefaultInfo(files = depset([output]))]

llzk_to_stablehlo = rule(
    implementation = _llzk_to_stablehlo_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".llzk", ".mlir"],
            mandatory = True,
            doc = "LLZK IR files",
        ),
        "prime": attr.string(
            default = "bn254",
            doc = "Prime modulus. Aliases: bn254. Or explicit: <value>:<type>",
        ),
        "_opt": attr.label(
            default = "//tools:llzk-to-shlo-opt",
            executable = True,
            cfg = "exec",
            doc = "The llzk-to-shlo-opt tool",
        ),
    },
    doc = "Converts LLZK IR to StableHLO.",
)

def circom_to_stablehlo(name, srcs, includes = [], prime = "bn254", **kwargs):
    """E2E macro: Circom -> LLZK -> StableHLO.

    This macro creates two targets:
    - {name}_llzk: Circom to LLZK conversion
    - {name}: LLZK to StableHLO conversion (final output)

    Args:
        name: Name of the target.
        srcs: Circom source files.
        includes: Include directories for circom imports.
        prime: Prime modulus with storage type.
        **kwargs: Additional arguments passed to the rules.
    """
    llzk_name = name + "_llzk"

    # Step 1: Circom -> LLZK (uses @circom//:circom built from source)
    circom_to_llzk(
        name = llzk_name,
        srcs = srcs,
        includes = includes,
        **kwargs
    )

    # Step 2: LLZK -> StableHLO
    llzk_to_stablehlo(
        name = name,
        srcs = [":" + llzk_name],
        prime = prime,
        **kwargs
    )

def golden_llzk_to_stablehlo(name, golden_llzk, prime = "bn254", **kwargs):
    """E2E macro: checked-in golden LLZK -> StableHLO.

    Use INSTEAD of circom_to_stablehlo for chips whose .json.gate is
    layout-sensitive (>= 2 sub-component-derived members in the main
    struct). project-llzk/circom emits non-deterministic struct.member
    ordering for such chips per process; the .json.gate maps fixed
    output offsets to .wtns wire indices, so live-circom output drifts
    the gate position-by-position across CI runs. Pinning a checked-in
    golden LLZK file makes the layout stable by construction.

    See CLAUDE.md "Multi-sub-component composite chips" for the full
    convention and the determinism tripwire that signals when this
    indirection can be retired.

    This macro creates two targets:
    - {name}_llzk: genrule that copies the golden into bazel-bin
    - {name}: LLZK to StableHLO conversion (final output)

    Args:
        name: Name of the target.
        golden_llzk: Label of the checked-in `.llzk.golden` file.
        prime: Prime modulus.
        **kwargs: Additional arguments passed to llzk_to_stablehlo.
    """
    llzk_name = name + "_llzk"

    # Step 1: copy the checked-in golden into bazel-bin/<...>/<name>_llzk.llzk
    native.genrule(
        name = llzk_name,
        srcs = [golden_llzk],
        outs = [llzk_name + ".llzk"],
        cmd = "cp $(SRCS) $@",
    )

    # Step 2: LLZK -> StableHLO (unchanged from circom_to_stablehlo)
    llzk_to_stablehlo(
        name = name,
        srcs = [":" + llzk_name],
        prime = prime,
        **kwargs
    )
