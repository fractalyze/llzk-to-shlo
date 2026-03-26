#!/bin/bash
# Copyright 2026 The llzk-to-shlo Authors.
#
# End-to-end demo: Circom → LLZK → StableHLO → GPU witness generation
#
# Usage:
#   ./examples/run_e2e.sh [--input "3,5"]
#
# Prerequisites:
#   - Build llzk-to-shlo-opt:  bazel build //tools:llzk-to-shlo-opt
#   - Build stablehlo_runner:  cd open-zkx && bazel build //zkx/tools/stablehlo_runner:stablehlo_runner_main
#   - Set STABLEHLO_RUNNER_PATH (or add to PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Tools
OPT="${PROJECT_DIR}/bazel-bin/tools/llzk-to-shlo-opt"
RUNNER="${STABLEHLO_RUNNER_PATH:-stablehlo_runner_main}"

# Input
INPUT="${1:---input=3,5}"
LLZK_FILE="${PROJECT_DIR}/examples/multiplier2/main.llzk"
STABLEHLO_FILE="/tmp/multiplier2.stablehlo.mlir"

echo "=== Step 1: LLZK IR (from Circom) ==="
echo "Circuit: Multiplier2 (out = in1 * in2)"
echo "Input LLZK: ${LLZK_FILE}"
echo

echo "=== Step 2: LLZK → StableHLO ==="
"${OPT}" --llzk-to-stablehlo "${LLZK_FILE}" -o "${STABLEHLO_FILE}" 2>&1
echo "Output StableHLO: ${STABLEHLO_FILE}"
echo
cat "${STABLEHLO_FILE}"
echo

echo "=== Step 3: StableHLO → GPU Execution ==="
if command -v "${RUNNER}" &>/dev/null || [ -f "${RUNNER}" ]; then
    echo "Running on GPU with input: ${INPUT}"
    "${RUNNER}" "${STABLEHLO_FILE}" "${INPUT}" --print_output=true
else
    echo "stablehlo_runner not found. Set STABLEHLO_RUNNER_PATH."
    echo "Build from open-zkx:"
    echo "  cd open-zkx && bazel build //zkx/tools/stablehlo_runner:stablehlo_runner_main"
    exit 1
fi
