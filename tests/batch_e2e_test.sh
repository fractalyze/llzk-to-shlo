#!/bin/bash
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

# Batch StableHLO E2E regression test.
# Tests: circom → LLZK (concrete) → StableHLO → batch-stablehlo(N=4)
# Verifies that the number of successful conversions doesn't regress.

# --- begin runfiles.bash initialization v3 ---
set -uo pipefail
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }
set -e
# --- end runfiles.bash initialization v3 ---

OPT=$(rlocation "llzk_to_shlo/tools/llzk-to-shlo-opt")
CIRCOM=$(rlocation "circom/circom_wrapper.sh")

# circom_benchmarks is a directory; find its root via a known file.
BENCH_MARKER=$(rlocation "circom_benchmarks/applications/fulladder/src/main.circom" 2>/dev/null || echo "")
if [ -z "$BENCH_MARKER" ]; then
  echo "SKIP: circom_benchmarks not found in runfiles"
  exit 0
fi
BENCH=$(cd "$(dirname "$BENCH_MARKER")/../../.." && pwd)

if [ ! -f "$OPT" ]; then
  echo "SKIP: llzk-to-shlo-opt not found"
  exit 0
fi
if [ ! -f "$CIRCOM" ] && ! command -v "$CIRCOM" &>/dev/null; then
  echo "SKIP: circom not found (set CIRCOM_PATH env)"
  exit 0
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Include paths for circom
INCLUDES=(
  -l "$BENCH/libs"
  -l "$BENCH/libs/circomlib/circuits"
  -l "$BENCH/libs/circom-ecdsa/circuits"
  -l "$BENCH/libs/circom-pairing/circuits"
  -l "$BENCH/libs/circomlib-matrix/circuits"
  -l "$BENCH/libs/ed25519"
  -l "$BENCH/libs/zksolid-libs/circuits"
  -l "$BENCH/applications"
)

CIRCOM_OK=0
SHLO_OK=0; SHLO_FAIL=0
BATCH_OK=0; BATCH_FAIL=0
TOTAL=0

echo "=== Batch StableHLO E2E Regression Test ==="

while IFS= read -r circom_file; do
  name=$(echo "$circom_file" | sed "s|$BENCH/applications/||" | sed 's|\.circom$||;s|/|_|g')
  src_dir=$(dirname "$circom_file")
  outdir="$TMPDIR/out/$name"
  mkdir -p "$outdir"

  # Step 1: circom → LLZK (concrete)
  if ! $CIRCOM "$circom_file" --llzk concrete -o "$outdir/" \
    -l "$src_dir" "${INCLUDES[@]}" >/dev/null 2>/dev/null; then
    continue
  fi
  llzk_file=$(find "$outdir" -name "*.llzk" | head -1)
  [ -z "$llzk_file" ] && continue
  CIRCOM_OK=$((CIRCOM_OK + 1))
  TOTAL=$((TOTAL + 1))

  # Step 2: LLZK → StableHLO
  if ! $OPT --simplify-sub-components --llzk-to-stablehlo="prime=2013265921:i32" \
    "$llzk_file" -o "$outdir/shlo.mlir" 2>/dev/null; then
    SHLO_FAIL=$((SHLO_FAIL + 1))
    continue
  fi
  SHLO_OK=$((SHLO_OK + 1))

  # Step 3: StableHLO → Batch(N=4)
  if ! $OPT --batch-stablehlo="batch-size=4" \
    "$outdir/shlo.mlir" -o "$outdir/batch.mlir" 2>/dev/null; then
    BATCH_FAIL=$((BATCH_FAIL + 1))
    continue
  fi
  BATCH_OK=$((BATCH_OK + 1))

done < <(find "$BENCH/applications" -name "*.circom" \
         -exec grep -l "component main" {} \; | sort)

echo ""
echo "  Circom → LLZK (concrete): $CIRCOM_OK"
echo "  LLZK → StableHLO:         $SHLO_OK / $TOTAL  (fail: $SHLO_FAIL)"
echo "  StableHLO → Batch(N=4):   $BATCH_OK / $SHLO_OK  (fail: $BATCH_FAIL)"
echo ""

# --- Regression thresholds ---
# These are the minimum expected counts. If any drops below, the test fails.
# Update these when new lowering patterns are added.
MIN_SHLO=41
MIN_BATCH=41

EXIT_CODE=0

if [ "$SHLO_OK" -lt "$MIN_SHLO" ]; then
  echo "FAIL: StableHLO conversions regressed: $SHLO_OK < $MIN_SHLO"
  EXIT_CODE=1
fi

if [ "$BATCH_OK" -lt "$MIN_BATCH" ]; then
  echo "FAIL: Batch conversions regressed: $BATCH_OK < $MIN_BATCH"
  EXIT_CODE=1
fi

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "PASS (StableHLO=$SHLO_OK >= $MIN_SHLO, Batch=$BATCH_OK >= $MIN_BATCH)"
fi

exit $EXIT_CODE
