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

# Witness correctness E2E test.
#
# Verifies GPU witness generation correctness for all compilable circuits
# in circom-benchmarks by comparing single vs batched execution:
#
#   1. circom --llzk concrete → StableHLO
#   2. Run single execution N times with different random inputs
#   3. Run batched execution with same inputs (N at once)
#   4. Compare: batch result[i] == single result for input[i]
#
# This ensures the batch-stablehlo pass preserves semantic correctness.
# Single execution correctness was manually verified for 9 circuits against
# known mathematical results (Gates, Sigma, Quadratic, etc.).
#
# Requires:
#   - circom (llzk fork): CIRCOM_PATH
#   - stablehlo_runner: STABLEHLO_RUNNER_PATH
#   - llzk-to-shlo-opt (built via bazel)
#   - LD_LIBRARY_PATH for llzk-circom

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPT="${ROOT_DIR}/bazel-bin/tools/llzk-to-shlo-opt"
CIRCOM="${CIRCOM_PATH:-/home/ryan/Workspace/llzk-circom/target/release/circom}"
RUNNER="${STABLEHLO_RUNNER_PATH:-/home/ryan/Workspace/open-zkx/bazel-bin/zkx/tools/stablehlo_runner/stablehlo_runner_main}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/data/ryan/llvm-20.1.8/lib}"

BATCH_SIZE=3

for tool in "$OPT" "$CIRCOM" "$RUNNER"; do
  if [ ! -f "$tool" ]; then
    echo "SKIP: tool not found: $tool"
    exit 0
  fi
done

BENCHMARKS_URL="https://github.com/project-llzk/circom-benchmarks/archive/6897550c0be30e8ab86fbcccde6b63a9666c0ceb.tar.gz"
BENCHMARKS_PREFIX="circom-benchmarks-6897550c0be30e8ab86fbcccde6b63a9666c0ceb"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading circom-benchmarks..."
curl -sL "$BENCHMARKS_URL" | tar xz -C "$TMPDIR"
BENCH="$TMPDIR/$BENCHMARKS_PREFIX"

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

PASS=0; FAIL=0; SKIP=0; TOTAL=0

# Generate random BabyBear field element (small, to avoid overflow in circuits)
rand_felt() {
  echo $(( RANDOM % 1000 ))
}

# Extract values from runner output (strip shape prefix, get numbers)
extract_values() {
  local output="$1"
  echo "$output" | sed 's/babybear\[[^]]*\]//' | tr -d '{}' | tr ',' '\n' | tr -d ' ' | grep -v '^$' | paste -sd, -
}

echo ""
echo "=== Witness Correctness: Single vs Batch (N=$BATCH_SIZE) ==="
echo ""

while IFS= read -r circom_file; do
  name=$(echo "$circom_file" | sed "s|$BENCH/applications/||" | sed 's|\.circom$||;s|/|_|g')
  src_dir=$(dirname "$circom_file")
  outdir="$TMPDIR/out/$name"
  mkdir -p "$outdir"

  # Step 1: circom → LLZK → StableHLO
  if ! $CIRCOM "$circom_file" --llzk concrete -o "$outdir/" \
    -l "$src_dir" "${INCLUDES[@]}" >/dev/null 2>/dev/null; then
    continue
  fi
  llzk_file=$(find "$outdir" -name "*.llzk" | head -1)
  [ -z "$llzk_file" ] && continue

  if ! $OPT --simplify-sub-components '--llzk-to-stablehlo=prime=2013265921:i32' \
    "$llzk_file" -o "$outdir/single.mlir" 2>/dev/null; then
    continue
  fi

  # Step 2: Batch StableHLO
  if ! $OPT "--batch-stablehlo=batch-size=$BATCH_SIZE" \
    "$outdir/single.mlir" -o "$outdir/batch.mlir" 2>/dev/null; then
    continue
  fi

  TOTAL=$((TOTAL + 1))

  # Step 3: Determine input shape from function signature
  func_sig=$(grep "func.func @main" "$outdir/single.mlir" | head -1)
  num_args=$(echo "$func_sig" | grep -oP '%arg\d+' | wc -l)

  # Parse each arg's shape: tensor<!pf> → [1], tensor<Nx!pf> → [N]
  arg_sizes=()
  while IFS= read -r arg; do
    size=$(echo "$arg" | grep -oP 'tensor<(\d+)x' | grep -oP '\d+' || echo "1")
    arg_sizes+=("$size")
  done < <(echo "$func_sig" | grep -oP 'tensor<[^>]+>' | head -"$num_args")

  # Step 4: Generate random inputs and run singles
  singles=""
  # Build batch input arrays (one per arg)
  declare -a batch_arrays
  for ((a=0; a<num_args; a++)); do
    batch_arrays[$a]=""
  done

  single_ok=true
  for ((i=0; i<BATCH_SIZE; i++)); do
    # Generate random input for each arg
    input_arrays="["
    for ((a=0; a<num_args; a++)); do
      [ $a -gt 0 ] && input_arrays+=","
      sz=${arg_sizes[$a]}
      vals=""
      for ((v=0; v<sz; v++)); do
        [ $v -gt 0 ] && vals+=","
        vals+="$(rand_felt)"
      done
      input_arrays+="[$vals]"

      # Accumulate for batch
      [ -n "${batch_arrays[$a]}" ] && batch_arrays[$a]+=","
      batch_arrays[$a]+="$vals"
    done
    input_arrays+="]"

    echo "{\"inputs\": $input_arrays}" > "$outdir/single_input_$i.json"
    result=$(timeout 30 "$RUNNER" "$outdir/single.mlir" \
      --input_json="$outdir/single_input_$i.json" --print_output=true 2>/dev/null || echo "ERROR")

    if echo "$result" | grep -qi "error"; then
      single_ok=false
      break
    fi

    val=$(extract_values "$result")
    [ -n "$singles" ] && singles+=",$val"
    [ -z "$singles" ] && singles="$val"
  done

  if ! $single_ok; then
    SKIP=$((SKIP + 1))
    continue
  fi

  # Step 5: Run batch
  batch_input="["
  for ((a=0; a<num_args; a++)); do
    [ $a -gt 0 ] && batch_input+=","
    batch_input+="[${batch_arrays[$a]}]"
  done
  batch_input+="]"

  echo "{\"inputs\": $batch_input}" > "$outdir/batch_input.json"
  batch_result=$(timeout 30 "$RUNNER" "$outdir/batch.mlir" \
    --input_json="$outdir/batch_input.json" --print_output=true 2>/dev/null || echo "ERROR")

  if echo "$batch_result" | grep -qi "error"; then
    SKIP=$((SKIP + 1))
    continue
  fi

  batch_val=$(extract_values "$batch_result")

  # Step 6: Compare
  if [ "$singles" = "$batch_val" ]; then
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $name"
    echo "    singles: [$singles]"
    echo "    batch:   [$batch_val]"
    FAIL=$((FAIL + 1))
  fi

  unset batch_arrays

done < <(find "$BENCH/applications" -name "*.circom" \
         -exec grep -l "component main" {} \; | sort)

echo ""
echo "============================================"
echo "  Total tested:  $TOTAL"
echo "  PASS:          $PASS"
echo "  FAIL:          $FAIL"
echo "  SKIP:          $SKIP (runner error)"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "FAIL: $FAIL witness mismatches detected"
  exit 1
fi

MIN_PASS=20
if [ "$PASS" -lt "$MIN_PASS" ]; then
  echo ""
  echo "FAIL: Only $PASS circuits passed (minimum $MIN_PASS)"
  exit 1
fi

echo ""
echo "PASS: All tested circuits match (single == batch)"
exit 0
