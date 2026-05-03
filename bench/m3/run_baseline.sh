#!/usr/bin/env bash
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
#
# circom-native C++ baseline runner for the M3 measurement harness.
#
# Build steps (one-time per circuit, cached under bench/m3/_cpu_build/):
#   1. circom <main.circom> -c -o build_dir   # emit C++ witness gen
#   2. make -C build_dir/<name>_cpp           # link witness binary
#
# Measurement: run the witness binary N times sequentially with random inputs
# and report wall-clock total + throughput. Other CSV stages are zero (no
# compile/jit/kernel/d2h analog for native C++).
#
# Usage (invoked from bench/m3/run.sh):
#   run_baseline.sh <circuit_label> <N> <bazel_target> <csv_out> [iterations=1]
#
# `iterations` runs the N-loop that many times and reports the median total_ms,
# matching the gpu_zkx path's median-of-iterations methodology. Default 1 since
# the cpu_circom path's wall-clock already scales linearly with N (more samples
# come for free); pass 3+ if you need extra noise reduction at the cost of
# proportional wall-clock.

set -euo pipefail

CIRCUIT_LABEL="$1"
N="$2"
TARGET="$3"
CSV_OUT="$4"
ITERATIONS="${5:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="$SCRIPT_DIR/_cpu_build"
mkdir -p "$BUILD_ROOT"

: "${CIRCOM_PATH:?CIRCOM_PATH must point at the llzk-fork circom binary}"
CIRCOM="$CIRCOM_PATH"
if [[ ! -x "$CIRCOM" ]]; then
  echo "[run_baseline] circom binary not executable: $CIRCOM" >&2
  exit 1
fi
# The llzk fork of circom is a Rust binary that dynamically loads
# libMLIR.so.20.1; LD_LIBRARY_PATH must point at the same llvm-20 install used
# to build it. The runner does NOT default a path — set LD_LIBRARY_PATH to your
# llvm-20 lib dir before invoking the harness.
: "${LD_LIBRARY_PATH:?LD_LIBRARY_PATH must include the llvm-20 lib dir for circom}"

# Locate the .circom main file. The bazel target's `srcs` already encodes
# this path; rather than parsing BUILD.bazel we resolve via the unpacked
# circom_benchmarks repo. Bazel may keep external/ either next to execroot/
# or inside it depending on version, so check both.
BENCHMARKS_ROOT=""
for cand in \
  "$ROOT_DIR/bazel-llzk-to-shlo2/external/circom_benchmarks" \
  "$(bazel info execution_root 2>/dev/null)/external/circom_benchmarks" \
  "$(bazel info output_base 2>/dev/null)/external/circom_benchmarks"; do
  if [[ -d "$cand/applications" ]]; then
    BENCHMARKS_ROOT="$cand"
    break
  fi
done
if [[ -z "$BENCHMARKS_ROOT" ]]; then
  echo "[run_baseline] circom_benchmarks not found; run 'bazel build //examples:${TARGET}' first" >&2
  exit 1
fi

# Map bazel target name → main .circom path. Extend in lockstep with run.sh's
# alias table when adding circuits.
declare -A SRC_FOR=(
  [montgomerydouble]="applications/MontgomeryDouble/src/main.circom"
  [montgomeryadd]="applications/MontgomeryAdd/src/main.circom"
  [num2bits]="applications/Num2Bits/src/main.circom"
  [aes_256_encrypt]="applications/aes-circom/src/aes_256_encrypt_test.circom"
  [aes_256_ctr]="applications/aes-circom/src/aes_256_ctr_test.circom"
  [aes_256_key_expansion]="applications/aes-circom/src/aes_256_key_expansion_test.circom"
  [iden3_verify_credential_subject]="applications/iden3-core/src/utils_verifyCredentialSubject.circom"
  [iden3_verify_expiration_time]="applications/iden3-core/src/utils_verifyExpirationTime.circom"
  [iden3_intest]="applications/iden3-core/src/inTest.circom"
  [iden3_querytest]="applications/iden3-core/src/queryTest.circom"
  [iden3_get_value_by_index]="applications/iden3-core/src/utils_GetValueByIndex.circom"
  [iden3_get_claim_expiration]="applications/iden3-core/src/utils_getClaimExpiration.circom"
  [iden3_get_claim_subject]="applications/iden3-core/src/utils_getClaimSubjectOtherIden.circom"
  [iden3_get_subject_location]="applications/iden3-core/src/utils_getSubjectLocation.circom"
  [iden3_is_expirable]="applications/iden3-core/src/utils_isExpirable.circom"
  [iden3_is_updatable]="applications/iden3-core/src/utils_isUpdatable.circom"
  [iden3_check_iden_state]="applications/iden3-core/src/utils_checkIdenStateMatchesRoots.circom"
  [iden3_verify_claim_sig]="applications/iden3-core/src/utils_verifyCredentialNotRevoked.circom"
  [keccak_chi]="applications/keccak256-circom/src/chi_test.circom"
  [keccak_iota3]="applications/keccak256-circom/src/iota3_test.circom"
  [keccak_iota10]="applications/keccak256-circom/src/iota10_test.circom"
  [keccak_round0]="applications/keccak256-circom/src/keccakfRound0_test.circom"
  [keccak_round20]="applications/keccak256-circom/src/keccakfRound20_test.circom"
  [keccak_pad]="applications/keccak256-circom/src/pad_test.circom"
  [keccak_rhopi]="applications/keccak256-circom/src/rhopi_test.circom"
  [keccak_squeeze]="applications/keccak256-circom/src/squeeze_test.circom"
  [keccak_theta]="applications/keccak256-circom/src/theta_test.circom"
  [maci_calculate_total]="applications/maci/src/calculateTotal_test.circom"
  [maci_decrypt]="applications/maci/src/decrypt_test.circom"
  [maci_quin_generate_path_indices]="applications/maci/src/quinGeneratePathIndices_test.circom"
  [maci_quin_selector]="applications/maci/src/quinSelector_test.circom"
)
REL="${SRC_FOR[$TARGET]:-}"
if [[ -z "$REL" ]]; then
  echo "[run_baseline] No circom source mapping for target '$TARGET'." >&2
  echo "[run_baseline] Add '$TARGET' to SRC_FOR in run_baseline.sh." >&2
  exit 1
fi
MAIN="$BENCHMARKS_ROOT/$REL"
if [[ ! -f "$MAIN" ]]; then
  echo "[run_baseline] main .circom not found at $MAIN" >&2
  exit 1
fi

CIRCUIT_BUILD="$BUILD_ROOT/$TARGET"
mkdir -p "$CIRCUIT_BUILD"

# Re-run circom whenever the .circom main is newer than its generated _cpp
# output dir. circom is fast (<1s for these circuits) and silent reuse of a
# stale generated dir caused review-time confusion.
CPP_DIR=$(find "$CIRCUIT_BUILD" -maxdepth 1 -type d -name "*_cpp" | head -1)
if [[ -z "$CPP_DIR" || "$MAIN" -nt "$CPP_DIR" ]]; then
  echo "[run_baseline] generating C++ witness gen for $TARGET ..."
  INCLUDES=(
    -l "$BENCHMARKS_ROOT/libs"
    -l "$BENCHMARKS_ROOT/libs/circomlib/circuits"
    -l "$BENCHMARKS_ROOT/libs/circom-ecdsa/circuits"
    -l "$BENCHMARKS_ROOT/libs/circom-pairing/circuits"
    -l "$BENCHMARKS_ROOT/libs/circomlib-matrix/circuits"
    -l "$BENCHMARKS_ROOT/libs/ed25519"
    -l "$BENCHMARKS_ROOT/libs/zksolid-libs/circuits"
    -l "$BENCHMARKS_ROOT/applications"
    -l "$(dirname "$MAIN")"
  )
  "$CIRCOM" "$MAIN" -c -o "$CIRCUIT_BUILD" "${INCLUDES[@]}"
  CPP_DIR=$(find "$CIRCUIT_BUILD" -maxdepth 1 -type d -name "*_cpp" | head -1)
fi
if [[ -z "$CPP_DIR" ]]; then
  echo "[run_baseline] no *_cpp dir under $CIRCUIT_BUILD (circom -c failed?)" >&2
  exit 1
fi
WIT_NAME="$(basename "$CPP_DIR")"
WIT_NAME="${WIT_NAME%_cpp}"
WIT_BIN="$CPP_DIR/$WIT_NAME"

# `make` is incremental; let it re-link only when sources actually changed
# instead of guarding by an existence check that misses staleness.
make -C "$CPP_DIR" >"$CIRCUIT_BUILD/build.log" 2>&1 || {
  echo "[run_baseline] make failed; see $CIRCUIT_BUILD/build.log" >&2
  tail -30 "$CIRCUIT_BUILD/build.log" >&2
  exit 1
}

# circom-benchmarks does NOT ship per-circuit input JSONs and circom .sym
# files do not separate input from output signals, so we keep a hand-curated
# fixture per target under bench/m3/inputs/.
INPUT_FIXTURE="$SCRIPT_DIR/inputs/$TARGET.json"
INPUT_JSON="$CIRCUIT_BUILD/input.json"
if [[ ! -f "$INPUT_FIXTURE" ]]; then
  echo "[run_baseline] missing input fixture: $INPUT_FIXTURE" >&2
  echo "[run_baseline] create one with the shape declared by '$REL'." >&2
  exit 1
fi
cp "$INPUT_FIXTURE" "$INPUT_JSON"

# `date +%s%N` is GNU-only (BSD `date` on macOS lacks %N); python3 keeps the
# baseline script portable to non-Linux dev machines.
now_ns() { python3 -c 'import time; print(int(time.time() * 1e9))'; }

WTNS_OUT="$CIRCUIT_BUILD/witness.wtns"
echo "[run_baseline] running $TARGET x $N sequentially x $ITERATIONS iter(s) ..."
SAMPLES_NS=()
for ((iter=0; iter<ITERATIONS; iter++)); do
  START_NS=$(now_ns)
  for ((i=0; i<N; i++)); do
    "$WIT_BIN" "$INPUT_JSON" "$WTNS_OUT" >/dev/null 2>&1 || {
      echo "[run_baseline] witness binary failed on iter=$iter run=$i" >&2
      exit 1
    }
  done
  END_NS=$(now_ns)
  SAMPLES_NS+=("$((END_NS - START_NS))")
done

read -r MEDIAN_NS <<<"$(python3 -c '
import sys, statistics
samples = [int(x) for x in sys.argv[1:]]
print(int(statistics.median(samples)))
' "${SAMPLES_NS[@]}")"
TOTAL_MS=$(awk -v ns="$MEDIAN_NS" 'BEGIN{printf "%.6f", ns/1e6}')
THROUGHPUT=$(awk -v n="$N" -v ns="$MEDIAN_NS" 'BEGIN{
  if (ns <= 0) { print "0.000"; exit }
  printf "%.3f", n*1e9/ns
}')

# Pull the canonical CSV header + per-stage / backend labels out of
# csv_schema.h so the baseline script and the C++ runner cannot drift apart.
# Each constant is extracted by exact name to avoid order-of-definition or
# false-match-in-comment hazards. clang-format may wrap the assignment onto
# the next line for long values (kCsvHeader), so join the matched line with
# the one after it before splitting on the quote delimiter.
SCHEMA_H="$SCRIPT_DIR/csv_schema.h"
extract_schema_val() {
  grep -w "$1" -A1 "$SCHEMA_H" | tr '\n' ' ' | awk -F'"' '{print $2}'
}
CSV_HEADER=$(extract_schema_val kCsvHeader)
CPU_BACKEND=$(extract_schema_val kBackendCpuCircom)
STAGE_COMPILE=$(extract_schema_val kStageCompile)
STAGE_JIT=$(extract_schema_val kStageJit)
STAGE_KERNEL=$(extract_schema_val kStageKernel)
STAGE_D2H=$(extract_schema_val kStageD2H)
STAGE_TOTAL=$(extract_schema_val kStageTotal)

if [[ ! -f "$CSV_OUT" ]]; then
  echo "$CSV_HEADER" >"$CSV_OUT"
fi
{
  for stage in "$STAGE_COMPILE" "$STAGE_JIT" "$STAGE_KERNEL" "$STAGE_D2H"; do
    echo "${CIRCUIT_LABEL},${CPU_BACKEND},${N},${stage},0.000000,"
  done
  echo "${CIRCUIT_LABEL},${CPU_BACKEND},${N},${STAGE_TOTAL},${TOTAL_MS},${THROUGHPUT}"
} >>"$CSV_OUT"

echo "[run_baseline] $TARGET N=$N iter=${ITERATIONS} total(med)=${TOTAL_MS}ms throughput=${THROUGHPUT} wits/s -> $CSV_OUT"
