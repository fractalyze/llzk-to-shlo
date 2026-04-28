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
# M3 measurement harness top-level dispatcher.
#
# Usage:
#   bash bench/m3/run.sh <circuit> <N> <backend> [iterations]
#
# Args:
#   circuit:  display label written to CSV (e.g. MontgomeryDouble). Aliased to
#             a bazel target name via the table below; falls back to using the
#             label as the bazel target name when no alias exists.
#   N:        batch size. {1, 64, 4096, 65536, 262144} per shared contract.
#   backend:  gpu_zkx | cpu_circom
#   iterations: optional, default 3 (post-warmup median).
#
# Output:
#   bench/m3/results/<circuit>_<backend>.csv (appended).
#
# Notes:
#   - For gpu_zkx: builds llzk-to-shlo-opt and m3_runner via bazel; runs on the
#     local GPU. For N>1, batches the StableHLO before running.
#   - For cpu_circom: dispatches to bench/m3/run_baseline.sh.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <circuit> <N> <backend> [iterations]" >&2
  exit 2
fi

CIRCUIT_LABEL="$1"
N="$2"
BACKEND="$3"
ITERATIONS="${4:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
CSV_OUT="$RESULTS_DIR/${CIRCUIT_LABEL}_${BACKEND}.csv"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Display-name → bazel-target alias. Extend as Phase 1/2 circuits get measured.
declare -A ALIAS=(
  [MontgomeryDouble]=montgomerydouble
  [MontgomeryAdd]=montgomeryadd
  [Multiplier2]=simple
  [Num2Bits]=num2bits
  [AES-256-encrypt]=aes_256_encrypt
  [AES-256-ctr]=aes_256_ctr
  [AES-256-key-expansion]=aes_256_key_expansion
  [iden3_verify_credential_subject]=iden3_verify_credential_subject
  [SHA-256]=sha256
  [keccak_chi]=keccak_chi
  [keccak_iota3]=keccak_iota3
  [keccak_iota10]=keccak_iota10
  [keccak_round0]=keccak_round0
  [keccak_round20]=keccak_round20
  [keccak_pad]=keccak_pad
  [keccak_rhopi]=keccak_rhopi
  [keccak_squeeze]=keccak_squeeze
  [keccak_theta]=keccak_theta
)
TARGET="${ALIAS[$CIRCUIT_LABEL]:-$CIRCUIT_LABEL}"

case "$BACKEND" in
  gpu_zkx)
    ;;
  cpu_circom)
    # Pass ITERATIONS through so the cpu_circom path also produces a median
    # (parity with gpu_zkx). Note: cpu_circom wall-clock scales as N x
    # ITERATIONS sequentially — for the M3 N grid {1..262144}, callers may
    # want to override with a smaller count for the largest N values.
    exec bash "$SCRIPT_DIR/run_baseline.sh" \
      "$CIRCUIT_LABEL" "$N" "$TARGET" "$CSV_OUT" "$ITERATIONS"
    ;;
  *)
    echo "Unknown backend: $BACKEND (expected gpu_zkx or cpu_circom)" >&2
    exit 2
    ;;
esac

# ---- gpu_zkx path ----
echo "[run.sh] gpu_zkx circuit=$CIRCUIT_LABEL target=$TARGET N=$N iterations=$ITERATIONS"

# Build single StableHLO + opt + m3_runner. m3_runner pulls in the open-zkx
# CUDA-enabled GPU compiler stack, so we always use --config=cuda_clang_official
# (clang-20 + nvcc); the lighter-weight //tools and //examples targets ignore
# the flag and rebuild only if their inputs changed.
BAZEL_CONFIG="${BAZEL_CONFIG:-cuda_clang_official}"
cd "$ROOT_DIR"
# `circom_to_stablehlo` runs the llzk-fork circom inside the bazel sandbox.
# That binary needs libMLIR.so.20.1; the Nix-built `/usr/local/bin/circom`
# (CI_SETUP_GUIDE.md) carries the right RPATH, so do NOT set CIRCOM_PATH or
# inject LD_LIBRARY_PATH via --action_env when invoking bazel — it would
# poison clang in the same sandbox with a mismatched libLLVM. Set CIRCOM_PATH
# only for the cpu_circom path (run_baseline.sh) where we run circom directly.
bazel build "--config=${BAZEL_CONFIG}" \
  "//examples:${TARGET}" //tools:llzk-to-shlo-opt //bench/m3:m3_runner

SINGLE_MLIR="$ROOT_DIR/bazel-bin/examples/${TARGET}.stablehlo.mlir"
if [[ ! -f "$SINGLE_MLIR" ]]; then
  echo "[run.sh] StableHLO output not found at $SINGLE_MLIR" >&2
  exit 1
fi

# N==1 keeps the unbatched MLIR shape that M2 Nsight measured against.
# N>1  prepends a leading batch dim via --batch-stablehlo.
if [[ "$N" == "1" ]]; then
  RUN_MLIR="$SINGLE_MLIR"
else
  RUN_MLIR="$TMP_DIR/batch_${N}.mlir"
  "$ROOT_DIR/bazel-bin/tools/llzk-to-shlo-opt" \
    "--batch-stablehlo=batch-size=$N" \
    "$SINGLE_MLIR" -o "$RUN_MLIR"
fi

# gpu_zkx and cpu_circom must consume the same JSON fixture so the
# per-witness comparison (PR-C) is meaningful. The fixture is required —
# m3_runner errors out if it's missing, matching run_baseline.sh's behavior.
INPUT_FIXTURE="$SCRIPT_DIR/inputs/${TARGET}.json"
if [[ ! -f "$INPUT_FIXTURE" ]]; then
  echo "[run.sh] missing input fixture: $INPUT_FIXTURE" >&2
  echo "[run.sh] create one with the shape declared by examples/${TARGET}." >&2
  exit 1
fi

# Correctness gate: per-circuit opt-in via the presence of a `.json.gate`
# sentinel next to the JSON fixture. Sentinel content is a space- or
# comma-separated list of .wtns wire indices (one per output Literal element);
# empty file means default contiguous [1..1+N). Requires <TARGET>.wtns committed
# next to the JSON.
SENTINEL="$SCRIPT_DIR/inputs/${TARGET}.json.gate"
WTNS_FIXTURE="$SCRIPT_DIR/inputs/${TARGET}.wtns"
GATE_FLAGS=()
if [[ -f "$SENTINEL" && "$N" == "1" ]]; then
  if [[ ! -f "$WTNS_FIXTURE" ]]; then
    echo "[run.sh] gate sentinel $SENTINEL present but $WTNS_FIXTURE missing" >&2
    exit 1
  fi
  GATE_INDICES="$(tr -s '[:space:]' ' ' <"$SENTINEL" | sed -e 's/^ *//' -e 's/ *$//')"
  GATE_FLAGS=(
    --correctness_gate=true
    --gate_wtns_path="$WTNS_FIXTURE"
    --gate_wtns_indices="$GATE_INDICES"
  )
  echo "[run.sh] correctness gate ON for $TARGET (.wtns=$WTNS_FIXTURE, indices='${GATE_INDICES:-<default contiguous>}')"
elif [[ -f "$SENTINEL" ]]; then
  # The gate is N=1 single-tensor only (see CLAUDE.md "M3 correctness gate
  # convention"); the .wtns file holds one witness, so a batched output of N
  # copies would have to tile indices N times. Skip rather than fail at N>1
  # so the same circuit can be both gated (at N=1) and measured (at N>1).
  echo "[run.sh] correctness gate SKIPPED for $TARGET at N=$N (gate is N=1 only)"
else
  echo "[run.sh] correctness gate SKIPPED for $TARGET — no $SENTINEL"
fi

"$ROOT_DIR/bazel-bin/bench/m3/m3_runner" \
  --circuit="$CIRCUIT_LABEL" \
  --N="$N" \
  --iterations="$ITERATIONS" \
  --csv_out="$CSV_OUT" \
  --append \
  --input_json="$INPUT_FIXTURE" \
  "${GATE_FLAGS[@]}" \
  "$RUN_MLIR"

echo "[run.sh] wrote $CSV_OUT"
