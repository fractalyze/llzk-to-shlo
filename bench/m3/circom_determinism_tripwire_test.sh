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

# Determinism tripwire for project-llzk/circom.
#
# Runs `circom --llzk concrete` twice on splicer_test.circom on the same host
# and asserts the two LLZK outputs are NOT byte-equal.
#
# - PASS while the upstream determinism bug exists (current state). The
#   golden-LLZK indirection in `examples/BUILD.bazel` (see CLAUDE.md
#   "Multi-sub-component composite chips") is required to gate composite chips
#   like maci_splicer.
# - FAIL once upstream lands the determinism fix. The failure is the signal
#   to retire the golden indirection: switch maci_splicer back to
#   `circom_to_stablehlo`, drop `examples/maci_splicer_llzk.llzk.golden`, and
#   remove the `golden_llzk_to_stablehlo` macro from `examples/e2e.bzl`.
#
# Mechanism (suspected — not yet source-confirmed): default Rust `HashMap`
# `RandomState` iteration order in circom's symbol-table or sub-component
# registry. See `memory/circom-non-determinism-llzk-shape-followup.md`.

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

CIRCOM=$(rlocation "circom/circom_wrapper.sh")
SRC=$(rlocation "circom_benchmarks/applications/maci/src/splicer_test.circom")

if [[ -z "$CIRCOM" || ! -x "$CIRCOM" ]]; then
  echo "SKIP: circom not found in runfiles (check @circom dependency and rebuild)"
  exit 0
fi
if [[ -z "$SRC" || ! -f "$SRC" ]]; then
  echo "FAIL: splicer_test.circom not found in runfiles"
  exit 1
fi

# SRC = <BENCH>/applications/maci/src/splicer_test.circom — climb 4 dirnames.
BENCH=$(dirname "$(dirname "$(dirname "$(dirname "$SRC")")")")

run_dir() {
  local out=$1
  mkdir -p "$out"
  "$CIRCOM" "$SRC" --llzk concrete --llzk_plaintext -o "$out" \
    -l "$BENCH/libs/circomlib/circuits" \
    -l "$BENCH/applications/maci/src" \
    >/dev/null
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run_dir "$WORK/r1"
run_dir "$WORK/r2"

LLZK1=$(find "$WORK/r1" -name "splicer_test.llzk" -print -quit)
LLZK2=$(find "$WORK/r2" -name "splicer_test.llzk" -print -quit)

if [[ -z "$LLZK1" || -z "$LLZK2" || ! -f "$LLZK1" || ! -f "$LLZK2" ]]; then
  echo "FAIL: circom did not emit splicer_test.llzk in both runs"
  ls -la "$WORK/r1" "$WORK/r2" || true
  exit 1
fi

MD5_1=$(md5sum "$LLZK1" | awk '{print $1}')
MD5_2=$(md5sum "$LLZK2" | awk '{print $1}')

echo "run1 md5: $MD5_1  ($(wc -c < "$LLZK1") bytes)"
echo "run2 md5: $MD5_2  ($(wc -c < "$LLZK2") bytes)"

if [[ "$MD5_1" == "$MD5_2" ]]; then
  echo
  echo "TRIPWIRE FIRED: circom emitted byte-identical LLZK twice in a row."
  echo "Upstream project-llzk/circom may have landed a determinism fix."
  echo "Action: retire the golden-LLZK indirection."
  echo "  - examples/BUILD.bazel: switch maci_splicer back to circom_to_stablehlo"
  echo "  - examples/maci_splicer_llzk.llzk.golden: delete"
  echo "  - examples/e2e.bzl: remove golden_llzk_to_stablehlo macro"
  echo "  - bench/m3/BUILD.bazel: drop this tripwire test"
  echo "See CLAUDE.md \"Multi-sub-component composite chips\"."
  exit 1
fi

echo
echo "PASS: circom is non-deterministic as expected (golden-LLZK indirection still required)."
exit 0
