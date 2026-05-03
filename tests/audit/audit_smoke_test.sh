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

# Smoke test for the witness_layout_audit tool.
#
# Runs the audit binary against two synthetic StableHLO modules — one whose
# DUS chain has only real values, one whose chain contains a length-16 splat
# zero (the AES-class silent-fallback pattern) — and asserts the human +
# JSON outputs report what we expect.

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

AUDIT=$(rlocation "llzk_to_shlo/tools/witness_layout_audit")
CLEAN=$(rlocation "llzk_to_shlo/tests/audit/synthetic_clean.mlir")
ORPHAN=$(rlocation "llzk_to_shlo/tests/audit/synthetic_orphan_zero.mlir")

PASS=0
FAIL=0

assert_contains() {
  local what="$1"
  local needle="$2"
  local haystack="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL [$what]: expected substring not found"
    echo "  needle:   $needle"
    echo "  output:"
    printf '%s\n' "$haystack" | sed 's/^/    /'
    FAIL=$((FAIL + 1))
  fi
}

# ---- clean fixture (no splat-zero chunks) ----
out=$("$AUDIT" "$CLEAN")
assert_contains "clean.human.chunks" "chunks=3"             "$out"
assert_contains "clean.human.totlen" "total_length=8"       "$out"
assert_contains "clean.human.zero"   "splat_zero=0"         "$out"
assert_contains "clean.human.add"    "stablehlo.add"        "$out"
assert_contains "clean.human.mul"    "stablehlo.multiply"   "$out"
assert_contains "clean.human.sub"    "stablehlo.subtract"   "$out"
# No chunk should be marked ZERO!!! in the clean fixture.
if [[ "$out" == *"ZERO!!!"* ]]; then
  echo "FAIL [clean.human.no-zero-mark]: clean fixture should not flag any chunks"
  FAIL=$((FAIL + 1))
else
  PASS=$((PASS + 1))
fi

out=$("$AUDIT" "$CLEAN" --format=json)
assert_contains "clean.json.zerocount" '"splat_zero_count":0' "$out"
assert_contains "clean.json.totlen"    '"total_length":8'     "$out"

# ---- orphan fixture (one length-16 splat-zero chunk) ----
out=$("$AUDIT" "$ORPHAN")
assert_contains "orphan.human.chunks"     "chunks=3"               "$out"
assert_contains "orphan.human.totlen"     "total_length=21"        "$out"
assert_contains "orphan.human.zero1"      "splat_zero=1"           "$out"
assert_contains "orphan.human.zerorow"    "ZERO!!!   | stablehlo.constant" "$out"

out=$("$AUDIT" "$ORPHAN" --format=json)
assert_contains "orphan.json.zerocount"   '"splat_zero_count":1'   "$out"
assert_contains "orphan.json.totlen"      '"total_length":21'      "$out"
assert_contains "orphan.json.is_splat"    '"is_splat_zero":true'   "$out"
assert_contains "orphan.json.length16"    '"length":16'            "$out"

# ---- error path: missing function ----
err_out=$("$AUDIT" "$CLEAN" --func=does_not_exist 2>&1) && err_rc=0 || err_rc=$?
if [[ "$err_rc" -ne 1 ]]; then
  echo "FAIL [missing-fn.exitcode]: expected rc=1, got $err_rc"
  FAIL=$((FAIL + 1))
else
  PASS=$((PASS + 1))
fi
assert_contains "missing-fn.message" "function @does_not_exist not found" "$err_out"

echo "witness_layout_audit smoke test: $PASS pass, $FAIL fail"
[[ "$FAIL" -eq 0 ]] || exit 1
