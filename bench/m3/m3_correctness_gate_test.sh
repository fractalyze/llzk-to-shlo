#!/bin/bash
# Copyright 2026 The llzk-to-shlo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================================================
#
# M3 correctness-gate regression test.
#
# Runs `m3_runner` with `--correctness_gate=true` for each gateable circuit
# and asserts the GPU output byte-equals the committed `.wtns` reference. A
# regression in any lowering pass that breaks witness correctness on a chip
# whose gate previously passed will turn this test red.
#
# Per-chip data flow:
#   1. <chip>.stablehlo.mlir   ← bazel build //examples:<chip>
#   2. <chip>.json             ← committed in bench/m3/inputs/
#   3. <chip>.wtns             ← committed in bench/m3/inputs/
#                                (regenerate via cpu_circom path if changed)
#   4. <chip>.json.gate        ← optional sentinel; empty = contiguous default
#
# Adding a new chip: add its bazel target to the `data` block of
# `m3_correctness_gate_test` in bench/m3/BUILD.bazel and append a row to the
# CHIPS table below.

set -euo pipefail

RDIR="${RUNFILES_DIR:-${TEST_SRCDIR:-}}"
if [[ -z "$RDIR" ]]; then
  echo "FAIL: bazel runfiles dir not set"
  exit 1
fi

# Bazel runfiles workspace dir. WORKSPACE.bazel sets `workspace(name="llzk_to_shlo")`
# so runfiles land under `<runfiles>/llzk_to_shlo/...` (NOT `_main/` — that's
# the bzlmod default which `common --noenable_bzlmod` in `.bazelrc` opts out of).
WS="llzk_to_shlo"
RUNNER="$RDIR/$WS/bench/m3/m3_runner"
if [[ ! -x "$RUNNER" ]]; then
  echo "FAIL: m3_runner not found at $RUNNER"
  exit 1
fi

# Resolve a runfile path under $WS, or fail with a clear message.
resolve() {
  local p="$RDIR/$WS/$1"
  [[ -e "$p" ]] || { echo "FAIL: missing runfile $1 (looked at $p)"; exit 1; }
  echo "$p"
}

# Each row: bazel-target-name (= input fixture name stem)
# All rows assume:
#   - examples/<target>.stablehlo.mlir is the lowered IR
#   - bench/m3/inputs/<target>.{json,wtns} exist
#   - bench/m3/inputs/<target>.json.gate exists (may be empty)
CHIPS=(
  iden3_get_claim_expiration
  iden3_get_claim_subject
  iden3_get_subject_location
  iden3_get_value_by_index
  iden3_intest
  iden3_is_expirable
  iden3_is_updatable
  iden3_querytest
  iden3_verify_credential_subject
  iden3_verify_expiration_time
  keccak_chi
  keccak_iota10
  keccak_iota3
  keccak_pad
  keccak_rhopi
  keccak_round0
  keccak_round20
  keccak_squeeze
  keccak_theta
  maci_calculate_total
  maci_quin_generate_path_indices
  maci_quin_selector
  montgomerydouble
)

FAIL=0
for chip in "${CHIPS[@]}"; do
  echo "=== $chip ==="
  mlir=$(resolve "examples/${chip}.stablehlo.mlir")
  json=$(resolve "bench/m3/inputs/${chip}.json")
  wtns=$(resolve "bench/m3/inputs/${chip}.wtns")
  gate=$(resolve "bench/m3/inputs/${chip}.json.gate")
  indices="$(tr -s '[:space:]' ' ' < "$gate" | sed 's/^ *//;s/ *$//')"
  set +e
  out=$("$RUNNER" "$mlir" \
    --circuit="$chip" --N=1 --iterations=1 --warmups=0 \
    --input_json="$json" \
    --correctness_gate=true \
    --gate_wtns_path="$wtns" \
    --gate_wtns_indices="$indices" 2>&1)
  rc=$?
  set -e
  if [[ $rc -ne 0 ]] || ! grep -q 'correctness gate PASSED' <<<"$out"; then
    echo "FAIL: $chip gate did not pass (rc=$rc)"
    echo "$out" | tail -10
    FAIL=$((FAIL + 1))
  else
    echo "  PASSED"
  fi
done

if [[ $FAIL -gt 0 ]]; then
  echo "FAIL: $FAIL chip(s) failed correctness gate"
  exit 1
fi
echo "PASS: all ${#CHIPS[@]} chip(s) gated"
exit 0
