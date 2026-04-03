#!/bin/bash
# Copyright 2026 The llzk-to-shlo Authors.
#
# GPU semantic verification tests.
# Converts LLZK circuits to StableHLO and runs them on GPU with known inputs,
# verifying the output matches expected values.
#
# Requires: stablehlo_runner_main (from open-zkx)

set -euo pipefail

# --- Tool paths (supports both bazel runfiles and direct execution) ---
# Bazel sets RUNFILES_DIR or TEST_SRCDIR for runfiles lookup.
RDIR="${RUNFILES_DIR:-${TEST_SRCDIR:-}}"
if [ -n "$RDIR" ] && [ -f "$RDIR/llzk_to_shlo/tools/llzk-to-shlo-opt" ]; then
  OPT="$RDIR/llzk_to_shlo/tools/llzk-to-shlo-opt"
  RUNNER="$RDIR/open_zkx/zkx/tools/stablehlo_runner/stablehlo_runner_main"
elif [ -n "${STABLEHLO_RUNNER_PATH:-}" ]; then
  # Direct execution with env var
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  OPT="${SCRIPT_DIR}/../bazel-bin/tools/llzk-to-shlo-opt"
  RUNNER="$STABLEHLO_RUNNER_PATH"
else
  echo "SKIP: cannot find tools (set STABLEHLO_RUNNER_PATH or run via bazel)"
  exit 0
fi

if [ ! -f "$RUNNER" ] && ! command -v "$RUNNER" &>/dev/null; then
  echo "SKIP: stablehlo_runner not found at $RUNNER"
  exit 0
fi
if [ ! -f "$OPT" ]; then
  echo "SKIP: llzk-to-shlo-opt not found at $OPT"
  exit 0
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PASS=0
FAIL=0

run_test() {
  local name="$1" circuit="$2" input_json="$3" expected="$4" opts="${5:-}"
  echo -n "  $name: "

  # Convert
  if ! $OPT $opts --llzk-to-stablehlo="prime=2013265921:i32" "$circuit" \
       > "$TMPDIR/circuit.mlir" 2>"$TMPDIR/convert_err.txt"; then
    echo "FAIL (conversion: $(head -1 "$TMPDIR/convert_err.txt"))"
    FAIL=$((FAIL + 1))
    return
  fi

  # Run on GPU
  echo "$input_json" > "$TMPDIR/input.json"
  local result
  result=$(timeout 30 "$RUNNER" "$TMPDIR/circuit.mlir" \
           --input_json="$TMPDIR/input.json" --print_output=true 2>&1) || {
    echo "FAIL (runner error: $(echo "$result" | tail -1))"
    FAIL=$((FAIL + 1))
    return
  }

  # Extract output values
  local values
  values=$(echo "$result" | grep -oP '\{[^}]+\}' | head -1 || echo "")

  if [ "$values" = "$expected" ]; then
    echo "PASS ($values)"
    PASS=$((PASS + 1))
  else
    echo "FAIL (expected $expected, got $values)"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== GPU Semantic Verification Tests ==="

# --- Gates: out = a * b ---
echo "Gates circuit:"
# Create inline LLZK for gates
cat > "$TMPDIR/gates.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Main<[]>>} {
  struct.def @Main<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@Main<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Main<[]>>
      %0 = felt.mul %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Main<[]>>, !felt.type
      function.return %self : !struct.type<@Main<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main<[]>>, %a: !felt.type, %b: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
LLZK
run_test "3 * 5 = 15" "$TMPDIR/gates.llzk" '{"inputs": [[3], [5]]}' '{15}'
run_test "7 * 8 = 56" "$TMPDIR/gates.llzk" '{"inputs": [[7], [8]]}' '{56}'
run_test "0 * 99 = 0" "$TMPDIR/gates.llzk" '{"inputs": [[0], [99]]}' '{0}'

# --- Sigma: out = x⁵ ---
echo "Sigma circuit (x⁵):"
cat > "$TMPDIR/sigma.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Sigma<[]>>} {
  struct.def @Sigma<[]> {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @in2 : !felt.type
    struct.member @in4 : !felt.type
    function.def @compute(%x: !felt.type) -> !struct.type<@Sigma<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sigma<[]>>
      %x2 = felt.mul %x, %x : !felt.type, !felt.type
      struct.writem %self[@in2] = %x2 : <@Sigma<[]>>, !felt.type
      %x4 = felt.mul %x2, %x2 : !felt.type, !felt.type
      struct.writem %self[@in4] = %x4 : <@Sigma<[]>>, !felt.type
      %x5 = felt.mul %x4, %x : !felt.type, !felt.type
      struct.writem %self[@out] = %x5 : <@Sigma<[]>>, !felt.type
      function.return %self : !struct.type<@Sigma<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sigma<[]>>, %x: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
LLZK
run_test "2⁵ = 32"  "$TMPDIR/sigma.llzk" '{"inputs": [[2]]}' '{32, 4, 16}'
run_test "3⁵ = 243" "$TMPDIR/sigma.llzk" '{"inputs": [[3]]}' '{243, 9, 81}'
run_test "1⁵ = 1"   "$TMPDIR/sigma.llzk" '{"inputs": [[1]]}' '{1, 1, 1}'

# --- NOT gate: out = 1 - in ---
echo "NOT gate:"
cat > "$TMPDIR/not.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Not<[]>>} {
  struct.def @Not<[]> {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type) -> !struct.type<@Not<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Not<[]>>
      %one = felt.const 1
      %0 = felt.sub %one, %a : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : <@Not<[]>>, !felt.type
      function.return %self : !struct.type<@Not<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Not<[]>>, %a: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
LLZK
run_test "NOT(0) = 1" "$TMPDIR/not.llzk" '{"inputs": [[0]]}' '{1}'
run_test "NOT(1) = 0" "$TMPDIR/not.llzk" '{"inputs": [[1]]}' '{0}'

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
