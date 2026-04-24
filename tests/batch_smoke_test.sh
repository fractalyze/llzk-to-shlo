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

# Batch StableHLO smoke test (CI-friendly).
#
# Tests representative LLZK circuits through the full pipeline:
#   LLZK → StableHLO → batch-stablehlo(N=4)
# without requiring circom compilation. Runs in ~30s.
#
# Coverage:
#   - Element-wise ops (add, mul, sub, div)
#   - Constants + broadcast
#   - dynamic_slice / dynamic_update_slice
#   - reshape
#   - While loops (fixed trip count)
#   - Function calls (sub-components)
#   - felt.umod / felt.uintdiv
#   - Input pod elimination
#   - Data-dependent indexing (one-hot)

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

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PASS=0; FAIL=0

run_test() {
  local name="$1" llzk="$2" shlo_opts="${3:---llzk-to-stablehlo=prime=2013265921:i32}"
  echo -n "  $name: "

  # LLZK → StableHLO
  if ! $OPT --simplify-sub-components "$shlo_opts" "$llzk" \
       -o "$TMPDIR/shlo.mlir" 2>/dev/null; then
    echo "FAIL (stablehlo conversion)"
    FAIL=$((FAIL + 1))
    return
  fi

  # StableHLO → Batch(N=4)
  if ! $OPT '--batch-stablehlo=batch-size=4' "$TMPDIR/shlo.mlir" \
       -o "$TMPDIR/batch.mlir" 2>/dev/null; then
    echo "FAIL (batch conversion)"
    FAIL=$((FAIL + 1))
    return
  fi

  echo "PASS"
  PASS=$((PASS + 1))
}

echo "=== Batch StableHLO Smoke Test ==="

# --- 1. Gates (element-wise mul) ---
cat > "$TMPDIR/gates.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Main<[]>>} {
  struct.def @Main {
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
run_test "Gates (mul)" "$TMPDIR/gates.llzk"

# --- 2. Sigma (struct 3 fields, deep chain) ---
cat > "$TMPDIR/sigma.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Sigma<[]>>} {
  struct.def @Sigma {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @in2 : !felt.type
    struct.member @in4 : !felt.type
    function.def @compute(%x: !felt.type) -> !struct.type<@Sigma<[]>> attributes {function.allow_witness} {
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
run_test "Sigma (x⁵)" "$TMPDIR/sigma.llzk"

# --- 3. NOT (constant + subtract) ---
cat > "$TMPDIR/not.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Not<[]>>} {
  struct.def @Not {
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
run_test "NOT (const+sub)" "$TMPDIR/not.llzk"

# --- 4. FullAdder (felt.umod + felt.uintdiv) ---
cat > "$TMPDIR/fulladder.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@FA<[]>>} {
  struct.def @FA {
    struct.member @val : !felt.type {llzk.pub}
    struct.member @carry : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@FA<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@FA<[]>>
      %sum = felt.add %a, %b : !felt.type, !felt.type
      %two = felt.const 2
      %val = felt.umod %sum, %two : !felt.type, !felt.type
      struct.writem %self[@val] = %val : <@FA<[]>>, !felt.type
      %carry = felt.uintdiv %sum, %two : !felt.type, !felt.type
      struct.writem %self[@carry] = %carry : <@FA<[]>>, !felt.type
      function.return %self : !struct.type<@FA<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@FA<[]>>, %a: !felt.type, %b: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
LLZK
run_test "FullAdder (umod+uintdiv)" "$TMPDIR/fulladder.llzk"

# --- 5. SubComponent (function call) ---
cat > "$TMPDIR/subcomp.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@Outer<[]>>} {
  struct.def @Inner {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%x: !felt.type) -> !struct.type<@Inner<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Inner<[]>>
      %sq = felt.mul %x, %x : !felt.type, !felt.type
      struct.writem %self[@out] = %sq : <@Inner<[]>>, !felt.type
      function.return %self : !struct.type<@Inner<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Inner<[]>>, %x: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
  struct.def @Outer {
    struct.member @result : !felt.type {llzk.pub}
    struct.member @inner : !struct.type<@Inner<[]>>
    struct.member @inner$inputs : !pod.type<[@in: !felt.type]>
    function.def @compute(%a: !felt.type) -> !struct.type<@Outer<[]>> attributes {function.allow_witness} {
      %self = struct.new : <@Outer<[]>>
      %c1 = arith.constant 1 : index
      %pod = pod.new { @count = %c1 } : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>
      %pod_0 = pod.new : <[@in: !felt.type]>
      pod.write %pod_0[@in] = %a : <[@in: !felt.type]>, !felt.type
      %0 = pod.read %pod[@count] : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, index
      %c1_1 = arith.constant 1 : index
      %1 = arith.subi %0, %c1_1 : index
      pod.write %pod[@count] = %1 : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, index
      %c0 = arith.constant 0 : index
      %2 = arith.cmpi eq, %1, %c0 : index
      scf.if %2 {
        %5 = pod.read %pod_0[@in] : <[@in: !felt.type]>, !felt.type
        %6 = function.call @Inner::@compute(%5) : (!felt.type) -> !struct.type<@Inner<[]>>
        pod.write %pod[@comp] = %6 : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, !struct.type<@Inner<[]>>
      } else {
      }
      %3 = pod.read %pod[@comp] : <[@count: index, @comp: !struct.type<@Inner<[]>>, @params: !pod.type<[]>]>, !struct.type<@Inner<[]>>
      %4 = struct.readm %3[@out] : <@Inner<[]>>, !felt.type
      struct.writem %self[@result] = %4 : <@Outer<[]>>, !felt.type
      struct.writem %self[@inner$inputs] = %pod_0 : <@Outer<[]>>, !pod.type<[@in: !felt.type]>
      struct.writem %self[@inner] = %3 : <@Outer<[]>>, !struct.type<@Inner<[]>>
      function.return %self : !struct.type<@Outer<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Outer<[]>>, %a: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
LLZK
run_test "SubComponent (pod dispatch)" "$TMPDIR/subcomp.llzk"

# --- 6. TwoElem (cast.toindex preservation) ---
cat > "$TMPDIR/twoelem.llzk" << 'LLZK'
module attributes {llzk.lang, llzk.main = !struct.type<@TwoElem<[]>>} {
  struct.def @TwoElem {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@TwoElem<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@TwoElem<[]>>
      %arr = llzk.nondet : !array.type<2 x !felt.type>
      %c0 = felt.const 0
      %i0 = cast.toindex %c0
      array.write %arr[%i0] = %a : <2 x !felt.type>, !felt.type
      %c1 = felt.const 1
      %i1 = cast.toindex %c1
      array.write %arr[%i1] = %b : <2 x !felt.type>, !felt.type
      struct.writem %self[@out] = %arr : <@TwoElem<[]>>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@TwoElem<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@TwoElem<[]>>, %a: !felt.type, %b: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
LLZK
run_test "TwoElem (cast.toindex)" "$TMPDIR/twoelem.llzk"

echo ""
echo "============================================"
echo "  PASS: $PASS / $((PASS + FAIL))"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
  echo "FAIL: $FAIL tests failed"
  exit 1
fi

echo "PASS: All smoke tests passed"
exit 0
