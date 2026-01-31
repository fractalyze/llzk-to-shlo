#!/bin/bash
# Copyright 2026 The llzk-to-shlo Authors.
# Licensed under the Apache License, Version 2.0.

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

LIT=$(rlocation "llvm-project/llvm/utils/lit/lit.py")
TEST_DIR=$(rlocation "llzk_to_shlo/tests/lit.site.cfg.py")
TEST_DIR=$(dirname "$TEST_DIR")
TOOL_DIR=$(rlocation "llzk_to_shlo/tools/llzk-to-shlo-opt")
TOOL_DIR=$(dirname "$TOOL_DIR")
FILECHECK=$(rlocation "llvm-project/llvm/FileCheck")
FILECHECK_DIR=$(dirname "$FILECHECK")

export PATH="${TOOL_DIR}:${FILECHECK_DIR}:${PATH}"

exec python3 "$LIT" "$TEST_DIR" -v "$@"
