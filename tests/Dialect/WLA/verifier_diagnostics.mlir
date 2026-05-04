// RUN: not llzk-to-shlo-opt %s -split-input-file -verify-diagnostics 2>&1 \
// RUN:   | FileCheck %s

// Per-entry sanity in `wla.layout`'s op verifier: each `#wla.signal`
// must have positive length and non-negative offset. Cross-entry
// invariants (no overlap, sorted, single op per module) belong to the
// emit/verify passes and are intentionally NOT enforced here.

module {
  // CHECK: error: 'wla.layout' op signal `@out` has non-positive length 0
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 0>
  ]
}

// -----

module {
  // CHECK: error: 'wla.layout' op signal `@arg0` has negative offset -3
  wla.layout signals = [
    #wla.signal<"@arg0", input, offset = -3, length = 8>
  ]
}

// -----

func.func @not_at_module_scope() {
  // CHECK: error: 'wla.layout' op expects parent op 'builtin.module'
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 1>
  ]
  return
}
