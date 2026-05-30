// RUN: not llzk-to-shlo-opt --simplify-sub-components="test-phase=NoSuchPhase" %s 2>&1 \
// RUN:   | FileCheck %s

// An unknown `test-phase=<name>` must fail loudly. A silent accept would
// let a typo'd lit fixture compare against whatever the pass does by
// default instead of the named phase's documented contract.

// CHECK: error: {{.*}}simplify-sub-components: unknown or unsupported test-phase 'NoSuchPhase'

module attributes {llzk.lang} {
}
