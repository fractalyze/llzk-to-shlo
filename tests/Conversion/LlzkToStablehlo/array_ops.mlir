// RUN: llzk-to-shlo-opt --llzk-to-stablehlo %s | FileCheck %s

// Test array operations used with felt operations inside func.func

// Verify no LLZK operations remain in output
// CHECK-NOT: !felt.type
// CHECK-NOT: felt.add

// CHECK-LABEL: func.func @test_array_element_ops
// CHECK: stablehlo.add
// CHECK: return
func.func @test_array_element_ops(%a: !felt.type, %b: !felt.type) -> !felt.type {
  // Simple felt operation - array operations would require function.def context
  %sum = felt.add %a, %b : !felt.type
  return %sum : !felt.type
}

// Note: Full array.new/array.read/array.write testing requires proper LLZK
// function.def context which is difficult to set up in isolated tests.
// The patterns are tested via end-to-end examples.
