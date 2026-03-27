// RUN: llzk-to-shlo-opt --llzk-to-stablehlo %s | FileCheck %s

// Verify no LLZK felt operations remain in output
// CHECK-NOT: !felt.type
// CHECK-NOT: felt.const
// CHECK-NOT: felt.add
// CHECK-NOT: felt.sub
// CHECK-NOT: felt.mul
// CHECK-NOT: felt.div
// CHECK-NOT: felt.neg

// Test felt.const conversion
// CHECK-LABEL: func.func @test_felt_const
// CHECK: stablehlo.constant
// CHECK-SAME: dense<42>
func.func @test_felt_const() -> !felt.type {
  %c = felt.const 42
  return %c : !felt.type
}

// Test felt.add conversion
// CHECK-LABEL: func.func @test_felt_add
// CHECK: stablehlo.add %{{.*}}, %{{.*}} : tensor<
func.func @test_felt_add(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %sum = felt.add %a, %b : !felt.type
  return %sum : !felt.type
}

// Test felt.sub conversion
// CHECK-LABEL: func.func @test_felt_sub
// CHECK: stablehlo.subtract %{{.*}}, %{{.*}} : tensor<
func.func @test_felt_sub(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %diff = felt.sub %a, %b : !felt.type
  return %diff : !felt.type
}

// Test felt.mul conversion
// CHECK-LABEL: func.func @test_felt_mul
// CHECK: stablehlo.multiply %{{.*}}, %{{.*}} : tensor<
func.func @test_felt_mul(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %prod = felt.mul %a, %b : !felt.type
  return %prod : !felt.type
}

// Test felt.div conversion
// CHECK-LABEL: func.func @test_felt_div
// CHECK: stablehlo.divide %{{.*}}, %{{.*}} : tensor<
func.func @test_felt_div(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %quot = felt.div %a, %b : !felt.type
  return %quot : !felt.type
}

// Test felt.neg conversion
// CHECK-LABEL: func.func @test_felt_neg
// CHECK: stablehlo.negate %{{.*}} : tensor<
func.func @test_felt_neg(%a: !felt.type) -> !felt.type {
  %neg = felt.neg %a : !felt.type
  return %neg : !felt.type
}

// Note: felt.pow and felt.shr/bit_and require function.def context and are
// tested via full circuit tests (sub_component.mlir, integration.mlir).

// Test combined operations
// CHECK-LABEL: func.func @test_compute
// CHECK: stablehlo.add
// CHECK: stablehlo.multiply
func.func @test_compute(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %sum = felt.add %a, %b : !felt.type
  %result = felt.mul %sum, %a : !felt.type
  return %result : !felt.type
}
