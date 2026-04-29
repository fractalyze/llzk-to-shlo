// RUN: llzk-to-shlo-opt --llzk-to-stablehlo="prime=2013265921:i32" %s | FileCheck %s

// `registerStructFieldOffsets` skips struct.member declarations that no
// `struct.writem` in the enclosing struct.def targets. This brings layout
// policy into line with `eliminateInputPods` /
// `eraseStructWritemForPodValues` (SimplifySubComponents.cpp), which erase
// writems for sub-component bookkeeping (`*$inputs`, single-struct/pod
// drains) but leave the matching `struct.member` declarations alive so
// `@constrain`-side `struct.readm` references stay valid. `@constrain` is
// dropped before partial conversion, so an unwritten field has no surviving
// op that needs its offset; reserving it would just inflate the witness
// tensor with permanently-zero cells.
//
// This test pins the invariant directly: @keep_a / @keep_arr / @keep_z have
// writems, @drop_inputs / @drop_solo do not. The flattened struct layout
// must contain only the keep'd members (1 + 3 + 1 = 5 cells), and DUS
// offsets must be 0 / 1 / 4 — proving the dropped members did not advance
// the running offset.

// CHECK-LABEL: func.func @main
// CHECK: dense<0> : tensor<5xi32>
// writem @keep_a → DUS at offset 0
// CHECK: %[[OFF_A:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[OFF_A]]
// writem @keep_arr → DUS at offset 1
// CHECK: %[[OFF_ARR:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[OFF_ARR]]
// writem @keep_z → DUS at offset 4 (NOT 5/6; @drop_inputs and @drop_solo
// must contribute 0 cells each)
// CHECK: %[[OFF_Z:.*]] = stablehlo.constant dense<4> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %{{.*}}, %[[OFF_Z]]
// CHECK: return

module attributes {llzk.lang, llzk.main = !struct.type<@LayoutSkip<[]>>} {
  struct.def @Drainee {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@Drainee<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Drainee<[]>>
      function.return %self : !struct.type<@Drainee<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Drainee<[]>>) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @LayoutSkip {
    struct.member @keep_a : !felt.type {llzk.pub}
    struct.member @keep_arr : !array.type<3 x !felt.type> {llzk.pub}
    // No writem targets these — they mimic post-simplify don't-care members
    // (`*$inputs`, single-struct drains).
    struct.member @drop_inputs : !pod.type<[@in: !felt.type]>
    struct.member @drop_solo : !struct.type<@Drainee<[]>>
    struct.member @keep_z : !felt.type {llzk.pub}
    function.def @compute(%x: !felt.type, %v: !array.type<3 x !felt.type>, %z: !felt.type) -> !struct.type<@LayoutSkip<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@LayoutSkip<[]>>
      struct.writem %self[@keep_a] = %x : <@LayoutSkip<[]>>, !felt.type
      struct.writem %self[@keep_arr] = %v : <@LayoutSkip<[]>>, !array.type<3 x !felt.type>
      struct.writem %self[@keep_z] = %z : <@LayoutSkip<[]>>, !felt.type
      function.return %self : !struct.type<@LayoutSkip<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@LayoutSkip<[]>>, %arg1: !felt.type, %arg2: !array.type<3 x !felt.type>, %arg3: !felt.type) attributes {function.allow_constraint} {
      function.return
    }
  }
}
