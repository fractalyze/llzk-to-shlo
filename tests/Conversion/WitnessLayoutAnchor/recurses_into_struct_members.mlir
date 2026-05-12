// RUN: llzk-to-shlo-opt --witness-layout-anchor %s | FileCheck %s

// `getMemberFlatSize` must recurse into struct-typed members and
// struct-array elements. The flat size of a nested `!struct.type<@Inner>`
// equals the sum of `@Inner`'s writem-targeted, non-pod members'
// (recursive) flat sizes — matching the offset-map invariant in
// `registerStructFieldOffsets` (LlzkToStablehlo.cpp). Without recursion,
// each struct-typed slot reports length 1 regardless of its real
// witness footprint, so the parent's downstream member offsets shift
// and the witness output tensor allocates too few cells.
//
// Layout shape this test pins:
//   const_one              [internal, length=1, offset=0]
//   @out                   [output,   length=2, offset=1]   (`Outer.@out`)
//   %arg0                  [input,    length=1, offset=3]
//   @inner_scalar          [internal, length=2, offset=4]   (Inner is 2 felts)
//   @inner_array           [internal, length=6, offset=6]   (3 * Inner flat = 6)
//
// Inner exposes two members; both are writem-targeted by Inner's @compute,
// so both contribute. @x being `{llzk.pub}` does not affect the inner's
// flat-size count — the inner filter is "writem-targeted non-pod only",
// consistent with how `registerStructFieldOffsets` walks the inner
// struct.def to register its per-field offsets.

module attributes {llzk.lang, llzk.main = !struct.type<@Outer::@Outer<[]>>} {
  // CHECK-LABEL: module
  // CHECK: wla.layout signals = [
  // CHECK-SAME: #wla.signal<"const_one", internal, offset = 0, length = 1>
  // CHECK-SAME: #wla.signal<"@out", output, offset = 1, length = 2>
  // CHECK-SAME: #wla.signal<"%arg0", input, offset = 3, length = 1>
  // CHECK-SAME: #wla.signal<"@inner_scalar", internal, offset = 4, length = 2>
  // CHECK-SAME: #wla.signal<"@inner_array", internal, offset = 6, length = 6>
  // CHECK-SAME: ]
  module @Inner {
    struct.def @Inner {
      struct.member @x : !felt.type<"bn128"> {llzk.pub}
      struct.member @y : !felt.type<"bn128">
      function.def @compute() -> !struct.type<@Inner::@Inner<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@Inner::@Inner<[]>>
        %vx = llzk.nondet : !felt.type<"bn128">
        %vy = llzk.nondet : !felt.type<"bn128">
        struct.writem %self[@x] = %vx : <@Inner::@Inner<[]>>, !felt.type<"bn128">
        struct.writem %self[@y] = %vy : <@Inner::@Inner<[]>>, !felt.type<"bn128">
        function.return %self : !struct.type<@Inner::@Inner<[]>>
      }
      function.def @constrain(%arg0: !struct.type<@Inner::@Inner<[]>>) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
  module @Outer {
    struct.def @Outer {
      struct.member @out : !array.type<2 x !felt.type<"bn128">> {llzk.pub}
      struct.member @inner_scalar : !struct.type<@Inner::@Inner<[]>>
      struct.member @inner_array : !array.type<3 x !struct.type<@Inner::@Inner<[]>>>
      function.def @compute(%arg0: !felt.type<"bn128">) -> !struct.type<@Outer::@Outer<[]>> attributes {function.allow_witness} {
        %self = struct.new : <@Outer::@Outer<[]>>
        %vout = llzk.nondet : !array.type<2 x !felt.type<"bn128">>
        %vscalar = llzk.nondet : !struct.type<@Inner::@Inner<[]>>
        %varr = llzk.nondet : !array.type<3 x !struct.type<@Inner::@Inner<[]>>>
        struct.writem %self[@out] = %vout
          : <@Outer::@Outer<[]>>, !array.type<2 x !felt.type<"bn128">>
        struct.writem %self[@inner_scalar] = %vscalar
          : <@Outer::@Outer<[]>>, !struct.type<@Inner::@Inner<[]>>
        struct.writem %self[@inner_array] = %varr
          : <@Outer::@Outer<[]>>, !array.type<3 x !struct.type<@Inner::@Inner<[]>>>
        function.return %self : !struct.type<@Outer::@Outer<[]>>
      }
      function.def @constrain(
          %arg0: !struct.type<@Outer::@Outer<[]>>,
          %arg1: !felt.type<"bn128">) attributes {function.allow_constraint} {
        function.return
      }
    }
  }
}
