// RUN: not llzk-to-shlo-opt --verify-witness-layout %s -o - 2>&1 \
// RUN:   | FileCheck %s

// `wla.layout` is present, but `@main` is multi-block. Witness-output
// functions are single-block by construction (StableHLO control flow
// is region-based, not block-based) — `collectChunks` should reject
// the function with a precise diagnostic rather than report a
// misleading operand count from the entry block's branch terminator.

// CHECK: error: 'func.func' op witness-output `func.return` is malformed: error: function @main must consist of exactly one block
module {
  wla.layout signals = [
    #wla.signal<"@out", output, offset = 0, length = 1>
  ]

  func.func @main() -> tensor<1xi32> {
    %v = stablehlo.constant dense<7> : tensor<1xi32>
    cf.br ^bb1
  ^bb1:
    return %v : tensor<1xi32>
  }
}
