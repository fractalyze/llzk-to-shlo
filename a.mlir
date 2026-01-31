func.func @test(%a: !felt.type, %b: !felt.type) -> !felt.type {
  %sum = "felt.add"(%a, %b) : (!felt.type, !felt.type) -> !felt.type
  return %sum : !felt.type
}
