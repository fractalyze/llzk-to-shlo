/* Copyright 2026 The llzk-to-shlo Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gtest/gtest.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::llzk_to_shlo {
namespace {

llzk::pod::NewPodOp makePodNewWithAttr(MLIRContext &context) {
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<llzk::pod::PODDialect>();
  auto loc = UnknownLoc::get(&context);
  OpBuilder builder(&context);

  auto zero =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  auto one =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));

  llzk::pod::RecordValue records[] = {{"idx_0", zero}, {"idx_1", one}};
  OperationState state(loc, llzk::pod::NewPodOp::getOperationName());
  llzk::pod::NewPodOp::build(builder, state,
                             llzk::pod::InitializedRecords(records));
  return cast<llzk::pod::NewPodOp>(Operation::create(state));
}

TEST(TypeConversionTest, ReadsInitializedRecordsFromAttrs) {
  MLIRContext context;
  auto podNew = makePodNewWithAttr(context);

  auto initAttr = getPodInitializedRecordsAttr(podNew);
  ASSERT_TRUE(initAttr);
  ASSERT_EQ(initAttr.size(), 2u);

  auto names = getPodInitializedRecords(podNew);
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "idx_0");
  EXPECT_EQ(names[1], "idx_1");

  podNew->erase();
}

} // namespace
} // namespace mlir::llzk_to_shlo
