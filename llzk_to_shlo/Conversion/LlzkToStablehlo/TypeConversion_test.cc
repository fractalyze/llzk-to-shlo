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

#include <string>

#include "gtest/gtest.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::llzk_to_shlo {
namespace {

ArrayAttr makeInitializedRecords(Builder &builder) {
  return builder.getArrayAttr(
      {builder.getStringAttr("idx_0"), builder.getStringAttr("idx_1")});
}

OwningOpRef<Operation *> makePodNewWithAttr(MLIRContext &context) {
  Builder builder(&context);
  OperationState state(UnknownLoc::get(&context), "pod.new");
  state.addAttribute("initializedRecords", makeInitializedRecords(builder));
  return OwningOpRef<Operation *>(Operation::create(state));
}

TEST(TypeConversionTest, ReadsInitializedRecordsFromAttrs) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  auto podNew = makePodNewWithAttr(context);
  ASSERT_TRUE(podNew);

  auto initAttr = getPodInitializedRecordsAttr(*podNew);
  ASSERT_TRUE(initAttr);
  ASSERT_EQ(initAttr.size(), 2u);

  auto names = getPodInitializedRecords(*podNew);
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "idx_0");
  EXPECT_EQ(names[1], "idx_1");
}

} // namespace
} // namespace mlir::llzk_to_shlo
