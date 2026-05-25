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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodInvariants.h"

#include "llzk/Dialect/POD/IR/Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Types.h"

namespace mlir::llzk_to_shlo {

void assertNoPodTypedWhileCarry(ModuleOp module) {
#ifndef NDEBUG
  module.walk([](scf::WhileOp w) {
    for (Type t : w->getResultTypes())
      assert(!isa<llzk::pod::PodType>(t) &&
             "pod-typed scf.while result survived SimplifySubComponents; a "
             "while-carry flatten/unpack phase was reordered or skipped "
             "(see docs/load-bearing-invariants.md)");
    for (Region &r : w->getRegions())
      for (BlockArgument a : r.front().getArguments())
        assert(!isa<llzk::pod::PodType>(a.getType()) &&
               "pod-typed scf.while carry survived SimplifySubComponents; a "
               "while-carry flatten/unpack phase was reordered or skipped "
               "(see docs/load-bearing-invariants.md)");
  });
#else
  (void)module;
#endif
}

} // namespace mlir::llzk_to_shlo
