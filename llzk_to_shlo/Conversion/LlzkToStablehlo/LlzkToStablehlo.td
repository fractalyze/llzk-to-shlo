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

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_LLZKTOSTABLEHLO_TD_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_LLZKTOSTABLEHLO_TD_

include "mlir/Pass/PassBase.td"

def LlzkToStablehlo : Pass<"llzk-to-stablehlo", "ModuleOp"> {
  let summary = "Lower LLZK dialect to StableHLO dialect.";
  let description = [{
    This pass lowers the LLZK dialect operations (felt, struct, array) from
    `compute()` functions to their StableHLO equivalents.

    Type conversions:
    - `!felt.type` -> `tensor<!field.pf<prime>>`
    - `!array.type<N x felt>` -> `tensor<N x !field.pf<prime>>`
    - `!struct.type<@Name>` -> `tensor<M x !field.pf<prime>>` (flattened)

    Operation mappings:
    - `felt.add/sub/mul/div/neg` -> `stablehlo.add/subtract/multiply/divide/negate`
    - `felt.inv` -> `stablehlo.divide` (1 / x)
    - `felt.const` -> `stablehlo.constant`
    - `struct.new` -> tensor construction
    - `struct.readm/writem` -> `stablehlo.slice/dynamic_update_slice`
    - `array.read/write` -> `stablehlo.dynamic_slice/dynamic_update_slice`
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::prime_ir::field::FieldDialect",
  ];
  let options = [
    Option<"prime", "prime", "std::string", /*default=*/"\"bn254\"",
           "Prime modulus. Accepts aliases (bn254) or "
           "explicit format <value>:<type> (e.g., 2013265921:i32). "
           "Default: bn254 (Circom's default curve)">,
    Option<"usePrimeFieldType", "use-prime-field-type", "bool",
           /*default=*/"true",
           "Use tensor<!field.pf<prime>> instead of tensor<iN>">,
    Option<"flagOrphanZeroWrites", "flag-orphan-zero-writes", "bool",
           /*default=*/"true",
           "Abort lowering if a `struct.writem`'s value resolves to a "
           "splat-zero `stablehlo.constant` of length >= 8 (likely an "
           "upstream-orphaned wire). Defaults to on after Wave 1 closure "
           "(PR #70 closed the upstream orphan class for "
           "aes_256_{ctr,encrypt,key_expansion}, fpmultiply, "
           "signed_fp_carry_mod_p). Pass `flag-orphan-zero-writes=false` "
           "to opt out for diagnostic spelunking on a known-broken chip. "
           "A future anchor + verify pass pair will replace this with an "
           "exact per-member check.">,
  ];
}

#endif  // LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_LLZKTOSTABLEHLO_TD_
