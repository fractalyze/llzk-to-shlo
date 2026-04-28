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

#ifndef BENCH_M3_JSON_INPUT_H_
#define BENCH_M3_JSON_INPUT_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/shape.h"

namespace llzk_to_shlo::bench_m3 {

// Builds a Literal of `shape` from a flat list of decimal-string values.
// Mirrors @open_zkx//zkx/tools/stablehlo_runner:ParseInputLiteral so a single
// JSON fixture can drive both gpu_zkx (this runner) and cpu_circom (circom
// witness binary). Token count must either equal `ShapeUtil::ElementsIn(shape)`
// or describe one witness whose contents are tiled across the leading batch
// dim added by --batch-stablehlo (token_count * shape.dimensions(0) ==
// num_elements). Per-element parsing routes to `NativeTypeFromDecString`,
// which dispatches prime field types to `FromDecString` and integer types
// to `SimpleAtoi`.
absl::StatusOr<zkx::Literal>
LiteralFromDecStrings(const zkx::Shape &shape,
                      const std::vector<std::string> &tokens);

// Parses a circom-style fixture (`{<signal>: scalar | flat-array}`) and maps
// each top-level key to one HLO parameter, IN JSON INSERTION ORDER. Signal
// names are not preserved through MLIR lowering (parameters surface as
// `%arg0`, `%arg1`, ...) so the contract is positional: 1st JSON key → param
// 0, 2nd → param 1, etc. The same fixture is fed unchanged to the circom
// witness binary in `run_baseline.sh`, which DOES use the names — shared
// schema is what makes the future per-witness comparison possible.
//
// Errors:
//   - NotFound        if json_path cannot be opened.
//   - InvalidArgument on JSON parse failure, root-not-object, key/param count
//     mismatch, nested array, non-numeric / non-string element, or element
//     count mismatch with the parameter shape.
absl::StatusOr<std::vector<zkx::Literal>>
ParseInputLiteralsFromJson(const zkx::HloModule &module,
                           const std::string &json_path);

} // namespace llzk_to_shlo::bench_m3

#endif // BENCH_M3_JSON_INPUT_H_
