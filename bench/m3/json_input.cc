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

#include "bench/m3/json_input.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/primitive_util.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace llzk_to_shlo::bench_m3 {
namespace {

// Flattens a JSON value (scalar number/string, or 1-D array of those) into a
// vector of decimal strings. Nested arrays and objects are rejected — the
// circom-side fixtures we share with `run_baseline.sh` always emit a flat
// signal: scalar or 1-D array.
absl::StatusOr<std::vector<std::string>>
DecStringsFromJsonValue(const nlohmann::ordered_json &val,
                        std::string_view key) {
  std::vector<std::string> tokens;
  auto append = [&](const nlohmann::ordered_json &elem) -> absl::Status {
    if (elem.is_string()) {
      tokens.push_back(elem.get<std::string>());
    } else if (elem.is_number_integer() || elem.is_number_unsigned()) {
      tokens.push_back(elem.dump());
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input \"", key,
          "\" element must be an integer or decimal string; got JSON type ",
          elem.type_name()));
    }
    return absl::OkStatus();
  };
  if (val.is_array()) {
    tokens.reserve(val.size());
    for (const auto &elem : val) {
      if (elem.is_array() || elem.is_object()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Input \"", key,
                         "\" must be flat (scalar or 1-D array of scalars); "
                         "nested structures are not supported"));
      }
      TF_RETURN_IF_ERROR(append(elem));
    }
  } else if (val.is_object()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input \"", key, "\" must be a scalar or 1-D array, not an object"));
  } else {
    TF_RETURN_IF_ERROR(append(val));
  }
  return tokens;
}

} // namespace

absl::StatusOr<zkx::Literal>
LiteralFromDecStrings(const zkx::Shape &shape,
                      const std::vector<std::string> &tokens) {
  int64_t num_elements = zkx::ShapeUtil::ElementsIn(shape);
  if (static_cast<int64_t>(tokens.size()) != num_elements) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", num_elements, " values for shape ",
                     shape.ToString(), " but got ", tokens.size()));
  }
  zkx::Literal literal(shape);
  zkx::PrimitiveType elem_type = shape.element_type();
  return zkx::primitive_util::PrimitiveTypeSwitch<absl::StatusOr<zkx::Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<zkx::Literal> {
        if constexpr (primitive_type_constant == zkx::PRIMITIVE_TYPE_INVALID ||
                      !zkx::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Unsupported element type: ", PrimitiveType_Name(elem_type)));
        } else {
          using NativeT =
              zkx::primitive_util::NativeTypeOf<primitive_type_constant>;
          auto data = literal.data<NativeT>();
          for (int64_t i = 0; i < num_elements; ++i) {
            TF_ASSIGN_OR_RETURN(
                data[i], zkx::primitive_util::NativeTypeFromDecString<NativeT>(
                             tokens[i]));
          }
          return std::move(literal);
        }
      },
      elem_type);
}

absl::StatusOr<std::vector<zkx::Literal>>
ParseInputLiteralsFromJson(const zkx::HloModule &module,
                           const std::string &json_path) {
  std::ifstream ifs(json_path);
  if (!ifs.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Cannot open --input_json file: ", json_path));
  }
  nlohmann::ordered_json j;
  try {
    ifs >> j;
  } catch (const nlohmann::json::parse_error &e) {
    return absl::InvalidArgumentError(
        absl::StrCat("JSON parse error in ", json_path, ": ", e.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError(
        absl::StrCat("--input_json root must be an object: ", json_path));
  }
  const auto *entry = module.entry_computation();
  const int64_t num_params = entry->num_parameters();
  if (static_cast<int64_t>(j.size()) != num_params) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Fixture ", json_path, " has ", j.size(),
        " top-level keys but module expects ", num_params, " parameters"));
  }
  std::vector<zkx::Literal> literals;
  literals.reserve(num_params);
  int64_t param_idx = 0;
  for (auto it = j.begin(); it != j.end(); ++it, ++param_idx) {
    const auto &shape = entry->parameter_instruction(param_idx)->shape();
    TF_ASSIGN_OR_RETURN(std::vector<std::string> tokens,
                        DecStringsFromJsonValue(it.value(), it.key()));
    TF_ASSIGN_OR_RETURN(zkx::Literal literal,
                        LiteralFromDecStrings(shape, tokens));
    literals.push_back(std::move(literal));
  }
  return literals;
}

} // namespace llzk_to_shlo::bench_m3
