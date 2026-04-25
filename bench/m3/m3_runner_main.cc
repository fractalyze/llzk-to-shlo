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

// M3 measurement-harness runner. Forked from
// @open_zkx//zkx/tools/stablehlo_runner:stablehlo_runner_main.cc; the upstream
// binary is the right place to land any non-measurement changes (input parsing,
// HLO dumping, etc.) — this file only owns per-stage timing and CSV emission.
//
// Usage (typically invoked via bench/m3/run.sh):
//   bazel run //bench/m3:m3_runner -- \
//     --circuit=MontgomeryDouble --N=1024 --iterations=3 --warmups=2 \
//     --csv_out=/path/to/results.csv --append \
//     --use_random_inputs \
//     /path/to/module.mlir

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "bench/m3/csv_schema.h"
#include "bench/m3/timer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"
#include "zkx/zkx_data.pb.h"

namespace llzk_to_shlo::bench_m3 {
namespace {

struct Options {
  std::string circuit;
  int32_t n = 1;
  int32_t iterations = 3;
  int32_t warmups = 2;
  std::string csv_out;
  bool append = false;
  bool use_random_inputs = true;
};

absl::StatusOr<std::vector<zkx::Literal>>
CreateInputLiterals(const zkx::HloModule &module, bool use_random_inputs) {
  std::vector<zkx::Literal> literals;
  const auto *entry = module.entry_computation();
  literals.reserve(entry->num_parameters());
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const auto &shape = entry->parameter_instruction(i)->shape();
    TF_ASSIGN_OR_RETURN(zkx::Literal literal,
                        zkx::MakeFakeLiteral(shape, use_random_inputs));
    literals.push_back(std::move(literal));
  }
  return literals;
}

// Throughput is only meaningful for the `total` stage; pass throughput < 0 to
// emit a blank in the CSV column for non-total rows.
void EmitRow(std::ostream &out, const std::string &circuit, const char *backend,
             int n, const char *stage, double time_ms,
             double throughput_wits_per_sec = -1.0) {
  out << circuit << ',' << backend << ',' << n << ',' << stage << ','
      << absl::StrFormat("%.6f", time_ms) << ',';
  if (throughput_wits_per_sec >= 0.0) {
    out << absl::StrFormat("%.3f", throughput_wits_per_sec);
  }
  out << '\n';
}

absl::Status RunHarness(const Options &options, const char *module_path) {
  std::string module_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), module_path, &module_text));
  // MLIRContext registers dialects on construction — keep that out of the
  // compile timer so the reported number is the actual lowering work.
  mlir::MLIRContext context;

  StageTimer compile_timer;
  TF_ASSIGN_OR_RETURN(auto stablehlo_module,
                      zkx::ParseStablehloModule(module_text, &context));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<zkx::HloModule> hlo_module,
                      zkx::ConvertStablehloToHloModule(*stablehlo_module));
  double compile_ms = compile_timer.ElapsedMs();

  TF_ASSIGN_OR_RETURN(auto platform, zkx::PlatformUtil::GetPlatform("gpu"));
  zkx::HloRunner runner(platform);

  TF_ASSIGN_OR_RETURN(
      std::vector<zkx::Literal> literals,
      CreateInputLiterals(*hlo_module, options.use_random_inputs));
  std::vector<const zkx::Literal *> literal_ptrs;
  literal_ptrs.reserve(literals.size());
  for (const zkx::Literal &literal : literals) {
    literal_ptrs.push_back(&literal);
  }

  StageTimer jit_timer;
  TF_ASSIGN_OR_RETURN(
      auto executable,
      runner.CreateExecutable(std::move(hlo_module), /*run_hlo_passes=*/true));
  double jit_ms = jit_timer.ElapsedMs();

  for (int32_t i = 0; i < options.warmups; ++i) {
    TF_RETURN_IF_ERROR(runner
                           .ExecuteWithExecutable(executable.get(),
                                                  literal_ptrs,
                                                  /*profile=*/nullptr)
                           .status());
  }

  std::vector<double> kernel_ms;
  std::vector<double> d2h_ms;
  std::vector<double> total_ms;
  kernel_ms.reserve(options.iterations);
  d2h_ms.reserve(options.iterations);
  total_ms.reserve(options.iterations);

  for (int32_t i = 0; i < options.iterations; ++i) {
    zkx::ExecutionProfile profile;
    StageTimer total_timer;
    TF_RETURN_IF_ERROR(
        runner.ExecuteWithExecutable(executable.get(), literal_ptrs, &profile)
            .status());
    double t_total_ms = total_timer.ElapsedMs();

    double t_kernel_ms = profile.compute_time_ns() / 1e6;
    double t_compute_and_xfer_ms = profile.compute_and_transfer_time_ns() / 1e6;
    double t_d2h_ms = std::max(0.0, t_compute_and_xfer_ms - t_kernel_ms);

    kernel_ms.push_back(t_kernel_ms);
    d2h_ms.push_back(t_d2h_ms);
    total_ms.push_back(t_total_ms);
  }

  double med_kernel = Median(kernel_ms);
  double med_d2h = Median(d2h_ms);
  double med_total = Median(total_ms);
  double throughput =
      (med_total > 0.0) ? (static_cast<double>(options.n) * 1000.0 / med_total)
                        : 0.0;

  std::ofstream ofs;
  std::ostream *out = &std::cout;
  if (!options.csv_out.empty()) {
    bool need_header = !options.append;
    if (options.append) {
      std::ifstream test(options.csv_out);
      need_header = !test.good();
    }
    ofs.open(options.csv_out, options.append ? std::ios::app : std::ios::out);
    if (!ofs.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open csv_out: ", options.csv_out));
    }
    if (need_header) {
      ofs << kCsvHeader << '\n';
    }
    out = &ofs;
  } else {
    *out << kCsvHeader << '\n';
  }

  const char *backend = kBackendGpuZkx;
  EmitRow(*out, options.circuit, backend, options.n, kStageCompile, compile_ms);
  EmitRow(*out, options.circuit, backend, options.n, kStageJit, jit_ms);
  EmitRow(*out, options.circuit, backend, options.n, kStageKernel, med_kernel);
  EmitRow(*out, options.circuit, backend, options.n, kStageD2H, med_d2h);
  EmitRow(*out, options.circuit, backend, options.n, kStageTotal, med_total,
          throughput);

  LOG(INFO) << "m3_runner: " << options.circuit << " N=" << options.n
            << " compile=" << compile_ms << "ms jit=" << jit_ms
            << "ms kernel(med)=" << med_kernel << "ms d2h(med)=" << med_d2h
            << "ms total(med)=" << med_total << "ms throughput=" << throughput
            << " wits/s";
  return absl::OkStatus();
}

} // namespace
} // namespace llzk_to_shlo::bench_m3

int main(int argc, char **argv) {
  llzk_to_shlo::bench_m3::Options options;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("circuit", &options.circuit,
                "Circuit name (free-form label written to CSV)."),
      tsl::Flag("N", &options.n,
                "Batch size N (free-form label written to CSV; must match "
                "the leading dim of the input MLIR)."),
      tsl::Flag("iterations", &options.iterations,
                "Number of post-warmup execution iterations to median over."),
      tsl::Flag("warmups", &options.warmups,
                "Untimed warmup iterations before measurement."),
      tsl::Flag("csv_out", &options.csv_out,
                "Path to write CSV output. If empty, writes to stdout."),
      tsl::Flag("append", &options.append,
                "Append to csv_out instead of overwriting (header is "
                "auto-suppressed when file already exists)."),
      tsl::Flag("use_random_inputs", &options.use_random_inputs,
                "Populate inputs with random data (otherwise zeros)."),
  };

  zkx::AppendDebugOptionsFlags(&flag_list);

  std::string usage = absl::StrCat(
      "M3 per-stage measurement runner.\n\nUsage:\n  ", argv[0],
      " [flags] <module.mlir>\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || argc < 2) {
    LOG(ERROR) << usage;
    return 1;
  }
  if (options.circuit.empty()) {
    LOG(ERROR) << "--circuit is required.\n" << usage;
    return 1;
  }
  if (options.iterations < 1) {
    LOG(ERROR) << "--iterations must be >= 1.\n";
    return 1;
  }
  if (options.warmups < 0) {
    LOG(ERROR) << "--warmups must be >= 0.\n";
    return 1;
  }

  absl::Status s = llzk_to_shlo::bench_m3::RunHarness(options, argv[1]);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return 1;
  }
  return 0;
}
