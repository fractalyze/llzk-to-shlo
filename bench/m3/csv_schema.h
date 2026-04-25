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

// Single source of truth for the M3 measurement-harness CSV format.
// run_baseline.sh extracts these strings via awk so the shell baseline
// stays in lock-step.

#ifndef LLZK_TO_SHLO_BENCH_M3_CSV_SCHEMA_H_
#define LLZK_TO_SHLO_BENCH_M3_CSV_SCHEMA_H_

namespace llzk_to_shlo::bench_m3 {

// Keep value on the same line as the name so run_baseline.sh can extract it
// with a simple awk -F'"'.
inline constexpr const char *kCsvHeader =
    "circuit,backend,N,stage,time_ms,throughput_wits_per_sec";

inline constexpr const char *kStageCompile = "compile";
inline constexpr const char *kStageJit = "jit";
inline constexpr const char *kStageKernel = "kernel";
inline constexpr const char *kStageD2H = "d2h";
inline constexpr const char *kStageTotal = "total";

inline constexpr const char *kBackendGpuZkx = "gpu_zkx";
inline constexpr const char *kBackendCpuCircom = "cpu_circom";

} // namespace llzk_to_shlo::bench_m3

#endif // LLZK_TO_SHLO_BENCH_M3_CSV_SCHEMA_H_
