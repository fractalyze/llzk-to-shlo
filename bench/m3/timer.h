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

#ifndef BENCH_M3_TIMER_H_
#define BENCH_M3_TIMER_H_

#include <algorithm>
#include <chrono>
#include <vector>

namespace llzk_to_shlo::bench_m3 {

class StageTimer {
public:
  StageTimer() : start_(std::chrono::steady_clock::now()) {}

  double ElapsedMs() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }

private:
  std::chrono::steady_clock::time_point start_;
};

// Median is preferred over mean: RTX 5090 ZKX JIT autotune occasionally
// re-tunes mid-run, producing one-off outliers that would skew a mean.
inline double Median(std::vector<double> samples) {
  if (samples.empty())
    return 0.0;
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  if (n % 2 == 1)
    return samples[n / 2];
  return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

} // namespace llzk_to_shlo::bench_m3

#endif // BENCH_M3_TIMER_H_
