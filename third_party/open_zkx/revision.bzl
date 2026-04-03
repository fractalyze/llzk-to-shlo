# Copyright 2026 The llzk-to-shlo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""open-zkx revision pin."""

# To update open-zkx to a new revision,
# a) update OPEN_ZKX_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -sL https://github.com/fractalyze/open-zkx/archive/<commit>.tar.gz | sha256sum
#    and update OPEN_ZKX_SHA256 with the result.

OPEN_ZKX_COMMIT = "d48e7ed8d426c56b6a2406e751caf934ebd2576f"
OPEN_ZKX_SHA256 = "4e4e444401a1240c5115993e834d451e9e2805a0bf2a320011987ae7eee83438"
