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

OPEN_ZKX_COMMIT = "3540457135a0afff4b2e0646887ad414fc7f79b2"
OPEN_ZKX_SHA256 = "09284665508a2704f930197c18ffa43cd3ac7249a401d27d7030953e224e202f"
