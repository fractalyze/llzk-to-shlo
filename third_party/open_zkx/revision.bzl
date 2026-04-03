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

OPEN_ZKX_COMMIT = "31ef3005f6785a5c99fd28049a3ef1549a596bf9"
OPEN_ZKX_SHA256 = "d0bf8852c750d1f3789adecf6d4afe340eb0beb97117002195ab6643fe0f9a9d"
