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

OPEN_ZKX_COMMIT = "1fbb59485cc695960ebdaed811447da076ad8bdd"
OPEN_ZKX_SHA256 = "859b80dbead8350d10fe5acb7588a778d44bb391ee1c7c4382c6cc2590addbf3"
