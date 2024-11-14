// Copyright 2005-2013 Brian Roark
// Copyright 2005-2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the 'License');
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an 'AS IS' BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include <fst/flags.h>

DEFINE_int64(backoff_label, 0, "Backoff label");
DEFINE_string(context_pattern1, "", "Context pattern for first model");
DEFINE_string(context_pattern2, "", "Context pattern for second model");
DEFINE_string(contexts, "", "Context patterns files (all FSTs)");
DEFINE_string(ofile, "", "Output file (prefix)");
DEFINE_string(method, "count_transfer",
              "One of \"count_transfer\", "
              "\"histogram_transfer\"");
DEFINE_int32(index, -1, "Specifies one FST as the destination (source)");
DEFINE_bool(transfer_from, false,
            "Transfer from (to) other FSTS to indexed FST");
DEFINE_bool(normalize, false, "Recompute backoff weights after transfer");
DEFINE_bool(complete, false, "Complete partial models");

int ngramtransfer_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramtransfer_main(argc, argv);
}
