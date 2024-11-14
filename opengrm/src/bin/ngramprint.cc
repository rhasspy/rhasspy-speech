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

DEFINE_bool(ARPA, false, "Print in ARPA format");
DEFINE_bool(backoff, false, "Show epsilon backoff transitions when printing");
DEFINE_bool(backoff_inline, false,
            "Show epsilon backoffs transitions inline with context as a third "
            "field if --backoff are being printed");
DEFINE_bool(negativelogs, false,
            "Show negative log probs/counts when printing");
DEFINE_bool(integers, false, "Show just integer counts when printing");
DEFINE_int64(backoff_label, 0, "Backoff label");
DEFINE_bool(check_consistency, false, "Check model consistency");
DEFINE_string(context_pattern, "", "Pattern of contexts to print");
DEFINE_bool(include_all_suffixes, false, "Include suffixes of contexts");
DEFINE_string(symbols, "",
              "Symbol table file. If not empty, causes it to be loaded from the"
              " specified file instead of using the one inside the input FST.");

int ngramprint_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramprint_main(argc, argv);
}
