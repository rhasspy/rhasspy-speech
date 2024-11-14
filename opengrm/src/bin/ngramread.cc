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

DEFINE_bool(ARPA, false, "Read model from ARPA format");
DEFINE_bool(renormalize_arpa, false,
            "If true, attempts to renormalize an unnormalized ARPA format "
            "model by normalizing the unigram state and recomputing the "
            "backoff weights.  Only used if --ARPA=true.");
DEFINE_string(symbols, "", "Label symbol table");
DEFINE_string(epsilon_symbol, "<epsilon>", "Label for epsilon transitions");
DEFINE_string(OOV_symbol, "<UNK>", "Class label for OOV symbols");
DECLARE_string(start_symbol);  // defined in ngram-output.cc
DECLARE_string(end_symbol);    // defined in ngram-output.cc

int ngramread_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramread_main(argc, argv);
}
