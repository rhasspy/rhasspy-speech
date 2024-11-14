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
#include <ngram/ngram-model.h>

DEFINE_string(method, "counts",
              "One of: \"counts\", \"histograms\", \"count_of_counts\", "
              "\"count_of_histograms\"");
DEFINE_int64(order, 3, "Set maximal order of ngrams to be counted");

// For counting:
DEFINE_bool(round_to_int, false, "Round all counts to integers");
DEFINE_bool(output_fst, true, "Output counts as fst (otherwise strings)");
DEFINE_bool(require_symbols, true, "Require symbol tables? (default: yes)");
DEFINE_double(
    add_to_symbol_unigram_count, 0.0,
    "Adds this amount to the unigram count of each word in the symbol table");

// For counting and histograms:
DEFINE_bool(epsilon_as_backoff, false,
            "Treat epsilon in the input Fsts as backoff");

// For count-of-counting:
DEFINE_string(context_pattern, "", "Pattern of contexts to count");

// For merging:
DEFINE_double(alpha, 1.0, "Weight for first FST");
DEFINE_double(beta, 1.0, "Weight for second (and subsequent) FST(s)");
DEFINE_bool(normalize, false, "Normalize resulting model");
DEFINE_int64(backoff_label, 0, "Backoff label");
DEFINE_double(norm_eps, ngram::kNormEps, "Normalization check epsilon");
DEFINE_bool(check_consistency, false, "Check model consistency");

int ngramcount_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramcount_main(argc, argv);
}
