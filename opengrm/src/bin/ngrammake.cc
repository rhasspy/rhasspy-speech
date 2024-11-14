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

DEFINE_double(witten_bell_k, 1, "Witten-Bell hyperparameter K");
DEFINE_double(discount_D, -1, "Absolute discount value D to use");
DEFINE_string(method, "katz",
              "One of: \"absolute\", \"katz\", \"kneser_ney\", "
              "\"presmoothed\", \"unsmoothed\", \"katz_frac\", "
              "\"witten_bell\"");
DEFINE_bool(backoff, false,
            "Use backoff smoothing (default: method dependent)");
DEFINE_bool(interpolate, false,
            "Use interpolated smoothing (default: method dependent)");
DEFINE_int64(bins, -1, "Number of bins for katz or absolute discounting");
DEFINE_int64(backoff_label, 0, "Backoff label");
DEFINE_double(norm_eps, ngram::kNormEps, "Normalization check epsilon");
DEFINE_bool(check_consistency, false, "Check model consistency");
DEFINE_string(count_of_counts, "", "Read count-of-counts from file");

int ngrammake_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngrammake_main(argc, argv);
}
