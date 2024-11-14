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

DEFINE_double(alpha, 1.0, "Weight for first FST");
DEFINE_double(beta, 1.0, "Weight for second (and subsequent) FST(s)");
DEFINE_string(context_pattern, "", "Context pattern for second FST");
DEFINE_string(contexts, "", "Context patterns file (all FSTs)");
DEFINE_bool(normalize, false, "Normalize resulting model");
DEFINE_string(method, "count_merge",
              "One of: \"context_merge\", \"count_merge\", \"model_merge\" "
              "\"bayes_model_merge\", \"histogram_merge\", \"replace_merge\"");
DEFINE_int32(max_replace_order, -1,
             "Maximum order to replace in replace_merge, ignored if < 1.");
DEFINE_string(ofile, "", "Output file");
DEFINE_int64(backoff_label, 0, "Backoff label");
DEFINE_double(norm_eps, ngram::kNormEps, "Normalization check epsilon");
DEFINE_bool(check_consistency, false, "Check model consistency");
DEFINE_bool(complete, false, "Complete partial models");
DEFINE_bool(round_to_int, false, "Round all merged counts to integers");

int ngrammerge_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngrammerge_main(argc, argv);
}
