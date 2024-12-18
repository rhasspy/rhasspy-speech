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

DEFINE_int64(backoff_label, 0, "Backoff label");
DEFINE_int32(max_bo_updates, 10, "Max iterations of backoff re-calculation");
DEFINE_double(norm_eps, ngram::kNormEps, "Normalization check epsilon");
DEFINE_bool(check_consistency, false, "Check model consistency");

int ngrammarginalize_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngrammarginalize_main(argc, argv);
}
