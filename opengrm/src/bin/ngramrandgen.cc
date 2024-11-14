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
#include <unistd.h>

#include <climits>
#include <ctime>

#include <fst/flags.h>

DEFINE_int32(max_length, INT_MAX, "Maximum sentence length");
DEFINE_int64(max_sents, 1, "Maximum number of sentences to produce");
DEFINE_int32(seed, time(nullptr) + getpid(), "Random seed");
DEFINE_bool(remove_epsilon, false, "Remove epsilons from generated strings");
DEFINE_bool(weighted, false,
            "Output tree weighted by sentence count vs. unweighted sentences");
DEFINE_bool(remove_total_weight, false,
            "Remove total weight when output weighted");

int ngramrandgen_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramrandgen_main(argc, argv);
}
