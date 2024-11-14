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

DEFINE_bool(use_phimatcher, false, "Use phi matcher and composition");
DEFINE_string(OOV_symbol, "", "Existing symbol for OOV class");
DEFINE_double(OOV_class_size, 10000, "Number of members of OOV class");
DEFINE_double(OOV_probability, 0, "Unigram probability for OOVs");
DEFINE_string(context_pattern, "",
              "Restrict perplexity computation to contexts defined by"
              " pattern (default: no restriction)");

int ngramperplexity_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramperplexity_main(argc, argv);
}
