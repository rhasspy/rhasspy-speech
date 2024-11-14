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

#include <ctime>

#include <fst/flags.h>

DEFINE_int32(max_length, 1000, "Maximum sentence length");
DEFINE_int32(seed, time(0) + getpid(), "Randomization seed");
DEFINE_int32(vocabulary_max, 5000, "maximum vocabulary size");
DEFINE_int32(mean_length, 100, "maximum mean string length");
DEFINE_int32(sample_max, 10000, "maximum sample corpus size");
DEFINE_int32(ngram_max, 3, "maximum n-gram order size");
DEFINE_string(directory, ".", "directory where files will be placed");
DEFINE_string(vars, "", "file name for outputting variable values");
DEFINE_double(thresh_max, 3, "maximum threshold size");

int ngramrandtest_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramrandtest_main(argc, argv);
}
