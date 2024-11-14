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
#include <fst/weight.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_string(ifile, "", "input file name");
DEFINE_string(syms, "", "input file symbols for compiling");
DEFINE_string(cfile, "", "file to compare input file to");
DEFINE_string(ofile, "", "file for output");

int ngramhisttest_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramhisttest_main(argc, argv);
}
