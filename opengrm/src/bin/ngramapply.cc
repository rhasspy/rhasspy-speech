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

DEFINE_string(bo_arc_type, "phi",
              "One of: \"phi\" (default), \"epsilon\", \"lexicographic\"");

int ngramapply_main(int argc, char** argv);
int main(int argc, char** argv) {
  return ngramapply_main(argc, argv);
}
