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
// Generates a context set of a given size from an input LM.

#include <memory>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/arc.h>
#include <fst/fst.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-model.h>

DECLARE_int64(contexts);
DECLARE_double(bigram_threshold);

int ngramcontext_main(int argc, char **argv) {
  std::string usage = "Generates a context set from an input LM.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] [in.fst] [out.fst]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (argc < 2 || argc > 3) {
    ShowUsage();
    return 1;
  }

  std::string in_name = argv[1];
  std::string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<fst::StdFst> in_fst(fst::StdFst::Read(in_name));
  if (!in_fst) return 1;

  ngram::NGramModel<fst::StdArc> ngram(*in_fst, 0, ngram::kNormEps, true);
  std::vector<std::string> contexts;
  ngram::NGramContext::FindContexts(ngram, FST_FLAGS_contexts,
                                    &contexts,
                                    FST_FLAGS_bigram_threshold);
  bool ret = ngram::NGramWriteContexts(out_name, contexts);

  return !ret;
}
