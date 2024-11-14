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
// Sorts an ngram LM in lexicographic state context order.

#include <cstring>
#include <memory>
#include <string>

#include <fst/flags.h>
#include <fst/arc.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-mutable-model.h>

DECLARE_bool(check_consistency);
DECLARE_int64(backoff_label);
DECLARE_double(norm_eps);

int ngramsort_main(int argc, char **argv) {
  std::string usage =
      "Sorts an ngram LM in lexicographic state context order.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] [in.fst [out.fst]]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  std::string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  std::string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<fst::StdMutableFst> fst(
      fst::StdMutableFst::Read(in_name, true));
  if (!fst) return 1;

  ngram::NGramMutableModel<fst::StdArc> ngramlm(
      fst.get(), FST_FLAGS_backoff_label,
      FST_FLAGS_norm_eps,
      /* state_ngrams= */ true, /* infinite_backoff= */ false);
  ngramlm.SortStates();
  ngramlm.InitModel();
  ngramlm.GetFst().Write(out_name);

  return 0;
}
