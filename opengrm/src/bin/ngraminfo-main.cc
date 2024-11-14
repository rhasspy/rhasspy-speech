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
// Prints out various information about n-gram language models.

#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/arc.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

DECLARE_bool(check_consistency);
DECLARE_double(norm_eps);

namespace ngram {

void PrintNGramInfo(const NGramModel<fst::StdArc> &ngram,
                    std::ostream &ostrm) {
  const fst::StdFst &fst = ngram.GetFst();
  std::vector<size_t> order_ngrams(ngram.HiOrder(), 0);
  size_t ngrams = 0;
  size_t backoffs = 0;
  size_t nfinal = 0;
  for (size_t s = 0; s < ngram.NumStates(); ++s) {
    int order = ngram.StateOrder(s);
    if (fst.Final(s) != fst::StdArc::Weight::Zero()) {
      ++nfinal;
      if (order > 0) ++order_ngrams[order - 1];
    }
    for (fst::ArcIterator<fst::StdFst> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      if (arc.ilabel == 0) {
        ++backoffs;
      } else {
        ++ngrams;
        if (order > 0) ++order_ngrams[order - 1];
      }
    }
  }

  std::ios_base::fmtflags old = ostrm.setf(std::ios::left);
  ostrm.width(50);
  ostrm << "# of states" << ngram.NumStates() << "\n";
  ostrm.width(50);
  ostrm << "# of ngram arcs" << ngrams << "\n";
  ostrm.width(50);
  ostrm << "# of backoff arcs" << backoffs << "\n";
  ostrm.width(50);
  ostrm << "initial state" << fst.Start() << "\n";
  ostrm.width(50);
  ostrm << "unigram state" << ngram.UnigramState() << "\n";
  ostrm.width(50);
  ostrm << "# of final states" << nfinal << "\n";

  ostrm.width(50);
  ostrm << "ngram order" << ngram.HiOrder() << "\n";
  for (int order = 1; order <= ngram.HiOrder(); ++order) {
    std::stringstream label;
    label << "# of " << order << "-grams";
    ostrm.width(50);
    ostrm << label.str() << order_ngrams[order - 1] << "\n";
  }
  ostrm.width(50);
  ostrm << "well-formed" << (ngram.CheckTopology() ? 'y' : 'n') << "\n";
  ostrm.width(50);
  ostrm << "normalized" << (ngram.CheckNormalization() ? 'y' : 'n') << "\n";
  ostrm.flush();
  ostrm.setf(old);
}

}  // namespace ngram

int ngraminfo_main(int argc, char **argv) {
  std::string usage =
      "Prints out various information about an LM.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] [in.fst [out.txt]]\n";

  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  std::string ifile = (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";

  std::unique_ptr<fst::StdMutableFst> fst(
      fst::StdMutableFst::Read(ifile, true));
  if (!fst) return 1;

  std::ofstream ofstrm;
  if (argc > 2 && (strcmp(argv[2], "-") != 0)) {
    ofstrm.open(argv[2]);
    if (!ofstrm) {
      LOG(ERROR) << argv[0] << ": Open failed, file = " << argv[2];
      return 1;
    }
  }
  std::ostream &ostrm = ofstrm.is_open() ? ofstrm : std::cout;
  ngram::NGramModel<fst::StdArc> ngram(
      *fst, 0, FST_FLAGS_norm_eps,
      FST_FLAGS_check_consistency);
  if (FST_FLAGS_check_consistency && !ngram.CheckTopology()) {
    NGRAMERROR() << "Bad ngram model topology";
    return 1;
  }
  ngram::PrintNGramInfo(ngram, ostrm);

  return 0;
}
