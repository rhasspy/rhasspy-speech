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
// Make model from raw counts or histograms.

#include <ngram/ngram-make.h>

#include <cstdint>
#include <string>

#include <fst/arc-map.h>
#include <fst/arc.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/script/fst-class.h>
#include <ngram/hist-arc.h>
#include <ngram/hist-mapper.h>
#include <ngram/ngram-absolute.h>
#include <ngram/ngram-katz.h>
#include <ngram/ngram-kneser-ney.h>
#include <ngram/ngram-unsmoothed.h>
#include <ngram/ngram-witten-bell.h>
#include <ngram/util.h>

namespace ngram {

using ::fst::StdArc;
using ::fst::StdFst;
using ::fst::StdMutableFst;

// Makes models from NGram count FSTs with StdArc counts.
bool NGramMakeModel(fst::StdMutableFst *fst, const std::string &method,
                    const fst::StdFst *ccfst, bool backoff,
                    bool interpolate, int64_t bins, double witten_bell_k,
                    double discount_D, int64_t backoff_label, double norm_eps,
                    bool check_consistency) {
  if (backoff && interpolate) {
    // Checks that these parameters make sense.  If both are false, defaults
    // to the default method for the smoothing method.  Both shouldn't be true.
    LOG(ERROR) << "NGramMakeModel only allows one of backoff or interpolated "
                  "to be set to true, not both";
    return false;
  }

  if (method == "kneser_ney") {
    ngram::NGramKneserNey ngram(fst, backoff, backoff_label, norm_eps,
                                check_consistency, discount_D, bins);
    if (ccfst) ngram.SetCountOfCounts(*ccfst);
    if (!ngram.MakeNGramModel()) {
      NGRAMERROR() << "NGramKneserNey: failed to make model";
      return false;
    }
  } else if (method == "absolute") {
    ngram::NGramAbsolute ngram(fst, backoff, backoff_label, norm_eps,
                               check_consistency, discount_D, bins);
    if (ccfst) ngram.SetCountOfCounts(*ccfst);
    if (!ngram.MakeNGramModel()) {
      NGRAMERROR() << "NGramAbsolute: failed to make model";
      return false;
    }
  } else if (method == "katz") {
    ngram::NGramKatz<fst::StdArc> ngram(fst, !interpolate, backoff_label,
                                            norm_eps, check_consistency, bins);
    if (ccfst) ngram.SetCountOfCounts(*ccfst);
    if (!ngram.MakeNGramModel()) {
      NGRAMERROR() << "NGramKatz: failed to make model";
      return false;
    }
  } else if (method == "witten_bell") {
    ngram::NGramWittenBell ngram(fst, backoff, backoff_label, norm_eps,
                                 check_consistency, witten_bell_k);
    if (!ngram.MakeNGramModel()) {
      NGRAMERROR() << "NGramWittenBell: failed to make model";
      return false;
    }
  } else if (method == "unsmoothed" || method == "presmoothed") {
    // presmoothed should only be used with randgen counts.
    bool prefix_norm = method == "unsmoothed" ? false : true;
    ngram::NGramUnsmoothed ngram(fst, !interpolate, prefix_norm, backoff_label,
                                 norm_eps, check_consistency);
    if (!ngram.MakeNGramModel()) {
      NGRAMERROR() << "NGramUnsmoothed: failed to make model";
      return false;
    }
  } else {
    LOG(ERROR) << "Model method " << method
               << " not processed by NGramMakeModel";
    return false;
  }
  return true;
}

// The same, but uses scripting FSTs.
bool NGramMakeModel(fst::script::MutableFstClass *fst,
                    const std::string &method,
                    const fst::script::FstClass *ccfst, bool backoff,
                    bool interpolate, int64_t bins, double witten_bell_k,
                    double discount_D, int64_t backoff_label, double norm_eps,
                    bool check_consistency) {
  StdMutableFst *typed_fst = fst->GetMutableFst<StdArc>();
  const StdFst *typed_ccfst = ccfst ? ccfst->GetFst<StdArc>() : nullptr;
  return NGramMakeModel(typed_fst, method, typed_ccfst, backoff, interpolate,
                        bins, witten_bell_k, discount_D, backoff_label,
                        norm_eps, check_consistency);
}

// Makes models from NGram count FSTs with HistogramArc counts.
bool NGramMakeHistModel(fst::MutableFst<ngram::HistogramArc> *hist_fst,
                        fst::StdMutableFst *fst, const std::string &method,
                        const fst::StdFst *ccfst, bool interpolate,
                        int64_t bins, int64_t backoff_label, double norm_eps,
                        bool check_consistency) {
  if (method == "katz_frac") {
    ngram::NGramKatz<ngram::HistogramArc> ngram(hist_fst, !interpolate,
                                                backoff_label, norm_eps,
                                                check_consistency, bins);
    if (ccfst) ngram.SetCountOfCounts(*ccfst);
    if (!ngram.MakeNGramModel()) {
      NGRAMERROR() << "NGramKatz(Frac): failed to make model";
      return false;
    }
    ArcMap(ngram.GetFst(), fst, ToStdArcMapper());
  } else {
    LOG(ERROR) << "Model method " << method
               << " not processed by NGramMakeHistModel";
    return false;
  }
  return true;
}

}  // namespace ngram
