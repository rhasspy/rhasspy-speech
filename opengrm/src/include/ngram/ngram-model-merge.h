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
// NGram model class for merging smoothed model FSTs.

#ifndef NGRAM_NGRAM_MODEL_MERGE_H_
#define NGRAM_NGRAM_MODEL_MERGE_H_

#include <fst/arc.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

class NGramModelMerge : public NGramMerge<fst::StdArc> {
 public:
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Constructs an NGramModelMerge object consisting of ngram model
  // to be merged.
  // Ownership of FST is retained by the caller.
  explicit NGramModelMerge(fst::StdMutableFst *infst1,
                           Label backoff_label = 0, double norm_eps = kNormEps,
                           bool check_consistency = false)
      : NGramMerge(infst1, backoff_label, norm_eps, check_consistency),
        merge_norm_(true) {
    if (!CheckNormalization()) {
      NGRAMERROR() << "NGramModelMerge: Model 1 must be normalized to"
                   << " use smoothing in merging";
      NGramModel::SetError();
    }
  }

  // Perform smooth-model merge with n-gram model specified by the FST argument
  // and mixing weights alpha and beta.
  void MergeNGramModels(const fst::StdFst &infst2, double alpha,
                        double beta, bool norm = false) {
    if (Error()) return;
    NGramModel<fst::StdArc> mod2(infst2);
    if (!mod2.CheckNormalization()) {
      NGRAMERROR() << "NGramModelMerge: Model 2 must be normalized to"
                   << " use smoothing in merging";
      NGramModel::SetError();
      return;
    }
    alpha_ = -log(alpha);
    beta_ = -log(beta);
    if (!NGramMerge<fst::StdArc>::MergeNGramModels(infst2, norm)) {
      NGRAMERROR() << "NGramModelMerge: Model merging failed";
      NGramModel::SetError();
      return;
    }
    if (!norm) merge_norm_ = false;
  }

 protected:
  // Specifies resultant weight when combining a weight from each FST
  Weight MergeWeights(StateId s1, StateId s2, Label label, Weight w1, Weight w2,
                      bool in_fst1, bool in_fst2) const override {
    if (label == BackoffLabel()) {  // don't modify (needed) backoff weights
      return in_fst1 ? w1.Value() : w2.Value();
    } else {
      return NegLogSum(w1.Value() + alpha_, w2.Value() + beta_);
    }
  }

  // Specifies normalization constant per state 'st' depending whether
  // state was present in one or boths FSTs.
  double NormWeight(StateId st, bool in_fst1, bool in_fst2) const override {
    return -NegLogSum(alpha_, beta_);
  }

 private:
  double alpha_;     // weight to scale model ngram1
  double beta_;      // weight to scale model ngram2
  bool merge_norm_;  // is the (possibly intermediate) result normalized?
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_MODEL_MERGE_H_
