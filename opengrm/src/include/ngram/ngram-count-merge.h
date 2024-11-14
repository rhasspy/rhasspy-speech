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
// NGram model class for merging count FSTs.

#ifndef NGRAM_NGRAM_COUNT_MERGE_H_
#define NGRAM_NGRAM_COUNT_MERGE_H_

#include <fst/arc.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

class NGramCountMerge : public NGramMerge<fst::StdArc> {
 public:
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Constructs an NGramCountMerge object consisting of ngram model
  // to be merged.
  // Ownership of FST is retained by the caller.
  explicit NGramCountMerge(fst::StdMutableFst *infst1,
                           Label backoff_label = 0, double norm_eps = kNormEps,
                           bool check_consistency = false)
      : NGramMerge(infst1, backoff_label, norm_eps, check_consistency) {}

  // Perform count-model merger with n-gram model specified by the FST argument
  // and mixing weights alpha and beta.
  void MergeNGramModels(const fst::StdFst &infst2, double alpha,
                        double beta, bool norm = false) {
    alpha_ = -log(alpha);
    beta_ = -log(beta);
    if (!NGramMerge<fst::StdArc>::MergeNGramModels(infst2, norm)) {
      NGRAMERROR() << "Count merging failed";
      NGramModel<fst::StdArc>::SetError();
    }
  }

 protected:
  // Specifies resultant weight when combining a weight from each FST.
  Weight MergeWeights(StateId s1, StateId s2, Label Label, Weight w1, Weight w2,
                      bool in_fst1, bool in_fst2) const override {
    if (in_fst1 && in_fst2) {
      return NegLogSum(w1.Value() + alpha_, w2.Value() + beta_);
    } else if (in_fst1) {
      return w1.Value() + alpha_;
    } else {
      return w2.Value() + beta_;
    }
  }

  // Specifies the normalization constant per state 'st' depending whether
  // state was present in one or boths FSTs.
  double NormWeight(StateId st, bool in_fst1, bool in_fst2) const override {
    if (in_fst1 && in_fst2) {
      return -NegLogSum(alpha_, beta_);
    } else if (in_fst1) {
      return -alpha_;
    } else {
      return -beta_;
    }
  }

  // Specifies if unshared arcs/final weights between the two
  // FSTs in a merge have a non-trivial merge. In particular, this
  // means MergeWeights() changes the arc or final weights; any
  // destination state changes are not relevant here. When false, more
  // efficient merging may be performed. If the arc/final_weight
  // comes from the first FST, then 'in_fst1' is true.
  bool MergeUnshared(bool in_fst1) const override {
    return (in_fst1) ? (alpha_ != 0.0) : (beta_ != 0.0);
  }

 private:
  double alpha_;  // weight to scale model ngram1
  double beta_;   // weight to scale model ngram2
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_COUNT_MERGE_H_
