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
// NGram model class for merging histogram FSTs.

#ifndef NGRAM_NGRAM_HIST_MERGE_H_
#define NGRAM_NGRAM_HIST_MERGE_H_

#include <array>
#include <cmath>

#include <fst/mutable-fst.h>
#include <ngram/hist-arc.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

class NGramHistMerge : public NGramMerge<HistogramArc> {
 public:
  typedef HistogramArc::StateId StateId;
  typedef HistogramArc::Label Label;

  // Constructs an NGramCountMerge object consisting of ngram model
  // to be merged.
  // Ownership of FST is retained by the caller.
  explicit NGramHistMerge(fst::MutableFst<HistogramArc> *infst1,
                          Label backoff_label = 0, double norm_eps = kNormEps,
                          bool check_consistency = false)
      : NGramMerge(infst1, backoff_label, norm_eps, check_consistency) {}

  // Perform count-model merger with n-gram model specified by the FST argument
  // and mixing weights alpha and beta.
  void MergeNGramModels(const fst::Fst<HistogramArc> &infst2, double alpha,
                        double beta, bool norm = false) {
    alpha_ = -log(alpha);
    beta_ = -log(beta);
    if (!NGramMerge::MergeNGramModels(infst2, norm)) {
      NGRAMERROR() << "Histogram count merging failed";
      NGramModel<HistogramArc>::SetError();
    }
  }

 protected:
  // Specifies resultant weight when combining a weight from each FST.
  Weight MergeWeights(StateId s1, StateId s2, Label Label, Weight w1, Weight w2,
                      bool in_fst1, bool in_fst2) const override {
    if (in_fst1 && in_fst2) {
      return NGramHistMerge::WeightSum(w1, w2);
    } else if (in_fst1) {
      return w1;
    } else {
      return w2;
    }
  }

  // TODO(vitalyk): this does nothing!
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
  // Add together two weights using addition for histogram weights.
  // Histogram weight is a tuple where first coordinate corresponds
  // to expected count and the rest K+1 coordinates indicate
  // the probability of observing index-1 occurrences of the n-gram
  // (associated with this weight).
  Weight WeightSum(Weight w1, Weight w2) const {
    std::array<fst::TropicalWeight, kHistogramBins> v;
    v.fill(fst::TropicalWeight::Zero());
    v[0] = NegLogSum(w1.Value(0).Value(), w2.Value(0).Value());

    for (int k = 0; k < v.size() - 1; k++) {
      for (int j = 0; j <= k; j++) {
        v[k + 1] = NegLogSum(v[k + 1].Value(), w1.Value(j + 1).Value() +
                                                   w2.Value(k - j + 1).Value());
      }
    }
    return fst::PowerWeight<fst::TropicalWeight, kHistogramBins>(
        v.begin(), v.end());
  }

  double alpha_;  // weight to scale model ngram1
  double beta_;   // weight to scale model ngram2
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_HIST_MERGE_H_
