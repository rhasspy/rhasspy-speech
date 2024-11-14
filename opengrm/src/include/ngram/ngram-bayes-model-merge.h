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
// Implements 'full' bayes model merging.

#ifndef NGRAM_NGRAM_BAYES_MODEL_MERGE_H_
#define NGRAM_NGRAM_BAYES_MODEL_MERGE_H_

#include <cmath>
#include <vector>

#include <fst/arc.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>

namespace ngram {

class NGramBayesModelMerge : public NGramMerge<fst::StdArc> {
 public:
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Constructs an NGramBayesModelMerge object consisting of ngram model
  // to be merged.
  // Ownership of FST is retained by the caller.
  explicit NGramBayesModelMerge(fst::StdMutableFst *infst1,
                                Label backoff_label = 0,
                                double norm_eps = kNormEps)
      : NGramMerge<fst::StdArc>(infst1, backoff_label, norm_eps, true) {
    if (!CheckNormalization()) {
      NGRAMERROR() << "NGramBayesModelMerge: Model 1 must be normalized to"
                   << " use smoothing in merging";
      NGramModel::SetError();
    }
  }

  // Perform smooth-model merge with n-gram model specified by the FST argument
  // and mixing weights alpha and beta. Resultant model will be normalized.
  void MergeNGramModels(const fst::StdFst &infst2, double alpha,
                        double beta) {
    NGramModel<fst::StdArc> mod2(infst2);
    if (!mod2.CheckNormalization()) {
      NGRAMERROR() << "NGramBayesModelMerge: Model 2 must be normalized to"
                   << " use smoothing in merging";
      NGramModel::SetError();
      return;
    }
    alpha_ = -log(alpha);
    beta_ = -log(beta);
    state_alpha_.clear();
    if (!NGramMerge<fst::StdArc>::MergeNGramModels(infst2, true)) {
      NGRAMERROR() << "NGramBayesModelMerge: Model merging failed";
      NGramModel::SetError();
    }
  }

 protected:
  // Specifies resultant weight when combining a weight from each FST
  fst::StdArc::Weight MergeWeights(StateId s1, StateId s2, Label label,
                                       fst::StdArc::Weight w1,
                                       fst::StdArc::Weight w2, bool in_fst1,
                                       bool in_fst2) const override {
    if (label == BackoffLabel()) {  // don't modify (needed) backoff weights
      return in_fst1 ? w1.Value() : w2.Value();
    } else {
      StateId st = in_fst1 ? s1 : ExactMap2To1(s2);
      double alpha = StateAlpha(st);
      double beta = NegLogDiff(0.0, alpha);
      return NegLogSum(w1.Value() + alpha, w2.Value() + beta);
    }
  }

 private:
  // normalized state weight to scale model ngram1
  double StateAlpha(StateId st) const {
    while (st >= state_alpha_.size()) state_alpha_.push_back(-1.0);
    if (state_alpha_[st] < 0.0) {
      const std::vector<Label> &ngram = StateNGram(st);

      // -log p(h|k), k=1,2
      double w1 = ScalarValue(GetNGramCost(ngram));
      double w2 = ScalarValue(NGram2().GetNGramCost(ngram));

      // p(k|h) = p(h|k) p(k) / sum_k' p(h|k') p(k')
      state_alpha_[st] = w1 + alpha_;

      // Only normalize non-infinite cost (to avoid potential NaN issues).
      // If state_alpha_[st] = inf (i.e., p = 0), then normalized is also inf.
      if (state_alpha_[st] < fst::StdArc::Weight::Zero().Value())
        state_alpha_[st] -= NegLogSum(w1 + alpha_, w2 + beta_);
    }
    return state_alpha_[st];
  }

  double alpha_;  // global weight to scale model ngram1
  double beta_;   // global weight to scale model ngram2

  // stored normalized state weight to scale model ngram1
  mutable std::vector<double> state_alpha_;
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_BAYES_MODEL_MERGE_H_
