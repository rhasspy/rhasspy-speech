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
// Seymore and Rosenfeld style model shrinking derived class.

#ifndef NGRAM_NGRAM_SEYMORE_SHRINK_H_
#define NGRAM_NGRAM_SEYMORE_SHRINK_H_

#include <fst/arc.h>
#include <ngram/ngram-shrink.h>

namespace ngram {

class NGramSeymoreShrink : public NGramShrink<fst::StdArc> {
 public:
  // Constructs an NGramSeymoreShrink object that prunes an LM using the
  // Seymore-Rosenfeld criterion.
  NGramSeymoreShrink(fst::StdMutableFst *infst, double theta,
                     int shrink_opt = 0, double tot_uni = -1.0,
                     Label backoff_label = 0, double norm_eps = kNormEps,
                     bool check_consistency = false)
      : NGramShrink<fst::StdArc>(infst, shrink_opt, tot_uni, backoff_label,
                                     norm_eps, check_consistency),
        theta_(theta) {}

  // Shrink n-gram model, based on initialized parameters (requires normalized).
  // No ngrams smaller than min_order will be pruned; min_order must be at
  // least 2 (the default value).
  bool ShrinkNGramModel(int min_order = 2) {
    return NGramShrink<fst::StdArc>::ShrinkNGramModel(
        /* require_norm = */ true, min_order);
  }

  // Returns a theta that will yield the target number of ngrams and no more.
  // No ngrams smaller than min_order will be pruned; min_order must be at
  // least 2 (the default value).
  void CalculateTheta(int target_number_of_ngrams, int min_order = 2) {
    theta_ = ThetaForMaxNGrams(target_number_of_ngrams, min_order);
  }

  // provide the pruning threshold
  double GetTheta(StateId state) const override { return theta_; }

 protected:
  // Compute shrink score for transition based on Seymore/Rosenfeld formula
  // N(w,h) [ log p(w|h) - log p'(w|h) ] where N(w,h) is discounted frequency
  double ShrinkScore(const ShrinkStateStats &state,
                     const ShrinkArcStats &arc) const override {
    if (arc.log_prob == -fst::StdArc::Weight::Zero().Value() ||
        state.log_prob == -fst::StdArc::Weight::Zero().Value()) {
      return -fst::StdArc::Weight::Zero().Value();
    }
    double score;
    score = arc.log_prob - CalcNewLogBackoff(arc);
    score -= arc.log_backoff_prob;
    score *= GetTotalUnigramCount();
    score *= exp(state.log_prob + arc.log_prob);
    return score;
  }

 private:
  double theta_;  // Shrinking parameter
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_SEYMORE_SHRINK_H_
