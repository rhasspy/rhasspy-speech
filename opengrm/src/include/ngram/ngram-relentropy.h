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
// Stolcke relative entropy style model shrinking derived class.

#ifndef NGRAM_NGRAM_RELENTROPY_H_
#define NGRAM_NGRAM_RELENTROPY_H_

#include <fst/arc.h>
#include <ngram/ngram-shrink.h>

namespace ngram {

class NGramRelEntropy : public NGramShrink<fst::StdArc> {
 public:
  // Constructs an NGramRelEntropy object that prunes an LM using relative
  // entropy criterion.
  NGramRelEntropy(fst::StdMutableFst *infst, double theta,
                  int shrink_opt = 0, double tot_uni = -1.0,
                  Label backoff_label = 0, double norm_eps = kNormEps,
                  bool check_consistency = false)
      : NGramShrink<fst::StdArc>(infst, shrink_opt, tot_uni, backoff_label,
                                     norm_eps, check_consistency) {
    // Threshold provided in real domain, convert to log
    theta_ = log(theta + 1);  // e^D - 1 <= theta_ -> D <= log(theta_ + 1)
  }

  // Shrink n-gram model, based on initialized parameters. No ngrams smaller
  // than min_order will be pruned; min_order must be at least 2 (the default
  // value).
  bool ShrinkNGramModel(int min_order = 2) {
    return NGramShrink<fst::StdArc>::ShrinkNGramModel(
        /* require_norm = */ true, min_order);
  }

  // Returns a theta that will yield the target number of ngrams and no more.
  // In relative entropy shrinking, theta is initially in real domain, then
  // converted to log domain for pruning.  In this function we convert back
  // from log domain to real domain for the threshold.  Default minimum order
  // is bigrams (2), which is the minimum possible.
  void CalculateTheta(int target_number_of_ngrams, int min_order = 2) {
    theta_ = ThetaForMaxNGrams(target_number_of_ngrams, min_order);
  }

 protected:
  // provide the pruning threshold
  double GetTheta(StateId state) const override { return theta_; }

  // Compute shrink score for transition based on Stolcke (KL) formula
  // D(p||p') = -p(h) { p(w|h) [ log p(w|h') + log \alpha'(h) - log p(w|h) ] +
  //            \alpha_numerator(h) [ log \alpha'(h) - log \alpha (h) ] }
  // return exp(D(p||p')) - 1
  double ShrinkScore(const ShrinkStateStats &state,
                     const ShrinkArcStats &arc) const override {
    if (arc.log_prob == -fst::StdArc::Weight::Zero().Value() ||
        state.log_prob == -fst::StdArc::Weight::Zero().Value()) {
      return -fst::StdArc::Weight::Zero().Value();
    }
    double new_log_backoff = CalcNewLogBackoff(arc);
    double score = arc.log_backoff_prob + new_log_backoff - arc.log_prob;
    double secondterm =
        new_log_backoff + (GetNLogBackoffNum() - GetNLogBackoffDenom());
    secondterm *= exp(-GetNLogBackoffNum());
    score *= exp(arc.log_prob);
    score += secondterm;
    score *= -exp(state.log_prob);
    return score;
  }

 private:
  double theta_;  // Shrinking parameter
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_RELENTROPY_H_
