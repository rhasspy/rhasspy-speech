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
// NGram model class for marginalizing the model.

#ifndef NGRAM_NGRAM_MARGINALIZE_H_
#define NGRAM_NGRAM_MARGINALIZE_H_

#include <vector>

#include <fst/arc.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-model.h>
#include <ngram/ngram-mutable-model.h>
#include <ngram/util.h>

namespace ngram {

class NGramMarginal : public NGramMutableModel<fst::StdArc> {
 public:
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;
  typedef fst::StdArc::Weight Weight;

  // Construct an NGramMarginal object, including an NGramModel and parameters
  explicit NGramMarginal(fst::StdMutableFst *infst, Label backoff_label = 0,
                         double norm_eps = kNormEps, int max_bo_updates = 10,
                         bool check_consistency = false)
      : NGramMutableModel<fst::StdArc>(
            infst, backoff_label, norm_eps,
            /* state_ngrams= */ check_consistency,
            /* infinite_backoff= */ false),
        max_bo_updates_(max_bo_updates) {
    ns_ = infst->NumStates();
    for (StateId st = 0; st < ns_; ++st)
      marginal_stats_.push_back(MarginalStateStats());
  }

  // Marginalize n-gram model, based on initialized parameters.  Returns
  // true on success.
  bool MarginalizeNGramModel() {
    if (!CheckNormalization()) {  // requires normalized model
      NGRAMERROR() << "NGramMarginal: Model not normalized;"
                   << " Model must be normalized for this estimation method";
      NGramModel<fst::StdArc>::SetError();
      return false;
    }
    if (!CalculateStateProbs()) {  // calculate p(h)
      NGRAMERROR() << "NGramMarginal: state probability calculation failed";
      NGramModel<fst::StdArc>::SetError();
      return false;
    }
    CalculateNewWeights();         // calculate new arc weights
    RecalcBackoff();               // re-calcs backoff weights
    if (!CheckNormalization()) {   // model should be normalized
      NGRAMERROR() << "NGramMarginal: Marginalized model not fully normalized";
      NGramModel<fst::StdArc>::SetError();
      return false;
    }
    return true;
  }

 private:
  struct MarginalStateStats {
    double log_prob;         // log probability of history represented by state
    double sum_ho_log_prob;  // log sum prob of hi-order states backing off
    double sum_ho_log_prob_w_bo;  // log sum prob including backoff weights
    std::vector<StateId> hi_states;    // vector of states backing off to this
    std::vector<double> arc_found;     // arc sums from states with arc
    std::vector<double> arc_notfound;  // arc sums from states without arc

    MarginalStateStats()
        : log_prob(0),
          sum_ho_log_prob(-fst::LogArc::Weight::Zero().Value()),
          sum_ho_log_prob_w_bo(-fst::LogArc::Weight::Zero().Value()) {}
  };

  // Calculates state marginal probs, and sums higher order log probs
  // Returns true on success.
  bool CalculateStateProbs();

  void DiffProbs(std::vector<double> *weights) const;

  // Establish re-calculated denominator value (log_prob is minimum)
  double SaneDenominator(double total_wbo, double found_sum, double log_prob) {
    double minval = fmax(-33, log_prob),
           ret = (found_sum > total_wbo) ? -NegLogDiff(total_wbo, found_sum)
                                         : minval;
    return fmax(ret, minval);
  }

  // Add value to vector of probabilities of arcs found at higher states
  void AddToArcFound(StateId st, size_t idx, double val) {
    marginal_stats_[st].arc_found[idx] =
        -NegLogSum(-marginal_stats_[st].arc_found[idx], val);
  }

  // Add value to vector of probabilities of arcs not found at higher states
  void AddToArcNotFound(StateId st, size_t idx, double val) {
    marginal_stats_[st].arc_notfound[idx] =
        -NegLogSum(-marginal_stats_[st].arc_notfound[idx], val);
  }

  // Subt value from vector of probabilities of arcs not found at higher states
  void SubtFromArcNotFound(StateId st, size_t idx, double val) {
    marginal_stats_[st].arc_notfound[idx] =
        SaneDenominator(-marginal_stats_[st].arc_notfound[idx], val,
                        marginal_stats_[st].log_prob);
  }

  // Returns change in state norm to control mass reserved for backoff.
  double MassReservedForBackoff(StateId st,
                                double original_norm, double new_norm) {
    if (st == UnigramState() || new_norm <= 0.0) {
      // Keeps original mass reserved for backoff.
      if (st != UnigramState()) {
        VLOG(2) << "NGramMarginalize: using original reserved mass: st: "
                << st;
      }
      return new_norm - original_norm;
    } else {
      // Keeps new mass reserved for backoff.
      return 0.0;
    }
  }


  // Initialize every arc 'not there' value with total; or reset index to -1
  void SetArcIndices(StateId st, bool initialize);

  // Get idx from current state
  int GetCurrentArcIndex(int label);

  // Function to set accumulators based on whether arc was found or not
  // arc_found accumulates the second term in numerator of formula
  // arc_notfound accumulates the denominator of formula
  void UpdateAccum(StateId st, StateId bst, size_t idx, size_t hidx,
                   bool update_found, double arcvalue, double bo_weight);

  // Scan through arcs in a state and collect higher order statistics
  void HigherOrderArcSum(StateId st, bool sum_found);

  // Scan through states backing off to state and collect statistics
  void HigherOrderStateSum(StateId st);

  // Calculate arc weight while ensuring resulting value is sane
  double SaneArcWeight(StateId st, size_t idx, double prob);

  // Use statistics to determine arc weights
  double GetSaneArcWeights(StateId st, std::vector<double> *wts);

  // Set arc weights with gathered weights
  void SetSaneArcWeights(StateId st, std::vector<double> *wts, double norm);

  // Use statistics to determine arc weights, second time around (new backoffs)
  // Add in formerly subtracted denominator, subtract updated denominator
  double UpdSaneArcWeights(StateId st, std::vector<double> *wts,
                           std::vector<double> *hold_notfound);

  // Recalculate backoff weights of all higher order states; indicate if changed
  bool StateHigherOrderBackoffRecalc(StateId st);

  // Recalculate backoff weights of higher order states;
  // if updated, recalculate arcs based on this.
  bool HigherOrderBackoffRecalc(StateId st, std::vector<double> *wts,
                                double *norm);

  // Calculate new weights enforcing marginalization constraints
  // P(w, h') - sum_of_found gamma(w | h) p(h) / sum_of_not alpha_h p(h)
  void CalculateNewStateWeights(StateId st);

  // Calculate state weights from highest order to lowest order
  void CalculateNewWeights();

  StateId ns_;
  std::vector<MarginalStateStats> marginal_stats_;
  std::vector<int> indices_;
  int max_bo_updates_;
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_MARGINALIZE_H_
