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

#include <ngram/ngram-marginalize.h>

#include <cstddef>
#include <vector>

#include <fst/arc.h>
#include <fst/expanded-fst.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

using ::fst::ArcIterator;
using ::fst::LogArc;
using ::fst::MATCH_INPUT;
using ::fst::Matcher;
using ::fst::MutableArcIterator;
using ::fst::StdArc;
using ::fst::StdExpandedFst;
using ::fst::StdFst;
using ::fst::StdMutableFst;

// Calculates state marginal probs, and sums higher order log probs
// Returns true with success.
bool NGramMarginal::CalculateStateProbs() {
  // The stationary distribution on the resultant marginally-constrained
  // FST is equal to the difference of the ascending probs on the
  // input 'summed' FST.
  std::vector<double> weights;
  if (!NGramModel::CalculateStateProbs(&weights, false))
    return false;
  DiffProbs(&weights);
  for (size_t s = 0; s < weights.size(); ++s) {  // initialize state values
    marginal_stats_[s].log_prob = log(weights[s]);
    marginal_stats_[s].sum_ho_log_prob_w_bo = log(weights[s]);
  }
  for (StateId st = 0; st < ns_; ++st) {
    StateId bst = GetBackoff(st, nullptr);
    if (bst >= 0) {       // if state backs off to another state
      marginal_stats_[bst].hi_states.push_back(st);  // add to list
      marginal_stats_[bst].sum_ho_log_prob =
          -NegLogSum(-marginal_stats_[bst].sum_ho_log_prob,
                     -marginal_stats_[st].log_prob);  // add to ho_prob
    }
  }
  return true;
}

// Subtracts higher-order probs from lower orders and normalizes.
void NGramMarginal::DiffProbs(std::vector<double> *weights) const {
  // Removes higher-order mass.
  for (int order = 2; order <= HiOrder(); ++order) {
    for (StateId st = 0; st < GetExpandedFst().NumStates(); ++st) {
      if (StateOrder(st) == order) {
        StateId bst = GetBackoff(st, nullptr);
        (*weights)[bst] -= (*weights)[st];
      }
    }
  }
  // Bounds and finds sum.
  double sum = 0.0;
  const double kEffectiveZero = 1.0e-20;
  for (size_t st = 0; st < weights->size(); ++st) {
    if ((*weights)[st] < kEffectiveZero)
      (*weights)[st] = kEffectiveZero;
    sum += (*weights)[st];
  }
  // Normalizes.
  for (size_t st = 0; st < weights->size(); ++st)
    (*weights)[st] /= sum;
}

// Function to set accumulators based on whether arc was found or not
// arc_found accumulates the second term in numerator of formula
// arc_notfound accumulates the denominator of formula
// update_found boolean indicates whether found arc stats should be updated
void NGramMarginal::UpdateAccum(StateId st, StateId bst, size_t idx,
                                size_t hidx, bool update_found, double arcvalue,
                                double bo_weight) {
  if (update_found) {  // updating found items, too
    if (!marginal_stats_[bst].arc_found.empty()) {  // not highest order
      // add in the arc_found values from higher order state
      AddToArcFound(st, idx, -marginal_stats_[bst].arc_found[hidx]);
      // add in the arc value from all states using the bst arc
      AddToArcFound(st, idx,
                    arcvalue - marginal_stats_[bst].arc_notfound[hidx]);
    } else {
      AddToArcFound(st, idx, arcvalue - marginal_stats_[bst].log_prob);
    }
  }
  SubtFromArcNotFound(st, idx,
                      bo_weight - marginal_stats_[bst].sum_ho_log_prob_w_bo);
}

// initialize every arc 'not there' value with total; or reset index to -1
void NGramMarginal::SetArcIndices(StateId st, bool initialize) {
  size_t idx = 0;
  if (initialize && GetFst().Final(st) != StdArc::Weight::Zero())
    marginal_stats_[st].arc_notfound[idx] =
        marginal_stats_[st].sum_ho_log_prob_w_bo;
  for (ArcIterator<StdExpandedFst> aiter(GetExpandedFst(), st); !aiter.Done();
       aiter.Next()) {
    StdArc arc = aiter.Value();
    ++idx;
    if (arc.ilabel == BackoffLabel()) continue;  // ignore backoff arc
    while (indices_.size() <= arc.ilabel + 1) indices_.push_back(-1);
    indices_[arc.ilabel] = initialize ? idx : -1;
    if (initialize)  // if initializing, set backoff sums for arc
      marginal_stats_[st].arc_notfound[idx] =
          marginal_stats_[st].sum_ho_log_prob_w_bo;
  }
}

// Get idx from current state
int NGramMarginal::GetCurrentArcIndex(int label) {
  while (indices_.size() <= label + 1) indices_.push_back(-1);
  return indices_[label];
}

// Scan through arcs in a state and collect higher order statistics.
// Later iterations only update the 'not_found' statistics, hence sum_found bool
void NGramMarginal::HigherOrderArcSum(StateId st, bool sum_found) {
  SetArcIndices(st, true);
  for (size_t i = 0; i < marginal_stats_[st].hi_states.size(); ++i) {
    Matcher<StdFst> matcher(GetFst(), MATCH_INPUT);
    matcher.SetState(st);
    StateId bst = marginal_stats_[st].hi_states[i];  // higher order state
    size_t idx = 0, hidx = 0;                        // idx = 0 is </s>
    Weight bo_weight;
    GetBackoff(bst, &bo_weight);
    if (GetFst().Final(bst) != StdArc::Weight::Zero())
      UpdateAccum(st, bst, idx, hidx, sum_found, GetFst().Final(bst).Value(),
                  ScalarValue(bo_weight));
    for (ArcIterator<StdExpandedFst> aiter(GetExpandedFst(), bst);
         !aiter.Done(); aiter.Next()) {
      StdArc arc = aiter.Value();
      ++hidx;
      if (arc.ilabel == BackoffLabel()) continue;  // ignore backoff arc
      if (!matcher.Find(arc.ilabel)) {
        NGRAMERROR() << "lower order arc not found";
        NGramModel<StdArc>::SetError();
        return;
      }
      StdArc barc = matcher.Value();
      int val = GetCurrentArcIndex(barc.ilabel);  // get idx from current state
      if (val < 0) {
        NGRAMERROR() << "lower order arc index not set";
        NGramModel<StdArc>::SetError();
        return;
      }
      idx = val;
      UpdateAccum(st, bst, idx, hidx, sum_found, arc.weight.Value(),
                  ScalarValue(bo_weight));
    }
  }
  SetArcIndices(st, false);
}

// Scan through states backing off to state and collect statistics
void NGramMarginal::HigherOrderStateSum(StateId st) {
  // initialize arc_found and arc_notfound values for state
  bool first_iteration = marginal_stats_[st].arc_found.empty();
  for (size_t i = 0; i <= GetExpandedFst().NumArcs(st); ++i) {
    if (first_iteration) {  // need to push back onto vector
      marginal_stats_[st].arc_found.push_back(-LogArc::Weight::Zero().Value());
      // value in arc_notfound includes residual mass at state itself
      marginal_stats_[st].arc_notfound.push_back(marginal_stats_[st].log_prob);
    } else {  // already allocated
      marginal_stats_[st].arc_found[i] = -LogArc::Weight::Zero().Value();
      marginal_stats_[st].arc_notfound[i] = marginal_stats_[st].log_prob;
    }
  }
  // for recursive accumulation of higher order probabilities
  for (size_t i = 0; i < marginal_stats_[st].hi_states.size(); ++i) {
    StateId bst = marginal_stats_[st].hi_states[i];  // bst backs off to st
    Weight bo_weight;
    GetBackoff(bst, &bo_weight);
    if (!marginal_stats_[bst].arc_found.empty())  // if not highest order
      marginal_stats_[st].sum_ho_log_prob =  // otherwise already accumulated
          -NegLogSum(-marginal_stats_[st].sum_ho_log_prob,
                     -marginal_stats_[bst].sum_ho_log_prob);
    marginal_stats_[st].sum_ho_log_prob_w_bo = -NegLogSum(
        -marginal_stats_[st].sum_ho_log_prob_w_bo,
        -marginal_stats_[bst].sum_ho_log_prob_w_bo + ScalarValue(bo_weight));
  }
}

// Calculate arc weight while ensuring resulting value is sane
double NGramMarginal::SaneArcWeight(StateId st, size_t idx, double prob) {
  double has = -marginal_stats_[st].arc_found[idx];
  if (has <= prob) {  // numerator <= 0; set to small default value
    VLOG(2) << "NGramMarginalize: non-positive arc weight set to kFloatEps: "
            << "st: "  << st << " idx: " << idx;
    prob = -log(kFloatEps);
  } else {
    prob = NegLogDiff(prob, has);
  }
  prob -= -marginal_stats_[st].arc_notfound[idx];
  return prob;
}

// Use statistics to determine arc weights
double NGramMarginal::GetSaneArcWeights(StateId st, std::vector<double> *wts) {
  size_t idx = 0;
  double original_norm = GetFst().Final(st).Value(), new_norm = original_norm,
         KahanVal1 = 0, KahanVal2 = 0,  // Values for Kahan summation algorithm
      total_state_sum = -NegLogSum(-marginal_stats_[st].log_prob,
                                   -marginal_stats_[st].sum_ho_log_prob);
  if (GetFst().Final(st).Value() != LogArc::Weight::Zero().Value())
    new_norm = SaneArcWeight(st, idx, original_norm - total_state_sum);
  wts->push_back(new_norm);
  for (ArcIterator<StdExpandedFst> aiter(GetExpandedFst(), st); !aiter.Done();
       aiter.Next()) {
    StdArc arc = aiter.Value();
    idx++;
    wts->push_back(0);
    if (arc.ilabel == BackoffLabel()) continue;  // ignore backoff arc
    original_norm = NegLogSum(original_norm, arc.weight.Value(), &KahanVal1);
    (*wts)[idx] = SaneArcWeight(st, idx, arc.weight.Value() - total_state_sum);
    new_norm = NegLogSum(new_norm, (*wts)[idx], &KahanVal2);
  }

  return MassReservedForBackoff(st, original_norm, new_norm);
}

// Set arc weights with gathered weights
void NGramMarginal::SetSaneArcWeights(StateId st, std::vector<double> *wts,
                                      double norm) {
  int idx = 0;
  (*wts)[idx] -= norm;                         // normalize first
  GetMutableFst()->SetFinal(st, (*wts)[idx]);  // then set value
  for (MutableArcIterator<StdMutableFst> aiter(GetMutableFst(), st);
       !aiter.Done(); aiter.Next()) {
    StdArc arc = aiter.Value();
    idx++;
    if (arc.ilabel == BackoffLabel()) continue;
    (*wts)[idx] -= norm;
    arc.weight = (*wts)[idx];
    aiter.SetValue(arc);
  }
}

// Use statistics to determine arc weights, second time around (new backoffs)
// Add in formerly subtracted denominator, subtract updated denominator
double NGramMarginal::UpdSaneArcWeights(StateId st, std::vector<double> *wts,
                                        std::vector<double> *hold_notfound) {
  size_t idx = 0;
  double orig_norm = (*wts)[0];  // keep track of original normalization
  (*wts)[0] -= (*hold_notfound)[idx] - marginal_stats_[st].arc_notfound[idx];
  double norm = (*wts)[0];
  for (ArcIterator<StdExpandedFst> aiter(GetExpandedFst(), st); !aiter.Done();
       aiter.Next()) {
    StdArc arc = aiter.Value();
    idx++;
    if (arc.ilabel == BackoffLabel()) continue;
    orig_norm = NegLogSum(orig_norm, (*wts)[idx]);
    (*wts)[idx] -=
        (*hold_notfound)[idx] - marginal_stats_[st].arc_notfound[idx];
    norm = NegLogSum(norm, (*wts)[idx]);
  }

  return MassReservedForBackoff(st, orig_norm, norm);
}

// Recalculate backoff weights of all higher order states; indicate if changed
bool NGramMarginal::StateHigherOrderBackoffRecalc(StateId st) {
  bool upd = false;
  marginal_stats_[st].sum_ho_log_prob_w_bo = marginal_stats_[st].log_prob;
  for (size_t i = 0; i < marginal_stats_[st].hi_states.size(); ++i) {
    StateId bst = marginal_stats_[st].hi_states[i];
    Weight bo_weight, new_bo_weight;
    GetBackoff(bst, &bo_weight);
    RecalcBackoff(bst);
    GetBackoff(bst, &new_bo_weight);
    marginal_stats_[st].sum_ho_log_prob_w_bo =
        -NegLogSum(-marginal_stats_[st].sum_ho_log_prob_w_bo,
                   -marginal_stats_[bst].sum_ho_log_prob_w_bo +
                       ScalarValue(new_bo_weight));
    if (fabs(ScalarValue(bo_weight) - ScalarValue(new_bo_weight)) < kNormEps)
      continue;
    upd = true;
  }
  return upd;
}

// Recalculate backoff weights of higher order states;
// if updated, recalculate arcs based on this.
bool NGramMarginal::HigherOrderBackoffRecalc(StateId st,
                                             std::vector<double> *wts,
                                             double *norm) {
  if (StateHigherOrderBackoffRecalc(st)) {  // recalculate denominators of arcs
    std::vector<double> hold_notfound;      // to hold prior not_found values
    for (size_t i = 0; i < marginal_stats_[st].arc_notfound.size(); ++i) {
      hold_notfound.push_back(marginal_stats_[st].arc_notfound[i]);
      marginal_stats_[st].arc_notfound[i] = marginal_stats_[st].log_prob;
    }
    HigherOrderArcSum(st, false);  // perform arc sum, but don't update found
    (*norm) = UpdSaneArcWeights(st, wts, &hold_notfound);
    return true;
  }
  return false;
}

// Calculate new weights enforcing marginalization constraints
// P(w, h') - sum_of_found gamma(w | h) p(h) / sum_of_not alpha_h p(h)
void NGramMarginal::CalculateNewStateWeights(StateId st) {
  HigherOrderStateSum(st);      // collect stats from higher order states
  HigherOrderArcSum(st, true);  // collect stats for all arcs
  std::vector<double> wts;      // For storing resulting arc weights
  double norm = GetSaneArcWeights(st, &wts);
  bool need_upd = true;
  int upd_count = 0;
  while (need_upd && (upd_count == 0 || upd_count < max_bo_updates_)) {
    SetSaneArcWeights(st, &wts, norm);  // assign new weights to arcs
    need_upd = (max_bo_updates_ > 0) ?
        HigherOrderBackoffRecalc(st, &wts, &norm) : false;
    ++upd_count;
  }
}

// Calculate state weights from highest order to lowest order
void NGramMarginal::CalculateNewWeights() {
  for (int order = HiOrder() - 1; order >= 1; --order) {  // for each order
    for (StateId st = 0; st < GetExpandedFst().NumStates(); ++st) {  // all st
      if (StateOrder(st) == order &&  // if state is the current order and
          !marginal_stats_[st].hi_states.empty()) {  // is backed off to
        CalculateNewStateWeights(st);
      }
    }
  }
}

}  // namespace ngram
