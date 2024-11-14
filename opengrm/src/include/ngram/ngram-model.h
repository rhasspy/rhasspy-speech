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
// NGram model class.

#ifndef NGRAM_NGRAM_MODEL_H_
#define NGRAM_NGRAM_MODEL_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <ostream>
#include <vector>

#include <fst/fst.h>
#include <fst/matcher.h>
#include <ngram/hist-arc.h>
#include <ngram/util.h>

namespace ngram {

// Default normalization constant (e.g., for checks)
const double kNormEps = 0.001;
const double kFloatEps = 0.000001;
const double kInfBackoff = 99.00;

// Calculate - log( exp(a - b) + 1 ) for use in high precision NegLogSum
static double NegLogDeltaValue(double a, double b, double *c) {
  double x = exp(a - b), delta = -log(x + 1);
  if (x < kNormEps) {  // for small x, use Mercator Series to calculate
    delta = -x;
    for (int j = 2; j <= 4; ++j) delta += pow(-x, j) / j;
  }
  if (c) delta -= *c;  // Sum correction from Kahan formula (if using)
  return delta;
}

// Precision method for summing reals and saving negative logs
// -log( exp(-a) + exp(-b) ) = a - log( exp(a - b) + 1 )
// Uses Mercator series and Kahan formula for additional numerical stability
static double NegLogSum(double a, double b, double *c) {
  if (a == fst::StdArc::Weight::Zero().Value()) return b;
  if (b == fst::StdArc::Weight::Zero().Value()) return a;
  if (a > b) return NegLogSum(b, a, c);
  double delta = NegLogDeltaValue(a, b, c), val = a + delta;
  if (c) *c = (val - a) - delta;  // update sum correction for Kahan formula
  return val;
}

// Summing reals and saving negative logs, no Kahan formula (backwards compat)
static double NegLogSum(double a, double b) { return NegLogSum(a, b, nullptr); }

// Negative log of difference: -log(exp^{-a} - exp^{-b}).
// FRAGILE: assumes exp^{-a} >= exp^{-b}
// Sets (but does not clear) optional error field.
static double NegLogDiff(double a, double b, bool *error = nullptr) {
  if (b == fst::StdArc::Weight::Zero().Value()) return a;
  if (a >= b) {
    if (a - b >= kNormEps) {  // not equal within fp error
      NGRAMERROR() << "NegLogDiff: undefined " << a << " " << b;
      if (error) *error = true;
    }
    return fst::StdArc::Weight::Zero().Value();
  }
  return b - log(exp(b - a) - 1);
}

template <class Arc>
class NGramModel {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  // Construct an NGramModel object, consisting of the FST and some
  // information about the states under the assumption that the FST is
  // a model. The 'backoff_label' is what is followed when there is no
  // word match at a given order. The 'norm_eps' is the epsilon used
  // in checking weight normalization.  If 'state_ngrams' is true,
  // this class explicitly finds, checks the consistency of and stores
  // the ngram that must be read to reach each state (normally false
  // to save some time and space).
  NGramModel(const fst::Fst<Arc> &infst, Label backoff_label,
             double norm_eps, bool state_ngrams)
      : fst_(infst),
        backoff_label_(backoff_label),
        norm_eps_(norm_eps),
        have_state_ngrams_(state_ngrams),
        error_(false) {
    InitModel();
  }

  // Same as above, but requires the FST and the backoff label.
  NGramModel(const fst::Fst<Arc> &infst, Label backoff_label)
      : fst_(infst),
        backoff_label_(backoff_label),
        norm_eps_(kNormEps),
        have_state_ngrams_(false),
        error_(false) {
    InitModel();
  }

  // Same as above, but uses defaults for most of the parameters.
  explicit NGramModel(const fst::Fst<Arc> &infst)
      : fst_(infst),
        backoff_label_(0),
        norm_eps_(kNormEps),
        have_state_ngrams_(false),
        error_(false) {
    InitModel();
  }

  virtual ~NGramModel() = default;

  // Number of states in the LM fst
  StateId NumStates() const { return nstates_; }

  // Size of ngram model is the sum of the number of states and number of arcs
  int64_t GetSize() const {
    int64_t size = 0;
    for (StateId st = 0; st < nstates_; ++st)
      size += fst_.NumArcs(st) + 1;  // number of arcs + 1 state
    return size;
  }

  // Returns highest order
  int HiOrder() const { return hi_order_; }

  // Returns order of a given state
  // Order 1 is unigram, order 2 is bigram, and so on.
  int StateOrder(StateId state) const {
    if (state >= 0 && state < nstates_)
      return state_orders_[state];
    else
      return -1;
  }

  // Gets the state present in the model corresponding to the longest prefix of
  // the specified n-gram context.  If the longest prefix is the empty string,
  // then this method returns the unigram state. '0' signifies super-initial
  // 'word' in the ngram.
  StateId NGramState(const std::vector<Label> &ngram) const {
    StateId state = UnigramState();
    if (state < 0) state = GetFst().Start();
    fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);
    for (auto it = ngram.begin(); it != ngram.end(); ++it) {
      if (*it == 0) {
        state = fst_.Start();
        continue;
      }
      matcher.SetState(state);
      if (!matcher.Find(*it)) break;
      const Arc &arc = matcher.Value();
      state = arc.nextstate;
    }
    return state;
  }

  // Returns n-gram that must be read to reach 'state'. '0' signifies
  // super-initial 'word' in the ngram. Constructor argument 'state_ngrams' must
  // be true.
  const std::vector<Label> &StateNGram(StateId state) const {
    if (!have_state_ngrams_) {
      NGRAMERROR() << "NGramModel: state ngrams not available";
      return empty_label_vector_;
    }
    return state_ngrams_[state];
  }

  // Unigram state
  StateId UnigramState() const { return unigram_; }

  // Returns the unigram cost of requested symbol if found (inf otherwise)
  double GetSymbolUnigramCost(Label symbol) const {
    fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);
    StateId st = unigram_;
    if (st < 0) st = fst_.Start();
    matcher.SetState(st);
    if (matcher.Find(symbol)) {
      Arc arc = matcher.Value();
      return ScalarValue(arc.weight);
    } else {
      return ScalarValue(Arc::Weight::Zero());
    }
  }

  // Label of backoff transitions
  Label BackoffLabel() const { return backoff_label_; }

  // Find the backoff state for a given state st, and provide bocost if req'd
  StateId GetBackoff(StateId st, Weight *bocost) const {
    StateId backoff = -1;
    fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);
    matcher.SetState(st);
    if (matcher.Find(backoff_label_)) {
      for (; !matcher.Done(); matcher.Next()) {
        Arc arc = matcher.Value();
        if (arc.ilabel == fst::kNoLabel) continue;  // non-consuming symbol
        backoff = arc.nextstate;
        if (bocost != nullptr) bocost[0] = arc.weight;
      }
    }
    return backoff;
  }

  // Verifies LM topology is sane.
  bool CheckTopology() const {
    ascending_ngrams_ = 0;
    // Checks state topology
    for (StateId st = 0; st < nstates_; ++st)
      if (!CheckTopologyState(st)) return false;
    // All but start and unigram state should have a unique ascending ngram arc
    if (unigram_ != -1 && ascending_ngrams_ != nstates_ - 2) {
      VLOG(1) << "Incomplete # of ascending n-grams: " << ascending_ngrams_;
      return false;
    }
    return true;
  }

  // Iterates through all states and validate that they are fully normalized.
  bool CheckNormalization() const {
    if (Error()) return false;
    for (StateId st = 0; st < nstates_; ++st) {
      if (!CheckNormalizationState(st)) {
        VLOG(1) << "Failed normalization check at " << st;
        return false;
      }
    }
    return true;
  }

  // Calculate backoff cost from neglog sums of hi and low order arcs
  double CalculateBackoffCost(double hi_neglog_sum, double low_neglog_sum,
                              bool infinite_backoff = false) const {
    double nlog_backoff_num, nlog_backoff_denom;  // backoff cost and factors
    bool return_inf = CalculateBackoffFactors(
        hi_neglog_sum, low_neglog_sum, &nlog_backoff_num, &nlog_backoff_denom,
        infinite_backoff);
    if (return_inf) return kInfBackoff;  // backoff cost is 'infinite'
    return nlog_backoff_num - nlog_backoff_denom;
  }

  // Calculates the numerator and denominator for assigning backoff cost
  bool CalculateBackoffFactors(double hi_neglog_sum, double low_neglog_sum,
                               double *nlog_backoff_num,
                               double *nlog_backoff_denom,
                               bool infinite_backoff = false) const {
    double effective_zero = kNormEps * kFloatEps, effective_nlog_zero = 99.0;
    if (infinite_backoff && hi_neglog_sum <= kFloatEps)  // unsmoothed and p=1
      return true;
    if (hi_neglog_sum < effective_zero) hi_neglog_sum = effective_zero;
    if (low_neglog_sum < effective_zero) low_neglog_sum = effective_zero;
    if (low_neglog_sum <= 0 || hi_neglog_sum <= 0) return true;
    if (hi_neglog_sum > effective_nlog_zero) {
      *nlog_backoff_num = 0.0;
    } else {
      *nlog_backoff_num = NegLogDiff(0.0, hi_neglog_sum);
    }
    if (low_neglog_sum > effective_nlog_zero) {
      *nlog_backoff_denom = 0.0;
    } else {
      *nlog_backoff_denom = NegLogDiff(0.0, low_neglog_sum);
    }
    return false;
  }

  // Calculate marginal state probs.  By default, uses the product of
  // the order-ascending ngram transition probabilities. If 'stationary'
  // is true, instead computes the stationary distribution of the Markov
  // chain. Returns true on success.
  bool CalculateStateProbs(std::vector<double> *probs, bool stationary = false,
                           size_t maxiters = 10000) const {
    bool ret = true;
    if (stationary) {
      ret = StationaryStateProbs(probs, .999999, norm_eps_, maxiters);
    } else {
      NGramStateProbs(probs);
    }
    if (FST_FLAGS_v > 1) {
      for (size_t st = 0; st < probs->size(); ++st)
        std::cerr << "st: " << st << " log_prob: " << log((*probs)[st])
                  << std::endl;
    }
    return ret;
  }

  // fst::Fst const reference
  const fst::Fst<Arc> &GetFst() const { return fst_; }

  // Called at construction. If the model topology is mutated, this should
  // be re-called prior to any member function that depends on it.
  void InitModel() {
    using fst::kAcceptor;
    using fst::kIDeterministic;
    using fst::kILabelSorted;
    using fst::kNoLabel;
    using fst::kNoStateId;
    // unigram state is set to -1 for unigram models (in which case start
    // state is the unigram state, no need to store here)
    if (fst_.Start() == kNoLabel) {
      NGRAMERROR() << "NGramModel: Empty automaton";
      SetError();
      return;
    }
    uint64_t need_props = kAcceptor | kIDeterministic | kILabelSorted;
    uint64_t have_props = fst_.Properties(need_props, true);
    if (!(have_props & kAcceptor)) {
      NGRAMERROR() << "NGramModel: input not an acceptor";
      SetError();
      return;
    }
    if (!(have_props & kIDeterministic)) {
      NGRAMERROR() << "NGramModel: input not deterministic";
      SetError();
      return;
    }
    if (!(have_props & kILabelSorted)) {
      NGRAMERROR() << "NGramModel: input not label sorted";
      SetError();
      return;
    }

    if (!fst::CompatSymbols(fst_.InputSymbols(), fst_.OutputSymbols())) {
      NGRAMERROR() << "NGramModel: input and output symbol tables do not match";
      SetError();
      return;
    }

    nstates_ = CountStates(fst_);
    unigram_ = GetBackoff(fst_.Start(), nullptr);  // set the unigram state
    ComputeStateOrders();
    if (!CheckTopology()) {
      NGRAMERROR() << "NGramModel: bad ngram model topology";
      SetError();
      return;
    }
  }

  // Accessor function for the norm_eps_ parameter
  double NormEps() const { return norm_eps_; }

  // Calculates number of n-grams at state
  int NumNGrams(StateId st) {
    int num_ngrams = fst_.NumArcs(st);  // arcs are n-grams
    if (GetBackoff(st, nullptr) >= 0)   // except one arc, backoff arc
      num_ngrams--;
    if (ScalarValue(fst_.Final(st)) !=
        ScalarValue(Arc::Weight::Zero()))  // </s> n-gram
      num_ngrams++;
    return num_ngrams;
  }

  // Returns the cost assigned by model to an n-gram.  '0' signifies
  // super-initial and super-final 'words'.  If the n-gram begins with
  // '0', the computation begins at the start state and the initial
  // weight is applied; otherwise the computation begins at the unigram
  // state.  If the n-gram ends with '0' (distinct from from an initial
  // '0'), the final weight is applied.
  Weight GetNGramCost(const std::vector<Label> &ngram) const {
    if (ngram.empty()) return Weight::One();

    StateId st = ngram.front() == 0 || unigram_ < 0 ? fst_.Start() : unigram_;

    // p(<s>) = p(</s>)
    Weight cost = ngram.front() == 0 && unigram_ >= 0 ? fst_.Final(unigram_)
                                                      : Weight::One();

    fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);

    for (int n = 0; n < ngram.size(); ++n) {
      Label label = ngram[n];
      if (label == 0) {
        if (n == 0) continue;  // super-initial word
        if (n != ngram.size() - 1) {
          NGRAMERROR() << "end-of-string is not the super-final word";
          return Weight::Zero();
        }
        while (fst_.Final(st) == Weight::Zero()) {
          Weight bocost;
          st = GetBackoff(st, &bocost);
          if (st < 0) {
            return Weight::Zero();
          }
          cost = Times(cost, bocost);
        }
        cost = Times(cost, fst_.Final(st));
      } else {
        while (true) {
          matcher.SetState(st);
          if (matcher.Find(label)) {
            Arc arc = matcher.Value();
            st = arc.nextstate;
            cost = Times(cost, arc.weight);
            break;
          } else {
            Weight bocost;
            st = GetBackoff(st, &bocost);
            if (st < 0) {
              return Weight::Zero();
            }
            cost = Times(cost, bocost);
          }
        }
      }
    }

    return cost;
  }

  // Mimic a phi matcher: follow backoff links until final state found
  Weight FinalCostInModel(StateId mst, int *order) const {
    Weight cost = Arc::Weight::One();
    while (fst_.Final(mst) == Arc::Weight::Zero()) {
      fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);
      matcher.SetState(mst);
      if (matcher.Find(backoff_label_)) {
        for (; !matcher.Done(); matcher.Next()) {
          Arc arc = matcher.Value();
          if (arc.ilabel == backoff_label_) {
            mst = arc.nextstate;             // make current state backoff state
            cost = Times(cost, arc.weight);  // add in backoff cost
          }
        }
      } else {
        NGRAMERROR() << "NGramModel: No final cost in model: " << mst;
        return Arc::Weight::Zero();
      }
    }
    *order = state_orders_[mst];
    // TODO(vitalyk): take care of value call
    cost = Times(cost, fst_.Final(mst));
    return cost;
  }

  // Change data for a state that would normally be computed
  // by InitModel; this allows incremental updates
  void UpdateState(StateId st, int order, bool unigram_state,
                   const std::vector<Label> *ngram = 0) {
    if (have_state_ngrams_ && !ngram) {
      NGRAMERROR() << "NGramModel::UpdateState: no ngram provided";
      SetError();
      return;
    }
    if (state_orders_.size() < st) {
      NGRAMERROR() << "NGramModel::UpdateState: bad state: " << st;
      SetError();
      return;
    }
    if (order > hi_order_) hi_order_ = order;

    if (state_orders_.size() == st) {  // add state info
      state_orders_.push_back(order);
      if (ngram) state_ngrams_.push_back(*ngram);
      ++nstates_;
    } else {  // modifies state info
      state_orders_[st] = order;
      if (ngram) state_ngrams_.push_back(*ngram);
    }

    if (unigram_state) unigram_ = nstates_;
  }

  // Returns a scalar value associated with a weight
  static double ScalarValue(Weight w);

  // Returns a weight that represents unit count for this model
  static Weight UnitCount();

  // Returns a factor used to scale backoff mass in interpolated models
  static double FactorValue(Weight w);

  // Returns the final for state st
  Weight GetFinalWeight(StateId st) const { return fst_.Final(st); }

  // Returns the backoff cost for state st
  Weight GetBackoffCost(StateId st) const {
    Weight bocost;
    StateId bo = GetBackoff(st, &bocost);
    if (bo < 0)  // if no backoff arc found
      bocost = Arc::Weight::Zero();
    return bocost;
  }

  // Estimate total unigram count based on probabilities in unigram state
  // The difference between two smallest probs should be 1/N, return reciprocal
  double EstimateTotalUnigramCount() const {
    StateId st = UnigramState();
    bool first = true;
    double max = fst::LogArc::Weight::Zero().Value(), nextmax = max;
    if (st < 0) st = GetFst().Start();  // if model unigram, use Start()
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(GetFst(), st);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == BackoffLabel()) continue;
      if (first || ScalarValue(arc.weight) > max) {
        // maximum negative log prob case
        nextmax = max;  // keep both max and nextmax (to calculate diff)
        max = ScalarValue(arc.weight);
        first = false;
      } else if (ScalarValue(arc.weight) < max &&
                 ScalarValue(arc.weight) > nextmax) {
        nextmax = ScalarValue(arc.weight);
      }
    }
    if (nextmax == fst::LogArc::Weight::Zero().Value()) return exp(max);
    return exp(NegLogDiff(nextmax, max));
  }

  // Returns true if model in a bad state/not a proper LM.
  bool Error() const { return error_; }

 protected:
  void SetError() { error_ = true; }

  // Shadows in this class to catch errors.
  double NegLogDiff(double a, double b) const {
    return ngram::NegLogDiff(a, b, &error_);
  }

  // Fills a vector with the counts of each state, based on prefix count
  void FillStateCounts(std::vector<double> *state_counts) {
    for (int i = 0; i < nstates_; i++)
      state_counts->push_back(ScalarValue(Arc::Weight::Zero()));
    WalkStatesForCount(state_counts);
  }

  // Collect backoff arc weights in a vector
  bool FillBackoffArcWeights(StateId st, StateId bo,
                             std::vector<double> *bo_arc_weight) const {
    fst::Matcher<fst::Fst<Arc>> matcher(
        fst_, fst::MATCH_INPUT);  // for querying backoff
    matcher.SetState(bo);
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == backoff_label_) continue;
      if (matcher.Find(arc.ilabel)) {
        Arc barc = matcher.Value();
        // Note that we allow to scale the backoff weight by
        // a value that depends on the weight of the ngram.
        // For instance, the fractional count model mixes in a fraction
        // of lower order mass proportional to the frequency of event
        // that ngram occurs zero times. So for this model we scale
        // backoff weights by these frequences.
        // For all the other models this scaling factor defaults to 0.0
        // (unity in log semiring).
        bo_arc_weight->push_back(ScalarValue(barc.weight) +
                                 FactorValue(arc.weight));
      } else {
        NGRAMERROR() << "NGramModel: lower order arc missing: " << st;
        return false;
      }
    }
    return true;
  }

  // Uses iterator in place of matcher for arc iterators; allows
  // getting Position(). NB: begins search from current position.
  bool FindArc(fst::ArcIterator<fst::Fst<Arc>> *biter,
               Label label) const {
    while (!biter->Done()) {  // scan through arcs
      Arc barc = biter->Value();
      if (barc.ilabel == label)
        return true;                 // if label matches, true
      else if (barc.ilabel < label)  // if less than value, go to next
        biter->Next();
      else
        return false;  // otherwise no match
    }
    return false;  // no match found
  }

  // Finds the arc weight associated with a label at a state
  Weight FindArcWeight(StateId st, Label label) const {
    Weight cost = Arc::Weight::Zero();
    fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);
    matcher.SetState(st);
    if (matcher.Find(label)) {
      Arc arc = matcher.Value();
      cost = arc.weight;
    }
    return cost;
  }

  // Mimic a phi matcher: follow backoff arcs until label found or no backoff
  bool FindNGramInModel(StateId *mst, int *order, Label label,
                        double *cost) const {
    if (label < 0) return false;
    StateId currstate = *mst;
    *cost = 0;
    *mst = -1;
    while (*mst < 0) {
      fst::Matcher<fst::Fst<Arc>> matcher(fst_, fst::MATCH_INPUT);
      matcher.SetState(currstate);
      if (matcher.Find(label)) {  // arc found out of current state
        Arc arc = matcher.Value();
        *order = state_orders_[currstate];
        *mst = arc.nextstate;  // assign destination as new model state
        *cost += ScalarValue(arc.weight);         // add cost to total
      } else if (matcher.Find(backoff_label_)) {  // follow backoff arc
        currstate = -1;
        for (; !matcher.Done(); matcher.Next()) {
          Arc arc = matcher.Value();
          if (arc.ilabel == backoff_label_) {
            currstate = arc.nextstate;  // make current state backoff state
            *cost += ScalarValue(arc.weight);  // add in backoff cost
          }
        }
        if (currstate < 0) return false;
      } else {
        return false;  // Found label in symbol list, but not in model
      }
    }
    return true;
  }

  // Sum final + arc probs out of state and for same transitions out of backoff
  bool CalcBONegLogSums(StateId st, double *hi_neglog_sum,
                        double *low_neglog_sum, bool infinite_backoff = false,
                        bool unigram = false) const {
    StateId bo = GetBackoff(st, nullptr);
    if (bo < 0 && !unigram) return false;  // only calc for states that backoff
    *low_neglog_sum = *hi_neglog_sum =     // final costs initialize the sum
        ScalarValue(fst_.Final(st));
    // if st is final
    if (bo >= 0 && *hi_neglog_sum != ScalarValue(Arc::Weight::Zero()))
      // re-initialize lower sum
      *low_neglog_sum = ScalarValue(fst_.Final(bo));
    CalcArcNegLogSums(st, bo, hi_neglog_sum, low_neglog_sum, infinite_backoff);
    return true;
  }

  // Prints state ngram to a stream
  bool PrintStateNGram(StateId st, std::ostream &ostrm = std::cerr) const {
    ostrm << "state: " << st << " order: " << state_orders_[st] << " ngram: ";
    for (int i = 0; i < state_ngrams_[st].size(); ++i)
      ostrm << state_ngrams_[st][i] << " ";
    ostrm << "\n";
    return true;
  }

  // Modifies n-gram weights according to printing parameters
  static double WeightRep(double wt, bool neglogs, bool intcnts) {
    if (!neglogs || intcnts) wt = exp(-wt);
    if (intcnts) wt = round(wt);
    return wt;
  }

 private:
  // Iterate through arcs, accumulate neglog probs from arcs and their backoffs
  bool CalcArcNegLogSums(StateId st, StateId bo, double *hi_sum,
                         double *low_sum, bool infinite_backoff = false) const {
    // correction values for Kahan summation
    double KahanVal1 = 0, KahanVal2 = 0;
    double init_low = *low_sum;
    fst::Matcher<fst::Fst<Arc>> matcher(
        fst_, fst::MATCH_INPUT);  // for querying backoff
    if (bo >= 0) matcher.SetState(bo);
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == backoff_label_) continue;
      if (bo < 0 || matcher.Find(arc.ilabel)) {
        if (bo >= 0) {
          Arc barc = matcher.Value();
          *low_sum =  // sum of lower order probs of the same labels
              NegLogSum(*low_sum, ScalarValue(barc.weight), &KahanVal2);
        }
        *hi_sum =  // sum of higher order probs
            NegLogSum(*hi_sum, ScalarValue(arc.weight), &KahanVal1);
      } else {
        NGRAMERROR() << "NGramModel: No arc label match in backoff state: "
                     << st;
        return false;
      }
    }
    if (bo >= 0 && infinite_backoff && *low_sum == 0.0)  // ok for unsmoothed
      return true;
    if (bo >= 0 && *low_sum <= 0.0) {
      VLOG(2) << "lower order sum less than zero: " << st << " " << *low_sum;
      double start_low = ScalarValue(Arc::Weight::Zero());
      if (init_low == start_low) start_low = ScalarValue(fst_.Final(bo));
      *low_sum = CalcBruteLowSum(st, bo, start_low);
      VLOG(2) << "new lower order sum: " << st << " " << *low_sum;
    }
    return true;
  }

  // Iterate through arcs, accumulate neglog probs from arcs and their backoffs
  // Used in case the more efficient method fails to produce a sane value
  double CalcBruteLowSum(StateId st, StateId bo, double start_low) const {
    double low_sum = start_low, KahanVal = 0;
    fst::Matcher<fst::Fst<Arc>> matcher(
        fst_, fst::MATCH_INPUT);  // for querying backoff
    matcher.SetState(bo);
    fst::ArcIterator<fst::Fst<Arc>> biter(fst_, bo);
    Arc barc;
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == backoff_label_) continue;
      barc = biter.Value();
      while (!biter.Done() && barc.ilabel < arc.ilabel) {  // linear scan
        if (barc.ilabel != backoff_label_)
          low_sum =  // sum of lower order probs of different labels
              NegLogSum(low_sum, ScalarValue(barc.weight), &KahanVal);
        biter.Next();
        if (!biter.Done()) barc = biter.Value();
      }
      if (!biter.Done() && barc.ilabel == arc.ilabel) {
        biter.Next();
        if (!biter.Done()) barc = biter.Value();
      }
      if (biter.Done()) break;
    }
    while (!biter.Done()) {  // linear scan
      if (barc.ilabel != backoff_label_)
        low_sum =  // sum of lower order probs of different labels
            NegLogSum(low_sum, ScalarValue(barc.weight), &KahanVal);
      biter.Next();
      if (!biter.Done()) barc = biter.Value();
    }
    return NegLogDiff(0.0, low_sum);
  }

  // Traverse n-gram fst and record each state's n-gram order, return highest
  void ComputeStateOrders() {
    state_orders_.clear();
    state_orders_.resize(nstates_, -1);

    if (have_state_ngrams_) {
      state_ngrams_.clear();
      state_ngrams_.resize(nstates_);
    }

    hi_order_ = 1;  // calculate highest order in the model
    std::deque<StateId> state_queue;
    if (unigram_ != fst::kNoStateId) {
      state_orders_[unigram_] = 1;
      state_queue.push_back(unigram_);
      state_orders_[fst_.Start()] = hi_order_ = 2;
      state_queue.push_back(fst_.Start());
      if (have_state_ngrams_)
        state_ngrams_[fst_.Start()].push_back(0);  // initial context
    } else {
      state_orders_[fst_.Start()] = 1;
      state_queue.push_back(fst_.Start());
    }

    while (!state_queue.empty()) {
      StateId state = state_queue.front();
      state_queue.pop_front();
      for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, state);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (state_orders_[arc.nextstate] == -1) {
          state_orders_[arc.nextstate] = state_orders_[state] + 1;
          if (have_state_ngrams_) {
            state_ngrams_[arc.nextstate] = state_ngrams_[state];
            state_ngrams_[arc.nextstate].push_back(arc.ilabel);
          }
          if (state_orders_[state] >= hi_order_)
            hi_order_ = state_orders_[state] + 1;
          state_queue.push_back(arc.nextstate);
        }
      }
    }
  }

  // Ensure correct n-gram topology for a given state.
  bool CheckTopologyState(StateId st) const {
    if (unigram_ == -1) {  // unigram model
      if (fst_.Final(fst_.Start()) == Arc::Weight::Zero()) {
        VLOG(1) << "CheckTopology: bad final weight for start state";
        return false;
      } else {
        return true;
      }
    }

    StateId bos = GetBackoff(st, nullptr);
    fst::Matcher<fst::Fst<Arc>> matcher(
        fst_, fst::MATCH_INPUT);  // for querying backoff

    if (st == unigram_) {  // unigram state
      if (fst_.Final(unigram_) == Arc::Weight::Zero()) {
        VLOG(1) << "CheckTopology: bad final weight for unigram state: "
                << unigram_;
        return false;
      } else if (have_state_ngrams_ && !state_ngrams_[unigram_].empty()) {
        VLOG(1) << "CheckTopology: bad unigram state: " << unigram_;
        return false;
      }
    } else {  // non-unigram state
      if (bos == -1) {
        VLOG(1) << "CheckTopology: no backoff state: " << st;
        return false;
      }

      if (fst_.Final(st) != Arc::Weight::Zero() &&
          fst_.Final(bos) == Arc::Weight::Zero()) {
        VLOG(1) << "CheckTopology: bad final weight for backoff state: " << st;
        return false;
      }

      if (StateOrder(st) != StateOrder(bos) + 1) {
        VLOG(1) << "CheckTopology: bad backoff arc from: " << st
                << " with order: " << StateOrder(st) << " to state: " << bos
                << " with order: " << StateOrder(bos);
        return false;
      }
      matcher.SetState(bos);
    }

    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();

      if (StateOrder(st) < StateOrder(arc.nextstate)) ++ascending_ngrams_;

      if (have_state_ngrams_ && !CheckStateNGrams(st, arc)) {
        VLOG(1) << "CheckTopology: inconsistent n-gram states: " << st << " -- "
                << arc.ilabel << "/" << arc.weight << " -> " << arc.nextstate;
        return false;
      }

      if (st != unigram_) {
        if (arc.ilabel == backoff_label_) continue;
        if (!matcher.Find(arc.ilabel)) {
          VLOG(1) << "CheckTopology: unmatched arc at backoff state: "
                  << arc.ilabel << "/" << arc.weight << " for state: " << st;
          return false;
        }
      }
    }
    return true;
  }

  // Checks state ngrams for consistency
  bool CheckStateNGrams(StateId st, const Arc &arc) const {
    std::vector<Label> state_ngram;
    bool boa = arc.ilabel == backoff_label_;

    int j = state_orders_[st] - state_orders_[arc.nextstate] + (boa ? 0 : 1);
    if (j < 0) return false;

    for (int i = j; i < state_ngrams_[st].size(); ++i)
      state_ngram.push_back(state_ngrams_[st][i]);
    if (!boa && j <= state_ngrams_[st].size())
      state_ngram.push_back(arc.ilabel);

    return state_ngram == state_ngrams_[arc.nextstate];
  }

  // Ensure normalization for a given state to error epsilon
  // sum of state probs + exp(-backoff_cost) - sum of arc backoff probs = 1
  bool CheckNormalizationState(StateId st) const {
    double Norm, Norm1;
    Weight bocost = Weight::NoWeight();
    StateId bo = GetBackoff(st, &bocost);
    // final costs initialize the sum
    Norm = Norm1 = ScalarValue(fst_.Final(st));
    if (bo >= 0 && Norm != ScalarValue(Arc::Weight::Zero()))  // if st is final
      Norm1 = ScalarValue(fst_.Final(bo));  // re-initialize lower sum
    if (!CalcArcNegLogSums(st, bo, &Norm, &Norm1,
                           (ScalarValue(bocost) == kInfBackoff))) {
      return false;
    }
    return EvaluateNormalization(st, bo, ScalarValue(bocost), Norm, Norm1);
  }

  // For accumulated negative log probabilities, test for normalization
  bool EvaluateNormalization(StateId st, StateId bo, double bocost, double norm,
                             double norm1) const {
    double newnorm = norm;
    if (bo >= 0) {
      newnorm = NegLogSum(norm, bocost);
      if (newnorm < norm1 + bocost)
        newnorm = NegLogDiff(newnorm, norm1 + bocost);
      else
        newnorm = NegLogDiff(norm1 + bocost, newnorm);
    }
    // NOTE: can we automatically derive an appropriate epsilon?
    if (fabs(newnorm) > norm_eps_ &&  // not normalized
        (bo < 0 || !ReevaluateNormalization(st, bocost, norm, norm1))) {
      VLOG(2) << "State ID: " << st << "; " << fst_.NumArcs(st) << " arcs;"
              << "  -log(sum(P)) = " << newnorm << ", should be 0";
      VLOG(2) << norm << " " << norm1;
      return false;
    }
    return true;
  }

  // For accumulated negative log probabilities, a 2nd test for normalization
  // Intended for states with very high magnitude backoff cost, which makes
  // previous test unreliable
  bool ReevaluateNormalization(StateId st, double bocost, double norm,
                               double norm1) const {
    double newalpha = CalculateBackoffCost(norm, norm1);
    // NOTE: can we automatically derive an appropriate epsilon?
    VLOG(2) << "Required re-evaluation of normalization: state " << st << " "
            << norm << " " << norm1 << " " << newalpha << " " << norm_eps_;
    if (fabs(newalpha - bocost) > norm_eps_) return false;
    return true;
  }

  // Collects prefix counts for arcs out of a specific state
  void CollectPrefixCounts(std::vector<double> *state_counts,
                           StateId st) const {
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != backoff_label_ &&  // only counting non-backoff arcs
          state_orders_[st] < state_orders_[arc.nextstate]) {  // that + order
        (*state_counts)[arc.nextstate] = ScalarValue(arc.weight);
        CollectPrefixCounts(state_counts, arc.nextstate);
      }
    }
  }

  // Walks model automaton to collect prefix counts for each state
  void WalkStatesForCount(std::vector<double> *state_counts) const {
    if (unigram_ != -1) {
      (*state_counts)[fst_.Start()] = ScalarValue(fst_.Final(unigram_));
      CollectPrefixCounts(state_counts, unigram_);
    }
    CollectPrefixCounts(state_counts, fst_.Start());
  }

  // checks non-negativity of weight and uses +;
  // Test to see if model came from pre-summing a mixture
  // Should have: backoff weights > 0; higher order always higher prob (summed)
  bool MixtureConsistent() const {
    fst::Matcher<fst::Fst<Arc>> matcher(
        fst_, fst::MATCH_INPUT);  // for querying backoff
    for (StateId st = 0; st < nstates_; ++st) {
      Weight bocost;
      StateId bo = GetBackoff(st, &bocost);
      if (bo >= 0) {     // if bigram or higher order
        if (bocost < 0)  // Backoff cost > 0 (can't happen with mixture)
          return false;
        matcher.SetState(bo);
        for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st);
             !aiter.Done(); aiter.Next()) {
          Arc arc = aiter.Value();
          if (arc.ilabel == backoff_label_) {
            continue;
          }
          if (matcher.Find(arc.ilabel)) {
            Arc barc = matcher.Value();
            if (ScalarValue(arc.weight) >
                ScalarValue(barc.weight) + ScalarValue(bocost)) {
              return false;  // L P + (1-L) P' < (1-L) P' (can't happen w/mix)
            }
          } else {
            NGRAMERROR() << "NGramModel: lower order arc missing: " << st;
            SetError();
            return false;
          }
        }
        if (ScalarValue(fst_.Final(st)) != ScalarValue(Arc::Weight::Zero()) &&
            ScalarValue(fst_.Final(st)) >
                SclarValue(fst_.Final(bo)) + ScalarValue(bocost))
          return false;  // final cost doesn't sum
      }
    }
    return true;
  }

  // At a given state, calculate the marginal prob p(h) based on
  // the smoothed, order-ascending n-gram transition probabilities.
  void NGramStateProb(StateId st, std::vector<double> *probs) const {
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == backoff_label_) continue;
      if (state_orders_[arc.nextstate] > state_orders_[st]) {
        (*probs)[arc.nextstate] = (*probs)[st] * exp(-ScalarValue(arc.weight));
        NGramStateProb(arc.nextstate, probs);
      }
    }
  }

  // Calculate marginal state probs as the product of the smoothed,
  // order-ascending ngram transition probablities: p(abc) =
  // p(a)p(b|a)p(c|ba) (odd w/KN)
  void NGramStateProbs(std::vector<double> *probs, bool norm = false) const {
    probs->clear();
    probs->resize(nstates_, 0.0);
    if (unigram_ < 0) {
      // p(unigram state) = 1
      (*probs)[fst_.Start()] = 1.0;
    } else {
      // p(unigram state) = 1
      (*probs)[unigram_] = 1.0;
      NGramStateProb(unigram_, probs);
      // p(<s>) = p(</s>)
      (*probs)[fst_.Start()] = exp(-ScalarValue(fst_.Final(unigram_)));
    }
    NGramStateProb(fst_.Start(), probs);

    if (norm) {  // Normalize result, as a starting point for the power method
      double sum = 0.0;
      for (size_t st = 0; st < probs->size(); ++st) sum += (*probs)[st];
      for (size_t st = 0; st < probs->size(); ++st) (*probs)[st] /= sum;
    }
  }

  // Exponentiates the weights
  // At a given state, calculate one step of the power method
  // for the stationary distribution of the closure of the
  // LM with re-entry probability 'alpha'.
  void StationaryStateProb(StateId st, std::vector<double> *init_probs,
                           std::vector<double> *probs, double alpha) const {
    fst::Matcher<fst::Fst<Arc>> matcher(
        fst_, fst::MATCH_INPUT);  // for querying backoff
    Weight bocost;
    StateId bo = GetBackoff(st, &bocost);
    if (bo != -1) {
      // Treats backoff like an epsilon transition
      matcher.SetState(bo);
      (*init_probs)[bo] += (*init_probs)[st] * exp(-ScalarValue(bocost));
    }

    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst_, st); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == backoff_label_) continue;
      (*probs)[arc.nextstate] +=
          (*init_probs)[st] * exp(-ScalarValue(arc.weight));
      if (bo != -1 && matcher.Find(arc.ilabel)) {
        // Subtracts corrective weight for backed-off arc
        const Arc &barc = matcher.Value();
        (*probs)[barc.nextstate] -=
            (*init_probs)[st] *
            exp(-ScalarValue(barc.weight) - ScalarValue(bocost));
      }
    }

    if (ScalarValue(fst_.Final(st)) != ScalarValue(Weight::Zero())) {
      (*probs)[fst_.Start()] +=
          (*init_probs)[st] * exp(-ScalarValue(fst_.Final(st))) * alpha;
      if (bo != -1) {
        // Subtracts corrective weight for backed-off superfinal arc
        (*probs)[fst_.Start()] -=
            (*init_probs)[st] *
            exp(-ScalarValue(fst_.Final(bo)) - ScalarValue(bocost)) * alpha;
      }
    }
  }

  // Calculate marginal state probs as the stationary distribution
  // of the Markov chain consisting of the closure of the LM
  // with re-entry probability 'alpha'. The convergence is controlled
  // by 'converge_eps' and the number of iterations by 'maxiters'
  // Returns true on convergence.
  bool StationaryStateProbs(std::vector<double> *probs, double alpha,
                            double converge_eps, size_t maxiters) const {
    std::vector<double> init_probs, last_probs;
    // Initialize based on ngram transition probabilities
    NGramStateProbs(&init_probs, true);
    last_probs = init_probs;

    size_t changed;
    size_t iters = 0;
    do {
      probs->clear();
      probs->resize(nstates_, 0.0);
      for (int order = hi_order_; order > 0; --order) {
        for (size_t st = 0; st < nstates_; ++st) {
          if (state_orders_[st] == order)
            StationaryStateProb(st, &init_probs, probs, alpha);
        }
      }

      changed = 0;
      for (size_t st = 0; st < nstates_; ++st) {
        if (fabs((*probs)[st] - last_probs[st]) > converge_eps * last_probs[st])
          ++changed;
        last_probs[st] = init_probs[st] = (*probs)[st];
      }
      VLOG(2) << "NGramModel::StationaryStateProbs: state probs changed: "
              << changed;
      if (++iters > maxiters) return false;
    } while (changed > 0);
    return true;
  }

  const fst::Fst<Arc> &fst_;
  StateId unigram_;                // unigram state
  Label backoff_label_;            // label of backoff transitions
  StateId nstates_;                // number of states in LM
  int hi_order_;                   // highest order in the model
  double norm_eps_;                // epsilon diff allowed to ensure normalized
  std::vector<int> state_orders_;  // order of each state
  bool have_state_ngrams_;         // compute and store state n-gram info
  mutable size_t ascending_ngrams_;  // # of n-gram arcs that increase order
  std::vector<std::vector<Label>>
      state_ngrams_;  // n-gram always read to reach state
  const std::vector<Label> empty_label_vector_;
  mutable bool error_;

  NGramModel(const NGramModel &) = delete;
  NGramModel &operator=(const NGramModel &) = delete;
};

template <typename T>
double NGramModel<T>::ScalarValue(NGramModel<T>::Weight w) {
  return w.Value();
}

template <>
double inline NGramModel<HistogramArc>::ScalarValue(
    NGramModel<HistogramArc>::Weight w) {
  return w.Value(0).Value();
}

template <typename Arc>
typename Arc::Weight NGramModel<Arc>::UnitCount() {
  return Arc::Weight::One();
}

template <>
inline typename HistogramArc::Weight NGramModel<HistogramArc>::UnitCount() {
  std::array<fst::StdArc::Weight, kHistogramBins> weights;
  weights.fill(fst::StdArc::Weight::Zero());
  if (kHistogramBins > 0) weights[0] = fst::StdArc::Weight::One();
  if (kHistogramBins > 2) weights[2] = fst::StdArc::Weight::One();
  static const fst::PowerWeight<fst::StdArc::Weight, kHistogramBins>
      one(weights.begin(), weights.end());
  return one;
}

template <typename T>
inline double NGramModel<T>::FactorValue(NGramModel<T>::Weight w) {
  return 0.0;
}

template <>
inline double NGramModel<HistogramArc>::FactorValue(
    NGramModel<HistogramArc>::Weight w) {
  return w.Value(1).Value();
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_MODEL_H_
