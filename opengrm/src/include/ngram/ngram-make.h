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
// NGram model class for making a model from raw counts or histograms.

#ifndef NGRAM_NGRAM_MAKE_H_
#define NGRAM_NGRAM_MAKE_H_

#include <cstdint>
#include <string>
#include <vector>

#include <fst/expanded-fst.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/script/fst-class.h>
#include <ngram/hist-arc.h>
#include <ngram/ngram-model.h>
#include <ngram/ngram-mutable-model.h>
#include <ngram/util.h>

namespace ngram {

template <class Arc>
class NGramMake : public NGramMutableModel<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  using NGramModel<Arc>::Error;
  using NGramMutableModel<Arc>::GetFst;
  using NGramMutableModel<Arc>::GetMutableFst;
  using NGramMutableModel<Arc>::GetExpandedFst;
  using NGramMutableModel<Arc>::BackoffLabel;
  using NGramMutableModel<Arc>::InitModel;
  using NGramMutableModel<Arc>::CheckNormalization;
  using NGramMutableModel<Arc>::HiOrder;
  using NGramMutableModel<Arc>::RecalcBackoff;
  using NGramMutableModel<Arc>::GetBackoff;
  using NGramMutableModel<Arc>::StateOrder;
  using NGramMutableModel<Arc>::FillBackoffArcWeights;
  using NGramMutableModel<Arc>::ScaleStateWeight;
  using NGramMutableModel<Arc>::ScalarValue;
  using NGramMutableModel<Arc>::SetScalarValue;
  using NGramMutableModel<Arc>::NGramMutableModel;
  using NGramMutableModel<Arc>::FactorValue;

  // Construct NGramMake object, consisting of the FST and some
  // information about the states under the assumption that the FST is a model.
  // Ownership of the FST is retained by the caller.
  NGramMake(fst::MutableFst<Arc> *infst, bool backoff,
            Label backoff_label = 0, double norm_eps = kNormEps,
            bool check_consistency = false, bool infinite_backoff = false)
      : NGramMutableModel<Arc>(infst, backoff_label, norm_eps,
                               check_consistency, infinite_backoff),
        backoff_(backoff) {}

  ~NGramMake() override = default;

  // Normalizes n-gram counts and smoothes to create an n-gram model.
  // Returns true on success and false on failure.
  virtual bool MakeNGramModel() {
    if (Error()) return false;
    for (StateId st = 0; st < GetExpandedFst().NumStates(); ++st) {
      has_all_ngrams_.push_back(false);
    }
    for (int order = 1; order <= HiOrder(); ++order) {
      for (StateId st = 0; st < GetExpandedFst().NumStates(); ++st) {
        if (StateOrder(st) == order) {
          // Smoothes all states in the model, in ascending state-order order.
          SmoothState(st);
          if (Error()) {
            NGRAMERROR() << "NGramMake: Error in smoothing state " << st;
            return false;
          }
        }
      }
    }
    InitModel();                // Recalculate state info
    if (Error()) {
      NGRAMERROR() << "NGramMake: Error in recalculating state info";
      return false;
    } else {
      RecalcBackoff();            // Recalculate the backoff costs
      if (!CheckNormalization()) {  // Ensures model is fully normalized
        NGRAMERROR() << "NGramMake: Final model not fully normalized";
        return false;
      }
    }
    return true;
  }

 protected:
  // Return negative log discounted count for provided negative log count
  // Need to override if some discounting is done during smoothing
  // Default can be used by non-discounting methods, e.g., Witten-Bell
  virtual double GetDiscount(Weight nlog_count, int order) {
    return ScalarValue(nlog_count);
  }

  // Additional count mass at state if nothing reserved via smoothing method
  // Override if method requires less or more; usmoothed should be zero
  // Default can be used by most methods
  virtual double EpsilonMassIfNoneReserved() const { return 1.0; }

  // Return high order count mass (sum of discounted counts)
  // Need to override if high order mass is not defined by discounts
  // Default can be used by discounting methods, e.g., Katz or Absolute Disc.
  virtual double CalculateHiOrderMass(const std::vector<double> &discounts,
                                      double nlog_count) const {
    double discount_norm = discounts[0];          // discounted count of </s>
    double KahanVal = 0;                          // Value for Kahan summation
    for (int i = 1; i < discounts.size(); ++i) {  // Sum discount counts
      discount_norm = NegLogSum(discount_norm, discounts[i], &KahanVal);
    }
    return discount_norm;
  }

  // Return normalization constant given the count and state
  // Need to override if normalization constant is not just the count
  // Default can be used if the normalizing constant is just count
  virtual double CalculateTotalMass(double nlog_count, StateId st) {
    return nlog_count;
  }

 private:
  // Normalize and smooth states, using parameterized smoothing method
  void SmoothState(StateId st) {
    std::vector<double> discounts;  // collect discounted counts for later use.
    double nlog_count_sum = CollectDiscounts(st, &discounts), nlog_stored_sum;
    Weight nlog_stored_sum_weight;
    if (GetBackoff(st, &nlog_stored_sum_weight) < 0) {
      has_all_ngrams_[st] = true;
      ScaleStateWeight(st, -nlog_count_sum);  // no backoff arc, unsmoothed
    } else {
      nlog_stored_sum = ScalarValue(nlog_stored_sum_weight);
      // Calculate total count mass and higher order count mass to normalize
      double total_mass = CalculateTotalMass(nlog_stored_sum, st);
      double hi_order_mass = CalculateHiOrderMass(discounts, nlog_stored_sum);
      has_all_ngrams_[st] = HasAllArcsInBackoff(st);
      if (has_all_ngrams_[st] && total_mass < hi_order_mass) {
        discounts[0] =
            NegLogSum(discounts[0], NegLogDiff(total_mass, hi_order_mass));
        hi_order_mass = total_mass;
      }
      double low_order_mass;
      if (total_mass >= hi_order_mass &&  // if approx equal
          fabs(total_mass - hi_order_mass) < kFloatEps)
        total_mass = hi_order_mass;  // then make equal, for later testing
      if (has_all_ngrams_[st] ||
          (total_mass == hi_order_mass && EpsilonMassIfNoneReserved() <= 0)) {
        low_order_mass = kInfBackoff;
      } else {
        if (total_mass == hi_order_mass) {  // if no mass reserved, add eps
          total_mass = -log(exp(-total_mass) + EpsilonMassIfNoneReserved());
        }
        low_order_mass = NegLogDiff(total_mass, hi_order_mass);
      }
      NormalizeStateArcs(st, total_mass, low_order_mass - total_mass,
                         discounts);
    }
  }

  // Calculate smoothed value for arc out of a state
  Weight SmoothVal(double discount_cnt, double norm, double neglog_bo_prob,
                   double backoff_weight) {
    double value = discount_cnt - norm;
    if (!backoff_) {
      double mixvalue = neglog_bo_prob + backoff_weight;
      value = NegLogSum(value, mixvalue);
    }
    Weight w = Weight::Zero();
    SetScalarValue(&w, value);
    return w;
  }

  // Checks to see if all n-grams already represented at state
  bool HasAllArcsInBackoff(StateId st) {
    StateId bo = GetBackoff(st, nullptr);
    if (!has_all_ngrams_[bo]) return false;  // backoff state doesn't have all
    size_t starcs = GetFst().NumArcs(st), boarcs = GetFst().NumArcs(bo);
    if (boarcs > starcs) return false;  // arcs at backoff not in current state
    if (ScalarValue(GetFst().Final(bo)) !=
        ScalarValue(Arc::Weight::Zero()))  // count </s> symbol
      boarcs++;
    if (GetBackoff(bo, nullptr) >= 0) boarcs--;  // don't count backoff arc
    if (ScalarValue(GetFst().Final(st)) !=
        ScalarValue(Arc::Weight::Zero()))  // count </s> symbol
      starcs++;
    starcs--;  // don't count backoff arc
    if (boarcs == starcs) return true;
    return false;
  }

  // Calculate smoothed values for all arcs leaving a state
  void NormalizeStateArcs(StateId st, double norm, double neglog_bo_prob,
                          const std::vector<double> &discounts) {
    StateId bo = GetBackoff(st, nullptr);
    if (ScalarValue(GetFst().Final(st)) != ScalarValue(Arc::Weight::Zero())) {
      GetMutableFst()->SetFinal(st,
                                SmoothVal(discounts[0], norm, neglog_bo_prob,
                                          ScalarValue(GetFst().Final(bo)) +
                                              FactorValue(GetFst().Final(st))));
    }
    std::vector<double> bo_arc_weight;
    // fill backoff weight vector
    if (!FillBackoffArcWeights(st, bo, &bo_arc_weight)) {
      NGRAMERROR() << "NGramMake: could not fill backoff arc weights";
      return;
    }
    int arc_counter = 0;     // index into backoff weights
    int discount_index = 1;  // index into discounts (off by one, for </s>)
    for (fst::MutableArcIterator<fst::MutableFst<Arc>> aiter(
             GetMutableFst(), st);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != BackoffLabel()) {  // backoff weights calculated later
        arc.weight = SmoothVal(discounts[discount_index++], norm,
                               neglog_bo_prob, bo_arc_weight[arc_counter++]);
        aiter.SetValue(arc);
      }
    }
  }

  // Collects discounted counts into vector, and returns -log(sum(counts))
  // If no discounting, vector collects undiscounted counts
  double CollectDiscounts(StateId st, std::vector<double> *discounts) {
    double nlog_count_sum = ScalarValue(GetFst().Final(st));
    double KahanVal = 0.0;
    int order = StateOrder(st) - 1;  // for retrieving discount parameters
    discounts->push_back(GetDiscount(GetFst().Final(st), order));
    for (fst::ArcIterator<fst::ExpandedFst<Arc>> aiter(GetExpandedFst(),
                                                               st);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != BackoffLabel()) {  // skip backoff arc
        nlog_count_sum =
            NegLogSum(nlog_count_sum, ScalarValue(arc.weight), &KahanVal);
        discounts->push_back(GetDiscount(arc.weight, order));
      }
    }
    return nlog_count_sum;
  }

  std::vector<bool> has_all_ngrams_;
  bool backoff_;  // whether to make the model as backoff or mixture model
};

// Makes models from NGram count FSTs with StdArc counts.
bool NGramMakeModel(fst::StdMutableFst *fst, const std::string &method,
                    const fst::StdFst *ccfst = nullptr,
                    bool backoff = false, bool interpolate = false,
                    int64_t bins = -1, double witten_bell_k = 1,
                    double discount_D = -1.0, int64_t backoff_label = 0,
                    double norm_eps = kNormEps, bool check_consistency = false);

// The same, but uses scripting FSTs.
bool NGramMakeModel(fst::script::MutableFstClass *fst,
                    const std::string &method,
                    const fst::script::FstClass *ccfst = nullptr,
                    bool backoff = false, bool interpolate = false,
                    int64_t bins = -1, double witten_bell_k = 1,
                    double discount_D = -1.0, int64_t backoff_label = 0,
                    double norm_eps = kNormEps, bool check_consistency = false);

// Makes models from NGram count FSTs with HistogramArc counts.
bool NGramMakeHistModel(fst::MutableFst<ngram::HistogramArc> *hist_fst,
                        fst::StdMutableFst *fst, const std::string &method,
                        const fst::StdFst *ccfst = nullptr,
                        bool interpolate = false, int64_t bins = -1,
                        int64_t backoff_label = 0, double norm_eps = kNormEps,
                        bool check_consistency = false);

// TODO(kbg): Figure out how to make this compatible with scripting interface.

}  // namespace ngram

#endif  // NGRAM_NGRAM_MAKE_H_
