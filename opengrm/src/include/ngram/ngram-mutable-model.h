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
// NGram mutable model class.

#ifndef NGRAM_NGRAM_MUTABLE_MODEL_H_
#define NGRAM_NGRAM_MUTABLE_MODEL_H_

#include <algorithm>
#include <cstddef>
#include <vector>

#include <fst/arcsort.h>
#include <fst/expanded-fst.h>
#include <fst/mutable-fst.h>
#include <fst/statesort.h>
#include <ngram/hist-arc.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

template <class Arc>
class NGramMutableModel : public NGramModel<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  using NGramModel<Arc>::GetBackoff;
  using NGramModel<Arc>::NumNGrams;
  using NGramModel<Arc>::GetFst;
  using NGramModel<Arc>::BackoffLabel;
  using NGramModel<Arc>::NumStates;
  using NGramModel<Arc>::UnigramState;
  using NGramModel<Arc>::CalcBONegLogSums;
  using NGramModel<Arc>::CalculateBackoffCost;
  using NGramModel<Arc>::ScalarValue;

  // Constructs an NGramMutableModel object, derived from NGramModel,
  // that adds mutable methods such as backoff normalization.
  // Ownership of the FST is retained by the caller.
  explicit NGramMutableModel(fst::MutableFst<Arc> *infst,
                             Label backoff_label, double norm_eps,
                             bool state_ngrams, bool infinite_backoff = false)
      : NGramModel<Arc>(*infst, backoff_label, norm_eps, state_ngrams),
        infinite_backoff_(infinite_backoff),
        mutable_fst_(infst) {}

  // Same as above, but requires the FST and the backoff label.
  NGramMutableModel(fst::MutableFst<Arc> *infst, Label backoff_label)
      : NGramModel<Arc>(*infst, backoff_label),
        infinite_backoff_(false),
        mutable_fst_(infst) {}

  // Same as above, but uses default parameters where possible.
  explicit NGramMutableModel(fst::MutableFst<Arc> *infst)
      : NGramModel<Arc>(*infst),
        infinite_backoff_(false),
        mutable_fst_(infst) {}

  // ExpandedFst const reference
  const fst::ExpandedFst<Arc> &GetExpandedFst() const {
    return *mutable_fst_;
  }

  // Mutable Fst pointer
  fst::MutableFst<Arc> *GetMutableFst() { return mutable_fst_; }

  // For given state, recalculates backoff cost, assigns to backoff arc
  void RecalcBackoff(StateId st) {
    double hi_neglog_sum, low_neglog_sum;
    if (CalcBONegLogSums(st, &hi_neglog_sum, &low_neglog_sum,
                         infinite_backoff_)) {
      UpdateBackoffCost(st, hi_neglog_sum, low_neglog_sum);
    }
  }

  // For all states, recalculates backoff cost, assigns to backoff arc
  // (if exists)
  void RecalcBackoff() {
    for (StateId st = 0; st < mutable_fst_->NumStates(); ++st) {
      if (NGramModel<Arc>::Error()) return;
      RecalcBackoff(st);
    }
  }

  // Scales weights in the whole model
  void ScaleWeights(double scale) {
    for (StateId st = 0; st < mutable_fst_->NumStates(); ++st)
      ScaleStateWeight(st, scale);
  }

  // Looks for infinite backoff cost in model, sets flag to allow if found
  void SetAllowInfiniteBO() {
    for (StateId s = 0; s < NumStates(); ++s) {
      Weight bocost;
      StateId bo = GetBackoff(s, &bocost);
      if (bo >= 0 && ScalarValue(bocost) >= kInfBackoff) {
        infinite_backoff_ = true;  // found an 'infinite' backoff, so true
        return;
      }
    }
  }

  // Sorts states in ngram-context lexicographic order.
  void SortStates() {
    std::vector<StateId> order(NumStates()), inv_order(NumStates());
    for (StateId s = 0; s < NumStates(); ++s) order[s] = s;
    std::sort(order.begin(), order.end(), StateCompare(*this));
    for (StateId s = 0; s < NumStates(); ++s) inv_order[order[s]] = s;
    StateSort(mutable_fst_, inv_order);
  }

  // Set a scalar value of a given weight to a specified value
  void SetScalarValue(Weight *w, double scalar);

  // Scale given weight by a given scalar
  Weight ScaleWeight(Weight w, double scalar);

 protected:
  Weight GetBackoffFinalCost(StateId st) const {
    if (mutable_fst_->Final(st) != Arc::Weight::Zero()) {
      return mutable_fst_->Final(st);
    }
    Weight fcost;
    StateId bo = GetBackoff(st, &fcost);
    fcost = Times(fcost, GetBackoffFinalCost(bo));
    if (fcost != Arc::Weight::Zero()) {
      mutable_fst_->SetFinal(st, fcost);
    }
    return fcost;
  }

  // Uses iterator in place of matcher for mutable arc iterators,
  // avoids full copy and allows getting Position(). NB: begins
  // search from current position.
  bool FindMutableArc(
      fst::MutableArcIterator<fst::MutableFst<Arc>> *biter,
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

  // Scale weights by some factor, for normalizing and use in model merging
  void ScaleStateWeight(StateId st, double scale) {
    if (mutable_fst_->Final(st) != Arc::Weight::Zero()) {
      mutable_fst_->SetFinal(st, ScaleWeight(mutable_fst_->Final(st), scale));
    }
    for (fst::MutableArcIterator<fst::MutableFst<Arc>> aiter(
             mutable_fst_, st);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != BackoffLabel()) {  // only scaling non-backoff arcs
        arc.weight = ScaleWeight(arc.weight, scale);
        aiter.SetValue(arc);
      }
    }
  }

  // Sorts arcs in state in ilabel order.
  void SortArcs(StateId s) {
    fst::ILabelCompare<Arc> comp;
    std::vector<Arc> arcs;
    for (fst::ArcIterator<fst::MutableFst<Arc>> aiter(*mutable_fst_, s);
         !aiter.Done(); aiter.Next())
      arcs.push_back(aiter.Value());
    std::sort(arcs.begin(), arcs.end(), comp);
    mutable_fst_->DeleteArcs(s);
    for (size_t a = 0; a < arcs.size(); ++a) mutable_fst_->AddArc(s, arcs[a]);
  }

  // Replace backoff weight with -log p(backoff)
  void DeBackoffNGramModel() {
    for (StateId st = 0; st < mutable_fst_->NumStates(); ++st) {
      double hi_neglog_sum, low_neglog_sum;
      if (CalcBONegLogSums(st, &hi_neglog_sum, &low_neglog_sum)) {
        fst::MutableArcIterator<fst::MutableFst<Arc>> aiter(
            mutable_fst_, st);
        if (FindMutableArc(&aiter, BackoffLabel())) {
          Arc arc = aiter.Value();
          SetScalarValue(&arc.weight, -log(1 - exp(-hi_neglog_sum)));
          aiter.SetValue(arc);
        } else {
          NGRAMERROR() << "NGramMutableModel: No backoff arc found: " << st;
          NGramModel<Arc>::SetError();
          return;
        }
      }
    }
  }

 private:
  // Calculate and assign backoff cost from neglog
  // sums of hi and low order arcs
  void UpdateBackoffCost(StateId st, double hi_neglog_sum,
                         double low_neglog_sum) {
    double alpha =
        CalculateBackoffCost(hi_neglog_sum, low_neglog_sum, infinite_backoff_);
    AdjustCompleteStates(st, &alpha);
    fst::MutableArcIterator<fst::MutableFst<Arc>> aiter(mutable_fst_,
                                                                st);
    if (FindMutableArc(&aiter, BackoffLabel())) {
      Arc arc = aiter.Value();
      SetScalarValue(&arc.weight, alpha);
      aiter.SetValue(arc);
    } else {
      NGRAMERROR() << "NGramMutableModel: No backoff arc found: " << st;
      NGramModel<Arc>::SetError();
    }
  }

  // Sets alpha to kInfBackoff for states with every possible n-gram
  void AdjustCompleteStates(StateId st, double *alpha) {
    int unigram_state = UnigramState();
    if (unigram_state < 0) unigram_state = GetFst().Start();
    if (NumNGrams(unigram_state) == NumNGrams(st)) *alpha = kInfBackoff;
  }

  // Scan arcs and remove lower order from arc weight
  void UnSumState(StateId st) {
    Weight bocost;
    StateId bo = GetBackoff(st, &bocost);
    for (fst::MutableArcIterator<fst::MutableFst<Arc>> aiter(
             mutable_fst_, st);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == BackoffLabel()) continue;
      SetScalarValue(&arc.weight,
                     NegLogDiff(ScalarValue(arc.weight),
                                ScalarValue(FindArcWeight(bo, arc.ilabel)) +
                                    ScalarValue(bocost)));
      aiter.SetValue(arc);
    }
    if (ScalarValue(mutable_fst_->Final(st)) !=
        ScalarValue(Arc::Weight::Zero())) {
      Weight w = mutable_fst_->Final(st);
      SetScalarValue(&w, NegLogDiff(ScalarValue(mutable_fst_->Final(st)),
                                    ScalarValue(mutable_fst_->Final(bo)) +
                                        ScalarValue(bocost)));
      mutable_fst_->SetFinal(st, w);
    }
  }

  class StateCompare {
   public:
    explicit StateCompare(const NGramModel<Arc> &ngramlm) : ngramlm_(ngramlm) {}

    bool operator()(StateId s1, StateId s2) const {
      std::vector<Label> ngram1 = ngramlm_.StateNGram(s1);
      std::vector<Label> ngram2 = ngramlm_.StateNGram(s2);
      return std::lexicographical_compare(ngram1.begin(), ngram1.end(),
                                          ngram2.begin(), ngram2.end());
    }

   private:
    const NGramModel<Arc> &ngramlm_;
  };

  bool infinite_backoff_;
  fst::MutableFst<Arc> *mutable_fst_;
};

template <typename Arc>
void NGramMutableModel<Arc>::SetScalarValue(
    typename NGramMutableModel<Arc>::Weight *w, double scalar) {
  *w = scalar;
}

template <>
inline void NGramMutableModel<HistogramArc>::SetScalarValue(
    NGramMutableModel<HistogramArc>::Weight *w, double scalar) {
  w->SetValue(0, scalar);
}

template <typename Arc>
typename NGramMutableModel<Arc>::Weight NGramMutableModel<Arc>::ScaleWeight(
    NGramMutableModel<Arc>::Weight w, double scalar) {
  return Times(scalar, w);
}

template <>
inline NGramMutableModel<HistogramArc>::Weight
NGramMutableModel<HistogramArc>::ScaleWeight(
    NGramMutableModel<HistogramArc>::Weight w, double scalar) {
  w.SetValue(0, Times(w.Value(0), scalar));
  return w;
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_MUTABLE_MODEL_H_
