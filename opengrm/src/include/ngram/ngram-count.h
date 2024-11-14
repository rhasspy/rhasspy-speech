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
// NGram counting class.

#ifndef NGRAM_NGRAM_COUNT_H_
#define NGRAM_NGRAM_COUNT_H_

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fst/log.h>
#include <fst/extensions/far/far.h>
#include <fst/arc.h>
#include <fst/connect.h>
#include <fst/fst.h>
#include <fst/fstlib.h>
#include <fst/mutable-fst.h>
#include <fst/properties.h>
#include <fst/symbol-table.h>
#include <fst/vector-fst.h>
#include <ngram/hist-arc.h>
#include <ngram/ngram-count-of-counts.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>
#include <unordered_map>

namespace ngram {

// NGramCounter class.
template <class Weight, class Label = int32_t>
class NGramCounter {
 public:
  // Construct an NGramCounter object counting n-grams of order less or equal to
  // 'order'. When 'epsilon_as_backoff' is 'true', the epsilon transition in the
  // input Fst are treated as failure backoff transitions and would trigger the
  // length of the current context to be decreased by one ("pop front").
  explicit NGramCounter(size_t order, bool epsilon_as_backoff = false,
                        float delta = 1e-9F)
      : order_(order),
        pair_arc_maps_(order),
        epsilon_as_backoff_(epsilon_as_backoff),
        delta_(delta),
        error_(false) {
    if (order < 1) {
      NGRAMERROR() << "NGramCounter: order must be positive";
      SetError();
      return;
    }
    backoff_ = states_.size();
    states_.push_back(CountState(-1, 1, Weight::Zero(), -1));
    if (order == 1) {
      initial_ = backoff_;
    } else {
      initial_ = states_.size();
      states_.push_back(CountState(backoff_, 2, Weight::Zero(), -1));
    }
  }

  // Extract counts from the input acyclic FST.  Return 'true' if counting
  // from the FST was successful and false otherwise. If necessary, a copy is
  // taken and this is sent to the MutableFst overload.
  template <class Arc>
  bool Count(const fst::Fst<Arc> &fst) {
    if (Error()) return false;
    if (fst.Start() == fst::kNoStateId) return false;
    if (fst.Properties(fst::kString, false) == fst::kString) {
      return CountFromStringFst(fst);
    } else if (fst.Properties(fst::kTopSorted, true)) {
      return CountFromTopSortedFst(fst);
    } else {
      fst::VectorFst<Arc> vfst(fst);
      return Count(&vfst);
    }
  }

  // Extract counts from input mutable acyclic FST, top-sorting the input FST
  // if necessary. Returns 'true' when the counting from
  // the Fst was successful and false otherwise.
  template <class Arc>
  bool Count(fst::MutableFst<Arc> *fst) {
    if (Error()) return false;
    if (fst->Start() == fst::kNoStateId) return false;
    if (fst->Properties(fst::kString, true) == fst::kString) {
      return CountFromStringFst(*fst);
    }
    if (fst->Properties(fst::kCoAccessible, true) !=
        fst::kCoAccessible) {
      LOG(WARNING) << "NGramCounter::Count: input FST is not coaccessible";
      fst::Connect(fst);
      // The FSA could be fully coinaccessible.
      if (fst->Start() == fst::kNoStateId) return false;
    }
    if (fst->Properties(fst::kTopSorted, true) != fst::kTopSorted) {
      if (!fst::TopSort(fst)) {
        // TODO(allauzen): support key in error message.
        LOG(ERROR) << "NGramCounter::Count: input FST is not acyclic";
        return false;
      }
    }
    return CountFromTopSortedFst(*fst);
  }

  // Get an FST representation of the ngram counts.
  template <class Arc>
  void GetFst(fst::MutableFst<Arc> *fst) {
    fst->DeleteStates();
    if (Error()) return;
    for (size_t s = 0; s < states_.size(); ++s) {
      fst->AddState();
      fst->SetFinal(s, states_[s].final_count.Value());
      if (states_[s].backoff_state != -1)
        fst->AddArc(s, Arc(0, 0, Arc::Weight::Zero(),
                           states_[s].backoff_state));
    }
    for (size_t a = 0; a < arcs_.size(); ++a) {
      const CountArc &arc = arcs_[a];
      fst->AddArc(arc.origin, Arc(arc.label, arc.label, arc.count.Value(),
                                  arc.destination));
    }
    fst->SetStart(initial_);
    StateCounts(fst);
  }

  // Returns strings of ngram counts, in reverse context order, e.g., for the
  // ngram "feed the angry duck", returns "<{angry,the,feed}, <duck,count>>".
  template <class Arc>
  void GetReverseContextNGrams(
      std::vector<std::pair<std::vector<int>, std::pair<Label, double>>>
          *ngram_counts) {
    if (Error()) return;
    std::vector<int> incoming_words(states_.size(), -1);
    std::vector<int> previous_states(states_.size(), -1);
    if (order_ > 1) incoming_words[NGramStartState()] = 0;
    for (size_t a = 0; a < arcs_.size(); ++a) {
      const CountArc &arc = arcs_[a];
      if (states_[arc.origin].order < states_[arc.destination].order) {
        previous_states[arc.destination] = arc.origin;
        incoming_words[arc.destination] = arc.label;
      }
    }
    std::vector<std::vector<int>> reverse_context(states_.size());
    for (size_t s = 0; s < states_.size(); ++s) {
      int ps = s;
      while (ps >= 0) {
        if (incoming_words[ps] >= 0)
          reverse_context[s].push_back(incoming_words[ps]);
        ps = previous_states[ps];
      }
      if (states_[s].final_count.Value() != Weight::Zero().Value()) {
        ngram_counts->push_back(
            std::make_pair(reverse_context[s],
                           std::make_pair(0, states_[s].final_count.Value())));
      }
    }
    for (size_t a = 0; a < arcs_.size(); ++a) {
      const CountArc &arc = arcs_[a];
      ngram_counts->push_back(
          std::make_pair(reverse_context[arc.origin],
                         std::make_pair(arc.label, arc.count.Value())));
    }
  }

  // Given a state ID and a label, returns the ID of the corresponding
  // arc, creating the arc if it does not exist already.
  ssize_t FindArc(ssize_t state_id, Label label) {
    const CountState &count_state = states_[state_id];
    // First determines if there already exists a corresponding arc.
    if (count_state.first_arc != -1) {
      if (arcs_[count_state.first_arc].label == label)
        return count_state.first_arc;
      const PairArcMap &arc_map = pair_arc_maps_[count_state.order - 1];
      const auto iter = arc_map.find(std::make_pair(label, state_id));
      if (iter != arc_map.end()) return iter->second;
    }
    // Otherwise, this arc needs to be created.
    return AddArc(state_id, label);
  }

  // Gets the start state of the counts (<s>).
  ssize_t NGramStartState() { return initial_; }

  // Gets the unigram state of the counts.
  ssize_t NGramUnigramState() { return backoff_; }

  // Gets the backoff state for a given state.
  ssize_t NGramBackoffState(ssize_t state_id) {
    return states_[state_id].backoff_state;
  }

  // Gets the next state from a found arc.
  ssize_t NGramNextState(ssize_t arc_id) {
    if (arc_id < 0 || arc_id >= arcs_.size()) return -1;
    return arcs_[arc_id].destination;
  }

  // Sets the weight for an n-gram ending with the stop symbol </s>.
  bool SetFinalNGramWeight(ssize_t state_id, Weight weight) {
    if (state_id < 0 || state_id >= states_.size()) return false;
    states_[state_id].final_count = weight;
    return true;
  }

  // Adds count to the unigram count of all symbols in the symbol table.
  void AddCountToSymbolUnigrams(const fst::SymbolTable &syms,
                                       Weight neg_log_count) {
    for (const auto &sitem : syms) {
      if (sitem.Label() == 0) {
        // If label == 0, updates final cost of unigram state.
        UpdateFinalCount(NGramUnigramState(), neg_log_count);
      } else {
        UpdateCount(NGramUnigramState(), sitem.Label(), neg_log_count);
      }
    }
  }

  // Sets the weight for a found n-gram.
  bool SetNGramWeight(ssize_t arc_id, Weight weight) {
    if (arc_id < 0 || arc_id >= arcs_.size()) return false;
    arcs_[arc_id].count = weight;
    return true;
  }

  // Size of ngram model is the sum of the number of states and number of arcs.
  ssize_t GetSize() const { return states_.size() + arcs_.size(); }

  // Returns true if counter setup is in a bad state.
  bool Error() const { return error_; }

 protected:
  void SetError() { error_ = true; }

 private:
  // Data representation for a state.
  struct CountState {
    ssize_t backoff_state;  // ID of the backoff state for the current state.
    size_t order;           // N-gram order of the state (of the outgoing arcs).
    Weight final_count;     // Count for n-gram corresponding to superfinal arc.
    ssize_t first_arc;      // ID of the first outgoing arc at that state.

    CountState(ssize_t s, size_t o, Weight c, ssize_t a)
        : backoff_state(s), order(o), final_count(c), first_arc(a) {}
  };

  // Data represention for an arc.
  struct CountArc {
    ssize_t origin;       // ID of the origin state for this arc.
    ssize_t destination;  // ID of the destination state for this arc.
    Label label;          // Label.
    Weight count;         // Count of the n-gram corresponding to this arc.
    ssize_t backoff_arc;  // ID of backoff arc.

    CountArc(ssize_t o, size_t d, Label l, Weight c, ssize_t b)
        : origin(o), destination(d), label(l), count(c), backoff_arc(b) {}
  };

  // A pair (Label, State ID) or (State ID, State ID)
  using Pair = std::pair<ssize_t, ssize_t>;

  struct PairHash {
    size_t operator()(const Pair &p) const {
      return (static_cast<size_t>(p.first) * 55697) ^
             (static_cast<size_t>(p.second) * 54631);
      // TODO(allauzen): run benchmark using Compose's hash function
      // return static_cast<size_t>(p.first + p.second * 7853);
    }
  };

  // TODO(allauzen): run benchmark using map instead of unordered map
  using PairArcMap = std::unordered_map<Pair, size_t, PairHash>;

  // Creates the arc corresponding to label 'label' out of the state
  // with ID 'state_id'.
  size_t AddArc(ssize_t state_id, Label label) {
    const auto count_state = states_[state_id];
    const ssize_t arc_id = arcs_.size();
    // Updates the hash entry for the new arc.
    if (count_state.first_arc == -1) {
      states_[state_id].first_arc = arc_id;
    } else {
      pair_arc_maps_[count_state.order - 1].emplace(
          std::make_pair(label, state_id), arc_id);
    }
    // Pre-fills arc with values valid when order_ == 1 and returns
    // if nothing else needs to be done.
    arcs_.emplace_back(state_id, initial_, label, Weight::Zero(), -1);
    if (order_ == 1) return arc_id;
    // First compute the backoff arc
    const ssize_t backoff_arc = count_state.backoff_state == -1
                                    ? -1
                                    : FindArc(count_state.backoff_state, label);
    // Second compute the destination state.
    ssize_t destination;
    if (count_state.order == order_) {
      // The destination state is the destination of the backoff arc.
      destination = arcs_[backoff_arc].destination;
    } else {
      // The destination state needs to be created.
      destination = states_.size();
      CountState next_count_state(
          backoff_arc == -1 ? backoff_ : arcs_[backoff_arc].destination,
          count_state.order + 1, Weight::Zero(), -1);
      states_.push_back(next_count_state);
    }
    // Updates destination and backoff_arc with the newly computed values.
    arcs_[arc_id].destination = destination;
    arcs_[arc_id].backoff_arc = backoff_arc;
    return arc_id;
  }

  // Increase the count of n-gram corresponding to the arc labeled 'label'
  // out of state of ID 'state_id' by 'count'.
  ssize_t UpdateCount(ssize_t state_id, Label label, Weight count) {
    ssize_t arc_id = FindArc(state_id, label);
    const ssize_t nextstate_id = arcs_[arc_id].destination;
    while (arc_id != -1) {
      arcs_[arc_id].count = Plus(arcs_[arc_id].count, count);
      arc_id = arcs_[arc_id].backoff_arc;
    }
    return nextstate_id;
  }

  // Increase the count of n-gram corresponding to the super-final arc
  // out of state of ID 'state_id' by 'count'.
  void UpdateFinalCount(ssize_t state_id, Weight count) {
    while (state_id != -1) {
      states_[state_id].final_count =
          Plus(states_[state_id].final_count, count);
      state_id = states_[state_id].backoff_state;
    }
  }

  // Puts the sum of counts of non-backoff arcs leaving s on the backoff arc.
  template <class Arc>
  void StateCounts(fst::MutableFst<Arc> *fst) {
    for (size_t s = 0; s < states_.size(); ++s) {
      Weight state_count = states_[s].final_count;
      if (states_[s].backoff_state != -1) {
        fst::MutableArcIterator<fst::MutableFst<Arc>> aiter(fst, s);
        ssize_t bo_pos = -1;
        for (; !aiter.Done(); aiter.Next()) {
          const auto &arc = aiter.Value();
          if (arc.ilabel != 0) {
            state_count = Plus(state_count, arc.weight.Value());
          } else {
            bo_pos = aiter.Position();
          }
        }
        if (bo_pos < 0) {
          NGRAMERROR() << "backoff arc not found";
          SetError();
          return;
        }
        aiter.Seek(bo_pos);
        auto arc = aiter.Value();
        arc.weight = state_count.Value();
        aiter.SetValue(arc);
      }
    }
  }

  template <class Arc>
  bool CountFromTopSortedFst(const fst::Fst<Arc> &fst);

  template <class Arc>
  bool CountFromStringFst(const fst::Fst<Arc> &fst);

  struct PairCompare {
    bool operator()(const Pair &p1, const Pair &p2) const {
      return p1.first == p2.first ? p1.second > p2.second : p1.first > p2.first;
    }
  };

  size_t order_;                    // Maximal order of n-gram being counted
  std::vector<CountState> states_;  // Vector mapping state IDs to CountStates
  std::vector<CountArc> arcs_;      // Vector mapping arc IDs to CountArcs
  ssize_t initial_;                 // ID of start state
  ssize_t backoff_;                 // ID of unigram/backoff state
  std::vector<PairArcMap> pair_arc_maps_;  // Maps pairs to arc IDs.
  bool epsilon_as_backoff_;  // Treat epsilons as backoff trans. in input Fsts
  float delta_;              // Delta value used by shortest-distance
  bool error_;

  NGramCounter(const NGramCounter &) = delete;
  NGramCounter &operator=(const NGramCounter &) = delete;
};

template <class Weight, class Label>
template <class Arc>
bool NGramCounter<Weight, Label>::CountFromStringFst(
    const fst::Fst<Arc> &fst) {
  if (!fst.Properties(fst::kString, false)) {
    NGRAMERROR() << "Input FST is not a string";
    return false;
  }
  ssize_t count_state = initial_;
  auto fst_state = fst.Start();
  Weight weight = fst.Properties(fst::kUnweighted, false)
                      ? Weight::One()
                      : Weight(ShortestDistance(fst).Value());
  while (fst.Final(fst_state) == Arc::Weight::Zero()) {
    fst::ArcIterator<fst::Fst<Arc>> aiter(fst, fst_state);
    const auto &arc = aiter.Value();
    if (arc.ilabel) {
      count_state = UpdateCount(count_state, arc.ilabel, weight);
    } else if (epsilon_as_backoff_) {
      const ssize_t next_count_state = NGramBackoffState(count_state);
      count_state = next_count_state == -1 ? count_state : next_count_state;
    }
    fst_state = arc.nextstate;
    aiter.Next();
    if (!aiter.Done()) {
      NGRAMERROR() << "More than one arc leaving state " << fst_state;
      return false;
    }
  }
  UpdateFinalCount(count_state, weight);
  return true;
}

template <class Weight, class Label>
template <class Arc>
bool NGramCounter<Weight, Label>::CountFromTopSortedFst(
    const fst::Fst<Arc> &fst) {
  if (!fst.Properties(fst::kTopSorted, false)) {
    NGRAMERROR() << "Input not topologically sorted";
    return false;
  }
  // Computes shortest-distances from the initial state and to the final
  // states.
  std::vector<typename Arc::Weight> fdistance;
  ShortestDistance(fst, &fdistance, true, delta_);
  std::vector<Pair> heap;
  std::unordered_map<Pair, typename Arc::Weight, PairHash> pair2weight;
  PairCompare compare;
  Pair start_pair = std::make_pair(fst.Start(), initial_);
  pair2weight[start_pair] = Arc::Weight::One();
  heap.push_back(start_pair);
  std::push_heap(heap.begin(), heap.end(), compare);
  size_t i = 0;
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), compare);
    const auto current_pair = heap.back();
    const auto fst_state = current_pair.first;
    const ssize_t count_state = current_pair.second;
    auto current_weight = pair2weight[current_pair];
    pair2weight.erase(current_pair);
    heap.pop_back();
    ++i;
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst, fst_state);
         !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      Pair next_pair(arc.nextstate, count_state);
      if (arc.ilabel) {
        Weight count =  Times(current_weight,
                              Times(arc.weight,
                                    fdistance[arc.nextstate])).Value();
        next_pair.second = UpdateCount(count_state, arc.ilabel, count.Value());
      } else if (epsilon_as_backoff_) {
        const ssize_t next_count_state = NGramBackoffState(count_state);
        next_pair.second =
            next_count_state == -1 ? count_state : next_count_state;
      }
      const auto next_weight = Times(current_weight, arc.weight);
      const auto iter = pair2weight.find(next_pair);
      if (iter == pair2weight.end()) {  // If pair not in heap, inserts it.
        pair2weight[next_pair] = next_weight;
        heap.push_back(next_pair);
        std::push_heap(heap.begin(), heap.end(), compare);
      } else {  // Otherwise, updates the weight stored for it.
        iter->second = Plus(iter->second, next_weight);
      }
    }
    if (fst.Final(fst_state) != Arc::Weight::Zero()) {
      UpdateFinalCount(count_state,
                       Times(current_weight, fst.Final(fst_state)).Value());
    }
  }
  return true;
}

// Computes ngram counts and returns ngram format FST.
bool GetNGramCounts(fst::FarReader<fst::StdArc> *far_reader,
                    fst::StdMutableFst *fst, int order,
                    bool require_symbols = true,
                    bool epsilon_as_backoff = false, bool round_to_int = false,
                    double add_to_symbol_unigram_count = 0.0);

bool GetNGramCounts(fst::FarReader<fst::StdArc> *far_reader,
                    std::vector<std::string> *ngrams, int order,
                    bool epsilon_as_backoff = false,
                    double add_to_symbol_unigram_count = 0.0);

// Computes counts using the HistogramArc template.
bool GetNGramHistograms(fst::FarReader<fst::StdArc> *far_reader,
                        fst::VectorFst<HistogramArc> *fst, int order,
                        bool epsilon_as_backoff = false, int backoff_label = 0,
                        double norm_eps = kNormEps,
                        bool check_consistency = false, bool normalize = false,
                        double alpha = 1.0, double beta = 1.0);

// Computes count-of-counts.
template <class Arc>
void GetNGramCountOfCounts(const fst::Fst<Arc> &fst,
                           fst::StdMutableFst *ccfst, int in_order,
                           std::string_view context_pattern) {
  NGramModel<Arc> ngram(fst, 0, kNormEps, !context_pattern.empty());
  const auto order = ngram.HiOrder() > in_order ? ngram.HiOrder() : in_order;
  NGramCountOfCounts<Arc> count_of_counts(context_pattern, order);
  count_of_counts.CalculateCounts(ngram);
  count_of_counts.GetFst(ccfst);
}
}  // namespace ngram

#endif  // NGRAM_NGRAM_COUNT_H_
