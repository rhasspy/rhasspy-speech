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
// Model shrinking derived class for a list of n-grams to be pruned.

#ifndef NGRAM_NGRAM_LIST_PRUNE_H_
#define NGRAM_NGRAM_LIST_PRUNE_H_

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <fst/arc.h>
#include <fst/expanded-fst.h>
#include <fst/fst.h>
#include <fst/matcher.h>
#include <fst/mutable-fst.h>
#include <fst/symbol-table.h>
#include <ngram/ngram-model.h>
#include <ngram/ngram-shrink.h>
#include <ngram/util.h>
#include <unordered_set>

namespace ngram {

// Context-restricting pruning
class NGramListPrune : public NGramShrink<fst::StdArc> {
 public:
  // Constructs an NGramShrink object, including an NGramModel and
  // parameters.  A set of word label vectors represent n-grams to be pruned.
  // The specified n-grams will be removed from the model, along with any
  // n-grams that have a specified n-gram as a substring.  Unigrams will not be
  // removed, but unigrams in the list will cause the removal of any
  // non-unigrams that have that unigram as a substring.
  NGramListPrune(fst::StdMutableFst *infst,
                 const std::set<std::vector<Label>> &ngrams_to_prune,
                 int shrink_opt = 0, double tot_uni = -1.0,
                 Label backoff_label = 0, double norm_eps = kNormEps,
                 bool check_consistency = false)
      : NGramShrink(infst, shrink_opt, tot_uni, backoff_label, norm_eps,
                    check_consistency),
        ngrams_to_prune_(ngrams_to_prune) {
    SetModelNormBool(CheckNormalization());
    InitializeNGramsToPrune();
  }

  ~NGramListPrune() override = default;

  // Shrinks n-gram model, based on initialized parameters
  bool ShrinkNGramModel(int min_order = 2) {
    return NGramShrink<fst::StdArc>::ShrinkNGramModel(
        /*require_norm=*/false, min_order);
  }

 protected:
  // Gives the pruning threshold.
  double GetTheta(StateId state) const override { return 0.0; }

  double ShrinkScore(const ShrinkStateStats &state,
                     const ShrinkArcStats &arc) const override {
    // Returns -1.0 if destination state is on a prune path or the origin state
    // and label are a highest level arc to be removed.
    return (state_on_prune_path_[state.state] ||
            (arc.dest >= 0 && state_on_prune_path_[arc.dest]) ||
            highest_order_prune_arcs_.find(std::make_pair(
                state.state, arc.label)) != highest_order_prune_arcs_.end())
               ? -1.0
               : 1.0;
  }

 private:
  // A pair (Label, State ID) or (State ID, State ID)
  using Pair = std::pair<ssize_t, ssize_t>;

  // TODO(roark): unify with struct in ngram-count.h and benchmark.
  struct PairHash {
    size_t operator()(const Pair &p) const {
      return (static_cast<size_t>(p.first) * 55697) ^
             (static_cast<size_t>(p.second) * 54631);
    }
  };

  // Initial discovery of target n-grams in the model, marking of destination
  // states if n-gram is ascending, and adding to a hash set for the highest
  // order.
  void InitializeStateVectors() {
    state_on_prune_path_.resize(GetMutableFst()->NumStates(), false);
    fst::Matcher<fst::Fst<fst::StdArc>> matcher(
        GetFst(), fst::MATCH_INPUT);
    for (auto ngram : ngrams_to_prune_) {
      StateId origin_state = fst::kNoStateId;
      StateId curr_state =
          UnigramState() < 0 ? GetMutableFst()->Start() : UnigramState();
      bool ngram_found = true;
      for (const auto label : ngram) {
        // Ascends from unigram state to end of ngram.
        if (label < 0) {
          ngram_found = false;
        } else {
          bool still_ascending =
              origin_state == fst::kNoStateId ||
              StateOrder(origin_state) < StateOrder(curr_state);
          matcher.SetState(curr_state);
          if (still_ascending && matcher.Find(label)) {
            fst::StdArc arc = matcher.Value();
            origin_state = curr_state;
            curr_state = arc.nextstate;
          } else {
            // Either word label not found or no longer ascending to higher
            // orders.  In either condition, n-gram is not in model.
            ngram_found = false;
          }
        }
        if (!ngram_found) {
          LOG(WARNING) << "ngram not found in model, nothing to remove";
          break;
        }
      }
      if (ngram_found) {
        if (StateOrder(origin_state) < StateOrder(curr_state)) {
          // Target n-gram has an ascending state, which is marked.  Any n-gram
          // pointing to this state and any subsequent n-grams should be pruned.
          state_on_prune_path_[curr_state] = true;
        } else {
          // Target n-gram in model at highest order, just this arc is pruned.
          highest_order_prune_origin_state_.insert(origin_state);
          highest_order_prune_arcs_.insert(
              std::make_pair(origin_state, ngram.back()));
        }
      }
    }
  }

  // Finds states and arcs at higher orders that should also be removed.
  bool CollectSuffixInfo() {
    std::vector<std::vector<StateId>> order_states(HiOrder() + 1);
    for (StateId s = 0; s < GetMutableFst()->NumStates(); ++s) {
      int state_order = StateOrder(s);
      if (state_order >= 0 && !state_on_prune_path_[s])
        order_states[state_order].push_back(s);
    }
    for (int order = 1; order <= HiOrder(); ++order) {
      for (auto s : order_states[order]) {
        fst::ArcIterator<fst::ExpandedFst<fst::StdArc>> aiter(
            GetExpandedFst(), s);
        fst::StdArc arc = aiter.Value();
        if (arc.ilabel == 0) {
          StateId backoff_state = arc.nextstate;
          if (state_on_prune_path_[backoff_state]) {
            state_on_prune_path_[s] = true;
          } else if (highest_order_prune_origin_state_.find(backoff_state) !=
                     highest_order_prune_origin_state_.end()) {
            for (; !aiter.Done(); aiter.Next()) {
              arc = aiter.Value();
              if (arc.ilabel == 0) continue;
              if (highest_order_prune_arcs_.find(
                      std::make_pair(backoff_state, arc.ilabel)) !=
                  highest_order_prune_arcs_.end()) {
                if (StateOrder(s) < StateOrder(arc.nextstate)) {
                  NGRAMERROR() << "Ascending arc, should be non-ascending.";
                  NGramModel<fst::StdArc>::SetError();
                  return false;
                }
                highest_order_prune_arcs_.insert(std::make_pair(s, arc.ilabel));
              }
            }
          }
        }
      }
    }
    return true;
  }

  // Ascend to all subsequent states and mark them if on prune path.
  void AscendAndMark(StateId s) {
    if (state_on_prune_path_[s]) {
      for (fst::ArcIterator<fst::ExpandedFst<fst::StdArc>> aiter(
               GetExpandedFst(), s);
           !aiter.Done(); aiter.Next()) {
        fst::StdArc arc = aiter.Value();
        if (arc.ilabel != 0 && !state_on_prune_path_[arc.nextstate] &&
            StateOrder(s) < StateOrder(arc.nextstate)) {
          state_on_prune_path_[arc.nextstate] = true;
          AscendAndMark(arc.nextstate);
        }
      }
    }
  }

  // Sets up data structures to allow for shrink scores to be assigned.
  void InitializeNGramsToPrune() {
    InitializeStateVectors();
    if (CollectSuffixInfo()) {
      for (StateId s = 0; s < GetMutableFst()->NumStates(); ++s) {
        AscendAndMark(s);
      }
    }
  }

  std::unordered_set<StateId> highest_order_prune_origin_state_;
  std::unordered_set<std::pair<StateId, Label>, PairHash>
      highest_order_prune_arcs_;
  std::vector<bool> state_on_prune_path_;
  std::set<std::vector<Label>> ngrams_to_prune_;
};

// Given a vector of ngram strings, adds vector of ngram labels to set.
void GetNGramListToPrune(
    const std::vector<std::string> &ngrams_to_prune,
    const fst::SymbolTable *syms,
    std::set<std::vector<fst::StdArc::Label>> *ngram_list,
    bool retry_downcase = false);

}  // namespace ngram

#endif  // NGRAM_NGRAM_LIST_PRUNE_H_
