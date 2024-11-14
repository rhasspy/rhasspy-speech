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
// Class for splitting an NGram model into multiple parts by context.

#ifndef NGRAM_NGRAM_SPLIT_H_
#define NGRAM_NGRAM_SPLIT_H_

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <fst/arcsort.h>
#include <fst/fst.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

// Splits NGram model into multiple parts by context.
// Assumes outer context encompasses input.
template <class Arc>
class NGramSplit {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  // Split based on context patterns (sse ngram-context.h for more
  // information).
  NGramSplit(const fst::Fst<Arc> &fst,
             const std::vector<std::string> &context_patterns,
             Label backoff_label = 0, double norm_eps = kNormEps,
             bool include_all_suffixes = false);

  // Split based on context begin and end vectors (sse ngram-context.h
  // for more information).
  NGramSplit(const fst::Fst<Arc> &infst,
             const std::vector<std::vector<Label>> &contexts,
             Label backoff_label = 0, double norm_eps = kNormEps,
             bool include_all_suffixes = false);

  // Return next NGram component model.
  bool NextNGramModel(fst::MutableFst<Arc> *outfst) {
    outfst->DeleteStates();
    CreateSplitFst(model_.GetFst(), split_, outfst);
    ++split_;
    if (Error()) return false;
    return true;
  }

  // Indicates if no more components to return.
  bool Done() const { return split_ >= contexts_.size(); }

  // Returns true if split in a bad state.
  bool Error() const { return error_; }

 protected:
  void SetError() { error_ = true; }

 private:
  void SplitNGramModel(const fst::Fst<Arc> &fst);

  void CreateSplitFst(const fst::Fst<Arc> &fst, size_t context_idx,
                      fst::MutableFst<Arc> *split_fst);

  NGramModel<Arc> model_;
  std::vector<std::unique_ptr<NGramContext>>
      contexts_;  // Ordered contexts to split
  std::vector<std::vector<StateId>>
      context_states_;  // States needed for each context.
  size_t split_;
  bool include_all_suffixes_;
  bool error_;
};

// Split based on context patterns (sse ngram-context.h for more information).
template <typename Arc>
NGramSplit<Arc>::NGramSplit(const fst::Fst<Arc> &fst,
                            const std::vector<std::string> &context_patterns,
                            Label backoff_label, double norm_eps,
                            bool include_all_suffixes)
    : model_(fst, backoff_label, norm_eps, true),
      split_(0),
      include_all_suffixes_(include_all_suffixes),
      error_(false) {
  for (int i = 0; i < context_patterns.size(); ++i) {
    contexts_.push_back(std::unique_ptr<NGramContext>(
        new NGramContext(context_patterns[i], model_.HiOrder())));
  }
  SplitNGramModel(fst);
}

// Split based on context begin and end vectors (sse ngram-context.h
// for more information).
template <typename Arc>
NGramSplit<Arc>::NGramSplit(const fst::Fst<Arc> &fst,
                            const std::vector<std::vector<Label>> &contexts,
                            Label backoff_label, double norm_eps,
                            bool include_all_suffixes)
    : model_(fst, backoff_label, norm_eps, true),
      split_(0),
      include_all_suffixes_(include_all_suffixes),
      error_(false) {
  for (int i = 0; i < contexts.size() - 1; ++i) {
    contexts_.push_back(std::unique_ptr<NGramContext>(
        new NGramContext(contexts[i], contexts[i + 1], model_.HiOrder())));
  }
  SplitNGramModel(fst);
}

template <typename Arc>
void NGramSplit<Arc>::SplitNGramModel(const fst::Fst<Arc> &fst) {
  // For each state, compute the strict split it belongs to.
  std::vector<std::set<size_t>> state_splits(model_.NumStates());
  for (StateId state = 0; state < model_.NumStates(); ++state) {
    const std::vector<Label> &ngram = model_.StateNGram(state);
    for (size_t i = 0; i < contexts_.size(); ++i) {
      if (contexts_[i]->HasContext(ngram, include_all_suffixes_))
        state_splits[state].insert(i);
    }
  }

  // Make sure the start state is added to every split.
  for (size_t i = 0; i < contexts_.size(); ++i)
    state_splits[fst.Start()].insert(i);

  // For each state, compute the sets of splits it needs to belong
  // to maintain the core model topology: needed ascending arcs and
  // backoff arcs.
  for (int order = model_.HiOrder(); order > 0; --order) {
    for (StateId state = 0; state < model_.NumStates(); ++state) {
      if (model_.StateOrder(state) != order) continue;
      // Second get more splits from ascending arcs.
      for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst, state);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (model_.StateOrder(arc.nextstate) != model_.StateOrder(state) + 1)
          continue;
        for (std::set<size_t>::const_iterator iter =
                 state_splits[arc.nextstate].begin();
             iter != state_splits[arc.nextstate].end(); ++iter) {
          state_splits[state].insert(*iter);
        }
      }
      // TODO(allauzen): decide whether keeping backoff of out of context
      // states. If so, process backoff after ascending arcs.
      // First propage the current splits for 'state' to its backoff.
      StateId bo_state = model_.GetBackoff(state, nullptr);
      if (bo_state >= 0) {
        for (std::set<size_t>::const_iterator iter =
                 state_splits[state].begin();
             iter != state_splits[state].end(); ++iter) {
          state_splits[bo_state].insert(*iter);
        }
      }
    }
  }

  // Make sure the unigram state was added to every split.
  for (size_t i = 0; i < contexts_.size(); ++i) {
    if (state_splits[model_.UnigramState()].count(i) != 1) {
      NGRAMERROR() << "Unigram state not added to every split";
      SetError();
      return;
    }
  }

  // Create state vectors for every context.
  context_states_.resize(contexts_.size());
  for (StateId state = 0; state < model_.NumStates(); ++state) {
    for (std::set<size_t>::const_iterator iter = state_splits[state].begin();
         iter != state_splits[state].end(); ++iter) {
      context_states_[*iter].push_back(state);
    }
  }
}

template <typename Arc>
void NGramSplit<Arc>::CreateSplitFst(const fst::Fst<Arc> &fst,
                                     size_t context_idx,
                                     fst::MutableFst<Arc> *split_fst) {
  if (Error()) return;
  split_fst->SetInputSymbols(fst.InputSymbols());
  split_fst->SetOutputSymbols(fst.OutputSymbols());
  std::map<StateId, StateId> state_map;
  std::vector<bool> state_in_context(model_.NumStates());

  // Create all required states in each split, and compute
  // state map for each split.
  for (size_t i = 0; i < context_states_[context_idx].size(); ++i) {
    state_in_context[context_states_[context_idx][i]] = true;
    state_map[context_states_[context_idx][i]] = split_fst->AddState();
  }

  // Add all needed and strictly in context transitions to each split.
  for (size_t i = 0; i < context_states_[context_idx].size(); ++i) {
    StateId state = context_states_[context_idx][i];
    Weight bo_weight = NGramModel<Arc>::UnitCount();
    StateId bo_state = model_.GetBackoff(state, &bo_weight);

    if (bo_state != fst::kNoStateId) {
      // Add backoff arc
      split_fst->AddArc(state_map[state],
                        Arc(0, 0, bo_weight, state_map[bo_state]));
    }

    if (state == fst.Start()) split_fst->SetStart(state_map[state]);

    const std::vector<Label> &ngram = model_.StateNGram(state);
    bool in_context =
        contexts_[context_idx]->HasContext(ngram, include_all_suffixes_);

    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == model_.BackoffLabel()) continue;
      StateId nextstate = arc.nextstate;
      if (!in_context &&
          ((model_.StateOrder(nextstate) != model_.StateOrder(state) + 1) ||
           !state_in_context[nextstate])) {
        continue;
      }
      auto iter = state_map.find(nextstate);
      if (!in_context && state != fst.Start() && iter == state_map.end()) {
        NGRAMERROR() << "out of context n-gram with no destination state";
        SetError();
        return;
      }
      while (iter == state_map.end()) {
        nextstate = model_.GetBackoff(nextstate, nullptr);
        if (nextstate == fst::kNoStateId) {
          NGRAMERROR() << "backoff state not found for destination state";
          SetError();
          return;
        }
        iter = state_map.find(nextstate);
      }
      split_fst->AddArc(state_map[state],
                        Arc(arc.ilabel, arc.olabel, arc.weight, iter->second));
    }

    if (in_context &&
        NGramModel<Arc>::ScalarValue(fst.Final(state)) !=
            NGramModel<Arc>::ScalarValue(Weight::Zero())) {
      split_fst->SetFinal(state_map[state], fst.Final(state));
    }
  }
  ArcSort(split_fst, fst::ILabelCompare<Arc>());
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_SPLIT_H_
