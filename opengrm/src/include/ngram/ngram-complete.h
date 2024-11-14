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
// Complete a partial model by adding transitions to ensure proper topology
// as required by NGramModel.

#ifndef NGRAM_NGRAM_COMPLETE_H_
#define NGRAM_NGRAM_COMPLETE_H_

#include <set>
#include <vector>

#include <fst/arcsort.h>
#include <fst/fst.h>
#include <fst/matcher.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

// Ascends the NGram WFST from lower order states and collects state info.
template <class Arc>
bool AscendAndCollectStateInfo(
    const fst::Fst<Arc> &fst, int order, typename Arc::Label backoff_label,
    std::vector<std::vector<typename Arc::StateId>> *order_states,
    std::vector<int> *state_orders,
    std::vector<typename Arc::StateId> *backoff_states) {
  if (order >= order_states->size()) return false;
  for (int i = 0; i < (*order_states)[order].size(); ++i) {
    auto s = (*order_states)[order][i];
    if ((*state_orders)[s] != order) {
      NGRAMERROR() << "State " << s << " included in vector of states with "
                   << "order " << order << ", but that is not the case";
      return false;
    }
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == backoff_label) {
        (*backoff_states)[s] = arc.nextstate;
      } else if ((*state_orders)[arc.nextstate] <= 0) {
        if (order_states->size() <= order + 1) order_states->resize(order + 2);
        (*state_orders)[arc.nextstate] = order + 1;
        (*order_states)[order + 1].push_back(arc.nextstate);
      }
    }
    if (order > 1 && (*backoff_states)[s] == fst::kNoStateId) {
      NGRAMERROR() << "No backoff state for higher order state " << s;
      return false;
    }
  }
  return true;
}

template <class Arc>
bool NGramComplete(fst::MutableFst<Arc> *fst,
                   typename Arc::Label backoff_label = 0) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  if (fst->NumStates() < 2) return true;
  fst::ArcIterator<fst::Fst<Arc>> aiter(*fst, fst->Start());
  const Arc &arc = aiter.Value();
  if (arc.ilabel != backoff_label) {
    NGRAMERROR() << "First arc out of start state is not backoff arc";
    return false;
  }
  StateId unigram_state = arc.nextstate;
  std::vector<int> state_orders(fst->NumStates());
  std::vector<StateId> backoff_states(fst->NumStates(), fst::kNoStateId);
  std::vector<std::vector<StateId>> order_states(3);
  order_states[1].push_back(unigram_state);
  state_orders[unigram_state] = 1;
  order_states[2].push_back(fst->Start());
  state_orders[fst->Start()] = 2;
  backoff_states[fst->Start()] = unigram_state;
  int st_order = 1;
  while (st_order < order_states.size() && !order_states[st_order].empty()) {
    if (!AscendAndCollectStateInfo(*fst, st_order++, backoff_label,
                                   &order_states, &state_orders,
                                   &backoff_states)) {
      return false;
    }
  }

  for (StateId s = 0; s < fst->NumStates(); ++s) {
    if (s != unigram_state && backoff_states[s] == fst::kNoStateId) {
      NGRAMERROR() << "state with no backoff state: " << s;
      return false;
    }
    if (state_orders[s] <= 0) {
      NGRAMERROR() << "state not on ascending path: " << s;
      return false;
    }
  }

  std::vector<std::set<Label>> label_sets(fst->NumStates());
  std::set<StateId> new_final_states;

  std::unique_ptr<fst::Matcher<fst::Fst<Arc>>> matcher(
      new fst::Matcher<fst::Fst<Arc>>(*fst, fst::MATCH_INPUT));

  for (int order = order_states.size() - 1; order > 1; --order) {
    for (int idx = 0; idx < order_states[order].size(); ++idx) {
      StateId s = order_states[order][idx];
      StateId bs = backoff_states[s];
      if (state_orders[s] != state_orders[bs] + 1) {
        NGRAMERROR() << "State " << s << " backs off more than one order";
        return false;
      }
      matcher->SetState(bs);
      for (fst::ArcIterator<fst::Fst<Arc>> aiter(*fst, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel == backoff_label || matcher->Find(arc.ilabel)) continue;
        label_sets[bs].insert(arc.ilabel);
      }

      for (auto iter = label_sets[s].begin(); iter != label_sets[s].end();
           ++iter) {
        if (matcher->Find(*iter)) continue;
        label_sets[bs].insert(*iter);
      }

      if ((NGramModel<Arc>::ScalarValue(fst->Final(s)) !=
               NGramModel<Arc>::ScalarValue(Weight::Zero()) ||
           new_final_states.count(s) != 0) &&
          NGramModel<Arc>::ScalarValue(fst->Final(bs)) ==
              NGramModel<Arc>::ScalarValue(Weight::Zero())) {
        new_final_states.insert(bs);
      }
    }
  }

  for (int order = 1; order < order_states.size() - 1; ++order) {
    for (int idx = 0; idx < order_states[order].size(); ++idx) {
      StateId s = order_states[order][idx];
      if (label_sets[s].empty()) continue;
      StateId bs = backoff_states[s];
      std::unique_ptr<fst::Matcher<fst::Fst<Arc>>> updated_matcher(
          new fst::Matcher<fst::Fst<Arc>>(*fst, fst::MATCH_INPUT));
      if (bs < 0) {
        updated_matcher->SetState(s);
      } else {
        updated_matcher->SetState(bs);
      }

      std::vector<Arc> arcs;
      arcs.reserve(fst->NumArcs(s) + label_sets[s].size());

      for (auto iter = label_sets[s].begin(); iter != label_sets[s].end();
           ++iter) {
        StateId nextstate = unigram_state;
        if (bs >= 0 && updated_matcher->Find(*iter))
          nextstate = updated_matcher->Value().nextstate;
        arcs.push_back(
            Arc(*iter, *iter, NGramModel<Arc>::UnitCount(), nextstate));
      }

      if (arcs.empty()) {
        NGRAMERROR() << "No arcs found";
        return false;
      }

      for (fst::ArcIterator<fst::Fst<Arc>> aiter(*fst, s);
           !aiter.Done(); aiter.Next())
        arcs.push_back(aiter.Value());

      fst->DeleteArcs(s);
      std::sort(arcs.begin(), arcs.end(), fst::ILabelCompare<Arc>());

      for (size_t i = 0; i < arcs.size(); ++i) fst->AddArc(s, arcs[i]);
    }
  }

  for (auto iter = new_final_states.begin(); iter != new_final_states.end();
       ++iter) {
    fst->SetFinal(*iter, NGramModel<Arc>::UnitCount());
  }

  if (NGramModel<Arc>::ScalarValue(fst->Final(unigram_state)) ==
      NGramModel<Arc>::ScalarValue(Weight::Zero()))
    fst->SetFinal(unigram_state, NGramModel<Arc>::UnitCount());
  return true;
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_COMPLETE_H_
