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
// NGram model class for merging FSTs via replacing weights.

#ifndef NGRAM_NGRAM_REPLACE_MERGE_H_
#define NGRAM_NGRAM_REPLACE_MERGE_H_

#include <vector>

#include <fst/arc.h>
#include <fst/connect.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

class NGramReplaceMerge : public NGramMerge<fst::StdArc> {
 public:
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Constructs an NGramReplaceMerge object consisting of ngram model
  // to be merged.  Since normalization in this case is only handled by
  // recalculating the backoff weights, the general merging mechanism is not
  // asked to normalize. Ownership of FST is retained by the caller.
  explicit NGramReplaceMerge(fst::StdMutableFst *infst1,
                             Label backoff_label = 0,
                             double norm_eps = kNormEps,
                             bool check_consistency = false)
      : NGramMerge(infst1, backoff_label, norm_eps, check_consistency) {}

  // Performs replacement merge with n-gram model specified by the FST argument
  // and a maximum order.  For all orders up to the max_replace_order, it is
  // assumed that infst2 has a superset of n-grams contained in the first model.
  // Resulting model will have up to and including max_replace_order orders of
  // the model from infst2, and any orders above that max from infst1.
  void MergeNGramModels(const fst::StdFst &infst2,
                        int max_replace_order = -1, bool norm = false) {
    if (Error()) return;
    NGramModel<fst::StdArc> mod2(infst2);
    if (!NGramMerge<fst::StdArc>::MergeNGramModels(
            infst2, /* norm = */ false, max_replace_order)) {
      NGRAMERROR() << "NGramReplaceMerge: Model merging failed";
      NGramModel::SetError();
      return;
    }
    TrimTopology();
    if (norm) {
      RecalcBackoff();
      if (!CheckNormalization()) {
        NGRAMERROR() << "NGramReplaceMerge: Merged model not fully normalized";
        NGramModel::SetError();
        return;
      }
    }
  }

 protected:
  // Specifies resultant weight when combining a weight from each FST
  Weight MergeWeights(StateId s1, StateId s2, Label label, Weight w1, Weight w2,
                      bool in_fst1, bool in_fst2) const override {
    if (!in_fst2) {
      NGRAMERROR() << "n-grams in original model must be a subset of n-grams "
                      "in the updating model.";
    }
    return w2;
  }

  // Specifies normalization constant per state 'st' depending whether
  // state was present in one or boths FSTs.
  double NormWeight(StateId st, bool in_fst1, bool in_fst2) const override {
    return 0.0;
  }

  // Returns true to allow for max orders less than the model's high order.
  bool MaxOrderOkay(int order) const override { return true; }

 private:
  // Discards extra states from orders higher than the maximum merged order,
  // and ensures that preserved arcs point to correct preserved states.
  void TrimTopology() {
    std::vector<StateId> dest_states(NumStates());
    for (StateId st = 0; st < NumStates(); ++st) {
      // Destinations states are identity unless state is a dead end.  Sets
      // destination state to -1 for dead end states, to be adjusted later.
      dest_states[st] = GetFst().NumArcs(st) == 0 &&
                                ScalarValue(GetFst().Final(st)) ==
                                    ScalarValue(fst::StdArc::Weight::Zero())
                            ? -1
                            : st;
    }
    for (StateId st = 0; st < NumStates(); ++st) {
      // Ignores dead end states when adjusting arcs.
      if (dest_states[st] != st) continue;
      StateId bo = GetBackoff(st, nullptr);
      for (fst::MutableArcIterator<fst::MutableFst<fst::StdArc>>
               aiter(GetMutableFst(), st);
           !aiter.Done(); aiter.Next()) {
        fst::StdArc arc = aiter.Value();
        if (dest_states[arc.nextstate] != arc.nextstate) {
          // If the arc is pointing to a dead end state, change destination.
          if (dest_states[arc.nextstate] < 0) {
            // If the destination has not been set yet, find and store it.
            UpdateDestStates(bo, arc, &dest_states);
          }
          if (dest_states[arc.nextstate] < 0) {
            NGRAMERROR() << "Destination state not set.";
            NGramModel::SetError();
            return;
          }
          arc.nextstate = dest_states[arc.nextstate];
          aiter.SetValue(arc);
        }
      }
    }

    // Discards all dead-end states and re-initializes model information.
    Connect(GetMutableFst());
    InitModel();
  }

  // Finds correct destination for arcs pointing to dead end states.
  void UpdateDestStates(StateId st, const fst::StdArc &in_arc,
                        std::vector<StateId> *dest_states) {
    if ((*dest_states)[in_arc.nextstate] >= 0) {
      NGRAMERROR() << "Destination state already set.";
      NGramModel::SetError();
      return;
    }
    if (st < 0) {
      // No backoff state found, should point arc to unigram state.
      (*dest_states)[in_arc.nextstate] = UnigramState();
    } else {
      StateId bo = GetBackoff(st, nullptr);
      fst::Matcher<fst::StdFst> matcher(GetFst(), fst::MATCH_INPUT);
      matcher.SetState(st);
      if (!matcher.Find(in_arc.ilabel)) {
        NGRAMERROR() << "Could not find n-gram arc at backoff state.";
        NGramModel::SetError();
        return;
      }
      fst::StdArc arc = matcher.Value();
      if ((*dest_states)[arc.nextstate] < 0) {
        // This arc also needs a new destination state.
        UpdateDestStates(bo, arc, dest_states);
      }
      if ((*dest_states)[arc.nextstate] < 0) {
        NGRAMERROR() << "destination state not set.";
        NGramModel::SetError();
        return;
      }
      if (arc.nextstate != in_arc.nextstate) {
        // Takes destination from valid destination state of backoff n-gram.
        (*dest_states)[in_arc.nextstate] = (*dest_states)[arc.nextstate];
      }
    }
  }
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_REPLACE_MERGE_H_
