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
// NGram model class for merging specified contexts from one FST into another.

#ifndef NGRAM_NGRAM_CONTEXT_MERGE_H_
#define NGRAM_NGRAM_CONTEXT_MERGE_H_

#include <memory>
#include <vector>

#include <fst/arc.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

class NGramContextMerge : public NGramMerge<fst::StdArc> {
 public:
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Constructs an NGramContextMerge object consisting of ngram model to be
  // merged.
  // Ownership of FST is retained by the caller.
  explicit NGramContextMerge(fst::StdMutableFst *infst1,
                             Label backoff_label = 0,
                             double norm_eps = kNormEps,
                             bool check_consistency = false)
      : NGramMerge(infst1, backoff_label, norm_eps, true) {}

  // Perform context merger with n-gram model specified by the FST
  // argument and 'context_pattern' string. These contexts taken from
  // the second FST and added to the first FST, replacing any existing
  // shared arcs.  See 'ngram-context.h' for meaning of the context
  // specification.
  void MergeNGramModels(const fst::StdFst &infst2,
                        std::string_view context_pattern, bool norm = false) {
    context_ =
        std::make_unique<NGramExtendedContext>(context_pattern, HiOrder());
    if (!NGramMerge<fst::StdArc>::MergeNGramModels(infst2, norm)) {
      NGRAMERROR() << "Context merge failed";
      NGramModel<fst::StdArc>::SetError();
    }
  }

  // Perform context merger with n-gram model specified by the FST argument
  // and 'context_begin' and 'context_end' vectors. These contexts
  // taken from the second FST and added to the first FST, replacing any
  // existing shared arcs.  See 'ngram-context.h' for meaning of the
  // context specification.
  void MergeNGramModels(const fst::StdFst &infst2,
                        const std::vector<Label> &context_begin,
                        const std::vector<Label> &context_end,
                        bool norm = false) {
    context_ = std::make_unique<NGramExtendedContext>(context_begin,
                                                      context_end, HiOrder());
    if (!NGramMerge<fst::StdArc>::MergeNGramModels(infst2, norm)) {
      NGRAMERROR() << "Context merge failed";
      NGramModel<fst::StdArc>::SetError();
    }
  }

 protected:
  // Specifies resultant weight when combining a weight from each FST
  Weight MergeWeights(StateId s1, StateId s2, Label label, Weight w1, Weight w2,
                      bool in_fst1, bool in_fst2) const override {
    if (in_fst1 && in_fst2) {
      // Takes weight from w2 if in both and ngram is strictly in context.
      const std::vector<Label> &ngram = NGram2().StateNGram(s2);
      return context_->HasContext(ngram, false) ? w2.Value() : w1.Value();
    } else if (in_fst1) {
      return w1.Value();
    } else {
      return w2.Value();
    }
  }

  // Specifies if unshared arcs/final weights between the two
  // FSTs in a merge have a non-trivial merge. In particular, this
  // means MergeWeights() changes the arc or final weights; any
  // destination state changes are not relevant here. When false, more
  // efficient merging may be performed. If the arc/final_weight
  // comes from the first FST, then 'in_fst1' is true.
  bool MergeUnshared(bool in_fst1) const override { return false; }

 private:
  std::unique_ptr<NGramExtendedContext> context_;
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_CONTEXT_MERGE_H_
