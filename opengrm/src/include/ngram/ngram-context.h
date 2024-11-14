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
// Class to parse and maintain context specifications.

#ifndef NGRAM_NGRAM_CONTEXT_H_
#define NGRAM_NGRAM_CONTEXT_H_

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fst/arc.h>
#include <fst/fst.h>
#include <ngram/ngram-model.h>
#include <unordered_map>

namespace ngram {

// Represents a context interval.
class NGramContext {
 public:
  typedef fst::StdArc::Label Label;
  typedef fst::StdArc::StateId StateId;

  // Constructs a context specification from begin and end context
  // vectors.  If the context is less than the n-gram order - 1, it is
  // padded with 0 on the left. The begin and end context vectors
  // specify a (half-open) interval of highest-order state contexts in
  // an LM with the interval defined using the reverse lexicographic
  // order (i.e., on the reverse of the context). All suffixes of
  // these contexts are also included for proper backoff (when
  // include_all_suffixes = true).
  //
  // Example1: context_begin = {1,1,1,1} and context_end = {1,1,1,5} with
  // a 5-gram:
  //   specifies states that have a rightmost context in [1,5).
  //
  // Example2: context_begin = {1} and context_end = {5,6} with a 5-gram:
  //   same as context_begin = {0,0,0,1} and context_end = {0,0,5,6}.
  NGramContext(std::vector<Label> context_begin,
               std::vector<Label> context_end, int hi_order)
      : hi_order_(hi_order),
        context_begin_(std::move(context_begin)),
        context_end_(std::move(context_end)) {
    Init();
  }

  // Constructs a context specification from context pattern string.
  // Expected format: "w_1 ...w_m : v_1 ... v_n" where
  // the w_i and v_i are numeric word IDs and m,n are typically less than
  // the n-gram order. A word ID 0 signfies the initial word.
  //
  // Example: "1 1 1 1 : 1 1 1 5" signifies a begin context vector of
  //   {1,1,1,1} and and end context vector of {1,1,1,5}. See next constructor
  //   for the behavior with these vectors.
  NGramContext(std::string_view context_pattern, int hi_order)
      : hi_order_(hi_order) {
    ParseContextInterval(context_pattern, &context_begin_, &context_end_);
    Init();
  }

  // Null context
  NGramContext() : hi_order_(0) {}

  // Is ngram in context?  If 'include_all_suffixes' is true, then all
  // suffixes of the begin and end contexts are considered in
  // context. When false, true (reverse) lexicographic order is used.
  bool HasContext(const std::vector<Label>& ngram,
                  bool include_all_suffixes = true) const;

  // No/empty context requested?
  int NullContext() const { return context_begin_.empty(); }

  // Derives begin and end context vectors from input context pattern
  // string.
  static void ParseContextInterval(std::string_view context_pattern,
                                   std::vector<Label> *context_begin,
                                   std::vector<Label> *context_end);

  // Generates context string from begin and end context vectors
  static std::string GetContextString(const std::vector<Label> &context_begin,
                                      const std::vector<Label> &context_end) {
    std::ostringstream context_pattern_strm;
    for (int i = 0; i < context_begin.size(); ++i) {
      context_pattern_strm << context_begin[i] << " ";
    }
    context_pattern_strm << ":";
    for (int i = 0; i < context_end.size(); ++i) {
      context_pattern_strm << " " << context_end[i];
    }
    return context_pattern_strm.str();
  }

  // Given a n-gram model, returns 'ncontext' contexts balanced for
  // size.  Arg 'bigram_thresh' determines how overfull a context
  // bin has to be to force a split at a bigram context.  The model
  // must have state n-grams enabled.
  template <class Arc>
  static void FindContexts(const NGramModel<Arc> &model, int ncontexts,
                           std::vector<std::string> *contexts,
                           float bigram_thresh = 1.1) {
    std::vector<std::vector<typename Arc::Label>> begin_contexts;
    std::vector<std::vector<typename Arc::Label>> end_contexts;
    FindContexts(model, ncontexts, &begin_contexts, &end_contexts,
                 bigram_thresh);

    for (int i = 0; i < begin_contexts.size(); ++i)
      contexts->push_back(GetContextString(begin_contexts[i], end_contexts[i]));
  }

  // Given a n-gram model, returns 'ncontext' contexts balanced for size. Arg
  // 'bigram_thresh' determines how overfull a context bin has to be to force a
  // split at a bigram context. The model must have state n-grams enabled.
  template <class Arc>
  static void FindContexts(
      const NGramModel<Arc> &model, int ncontexts,
      std::vector<std::vector<typename Arc::Label>> *begin_contexts,
      std::vector<std::vector<typename Arc::Label>> *end_contexts,
      float bigram_thresh = 1.1);

  // Begin context as could be passed to class constructor
  std::vector<Label> GetContextBegin() const {
    std::vector<Label> ngram(context_begin_);
    while (ngram.size() > 1 && ngram.back() == 0) ngram.pop_back();
    std::reverse(ngram.begin(), ngram.end());
    return ngram;
  }

  // End context as could be passed to class constructor
  std::vector<Label> GetContextEnd() const {
    std::vector<Label> ngram(context_end_);
    while (ngram.size() > 1 && ngram.back() == 0) ngram.pop_back();
    std::reverse(ngram.begin(), ngram.end());
    return ngram;
  }

  // Context is reversed and padded to high-order
  const std::vector<Label>& GetReverseContextBegin() const {
    return context_begin_;
  }

  // Context is reversed and padded to high-order
  const std::vector<Label>& GetReverseContextEnd() const {
    return context_end_;
  }

  // Note order is wrt transitions not states in the model;
  // so state ngram.size() == 1 has order 2
  int GetHiOrder() const { return hi_order_; }

  // Changes hi order (which affects context padding)
  // Used by NGramExtendedContext to put several
  // NGramContexts on the same hi-order.
  void SetHiOrder(int hi_order) {
    if (hi_order > hi_order_) {
      if (!NullContext()) {
        context_begin_.resize(hi_order - 1, 0);
        context_end_.resize(hi_order - 1, 0);
      }
      hi_order_ = hi_order;
    }
  }

 private:
  void Init();

  int hi_order_;
  std::vector<Label> context_begin_;
  std::vector<Label> context_end_;
};

// Represents a set of disjoint context intervals.
class NGramExtendedContext {
 public:
  typedef fst::StdArc::Label Label;

  // Constructs a context specification om begin and end context vectors.
  // See the corresponding NGramContext constructor.
  NGramExtendedContext(const std::vector<Label> &context_begin,
                       const std::vector<Label> &context_end, int hi_order) {
    contexts_.push_back(NGramContext(context_begin, context_end, hi_order));
    Init(false);
  }

  // Constructs a context specification from an extended context
  // pattern string. An extended context pattern is a comma-separated
  // set of NGramContext context patterns that must be disjoint.
  // If 'merge_contexts' is true, adjacent contexts will be merged.
  NGramExtendedContext(std::string_view extended_context_pattern, int hi_order,
                       bool merge_contexts = true) {
    ParseContextIntervals(extended_context_pattern, hi_order, &contexts_);
    Init(merge_contexts);
  }

  // Constructs a context specification from a NGramContext vector.
  // If 'merge_contexts' is true, adjacent contexts will be merged.
  explicit NGramExtendedContext(const std::vector<NGramContext> &contexts,
                                bool merge_contexts = true)
      : contexts_(contexts) {
    Init(merge_contexts);
  }

  // Null context
  NGramExtendedContext() = default;

  // No/empty context requested?
  int NullContext() const { return contexts_.empty(); }

  // Is ngram in context?  If 'include_all_suffixes' is true, then all
  // suffixes of the begin and end contexts are considered in
  // context. When false, true (reverse) lexicographic order is used.
  bool HasContext(const std::vector<Label> &ngram,
                  bool include_all_suffixes = true) const;

  // Find NGramContext that matches context. Returns a null pointer
  // if no match or if the input is the null context.  If
  // 'include_all_suffixes' is true, then all suffixes of the begin
  // and end contexts are considered in context. When false, true
  // (reverse) lexicographic order is used.
  const NGramContext * GetContext(
      const std::vector<Label> &ngram, bool include_all_suffixes = true) const;

  // Derives NGramContext vector from input extended context pattern string.
  static void ParseContextIntervals(std::string_view extended_context_pattern,
                                    int hi_order,
                                    std::vector<NGramContext> *contexts);

  // Generates an extended context string from a vector of NGramContexts
  static std::string GetExtendedContextString(
      const std::vector<NGramContext> &contexts) {
    std::ostringstream extended_context_pattern_strm;
    for (size_t i = 0; i < contexts.size(); ++i) {
      if (i > 0) extended_context_pattern_strm << ",";
      const std::vector<Label> &context_begin = contexts[i].GetContextBegin();
      const std::vector<Label> &context_end = contexts[i].GetContextEnd();
      extended_context_pattern_strm
          << NGramContext::GetContextString(context_begin, context_end);
    }
    return extended_context_pattern_strm.str();
  }

  const std::vector<NGramContext> &GetContexts() const { return contexts_; }

 private:
  // Ensures disjoint, same hi-order and canonicalizes.
  void Init(bool merge_contexts);

  // Ensures contexts are non-empty, non-overlapping and that
  // the high orders made to match.
  bool CheckContexts();

  // Merges contexts_ in range [i, j] and write to k.
  void MergeContexts(size_t i, size_t j, size_t k);

  // Comparison function object on contexts
  struct ContextCompare {
    bool operator()(const NGramContext &c1, const NGramContext &c2) const {
      // Sorts by beginning of the context interval. Will be a total
      // order assuming the context intervals are disjoint.
      const std::vector<Label> &b1 = c1.GetReverseContextBegin();
      const std::vector<Label> &b2 = c2.GetReverseContextBegin();
      return std::lexicographical_compare(b1.begin(), b1.end(), b2.begin(),
                                          b2.end());
    };
  };

  std::vector<NGramContext> contexts_;
};

// Reads (possibly extended) context specifications form a file into a vector.
bool NGramReadContexts(const std::string &file,
                       std::vector<std::string> *contexts);

// Writes (possibly extended) context specifications from a vector to a file.
bool NGramWriteContexts(const std::string &file,
                        const std::vector<std::string> &contexts);

template <class Arc>
void NGramContext::FindContexts(
    const NGramModel<Arc> &model, int ncontexts,
    std::vector<std::vector<typename Arc::Label>> *begin_contexts,
    std::vector<std::vector<typename Arc::Label>> *end_contexts,
    float bigram_thresh) {
  using fst::kNoLabel;
  // state n-gram counts with given unigram suffix
  std::map<typename Arc::Label, size_t> suffix1_counts;
  // state n-gram counts with given (reversed) bigram suffix
  std::map<std::pair<typename Arc::Label, typename Arc::Label>, size_t>
      suffix2_counts;
  // state n-gram counts at a bigram state
  std::unordered_map<typename Arc::Label, size_t> bigram_counts;
  size_t total_count = 0;
  typename Arc::Label max_label = kNoLabel;

  for (StateId s = 0; s < model.NumStates(); ++s) {
    for (fst::ArcIterator<fst::Fst<Arc>> aiter(model.GetFst(), s);
         !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == kNoLabel || arc.ilabel > max_label)
        max_label = arc.ilabel;
    }
    const auto &ngram = model.StateNGram(s);
    typename Arc::Label l1 =
        !ngram.empty() ? ngram[ngram.size() - 1] : kNoLabel;
    typename Arc::Label l2 =
        ngram.size() > 1 ? ngram[ngram.size() - 2] : kNoLabel;
    if (l1 == kNoLabel) continue;
    suffix1_counts[l1] += model.GetFst().NumArcs(s);
    total_count += model.GetFst().NumArcs(s);

    if (l2 != kNoLabel) {
      suffix2_counts[std::make_pair(l1, l2)] += model.GetFst().NumArcs(s);
    } else {
      bigram_counts[l1] += model.GetFst().NumArcs(s);
    }
  }
  std::vector<typename Arc::Label> context;
  begin_contexts->clear();
  end_contexts->clear();
  begin_contexts->push_back(context);
  begin_contexts->back().push_back(0);
  ssize_t bin_count = 0;
  auto it1 = suffix1_counts.begin();

  while (it1 != suffix1_counts.end()) {
    auto suffix1 = it1->first;
    ssize_t delta1 = it1->second;
    ssize_t deltab = bigram_counts[suffix1];
    ++it1;
    if (it1 != suffix1_counts.end() &&
        (bin_count + delta1) * ncontexts < total_count) {
      // Continues to fill bin
      bin_count += delta1;
    } else if ((bin_count + delta1 - deltab) * ncontexts <
               bigram_thresh * total_count) {
      // Splits at a unigram state suffix when bin not overfull
      if (it1 == suffix1_counts.end()) {
        // Finalizes at end of contexts
        end_contexts->push_back(context);
        end_contexts->back().push_back(max_label + 1);
      } else {
        // Splits and continues with next unigram state suffix
        auto next_suffix1 = it1->first;
        end_contexts->push_back(context);
        begin_contexts->push_back(context);
        end_contexts->back().push_back(next_suffix1);
        begin_contexts->back().push_back(next_suffix1);
        total_count -= bin_count + delta1;
        --ncontexts;
        bin_count = 0;
      }
    } else {
      // Splits at a bigram state context
      total_count -= deltab;
      auto it2 = suffix2_counts.find(std::make_pair(suffix1, 0));
      while (it2 != suffix2_counts.end() && it2->first.first <= suffix1) {
        ssize_t delta2 = it2->second;
        ++it2;
        if (it1 != suffix1_counts.end() &&
            (bin_count + delta2) * ncontexts < total_count) {
          // Continues to fill bin
          bin_count += delta2;
        } else if (it1 == suffix1_counts.end()) {
          // Finalizes at end of contexts
          end_contexts->push_back(context);
          end_contexts->back().push_back(max_label + 1);
        } else if (it2 == suffix2_counts.end() || it2->first.first > suffix1) {
          // Splits and continues with next unigram state suffix
          auto next_suffix1 = it1->first;
          end_contexts->push_back(context);
          begin_contexts->push_back(context);
          end_contexts->back().push_back(next_suffix1);
          begin_contexts->back().push_back(next_suffix1);
          total_count -= bin_count + delta2;
          --ncontexts;
          bin_count = 0;
        } else {
          // Splits and continues with next bigram state suffix
          auto next_suffix2 = it2->first;
          end_contexts->push_back(context);
          begin_contexts->push_back(context);
          end_contexts->back().push_back(next_suffix2.second);
          end_contexts->back().push_back(next_suffix2.first);
          begin_contexts->back().push_back(next_suffix2.second);
          begin_contexts->back().push_back(next_suffix2.first);
          total_count -= bin_count + delta2;
          --ncontexts;
          bin_count = 0;
        }
      }
    }
  }
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_CONTEXT_H_
