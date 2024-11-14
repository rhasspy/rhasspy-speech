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

#include <ngram/ngram-context.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <fst/arc.h>
#include <fst/util.h>

namespace ngram {

using ::fst::StdArc;

namespace {
// Utility wrapper for HasContext that avoids copying the input vector.
class ReversedPaddedVector {
 public:
  using Label = StdArc::Label;

  // The input must outlive this.
  explicit ReversedPaddedVector(const std::vector<Label>& in, size_t len)
      : in_(in), len_(len) {}

  class Iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;
    using pointer = Label*;
    using reference = Label&;
    using value_type = Label;

    explicit Iterator(const ReversedPaddedVector* v, size_t i): i_(i), v_(v) {}

    Iterator(const Iterator&) = default;

    Iterator& operator=(const Iterator&) = default;

    Iterator& operator++() {
      i_ = std::min(i_ + 1, v_->len_);
      return *this;
    }

    Iterator operator++(int) {
      Iterator ret = *this;
      i_ = std::min(i_ + 1, v_->len_);
      return ret;
    }

    bool operator==(Iterator other) const {
      return i_ == other.i_ && v_ == other.v_;
    }

    bool operator!=(Iterator other) const {
      return i_ != other.i_ || v_ != other.v_;
    }

    value_type operator*() {
      return (i_ < v_->in_.size()) ? v_->in_[v_->in_.size() - i_ - 1] : 0;
    }

   private:
    size_t i_;
    const ReversedPaddedVector* v_;
  };

  Iterator begin() const { return Iterator(this, 0); }

  Iterator end() const { return Iterator(this, len_); }

 private:
  const std::vector<Label>& in_;
  const size_t len_;
};
}  // namespace

bool NGramContext::HasContext(const std::vector<Label> &ngram,
                              bool include_all_suffixes) const {
  if (NullContext())  // accept all
    return true;

  ReversedPaddedVector ngram_for_cmp(ngram, hi_order_ - 1);

  std::vector<Label>::const_iterator context_begin_end;
  if (include_all_suffixes) {
    // Truncate context_begin to ensure all state n-gram suffixes accepted
    context_begin_end = context_begin_.begin() + ngram.size();
  } else {
    // True (reverse) lexicographic order
    context_begin_end = context_begin_.end();
  }

  bool less_begin =
      std::lexicographical_compare(ngram_for_cmp.begin(), ngram_for_cmp.end(),
                                   context_begin_.begin(), context_begin_end);
  bool less_end =
      std::lexicographical_compare(ngram_for_cmp.begin(), ngram_for_cmp.end(),
                                   context_end_.begin(), context_end_.end());

  return !less_begin && less_end;
}

void NGramContext::ParseContextInterval(std::string_view context_pattern,
                                        std::vector<Label> *context_begin,
                                        std::vector<Label> *context_end) {
  context_begin->clear();
  context_end->clear();

  if (context_pattern.empty()) return;

  std::vector<std::string_view> contexts =
      StrSplit(context_pattern, ':', fst::SkipEmpty());
  if (contexts.size() != 2)
    LOG(FATAL) << "NGramContext: bad context pattern: " << context_pattern;
  std::vector<std::string_view> labels1 =
      StrSplit(contexts[0], ' ', fst::SkipEmpty());
  std::vector<std::string_view> labels2 =
      StrSplit(contexts[1], ' ', fst::SkipEmpty());
  for (int i = 0; i < labels1.size(); ++i) {
    Label label = fst::StrToInt64(labels1[i], "context begin", 1);
    context_begin->push_back(label);
  }
  for (int i = 0; i < labels2.size(); ++i) {
    Label label = fst::StrToInt64(labels2[i], "context end", 1);
    context_end->push_back(label);
  }
}

void NGramContext::Init() {
  if (NullContext()) return;

  std::reverse(context_begin_.begin(), context_begin_.end());
  std::reverse(context_end_.begin(), context_end_.end());
  if (context_begin_.size() >= hi_order_) hi_order_ = context_begin_.size() + 1;
  if (context_end_.size() >= hi_order_) hi_order_ = context_end_.size() + 1;
  context_begin_.resize(hi_order_ - 1, 0);
  context_end_.resize(hi_order_ - 1, 0);
  if (!std::lexicographical_compare(context_begin_.begin(),
                                    context_begin_.end(), context_end_.begin(),
                                    context_end_.end()))
    LOG(FATAL) << "NGramContext: bad context interval";
}

void NGramExtendedContext::Init(bool merge_contexts) {
  ContextCompare context_cmp;
  std::sort(contexts_.begin(), contexts_.end(), context_cmp);
  if (contexts_.empty()) return;

  if (contexts_.size() == 1 && contexts_[0].NullContext()) {
    contexts_.pop_back();
    return;
  }

  if (!CheckContexts()) {
    LOG(FATAL) << "NGramContexts: bad contexts";
  }

  if (merge_contexts) {
    size_t i = 0;
    size_t j = 1;
    size_t k = 0;
    for (; j < contexts_.size(); ++j) {
      const std::vector<Label> &e1 = contexts_[j - 1].GetReverseContextEnd();
      const std::vector<Label> &b2 = contexts_[j].GetReverseContextBegin();
      if (e1 != b2) {
        MergeContexts(i, j - 1, k++);
        i = j;
      }
    }
    MergeContexts(i, j - 1, k++);
    contexts_.resize(k);
  }
}

bool NGramExtendedContext::CheckContexts() {
  int hi_order = 0;
  for (size_t i = 0; i < contexts_.size(); ++i) {
    if (contexts_[i].NullContext()) {
      LOG(WARNING) << "CheckContexts: null context";
      return false;
    }
    if (hi_order < contexts_[i].GetHiOrder())
      hi_order = contexts_[i].GetHiOrder();
  }

  for (size_t i = 0; i < contexts_.size(); ++i)
    contexts_[i].SetHiOrder(hi_order);

  for (size_t i = 1; i < contexts_.size(); ++i) {
    const std::vector<Label> &e1 = contexts_[i - 1].GetReverseContextEnd();
    const std::vector<Label> &b2 = contexts_[i].GetReverseContextBegin();
    if (std::lexicographical_compare(b2.begin(), b2.end(), e1.begin(),
                                     e1.end())) {
      LOG(WARNING) << "CheckContexts: over-lapping context intevals";
      return false;
    };
  }

  return true;
}

void NGramExtendedContext::MergeContexts(size_t i, size_t j, size_t k) {
  if (i != j) {  // non-trivial merge?
    const std::vector<Label> &b1 = contexts_[i].GetContextBegin();
    const std::vector<Label> &e2 = contexts_[j].GetContextEnd();
    int hi_order = contexts_[0].GetHiOrder();
    NGramContext context(b1, e2, hi_order);
    contexts_[k] = context;
  } else if (i != k) {  // non-trivial move?
    contexts_[k] = contexts_[i];
  }
}

void NGramExtendedContext::ParseContextIntervals(
    std::string_view extended_context_pattern, int hi_order,
    std::vector<NGramContext> *contexts) {
  contexts->clear();
  std::vector<std::string_view> context_patterns =
      StrSplit(extended_context_pattern, ',', fst::SkipEmpty());

  for (size_t i = 0; i < context_patterns.size(); ++i)
    contexts->push_back(NGramContext(context_patterns[i], hi_order));
}

bool NGramExtendedContext::HasContext(const std::vector<Label> &ngram,
                                      bool include_all_suffixes) const {
  if (contexts_.empty())
    return true;
  else
    return GetContext(ngram, include_all_suffixes) != nullptr;
}

const NGramContext * NGramExtendedContext::GetContext(
    const std::vector<Label> &ngram, bool include_all_suffixes) const {
  std::vector<Label> ngram_end(ngram);
  if (ngram_end.empty()) ngram_end.push_back(0);
  std::vector<Label> ngram_beg(ngram_end);
  ++ngram_end[0];  // ensures non-empty interval below

  int hi_order = contexts_[0].GetHiOrder();
  NGramContext ngram_context(std::move(ngram_beg), std::move(ngram_end),
                             hi_order);
  ContextCompare context_cmp;
  auto it = std::upper_bound(contexts_.begin(), contexts_.end(), ngram_context,
                             context_cmp);
  if (it != contexts_.begin() &&
      (it - 1)->HasContext(ngram, include_all_suffixes)) {
    return &*(it - 1);  // strict match
  } else if (include_all_suffixes && it != contexts_.end() &&
             it->HasContext(ngram, include_all_suffixes)) {
    return &*it;  // suffix match
  } else {
    return nullptr;  // no match (or null context)
  }
}

bool NGramReadContexts(const std::string &file,
                       std::vector<std::string> *contexts) {
  contexts->clear();
  std::ifstream fstrm;
  fstrm.open(file);
  if (!fstrm) {
    LOG(ERROR) << "NGramReadContexts: Can't open file: " << file;
    return false;
  }
  std::istream &strm = fstrm.is_open() ? fstrm : std::cin;
  std::string line;
  while (std::getline(strm, line)) contexts->push_back(line);
  return true;
}

bool NGramWriteContexts(const std::string &file,
                        const std::vector<std::string> &contexts) {
  std::ofstream ofstrm;
  if (!file.empty()) {
    ofstrm.open(file);
    if (!ofstrm) {
      LOG(ERROR) << "NGramWriteContexts: Can't create file: " << file;
      return false;
    }
  }
  std::ostream &strm = ofstrm.is_open() ? ofstrm : std::cout;
  for (int i = 0; i < contexts.size(); ++i) strm << contexts[i] << std::endl;
  return true;
}

}  // namespace ngram
