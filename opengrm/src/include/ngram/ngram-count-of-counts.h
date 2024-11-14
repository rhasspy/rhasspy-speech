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
// Class for computing/accessing count-of-count bins for e.g., Katz and
// absolute discounting.

#ifndef NGRAM_NGRAM_COUNT_OF_COUNTS_H_
#define NGRAM_NGRAM_COUNT_OF_COUNTS_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/symbol-table.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

template <class Arc>
class NGramCountOfCounts {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  static constexpr int kMaxBins = 32;  // maximum # of bins allowed

  explicit NGramCountOfCounts(int bins = -1)
      : bins_(bins <= 0 || bins > kMaxBins ? kMaxBins : bins) {
    if (bins > kMaxBins)
      NGRAMERROR() << "NGramCountOfCounts: Number of bins too large: " << bins;
  }

  NGramCountOfCounts(std::string_view context_pattern, int order,
                     int bins = -1)
      : bins_(bins <= 0 || bins > kMaxBins ? kMaxBins : bins),
        context_(context_pattern, order) {
    if (bins > kMaxBins)
      NGRAMERROR() << "NGramCountOfCounts: Number of bins too large: " << bins;
  }

  NGramCountOfCounts(const std::vector<Label> &context_begin,
                     const std::vector<Label> &context_end, int order,
                     int bins = -1)
      : bins_(bins <= 0 || bins > kMaxBins ? kMaxBins : bins),
        context_(context_begin, context_end, order) {
    if (bins > kMaxBins)
      NGRAMERROR() << "NGramCountOfCounts: Number of bins too large: " << bins;
  }

  void CalculateCounts(const NGramModel<Arc> &model) {
    if (!histogram_.empty()) return;
    histogram_.resize(model.HiOrder());
    for (int order = 0; order < model.HiOrder(); ++order)  // for each order
      histogram_[order].resize(bins_ + 1, 0.0);            // space for bins + 1

    for (StateId st = 0; st < model.NumStates(); ++st) {  // get histograms
      if (!context_.NullContext()) {                      // restricted context
        const std::vector<Label> &ngram = model.StateNGram(st);
        if (!context_.HasContext(ngram, false)) continue;
      }
      int order = model.StateOrder(st) - 1;  // order starts from 0 here, not 1
      for (fst::ArcIterator<fst::Fst<Arc>> aiter(model.GetFst(), st);
           !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != model.BackoffLabel())  // no count from backoff
          IncrementBinCount(order, arc.weight, model);
      }
      IncrementBinCount(order, model.GetFst().Final(st), model);
    }
  }

  // Returns the number of bins
  int GetBins() const { return bins_; }

  // Put ngram in bin = count - 1 for 0 < count <= bins
  // include big counts when discounting, but not when building histograms
  // For counts > bins + 2, set weight = -log(bin + 2), gives same result.
  // This avoids issues with converting count to int for very large counts.
  int GetCountBin(double weight, int bins, bool includebig) const {
    double val = -log(double(bins + 2));
    if (weight > val) val = weight;
    int wt = round(exp(-val)) - 1;             // rounding count to integer
    if (wt < 0 || (!includebig && wt > bins))  // if bin should not be assigned
      wt = -1;
    else if (wt > bins)  // include big counts in highest bin discounting
      wt = bins;
    return wt;
  }

  // NB: unigram is order 0 here, etc.
  double Count(int order, int bin) const { return histogram_[order][bin]; }

  // Display input histogram
  void ShowCounts(const std::vector<std::vector<double>> &hist,
                  std::string_view label) const {
    int hi_order = hist.size();
    std::cerr << "Count bin   ";
    std::cerr << label;
    std::cerr << " Counts (";
    for (int order = 0; order < hi_order; ++order) {
      if (order > 0) std::cerr << "/";
      std::cerr << order + 1 << "-grams";
    }
    std::cerr << ")\n";
    for (int bin = 0; bin <= bins_; ++bin) {
      if (bin < bins_)
        std::cerr << "Count = " << bin + 1 << "   ";
      else
        std::cerr << "Count > " << bin << "   ";
      for (int order = 0; order < hi_order; ++order) {
        if (order > 0) std::cerr << "/";
        std::cerr << hist[order][bin];
      }
      std::cerr << "\n";
    }
  }

  // Display internal histogram
  void ShowCounts(std::string_view label) const {
    ShowCounts(histogram_, label);
  }

  // Get an Fst representation of the ngram count-of-counts
  void GetFst(fst::StdMutableFst *fst) const {
    std::unique_ptr<fst::SymbolTable> symbols(new fst::SymbolTable());

    fst->DeleteStates();
    StateId s = fst->AddState();
    fst->SetStart(s);
    int hi_order = histogram_.size();
    symbols->AddSymbol("<epsilon>", 0);
    double sum = kFloatEps;
    for (int order = 0; order < hi_order; ++order) {
      for (int bin = 0; bin <= bins_; ++bin) {
        // label encodes order and bin
        Label label = order * (kMaxBins + 1) + bin + 1;
        std::ostringstream strm;
        strm << "order=" << order << ",bin=" << bin;
        symbols->AddSymbol(strm.str(), label);
        using fst::StdArc;
        StdArc::Weight weight = -log(histogram_[order][bin]);
        if (bin > 0 && weight == StdArc::Weight::Zero()) continue;
        fst->AddArc(s, StdArc(label, label, weight, s));
        sum += histogram_[order][bin];
      }
    }
    fst->SetFinal(s, -log(sum));
    fst->SetInputSymbols(symbols.get());
    fst->SetOutputSymbols(symbols.get());
  }

  // Sets counts from count-of-counts FST
  void SetCounts(const fst::StdFst &fst) {
    histogram_.clear();
    if (fst.Start() == fst::kNoStateId) return;

    for (fst::ArcIterator<fst::StdFst> aiter(fst, 0); !aiter.Done();
         aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      // label encodes order and bin
      int bin = (arc.ilabel - 1) % (kMaxBins + 1);
      int order = (arc.ilabel - 1) / (kMaxBins + 1);
      while (order >= histogram_.size())
        histogram_.push_back(std::vector<double>(bins_ + 1, 0.0));
      if (bin <= bins_)
        histogram_[order][bin] = std::round(std::exp(-arc.weight.Value()));
    }
  }

 private:
  // Find bin for the value provided and increment the histogram for that bin
  void IncrementBinCount(int order, Weight value, const NGramModel<Arc> &model);
  //   int bin = GetCountBin(value, GetBins(), false);
  //   if (bin >= 0)
  //     histogram_[order][bin] = GetIncrement(value, histogram_[order][bin]);
  // }

  std::vector<std::vector<double>> histogram_;  // count histogram for orders
  int bins_;                          // Number of bins for discounting
  NGramContext context_;              // context specification
};

// Find bin for the value provided and increment the histogram for that bin
template <typename Arc>
void NGramCountOfCounts<Arc>::IncrementBinCount(
    int order, NGramCountOfCounts<Arc>::Weight value,
    const NGramModel<Arc> &model) {
  int bin = GetCountBin(model.ScalarValue(value), GetBins(), false);
  if (bin >= 0) ++histogram_[order][bin];
}

// Find bin for the value provided and increment the histogram for that bin
template <>
inline void NGramCountOfCounts<HistogramArc>::IncrementBinCount(
    int order, NGramCountOfCounts<HistogramArc>::Weight value,
    const NGramModel<HistogramArc> &model) {
  int n_bins = NGramCountOfCounts<HistogramArc>::GetBins() + 1;
  int cutoff = value.Length() - 1;
  int length = (cutoff > n_bins) ? n_bins : cutoff;
  for (int k = 0; k < length; k++) {
    histogram_[order][k] += std::exp(-value.Value(k + 1).Value());
  }
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_COUNT_OF_COUNTS_H_
