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
// Unsmoothed derived class for smoothing.

#ifndef NGRAM_NGRAM_UNSMOOTHED_H_
#define NGRAM_NGRAM_UNSMOOTHED_H_

#include <vector>

#include <fst/arc.h>
#include <ngram/ngram-make.h>
#include <ngram/ngram-model.h>

namespace ngram {

class NGramUnsmoothed : public NGramMake<fst::StdArc> {
 public:
  // Construct Unsmoothed object, consisting of the FST and some
  // information about the states under the assumption that the FST is a model.
  // Ownership of the FST is retained by the caller.
  explicit NGramUnsmoothed(fst::StdMutableFst *infst, bool backoff = true,
                           bool prefix_norm = true, Label backoff_label = 0,
                           double norm_eps = kNormEps,
                           bool check_consistency = false)
      : NGramMake(infst, backoff, backoff_label, norm_eps, check_consistency,
                  true) {
    SetNormCounts(prefix_norm);
  }

  // Make unsmoothed model
  bool MakeNGramModel() override { return NGramMake::MakeNGramModel(); }

 protected:
  // For unsmoothed model, do not add epsilon to count mass
  double EpsilonMassIfNoneReserved() const override {
    if (norm_counts_.empty())
      return 0;
    else
      return 1;
  }

  // For unsmoothed model, whole count is from high order
  double CalculateHiOrderMass(const std::vector<double> &discounts,
                              double nlog_count) const override {
    return nlog_count;
  }

  double CalculateTotalMass(double nlog_count, StateId st) override {
    if (norm_counts_.empty() ||
        norm_counts_[st] == fst::StdArc::Weight::Zero().Value())
      return nlog_count;
    else {
      if (norm_counts_[st] > nlog_count) {
        if (norm_counts_[st] - nlog_count > NormEps())
          LOG(ERROR) << "prefix normalization too small; using arc sum";
        return nlog_count;
      }
      return norm_counts_[st];
    }
  }

 private:
  void SetNormCounts(bool prefix_norm) {
    if (prefix_norm) FillStateCounts(&norm_counts_);
  }

  std::vector<double> norm_counts_;
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_UNSMOOTHED_H_
