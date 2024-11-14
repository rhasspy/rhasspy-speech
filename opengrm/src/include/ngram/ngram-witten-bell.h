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
// Witten-Bell derived class for smoothing.

#ifndef NGRAM_NGRAM_WITTEN_BELL_H_
#define NGRAM_NGRAM_WITTEN_BELL_H_

#include <cmath>
#include <vector>

#include <fst/arc.h>
#include <ngram/ngram-make.h>
#include <ngram/ngram-model.h>

namespace ngram {

class NGramWittenBell : public NGramMake<fst::StdArc> {
 public:
  // Construct NGramMake object, consisting of the FST and some
  // information about the states under the assumption that the FST is a model.
  // Ownership of the FST is retained by the caller.
  explicit NGramWittenBell(fst::StdMutableFst *infst, bool backoff = false,
                           Label backoff_label = 0, double norm_eps = kNormEps,
                           bool check_consistency = false,
                           double parameter = 1.0)
      : NGramMake(infst, backoff, backoff_label, norm_eps, check_consistency),
        parameter_(parameter) {}

  // Smooth model according to 'method' and parameters.
  bool MakeNGramModel() override { return NGramMake::MakeNGramModel(); }

 protected:
  // No discount, hence hi order mass is count
  double CalculateHiOrderMass(const std::vector<double> &discounts,
                              double nlog_count) const override {
    return nlog_count;
  }

  // Return Normalization constant for Witten Bell:
  // -log c(h) + K |{w:c(hw)>0}|
  double CalculateTotalMass(double nlog_count, StateId st) override {
    double ngcount = GetFst().NumArcs(st) - 1;
    if (GetFst().Final(st).Value() != fst::StdArc::Weight::Zero().Value())
      ++ngcount;  // count </s> if p > 0
    // Count mass allocated to lower order estimates: K |{w:c(hw)>0}|
    double low_order_mass = -log(ngcount) - log(parameter_);
    return NegLogSum(nlog_count, low_order_mass);
  }

 private:
  double parameter_;  // Witten-Bell K
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_WITTEN_BELL_H_
