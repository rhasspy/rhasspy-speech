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
// Absolute Discounting derived class for smoothing.

#include <ngram/ngram-absolute.h>

#include <cmath>
#include <vector>

#include <fst/arc.h>
#include <fst/arcsort.h>
#include <ngram/ngram-make.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

using ::fst::StdArc;
using ::fst::StdILabelCompare;

// Normalize n-gram counts and smooth to create an n-gram model
// Using Absolute Discounting Methods
//  'parameter': discount D
//   number of 'bins' used by Absolute Discounting (>=1)
bool NGramAbsolute::MakeNGramModel() {
  count_of_counts_.CalculateCounts(*this);
  CalculateDiscounts();
  if (FST_FLAGS_v > 0)
    count_of_counts_.ShowCounts(discount_, "Absolute discounts");
  return NGramMake::MakeNGramModel();
}

// Calculate discounts for each order
void NGramAbsolute::CalculateDiscounts() {
  discount_.clear();
  discount_.resize(HiOrder());

  for (int order = 0; order < HiOrder(); ++order) {
    discount_[order].resize(bins_ + 1, 0.0);  // space for bins + 1
    for (int bin = 0; bin < bins_; ++bin) CalculateAbsoluteDiscount(order, bin);
    // counts higher than largest bin are discounted at largest bin rate
    discount_[order][bins_] = discount_[order][bins_ - 1];
  }
}

// Return negative log discounted count for provided negative log count
double NGramAbsolute::GetDiscount(Weight neglogcount_weight, int order) {
  double neglogcount = ScalarValue(neglogcount_weight);
  double discounted = neglogcount, neglogdiscount;
  if (neglogcount == StdArc::Weight::Zero().Value())  // count = 0
    return neglogcount;
  int bin = count_of_counts_.GetCountBin(neglogcount, bins_, true);
  if (bin >= 0) {
    neglogdiscount = -log(discount_[order][bin]);
    if (neglogdiscount <= neglogcount)              // c - D <= 0
      discounted = StdArc::Weight::Zero().Value();  // set count to 0
    else
      discounted = NegLogDiff(neglogcount, neglogdiscount);  // subtract
  } else {
    NGRAMERROR() << "NGramAbsolute: No discount bin for discounting";
    NGramModel::SetError();
  }
  return discounted;
}

}  // namespace ngram
