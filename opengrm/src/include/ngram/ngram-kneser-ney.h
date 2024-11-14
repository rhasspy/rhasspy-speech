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
// Kneser-Ney derived class for smoothing.

#ifndef NGRAM_NGRAM_KNESER_NEY_H_
#define NGRAM_NGRAM_KNESER_NEY_H_

#include <vector>

#include <fst/mutable-fst.h>
#include <ngram/ngram-absolute.h>
#include <ngram/ngram-model.h>

namespace ngram {

class NGramKneserNey : public NGramAbsolute {
 public:
  // Construct NGramKneserNey object, consisting of the FST and some
  // information about the states under the assumption that the FST is a model.
  // Ownership of the FST is retained by the caller.
  explicit NGramKneserNey(fst::StdMutableFst *infst, bool backoff = false,
                          Label backoff_label = 0, double norm_eps = kNormEps,
                          bool check_consistency = false,
                          double parameter = -1.0, int bins = -1)
      : NGramAbsolute(infst, backoff, backoff_label, norm_eps,
                      check_consistency, parameter, bins) {}

  // Smooth model according to 'method' and parameters.
  bool MakeNGramModel() override;

 private:
  // Update arc and final values, either initializing or incrementing
  bool UpdateKneserNeyCounts(StateId st, bool increment);

  // Calculate update value, either for incrementing or removing hi order
  double CalcKNValue(bool increment, double hi_order_value,
                     double lo_order_value);

  // Update the backoff arc with total count
  bool UpdateTotalCount(StateId st);

  // Modify lower order counts according to Kneser Ney formula
  void AssignKneserNeyCounts();
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_KNESER_NEY_H_
