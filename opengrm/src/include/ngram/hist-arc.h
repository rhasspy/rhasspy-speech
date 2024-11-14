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
// An arc type for histograms.

#ifndef NGRAM_HIST_ARC_H_
#define NGRAM_HIST_ARC_H_

#include <cstddef>
#include <string>

#include <fst/arc.h>

namespace ngram {

// Number of bins in histogram arc weights; set to 2 more than the highest count
// that receives a Katz backoff estimate to allow for bins representing the 0
// count and counts higher than the cutoff.
constexpr size_t kHistogramBins = 7;

// HistogramArc is the Cartesian product of kHistogramBins StdArcs.
struct HistogramArc
    : public fst::PowerArc<fst::StdArc, kHistogramBins> {
  using Base = fst::PowerArc<fst::StdArc, ::ngram::kHistogramBins>;

  using Base::Base;

  static const std::string &Type() {
    static const auto *const type = new std::string("hist");
    return *type;
  }
};

}  // namespace ngram

#endif  // NGRAM_HIST_ARC_H_
