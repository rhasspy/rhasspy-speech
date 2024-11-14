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
// Mapper to/from histogram arc.

#ifndef NGRAM_HIST_MAPPER_H_
#define NGRAM_HIST_MAPPER_H_

#include <array>
#include <cmath>
#include <cstdint>

#include <fst/arc-map.h>
#include <fst/power-weight.h>
#include <fst/properties.h>
#include <ngram/hist-arc.h>

namespace ngram {

// Mapper from HistogramArc to StdArc.
//
// The value at index 0 of HistogramArc is a raw expected count
// which becomes the value of the mapped StdArc.
struct ToStdArcMapper {
  using FromArc = HistogramArc;
  using ToArc = fst::StdArc;

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, arc.olabel, arc.weight.Value(0).Value(),
                 arc.nextstate);
  }

  constexpr fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  constexpr fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  constexpr fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64_t Properties(uint64_t props) const {
    return props & fst::kWeightInvariantProperties;
  }
};

// Mapper from Arc to HistogramArc.
//
// The value at index 0 of HistogramArc is the raw expected count
// (weight value of the original arc). Note that HistogramArc stores
// frequencies of 0, 1, 2, ..., K at indices 1, 2, 3, ... , K + 1.
//
// For instance, if expected count is 1.1, then under this mapping
// HistogramArc will contain the following:
//
// value:  -log(1.1) -log(0) -log(1.1-1) -log(1.1-1) -log(0) ...
// index:         0        1           2           3       4 ...
template <class Arc>
class ToHistogramMapper {
 public:
  using FromArc = Arc;
  using ToArc = HistogramArc;
  using Weight =
      fst::PowerWeight<fst::TropicalWeight, ::ngram::kHistogramBins>;

  ToArc operator()(const Arc &arc) const {
    std::array<fst::TropicalWeight, ::ngram::kHistogramBins> weights;
    weights.fill(fst::TropicalWeight::Zero());
    const double val = arc.weight.Value();
    weights[0] = val;
    const double round_down = floor(exp(-val));
    const double round_up = round_down + 1;
    const auto index = static_cast<int>(round_up);
    if (index < weights.size() - 1) {
      weights[index + 1] = -log(exp(-val) - round_down);
    }
    if (index && index < weights.size()) {
      weights[index] = -log(round_up - exp(-val));
    }
    return ToArc(arc.ilabel, arc.olabel, Weight(weights.begin(), weights.end()),
                 arc.nextstate);
  }

  constexpr fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  constexpr fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  constexpr fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64_t Properties(uint64_t props) const {
    return props & fst::kWeightInvariantProperties;
  }
};

}  // namespace ngram

#endif  // NGRAM_HIST_MAPPER_H_
