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
// Implements the algorithm for using lexicographic semirings discussed in:
//
// Brian Roark, Richard Sproat and Izhak Shafran. 2011. "Lexicographic
// Semirings for Exact Automata Encoding of Sequence Models". ACL-HLT
// 2011, Portland, OR.
//
// The conversion back and forth between the W and the
// Lexicographic<W, W> semiring is handled by a map (fst::ArcMap). Also
// provided is a lightweight class to perform the composition, epsilon
// removal and determinization required by the method.

#ifndef NGRAM_LEXICOGRAPHIC_MAP_H_
#define NGRAM_LEXICOGRAPHIC_MAP_H_

#include <cstdint>

#include <fst/arc-map.h>
#include <fst/arc.h>
#include <fst/compose.h>
#include <fst/determinize.h>
#include <fst/fst.h>
#include <fst/properties.h>
#include <fst/rmepsilon.h>
#include <fst/vector-fst.h>
#include <ngram/ngram-model.h>

namespace ngram {

// The penalty for the first dimension of the lexicographic weight on
// the phi arc implemented as an epsilon arc. In the most common
// usage, where W=Tropical, this could just be 1 (or any positive
// number). But we make it 2 in case someone uses a semiring where
// fst::Power is really power and not (as in the Tropical),
// multiplication.

static const int32_t kBackoffPenalty = 2;

template <class A>
struct ToLexicographicMapper {
  using FromArc = A;
  using W = typename A::Weight;

  using ToArc = fst::LexicographicArc<W, W>;
  using LW = typename ToArc::Weight;

  explicit ToLexicographicMapper(NGramModel<fst::StdArc>* in_model)
      : model(in_model) {}

  ToArc operator()(const FromArc& arc) const {
    // 'Super-non-final' arc
    if (arc.nextstate == fst::kNoStateId && arc.weight == W::Zero()) {
      return ToArc(0, 0, LW(W::Zero(), arc.weight), fst::kNoStateId);
      // 'Super-final' arc
    } else if (arc.nextstate == fst::kNoStateId) {
      return ToArc(0, 0, LW(W::One(), arc.weight), fst::kNoStateId);
      // Epsilon label: in this case if it's an LM we need to check the
      // order of the backoff, unless this is Zero(), which can happen
      // in some topologies.
    } else if (arc.ilabel == 0 && arc.olabel == 0 && model) {
      if (arc.weight == W::Zero())
        return ToArc(arc.ilabel, arc.olabel, LW(arc.weight, arc.weight),
                     arc.nextstate);
      int expt = model->HiOrder() - model->StateOrder(arc.nextstate);
      return ToArc(arc.ilabel, arc.olabel,
                   LW(fst::Power<W>(kBackoffPenalty, expt), arc.weight),
                   arc.nextstate);
      // Real arc (called an "ngram" arc in Roark et al. 2011)
    } else {
      return ToArc(arc.ilabel, arc.olabel, LW(W::One(), arc.weight),
                   arc.nextstate);
    }
  }

  fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64_t Properties(uint64_t props) const {
    return fst::ProjectProperties(props, true) &
           fst::kWeightInvariantProperties;
  }

  NGramModel<fst::StdArc>* model;
};

template <class A>
struct FromLexicographicMapper {
  using ToArc = A;
  using W = typename A::Weight;
  using FromArc = fst::LexicographicArc<W, W>;
  using LW = typename FromArc::Weight;

  ToArc operator()(const FromArc& arc) const {
    // 'Super-final' arc and 'Super-non-final' arc
    if (arc.nextstate == fst::kNoStateId) {
      return ToArc(0, 0, W(arc.weight.Value2()), fst::kNoStateId);
    } else {
      return ToArc(arc.ilabel, arc.olabel, W(arc.weight.Value2()),
                   arc.nextstate);
    }
  }

  fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64_t Properties(uint64_t props) const {
    return fst::ProjectProperties(props, true) &
           fst::kWeightInvariantProperties;
  }
};

template <class A>
class LexicographicRescorer {
 public:
  using ToMapper = ToLexicographicMapper<A>;
  using FromMapper = FromLexicographicMapper<A>;

  using W = typename A::Weight;
  using ToArc = typename ToMapper::ToArc;

  LexicographicRescorer(fst::MutableFst<A>* lm,
                        NGramModel<fst::StdArc>* model) {
    fst::ArcMap(*lm, &lm_, ToMapper(model));
  }

  fst::VectorFst<A>* Rescore(fst::MutableFst<A>* lattice);

 private:
  fst::VectorFst<ToArc> lm_;
  fst::VectorFst<A> result_;
};

template <class A>
fst::VectorFst<A>* LexicographicRescorer<A>::Rescore(
    fst::MutableFst<A>* lattice) {
  fst::VectorFst<ToArc> lexlat;
  fst::ArcMap(*lattice, &lexlat, ToMapper(nullptr));
  fst::VectorFst<ToArc> comp;
  fst::Compose(lexlat, lm_, &comp);
  fst::RmEpsilon(&comp);
  fst::VectorFst<ToArc> det;
  fst::Determinize(comp, &det);
  fst::ArcMap(det, &result_, FromMapper());
  return &result_;
}

using StdLexicographicRescorer = LexicographicRescorer<fst::StdArc>;

}  // namespace ngram

#endif  // NGRAM_LEXICOGRAPHIC_MAP_H_
