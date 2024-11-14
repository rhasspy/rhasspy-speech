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
// Classes to generate random sentences from an LM or more generally
// paths through any FST where epsilons are treated as failure transitions.

#ifndef NGRAM_NGRAM_RANDGEN_H_
#define NGRAM_NGRAM_RANDGEN_H_

#include <sys/types.h>
#include <unistd.h>

#include <ctime>
#include <map>
#include <utility>

#include <fst/accumulator.h>

// Faster multinomial sampling possible if Gnu Scientific Library available.
#ifdef HAVE_GSL
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#endif  // HAVE_GSL

#include <fst/fst.h>
#include <fst/randgen.h>
#include <ngram/util.h>

namespace ngram {

// Same as FastLogProbArcSelector but treats *all* epsilons as
// failure transitions that have a backoff weight. The LM must
// be fully normalized.
template <class A>
class NGramArcSelector {
 public:
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;

  explicit NGramArcSelector(int seed = time(nullptr) + getpid()) : seed_(seed) {
    srand(seed);
  }

  // Samples one transition.
  size_t operator()(const fst::Fst<A> &fst, StateId s, double total_prob,
                    fst::CacheLogAccumulator<A> *accumulator) const {
    double r = rand() / (RAND_MAX + 1.0);
    // In effect, subtract out excess mass from the cumulative distribution.
    // Requires the backoff epsilon be the initial transition.
    double z = r + total_prob - 1.0;
    if (z <= 0.0) return 0;
    fst::ArcIterator<fst::Fst<A> > aiter(fst, s);
    return accumulator->LowerBound(-log(z), &aiter);
  }

  int Seed() const { return seed_; }

 private:
  int seed_;
  fst::WeightConvert<Weight, fst::LogWeight> to_log_weight_;
};

}  // namespace ngram

namespace fst {

// Specialization for NGramArcSelector.
template <class A>
class ArcSampler<A, ngram::NGramArcSelector<A> > {
 public:
  typedef ngram::NGramArcSelector<A> S;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;
  typedef typename A::Label Label;
  typedef CacheLogAccumulator<A> C;

  ArcSampler(const fst::Fst<A> &fst, const S &arc_selector,
             int max_length = INT_MAX)
      : fst_(fst),
        arc_selector_(arc_selector),
        max_length_(max_length),
        matcher_(fst_, MATCH_INPUT) {
    // Ensure the input FST has any epsilons as the initial transitions.
    if (!fst_.Properties(kILabelSorted, true))
      NGRAMERROR() << "ArcSampler:  is not input-label sorted";
    accumulator_.reset(new C());
    accumulator_->Init(fst);
#ifdef HAVE_GSL
    rng_ = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rng_, arc_selector.Seed());
#endif  // HAVE_GSL
  }

  ArcSampler(const ArcSampler<A, S> &sampler, const fst::Fst<A> *fst = 0)
      : fst_(fst ? *fst : sampler.fst_),
        arc_selector_(sampler.arc_selector_),
        max_length_(sampler.max_length_),
        matcher_(fst_, MATCH_INPUT) {
    if (fst) {
      accumulator_.reset(new C());
      accumulator_->Init(*fst);
    } else {  // shallow copy
      accumulator_.reset(new C(*sampler.accumulator_));
    }
  }

  ~ArcSampler() {
#ifdef HAVE_GSL
    gsl_rng_free(rng_);
#endif  // HAVE_GSL
  }

  bool Sample(const RandState<A> &rstate) {
    sample_map_.clear();
    forbidden_labels_.clear();

    if ((fst_.NumArcs(rstate.state_id) == 0 &&
         fst_.Final(rstate.state_id) == Weight::Zero()) ||
        rstate.length == max_length_) {
      Reset();
      return false;
    }

    double total_prob = TotalProb(rstate.state_id);

#ifdef HAVE_GSL
    if (fst_.NumArcs(rstate.state_id) + 1 < rstate.nsamples) {
      Weight numer_weight, denom_weight;
      BackoffWeight(rstate.state_id, total_prob, &numer_weight, &denom_weight);
      MultinomialSample(rstate, numer_weight);
      Reset();
      return true;
    }
#endif  // HAVE_GSL

    fst::ArcIterator<fst::Fst<A> > aiter(fst_, rstate.state_id);

    for (size_t i = 0; i < rstate.nsamples; ++i) {
      size_t pos = 0;
      Label label = kNoLabel;
      do {
        pos = arc_selector_(fst_, rstate.state_id, total_prob,
                            accumulator_.get());
        if (pos < fst_.NumArcs(rstate.state_id)) {
          aiter.Seek(pos);
          label = aiter.Value().ilabel;
        } else {
          label = kNoLabel;
        }
      } while (ForbiddenLabel(label, rstate));
      ++sample_map_[pos];
    }
    Reset();
    return true;
  }

  bool Done() const { return sample_iter_ == sample_map_.end(); }
  void Next() { ++sample_iter_; }
  std::pair<size_t, size_t> Value() const { return *sample_iter_; }
  void Reset() { sample_iter_ = sample_map_.begin(); }
  bool Error() const { return false; }

 private:
  double TotalProb(StateId s) {
    // Get cumulative weight at the state.
    fst::ArcIterator<fst::Fst<A> > aiter(fst_, s);
    accumulator_->SetState(s);
    Weight total_weight =
        accumulator_->Sum(fst_.Final(s), &aiter, 0, fst_.NumArcs(s));
    return exp(-to_log_weight_(total_weight).Value());
  }

  void BackoffWeight(StateId s, double total_prob, Weight *numer_weight,
                     Weight *denom_weight);

#ifdef HAVE_GSL
  void MultinomialSample(const RandState<A> &rstate, Weight fail_weight);
#endif  // HAVE_GSL

  bool ForbiddenLabel(Label l, const RandState<A> &rstate);

  const fst::Fst<A> &fst_;
  const S &arc_selector_;
  int max_length_;

  // Stores (N, K) as described for Value().
  std::map<size_t, size_t> sample_map_;
  std::map<size_t, size_t>::const_iterator sample_iter_;
  std::unique_ptr<C> accumulator_;

#ifdef HAVE_GSL
  gsl_rng *rng_;                   // GNU Sci Lib random number generator
  std::vector<double> pr_;         // multinomial parameters
  std::vector<unsigned int> pos_;  // sample positions
  std::vector<unsigned int> n_;    // sample counts
#endif                             // HAVE_GSL

  WeightConvert<fst::Log64Weight, Weight> to_weight_;
  WeightConvert<Weight, fst::Log64Weight> to_log_weight_;
  std::set<Label>
      forbidden_labels_;  // labels forbidden for failure transitions
  Matcher<fst::Fst<A> > matcher_;
};

// Finds and decomposes the backoff probability into its numerator and
// denominator.
template <class A>
void ArcSampler<A, ngram::NGramArcSelector<A> >::BackoffWeight(
    StateId s, double total, Weight *numer_weight, Weight *denom_weight) {
  // Get backoff prob.
  double backoff = 0.0;
  matcher_.SetState(s);
  matcher_.Find(0);
  for (; !matcher_.Done(); matcher_.Next()) {
    const A &arc = matcher_.Value();
    if (arc.ilabel != kNoLabel) {  // not an implicit epsilon loop
      backoff = exp(-to_log_weight_(arc.weight).Value());
      break;
    }
  }

  if (backoff == 0.0) {  // no backoff transition
    *numer_weight = Weight::Zero();
    *denom_weight = Weight::Zero();
    return;
  }

  // total = 1 - numer + backoff
  double numer = 1.0 + backoff - total;
  *numer_weight = to_weight_(-log(numer));

  // backoff = numer/denom
  double denom = numer / backoff;
  *denom_weight = to_weight_(-log(denom));
}

#ifdef HAVE_GSL
template <class A>
void ArcSampler<A, ngram::NGramArcSelector<A> >::MultinomialSample(
    const RandState<A> &rstate, Weight fail_weight) {
  pr_.clear();
  pos_.clear();
  n_.clear();
  size_t pos = 0;
  for (fst::ArcIterator<fst::Fst<A> > aiter(fst_, rstate.state_id);
       !aiter.Done(); aiter.Next(), ++pos) {
    const A &arc = aiter.Value();
    if (!ForbiddenLabel(arc.ilabel, rstate)) {
      pos_.push_back(pos);
      Weight weight = arc.ilabel == 0 ? fail_weight : arc.weight;
      pr_.push_back(exp(-to_log_weight_(weight).Value()));
    }
  }
  if (fst_.Final(rstate.state_id) != Weight::Zero() &&
      !ForbiddenLabel(kNoLabel, rstate)) {
    pos_.push_back(pos);
    pr_.push_back(exp(-to_log_weight_(fst_.Final(rstate.state_id)).Value()));
  }

  if (rstate.nsamples < UINT_MAX) {
    n_.resize(pr_.size());
    gsl_ran_multinomial(rng_, pr_.size(), rstate.nsamples, &(pr_[0]), &(n_[0]));
    for (size_t i = 0; i < n_.size(); ++i)
      if (n_[i] != 0) sample_map_[pos_[i]] = n_[i];
  } else {
    for (size_t i = 0; i < pr_.size(); ++i)
      sample_map_[pos_[i]] = ceil(pr_[i] * rstate.nsamples);
  }
}
#endif  // HAVE_GSL

template <class A>
bool ArcSampler<A, ngram::NGramArcSelector<A> >::ForbiddenLabel(
    Label l, const RandState<A> &rstate) {
  if (l == 0) return false;

  if (fst_.NumArcs(rstate.state_id) > rstate.nsamples) {
    for (const RandState<A> *rs = &rstate; rs->parent != nullptr;
         rs = rs->parent) {
      StateId parent_id = rs->parent->state_id;
      fst::ArcIterator<fst::Fst<A> > aiter(fst_, parent_id);
      aiter.Seek(rs->select);
      if (aiter.Value().ilabel != 0)  // not backoff transition
        return false;

      if (l == kNoLabel) {  // super-final label
        return fst_.Final(parent_id) != Weight::Zero();
      } else {
        matcher_.SetState(parent_id);
        if (matcher_.Find(l)) return true;
      }
    }
    return false;
  } else {
    if (forbidden_labels_.empty()) {
      for (const RandState<A> *rs = &rstate; rs->parent != nullptr;
           rs = rs->parent) {
        StateId parent_id = rs->parent->state_id;
        fst::ArcIterator<fst::Fst<A> > aiter(fst_, parent_id);
        aiter.Seek(rs->select);
        if (aiter.Value().ilabel != 0)  // not backoff transition
          break;

        for (aiter.Reset(); !aiter.Done(); aiter.Next()) {
          Label l = aiter.Value().ilabel;
          if (l != 0) forbidden_labels_.insert(l);
        }

        if (fst_.Final(parent_id) != Weight::Zero())
          forbidden_labels_.insert(kNoLabel);
      }
    }
    return forbidden_labels_.count(l) > 0;
  }
}

}  // namespace fst

#endif  // NGRAM_NGRAM_RANDGEN_H_
