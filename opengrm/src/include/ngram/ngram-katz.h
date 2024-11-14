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
// Katz backoff derived class for smoothing.

#ifndef NGRAM_NGRAM_KATZ_H_
#define NGRAM_NGRAM_KATZ_H_

#include <cmath>
#include <vector>

#include <fst/mutable-fst.h>
#include <ngram/ngram-count-of-counts.h>
#include <ngram/ngram-make.h>
#include <ngram/ngram-model.h>
#include <ngram/util.h>

namespace ngram {

template <class Arc>
class NGramKatz : public NGramMake<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  using NGramMake<Arc>::HiOrder;
  using NGramMake<Arc>::MakeNGramModel;
  using NGramMake<Arc>::ScalarValue;

  // Construct NGramMake object, consisting of the FST and some
  // information about the states under the assumption that the FST is a model.
  // Ownership of the FST is retained by the caller.
  explicit NGramKatz(fst::MutableFst<Arc> *infst, bool backoff = true,
                     Label backoff_label = 0, double norm_eps = kNormEps,
                     bool check_consistency = false, int bins = -1)
      : NGramMake<Arc>(infst, BackOffOrMixed(backoff), backoff_label, norm_eps,
                       check_consistency),
        bins_(bins <= 0 ? 5 : bins),
        count_of_counts_(bins_) {}

  // Normalize n-gram counts and smooth to create an n-gram model
  // Using Katz smoothing methods
  //   number of 'bins' used by Katz (>=1)
  bool MakeNGramModel() override {
    count_of_counts_.CalculateCounts(*this);
    CalculateDiscounts();
    if (FST_FLAGS_v > 0)
      count_of_counts_.ShowCounts(discount_, "Katz discounts");
    return NGramMake<Arc>::MakeNGramModel();
  }

  // Pass in count of counts (rather than computing them)
  void SetCountOfCounts(const fst::StdFst &fst) {
    count_of_counts_.SetCounts(fst);
  }

 protected:
  // Return negative log discounted count for provided negative log count
  double GetDiscount(Weight neglogcount, int order) override;

 private:
  // Katz discount for count r: (r*/r - rnorm) / (1 - rnorm)
  // r* = (r+1) n_{r+1} / n_r   and   rnorm = (bins + 1) n_{bins + 1} / n_1
  // stored with indices starting at 0, so bin 0 is count 1
  void CalculateKatzDiscount(int order, int bin, double rnorm) {
    if (bin < bins_) {
      double rstar = Offset(bin) + 1, denom = 1.0 - rnorm;
      if (count_of_counts_.Count(order, bin + 1) > 0.0)
        rstar *= count_of_counts_.Count(order, bin + 1);
      if (count_of_counts_.Count(order, bin) > 0.0)
        rstar /= count_of_counts_.Count(order, bin);
      discount_[order][bin] = rstar;
      if (Offset(bin) > 0) {
        discount_[order][bin] /= Offset(bin);
        discount_[order][bin] -= rnorm;
        if (denom > 0.0) {
          discount_[order][bin] /= denom;
        }
      }
    }
    if (bin == bins_ || discount_[order][bin] <= 0.0 ||
        discount_[order][bin] >= 1.0) {  // need to provide epsilon discount
      if (bin < bins_ && discount_[order][bin] != 1.0)
        VLOG(1) << "Histograms violating Good-Turing assumptions";
      discount_[order][bin] = 1.0;
      discount_[order][bin] -= kNormEps;  // everything discounted epsilon
    }
  }

  // Calculate Katz discounts for each order
  void CalculateDiscounts() {
    discount_.clear();
    discount_.resize(HiOrder());

    for (int order = 0; order < HiOrder(); ++order) {
      discount_[order].resize(bins_ + 1, 0.0);  // space for bins + 1
      double rnorm = SaneKatzRNorm(order);
      for (int bin = 0; bin <= bins_; ++bin)
        CalculateKatzDiscount(order, bin, rnorm);
    }
  }

  // Ratio of count mass for lowest undiscounted value to singleton count mass
  // We generalize to allow for count pruning which can lead to zero singletons
  // standard form: rnorm = ( bins_ + 1 ) n_{bins_ + 1} / n_1
  // generalized: rnorm = ( bins_ + 1 ) n_{bins_ + 1} / ( k n_k )
  //              for k = lowest count with non-zero observations
  double SaneKatzRNorm(int order) {
    double rnorm_numerator = Offset(bins_), rnorm_denom = 1;
    rnorm_numerator *= count_of_counts_.Count(order, bins_);
    // NB: Offset(0) = 0 means that first count is for n_0, which we skip
    int rlevel = 1 - Offset(0);

    while (rlevel < bins_ && count_of_counts_.Count(order, rlevel) <= 0) {
      ++rlevel;
      ++rnorm_denom;
    }
    rnorm_denom *= count_of_counts_.Count(order, rlevel);
    return rnorm_numerator / rnorm_denom;
  }

  // Offset between index of the count of counts and the count.
  // For usual count of counts, we store n_1 n_2 n_3 etc., so index
  // of n_k is k-1 and Offset(k-1) = k.
  // For count of histograms, we store n_0 n_1 n_2 etc. and so Offset(k) = k.
  int Offset(int bin);

  // User has a choice to make model purely backoff or mixed.
  // Histogram count models are always mixed and user's choice is ignored.
  // This is because histogram method requires a mixed model to
  // correctly account for the probability of ngrams observed in the lattice
  // but that may occur 0 times with positive probability.
  static bool BackOffOrMixed(bool backoff);

  int bins_;                                 // number of bins for discounting
  NGramCountOfCounts<Arc> count_of_counts_;  // count bins for orders
  std::vector<std::vector<double> > discount_;         // discount for bins
};

template <typename Arc>
double NGramKatz<Arc>::GetDiscount(typename Arc::Weight neglogcount_weight,
                                   int order) {
  // Returns the negative log discounted count of the ngram.
  // This is -log(kd_k) = -log(k) - log(d_k) for count k and discount d_k.
  double neglogcount = ScalarValue(neglogcount_weight);  // -log(k)
  double neglogdiscount;
  if (neglogcount == ScalarValue(Arc::Weight::Zero()))  // count = 0
    return neglogcount;
  int bin = count_of_counts_.GetCountBin(neglogcount, bins_, true);
  if (bin >= 0) {
    neglogdiscount = -log(discount_[order][bin]);
  } else {
    NGRAMERROR() << "NGramKatz: No discount bin for discounting";
    NGramModel<Arc>::SetError();
    neglogdiscount = ScalarValue(Arc::Weight::Zero());
  }
  return neglogcount + neglogdiscount;  // -log(k) - log(d_k) = -log(kd_k).
}

template <>
inline double NGramKatz<HistogramArc>::GetDiscount(
    HistogramArc::Weight neglogcount_weight, int order) {
  // Returns the negative log discounted expected count of the ngram.
  // This is -log[sum_{k > 0} q(k)c^*(k)] where q(k) is the probability of
  // observing k instances of the ngram and c^*(k) is the discounted count
  // for count k.  Note that  c^*(k) = kd_k for some discount
  // factor d_k. For k > K, d_k = 1, i.e., above the threshold K, counts are
  // undiscounted.  Let lambda = sum_{k > 0} q(k)k, i.e., the expected count
  // of the ngram.  Then
  // sum_{k > 0} q(k)c^*(k) = sum_{k > 0} q(k)kd_k
  //                        = sum_{k > 0} q(k)k(1 + (d_k - 1))
  //                        = lambda + sum_{k > 0} q(k)k(d_k - 1)
  //                        = lambda + sum_{k=1}^K q(k)k(d_k - 1)
  //                        = lambda - sum_{k=1}^K q(k)k(1 - d_k)
  // since d_k = 1 for k > K, hence the last term only required to sum up to K.
  // The scalar value of the histogram arc weight is the expected count.
  double neg_log_lambda = ScalarValue(neglogcount_weight);

  // Accumulates neg log of the final term above: sum_{k=1}^K q(k)k(1 - d_k).
  // To sum over k from 1 to K, we initialize value to -log(0.0).
  double neg_log_discounted = ScalarValue(HistogramArc::Weight::Zero());
  int cutoff = neglogcount_weight.Length() - 1;
  int length = (cutoff > bins_) ? bins_ : cutoff;  // length = K + 1
  for (int bin = 1; bin < length; bin++) {
    // Accumulates -log(q(k)) - log(k) - log(1 - d_k).
    neg_log_discounted = NegLogSum(
        neg_log_discounted, neglogcount_weight.Value(bin + 1).Value() -
                                log(1 - discount_[order][bin]) - log(bin));
  }

  // Truncate result for additional numerical stability
  if (neg_log_discounted > 99.0) {
    neg_log_discounted = ScalarValue(HistogramArc::Weight::Zero());
  }

  // Reserves epsilon from higher order probs if no mass was discounted.
  if (neg_log_discounted == ScalarValue(HistogramArc::Weight::Zero())) {
    neg_log_discounted = -log(1 - discount_[order][bins_]) + neg_log_lambda;
  }

  // Returns -log[ lambda - sum_{k=1}^K q(k)k(1 - d_k) ].
  return NegLogDiff(neg_log_lambda, neg_log_discounted);
}

template <typename Arc>
int NGramKatz<Arc>::Offset(int bin) {
  return bin + 1;
}

template <>
inline int NGramKatz<HistogramArc>::Offset(int bin) {
  return bin;
}

template <typename Arc>
bool NGramKatz<Arc>::BackOffOrMixed(bool backoff) {
  return backoff;
}

template <>
inline bool NGramKatz<HistogramArc>::BackOffOrMixed(bool backoff) {
  return false;
}

}  // namespace ngram

#endif  // NGRAM_NGRAM_KATZ_H_
