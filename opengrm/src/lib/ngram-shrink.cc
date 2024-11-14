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
#include <ngram/ngram-shrink.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <fst/arc.h>
#include <fst/mutable-fst.h>
#include <ngram/ngram-context-prune.h>
#include <ngram/ngram-count-prune.h>
#include <ngram/ngram-list-prune.h>
#include <ngram/ngram-relentropy.h>
#include <ngram/ngram-seymore-shrink.h>

namespace ngram {

namespace impl {
namespace {

// Checks several of the parameters to make sure they are consistent.
void CheckShrinkOptions(std::string_view method, int64_t target_num,
                        bool full_context, int min_order) {
  if (target_num >= 0) {
    if (method == "context_prune" || method == "count_prune") {
      LOG(ERROR) << "Target number of ngrams requires \"relative_entropy\" or "
                 << "\"seymore\" shrinking";
    } else if (!full_context) {
      LOG(ERROR) << "Target number of ngrams requires a full context";
    }
  }
  if (method == "context_prune" && min_order > 2) {
    LOG(ERROR) << "Context prune does not support a minimum order parameter";
  }
}

}  // namespace
}  // namespace impl

bool NGramShrinkModel(fst::StdMutableFst *fst, const std::string &method,
                      double tot_uni, double theta, int64_t target_num,
                      int32_t min_order, const std::string &count_pattern,
                      std::string_view context_pattern, int shrink_opt,
                      fst::StdArc::Label backoff_label, double norm_eps,
                      bool check_consistency) {
  return NGramShrinkModel(
      fst, method, std::set<std::vector<fst::StdArc::Label>>(), tot_uni,
      theta, target_num, min_order, count_pattern, context_pattern, shrink_opt,
      backoff_label, norm_eps, check_consistency);
}

// Makes model from NGram model FST with StdArc counts.
bool NGramShrinkModel(
    fst::StdMutableFst *fst, const std::string &method,
    const std::set<std::vector<fst::StdArc::Label>> &ngram_list,
    double tot_uni, double theta, int64_t target_num, int32_t min_order,
    const std::string &count_pattern, std::string_view context_pattern,
    int shrink_opt, fst::StdArc::Label backoff_label, double norm_eps,
    bool check_consistency) {
  bool full_context = context_pattern.empty();
  impl::CheckShrinkOptions(method, target_num, full_context, min_order);
  if (method == "list_prune") {
    NGramListPrune ngramsh(fst, ngram_list, shrink_opt, tot_uni, backoff_label,
                           norm_eps, check_consistency);
    ngramsh.ShrinkNGramModel(min_order);
    return !ngramsh.Error();
  } else if (method == "context_prune") {
    NGramContextPrune ngramsh(fst, context_pattern, shrink_opt, tot_uni,
                              backoff_label, norm_eps, check_consistency);
    ngramsh.ShrinkNGramModel();
    return !ngramsh.Error();
  } else if (method == "count_prune") {
    if (full_context) {
      NGramCountPrune ngramsh(fst, count_pattern, shrink_opt, tot_uni,
                              backoff_label, norm_eps, check_consistency);
      ngramsh.ShrinkNGramModel(min_order);
      return !ngramsh.Error();
    } else {
      NGramContextCountPrune ngramsh(fst, count_pattern, context_pattern,
                                     shrink_opt, tot_uni, backoff_label,
                                     norm_eps, check_consistency);
      ngramsh.ShrinkNGramModel(min_order);
      return !ngramsh.Error();
    }
  } else if (method == "relative_entropy") {
    if (full_context) {
      NGramRelEntropy ngramsh(fst, theta, shrink_opt, tot_uni, backoff_label,
                              norm_eps, check_consistency);
      if (target_num >= 0) {
        ngramsh.CalculateTheta(target_num, min_order);
        if (ngramsh.Error()) return false;
      }
      ngramsh.ShrinkNGramModel(min_order);
      return !ngramsh.Error();
    } else {
      NGramContextRelEntropy ngramsh(fst, theta, context_pattern, shrink_opt,
                                     tot_uni, backoff_label, norm_eps,
                                     check_consistency);
      ngramsh.ShrinkNGramModel(min_order);
      return !ngramsh.Error();
    }
  } else if (method == "seymore") {
    if (full_context) {
      NGramSeymoreShrink ngramsh(fst, theta, shrink_opt, tot_uni,
                                 backoff_label, norm_eps, check_consistency);
      if (target_num >= 0) {
        ngramsh.CalculateTheta(target_num, min_order);
        if (ngramsh.Error()) return false;
      }
      ngramsh.ShrinkNGramModel(min_order);
      return !ngramsh.Error();
    } else {
      NGramContextSeymoreShrink ngramsh(fst, theta, context_pattern, shrink_opt,
                                        tot_uni, backoff_label, norm_eps,
                                        check_consistency);
      ngramsh.ShrinkNGramModel(min_order);
      return !ngramsh.Error();
    }
  }
  LOG(ERROR) << "Unknown shrink method: " << method;
  return false;
}

}  // namespace ngram
