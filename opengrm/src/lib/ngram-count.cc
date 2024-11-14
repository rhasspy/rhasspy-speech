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
// Get counts from input strings.

#include <ngram/ngram-count.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fst/extensions/far/far.h>
#include <fst/arc-map.h>
#include <fst/arc.h>
#include <fst/arcsort.h>
#include <fst/float-weight.h>
#include <fst/mutable-fst.h>
#include <fst/properties.h>
#include <fst/symbol-table.h>
#include <fst/vector-fst.h>
#include <ngram/hist-arc.h>
#include <ngram/hist-mapper.h>
#include <ngram/ngram-hist-merge.h>

namespace ngram {

// Rounds -log count to values corresponding to the rounded integer count;
// reduces small floating point precision issues when dealing with int counts;
// primarily for testing that methods for deriving the same model are identical.
void RoundCountsToInt(fst::StdMutableFst *fst) {
  for (size_t s = 0; s < fst->NumStates(); ++s) {
    for (fst::MutableArcIterator<fst::StdMutableFst> aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      auto weight = std::round(std::exp(-arc.weight.Value()));
      arc.weight = -std::log(weight);
      aiter.SetValue(arc);
    }
    if (fst->Final(s) != fst::StdArc::Weight::Zero()) {
      auto weight = std::round(std::exp(-fst->Final(s).Value()));
      fst->SetFinal(s, -std::log(weight));
    }
  }
}

// Returns an n-gram string and double count from a (history, ngram) pair.
double GetNGramAndCount(
    const std::pair<std::vector<int>, std::pair<int, double>> &ngram_count,
    std::string *ngram, const fst::SymbolTable &syms) {
  std::vector<int> ngram_history = ngram_count.first;
  *ngram = "";
  for (size_t i = 0; i < ngram_history.size(); ++i) {
    std::string symbol =
        ngram_history[i] > 0 ? syms.Find(ngram_history[i]) : "<s>";
    *ngram += symbol + " ";
  }
  if (ngram_count.second.first > 0) {
    *ngram += syms.Find(ngram_count.second.first);
  } else {
    *ngram += "</s>";
  }
  return ngram_count.second.second;
}

// Gets ngram counts for the next fst in far_reader.
bool GetCounts(const std::string &countname,
               NGramCounter<fst::Log64Weight> *ngram_counter,
               fst::FarReader<fst::StdArc> *far_reader, int fstnumber,
               fst::SymbolTable *syms) {
  std::unique_ptr<const fst::StdVectorFst> ifst(
      new fst::StdVectorFst(*far_reader->GetFst()));
  if (!ifst) {
    LOG(ERROR) << countname << ": unable to read fst #" << fstnumber;
    return false;
  }

  bool counted = false;
  if (ifst->Properties(fst::kString, true)) {
    counted = ngram_counter->Count(*ifst);
  } else {
    fst::VectorFst<fst::Log64Arc> log_ifst;
    ArcMap(*ifst, &log_ifst, fst::StdToLog64Mapper());
    counted = ngram_counter->Count(&log_ifst);
  }
  if (!counted) LOG(ERROR) << countname << ": fst #" << fstnumber << " skipped";
  if (ifst->InputSymbols() != nullptr && syms->NumSymbols() == 0) {
    // Retains symbol table if available and not yet retained.
    *syms = *ifst->InputSymbols();
  }

  return true;
}

// Builds a count WFST from a single input.
bool GetSingleCountFst(fst::FarReader<fst::StdArc> *far_reader,
                       fst::StdMutableFst *fst, int fstnumber, int order,
                       bool epsilon_as_backoff) {
  NGramCounter<fst::Log64Weight> ngram_counter(order, epsilon_as_backoff);
  if (ngram_counter.Error()) {
    return false;
  }
  fst::SymbolTable syms;
  if (far_reader->Done() ||
      !GetCounts("ngramhistcount", &ngram_counter, far_reader, fstnumber,
                 &syms)) {
    return false;
  }
  ngram_counter.GetFst(fst);
  fst::ArcSort(fst, fst::StdILabelCompare());
  if (syms.NumSymbols() > 0) {
    fst->SetInputSymbols(&syms);
    fst->SetOutputSymbols(&syms);
  }
  return true;
}

// Computes counts using the HistogramArc template.
bool GetNGramHistograms(fst::FarReader<fst::StdArc> *far_reader,
                        fst::VectorFst<HistogramArc> *fst,
                        int order, bool epsilon_as_backoff, int backoff_label,
                        double norm_eps, bool check_consistency, bool normalize,
                        double alpha, double beta) {
  int fstnumber = 1;
  std::unique_ptr<NGramHistMerge> ngramrg;
  while (!far_reader->Done()) {
    fst::StdVectorFst in_fst;
    if (!GetSingleCountFst(far_reader, &in_fst, fstnumber, order,
                           epsilon_as_backoff)) {
      LOG(ERROR) << "failed to count fst number " << fstnumber;
      return false;
    }
    if (ngramrg == nullptr) {
      ArcMap(in_fst, fst, ToHistogramMapper<fst::StdArc>());
      ngramrg = std::make_unique<NGramHistMerge>(fst, backoff_label, norm_eps,
                                                 check_consistency);
    } else {
      fst::VectorFst<HistogramArc> hist_fst;
      ArcMap(in_fst, &hist_fst, ToHistogramMapper<fst::StdArc>());
      bool norm = normalize && far_reader->Done();
      ngramrg->MergeNGramModels(hist_fst, alpha, beta, norm);
    }
    far_reader->Next();
    ++fstnumber;
  }
  return true;
}

// Derives n-gram counts (and symbols) from input FAR reader.
bool GetNGramsAndSyms(fst::FarReader<fst::StdArc> *far_reader,
                      NGramCounter<fst::Log64Weight> *ngram_counter,
                      fst::SymbolTable *syms, bool require_symbols,
                      double add_to_symbol_unigram_count) {
  int fstnumber = 1;
  while (!far_reader->Done()) {
    if (!GetCounts("ngramcount", ngram_counter, far_reader, fstnumber, syms))
      return false;
    far_reader->Next();
    ++fstnumber;
  }
  if (require_symbols && syms->NumSymbols() == 0) {
    LOG(ERROR) << "None of the input FSTs had a symbol table";
    return false;
  }
  if (add_to_symbol_unigram_count > 0.0 && require_symbols) {
    ngram_counter->AddCountToSymbolUnigrams(
        *syms, /*neg_log_count=*/-log(add_to_symbol_unigram_count));
  }
  return true;
}

// Computes ngram counts and returns ngram format FST.
bool GetNGramCounts(fst::FarReader<fst::StdArc> *far_reader,
                    fst::StdMutableFst *fst, int order,
                    bool require_symbols, bool epsilon_as_backoff,
                    bool round_to_int, double add_to_symbol_unigram_count) {
  NGramCounter<fst::Log64Weight> ngram_counter(order, epsilon_as_backoff);
  fst::SymbolTable syms;
  if (!GetNGramsAndSyms(far_reader, &ngram_counter, &syms, require_symbols,
                        add_to_symbol_unigram_count)) {
    return false;
  }
  ngram_counter.GetFst(fst);
  fst::ArcSort(fst, fst::StdILabelCompare());
  if (syms.NumSymbols() > 0) {
    fst->SetInputSymbols(&syms);
    fst->SetOutputSymbols(&syms);
  }
  if (round_to_int) RoundCountsToInt(fst);
  return true;
}

// Computes ngram counts and returns vector of strings.
bool GetNGramCounts(fst::FarReader<fst::StdArc> *far_reader,
                    std::vector<std::string> *ngrams, int order,
                    bool epsilon_as_backoff,
                    double add_to_symbol_unigram_count) {
  NGramCounter<fst::Log64Weight> ngram_counter(order, epsilon_as_backoff);
  fst::SymbolTable syms;
  if (!GetNGramsAndSyms(far_reader, &ngram_counter, &syms,
                        /* require_symbols = */ true,
                        add_to_symbol_unigram_count)) {
    // Requires symbols from input far to output as vector of strings.
    return false;
  }
  std::vector<std::pair<std::vector<int>, std::pair<int, double>>> ngram_counts;
  ngram_counter.GetReverseContextNGrams<fst::StdArc>(&ngram_counts);
  for (size_t i = 0; i < ngram_counts.size(); ++i) {
    std::string ngram;
    double count = GetNGramAndCount(ngram_counts[i], &ngram, syms);
    ngrams->push_back(ngram + '\t' + std::to_string(count));
  }
  return true;
}

}  // namespace ngram
