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
#include <ngram/ngram-input.h>

#include <cmath>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <fst/arc.h>
#include <fst/arcsort.h>
#include <fst/connect.h>
#include <fst/float-weight.h>
#include <fst/fst.h>
#include <fst/matcher.h>
#include <fst/mutable-fst.h>
#include <fst/symbol-table.h>
#include <fst/vector-fst.h>
#include <ngram/ngram-count.h>
#include <ngram/ngram-model.h>
#include <ngram/ngram-mutable-model.h>
#include <ngram/util.h>
#include <cctype>
#include <cstdlib>

namespace ngram {

using ::fst::ArcIterator;
using ::fst::kNoStateId;
using ::fst::Log64Weight;
using ::fst::MATCH_INPUT;
using ::fst::Matcher;
using ::fst::MutableArcIterator;
using ::fst::MutableFst;
using ::fst::StateIterator;
using ::fst::StdArc;
using ::fst::StdILabelCompare;
using ::fst::StdMutableFst;
using ::fst::StdVectorFst;
using ::fst::SymbolTable;

void ReadTokenString(std::string_view str, std::vector<std::string> *words) {
  auto it = str.cbegin();
  while (it < str.cend()) {
    if (words->empty() || std::isspace(*it)) {
      while (std::isspace(*it)) ++it;       // Skips whitespace sequence.
      if (it != str.end()) words->emplace_back();  // Starts new empty word.
    } else {
      (*words)[words->size() - 1] += *it;  // Adds character to words.
      ++it;
    }
  }
}

NGramInput::NGramInput(std::istream &istrm, std::ostream &ostrm,
                       const std::string &symbols,
                       std::string_view epsilon_symbol,
                       std::string_view oov_symbol,
                       std::string_view start_symbol,
                       std::string_view end_symbol)
    : oov_symbol_(oov_symbol),
      start_symbol_(start_symbol),
      end_symbol_(end_symbol),
      istrm_(istrm),
      ostrm_(ostrm),
      error_(false) {
  InitializeSymbols(symbols, epsilon_symbol);
}

// Reads text input of three types: n-gram counts, ARPA model or text corpus.
// Outputs a model FST, a corpus FAR, or a symbol table.
bool NGramInput::ReadInput(bool ARPA, bool symbols, bool output,
                           bool renormalize_arpa) {
  if (Error()) return false;
  if (ARPA) {  // ARPA format model in, FST model out.
    return CompileARPAModel(output, renormalize_arpa);
  } else if (symbols) {
    return CompileSymbolTable(output);
  } else {  // Sorted list of ngrams counts in, FST out.
    return CompileNGramCounts(output);
  }
  return false;
}

void NGramInput::InitializeSymbols(const std::string &symbols,
                                   std::string_view epsilon_symbol) {
  if (Error()) return;
  if (symbols.empty()) {                           // Symbol table not provided.
    syms_ = std::make_unique<SymbolTable>(
        "NGramSymbols");  // Initializes symbol table
    syms_->AddSymbol(epsilon_symbol);
    add_symbols_ = true;
  } else {
    syms_.reset(SymbolTable::ReadText(symbols));
    if (!syms_) {
      NGRAMERROR() << "NGramInput: Could not read symbol table file: "
                   << symbols;
      SetError();
      return;
    }
    add_symbols_ = false;
  }
}

// Using whitespace as delimiter, reads token from string.
bool NGramInput::GetWhiteSpaceToken(std::string::const_iterator *it,
                                    const std::string &str,
                                    std::string *token) {
  // Skips the whitespace preceding the token.
  while (std::isspace(**it)) ++*it;
  // No further tokens to be found in string.
  if (*it == str.cend()) return false;
  while (*it < str.cend() && !std::isspace(**it)) {
    *token += **it;
    ++*it;
  }
  return true;
}

// Gets symbol label from table, adds it and ensure no duplicates if requested.
typename NGramInput::Label NGramInput::GetLabel(const std::string &word,
                                                bool add, bool dups) {
  auto label = syms_->Find(word);
  if (!add_symbols_) {  // Fixed symbol table provided.
    if (label == fst::kNoLabel) {
      label = syms_->Find(oov_symbol_);
      if (label == fst::kNoLabel) {
        NGRAMERROR() << "NGramInput: OOV symbol not found "
                     << "in given symbol table: " << oov_symbol_;
        SetError();
      }
    }
  } else if (add) {
    if (label == fst::kNoLabel) {
      label = syms_->AddSymbol(word);
    } else if (!dups) {  // Shouldn't find duplicate.
      NGRAMERROR() << "NGramInput: Symbol already found in list: " << word;
      SetError();
    }
  } else if (label == fst::kNoLabel) {
    NGRAMERROR() << "NGramInput: Symbol not found in list: " << word;
    SetError();
  }
  return label;
}

// GetLabel() if not <s> or </s>, otherwise sets appropriate bool values.
typename NGramInput::Label NGramInput::GetNGramLabel(
    const std::string &ngram_word, bool add, bool dups, bool *stsym,
    bool *endsym) {
  *stsym = false;
  *endsym = false;
  if (ngram_word == start_symbol_) {
    *stsym = true;
    return -1;
  } else if (ngram_word == end_symbol_) {
    *endsym = true;
    return -2;
  } else {
    return GetLabel(ngram_word, add, dups);
  }
}

// Iterates through words in the n-gram history to find current state.
typename NGramInput::StateId NGramInput::GetHistoryState(
    const std::vector<std::string> &words, const std::vector<Label> &last_labels,
    const std::vector<StateId> &last_states, StateId st) {
  for (auto i = 0; i < words.size() - 2; ++i) {
    bool stsym;
    bool endsym;
    const auto label = GetNGramLabel(words[i], false, false, &stsym, &endsym);
    if (Error()) return 0;
    if (last_labels.size() <= i || label != last_labels[i]) {
      std::string prefix = words[0];
      for (auto j = 1; j <= i; ++j) prefix += " " + words[j];
      NGRAMERROR() << "NGramInput: n-gram prefix (" << prefix
                   << ") not seen in previous n-gram";
      SetError();
      return 0;
    }
    st = last_states[i];  // Retrieves previously stored state.
  }
  return st;
}

// Uses string iterator to construct token, then gets the label for it.
typename NGramInput::Label NGramInput::ExtractNGramLabel(
    std::string::const_iterator *it, const std::string &str, bool add,
    bool dups, bool *stsym, bool *endsym) {
  std::string token;
  if (!GetWhiteSpaceToken(it, str, &token)) {
    NGRAMERROR() << "NGramInput: No token found when expected";
    SetError();
    return -1;
  }
  return GetNGramLabel(token, add, dups, stsym, endsym);
}

// Gets backoff state and backoff cost for state (following <epsilon> arc).
typename NGramInput::StateId NGramInput::GetBackoffAndCost(StateId st,
                                                           double *cost) {
  StateId backoff = -1;
  Label backoff_label = 0;  // <epsilon> is assumed to be label 0 here.
  Matcher<MutableFst<Arc>> matcher(*fst_, MATCH_INPUT);
  matcher.SetState(st);
  if (matcher.Find(backoff_label)) {
    for (; !matcher.Done(); matcher.Next()) {
      const auto &arc = matcher.Value();
      if (arc.ilabel == backoff_label) {
        backoff = arc.nextstate;
        if (cost) *cost = arc.weight.Value();
      }
    }
  }
  return backoff;
}

// Ensures matching with appropriate ARPA header strings.
bool NGramInput::ARPAHeaderStringMatch(std::string_view tomatch) {
  std::string str;
  if (!std::getline(istrm_, str)) {
    NGRAMERROR() << "Input stream read error";
    SetError();
    return false;
  }
  if (str != tomatch) {
    NGRAMERROR() << "NGramInput: ARPA header mismatch!  Line reads: " << str
                 << "  Line should read: " << tomatch;
    SetError();
    return false;
  }
  return true;
}

// When reading in string numerical tokens, ensures correct inf values.
void NGramInput::CheckInfVal(std::string_view token, double *val) {
  if (token == "-inf" || token == "-Infinity") {
    *val = -std::numeric_limits<double>::infinity();
  } else if (token == "inf" || token == "Infinity") {
    *val = std::numeric_limits<double>::infinity();
  }
}

// Reads the header at the top of the ARPA model file, collecting n-gram orders.
int NGramInput::ReadARPATopHeader(std::vector<int> *orders) {
  std::string str;
  // Scans the file until a \data\ record is found.
  while (std::getline(istrm_, str)) {
    if (str == "\\data\\") break;
  }
  if (!std::getline(istrm_, str)) {
    NGRAMERROR() << "Input stream read error, or no \\data\\ record found";
    SetError();
    return 0;
  }
  int order = 0;
  while (!str.empty()) {
    auto it = str.cbegin();
    while (it < str.cend() && *it != '=') it++;
    if (it == str.cend()) {
      NGRAMERROR()
          << "NGramInput: ARPA header mismatch!  No '=' in ngram count.";
      SetError();
      return 0;
    }
    ++it;
    int ngram_cnt;  // Must have n-gram count, fails if not found.
    if (!GetStringVal(&it, str, &ngram_cnt, nullptr)) {
      NGRAMERROR() << "NGramInput: ARPA header mismatch!  No ngram count.";
      SetError();
      return 0;
    }
    orders->push_back(ngram_cnt);
    if (ngram_cnt > 0) ++order;  // Some reported n-gram orders may be empty.
    if (!std::getline(istrm_, str)) {
      NGRAMERROR() << "Input stream read error";
      SetError();
      return 0;
    }
  }
  return order;
}

// Gets the destination state of an arc with the requested label.
typename NGramInput::StateId NGramInput::GetLabelNextState(StateId st,
                                                           Label label) {
  Matcher<MutableFst<Arc>> matcher(*fst_, MATCH_INPUT);
  matcher.SetState(st);
  if (matcher.Find(label)) {
    return matcher.Value().nextstate;
  } else {
    NGRAMERROR() << "NGramInput: Lower order prefix n-gram not found: ";
    SetError();
    return kNoStateId;
  }
}

// Gets the destination state of arc with requested label, assumed to exist.
typename NGramInput::StateId NGramInput::GetLabelNextStateNoFail(StateId st,
                                                                 Label label) {
  Matcher<MutableFst<Arc>> matcher(*fst_, MATCH_INPUT);
  matcher.SetState(st);
  if (matcher.Find(label)) {
    return matcher.Value().nextstate;
  } else {
    return kNoStateId;
  }
}

// Redirects high-order arcs in acyclic count format to proper next states.
void NGramInput::MakeCyclicTopology(StateId st, StateId bo,
                                    const std::vector<bool> &bo_incoming) {
  for (MutableArcIterator<StdMutableFst> aiter(fst_.get(), st); !aiter.Done();
       aiter.Next()) {
    auto arc = aiter.Value();
    auto nst = GetLabelNextState(bo, arc.ilabel);
    if (!bo_incoming[arc.nextstate] &&
        fst_->Final(arc.nextstate) == StdArc::Weight::Zero() &&
        fst_->NumArcs(arc.nextstate) == 0) {  // if nextstate not in model
      arc.nextstate = nst;  // point to state that will persist in the model
      aiter.SetValue(arc);
    } else {
      MakeCyclicTopology(arc.nextstate, nst, bo_incoming);
    }
  }
}

// GetLabelNextState() when arc exists; other results for <s> and </s>.
ssize_t NGramInput::NextStateFromLabel(
    ssize_t st, Label label, bool stsym, bool endsym,
    NGramCounter<Log64Weight> *ngram_counter) {
  if (Error()) return 0;
  if (stsym) {  // start symbol: <s>.
    return ngram_counter->NGramStartState();
  } else if (endsym) {  // end symbol: </s>.
    NGRAMERROR() << "NGramInput: stop symbol occurred in n-gram prefix";
    SetError();
    return 0;
  } else {
    ssize_t arc_id = ngram_counter->FindArc(st, label);
    return ngram_counter->NGramNextState(arc_id);
  }
}

// Extracts the token, finding the label and the appropriate destination state.
ssize_t NGramInput::GetNextState(std::string::const_iterator *it,
                                 const std::string &str, ssize_t st,
                                 NGramCounter<Log64Weight> *ngram_counter) {
  if (Error()) return 0;
  bool stsym = false;
  bool endsym = false;
  auto label = ExtractNGramLabel(it, str, /*add=*/false,
                                 /*dups=*/false, &stsym, &endsym);
  return NextStateFromLabel(st, label, stsym, endsym, ngram_counter);
}

// Reads the header for each of the n-gram orders in the ARPA format file
void NGramInput::ReadARPAOrderHeader(int order) {
  std::stringstream ss;
  ss << order + 1;
  const std::string tomatch = "\\" + ss.str() + "-grams:";
  ARPAHeaderStringMatch(tomatch);
}

// Adds an n-gram arc as appropriate, and records the state and label if
// required.
void NGramInput::AddNGramArc(StateId st, StateId nextstate, Label label,
                             bool stsym, bool endsym, double ngram_log_prob) {
  if (endsym) {  // </s> requires no arc, just final cost.
    fst_->SetFinal(st, ngram_log_prob);
  } else if (!stsym) {  // Creates arc from st to nextstate.
    fst_->AddArc(st, StdArc(label, label, ngram_log_prob, nextstate));
  }
}

// Reads in n-grams for the particular order.
void NGramInput::ReadARPAOrder(std::vector<int> *orders, int order,
                               std::vector<double> *boweights,
                               NGramCounter<Log64Weight> *ngram_counter) {
  std::string str;
  bool add_words = order == 0;
  for (auto i = 0; i < (*orders)[order]; ++i) {
    if (!std::getline(istrm_, str)) {
      NGRAMERROR() << "Input stream read error";
      SetError();
      return;
    }
    auto it = str.cbegin();
    double nlprob;
    double boprob;
    std::string token;
    if (!GetStringVal(&it, str, &nlprob, &token)) {
      NGRAMERROR() << "NGramInput: ARPA format mismatch!  No ngram log prob.";
      SetError();
      return;
    }
    NGramInput::CheckInfVal(token, &boprob);
    nlprob *= -log(10);  // Converts to neglog base e from log base 10.
    ssize_t st = ngram_counter->NGramUnigramState();
    StateId nextstate = fst::kNoStateId;
    for (int j = 0; j < order; ++j)  // Finds n-gram history state.
      st = GetNextState(&it, str, st, ngram_counter);
    if (Error()) return;
    bool stsym;   // stsym == 1 for <s>.
    bool endsym;  // endsym == 1 for </s>.
    Label label = ExtractNGramLabel(&it, str, add_words,
                                    /*dups=*/false, &stsym, &endsym);
    if (Error()) return;
    if (endsym) {
      ngram_counter->SetFinalNGramWeight(st, nlprob);
    } else if (!stsym) {
      // Tests for presence of all suffixes of n-gram.
      auto backoff_st = ngram_counter->NGramBackoffState(st);
      while (backoff_st >= 0) {
        ngram_counter->FindArc(backoff_st, label);
        backoff_st = ngram_counter->NGramBackoffState(backoff_st);
      }
      const auto arc_id = ngram_counter->FindArc(st, label);
      ngram_counter->SetNGramWeight(arc_id, nlprob);
      nextstate = ngram_counter->NGramNextState(arc_id);
    } else {
      nextstate = ngram_counter->NGramStartState();
    }
    if (GetStringVal(&it, str, &boprob, &token) &&
        (nextstate >= 0 || boprob != 0)) {  // Found non-zero backoff cost.
      if (nextstate == fst::kNoStateId) {
        NGRAMERROR() << "NGramInput: Have a backoff cost with no state ID!";
        SetError();
        return;
      }
      NGramInput::CheckInfVal(token, &boprob);
      boprob *= -log(10);  // Converts to neglog base e from log base 10.
      while (nextstate >= boweights->size())
        boweights->push_back(StdArc::Weight::Zero().Value());
      (*boweights)[nextstate] = boprob;
    }
  }
  // Blank line at end of n-gram order.
  if (!std::getline(istrm_, str)) {
    NGRAMERROR() << "Input stream read error";
    SetError();
    return;
  }
  if (!str.empty()) {
    NGRAMERROR() << "Expected blank line at end of n-grams";
    SetError();
  }
}

typename NGramInput::StateId NGramInput::FindNewDest(StateId st) {
  StateId newdest = st;
  if (fst_->NumArcs(st) > 1 || fst_->Final(st) != StdArc::Weight::Zero()) {
    return newdest;
  }
  MutableArcIterator<StdMutableFst> aiter(fst_.get(), st);
  const auto &arc = aiter.Value();
  if (arc.ilabel == 0) newdest = FindNewDest(arc.nextstate);
  return newdest;
}

void NGramInput::SetARPANGramDests() {
  std::vector<StateId> newdests;
  StateIterator<MutableFst<Arc>> siter(*fst_);
  for (; !siter.Done(); siter.Next()) {
    newdests.push_back(FindNewDest(siter.Value()));
  }
  siter.Reset();
  for (; !siter.Done(); siter.Next()) {
    for (MutableArcIterator<StdMutableFst> aiter(fst_.get(), siter.Value());
         !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      if (arc.ilabel == 0) continue;
      if (newdests[arc.nextstate] != arc.nextstate) {
        arc.nextstate = newdests[arc.nextstate];
        aiter.SetValue(arc);
      }
    }
  }
}

// Puts stored backoff weights on backoff arcs.
void NGramInput::SetARPABackoffWeights(const std::vector<double> &boweights) {
  for (StateIterator<MutableFst<Arc>> siter(*fst_); !siter.Done();
       siter.Next()) {
    const auto st = siter.Value();
    if (st < boweights.size()) {
      const auto boprob = boweights[st];
      MutableArcIterator<StdMutableFst> aiter(fst_.get(), st);
      auto arc = aiter.Value();
      if (arc.ilabel == 0 || boprob != StdArc::Weight::Zero().Value()) {
        if (arc.ilabel != 0) {
          NGRAMERROR() << "NGramInput: Have a backoff prob but no arc";
          SetError();
          return;
        } else {
          arc.weight = boprob;
        }
        aiter.SetValue(arc);
      }
    }
  }
}

double NGramInput::GetLowerOrderProb(StateId st, Label label) {
  Matcher<MutableFst<Arc>> matcher(*fst_, MATCH_INPUT);
  matcher.SetState(st);
  if (matcher.Find(label)) {
    return matcher.Value().weight.Value();
  }
  if (!matcher.Find(0)) {
    NGRAMERROR() << "NGramInput: No backoff probability";
    SetError();
    return StdArc::Weight::Zero().Value();
  }
  for (; !matcher.Done(); matcher.Next()) {
    const auto &arc = matcher.Value();
    if (arc.ilabel == 0) {
      return arc.weight.Value() + GetLowerOrderProb(arc.nextstate, label);
    }
  }
  NGRAMERROR() << "NGramInput: No backoff arc found";
  SetError();
  return StdArc::Weight::Zero().Value();
}

// Descends backoff arcs to find backoff final cost and sets it.
double NGramInput::GetFinalBackoff(StateId st) {
  if (fst_->Final(st) != StdArc::Weight::Zero()) return fst_->Final(st).Value();
  double bocost;
  auto bostate = GetBackoffAndCost(st, &bocost);
  if (bostate >= 0) fst_->SetFinal(st, bocost + GetFinalBackoff(bostate));
  return fst_->Final(st).Value();
}

void NGramInput::FillARPAHoles() {
  for (StateIterator<MutableFst<Arc>> siter(*fst_); !siter.Done();
       siter.Next()) {
    const auto st = siter.Value();
    double boprob;
    StateId bostate = kNoStateId;
    for (MutableArcIterator<StdMutableFst> aiter(fst_.get(), st); !aiter.Done();
         aiter.Next()) {
      auto arc = aiter.Value();
      if (arc.ilabel == 0) {
        boprob = arc.weight.Value();
        bostate = arc.nextstate;
      } else {
        if (arc.weight == StdArc::Weight::Zero()) {
          arc.weight = boprob + GetLowerOrderProb(bostate, arc.ilabel);
          if (Error()) return;
          aiter.SetValue(arc);
        }
      }
    }
    if (bostate >= 0 && fst_->Final(st) != StdArc::Weight::Zero() &&
        fst_->Final(bostate) == StdArc::Weight::Zero()) {
      GetFinalBackoff(bostate);
    }
  }
}

// Reads in headers and n-grams from an ARPA model text file and dumps resulting
// FST.
bool NGramInput::CompileARPAModel(bool output, bool renormalize) {
  std::vector<int> orders;
  ReadARPATopHeader(&orders);
  if (Error()) return false;
  std::vector<double> boweights;
  NGramCounter<Log64Weight> ngram_counter(orders.size());
  for (auto i = 0; i < orders.size(); i++) {  // Read n-grams of each order
    ReadARPAOrderHeader(i);
    if (Error()) return false;
    ReadARPAOrder(&orders, i, &boweights, &ngram_counter);
    if (Error()) return false;
  }
  ARPAHeaderStringMatch("\\end\\");  // Verify that everything parsed well
  if (Error()) return false;
  fst_ = std::make_unique<StdVectorFst>();
  ngram_counter.GetFst(fst_.get());
  static const StdILabelCompare icomp;
  ArcSort(fst_.get(), icomp);
  SetARPABackoffWeights(boweights);
  if (Error()) return false;
  FillARPAHoles();
  if (Error()) return false;
  SetARPANGramDests();
  Connect(fst_.get());
  if (renormalize) RenormalizeARPAModel();
  if (Error()) return false;
  DumpFst(true, output);
  return true;
}

// Renormalizes the ARPA format model if required.
void NGramInput::RenormalizeARPAModel() {
  NGramMutableModel<Arc> ngram_model(fst_.get());
  if (ngram_model.CheckNormalization() || ngram_model.Error()) return;
  auto st = ngram_model.UnigramState();
  if (st == fst::kNoStateId) st = fst_->Start();
  double renorm_val = ngram_model.ScalarValue(fst_->Final(st));
  double KahanVal = 0.0;
  for (ArcIterator<MutableFst<Arc>> aiter(*fst_, st); !aiter.Done();
       aiter.Next()) {
    Arc arc = aiter.Value();
    renorm_val =
        NegLogSum(renorm_val, ngram_model.ScalarValue(arc.weight), &KahanVal);
  }
  if (fst_->Final(st) != Arc::Weight::Zero()) {
    fst_->SetFinal(st, ngram_model.ScaleWeight(fst_->Final(st), -renorm_val));
  }
  for (MutableArcIterator<MutableFst<Arc>> aiter(fst_.get(), st); !aiter.Done();
       aiter.Next()) {
    Arc arc = aiter.Value();
    arc.weight = ngram_model.ScaleWeight(arc.weight, -renorm_val);
    aiter.SetValue(arc);
  }
  ngram_model.RecalcBackoff();
  if (!ngram_model.CheckNormalization()) {
    NGRAMERROR() << "ARPA model could not be renormalized";
    SetError();
  }
}

// Collects state level information prior to changing topology.
void NGramInput::SetStateBackoff(StateId st, StateId bo,
                                 std::vector<StateId> *bo_dest,
                                 std::vector<double> *total_cnt,
                                 std::vector<bool> *bo_incoming) {
  (*bo_dest)[st] = bo;  // Records the backoff state to be added later.

  // Records that state is backed off to by another state in the model.
  (*bo_incoming)[bo] = true;
  (*total_cnt)[st] = fst_->Final(st).Value();
  double correction_value = 0.0;
  for (MutableArcIterator<StdMutableFst> aiter(fst_.get(), st); !aiter.Done();
       aiter.Next()) {
    auto arc = aiter.Value();
    (*total_cnt)[st] =
        NegLogSum((*total_cnt)[st], arc.weight.Value(), &correction_value);
    auto nst = GetLabelNextState(bo, arc.ilabel);
    SetStateBackoff(arc.nextstate, nst, bo_dest, total_cnt, bo_incoming);
  }
}

// Creates re-entrant model topology from acyclic count automaton.
void NGramInput::AddBackoffAndCycles(StateId unigram, Label bo_label) {
  static const StdILabelCompare icomp;
  ArcSort(fst_.get(), icomp);  // Ensures arcs fully sorted.
  std::vector<StateId> bo_dest(fst_->NumStates(), kNoStateId);
  std::vector<bool> bo_incoming(fst_->NumStates(), false);
  std::vector<double> total_cnt(fst_->NumStates(),
                                StdArc::Weight::Zero().Value());

  // Stores all bigram states in a vector for ascending state functions.
  std::vector<StateId> bigram_states;
  if (fst_->Start() != unigram) bigram_states.push_back(fst_->Start());
  for (ArcIterator<StdMutableFst> aiter(*fst_, unigram); !aiter.Done();
       aiter.Next()) {
    bigram_states.push_back(aiter.Value().nextstate);
  }

  // Ascends to all states and collects state information.
  for (auto i = 0; i < bigram_states.size(); ++i) {
    SetStateBackoff(bigram_states[i], unigram, &bo_dest, &total_cnt,
                    &bo_incoming);
  }
  // Ascends to all states from unigram and makes topology cyclic.
  for (auto i = 0; i < bigram_states.size(); ++i) {
    MakeCyclicTopology(bigram_states[i], unigram, bo_incoming);
  }
  // Adds backoff arcs for each state in the topology.
  for (StateIterator<MutableFst<Arc>> siter(*fst_); !siter.Done();
       siter.Next()) {
    const auto st = siter.Value();
    if (bo_dest[st] >= 0)  // If backoff state has been recorded.
      fst_->AddArc(st, StdArc(bo_label, bo_label, total_cnt[st], bo_dest[st]));
  }
  ArcSort(fst_.get(), icomp);  // Resorts for new backoff arcs.
  Connect(fst_.get());  // Connects to dispose of states not in the model.
}

// Controls allocation of ARPA model start state.
void NGramInput::CheckInitState(const std::vector<std::string> &words,
                                StateId *init, StateId unigram, StateId start) {
  if (words.size() > 2 && *init == unigram) {  // 1st evidence order > 1.
    if (start >= 0) {
      // If the start unigram is already seen, we use this state as initial.
      *init = start;
    } else {
      // Otherwise, we need to create a start state.
      *init = fst_->AddState();
    }
    fst_->SetStart(*init);  // Sets it as start state.
  }
}

// Reads in n-gram tokens and counts from string.
double NGramInput::ReadNGramFromString(std::string str,
                                       std::vector<std::string> *words,
                                       StateId *init, StateId unigram,
                                       StateId start) {
  ReadTokenString(str, words);
  if (words->empty()) {
    NGRAMERROR() << "NGramInput: empty line in file: format error";
    SetError();
    return 0.0;
  }
  std::stringstream cnt_ss((*words)[words->size() - 1]);
  double ngram_count;
  cnt_ss >> ngram_count;
  CheckInitState(*words, init, unigram, start);  // Checks start state status.
  return -log(ngram_count);  // Counts are encoded in -log-space.
}

// Determines the next state for the final token of an n-gram.
typename NGramInput::StateId NGramInput::GetCntNextSt(StateId st,
                                                      StateId unigram,
                                                      StateId init,
                                                      StateId *start,
                                                      bool stsym, bool endsym) {
  StateId nextstate = -1;
  if (stsym) {            // Start symbol: <s>.
    if (st != unigram) {  // Should not occur.
      NGRAMERROR() << "NGramInput: start symbol occurred in n-gram suffix";
      SetError();
      return nextstate;
    }
    if (init == unigram)          // Don't know if model is order > 1 yet.
      *start = fst_->AddState();  // Creates state associated with <s>.
    else  // Already created 2nd order start state, stored as init.
      *start = init;
    nextstate = *start;
  } else if (!endsym) {  // Not a </s> symbol, hence need to create a state.
    nextstate = fst_->AddState();
  }
  return nextstate;
}

// Updates last label and state, for retrieval by following n-grams.
int NGramInput::UpdateLast(const std::vector<std::string> &words,
                           int longest_ngram, std::vector<Label> *last_labels,
                           std::vector<StateId> *last_states, Label label,
                           StateId nextst) {
  if (words.size() > longest_ngram + 1) {  // Adds a dimension to vectors.
    ++longest_ngram;
    last_labels->push_back(-1);
    last_states->push_back(-1);
  }
  (*last_labels)[words.size() - 2] = label;
  (*last_states)[words.size() - 2] = nextst;
  return longest_ngram;
}

// Converts a sorted n-gram count file to an FST.
bool NGramInput::CompileNGramCounts(bool output) {
  fst_ = std::make_unique<StdVectorFst>();  // Creates new FST.
  auto init = fst_->AddState();
  auto unigram = init;
  auto start = fst::kNoStateId;
  int longram = 0;  // Keeps track of longest observed n-gram in file.
  std::string str;
  std::vector<Label> last_labels;      // Stores labels from prior n-grams.
  std::vector<StateId> last_states;    // Stores states from prior n-grams.
  while (std::getline(istrm_, str)) {  // For each string...
    std::vector<std::string> words;
    // Reads in word tokens from string, and returns count.
    double ngram_count =
        ReadNGramFromString(str, &words, &init, unigram, start);
    if (Error()) return false;
    // Finds n-gram history state from prefix words in n-gram.
    auto st = GetHistoryState(words, last_labels, last_states, unigram);
    if (Error()) return false;
    bool stsym;
    bool endsym;
    // Gets label of word suffix of n-gram.
    auto label =
        GetNGramLabel(words[words.size() - 2], true, true, &stsym, &endsym);
    if (Error()) return false;
    // Gets the next state from history state and label.
    auto nextst = GetCntNextSt(st, unigram, init, &start, stsym, endsym);
    if (Error()) return false;
    // Adds arc.
    AddNGramArc(st, nextst, label, stsym, endsym, ngram_count);
    // Updates states and labels for subsequent n-grams.
    longram =
        UpdateLast(words, longram, &last_labels, &last_states, label, nextst);
  }
  // Sets init as start state for unigram model.
  if (init == unigram) fst_->SetStart(init);
  AddBackoffAndCycles(unigram, 0);  // Turns into reentrant OpenGrm format.
  DumpFst(true, output);
  return true;
}

// Tokenizes string and store labels in a vector for building an FST.
double NGramInput::FillStringLabels(const std::string &str,
                                    std::vector<Label> *labels,
                                    bool string_counts) {
  std::string token;
  auto it = str.cbegin();
  double count = 1.0;
  if (string_counts) {
    GetWhiteSpaceToken(&it, str, &token);
    count = atof(token.c_str());
    token.clear();
  }
  while (NGramInput::GetWhiteSpaceToken(&it, str, &token)) {
    labels->push_back(GetLabel(token, true, true));  // Stores index.
    token.clear();
  }
  return count;
}

// Converts text corpus to symbol table.
bool NGramInput::CompileSymbolTable(bool output) {
  for (std::string str; std::getline(istrm_, str);) {  // For each string.
    std::vector<Label> labels;
    FillStringLabels(str, &labels, false);
    if (Error()) return false;
  }
  if (!oov_symbol_.empty()) syms_->AddSymbol(oov_symbol_);
  if (output) syms_->WriteText(ostrm_);
  return true;
}

// Writes resulting FST to output stream.
void NGramInput::DumpFst(bool incl_symbols, bool output) {
  if (incl_symbols) {
    fst_->SetInputSymbols(syms_.get());
    fst_->SetOutputSymbols(syms_.get());
  }
  if (output) fst_->Write(ostrm_, fst::FstWriteOptions());
}

}  // namespace ngram
