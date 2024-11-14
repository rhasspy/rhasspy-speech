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
// NGram model class for reading in a model or text for building a model.

#ifndef NGRAM_NGRAM_INPUT_H_
#define NGRAM_NGRAM_INPUT_H_

#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <fst/arc.h>
#include <fst/float-weight.h>
#include <fst/fst.h>
#include <fst/matcher.h>
#include <fst/mutable-fst.h>
#include <fst/symbol-table.h>
#include <ngram/ngram-count.h>

namespace ngram {

// Reads vector of tokens from string.
void ReadTokenString(std::string_view str, std::vector<std::string> *words);

class NGramInput {
 public:
  using Arc = fst::StdArc;
  using Label = Arc::Label;
  using StateId = Arc::StateId;
  using Weight = Arc::Weight;

  using Counter = NGramCounter<fst::Log64Weight>;

  NGramInput(std::istream &istrm, std::ostream &ostrm,
             const std::string &symbols, std::string_view epsilon_symbol,
             std::string_view oov_symbol, std::string_view start_symbol,
             std::string_view end_symbol);

  // Reads text input of three types: n-gram counts, ARPA model or text corpus.
  // Outputs a model FST, a corpus FAR, or a symbol table.
  bool ReadInput(bool ARPA, bool symbols, bool output = true,
                 bool renormalize_arpa = false);

  const fst::MutableFst<Arc> *GetFst() const { return fst_.get(); }

  // Returns true if input setup is in a bad state.
  bool Error() const { return error_; }

 protected:
  void SetError() { error_ = true; }

 private:
  void InitializeSymbols(const std::string &symbols,
                         std::string_view epsilon_symbol);

  // Using whitespace as delimiter, reads token from string.
  static bool GetWhiteSpaceToken(std::string::const_iterator *it,
                                 const std::string &str, std::string *token);

  // Gets symbol label from table, adds it and ensure no duplicates if
  // requested.
  Label GetLabel(const std::string &word, bool add, bool dups);

  // GetLabel() if not <s> or </s>, otherwise sets appropriate bool values.
  Label GetNGramLabel(const std::string &ngram_word, bool add, bool dups,
                      bool *stsym, bool *endsym);

  // Uses string iterator to construct token, then gets the label for it.
  Label ExtractNGramLabel(std::string::const_iterator *it,
                          const std::string &str, bool add, bool dups,
                          bool *stsym, bool *endsym);

  // Gets backoff state and backoff cost for state (following <epsilon> arc).
  StateId GetBackoffAndCost(StateId st, double *cost);

  // Just returns backoff state.
  StateId GetBackoff(StateId st) { return GetBackoffAndCost(st, nullptr); }

  // Ensures matching with appropriate ARPA header strings.
  bool ARPAHeaderStringMatch(std::string_view tomatch);

  // Extracts string token and convert into value of type A.
  template <class A>
  bool GetStringVal(std::string::const_iterator *it, const std::string &str,
                    A *val, std::string *keeptoken) {
    std::string token;
    if (!GetWhiteSpaceToken(it, str, &token)) return false;
    std::stringstream ss(token);
    ss >> *val;
    if (keeptoken) *keeptoken = token;  // To store token string if needed.
    return true;
  }

  // When reading in string numerical tokens, ensures correct inf values.
  static void CheckInfVal(std::string_view token, double *val);

  // Reads the header at the top of the ARPA model file, collect n-gram orders
  int ReadARPATopHeader(std::vector<int> *orders);

  // Gets the destination state of an arc with the requested label.
  StateId GetLabelNextState(StateId st, Label label);

  // Gets the destination state of arc with requested label, assumed to exist.
  StateId GetLabelNextStateNoFail(StateId st, Label label);

  // GetLabelNextState() when arc exists; other results for <s> and </s>.
  ssize_t NextStateFromLabel(ssize_t st, Label label, bool stsym, bool endsym,
                             Counter *ngram_counter);

  // Extracts the token, finding the label and the appropriate destination
  // state.
  ssize_t GetNextState(std::string::const_iterator *it, const std::string &str,
                       ssize_t st, Counter *ngram_counter);

  // Reads the header for each of the n-gram orders in the ARPA format file.
  void ReadARPAOrderHeader(int order);

  // Adds an n-gram arc as appropriate, and records the state and label if
  // required.
  void AddNGramArc(StateId st, StateId nextstate, Label label, bool stsym,
                   bool endsym, double ngram_log_prob);

  // Reads in n-grams for the particular order.
  void ReadARPAOrder(
      std::vector<int> *orders, int order, std::vector<double> *boweights,
      NGramCounter<fst::LogWeightTpl<double>> *ngram_counter);

  StateId FindNewDest(StateId st);

  void SetARPANGramDests();

  // Puts stored backoff weights on backoff arcs.
  void SetARPABackoffWeights(const std::vector<double> &boweights);

  double GetLowerOrderProb(StateId st, Label label);

  // Descends backoff arcs to find backoff final cost and sets it.
  double GetFinalBackoff(StateId st);

  void FillARPAHoles();

  // Reads in headers and n-grams from an ARPA model text file and dumps
  // resulting FST.
  bool CompileARPAModel(bool output, bool renormalize);

  // Renormalizes the ARPA format model if required.
  void RenormalizeARPAModel();

  // Redirects high-order arcs in acyclic count format to proper next states.
  void MakeCyclicTopology(StateId st, StateId bo,
                          const std::vector<bool> &bo_incoming);

  // Collects state level information prior to changing topology.
  void SetStateBackoff(StateId st, StateId bo, std::vector<StateId> *bo_dest,
                       std::vector<double> *total_cnt,
                       std::vector<bool> *bo_incoming);

  // Creates re-entrant model topology from acyclic count automaton.
  void AddBackoffAndCycles(StateId unigram, Label bo_label);

  // Controls allocation of ARPA model start state.
  void CheckInitState(const std::vector<std::string> &words, StateId *init,
                      StateId unigram, StateId start);

  // Reads in N-gram tokens as well as count (last token) from string.
  double ReadNGramFromString(std::string str, std::vector<std::string> *words,
                             StateId *init, StateId unigram, StateId start);

  // Iterates through words in the n-gram history to find current state.
  StateId GetHistoryState(const std::vector<std::string> &words,
                          const std::vector<Label> &last_labels,
                          const std::vector<StateId> &last_states, StateId st);

  // Determines the next state for the final token of an n-gram.
  StateId GetCntNextSt(StateId st, StateId unigram, StateId init,
                       StateId *start, bool stsym, bool endsym);

  // Updates last label and state for retrieval with following n-grams.
  static int UpdateLast(const std::vector<std::string> &words, int longest_ngram,
                        std::vector<Label> *last_labels,
                        std::vector<StateId> *last_states, Label label,
                        StateId nextst);

  // Converts a sorted n-gram count file to an FST.
  bool CompileNGramCounts(bool output);

  // Tokenizes string and store labels in a vector for building an FST.
  double FillStringLabels(const std::string &str, std::vector<Label> *labels,
                          bool string_counts);

  // Converts text corpus to symbol table.
  bool CompileSymbolTable(bool output);

  // Writes resulting FST to output stream.
  void DumpFst(bool incl_symbols, bool output);

  std::unique_ptr<fst::MutableFst<Arc>> fst_;
  std::unique_ptr<fst::SymbolTable> syms_;
  bool add_symbols_;
  std::string oov_symbol_;
  std::string start_symbol_;
  std::string end_symbol_;
  std::istream &istrm_;
  std::ostream &ostrm_;
  bool error_;
};

}  // namespace ngram

#endif  // NGRAM_NGRAM_INPUT_H_
