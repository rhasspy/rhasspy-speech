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
// Prints a given n-gram model to various kinds of textual formats.

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include <fst/flags.h>
#include <fst/mutable-fst.h>
#include <fst/symbol-table.h>
#include <ngram/ngram-output.h>

DECLARE_bool(ARPA);
DECLARE_bool(backoff);
DECLARE_bool(backoff_inline);
DECLARE_bool(negativelogs);
DECLARE_bool(integers);
DECLARE_int64(backoff_label);
DECLARE_bool(check_consistency);
DECLARE_string(context_pattern);
DECLARE_bool(include_all_suffixes);
DECLARE_string(symbols);

int ngramprint_main(int argc, char **argv) {
  std::string usage = "Print n-gram counts and models.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] [in.fst [out.txt]]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  std::string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  std::string out_name =
      (argc > 2 && (strcmp(argv[2], "-") != 0)) ? argv[2] : "stdout";

  std::unique_ptr<fst::StdMutableFst> fst(
      fst::StdMutableFst::Read(in_name, true));
  if (!fst) return 1;

  if (!FST_FLAGS_symbols.empty()) {
    std::unique_ptr<fst::SymbolTable> syms(
        fst::SymbolTable::ReadText(FST_FLAGS_symbols));
    if (!syms) return 1;
    fst->SetInputSymbols(syms.get());
    fst->SetOutputSymbols(syms.get());
  }

  std::ofstream ofstrm;
  if (argc > 2 && (strcmp(argv[2], "-") != 0)) {
    ofstrm.open(argv[2]);
    if (!ofstrm) {
      LOG(ERROR) << argv[0] << ": Open failed, file = " << argv[2];
      return 1;
    }
  }
  std::ostream &ostrm = ofstrm.is_open() ? ofstrm : std::cout;

  if (FST_FLAGS_ARPA &&
      (FST_FLAGS_negativelogs || FST_FLAGS_integers)) {
    LOG(ERROR) << "Printing in ARPA format writes weights as log_{10} "
               << "probabilities. --ARPA cannot be used with "
               << "the --negativelogs and --integers flags.";
    return 1;
  }

  ngram::NGramOutput ngram(fst.get(), ostrm, FST_FLAGS_backoff_label,
                           FST_FLAGS_check_consistency,
                           FST_FLAGS_context_pattern,
                           FST_FLAGS_include_all_suffixes);

  // Parse --backoff and --backoff_inline flags, where --backoff takes precedent
  ngram::NGramOutput::ShowBackoff show_backoff =
      ngram::NGramOutput::ShowBackoff::NONE;
  if (FST_FLAGS_backoff) {
    show_backoff = ngram::NGramOutput::ShowBackoff::EPSILON;
    if (FST_FLAGS_backoff_inline)
      show_backoff = ngram::NGramOutput::ShowBackoff::INLINE;
  }

  ngram.ShowNGramModel(show_backoff, FST_FLAGS_negativelogs,
                       FST_FLAGS_integers,
                       FST_FLAGS_ARPA);
  return 0;
}
