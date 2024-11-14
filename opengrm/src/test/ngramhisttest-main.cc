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
// Compiles and tests equality of HistogramArc models during unit tests.
// Avoid complex shared object issues for equivalent FST library functions.

#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <string>

#include <fst/log.h>
#include <fst/symbol-table.h>
#include <fst/script/compile.h>
#include <fst/script/equal.h>
#include <fst/script/fst-class.h>

DECLARE_double(delta);
DECLARE_string(ifile);
DECLARE_string(syms);
DECLARE_string(cfile);
DECLARE_string(ofile);

int ngramhisttest_main(int argc, char **argv) {
  using fst::script::FstClass;
  std::string usage =
      "Compiles and tests equality of HistogramArc models.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (FST_FLAGS_ifile.empty() ||
      (FST_FLAGS_syms.empty() &&
       FST_FLAGS_cfile.empty())) {
    LOG(ERROR)
        << "The --ifile option and one of --syms and --cfile must be non-empty";
    return 1;
  }
  if (!FST_FLAGS_syms.empty() &&
      !FST_FLAGS_cfile.empty()) {
    LOG(ERROR) << "Both --syms and --cfile cannot be provided.  Give --syms to "
                  "compile the --ifile; give --cfile to compare with --ifile.";
    return 1;
  }
  if (FST_FLAGS_syms.empty()) {
    std::unique_ptr<FstClass> ifst1(FstClass::Read(FST_FLAGS_ifile));
    if (!ifst1) return 1;

    std::unique_ptr<FstClass> ifst2(FstClass::Read(FST_FLAGS_cfile));
    if (!ifst2) return 1;

    bool result =
        fst::script::Equal(*ifst1, *ifst2, FST_FLAGS_delta);
    if (!result) VLOG(1) << "FSTs are not equal.";

    return result ? 0 : 2;
  } else {
    std::unique_ptr<const fst::SymbolTable> syms(
        fst::SymbolTable::ReadText(FST_FLAGS_syms));
    if (!syms) return 1;
    std::unique_ptr<const fst::SymbolTable> ssyms;
    std::ifstream fstrm;
    fstrm.open(FST_FLAGS_ifile);
    if (!fstrm) {
      LOG(ERROR) << argv[0]
                 << ": Open failed, file = " << FST_FLAGS_ifile;
      return 1;
    }
    std::istream &istrm = fstrm.is_open() ? fstrm : std::cin;
    fst::script::Compile(
        istrm, FST_FLAGS_ifile, FST_FLAGS_ofile, "vector",
        "hist", syms.get(), syms.get(), ssyms.get(), false, true, true, true);
  }
  return 0;
}
