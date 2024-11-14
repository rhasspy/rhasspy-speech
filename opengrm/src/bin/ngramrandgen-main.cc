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
// Generates random sentences from an LM or more generally paths through any
// FST where epsilons are treated as failure transitions.

#include <unistd.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/extensions/far/far.h>
#include <fst/arc.h>
#include <fst/fst.h>
#include <fst/randgen.h>
#include <fst/vector-fst.h>
#include <ngram/ngram-randgen.h>

DECLARE_int32(max_length);
DECLARE_int64(max_sents);
DECLARE_int32(seed);
DECLARE_bool(remove_epsilon);
DECLARE_bool(weighted);
DECLARE_bool(remove_total_weight);

namespace {

// create FST label for far, with sufficient padding for the number
int CalcExtLen(int value, std::string * key, int far_len) {
  std::stringstream ss;
  ss << value;
  std::string extension;
  ss >> extension;
  if (key) {
    for (int i = extension.length(); i < far_len; i++) *key += "0";
    *key += extension;
  }
  return extension.length();
}

// Write given fst into open far_writer
void FarWriteFst(fst::FarWriter<fst::StdArc> *far_writer,
                 fst::StdFst *fst, int *far_incr, int far_len) {
  std::string key = "FST";
  CalcExtLen(++(*far_incr), &key, far_len);
  far_writer->Add(key, *fst);
}

// from the given state to a final state, create linear fst for string
void CreateStringFstFromPath(std::vector<int> *labels,
                             fst::StdVectorFst *nfst) {
  fst::StdArc::StateId ost = 0, dst = 1;
  nfst->AddState();
  nfst->SetStart(ost);
  for (int i = 0; i < labels->size(); i++) {
    nfst->AddState();
    nfst->AddArc(
        ost++, fst::StdArc((*labels)[i], (*labels)[i],
                               fst::StdArc::Weight::One().Value(), dst++));
  }
  nfst->SetFinal(ost, fst::StdArc::Weight::One().Value());
}

// Given an fst (tree) of strings, create far of string fsts
int WritePathsToFar(fst::StdFst *fst, fst::StdArc::StateId st,
                    std::vector<int> *labels,
                    fst::FarWriter<fst::StdArc> *far_writer,
                    int far_cnt, int far_len, int remove_epsilons) {
  for (fst::ArcIterator<fst::StdFst> aiter(*fst, st); !aiter.Done();
       aiter.Next()) {
    fst::StdArc arc = aiter.Value();
    if (!remove_epsilons || arc.ilabel != 0) labels->push_back(arc.ilabel);
    far_cnt = WritePathsToFar(fst, arc.nextstate, labels, far_writer, far_cnt,
                              far_len, remove_epsilons);
    if (!remove_epsilons || arc.ilabel != 0) labels->pop_back();
  }
  if (fst->Final(st) != fst::StdArc::Weight::Zero()) {
    fst::StdVectorFst nfst;
    if (far_cnt == 1) {  // assigns symbol table to first fst
      nfst.SetInputSymbols(fst->InputSymbols());
      nfst.SetOutputSymbols(fst->OutputSymbols());
    }
    CreateStringFstFromPath(labels, &nfst);
    FarWriteFst(far_writer, &nfst, &far_cnt, far_len);
  }
  return far_cnt;
}

}  // namespace

int ngramrandgen_main(int argc, char **argv) {
  std::string usage = "Generates random sentences from an LM.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] [in.fst [out.far]]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  VLOG(1) << argv[0] << ": Seed = " << FST_FLAGS_seed;

  std::string ifile = (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  std::string ofile = (argc > 2 && (strcmp(argv[2], "-") != 0)) ? argv[2] : "";

  std::unique_ptr<fst::StdFst> ifst(fst::StdFst::Read(ifile));
  if (!ifst) return 1;

  std::unique_ptr<fst::FarWriter<fst::StdArc>> far_writer(
      fst::FarWriter<fst::StdArc>::Create(
          ofile, fst::FarType::STLIST));  // type change for fst
  if (!far_writer) {
    LOG(ERROR) << "Can't open " << (ofile.empty() ? "stdout" : ofile)
               << " for writing";
    return 1;
  }
  fst::StdVectorFst ofst;

  ngram::NGramArcSelector<fst::StdArc> selector(
      FST_FLAGS_seed ? FST_FLAGS_seed : time(nullptr));
  fst::RandGenOptions<ngram::NGramArcSelector<fst::StdArc>> opts(
      selector, FST_FLAGS_max_length, FST_FLAGS_max_sents,
      FST_FLAGS_weighted, FST_FLAGS_remove_total_weight);
  fst::RandGen(*ifst, &ofst, opts);
  ofst.SetInputSymbols(ifst->InputSymbols());  // takes model symbol tables
  ofst.SetOutputSymbols(ifst->OutputSymbols());
  if (FST_FLAGS_weighted) {
    int far_incr = 1;
    int far_len = 1;
    FarWriteFst(far_writer.get(), &ofst, &far_incr, far_len);
  } else {
    int far_cnt = 1;
    int far_len = CalcExtLen(FST_FLAGS_max_sents, nullptr, 0);
    std::vector<int> labels;
    WritePathsToFar(&ofst, ofst.Start(), &labels, far_writer.get(), far_cnt,
                    far_len, FST_FLAGS_remove_epsilon);
  }

  return 0;
}
