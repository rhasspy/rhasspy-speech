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
// Transfers n-grams from a source model(s) to a destination model.

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/arc.h>
#include <fst/vector-fst.h>
#include <ngram/hist-arc.h>
#include <ngram/ngram-complete.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-transfer.h>
#include <ngram/util.h>

DECLARE_int64(backoff_label);
DECLARE_string(context_pattern1);
DECLARE_string(context_pattern2);
DECLARE_string(contexts);
DECLARE_string(ofile);
DECLARE_string(method);
DECLARE_int32(index);
DECLARE_bool(transfer_from);
DECLARE_bool(normalize);
DECLARE_bool(complete);

namespace {

template <class Arc>
bool ReadFst(const char *file, std::unique_ptr<fst::VectorFst<Arc>> *fst) {
  std::string in_name = (strcmp(file, "-") != 0) ? file : "";
  fst->reset(fst::VectorFst<Arc>::Read(file));
  if (!*fst ||
      (FST_FLAGS_complete && !ngram::NGramComplete(fst->get())))
    return false;
  return true;
}

bool GetContexts(int in_count, std::vector<std::string> *contexts) {
  contexts->clear();
  if (!FST_FLAGS_contexts.empty()) {
    ngram::NGramReadContexts(FST_FLAGS_contexts, contexts);
  } else if (!FST_FLAGS_context_pattern1.empty() &&
             !FST_FLAGS_context_pattern2.empty()) {
    contexts->push_back(FST_FLAGS_context_pattern1);
    contexts->push_back(FST_FLAGS_context_pattern2);
  } else {
    LOG(ERROR) << "Context patterns not specified";
    return false;
  }
  if (contexts->size() != in_count) {
    LOG(ERROR) << "Wrong number of context patterns specified";
    return false;
  }
  return true;
}

template <class Arc>
bool Transfer(std::string out_name_prefix, int in_count, char **argv) {
  std::unique_ptr<fst::VectorFst<Arc>> index_fst;
  if (!ReadFst<Arc>(argv[FST_FLAGS_index + 1], &index_fst))
    return false;

  std::vector<std::string> contexts;
  if (!GetContexts(in_count, &contexts)) return false;

  if (FST_FLAGS_transfer_from) {
    ngram::NGramTransfer<Arc> transfer(index_fst.get(),
                                       contexts[FST_FLAGS_index],
                                       FST_FLAGS_backoff_label);
    for (int src = 0; src < in_count; ++src) {
      if (src == FST_FLAGS_index) continue;

      std::unique_ptr<fst::VectorFst<Arc>> fst_src;
      if (!ReadFst<Arc>(argv[src + 1], &fst_src) ||
          !transfer.TransferNGramsFrom(*fst_src, contexts[src])) {
        return false;
      }
    }

    // Normalization occurs after all transfer has occurred.
    if (FST_FLAGS_normalize && !transfer.TransferNormalize()) {
      NGRAMERROR() << "Unable to normalize after transfer";
      return false;
    }
    index_fst->Write(out_name_prefix);  // no suffix in this case
  } else {
    ngram::NGramTransfer<Arc> transfer(*index_fst,
                                       contexts[FST_FLAGS_index],
                                       FST_FLAGS_backoff_label);
    for (int dest = 0; dest < in_count; ++dest) {
      if (dest == FST_FLAGS_index) continue;

      std::unique_ptr<fst::VectorFst<Arc>> fst_dest;
      if (!ReadFst<Arc>(argv[dest + 1], &fst_dest) ||
          !transfer.TransferNGramsTo(fst_dest.get(), contexts[dest])) {
        return false;
      }
      std::ostringstream suffix;
      suffix.width(5);
      suffix.fill('0');
      suffix << dest;
      std::string out_name = out_name_prefix + suffix.str();
      fst_dest->Write(out_name);
    }
  }
  return true;
}

}  // namespace

int ngramtransfer_main(int argc, char **argv) {
  std::string usage = "Transfer n-grams between models.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] --ofile=out.fst in1.fst in2.fst [in3.fst ...]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (argc < 3) {
    ShowUsage();
    return 1;
  }

  std::string out_name_prefix = FST_FLAGS_ofile.empty()
                                    ? (argc > 3 ? argv[3] : "")
                                    : FST_FLAGS_ofile;

  int in_count = FST_FLAGS_ofile.empty() ? 2 : argc - 1;

  if (FST_FLAGS_index < 0 ||
      FST_FLAGS_index >= in_count) {
    LOG(ERROR) << "Bad FST index: " << FST_FLAGS_index;
    return 1;
  }

  if (FST_FLAGS_method == "histogram_transfer") {
    if (!Transfer<ngram::HistogramArc>(out_name_prefix, in_count, argv))
      return 1;
  } else if (FST_FLAGS_method == "count_transfer") {
    if (!Transfer<fst::StdArc>(out_name_prefix, in_count, argv))
      return 1;
  } else {
    LOG(ERROR) << argv[0]
               << ": bad transfer method: " << FST_FLAGS_method;
    return 1;
  }
  return 0;
}
