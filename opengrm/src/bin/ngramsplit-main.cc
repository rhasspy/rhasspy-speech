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
// Splits an n-gram model based on given context patterns.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/extensions/far/far.h>
#include <fst/extensions/far/getters.h>
#include <fst/arc.h>
#include <fst/vector-fst.h>
#include <ngram/hist-arc.h>
#include <ngram/ngram-complete.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-split.h>
#include <ngram/util.h>

DECLARE_int64(backoff_label);
DECLARE_string(contexts);
DECLARE_string(method);
DECLARE_double(norm_eps);
DECLARE_bool(complete);
DECLARE_string(far_type);

namespace {

template <class Arc>
bool SplitToFsts(fst::VectorFst<Arc> *fst,
                 const std::vector<std::string> &context_patterns,
                 const std::string &out_name_prefix) {
  ngram::NGramSplit<Arc> split(*fst, context_patterns,
                               FST_FLAGS_backoff_label,
                               FST_FLAGS_norm_eps);
  for (int i = 0; !split.Done(); ++i) {
    fst::VectorFst<Arc> ofst;
    if (!split.NextNGramModel(&ofst)) return true;
    std::ostringstream suffix;
    suffix.width(5);
    suffix.fill('0');
    suffix << i;
    const auto out_name = out_name_prefix + suffix.str();
    if (!ofst.Write(out_name)) return true;
  }
  return false;
}

void GetSortedPatterns(const std::vector<std::string> &context_patterns,
                       std::vector<std::string> *sorted_full_patterns,
                       std::vector<std::string> *sorted_context_patterns) {
  for (size_t i = 0; i < context_patterns.size(); ++i) {
    sorted_full_patterns->push_back(context_patterns[i] + "/" +
                                    std::to_string(i));
  }
  std::sort(sorted_full_patterns->begin(), sorted_full_patterns->end());
  for (const auto &full_pattern : *sorted_full_patterns) {
    std::vector<std::string> split_pattern =
        StrSplit(full_pattern, '/', fst::SkipEmpty());
    CHECK_EQ(split_pattern.size(), 2);
    sorted_context_patterns->push_back(split_pattern[0]);
  }
}

template <class Arc>
bool SplitToFar(fst::VectorFst<Arc> *fst,
                const std::vector<std::string> &context_patterns,
                const std::string &out_name_prefix,
                const fst::FarType input_far_type) {
  std::vector<std::string> sorted_full_patterns;
  std::vector<std::string> sorted_context_patterns;
  GetSortedPatterns(context_patterns, &sorted_full_patterns,
                    &sorted_context_patterns);
  ngram::NGramSplit<Arc> split(*fst, sorted_context_patterns,
                               FST_FLAGS_backoff_label,
                               FST_FLAGS_norm_eps);
  std::unique_ptr<fst::FarWriter<Arc>> far_writer(
      fst::FarWriter<Arc>::Create(out_name_prefix + "split_fsts.far",
                                      input_far_type));
  if (!far_writer) {
    NGRAMERROR() << "Can't open " << out_name_prefix
                 << "split_fsts.far for writing";
    return true;
  }
  for (size_t i = 0; !split.Done(); ++i) {
    fst::VectorFst<Arc> ofst;
    if (!split.NextNGramModel(&ofst)) return true;
    CHECK_LT(i, context_patterns.size());
    far_writer->Add(sorted_full_patterns[i], ofst);
  }
  return false;
}

template <class Arc>
bool CountOrHistogramSplit(std::string_view in_name,
                           const std::vector<std::string> &context_patterns,
                           const std::string &out_name_prefix,
                           const std::string &input_far_type_str) {
  std::unique_ptr<fst::VectorFst<Arc>> fst(
      fst::VectorFst<Arc>::Read(in_name));
  if (!fst ||
      (FST_FLAGS_complete && !ngram::NGramComplete(fst.get()))) {
    return true;
  }
  if (input_far_type_str.empty()) {
    // When an empty string far_type is provided, this is not a FarType at all,
    // and it means there is a number of created output FSTs.
    return SplitToFsts(fst.get(), context_patterns, out_name_prefix);
  } else {
    // Otherwise, we expect the user plans to provide a well-formed FarType
    // string representation.
    fst::FarType far_type;
    if (!fst::script::GetFarType(input_far_type_str, &far_type)) {
      NGRAMERROR() << "Unknown or unsupported FAR type: " << input_far_type_str;
      return true;
    }
    if (!SplitToFar(fst.get(), context_patterns, out_name_prefix, far_type)) {
      return true;
    }
  }
  return false;
}

}  // namespace

int ngramsplit_main(int argc, char **argv) {
  std::string usage =
      "Split an n-gram model using context patterns.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options] in_fst [out_fsts_prefix]\n";
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (argc > 3 || argc < 2) {
    ShowUsage();
    return 1;
  }

  std::string in_name = strcmp(argv[1], "-") != 0 ? argv[1] : "";
  std::string out_name_prefix = argc > 2 ? argv[2] : in_name;

  std::vector<std::string> context_patterns;

  if (FST_FLAGS_contexts.empty()) {
    LOG(ERROR) << "Context patterns file need to be specified using "
               << "--contexts flag.";
    return 1;
  } else {
    ngram::NGramReadContexts(FST_FLAGS_contexts, &context_patterns);
  }

  if (FST_FLAGS_method == "count_split") {
    return CountOrHistogramSplit<fst::StdArc>(
        in_name, context_patterns, out_name_prefix,
        FST_FLAGS_far_type);
  } else if (FST_FLAGS_method == "histogram_split") {
    return CountOrHistogramSplit<ngram::HistogramArc>(
        in_name, context_patterns, out_name_prefix,
        FST_FLAGS_far_type);
  } else {
    LOG(ERROR) << argv[0]
               << ": bad split method: " << FST_FLAGS_method;
    return 1;
  }
  return 0;
}
