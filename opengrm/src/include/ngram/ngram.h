// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2016 Brian Roark and Google, Inc.
// This convenience file includes all other GRM NGram include files.

#ifndef NGRAM_NGRAM_H_
#define NGRAM_NGRAM_H_

#include <ngram/hist-arc.h>
#include <ngram/hist-mapper.h>
#include <ngram/lexicographic-map.h>
#include <ngram/ngram-absolute.h>
#include <ngram/ngram-bayes-model-merge.h>
#include <ngram/ngram-complete.h>
#include <ngram/ngram-context.h>
#include <ngram/ngram-context-merge.h>
#include <ngram/ngram-context-prune.h>
#include <ngram/ngram-count.h>
#include <ngram/ngram-count-merge.h>
#include <ngram/ngram-count-of-counts.h>
#include <ngram/ngram-count-prune.h>
#include <ngram/ngram-hist-merge.h>
#include <ngram/ngram-input.h>
#include <ngram/ngram-katz.h>
#include <ngram/ngram-kneser-ney.h>
#include <ngram/ngram-list-prune.h>
#include <ngram/ngram-make.h>
#include <ngram/ngram-marginalize.h>
#include <ngram/ngram-merge.h>
#include <ngram/ngram-model.h>
#include <ngram/ngram-model-merge.h>
#include <ngram/ngram-mutable-model.h>
#include <ngram/ngram-output.h>
#include <ngram/ngram-randgen.h>
#include <ngram/ngram-relentropy.h>
#include <ngram/ngram-replace-merge.h>
#include <ngram/ngram-seymore-shrink.h>
#include <ngram/ngram-shrink.h>
#include <ngram/ngram-split.h>
#include <ngram/ngram-transfer.h>
#include <ngram/ngram-unsmoothed.h>
#include <ngram/ngram-witten-bell.h>
#include <ngram/util.h>

#endif  // NGRAM_NGRAM_H_
