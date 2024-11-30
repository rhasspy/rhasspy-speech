// online2bin/online2-cli-nnet3-decode-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)
//           2018  Polish-Japanese Academy of Information Technology (Author:
//           Danijel Korzinek) 2024  Michael Hansen (mike@rhasspy.org)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-timing.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

#include <cstdio>
#include <string>
#include <unistd.h>

const std::size_t chunk_samples = 1024;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in audio from stdin and performs online\n"
        "decoding with neural nets (nnet3 setup), with iVector-based\n"
        "speaker adaptation and endpointing.\n"
        "Note: some configuration values and inputs are set via config\n"
        "files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-cli-nnet3-decode-faster-confidence [options] "
        "<nnet3-in> "
        "<fst-in> <word-symbol-table> <lattice-wspecifier>\n";

    ParseOptions po(usage);

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat samp_freq = 16000.0;

    po.Register(
        "samp-freq", &samp_freq,
        "Sampling frequency of the input signal (coded as 16-bit slinear).");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1), fst_rxfilename = po.GetArg(2),
                word_syms_filename = po.GetArg(3),
                clat_wspecifier = po.GetArg(4);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    BaseFloat frame_shift = feature_info.FrameShiftInSeconds();
    int32 frame_subsampling = decodable_opts.frame_subsampling_factor;
    BaseFloat time_unit = frame_shift * frame_subsampling;

    KALDI_VLOG(1) << "Loading AM...";

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    KALDI_VLOG(1) << "Loading FST...";

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (!word_syms_filename.empty())
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    CompactLatticeWriter clat_writer(clat_wspecifier);

    KALDI_VLOG(1) << "Initializing decoder...";

    OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
    SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                        decodable_info, *decode_fst,
                                        &feature_pipeline);
    decoder.InitDecoding();
    OnlineSilenceWeighting silence_weighting(
        trans_model, feature_info.silence_weighting_config,
        decodable_opts.frame_subsampling_factor);
    std::vector<std::pair<int32, BaseFloat>> delta_weights;

    int16_t samples[chunk_samples];

    // Read 16-bit samples from stdin until EOF.
    freopen(NULL, "rb", stdin);
    size_t samples_read = fread(samples, sizeof(int16_t), chunk_samples, stdin);
    while (samples_read > 0) {
      Vector<BaseFloat> wave_part(samples_read);
      for (std::size_t i = 0; i < samples_read; ++i) {
        wave_part(i) = static_cast<BaseFloat>(samples[i]);
      }

      feature_pipeline.AcceptWaveform(samp_freq, wave_part);
      decoder.AdvanceDecoding();
      samples_read = fread(samples, sizeof(int16_t), chunk_samples, stdin);
    }

    KALDI_VLOG(1) << "Finished decoding.";

    // Finish decoding
    feature_pipeline.InputFinished();

    decoder.AdvanceDecoding();
    decoder.FinalizeDecoding();

    // Write lattice
    CompactLattice lat;
    decoder.GetLattice(true, &lat);

    BaseFloat inv_acoustic_scale = 1.0 / decodable_opts.acoustic_scale;
    ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &lat);

    clat_writer.Write("utt", lat);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

  return 0;

} // main()
