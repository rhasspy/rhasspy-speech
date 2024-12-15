#include "stt.h"

#include <algorithm>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "alphabet.h"
#include "modelstate.h"

#include "tflitemodelstate.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "libstt"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...)
#define LOGE(...)
#endif // __ANDROID__

using std::vector;

template <typename T> void shift_buffer_left(vector<T> &buf, int shift_amount) {
  std::rotate(buf.begin(), buf.begin() + shift_amount, buf.end());
  buf.resize(buf.size() - shift_amount);
}

void StreamingState::feedAudioContent(const short *buffer,
                                      unsigned int buffer_size) {
  // Consume all the data that was passed in, processing full buffers if needed
  while (buffer_size > 0) {
    while (buffer_size > 0 && audio_buffer_.size() < model_->audio_win_len_) {
      // Convert i16 sample into f32
      float multiplier = 1.0f / (1 << 15);
      audio_buffer_.push_back((float)(*buffer) * multiplier);
      ++buffer;
      --buffer_size;
    }

    // If the buffer is full, process and shift it
    if (audio_buffer_.size() == model_->audio_win_len_) {
      processAudioWindow(audio_buffer_);
      // Shift data by one step
      shift_buffer_left(audio_buffer_, model_->audio_win_step_);
    }

    // Repeat until buffer empty
  }
}

// char*
void StreamingState::finishStream() { flushBuffers(true); }

void StreamingState::processAudioWindow(const vector<float> &buf) {
  // Compute MFCC features
  vector<float> mfcc;
  mfcc.reserve(model_->n_features_);
  model_->compute_mfcc(buf, mfcc);
  pushMfccBuffer(mfcc);
}

void StreamingState::flushBuffers(bool addZeroMfccVectors) {
  // Flush audio buffer
  processAudioWindow(audio_buffer_);

  if (addZeroMfccVectors) {
    // Add empty mfcc vectors at end of sample
    for (int i = 0; i < model_->n_context_; ++i) {
      addZeroMfccWindow();
    }
  }

  // Process batch if there's inputs to be processed
  if (batch_buffer_.size() > 0) {
    processBatch(batch_buffer_,
                 batch_buffer_.size() / model_->mfcc_feats_per_timestep_);
    batch_buffer_.resize(0);
  }
}

void StreamingState::addZeroMfccWindow() {
  vector<float> zero_buffer(model_->n_features_, 0.f);
  pushMfccBuffer(zero_buffer);
}

template <typename InputIt, typename OutputIt>
InputIt copy_up_to_n(InputIt from_begin, InputIt from_end, OutputIt to_begin,
                     int max_elems) {
  int next_copy_amount =
      std::min<int>(std::distance(from_begin, from_end), max_elems);
  std::copy_n(from_begin, next_copy_amount, to_begin);
  return from_begin + next_copy_amount;
}

void StreamingState::pushMfccBuffer(const vector<float> &buf) {
  auto start = buf.begin();
  auto end = buf.end();
  while (start != end) {
    // Copy from input buffer to mfcc_buffer, stopping if we have a full context
    // window
    start =
        copy_up_to_n(start, end, std::back_inserter(mfcc_buffer_),
                     model_->mfcc_feats_per_timestep_ - mfcc_buffer_.size());
    assert(mfcc_buffer_.size() <= model_->mfcc_feats_per_timestep_);

    // If we have a full context window
    if (mfcc_buffer_.size() == model_->mfcc_feats_per_timestep_) {
      processMfccWindow(mfcc_buffer_);
      // Shift data by one step of one mfcc feature vector
      shift_buffer_left(mfcc_buffer_, model_->n_features_);
    }
  }
}

void StreamingState::processMfccWindow(const vector<float> &buf) {
  auto start = buf.begin();
  auto end = buf.end();
  while (start != end) {
    // Copy from input buffer to batch_buffer, stopping if we have a full batch
    start = copy_up_to_n(start, end, std::back_inserter(batch_buffer_),
                         model_->n_steps_ * model_->mfcc_feats_per_timestep_ -
                             batch_buffer_.size());
    assert(batch_buffer_.size() <=
           model_->n_steps_ * model_->mfcc_feats_per_timestep_);

    // If we have a full batch
    if (batch_buffer_.size() ==
        model_->n_steps_ * model_->mfcc_feats_per_timestep_) {
      processBatch(batch_buffer_, model_->n_steps_);
      batch_buffer_.resize(0);
    }
  }
}

void StreamingState::processBatch(const vector<float> &buf,
                                  unsigned int n_steps) {
  vector<float> logits;
  model_->infer(buf, n_steps, previous_state_c_, previous_state_h_, logits,
                previous_state_c_, previous_state_h_);

  const size_t num_classes = model_->alphabet_.GetSize() + 1; // +1 for blank
  const int n_frames = logits.size() / (ModelState::BATCH_SIZE * num_classes);

  // Convert logits to double
  auto logits_iter = logits.begin();
  bool new_frame = true;
  std::vector<double> *frame_probs = nullptr;

  while (logits_iter != logits.end()) {
    if (new_frame) {
      probs_.emplace_back();
      frame_probs = &probs_[probs_.size() - 1];
      new_frame = false;
    }

    if (!frame_probs) {
      break;
    }

    frame_probs->push_back(*logits_iter);
    if (frame_probs->size() == num_classes) {
      new_frame = true;
    }

    ++logits_iter;
  }
}

int CreateModelImpl(const char *aModelString, bool init_from_bytes,
                    ModelState **retval, unsigned int aBufferSize = 0) {
  *retval = nullptr;

  if ((!init_from_bytes && !strlen(aModelString)) ||
      (init_from_bytes && !aBufferSize)) {
    std::cerr << "No model specified, cannot continue." << std::endl;
    return STT_ERR_NO_MODEL;
  }

  std::unique_ptr<ModelState> model(new TFLiteModelState());

  if (!model) {
    std::cerr << "Could not allocate model state." << std::endl;
    return STT_ERR_FAIL_CREATE_MODEL;
  }

  int err = model->init(aModelString, init_from_bytes, aBufferSize);
  if (err != STT_ERR_OK) {
    return err;
  }

  *retval = model.release();
  return STT_ERR_OK;
}

ModelState *STT_CreateModel(const char *aModelPath) {
  ModelState *model = nullptr;
  if (CreateModelImpl(aModelPath, false, &model) != STT_ERR_OK) {
    return nullptr;
  }

  return model;
}

int STT_CreateModelFromBuffer(const char *aModelBuffer,
                              unsigned int aBufferSize, ModelState **retval) {
  return CreateModelImpl(aModelBuffer, true, retval, aBufferSize);
}

int STT_GetModelSampleRate(const ModelState *aCtx) {
  return aCtx->sample_rate_;
}

void STT_FreeModel(ModelState *ctx) { delete ctx; }

StreamingState *STT_CreateStream(ModelState *aCtx) {
  if (!aCtx) {
    return nullptr;
  }

  std::unique_ptr<StreamingState> ctx(new StreamingState());

  ctx->audio_buffer_.reserve(aCtx->audio_win_len_);
  ctx->mfcc_buffer_.reserve(aCtx->mfcc_feats_per_timestep_);
  ctx->mfcc_buffer_.resize(aCtx->n_features_ * aCtx->n_context_, 0.f);
  ctx->batch_buffer_.reserve(aCtx->n_steps_ * aCtx->mfcc_feats_per_timestep_);
  ctx->previous_state_c_.resize(aCtx->state_size_, 0.f);
  ctx->previous_state_h_.resize(aCtx->state_size_, 0.f);
  ctx->model_ = aCtx;

  return ctx.release();
}

void STT_FeedAudioContent(StreamingState *aSctx, const short *aBuffer,
                          unsigned int aBufferSize) {
  aSctx->feedAudioContent(aBuffer, aBufferSize);
}

void STT_FreeStream(StreamingState *aSctx) { delete aSctx; }

vector<vector<double>> &STT_FinishStream(StreamingState *aSctx) {
  aSctx->finishStream();
  return aSctx->getProbs();
}
