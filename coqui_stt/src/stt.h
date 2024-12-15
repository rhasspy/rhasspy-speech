#ifndef STT_H_
#define STT_H_

#include <vector>

#include "modelstate.h"

using std::vector;

struct StreamingState {
  vector<float> audio_buffer_;
  vector<float> mfcc_buffer_;
  vector<float> batch_buffer_;
  vector<float> previous_state_c_;
  vector<float> previous_state_h_;

  vector<vector<double>> probs_;
  ModelState *model_;

  StreamingState(){};
  ~StreamingState(){};

  void feedAudioContent(const short *buffer, unsigned int buffer_size);
  void flushBuffers(bool addZeroMfccVectors = false);
  void finishStream();

  void processAudioWindow(const vector<float> &buf);
  void processMfccWindow(const vector<float> &buf);
  void pushMfccBuffer(const vector<float> &buf);
  void addZeroMfccWindow();
  void processBatch(const vector<float> &buf, unsigned int n_steps);

  vector<vector<double>> &getProbs() { return this->probs_; }
};

ModelState *STT_CreateModel(const char *aModelPath);
StreamingState *STT_CreateStream(ModelState *aCtx);
void STT_FreeStream(StreamingState *aSctx);
void STT_FreeModel(ModelState *ctx);
void STT_FeedAudioContent(StreamingState *aSctx, const short *aBuffer,
                          unsigned int aBufferSize);
vector<vector<double>> &STT_FinishStream(StreamingState *aSctx);

#endif // STT_H_
