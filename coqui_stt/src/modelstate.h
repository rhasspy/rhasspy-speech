#ifndef MODELSTATE_H
#define MODELSTATE_H

#define STT_ERR_OK 0x0000
#define STT_ERR_NO_MODEL 0x1000
#define STT_ERR_INVALID_ALPHABET 0x2000
#define STT_ERR_INVALID_SHAPE 0x2001
#define STT_ERR_INVALID_SCORER 0x2002
#define STT_ERR_MODEL_INCOMPATIBLE 0x2003
#define STT_ERR_SCORER_NOT_ENABLED 0x2004
#define STT_ERR_SCORER_UNREADABLE 0x2005
#define STT_ERR_SCORER_INVALID_LM 0x2006
#define STT_ERR_SCORER_NO_TRIE 0x2007
#define STT_ERR_SCORER_INVALID_TRIE 0x2008
#define STT_ERR_SCORER_VERSION_MISMATCH 0x2009
#define STT_ERR_FAIL_INIT_MMAP 0x3000
#define STT_ERR_FAIL_INIT_SESS 0x3001
#define STT_ERR_FAIL_INTERPRETER 0x3002
#define STT_ERR_FAIL_RUN_SESS 0x3003
#define STT_ERR_FAIL_CREATE_STREAM 0x3004
#define STT_ERR_FAIL_READ_PROTOBUF 0x3005
#define STT_ERR_FAIL_CREATE_SESS 0x3006
#define STT_ERR_FAIL_CREATE_MODEL 0x3007
#define STT_ERR_FAIL_INSERT_HOTWORD 0x3008
#define STT_ERR_FAIL_CLEAR_HOTWORD 0x3009
#define STT_ERR_FAIL_ERASE_HOTWORD 0x3010

#include <vector>

// #include "coqui-stt.h"
#include "alphabet.h"

// #include "ctcdecode/scorer.h"
// #include "ctcdecode/output.h"

struct DecoderState;

struct ModelState {
  static constexpr unsigned int BATCH_SIZE = 1;

  Alphabet alphabet_;
  unsigned int n_steps_;
  unsigned int n_context_;
  unsigned int n_features_;
  unsigned int mfcc_feats_per_timestep_;
  unsigned int sample_rate_;
  unsigned int audio_win_len_;
  unsigned int audio_win_step_;
  unsigned int state_size_;

  ModelState();
  virtual ~ModelState();

  virtual int init(const char *model_string, bool init_from_bytes,
                   std::size_t bufferSize);

  virtual void compute_mfcc(const std::vector<float> &audio_buffer,
                            std::vector<float> &mfcc_output) = 0;

  /**
   * @brief Do a single inference step in the acoustic model, with:
   *          input=mfcc
   *          input_lengths=[n_frames]
   *
   * @param mfcc batch input data
   * @param n_frames number of timesteps in the data
   *
   * @param[out] output_logits Where to store computed logits.
   */
  virtual void infer(const std::vector<float> &mfcc, unsigned int n_frames,
                     const std::vector<float> &previous_state_c,
                     const std::vector<float> &previous_state_h,
                     std::vector<float> &logits_output,
                     std::vector<float> &state_c_output,
                     std::vector<float> &state_h_output) = 0;

  Alphabet &getAlphabet() { return this->alphabet_; }
};

#endif // MODELSTATE_H
