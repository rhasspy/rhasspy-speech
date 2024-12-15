#include <cstdint>
#include <iostream>

#include "stt.h"

auto main(int argc, char *argv[]) -> int {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model>\n";
    return 1;
  }

  ModelState *model = STT_CreateModel(argv[1]);

  bool processingRequests = true;
  while (processingRequests) {
    StreamingState *streamingState = STT_CreateStream(model);

    while (true) {
      // Read the length prefix
      uint32_t length = 0;
      std::cin.read(reinterpret_cast<char *>(&length), sizeof(length));
      if (std::cin.gcount() == 0) {
        // Error
        processingRequests = false;
        break;
      }

      if (length == 0) {
        // Ready to process
        break;
      }

      std::vector<char> buffer(length);
      uint32_t bytesRead = 0;
      while (bytesRead < length) {
        std::cin.read(buffer.data(), length - bytesRead);
        if (std::cin.gcount() > 0) {
          bytesRead += std::cin.gcount();
        } else {
          // Error
          processingRequests = false;
          break;
        }
      }

      if (!processingRequests) {
        // Error
        break;
      }

      STT_FeedAudioContent(streamingState,
                           reinterpret_cast<short *>(buffer.data()),
                           buffer.size() / sizeof(short));
    }

    if (!processingRequests) {
      // Error
      STT_FreeStream(streamingState);
      streamingState = nullptr;
      break;
    }

    // Flush buffers
    STT_FinishStream(streamingState);

    std::cerr << "Frames: " << streamingState->getProbs().size() << std::endl;

    for (auto &frameProbs : streamingState->getProbs()) {
      for (auto charProb : frameProbs) {
        std::cout << charProb << " ";
      }

      std::cout << std::endl;
    }

    // Probabilities are done
    std::cout << std::endl;

    STT_FreeStream(streamingState);
    streamingState = nullptr;
  }

  STT_FreeModel(model);
  model = nullptr;

  return 0;
}
