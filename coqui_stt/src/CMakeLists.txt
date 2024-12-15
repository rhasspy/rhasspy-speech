cmake_minimum_required(VERSION 3.10)
project(coqui_stt)

add_subdirectory(../tensorflow/tensorflow/lite ../tensorflow/tensorflow/lite/build)

add_executable(stt_onlyprobs
  tflitemodelstate.cc modelstate.cc alphabet.cc stt.cc ctcdecode/decoder_utils.cpp stt_onlyprobs.cpp
  flashlight/lib/common/String.cpp
  flashlight/lib/common/System.cpp
  flashlight/lib/text/dictionary/Dictionary.cpp
  flashlight/lib/text/dictionary/Utils.cpp
)
target_include_directories(stt_onlyprobs PRIVATE ../tensorflow .)
target_link_libraries(stt_onlyprobs PRIVATE tensorflow-lite)

install(TARGETS stt_onlyprobs DESTINATION .)
