# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
add_kaldi_executable(NAME compute-fbank-online-batched-cuda SOURCES compute-fbank-online-batched-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME compute-mfcc-online-batched-cuda SOURCES compute-mfcc-online-batched-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME apply-cmvn-online-cuda SOURCES apply-cmvn-online-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME compute-online-feats-cuda SOURCES compute-online-feats-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME compute-mfcc-feats-cuda SOURCES compute-mfcc-feats-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME compute-fbank-feats-cuda SOURCES compute-fbank-feats-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME apply-batched-cmvn-online-cuda SOURCES apply-batched-cmvn-online-cuda.cc DEPENDS kaldi-cudafeat)
add_kaldi_executable(NAME compute-online-feats-batched-cuda SOURCES compute-online-feats-batched-cuda.cc DEPENDS kaldi-cudafeat)
