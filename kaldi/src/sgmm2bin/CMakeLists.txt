# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
add_kaldi_executable(NAME sgmm2-est-ebw SOURCES sgmm2-est-ebw.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-latgen-faster SOURCES sgmm2-latgen-faster.cc DEPENDS kaldi-sgmm2 kaldi-decoder kaldi-hmm)
add_kaldi_executable(NAME sgmm2-acc-stats SOURCES sgmm2-acc-stats.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-project SOURCES sgmm2-project.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-sum-accs SOURCES sgmm2-sum-accs.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-latgen-faster-parallel SOURCES sgmm2-latgen-faster-parallel.cc DEPENDS kaldi-sgmm2 kaldi-decoder kaldi-hmm)
add_kaldi_executable(NAME sgmm2-est-spkvecs SOURCES sgmm2-est-spkvecs.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-comp-prexform SOURCES sgmm2-comp-prexform.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME init-ubm SOURCES init-ubm.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-est-spkvecs-gpost SOURCES sgmm2-est-spkvecs-gpost.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-init SOURCES sgmm2-init.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-est-fmllr SOURCES sgmm2-est-fmllr.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-acc-stats2 SOURCES sgmm2-acc-stats2.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-acc-stats-gpost SOURCES sgmm2-acc-stats-gpost.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-gselect SOURCES sgmm2-gselect.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-align-compiled SOURCES sgmm2-align-compiled.cc DEPENDS kaldi-sgmm2 kaldi-decoder kaldi-hmm)
add_kaldi_executable(NAME sgmm2-est SOURCES sgmm2-est.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-rescore-lattice SOURCES sgmm2-rescore-lattice.cc DEPENDS kaldi-sgmm2 kaldi-hmm kaldi-lat)
add_kaldi_executable(NAME sgmm2-copy SOURCES sgmm2-copy.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-post-to-gpost SOURCES sgmm2-post-to-gpost.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
add_kaldi_executable(NAME sgmm2-info SOURCES sgmm2-info.cc DEPENDS kaldi-sgmm2 kaldi-hmm)
