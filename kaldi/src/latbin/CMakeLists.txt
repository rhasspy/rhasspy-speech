# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
add_kaldi_executable(NAME nbest-to-linear SOURCES nbest-to-linear.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-depth-per-frame SOURCES lattice-depth-per-frame.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-1best SOURCES lattice-1best.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-add-trans-probs SOURCES lattice-add-trans-probs.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-prune SOURCES lattice-prune.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-equivalent SOURCES lattice-equivalent.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-compose SOURCES lattice-compose.cc DEPENDS kaldi-lat kaldi-fstext)
add_kaldi_executable(NAME lattice-limit-depth SOURCES lattice-limit-depth.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-reverse SOURCES lattice-reverse.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-fst SOURCES lattice-to-fst.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-scale SOURCES lattice-scale.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-copy-backoff SOURCES lattice-copy-backoff.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-determinize SOURCES lattice-determinize.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-lmrescore-pruned SOURCES lattice-lmrescore-pruned.cc DEPENDS kaldi-lat kaldi-fstext kaldi-rnnlm)
add_kaldi_executable(NAME lattice-determinize-pruned-parallel SOURCES lattice-determinize-pruned-parallel.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-best-path SOURCES lattice-best-path.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-oracle SOURCES lattice-oracle.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-align-phones SOURCES lattice-align-phones.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-copy SOURCES lattice-copy.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-lmrescore-rnnlm SOURCES lattice-lmrescore-rnnlm.cc DEPENDS kaldi-lat kaldi-fstext kaldi-rnnlm)
add_kaldi_executable(NAME lattice-lmrescore-kaldi-rnnlm SOURCES lattice-lmrescore-kaldi-rnnlm.cc DEPENDS kaldi-lat kaldi-fstext kaldi-rnnlm)
add_kaldi_executable(NAME lattice-align-words-lexicon SOURCES lattice-align-words-lexicon.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-union SOURCES lattice-union.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-difference SOURCES lattice-difference.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-mbr-decode SOURCES lattice-mbr-decode.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-ctm-conf SOURCES lattice-to-ctm-conf.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-push SOURCES lattice-push.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-arc-post SOURCES lattice-arc-post.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-confidence SOURCES lattice-confidence.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-determinize-phone-pruned-parallel SOURCES lattice-determinize-phone-pruned-parallel.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME nbest-to-lattice SOURCES nbest-to-lattice.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME nbest-to-prons SOURCES nbest-to-prons.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-expand-ngram SOURCES lattice-expand-ngram.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-project SOURCES lattice-project.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-combine SOURCES lattice-combine.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-lmrescore-const-arpa SOURCES lattice-lmrescore-const-arpa.cc DEPENDS kaldi-lat kaldi-fstext kaldi-rnnlm)
add_kaldi_executable(NAME lattice-determinize-pruned SOURCES lattice-determinize-pruned.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-determinize-non-compact SOURCES lattice-determinize-non-compact.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-depth SOURCES lattice-depth.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-post SOURCES lattice-to-post.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-boost-ali SOURCES lattice-boost-ali.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-minimize SOURCES lattice-minimize.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-phone-lattice SOURCES lattice-to-phone-lattice.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME nbest-to-ctm SOURCES nbest-to-ctm.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-align-words SOURCES lattice-align-words.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-mpe-post SOURCES lattice-to-mpe-post.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-lmrescore SOURCES lattice-lmrescore.cc DEPENDS kaldi-lat kaldi-fstext)
add_kaldi_executable(NAME lattice-add-penalty SOURCES lattice-add-penalty.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME linear-to-nbest SOURCES linear-to-nbest.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-interp SOURCES lattice-interp.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-rmali SOURCES lattice-rmali.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-lmrescore-kaldi-rnnlm-pruned SOURCES lattice-lmrescore-kaldi-rnnlm-pruned.cc DEPENDS kaldi-lat kaldi-fstext kaldi-rnnlm)
add_kaldi_executable(NAME lattice-determinize-phone-pruned SOURCES lattice-determinize-phone-pruned.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-nbest SOURCES lattice-to-nbest.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-to-smbr-post SOURCES lattice-to-smbr-post.cc DEPENDS kaldi-lat)
add_kaldi_executable(NAME lattice-rescore-mapped SOURCES lattice-rescore-mapped.cc DEPENDS kaldi-lat)
