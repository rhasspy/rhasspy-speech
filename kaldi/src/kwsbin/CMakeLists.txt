# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
add_kaldi_executable(NAME kws-search SOURCES kws-search.cc DEPENDS kaldi-kws)
add_kaldi_executable(NAME print-proxy-keywords SOURCES print-proxy-keywords.cc DEPENDS kaldi-kws)
add_kaldi_executable(NAME compute-atwv SOURCES compute-atwv.cc DEPENDS kaldi-kws)
add_kaldi_executable(NAME lattice-to-kws-index SOURCES lattice-to-kws-index.cc DEPENDS kaldi-kws)
add_kaldi_executable(NAME transcripts-to-fsts SOURCES transcripts-to-fsts.cc DEPENDS kaldi-kws kaldi-fstext)
add_kaldi_executable(NAME generate-proxy-keywords SOURCES generate-proxy-keywords.cc DEPENDS kaldi-kws kaldi-fstext)
add_kaldi_executable(NAME kws-index-union SOURCES kws-index-union.cc DEPENDS kaldi-kws)
