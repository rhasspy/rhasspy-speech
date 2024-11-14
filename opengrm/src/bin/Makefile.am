AM_CPPFLAGS = -I$(srcdir)/../include
AM_LDFLAGS = -L/usr/local/lib/fst -lfstfarscript -lfstfar -lfstscript -lfst \
             -lm -ldl

bin_PROGRAMS = ngramapply \
               ngramcontext \
               ngramcount \
               ngraminfo \
               ngrammake \
               ngrammarginalize \
               ngrammerge \
               ngramperplexity \
               ngramprint \
               ngramrandgen \
               ngramread \
               ngramshrink \
               ngramsort \
               ngramsplit \
               ngramsymbols \
               ngramtransfer

dist_noinst_SCRIPTS = ngramdisttrain.sh ngramfractrain.sh

ngramapply_SOURCES = ngramapply.cc ngramapply-main.cc
ngramapply_LDADD = ../lib/libngram.la

ngramcontext_SOURCES = ngramcontext.cc ngramcontext-main.cc
ngramcontext_LDADD = ../lib/libngram.la

ngramcount_SOURCES = ngramcount.cc ngramcount-main.cc
ngramcount_LDADD = ../lib/libngram.la ../lib/libngramhist.la

ngraminfo_SOURCES = ngraminfo.cc ngraminfo-main.cc
ngraminfo_LDADD = ../lib/libngram.la

ngrammake_SOURCES = ngrammake.cc ngrammake-main.cc
ngrammake_LDADD = ../lib/libngram.la ../lib/libngramhist.la

ngrammarginalize_SOURCES = ngrammarginalize.cc ngrammarginalize-main.cc
ngrammarginalize_LDADD = ../lib/libngram.la

ngrammerge_SOURCES = ngrammerge.cc ngrammerge-main.cc
ngrammerge_LDADD = ../lib/libngram.la ../lib/libngramhist.la

ngramperplexity_SOURCES = ngramperplexity.cc ngramperplexity-main.cc
ngramperplexity_LDADD = ../lib/libngram.la

ngramprint_SOURCES = ngramprint.cc ngramprint-main.cc
ngramprint_LDADD = ../lib/libngram.la

ngramrandgen_SOURCES = ngramrandgen.cc ngramrandgen-main.cc
ngramrandgen_LDADD = ../lib/libngram.la

ngramread_SOURCES = ngramread.cc ngramread-main.cc
ngramread_LDADD = ../lib/libngram.la

ngramshrink_SOURCES = ngramshrink.cc ngramshrink-main.cc
ngramshrink_LDADD = ../lib/libngram.la

ngramsort_SOURCES = ngramsort.cc ngramsort-main.cc
ngramsort_LDADD = ../lib/libngram.la

ngramsplit_SOURCES = ngramsplit.cc ngramsplit-main.cc
ngramsplit_LDADD = ../lib/libngram.la ../lib/libngramhist.la

ngramsymbols_SOURCES = ngramsymbols.cc ngramsymbols-main.cc
ngramsymbols_LDADD = ../lib/libngram.la

ngramtransfer_SOURCES = ngramtransfer.cc ngramtransfer-main.cc
ngramtransfer_LDADD = ../lib/libngram.la ../lib/libngramhist.la
