AM_CPPFLAGS = -I$(srcdir)/../include

lib_LTLIBRARIES = libngram.la libngramhist.la hist-arc.la

libngram_la_SOURCES = ngram-absolute.cc \
                      ngram-context.cc \
                      ngram-count.cc \
                      ngram-count-prune.cc \
                      ngram-input.cc \
                      ngram-kneser-ney.cc \
                      ngram-list-prune.cc \
                      ngram-make.cc \
                      ngram-marginalize.cc \
                      ngram-output.cc \
                      ngram-shrink.cc \
                      util.cc
libngram_la_LDFLAGS = -version-info 1316:0:0 -lfst -lm
libngram_la_LIBADD = $(DL_LIBS)

libngramhist_la_SOURCES = hist-arc.cc
libngramhist_la_LDFLAGS = -version-info 1316:0:0 -lfst -lfstscript -lm
libngramhist_la_LIBADD =  $(DL_LIBS)

hist_arc_la_SOURCES = hist-arc.cc
hist_arc_la_LDFLAGS = -module
