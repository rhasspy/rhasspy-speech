AM_CPPFLAGS = -I$(srcdir)/../include
AM_LDFLAGS = -L/usr/local/lib/fst -lfstfar -lfst -lm -ldl

bin_PROGRAMS = ngramhisttest ngramrandtest

ngramhisttest_SOURCES = ngramhisttest.cc ngramhisttest-main.cc
ngramhisttest_LDADD = -lfstscript ../lib/libngram.la ../lib/libngramhist.la

ngramrandtest_SOURCES = ngramrandtest.cc ngramrandtest-main.cc
ngramrandtest_LDADD = ../lib/libngram.la

dist_check_SCRIPTS = disttestsetup.sh \
                     ngramapply_test.sh \
                     ngramcount_histograms_test.sh \
                     ngramcount_test.sh \
                     ngramdistrand.sh \
                     ngramdistcount_test.sh \
                     ngramdistmake_test.sh \
                     ngramdistshrink_test.sh \
                     ngramdistwittenbell_test.sh \
                     ngramfracdistmake_test.sh \
                     ngramfracdistshrink_test.sh \
                     ngraminfo_test.sh \
                     ngrammake_test.sh \
                     ngrammarginalize_test.sh \
                     ngrammerge_test.sh \
                     ngramperplexity_test.sh \
                     ngramprint_test.sh \
                     ngramrandgen_test.sh \
                     ngramrand_test.sh \
                     ngramshrink_test.sh \
                     ngramsymbols_test.sh

dist_noinst_DATA = testdata/ab.sym \
                   testdata/earnest-absolute.mod.sym \
                   testdata/earnest-absolute.mod.txt \
                   testdata/earnest.arpa \
                   testdata/earnest.cnt_of_cnts.sym \
                   testdata/earnest.cnt_of_cnts.txt \
                   testdata/earnest.cnt.print \
                   testdata/earnest.cnts.sym \
                   testdata/earnest.cnts.txt \
                   testdata/earnest-count_prune.pru.sym \
                   testdata/earnest-count_prune.pru.txt \
                   testdata/earnest-det.cnts.sym \
                   testdata/earnest-det.cnts.txt \
                   testdata/earnest.det.sym \
                   testdata/earnest.det.txt \
                   testdata/earnest-fst.cnts.sym \
                   testdata/earnest-fst.cnts.txt \
                   testdata/earnest.fst.sym \
                   testdata/earnest.fst.txt \
                   testdata/earnest.info \
                   testdata/earnest-katz.marg.mod.sym \
                   testdata/earnest-katz.marg.mod.txt \
                   testdata/earnest-katz.mod.sym \
                   testdata/earnest-katz.mod.txt \
                   testdata/earnest-kneser_ney.mod.sym \
                   testdata/earnest-kneser_ney.mod.txt \
                   testdata/earnest-min.cnts.sym \
                   testdata/earnest-min.cnts.txt \
                   testdata/earnest.min.sym \
                   testdata/earnest.min.txt \
                   testdata/earnest.mod.sym \
                   testdata/earnest.mod.txt \
                   testdata/earnest.mrg.norm.sym \
                   testdata/earnest.mrg.norm.txt \
                   testdata/earnest.mrg.smooth.norm.sym \
                   testdata/earnest.mrg.smooth.norm.txt \
                   testdata/earnest.mrg.smooth.sym \
                   testdata/earnest.mrg.smooth.txt \
                   testdata/earnest.mrg.sym \
                   testdata/earnest.mrg.txt \
                   testdata/earnest.perp \
                   testdata/earnest.randgen.apply.FSTtxt.tgz \
                   testdata/earnest.randgen.apply.sym \
                   testdata/earnest.randgen.sym \
                   testdata/earnest.randgen.txt \
                   testdata/earnest-relative_entropy.pru.sym \
                   testdata/earnest-relative_entropy.pru.txt \
                   testdata/earnest-seymore.pru.sym \
                   testdata/earnest-seymore.pru.txt \
                   testdata/earnest.sym \
                   testdata/earnest.txt \
                   testdata/earnest-unsmoothed.mod.sym \
                   testdata/earnest-unsmoothed.mod.txt \
                   testdata/earnest-witten_bell.mod.sym \
                   testdata/earnest-witten_bell.mod.txt \
                   testdata/fst1.hist.txt \
                   testdata/fst2.hist.txt \
                   testdata/hist.ref.txt \
                   testdata/single_fst_ref.txt \
                   testdata/single_fst.txt

TESTS = ngramapply_test.sh \
        ngramcount_histograms_test.sh \
        ngramcount_test.sh \
        ngramdistcount_test.sh \
        ngramdistmake_test.sh \
        ngramdistshrink_test.sh \
        ngramdistwittenbell_test.sh \
        ngramfracdistmake_test.sh \
        ngramfracdistshrink_test.sh \
        ngraminfo_test.sh \
        ngrammake_test.sh \
        ngrammarginalize_test.sh \
        ngrammerge_test.sh \
        ngramperplexity_test.sh \
        ngramprint_test.sh \
        ngramrandgen_test.sh \
        ngramrand_test.sh \
        ngramshrink_test.sh \
        ngramsymbols_test.sh
