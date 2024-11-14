#!/bin/bash
# Tests that ngramcount computes correct histogram counts.

set -eou pipefail

readonly BIN="../bin"
readonly TESTDATA="${srcdir}/testdata"
readonly TEST_TMPDIR="${TEST_TMPDIR:-$(mktemp -d)}"

compile_test_fst() {
  fstcompile \
    --isymbols="${TESTDATA}/${2}.sym" \
    --osymbols="${TESTDATA}/${2}.sym" \
    --keep_isymbols \
    --keep_osymbols \
    --keep_state_numbering \
    "${TESTDATA}/${1}.txt" \
    "${TEST_TMPDIR}/${1}.ref"
}

compile_test_fst single_fst ab
farcreate \
  "${TEST_TMPDIR}/single_fst.ref" \
  "${TEST_TMPDIR}/single_fst.far"
"./ngramhisttest" \
  --ifile="${TESTDATA}/single_fst_ref.txt" \
  --syms="${TESTDATA}/ab.sym" \
  --ofile="${TEST_TMPDIR}/single_fst_ref.ref"
"${BIN}/ngramcount" \
  --order=3 \
  --method=histograms \
  "${TEST_TMPDIR}/single_fst.far" \
  "${TEST_TMPDIR}/single_fst.cnts"
"./ngramhisttest" \
  --ifile="${TEST_TMPDIR}/single_fst.cnts" \
  --cfile="${TEST_TMPDIR}/single_fst_ref.ref"

"./ngramhisttest" \
  --ifile="${TESTDATA}/hist.ref.txt" \
  --syms="${TESTDATA}/ab.sym" \
  --ofile="${TEST_TMPDIR}/hist.ref.ref"
compile_test_fst fst1.hist ab
compile_test_fst fst2.hist ab
farcreate \
  "${TEST_TMPDIR}/fst1.hist.ref" \
  "${TEST_TMPDIR}/fst2.hist.ref" \
  "${TEST_TMPDIR}/test.far"
"${BIN}/ngramcount" \
   --order=2 \
   --method=histograms \
  "${TEST_TMPDIR}/test.far" \
  "${TEST_TMPDIR}/test.cnts"
"./ngramhisttest" \
  --ifile="${TEST_TMPDIR}/test.cnts" \
  --cfile="${TEST_TMPDIR}/hist.ref.ref"
