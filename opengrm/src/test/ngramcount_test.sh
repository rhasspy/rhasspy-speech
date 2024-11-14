#!/bin/bash
# Tests the command line binary ngramcount.

set -eou pipefail

readonly BIN="../bin"
readonly TESTDATA="${srcdir}/testdata"
readonly TEST_TMPDIR="${TEST_TMPDIR:-$(mktemp -d)}"

compile_test_fst() {
  fstcompile \
    --isymbols="${TESTDATA}/${1}.sym" \
    --osymbols="${TESTDATA}/${1}.sym" \
    --keep_isymbols \
    --keep_osymbols \
    --keep_state_numbering \
    "${TESTDATA}/${1}.txt" \
    "${TEST_TMPDIR}/${1}.ref"
}

compile_test_far() {
  compile_test_fst "${1}"
  farcreate \
    "${TEST_TMPDIR}/${1}.ref" \
    "${TEST_TMPDIR}/${1}.far"
}

farcompilestrings \
  --fst_type=compact \
  --symbols="${TESTDATA}/earnest.sym" \
  --keep_symbols \
  "${TESTDATA}/earnest.txt" \
  "${TEST_TMPDIR}/earnest.far"

compile_test_fst earnest.cnts
# Counting from a FAR of string FSTs.
"${BIN}/ngramcount" \
  --order=5 \
  "${TEST_TMPDIR}/earnest.far" \
  "${TEST_TMPDIR}/earnest.cnts"
fstequal \
  "${TEST_TMPDIR}/earnest.cnts.ref" \
  "${TEST_TMPDIR}/earnest.cnts"

compile_test_far earnest.fst
compile_test_fst earnest-fst.cnts
# Counting from an FST representing a union of paths.
"${BIN}/ngramcount" \
  --order=5 \
  "${TEST_TMPDIR}/earnest.fst.far" \
  "${TEST_TMPDIR}/earnest.cnts"
fstequal \
  "${TEST_TMPDIR}/earnest-fst.cnts.ref" \
  "${TEST_TMPDIR}/earnest.cnts"

compile_test_far earnest.det
compile_test_fst earnest-det.cnts
# Counting from the deterministic "tree" FST representing the corpus.
"${BIN}/ngramcount" \
  --order=5 \
  "${TEST_TMPDIR}/earnest.det.far" \
  "${TEST_TMPDIR}/earnest-det.cnts"
fstequal \
  "${TEST_TMPDIR}/earnest-det.cnts.ref" \
  "${TEST_TMPDIR}/earnest-det.cnts"

compile_test_far earnest.min
compile_test_fst earnest-min.cnts
# Counting from the minimal deterministic FST representing the corpus.
"${BIN}/ngramcount" \
  --order=5 \
  "${TEST_TMPDIR}/earnest.min.far" \
  "${TEST_TMPDIR}/earnest-min.cnts"
fstequal \
  "${TEST_TMPDIR}/earnest-min.cnts.ref" \
  "${TEST_TMPDIR}/earnest-min.cnts"

compile_test_fst earnest.cnt_of_cnts
# Counting from counts.
"${BIN}/ngramcount" \
  --method=count_of_counts \
  "${TEST_TMPDIR}/earnest.cnts.ref" \
  "${TEST_TMPDIR}/earnest.cnt_of_cnts"
fstequal \
  "${TEST_TMPDIR}/earnest.cnt_of_cnts.ref" \
  "${TEST_TMPDIR}/earnest.cnt_of_cnts"
