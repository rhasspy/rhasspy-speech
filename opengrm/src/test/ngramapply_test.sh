#!/bin/bash
# Tests the command line binary ngramapply.

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

compile_test_fst earnest-witten_bell.mod

# Create FARs.
farcompilestrings \
  --fst_type=compact \
  --key_prefix="FST" \
  --generate_keys=4 \
  --symbols="${TESTDATA}/earnest.randgen.apply.sym" \
  --keep_symbols \
  "${TESTDATA}/earnest.randgen.txt" \
  "${TEST_TMPDIR}/earnest.far"
tar -xzf \
  "${TESTDATA}/earnest.randgen.apply.FSTtxt.tgz" \
  -C \
  "${TEST_TMPDIR}"
fstcompile \
  --isymbols="${TESTDATA}/earnest.randgen.sym" \
  --osymbols="${TESTDATA}/earnest.randgen.sym" \
  --keep_state_numbering \
  "${TEST_TMPDIR}/FST0001.txt" \
  "${TEST_TMPDIR}/FST0001"
ls "${TEST_TMPDIR}/FST"????.txt \
  | grep -v FST0001 \
  | sed 's/.txt$//g' \
  | while read I; do
    fstcompile \
    "${I}.txt" \
    "${I}"
  done
farcreate \
    "${TEST_TMPDIR}/FST"???? "${TEST_TMPDIR}/earnest.apply.far.ref"

"${BIN}/ngramapply" \
  "${TEST_TMPDIR}/earnest-witten_bell.mod.ref" \
  "${TEST_TMPDIR}/earnest.far" \
  "${TEST_TMPDIR}/earnest.apply.far"

farequal \
  "${TEST_TMPDIR}/earnest.apply.far.ref" \
  "${TEST_TMPDIR}/earnest.apply.far"
