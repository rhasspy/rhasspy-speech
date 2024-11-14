#!/bin/bash
# Tests the command line binary ngrammarginalize.

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

compile_test_fst earnest-katz.mod
compile_test_fst earnest-katz.marg.mod
"${BIN}/ngrammarginalize" \
  "${TEST_TMPDIR}/earnest-katz.mod.ref" \
  "${TEST_TMPDIR}/earnest-katz.marg.mod"

fstequal \
  "${TEST_TMPDIR}/earnest-katz.marg.mod.ref" \
  "${TEST_TMPDIR}/earnest-katz.marg.mod"
