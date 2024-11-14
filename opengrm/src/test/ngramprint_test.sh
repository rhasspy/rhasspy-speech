#!/bin/bash
# Tests the command line binary ngramprint and ngramread.

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
"${BIN}/ngramprint" \
  --ARPA \
  --check_consistency \
  "${TEST_TMPDIR}/earnest-witten_bell.mod.ref" \
  "${TEST_TMPDIR}/earnest.arpa"

cmp "${TESTDATA}/earnest.arpa" "${TEST_TMPDIR}/earnest.arpa"

"${BIN}/ngramread" --ARPA "${TESTDATA}/earnest.arpa" "${TEST_TMPDIR}/earnest.arpa.mod"

"${BIN}/ngramprint" \
  --ARPA \
  --check_consistency \
  "${TEST_TMPDIR}/earnest.arpa.mod" \
  | "${BIN}/ngramread" \
  --ARPA \
  - \
  "${TEST_TMPDIR}/earnest.arpa.mod2"

fstequal \
  "${TEST_TMPDIR}/earnest.arpa.mod" \
  "${TEST_TMPDIR}/earnest.arpa.mod2"

compile_test_fst earnest.cnts
"${BIN}/ngramprint" \
  --check_consistency \
  "${TEST_TMPDIR}/earnest.cnts.ref" \
  "${TEST_TMPDIR}/earnest.cnt.print"

cmp "${TESTDATA}/earnest.cnt.print" "${TEST_TMPDIR}/earnest.cnt.print"

"${BIN}/ngramread" \
  --symbols="${TESTDATA}/earnest.sym" \
  "${TESTDATA}/earnest.cnt.print" \
  "${TEST_TMPDIR}/earnest.cnts"

"${BIN}/ngramprint" \
  --check_consistency \
  "${TEST_TMPDIR}/earnest.cnts" \
  | "${BIN}/ngramread" \
  --symbols="${TESTDATA}/earnest.sym" \
  - \
  "${TEST_TMPDIR}/earnest.cnts2"

fstequal \
  "${TEST_TMPDIR}/earnest.cnts" \
  "${TEST_TMPDIR}/earnest.cnts2"
