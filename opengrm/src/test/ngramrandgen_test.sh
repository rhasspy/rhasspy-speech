#!/bin/bash
# Tests the command line binary ngramrandgen.

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

echo "a" > "${TEST_TMPDIR}/randgen.far"
echo "b" > "${TEST_TMPDIR}/randgen2.far"
compile_test_fst earnest.mod
"${BIN}/ngramrandgen" \
  --max_sents=1000 \
  --seed=12 \
  "${TEST_TMPDIR}/earnest.mod.ref" \
  "${TEST_TMPDIR}/randgen.far"
"${BIN}/ngramrandgen" \
  --max_sents=1000 \
  --seed=12 \
  "${TEST_TMPDIR}/earnest.mod.ref" \
  "${TEST_TMPDIR}/randgen2.far"

farequal \
  "${TEST_TMPDIR}/randgen.far" \
  "${TEST_TMPDIR}/randgen2.far"
