#!/bin/bash
# Tests the command line binary ngrammerge.

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

compile_test_fst earnest-absolute.mod
compile_test_fst earnest-seymore.pru
compile_test_fst earnest.mrg
"${BIN}/ngrammerge" \
  --check_consistency \
  --method=count_merge \
  "${TEST_TMPDIR}/earnest-absolute.mod.ref" \
  "${TEST_TMPDIR}/earnest-seymore.pru.ref" \
  "${TEST_TMPDIR}/earnest.mrg"

fstequal \
  "${TEST_TMPDIR}/earnest.mrg.ref" \
  "${TEST_TMPDIR}/earnest.mrg"

compile_test_fst earnest.mrg.norm
"${BIN}/ngrammerge" \
  --check_consistency \
  --method=count_merge \
  --normalize \
  "${TEST_TMPDIR}/earnest-absolute.mod.ref" \
  "${TEST_TMPDIR}/earnest-seymore.pru.ref" \
  "${TEST_TMPDIR}/earnest.mrg.norm"

fstequal \
  "${TEST_TMPDIR}/earnest.mrg.norm.ref" \
  "${TEST_TMPDIR}/earnest.mrg.norm"

compile_test_fst earnest.mrg.smooth
"${BIN}/ngrammerge" \
  --check_consistency \
  --method=model_merge \
  "${TEST_TMPDIR}/earnest-absolute.mod.ref" \
  "${TEST_TMPDIR}/earnest-seymore.pru.ref" \
  "${TEST_TMPDIR}/earnest.mrg.smooth"

fstequal \
  "${TEST_TMPDIR}/earnest.mrg.smooth.ref" \
  "${TEST_TMPDIR}/earnest.mrg.smooth"

compile_test_fst earnest.mrg.smooth.norm
"${BIN}/ngrammerge" \
  --check_consistency \
  --method=model_merge \
  --normalize \
  "${TEST_TMPDIR}/earnest-absolute.mod.ref" \
  "${TEST_TMPDIR}/earnest-seymore.pru.ref" \
  "${TEST_TMPDIR}/earnest.mrg.smooth.norm"

fstequal \
  "${TEST_TMPDIR}/earnest.mrg.smooth.norm.ref" \
  "${TEST_TMPDIR}/earnest.mrg.smooth.norm"
