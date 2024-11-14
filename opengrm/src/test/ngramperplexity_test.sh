#!/bin/bash
# Tests the command line binary ngramperplexity.

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

# Compile strings.
farcompilestrings \
  --fst_type=compact \
  --symbols="${TESTDATA}/earnest.sym" \
  --keep_symbols \
  "${TESTDATA}/earnest.txt" \
  "${TEST_TMPDIR}/earnest.far"

compile_test_fst earnest-witten_bell.mod
"${BIN}/ngramperplexity" \
  --OOV_probability=0.01 \
  "${TEST_TMPDIR}/earnest-witten_bell.mod.ref" \
  "${TEST_TMPDIR}/earnest.far" \
  "${TEST_TMPDIR}/earnest.perp"

file "${TESTDATA}/earnest.perp"
file "${TEST_TMPDIR}/earnest.perp"
cmp "${TESTDATA}/earnest.perp" "${TEST_TMPDIR}/earnest.perp"
